#!/usr/bin/env python3
"""
VLA Run — 用 OpenVLA 驱动两个 AI2-THOR 机器人。

每个 robot 加载一个 OpenVLA 实例（4-bit 量化），
接收自己的视野图像 + 自然语言指令，
推理出下一步动作并执行。

Usage:
    # 双 robot，各一张卡（推荐，2×3080 Ti 完美匹配）
    python3 vla_run.py --show --steps 100

    # 双 robot 共享同一张卡
    python3 vla_run.py --show --steps 100 --single-gpu

    # 连续动作模式（TeleportFull 直接设位姿）
    python3 vla_run.py --show --steps 100 --continuous

    # 跑完不关窗口
    python3 vla_run.py --show --keep-open --steps 200
"""

import argparse
import math
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_agent_env import MultiAgentEnv
from agents.vla_agent import VLAgent


def find_available_positions(scene: str) -> list:
    """各个场景的初始位置（与 run.py 一致）"""
    Y = 0.900999128818512
    positions = {
        "FloorPlan1": [{"x": -1.25, "y": Y, "z": 1.5},
                       {"x": 1.5, "y": Y, "z": -1.25}],
        "FloorPlan2": [{"x": -1.8, "y": Y, "z": 1.0},
                       {"x": 1.8, "y": Y, "z": -1.0}],
        "FloorPlan3": [{"x": -1.2, "y": Y, "z": 1.2},
                       {"x": 1.2, "y": Y, "z": -1.2}],
        "FloorPlan4": [{"x": -1.5, "y": Y, "z": 1.0},
                       {"x": 1.0, "y": Y, "z": -1.5}],
    }
    return positions.get(scene, [
        {"x": -1.25, "y": Y, "z": 1.5},
        {"x": 1.5, "y": Y, "z": -1.25},
    ])


def setup_agents(env, scene_name, output_dir):
    """创建两个 AgentState，互相面对。返回 (agent_a, agent_b)。"""
    positions = find_available_positions(scene_name)
    print(f"\n初始位置:")
    for i, p in enumerate(positions):
        print(f"  Agent {i}: ({p['x']:.2f}, {p['z']:.2f})")

    dx = positions[1]["x"] - positions[0]["x"]
    dz = positions[1]["z"] - positions[0]["z"]
    angle_a = math.degrees(math.atan2(-dx, dz))
    angle_b = angle_a + 180

    agent_a = env.add_agent("RobotA", positions[0],
                            {"x": 0, "y": angle_a, "z": 0},
                            task="Observe RobotB")
    agent_b = env.add_agent("RobotB", positions[1],
                            {"x": 0, "y": angle_b, "z": 0},
                            task="Observe RobotA")
    return agent_a, agent_b


def run_vla_loop(env, agent_a, agent_b, vla_a, vla_b,
                 max_steps, show, keep_open, output_dir):
    """主循环：两个 robot 各自用 VLA 推理动作并执行。"""
    if show:
        import cv2
        WIN_A = "RobotA — VLA"
        WIN_B = "RobotB — VLA"
        cv2.namedWindow(WIN_A, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_B, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_A, 600, 600)
        cv2.resizeWindow(WIN_B, 600, 600)

    print(f"\n{'='*60}")
    print(f"🤖 VLA Run — OpenVLA 驱动双机器人")
    print(f"   RobotA 指令: \"{vla_a.instruction}\"")
    print(f"   RobotB 指令: \"{vla_b.instruction}\"")
    print(f"   步数: {'无限' if max_steps <= 0 else max_steps}")
    if show:
        print(f"   窗口按 'q' 或 ESC 随时退出")
    print(f"{'='*60}\n")

    step = 0
    stats = {"RobotA": {"ok": 0, "fail": 0},
             "RobotB": {"ok": 0, "fail": 0}}

    def _show_window(name, window, agent, obs, step, ok):
        img = obs["image"][:, :, ::-1].copy()
        pos = agent.position
        cv2.putText(img, f"{agent.name} | Step {step}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, f"Pos: ({pos['x']:.2f}, {pos['z']:.2f})",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 0), 1)
        cv2.putText(img, f"{'✓' if ok else '✗'}",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if ok else (0, 0, 255), 2)
        cv2.imshow(window, img)

    try:
        while True:
            if max_steps > 0 and step >= max_steps:
                break

            # ── RobotA ──
            obs_a = env.observe_agent(agent_a)
            if obs_a:
                action_a = vla_a(obs_a)
                result_a = env.step_agent(agent_a, action_a)
                ok_a = result_a.get("lastActionSuccess", False)
                stats["RobotA"]["ok" if ok_a else "fail"] += 1
                if ok_a and result_a.get("agent", {}).get("position"):
                    agent_a.position = result_a["agent"]["position"]
                if show and obs_a.get("image") is not None:
                    _show_window("A", WIN_A, agent_a, obs_a, step, ok_a)

            # ── RobotB ──
            obs_b = env.observe_agent(agent_b)
            if obs_b:
                action_b = vla_b(obs_b)
                result_b = env.step_agent(agent_b, action_b)
                ok_b = result_b.get("lastActionSuccess", False)
                stats["RobotB"]["ok" if ok_b else "fail"] += 1
                if ok_b and result_b.get("agent", {}).get("position"):
                    agent_b.position = result_b["agent"]["position"]
                if show and obs_b.get("image") is not None:
                    _show_window("B", WIN_B, agent_b, obs_b, step, ok_b)

            # ── 打印状态 ──
            act_a = action_a.get("action", "?") if obs_a else "?"
            act_b = action_b.get("action", "?") if obs_b else "?"
            pos_a = agent_a.position
            pos_b = agent_b.position
            print(f"[Step {step:03d}] "
                  f"A: {act_a:12s} {'✓' if ok_a else '✗'} "
                  f"@ ({pos_a['x']:.2f}, {pos_a['z']:.2f})  |  "
                  f"B: {act_b:12s} {'✓' if ok_b else '✗'} "
                  f"@ ({pos_b['x']:.2f}, {pos_b['z']:.2f})")

            step += 1

            # ── 保存图片 ──
            if obs_a and obs_a.get("image") is not None:
                import cv2
                cv2.imwrite(
                    str(output_dir / f"step_{step:04d}_RobotA.png"),
                    obs_a["image"][:, :, ::-1])
            if obs_b and obs_b.get("image") is not None:
                import cv2
                cv2.imwrite(
                    str(output_dir / f"step_{step:04d}_RobotB.png"),
                    obs_b["image"][:, :, ::-1])

            # ── 退出检查 ──
            if show:
                key = cv2.waitKey(50) & 0xFF
                if key in (ord('q'), 27):
                    print("[退出] 用户按 Q")
                    break

        # ── Keep-open ──
        if show and keep_open:
            print(f"\n[KeepOpen] 步数完成，窗口保持。按 'q' 关闭。")
            while True:
                for obs, win, agent in [
                    (env.observe_agent(agent_a), WIN_A, agent_a),
                    (env.observe_agent(agent_b), WIN_B, agent_b),
                ]:
                    if obs.get("image") is not None:
                        img = obs["image"][:, :, ::-1].copy()
                        cv2.putText(img, f"{agent.name} | IDLE",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 255, 255), 2)
                        cv2.imshow(win, img)
                key = cv2.waitKey(500) & 0xFF
                if key in (ord('q'), 27):
                    break

    finally:
        if show:
            import cv2
            cv2.destroyAllWindows()

    # ── 统计 ──
    print(f"\n{'='*60}")
    print(f"📊 运行统计（{step} 步）")
    for name, s in stats.items():
        total = s["ok"] + s["fail"]
        rate = s["ok"] / total * 100 if total > 0 else 0
        print(f"   {name}: 成功 {s['ok']}/{total} ({rate:.1f}%)")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="VLA Run — OpenVLA 驱动 AI2-THOR 双机器人")
    parser.add_argument("--scene", default="FloorPlan1",
                        help="AI2-THOR 场景 (默认 FloorPlan1)")
    parser.add_argument("--headless", action="store_true", default=True,
                        help="CloudRendering 头模式")
    parser.add_argument("--no-headless", action="store_false",
                        dest="headless", help="显示 Unity 窗口")
    parser.add_argument("--output", default="output",
                        help="输出目录")
    parser.add_argument("--steps", type=int, default=50,
                        help="步数 (0 = 无限)")
    parser.add_argument("--show", action="store_true", default=True,
                        help="显示 OpenCV 窗口")
    parser.add_argument("--keep-open", action="store_true", default=False,
                        help="步数完成后保持窗口")
    parser.add_argument("--single-gpu", action="store_true", default=False,
                        help="两个 VLA 共享一张卡")
    parser.add_argument("--continuous", action="store_true", default=False,
                        help="使用 Teleport 连续动作模式取代离散映射")
    parser.add_argument("--model", default="openvla/openvla-7b",
                        help="模型 ID")
    parser.add_argument("--instruct-a",
                        default="go to the nearest object and pick it up",
                        help="RobotA 指令")
    parser.add_argument("--instruct-b",
                        default="explore the room and find the vase",
                        help="RobotB 指令")
    parser.add_argument("--unnorm-key", default=None,
                        help="反归一化 key (OpenVLA 训练数据集)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🤖 VLA Run — OpenVLA 驱动双 AI2-THOR 机器人")
    print("=" * 60)

    with MultiAgentEnv(scene=args.scene, headless=args.headless,
                       save_dir=str(output_dir)) as env:

        agent_a, agent_b = setup_agents(env, args.scene, output_dir)
        print(f"\n场景物体数: "
              f"{len(env.controller.last_event.metadata.get('objects', []))}")

        # ── 加载 VLA ──
        mode = "continuous" if args.continuous else "discrete"

        if args.single_gpu:
            # 两个 robot 用同一张卡
            vla_a = VLAgent("RobotA", args.instruct_a,
                            model_id=args.model, device="cuda:0",
                            mode=mode, unnorm_key=args.unnorm_key)
            vla_b = VLAgent("RobotB", args.instruct_b,
                            model_id=args.model, device="cuda:0",
                            mode=mode, unnorm_key=args.unnorm_key)
        else:
            # 双卡：各占一张 (0 和 1)
            vla_a = VLAgent("RobotA", args.instruct_a,
                            model_id=args.model, device="cuda:0",
                            mode=mode, unnorm_key=args.unnorm_key)
            vla_b = VLAgent("RobotB", args.instruct_b,
                            model_id=args.model, device="cuda:1",
                            mode=mode, unnorm_key=args.unnorm_key)

        # ── 运行 ──
        run_vla_loop(env, agent_a, agent_b, vla_a, vla_b,
                     args.steps, args.show, args.keep_open, output_dir)

        print(f"\n--- 完成 ---")
        print(f"输出: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
