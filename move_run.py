#!/usr/bin/env python3
"""
Move Run — 两个机器人持续向前移动，观察长时间运动行为。

相当于把 dummy_agent 的 "Pass" 换成 "MoveAhead"，
让两个 robot 一直往前走，并实时用 OpenCV 窗口展示视野。

Usage:
    python move_run.py --show                     # 跑 100 步，窗口实时看
    python move_run.py --show --steps 500         # 跑 500 步
    python move_run.py --show --keep-open         # 跑完窗口不关
    python move_run.py --show --steps 0 --keep-open  # 无限跑
"""

import argparse
import sys
import os
import math
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_agent_env import MultiAgentEnv


def find_available_positions(scene: str) -> list:
    """Same as run.py — return two spawn positions."""
    Y = 0.900999128818512
    positions = {
        "FloorPlan1": [{"x": -1.25, "y": Y, "z": 1.5}, {"x": 1.5, "y": Y, "z": -1.25}],
        "FloorPlan2": [{"x": -1.8, "y": Y, "z": 1.0}, {"x": 1.8, "y": Y, "z": -1.0}],
        "FloorPlan3": [{"x": -1.2, "y": Y, "z": 1.2}, {"x": 1.2, "y": Y, "z": -1.2}],
        "FloorPlan4": [{"x": -1.5, "y": Y, "z": 1.0}, {"x": 1.0, "y": Y, "z": -1.5}],
    }
    return positions.get(scene, [
        {"x": -1.25, "y": Y, "z": 1.5}, {"x": 1.5, "y": Y, "z": -1.25},
    ])


def make_forward_agent(name: str):
    """创建一个只会向前走的 agent 函数。"""
    def agent_fn(obs):
        return {"action": "MoveAhead"}
    agent_fn.__name__ = name
    return agent_fn


def run_move_loop(env, agent_a, agent_b, max_steps, keep_open):
    """主循环：两个机器人不断 MoveAhead，OpenCV 窗口展示。"""
    import cv2

    WIN_A = "RobotA View — Moving Forward"
    WIN_B = "RobotB View — Moving Forward"
    cv2.namedWindow(WIN_A, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_B, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_A, 600, 600)
    cv2.resizeWindow(WIN_B, 600, 600)

    print(f"\n{'='*60}")
    print(f"🚀 Move Run — 持续向前移动")
    print(f"   窗口: '{WIN_A}' 和 '{WIN_B}'")
    print(f"   最大步数: {'无限' if max_steps <= 0 else max_steps}")
    print(f"   按 'q' 或 ESC 随时退出")
    print(f"{'='*60}\n")

    step = 0
    success_a = 0
    success_b = 0
    fail_a = 0
    fail_b = 0

    try:
        while True:
            if max_steps > 0 and step >= max_steps:
                print(f"\n[完成] 到达 {max_steps} 步")
                break

            # ── RobotA ──
            obs_a = env.observe_agent(agent_a)
            result_a = env.step_agent(agent_a, {"action": "MoveAhead"})
            ok_a = result_a.get("lastActionSuccess", False)
            if ok_a:
                success_a += 1
                new_pos = result_a.get("agent", {}).get("position")
                if new_pos:
                    agent_a.position = new_pos
            else:
                fail_a += 1

            # ── RobotB ──
            obs_b = env.observe_agent(agent_b)
            result_b = env.step_agent(agent_b, {"action": "MoveAhead"})
            ok_b = result_b.get("lastActionSuccess", False)
            if ok_b:
                success_b += 1
                new_pos = result_b.get("agent", {}).get("position")
                if new_pos:
                    agent_b.position = new_pos
            else:
                fail_b += 1

            # ── Show ──
            step_str = f"Step {step}"
            if obs_a.get("image") is not None:
                img_a = obs_a["image"][:, :, ::-1].copy()
                pos_a = agent_a.position
                cv2.putText(img_a, f"RobotA | {step_str}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(img_a, f"Pos: ({pos_a['x']:.2f}, {pos_a['z']:.2f})", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(img_a, f"Move: {'OK' if ok_a else 'FAIL'}", (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0) if ok_a else (0, 0, 255), 1)
                cv2.imshow(WIN_A, img_a)

            if obs_b.get("image") is not None:
                img_b = obs_b["image"][:, :, ::-1].copy()
                pos_b = agent_b.position
                cv2.putText(img_b, f"RobotB | {step_str}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(img_b, f"Pos: ({pos_b['x']:.2f}, {pos_b['z']:.2f})", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(img_b, f"Move: {'OK' if ok_b else 'FAIL'}", (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0) if ok_b else (0, 0, 255), 1)
                cv2.imshow(WIN_B, img_b)

            step += 1
            status = f"[{step_str}] RobotA: {'OK' if ok_a else 'FAIL'} @ ({pos_a['x']:.2f}, {pos_a['z']:.2f})  |  RobotB: {'OK' if ok_b else 'FAIL'} @ ({pos_b['x']:.2f}, {pos_b['z']:.2f})"
            print(status)

            # ── Check exit ──
            key = cv2.waitKey(100) & 0xFF
            if key in (ord('q'), 27):
                print("[退出] 用户按 Q")
                break

        # ── Keep-open 阶段 ──
        if keep_open:
            print(f"\n[KeepOpen] 移动结束。窗口保持打开，按 'q' 或 ESC 关闭。")
            while True:
                obs_a = env.observe_agent(agent_a)
                obs_b = env.observe_agent(agent_b)
                if obs_a.get("image") is not None:
                    img = obs_a["image"][:, :, ::-1].copy()
                    cv2.putText(img, "RobotA | IDLE", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.imshow(WIN_A, img)
                if obs_b.get("image") is not None:
                    img = obs_b["image"][:, :, ::-1].copy()
                    cv2.putText(img, "RobotB | IDLE", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.imshow(WIN_B, img)
                key = cv2.waitKey(500) & 0xFF
                if key in (ord('q'), 27):
                    break
    finally:
        cv2.destroyAllWindows()

    # ── 统计 ──
    print(f"\n{'='*60}")
    print(f"📊 移动统计")
    print(f"   RobotA: 成功 {success_a} 次, 失败 {fail_a} 次")
    print(f"   RobotB: 成功 {success_b} 次, 失败 {fail_b} 次")
    final_pos_a = agent_a.position
    final_pos_b = agent_b.position
    print(f"   RobotA 最终: ({final_pos_a['x']:.2f}, {final_pos_a['z']:.2f})")
    print(f"   RobotB 最终: ({final_pos_b['x']:.2f}, {final_pos_b['z']:.2f})")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Move Run — 两个机器人持续向前移动")
    parser.add_argument("--scene", default="FloorPlan1",
                        help="AI2-THOR 场景名 (默认: FloorPlan1)")
    parser.add_argument("--headless", action="store_true", default=True,
                        help="头模式 CloudRendering")
    parser.add_argument("--no-headless", action="store_false", dest="headless",
                        help="显示 Unity 窗口")
    parser.add_argument("--output", default="output",
                        help="输出目录")
    parser.add_argument("--steps", type=int, default=100,
                        help="移动步数 (0 = 无限)")
    parser.add_argument("--show", action="store_true", default=True,
                        help="显示 OpenCV 窗口 (默认开启)")
    parser.add_argument("--keep-open", action="store_true", default=False,
                        help="移动结束后保持窗口打开")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🤖 Move Run — Multi-Agent 持续向前移动")
    print("=" * 60)

    with MultiAgentEnv(scene=args.scene, headless=args.headless,
                       save_dir=args.output) as env:
        # ── 放置两个 agent ──
        positions = find_available_positions(args.scene)
        print(f"初始位置:")
        for i, p in enumerate(positions):
            print(f"  Agent {i}: ({p['x']:.2f}, {p['z']:.2f})")

        dx = positions[1]["x"] - positions[0]["x"]
        dz = positions[1]["z"] - positions[0]["z"]
        angle_a = math.degrees(math.atan2(-dx, dz))
        angle_b = angle_a + 180

        agent_a = env.add_agent("RobotA", positions[0],
                                {"x": 0, "y": angle_a, "z": 0})
        agent_b = env.add_agent("RobotB", positions[1],
                                {"x": 0, "y": angle_b, "z": 0})

        print(f"\n场景物体数: {len(env.controller.last_event.metadata.get('objects', []))}")

        # ── 开始移动 ──
        run_move_loop(env, agent_a, agent_b, args.steps, args.keep_open)

        print(f"\n--- 完成 ---")
        print(f"输出目录: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
