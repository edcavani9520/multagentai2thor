#!/usr/bin/env python3
"""
Custom Agent Run — 用你训练好的离散动作模型驱动 AI2-THOR 机器人。

你的模型只需满足:
    model(image: np.ndarray, instruction: str) → [base_action, modifier]

用法:
    python3 custom_run.py --show --steps 200
    python3 custom_run.py --show --steps 200 --no-model   # 纯规则模式 (只返回Pass)

集成你的模型:
    # 方式1: 直接传 callable
    from agents.custom_agent import CustomDiscreteAgent
    agent = CustomDiscreteAgent("RobotA", model=my_model, instruction="pick up")

    # 方式2: 用 VLM + 规则映射 (LLaVA/Qwen-VL 输出文本)
    from agents.custom_agent import PromptedDiscreteAgent
    agent = PromptedDiscreteAgent("RobotA", vlm=my_vlm, instruction="pick up")
"""

import argparse
import math
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_agent_env import MultiAgentEnv
from agents.custom_agent import CustomDiscreteAgent, PromptedDiscreteAgent


def find_available_positions(scene: str) -> list:
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


def setup_agents(env, scene_name):
    positions = find_available_positions(scene_name)
    print(f"\n初始位置:")
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
    return agent_a, agent_b


def run_loop(env, agent_a, agent_b, model_a, model_b,
             max_steps, show, keep_open, output_dir):
    if show:
        import cv2
        WIN_A = "RobotA — Your Model"
        WIN_B = "RobotB — Your Model"
        cv2.namedWindow(WIN_A, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_B, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_A, 600, 600)
        cv2.resizeWindow(WIN_B, 600, 600)

    print(f"\n{'='*60}")
    print(f"🤖 Custom Agent Run")
    print(f"   RobotA: \"{model_a.instruction}\"")
    print(f"   RobotB: \"{model_b.instruction}\"")
    print(f"   步数: {'无限' if max_steps <= 0 else max_steps}")
    print(f"{'='*60}\n")

    step = 0
    stats = {"RobotA": {"ok": 0, "fail": 0},
             "RobotB": {"ok": 0, "fail": 0}}

    try:
        while True:
            if max_steps > 0 and step >= max_steps:
                break

            for agent, model in [(agent_a, model_a), (agent_b, model_b)]:
                obs = env.observe_agent(agent)
                if not obs:
                    continue

                action = model(obs)
                result = env.step_agent(agent, action)
                ok = result.get("lastActionSuccess", False)
                name = agent.name
                stats[name]["ok" if ok else "fail"] += 1

                if ok and result.get("agent", {}).get("position"):
                    agent.position = result["agent"]["position"]

                if show and obs.get("image") is not None:
                    img = obs["image"][:, :, ::-1].copy()
                    pos = agent.position
                    import cv2
                    cv2.putText(img, f"{name} | Step {step}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)
                    cv2.putText(img, f"{action.get('action','?')} "
                                f"{'✓' if ok else '✗'}",
                                (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0) if ok else (0, 0, 255), 2)
                    cv2.putText(img, f"Pos: ({pos['x']:.2f}, {pos['z']:.2f})",
                                (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 0), 1)
                    win = WIN_A if name == "RobotA" else WIN_B
                    cv2.imshow(win, img)

                action_name = action.get("action", "?")
                raw = action.get("_raw_output", "")
                print(f"  [Step {step:03d}] {name}: {action_name:12s} "
                      f"[{'✓' if ok else '✗'}]  raw={raw}")

            step += 1

            if show:
                import cv2
                key = cv2.waitKey(100) & 0xFF
                if key in (ord('q'), 27):
                    print("[退出]")
                    break

        if show and keep_open:
            import cv2
            print("[KeepOpen] 按 q 关闭")
            while True:
                key = cv2.waitKey(500) & 0xFF
                if key in (ord('q'), 27):
                    break

    finally:
        if show:
            import cv2
            cv2.destroyAllWindows()

    print(f"\n{'='*60}")
    print(f"📊 统计 ({step} 步)")
    for name, s in stats.items():
        total = s["ok"] + s["fail"]
        rate = s["ok"] / total * 100 if total else 0
        print(f"   {name}: 成功 {s['ok']}/{total} ({rate:.1f}%)")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Custom Agent Run — 接入你的离散动作模型")
    parser.add_argument("--scene", default="FloorPlan1")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", action="store_false", dest="headless")
    parser.add_argument("--output", default="output")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--show", action="store_true", default=True)
    parser.add_argument("--keep-open", action="store_true", default=False)
    parser.add_argument("--no-model", action="store_true", default=False,
                        help="无模型模式（只返回 Pass，测试环境是否正常）")
    parser.add_argument("--instruct-a",
                        default="explore the room",
                        help="RobotA 指令")
    parser.add_argument("--instruct-b",
                        default="find the bowl",
                        help="RobotB 指令")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🤖 Custom Agent Run — 接入你的模型")
    print("=" * 60)

    with MultiAgentEnv(scene=args.scene, headless=args.headless,
                       save_dir=str(output_dir)) as env:

        agent_a, agent_b = setup_agents(env, args.scene)
        print(f"\n场景物体: "
              f"{len(env.controller.last_event.metadata.get('objects', []))}")

        # ── 创建 Agent ──
        # 在这里改成你的真实模型加载！
        if args.no_model:
            model_a = CustomDiscreteAgent("RobotA", model=None,
                                          instruction=args.instruct_a)
            model_b = CustomDiscreteAgent("RobotB", model=None,
                                          instruction=args.instruct_b)
        else:
            # ── TODO: 在这里加载你的真实模型 ──
            # 例如:
            #   from your_model import load_model
            #   my_model = load_model("/path/to/checkpoint.pth")
            #   model_a = CustomDiscreteAgent("RobotA", model=my_model, ...)
            #   model_b = CustomDiscreteAgent("RobotB", model=my_model, ...)
            #
            # 你的模型签名: model(image, instruction) → ["moveahead", "none"]
            print("\n⚠️  请先在 custom_run.py 中加载你的模型！")
            print("   参考 agents/custom_agent.py 的 CustomDiscreteAgent")
            model_a = CustomDiscreteAgent("RobotA", model=None,
                                          instruction=args.instruct_a)
            model_b = CustomDiscreteAgent("RobotB", model=None,
                                          instruction=args.instruct_b)

        run_loop(env, agent_a, agent_b, model_a, model_b,
                 args.steps, args.show, args.keep_open, output_dir)


if __name__ == "__main__":
    main()
