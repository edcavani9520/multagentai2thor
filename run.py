#!/usr/bin/env python3
"""
Multi-Agent AI2-THOR Framework
================================
Demo: two dummy agents that can see each other (via proxy objects).

Usage:
    python run.py                          # 默认自动运行
    python run.py --show                   # OpenCV 窗口实时看两个 robot 视角
    python run.py --interactive            # 键盘操控 robot 在场景里走
    python run.py --real-agent             # 用同学的 agent 替换 dummy
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_agent_env import MultiAgentEnv
from agents.dummy_agent import make_dummy_agent

try:
    from agents.colleague_agent_stub import colleague_agent
    HAS_COLLEAGUE = True
except ImportError:
    HAS_COLLEAGUE = False


def find_available_positions(scene: str) -> list:
    Y = 0.900999128818512
    positions = {
        "FloorPlan1": [ {"x": -1.25, "y": Y, "z": 1.5},  {"x": 1.5, "y": Y, "z": -1.25} ],
        "FloorPlan2": [ {"x": -1.8, "y": Y, "z": 1.0},  {"x": 1.8, "y": Y, "z": -1.0} ],
        "FloorPlan3": [ {"x": -1.2, "y": Y, "z": 1.2},  {"x": 1.2, "y": Y, "z": -1.2} ],
        "FloorPlan4": [ {"x": -1.5, "y": Y, "z": 1.0},  {"x": 1.0, "y": Y, "z": -1.5} ],
    }
    return positions.get(scene, [
        {"x": -1.25, "y": Y, "z": 1.5}, {"x": 1.5, "y": Y, "z": -1.25},
    ])


def _run_interactive(env, agents):
    """
    键盘操控模式。用 WASD 控制当前选中的 robot 在场景里行走。
    """
    import cv2

    WIN = "Interactive Control"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 640, 640)

    active_idx = 0
    agents_list = agents

    # Help overlay
    HELP = (
        "W/S 前进/后退 | A/D 旋转 | ↑/↓ 俯仰 | Tab 切换robot | Space 拾取最近物体 | "
        "P 放置 | O 打开 | C 关闭 | q/ESC 退出"
    )

    print(f"\n{'='*60}")
    print(f"🎮 Interactive Mode")
    print(f"   WASD   — 移动 / 旋转")
    print(f"   ↑/↓    — 俯仰")
    print(f"   Tab    — 切换操控哪个 robot")
    print(f"   Space  — 拾取最近的物体")
    print(f"   P      — 放置手中物体")
    print(f"   O/C    — 打开/关闭")
    print(f"   q/ESC  — 退出")
    print(f"{'='*60}\n")

    try:
        while True:
            agent = agents_list[active_idx]

            # Capture current agent's view
            obs = env.observe_agent(agent)
            if obs.get("image") is None:
                continue

            img = obs["image"][:, :, ::-1].copy()
            h, w = img.shape[:2]

            # Overlay info
            label = f"▶ Active: {agent.name}  |  Pos: ({agent.position['x']:.2f}, {agent.position['z']:.2f})"
            cv2.putText(img, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 2)
            cv2.putText(img, HELP, (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (200, 200, 200), 1)
            cv2.imshow(WIN, img)

            key = cv2.waitKey(50) & 0xFF

            # ── Quit ──
            if key in (ord('q'), 27):
                print("[Interactive] Quit")
                break

            # ── Switch agent ──
            if key == 9:  # Tab
                active_idx = (active_idx + 1) % len(agents_list)
                print(f"[Interactive] Switched to {agents_list[active_idx].name}")
                continue

            # ── Movement ──
            action = None
            if key == ord('w') or key == ord('W'):
                action = {"action": "MoveAhead"}
            elif key == ord('s') or key == ord('S'):
                action = {"action": "MoveBack"}
            elif key == ord('a') or key == ord('A'):
                action = {"action": "MoveLeft"}
            elif key == ord('d') or key == ord('D'):
                action = {"action": "MoveRight"}
            elif key == 82:  # Up arrow
                action = {"action": "LookUp"}
            elif key == 84:  # Down arrow
                action = {"action": "LookDown"}
            elif key == 81:  # Left arrow
                action = {"action": "RotateLeft"}
            elif key == 83:  # Right arrow
                action = {"action": "RotateRight"}

            # ── Object Interaction ──
            elif key == ord(' '):  # Space → pickup nearest
                nearest = None
                min_dist = float('inf')
                for obj in obs.get("objects", []):
                    d = obj.get("distance", 999)
                    if d < min_dist and obj["objectType"] not in (
                        "Floor", "Wall", "Window", "Cabinet", "CounterTop",
                        "Drawer", "Shelf", "ShelvingUnit", "Door"):
                        # Skip large furniture, prefer small objects
                        pass
                    if d < min_dist:
                        # Just pick the closest object overall
                        min_dist = d
                        nearest = obj.get("objectId", "")
                if nearest and min_dist < 2.0:
                    action = {"action": "PickupObject", "objectId": nearest}
                    print(f"  Pickup: {nearest}")
                else:
                    action = {"action": "Pass"}
                    print(f"  No pickup target within 2m (closest: {min_dist:.2f})")

            elif key == ord('p') or key == ord('P'):
                # PutObject — find a receptacle
                receptacles = [o for o in obs.get("objects", [])
                               if o.get("receptacle", False) and o.get("objectId", "")]
                if receptacles:
                    rid = receptacles[0]["objectId"]
                    action = {"action": "PutObject", "objectId": rid}
                else:
                    action = {"action": "DropHandObject"}

            elif key == ord('o') or key == ord('O'):
                # OpenObject — find nearest openable
                openable = [o for o in obs.get("objects", [])
                            if o.get("openable", False) and o.get("objectId", "")]
                if openable:
                    action = {"action": "OpenObject", "objectId": openable[0]["objectId"]}

            elif key == ord('c') or key == ord('C'):
                openable = [o for o in obs.get("objects", [])
                            if o.get("openable", False) and o.get("objectId", "")]
                if openable:
                    action = {"action": "CloseObject", "objectId": openable[0]["objectId"]}

            # Execute
            if action:
                result = env.step_agent(agent, action)
                ok = result.get("lastActionSuccess", False)
                # Update agent position from result
                if ok:
                    new_pos = result.get("agent", {}).get("position")
                    if new_pos:
                        agent.position = new_pos
                if key not in (9,):  # Don't spam for Tab switches
                    action_name = action.get("action", "?")
                    cv2.putText(img, f"{'✓' if ok else '✗'} {action_name}",
                                (12, 58), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0) if ok else (0, 0, 255), 2)

    finally:
        cv2.destroyAllWindows()


def _run_with_gui(env, agent_a, agent_b, agent_fns, max_steps, output_dir,
                   keep_open=False):
    import cv2

    W_A = "RobotA View"
    W_B = "RobotB View"
    cv2.namedWindow(W_A, cv2.WINDOW_NORMAL)
    cv2.namedWindow(W_B, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(W_A, 600, 600)
    cv2.resizeWindow(W_B, 600, 600)

    print(f"\n{'='*60}")
    print(f"[GUI] Windows opened: '{W_A}' and '{W_B}'")
    print(f"[GUI] Press 'q' or ESC to quit")
    print(f"{'='*60}\n")

    try:
        for step in range(max_steps):
            obs_a = env.observe_agent(agent_a)
            obs_b = env.observe_agent(agent_b)

            if obs_a.get("image") is not None:
                img_a = obs_a["image"][:, :, ::-1].copy()
                cv2.putText(img_a, f"RobotA | Step {step}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(W_A, img_a)
                cv2.imwrite(str(output_dir / f"step_{step:04d}_RobotA.png"), img_a)

            if obs_b.get("image") is not None:
                img_b = obs_b["image"][:, :, ::-1].copy()
                cv2.putText(img_b, f"RobotB | Step {step}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(W_B, img_b)
                cv2.imwrite(str(output_dir / f"step_{step:04d}_RobotB.png"), img_b)

            key = cv2.waitKey(100) & 0xFF
            if key in (ord('q'), 27):
                print("[GUI] Quit by user")
                return

            for agent in [agent_a, agent_b]:
                if agent.agent_id not in agent_fns:
                    continue
                fn = agent_fns[agent.agent_id]
                cur_obs = obs_a if agent.agent_id == 0 else obs_b
                try:
                    action = fn(cur_obs)
                except Exception as e:
                    print(f"[GUI] {agent.name} error: {e}")
                    continue
                result = env.step_agent(agent, action)
                ok = result.get("lastActionSuccess", False)
                print(f"  [Step {step}] {agent.name} → {action.get('action','?')} "
                      f"[{'OK' if ok else 'FAIL'}]")

        if keep_open:
            print(f"\n[KeepOpen] Steps done. Windows stay open. Press 'q' or ESC to close.")
            while True:
                obs_a = env.observe_agent(agent_a)
                obs_b = env.observe_agent(agent_b)
                if obs_a.get("image") is not None:
                    img = obs_a["image"][:, :, ::-1].copy()
                    cv2.putText(img, "RobotA | IDLE", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.imshow(W_A, img)
                if obs_b.get("image") is not None:
                    img = obs_b["image"][:, :, ::-1].copy()
                    cv2.putText(img, "RobotB | IDLE", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.imshow(W_B, img)
                key = cv2.waitKey(500) & 0xFF
                if key in (ord('q'), 27):
                    break
    finally:
        cv2.destroyAllWindows()


def _setup_agents(env, scene_name, output_dir):
    """Register two agents and place proxy objects. Returns (agent_a, agent_b)."""
    import math
    positions = find_available_positions(scene_name)
    print(f"Agent positions:")
    for i, p in enumerate(positions):
        print(f"  {i}: ({p['x']:.2f}, {p['z']:.2f})")

    dx = positions[1]["x"] - positions[0]["x"]
    dz = positions[1]["z"] - positions[0]["z"]
    angle_a = math.degrees(math.atan2(-dx, dz))
    angle_b = angle_a + 180

    agent_a = env.add_agent("RobotA", positions[0],
                            {"x": 0, "y": angle_a, "z": 0}, task="Observe RobotB")
    agent_b = env.add_agent("RobotB", positions[1],
                            {"x": 0, "y": angle_b, "z": 0}, task="Observe RobotA")

    # Place proxy objects
    mid_x = (positions[0]["x"] + positions[1]["x"]) / 2
    mid_z = (positions[0]["z"] + positions[1]["z"]) / 2
    env.place_proxy_object("Bowl", {"x": mid_x + 0.5, "y": positions[0]["y"], "z": mid_z})
    env.place_proxy_object("Vase", {"x": mid_x - 0.5, "y": positions[1]["y"], "z": mid_z})

    return agent_a, agent_b


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent AI2-THOR Framework")
    parser.add_argument("--scene", default="FloorPlan1",
                        help="AI2-THOR scene name (default: FloorPlan1)")
    parser.add_argument("--headless", action="store_true", default=True,
                        help="Run headless with CloudRendering")
    parser.add_argument("--no-headless", action="store_false", dest="headless",
                        help="Show Unity window")
    parser.add_argument("--output", default="output",
                        help="Output directory for images/logs")
    parser.add_argument("--steps", type=int, default=5,
                        help="Round-robin steps per agent (auto mode)")
    parser.add_argument("--real-agent", action="store_true",
                        help="Use colleague's agent instead of dummy")
    parser.add_argument("--show", action="store_true", default=False,
                        help="Open two OpenCV windows (auto mode)")
    parser.add_argument("--interactive", action="store_true", default=False,
                        help="🎮 Keyboard control: WASD move, Tab switch robot")
    parser.add_argument("--keep-open", action="store_true", default=False,
                        help="Keep windows open after run finishes")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Multi-Agent AI2-THOR Framework")
    print("=" * 60)

    with MultiAgentEnv(scene=args.scene, headless=args.headless,
                       save_dir=args.output) as env:

        agent_a, agent_b = _setup_agents(env, args.scene, output_dir)

        print(f"\nScene has {len(env.controller.last_event.metadata.get('objects', []))} objects")

        # Agent functions
        agent_fns = {}
        agent_fns[agent_a.agent_id] = make_dummy_agent("RobotA", log_dir=str(output_dir))
        agent_fns[agent_b.agent_id] = make_dummy_agent("RobotB", log_dir=str(output_dir))

        # Initial snapshots
        print(f"\n--- Initial Views ---")
        for agent in [agent_a, agent_b]:
            obs = env.observe_agent(agent)
            if obs.get("image") is not None:
                import cv2
                cv2.imwrite(str(output_dir / f"initial_view_{agent.name}.png"),
                            obs["image"][:, :, ::-1])
            print(f"  {agent.name} @ ({agent.position['x']:.2f}, {agent.position['z']:.2f})")

        # Mode selection
        if args.interactive:
            _run_interactive(env, [agent_a, agent_b])
        elif args.show:
            _run_with_gui(env, agent_a, agent_b, agent_fns,
                          args.steps, output_dir, keep_open=args.keep_open)
        else:
            env.run_agents_round_robin(agent_fns,
                                       max_steps=args.steps,
                                       save_images=True)
            if args.keep_open and not args.headless:
                print(f"\n[KeepOpen] Unity window stays open. Press Ctrl+C to exit.")
                try:
                    while True:
                        import time
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass

        print(f"\n--- Done ---")
        print(f"Output: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
