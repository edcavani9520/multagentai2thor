"""Small ProcTHOR scene smoke benchmark.

This runner intentionally talks to AI2-THOR directly. It is useful for checking
that a ProcTHOR house JSON can be loaded before wiring it into the HTTP receiver
or a larger task/evaluator framework.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a ProcTHOR scene smoke benchmark")
    parser.add_argument(
        "--scene-json",
        type=Path,
        required=True,
        help="A house JSON, JSON.GZ, or a directory containing them.",
    )
    parser.add_argument(
        "--local-executable-path",
        type=Path,
        default=None,
        help="Local AI2-THOR executable. Defaults to AI2THOR_EXECUTABLE.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/procthor_benchmark"),
        help="Directory for the summary and optional frames.",
    )
    parser.add_argument("--max-scenes", type=int, default=1)
    parser.add_argument("--move-steps", type=int, default=1)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument(
        "--x-display",
        default=os.environ.get("DISPLAY"),
        help="X display for rendered mode. Omit --headless when frames are needed.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without a display. Rendering may not work on Null graphics.",
    )
    parser.add_argument("--skip-render", action="store_true")
    return parser.parse_args()


def scene_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Scene path does not exist: {path}")
    paths = sorted(
        item
        for item in path.iterdir()
        if item.name != "manifest.json"
        and (item.suffix.lower() == ".json" or item.name.lower().endswith(".json.gz"))
    )
    if not paths:
        raise FileNotFoundError(f"No JSON or JSON.GZ scenes found in {path}")
    return paths


def load_house(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.name.lower().endswith(".json.gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        house = json.load(handle)
    if not isinstance(house, dict):
        raise ValueError(f"Scene must be a JSON object: {path}")
    return house


def agent_pose(house: dict[str, Any]) -> dict[str, Any]:
    """Read ProcTHOR's nested metadata pose, with a performerStart fallback."""
    agent = house.get("metadata", {}).get("agent")
    if isinstance(agent, dict) and isinstance(agent.get("position"), dict):
        position = agent["position"]
        rotation = agent.get("rotation", {})
        if isinstance(rotation, dict):
            rotation = rotation.get("y", 0)
        return {
            "x": float(position.get("x", 0)),
            "y": float(position.get("y", 0.95)),
            "z": float(position.get("z", 0)),
            "rotation": float(rotation),
            "horizon": float(agent.get("horizon", 30)),
            "standing": bool(agent.get("standing", True)),
        }

    start = house.get("performerStart", {})
    position = start.get("position", {})
    rotation = start.get("rotation", 0)
    if isinstance(rotation, dict):
        rotation = rotation.get("y", 0)
    return {
        "x": float(position.get("x", 0)),
        "y": float(position.get("y", 0.95)),
        "z": float(position.get("z", 0)),
        "rotation": float(rotation),
        "horizon": 30.0,
        "standing": True,
    }


def action_ok(event: Any) -> bool:
    return bool(event.metadata.get("lastActionSuccess"))


def action_error(event: Any) -> str:
    return str(event.metadata.get("errorMessage") or "")[:500]


def run_scene(controller: Any, path: Path, output_dir: Path, move_steps: int, skip_render: bool) -> dict[str, Any]:
    started = time.monotonic()
    result: dict[str, Any] = {"scene": str(path), "success": False}
    try:
        house = load_house(path)
        pose = agent_pose(house)
        result["schema"] = house.get("metadata", {}).get("schema")
        result["pose"] = pose
        result["rooms"] = len(house.get("rooms", []))
        result["objects_in_house_json"] = len(house.get("objects", []))

        controller.reset(renderImage=False)
        event = controller.step(action="CreateHouse", house=house)
        result["create_house"] = action_ok(event)
        if not result["create_house"]:
            result["error"] = action_error(event)
            return result

        event = controller.step(
            action="TeleportFull",
            x=pose["x"],
            y=pose["y"],
            z=pose["z"],
            rotation=pose["rotation"],
            horizon=pose["horizon"],
            standing=pose["standing"],
        )
        result["teleport"] = action_ok(event)
        if not result["teleport"]:
            result["error"] = action_error(event)
            return result

        event = controller.step(action="GetReachablePositions")
        reachable = event.metadata.get("actionReturn") or []
        result["reachable_positions"] = len(reachable)
        result["reachable_query"] = action_ok(event)
        if not result["reachable_query"]:
            result["error"] = action_error(event)
            return result

        move_results = []
        for _ in range(max(0, move_steps)):
            event = controller.step(action="MoveAhead", moveMagnitude=0.25)
            move_results.append(action_ok(event))
        result["move_results"] = move_results

        if not skip_render:
            event = controller.step(action="Pass", renderImage=True)
            result["observation"] = action_ok(event)
            result["frame_shape"] = list(event.frame.shape) if event.frame is not None else None
            result["runtime_objects"] = len(event.metadata.get("objects", []))
            if event.frame is not None:
                from PIL import Image

                frame_path = output_dir / f"{path.stem.replace('.', '_')}.png"
                Image.fromarray(event.frame).save(frame_path)
                result["frame"] = str(frame_path)
        result["success"] = True
    except Exception as exc:  # Keep later scenes running in a batch.
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return result


def main() -> int:
    args = parse_args()
    paths = scene_paths(args.scene_json)[: max(1, args.max_scenes)]
    executable = args.local_executable_path or os.environ.get("AI2THOR_EXECUTABLE")
    if not executable:
        raise SystemExit("Pass --local-executable-path or set AI2THOR_EXECUTABLE")

    from ai2thor.controller import Controller

    args.output_dir.mkdir(parents=True, exist_ok=True)
    controller = None
    results: list[dict[str, Any]] = []
    try:
        controller = Controller(
            local_executable_path=str(executable),
            scene="Procedural",
            width=args.width,
            height=args.height,
            headless=args.headless,
            x_display=args.x_display,
            server_start_timeout=60,
            server_timeout=60,
        )
        for path in paths:
            print(f"Running {path}...", flush=True)
            result = run_scene(controller, path, args.output_dir, args.move_steps, args.skip_render)
            results.append(result)
            print(json.dumps(result, ensure_ascii=True), flush=True)
    finally:
        if controller is not None:
            controller.stop()

    summary = {
        "scenes": len(results),
        "successful_scenes": sum(bool(item.get("success")) for item in results),
        "results": results,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Summary: {summary_path}", flush=True)
    return 0 if summary["successful_scenes"] == summary["scenes"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
