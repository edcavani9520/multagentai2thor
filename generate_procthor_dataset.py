"""Generate deterministic, current-build-compatible ProcTHOR test scenes.

The installed ProcTHOR package provides the official HouseGenerator and
PROCTHOR10K_ROOM_SPEC_SAMPLER. This wrapper makes its sampling deterministic,
keeps the local AI2-THOR compatibility conversion in one place, and writes a
manifest for later benchmark reproducibility.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic ProcTHOR test scenes")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument(
        "--local-executable-path",
        type=Path,
        default=None,
        help="Local AI2-THOR executable. Defaults to AI2THOR_EXECUTABLE.",
    )
    parser.add_argument("--x-display", default=os.environ.get("DISPLAY"))
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--include-small-objects",
        action="store_true",
        help="Attempt the full official generation stage. It is not compatible with the current local build.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def event_ok(event: Any) -> bool:
    return bool(event.metadata.get("lastActionSuccess"))


def event_error(event: Any) -> str:
    return str(event.metadata.get("errorMessage") or "")[:500]


def material(value: Any) -> Any:
    if isinstance(value, str):
        return {"name": value, "color": {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0}}
    if isinstance(value, dict):
        result = dict(value)
        if "name" not in result and "material" in result:
            result["name"] = result.pop("material")
        result.setdefault("color", {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0})
        return result
    return value


def converted_house(raw_house: dict[str, Any]) -> dict[str, Any]:
    """Translate ProcTHOR 0.0.1 data into the schema accepted by this build."""
    data = json.loads(json.dumps(raw_house))

    for room in data.get("rooms", []):
        for key in ("floorMaterial", "wallMaterial", "ceilingMaterial"):
            if key in room:
                room[key] = material(room[key])
        for ceiling in room.get("ceilings", []):
            if "material" in ceiling:
                ceiling["material"] = material(ceiling["material"])
            if "materialProperties" in ceiling:
                ceiling["material"] = material(ceiling.pop("materialProperties"))
        room.setdefault("children", [])

    for wall in data.get("walls", []):
        wall["material"] = material(wall.get("material", "Drywall"))
        if wall.get("id", "").split("|")[1:2] == ["exterior"]:
            wall["roomId"] = "exterior"

    for opening in [*data.get("doors", []), *data.get("windows", [])]:
        if "material" in opening:
            opening["material"] = material(opening["material"])
        if "materialProperties" in opening:
            opening["material"] = material(opening.pop("materialProperties"))
        if "boundingBox" in opening:
            bounds = opening.pop("boundingBox")
            opening["holePolygon"] = [bounds["min"], bounds["max"]]
        if "assetOffset" in opening and "holePolygon" in opening:
            offset = opening.pop("assetOffset")
            low, high = opening["holePolygon"]
            opening["assetPosition"] = {
                "x": low["x"] + offset.get("x", 0) + (high["x"] - low["x"]) / 2.0,
                "y": low["y"] + offset.get("y", 0),
                "z": 0,
            }

    for obj in data.get("objects", []):
        if "material" in obj:
            obj["material"] = material(obj["material"])
        if "materialProperties" in obj:
            obj["material"] = material(obj.pop("materialProperties"))
        if "color" in obj:
            obj["material"] = obj.get("material", material("Drywall"))
            obj["material"]["color"] = obj.pop("color")

    parameters = data.get("proceduralParameters")
    if isinstance(parameters, dict):
        if "ceilingMaterial" in parameters:
            parameters["ceilingMaterial"] = material(parameters["ceilingMaterial"])
        if "ceilingColor" in parameters and isinstance(parameters.get("ceilingMaterial"), dict):
            parameters["ceilingMaterial"]["color"] = parameters.pop("ceilingColor")
        for prefix in ("ceilingMaterial", "floorMaterial", "wallMaterial"):
            x_divisor = parameters.pop(f"{prefix}TilingXDivisor", None)
            y_divisor = parameters.pop(f"{prefix}TilingYDivisor", None)
            target = parameters.get(prefix)
            if isinstance(target, dict):
                if x_divisor is not None:
                    target["tilingDivisorX"] = x_divisor
                if y_divisor is not None:
                    target["tilingDivisorY"] = y_divisor

    metadata = data.setdefault("metadata", {})
    metadata["schema"] = "1.0.0"
    data.setdefault("doors", [])
    data.setdefault("windows", [])
    data.setdefault("wallPolys", [])
    data.setdefault(
        "proceduralParameters",
        {
            "ceilingMaterial": material("Drywall"),
            "lights": [],
            "reflections": [],
            "skyboxId": "Outdoor",
            "receptacleHeight": 0.3,
        },
    )

    agent = metadata.get("agent", {})
    position = agent.get("position", {}) if isinstance(agent, dict) else {}
    rotation = agent.get("rotation", {}) if isinstance(agent, dict) else {}
    if not isinstance(rotation, dict):
        rotation = {"x": 0, "y": rotation, "z": 0}
    data["performerStart"] = {
        "position": {
            "x": position.get("x", 0),
            "y": position.get("y", 0.95),
            "z": position.get("z", 0),
        },
        "rotation": {
            "x": rotation.get("x", 0),
            "y": rotation.get("y", 0),
            "z": rotation.get("z", 0),
        },
    }
    return data


def agent_arguments(house: dict[str, Any]) -> dict[str, Any]:
    agent = house.get("metadata", {}).get("agent", {})
    position = agent.get("position", {})
    rotation = agent.get("rotation", {})
    if isinstance(rotation, dict):
        rotation = rotation.get("y", 0)
    return {
        "x": position.get("x", 0),
        "y": position.get("y", 0.95),
        "z": position.get("z", 0),
        "rotation": rotation,
        "horizon": agent.get("horizon", 30),
        "standing": agent.get("standing", True),
        "renderImage": False,
    }


def validate_house(controller: Any, house: dict[str, Any]) -> dict[str, Any]:
    controller.reset(renderImage=False)
    event = controller.step(action="CreateHouse", house=house, renderImage=False)
    if not event_ok(event):
        return {"valid": False, "stage": "CreateHouse", "error": event_error(event)}

    event = controller.step(action="TeleportFull", **agent_arguments(house))
    if not event_ok(event):
        return {"valid": False, "stage": "TeleportFull", "error": event_error(event)}

    event = controller.step(action="GetReachablePositions", renderImage=False)
    reachable = event.metadata.get("actionReturn") or []
    if not event_ok(event) or not reachable:
        return {
            "valid": False,
            "stage": "GetReachablePositions",
            "error": event_error(event) or "No reachable positions returned",
        }
    return {"valid": True, "reachable_positions": len(reachable)}


def write_json_gz(path: Path, house: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(house, handle, ensure_ascii=True)
    temporary.replace(path)


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def load_manifest(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("records"), list):
                return loaded
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "format": "procthor-compatible-v1",
        "split": args.split,
        "start_seed": args.start_seed,
        "count": args.count,
        "small_objects_skipped": not args.include_small_objects,
        "records": [],
    }


def load_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Scene is not a JSON object: {path}")
    return loaded


def replace_record(manifest: dict[str, Any], record: dict[str, Any]) -> None:
    records = manifest["records"]
    for index, existing in enumerate(records):
        if existing.get("seed") == record["seed"]:
            records[index] = record
            return
    records.append(record)


def main() -> int:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be positive")
    executable = args.local_executable_path or os.environ.get("AI2THOR_EXECUTABLE")
    if not executable:
        raise SystemExit("Pass --local-executable-path or set AI2THOR_EXECUTABLE")

    from ai2thor.controller import Controller
    import procthor.generation as procthor_generation
    from procthor.generation import HouseGenerator, PROCTHOR10K_ROOM_SPEC_SAMPLER

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = load_manifest(manifest_path, args)
    manifest.setdefault("format", "procthor-compatible-v1")
    manifest.setdefault("split", args.split)
    manifest.setdefault("small_objects_skipped", not args.include_small_objects)
    manifest.setdefault("records", [])
    existing_by_seed = {
        record.get("seed"): record
        for record in manifest["records"]
        if isinstance(record, dict) and isinstance(record.get("seed"), int)
    }

    if not args.include_small_objects:
        # The old ProcTHOR small-object stage requires an incompatible CreateHouse schema.
        procthor_generation.add_small_objects = lambda *unused_args, **unused_kwargs: None

    controller = None
    saved_this_run = 0
    try:
        controller = Controller(
            local_executable_path=str(executable),
            scene="Procedural",
            quality="Low",
            width=args.width,
            height=args.height,
            headless=args.headless,
            x_display=args.x_display,
            server_start_timeout=60,
            server_timeout=60,
        )
        for seed in range(args.start_seed, args.start_seed + args.count):
            started = time.monotonic()
            output_path = args.output_dir / f"house_seed_{seed:05d}.json.gz"
            record: dict[str, Any] = {"seed": seed, "path": str(output_path), "saved": False}
            print(f"Generating seed={seed}...", flush=True)
            try:
                if output_path.exists() and not args.overwrite:
                    previous = existing_by_seed.get(seed)
                    if previous and previous.get("saved"):
                        record = dict(previous)
                        record["path"] = str(output_path)
                    else:
                        compatible = load_json_gz(output_path)
                        validation = validate_house(controller, compatible)
                        record.update(
                            {
                                "room_spec_id": compatible.get("metadata", {}).get("roomSpecId"),
                                "rooms": len(compatible.get("rooms", [])),
                                "objects_in_house_json": len(compatible.get("objects", [])),
                                "validation": validation,
                                "saved": validation["valid"],
                                "status": "existing_verified" if validation["valid"] else "existing_invalid",
                            }
                        )
                else:
                    generator = HouseGenerator(
                        split=args.split,
                        seed=seed,
                        room_spec_sampler=PROCTHOR10K_ROOM_SPEC_SAMPLER,
                        controller=controller,
                    )
                    house, _ = generator.sample()
                    compatible = converted_house(house.data)
                    validation = validate_house(controller, compatible)
                    record.update(
                        {
                            "room_spec_id": compatible.get("metadata", {}).get("roomSpecId"),
                            "rooms": len(compatible.get("rooms", [])),
                            "objects_in_house_json": len(compatible.get("objects", [])),
                            "validation": validation,
                        }
                    )
                    if validation["valid"]:
                        write_json_gz(output_path, compatible)
                        record.update({"saved": True, "status": "generated"})
                    else:
                        record["status"] = "validation_failed"
            except Exception as exc:
                record.update({"status": "generation_failed", "error": f"{type(exc).__name__}: {exc}"})
            finally:
                record["elapsed_seconds"] = round(time.monotonic() - started, 3)
                replace_record(manifest, record)
                write_manifest(manifest_path, manifest)
                print(json.dumps(record, ensure_ascii=True), flush=True)
                saved_this_run += int(bool(record["saved"]))
    finally:
        if controller is not None:
            controller.stop()

    total_saved = sum(1 for record in manifest["records"] if record.get("saved"))
    print(
        f"This run saved {saved_this_run}/{args.count}; total saved: {total_saved}. "
        f"Manifest: {manifest_path}",
        flush=True,
    )
    return 0 if saved_this_run == args.count else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
