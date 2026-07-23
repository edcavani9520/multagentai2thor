#!/usr/bin/env python3
"""Run a lightweight EMAS planning -> relay -> AI2-THOR smoke test.

The full EMAS hybrid loop currently constructs ConceptGraphs in its own tree.
This runner instead converts the receiver's object metadata into the small
subgraph schema needed by EMAS task decomposition, then uses EMAS allocation
and this repository's HTTP relay bridge. It is intended to validate the cross
repository contract before replacing EMAS's file adapter in production.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from emas_relay_bridge import RelayTaskClient, dispatch_assignments, save_json


DEFAULT_EMAS_ROOT = Path("/home/kinova-1/EMAS/225010231/mwl/EMAS")


def get_json(url: str, timeout: float) -> dict[str, Any]:
    with urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object from {url}")
    return payload


def receiver_subgraph(state: dict[str, Any], task: str) -> dict[str, Any]:
    nodes = []
    for index, obj in enumerate(state.get("objects") or []):
        if not isinstance(obj, dict):
            continue
        object_id = str(obj.get("id") or f"object-{index}")
        object_type = str(obj.get("type") or "object")
        nodes.append(
            {
                "pruned_id": index,
                "original_id": object_id,
                "object_tag": object_type,
                "caption": object_type,
                "possible_tags": [object_type],
                "position": obj.get("position"),
                "visible": obj.get("visible"),
            }
        )
    return {"task": task, "task_spec": {"source": "ai2thor_receiver_state"}, "seed_nodes": [], "nodes": nodes, "triples": []}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test EMAS planning through relay_task_server.py")
    parser.add_argument("--task", required=True)
    parser.add_argument("--emas-root", type=Path, default=DEFAULT_EMAS_ROOT)
    parser.add_argument("--receiver-url", default="http://127.0.0.1:19000")
    parser.add_argument("--relay-url", default="http://127.0.0.1:18080")
    parser.add_argument("--primary-robot-id", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--relay-strategy", choices=["agent", "rules"], default="rules")
    parser.add_argument("--max-actions", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.emas_root.is_dir():
        raise SystemExit(f"EMAS root does not exist: {args.emas_root}")
    sys.path.insert(0, str(args.emas_root))
    from planning.task_graph import decompose_task_to_graph
    from planning.utils.task_allocation import allocate_subtasks_to_agents

    receiver = args.receiver_url.rstrip("/")
    state = get_json(f"{receiver}/state?robot_id={args.primary_robot_id}", args.timeout)
    robots = get_json(f"{receiver}/robots", args.timeout).get("robots") or []
    robot_ids = [int(robot["robot_id"]) for robot in robots if isinstance(robot, dict) and "robot_id" in robot]
    if not robot_ids:
        robot_ids = [args.primary_robot_id]

    subgraph = receiver_subgraph(state, args.task)
    task_graph = decompose_task_to_graph(args.task, subgraph, use_qwen=False)
    allocation = allocate_subtasks_to_agents(
        task_graph,
        metadata_source={"agents": [{"agent_id": str(robot_id), "status": "idle"} for robot_id in robot_ids]},
        scene_context=subgraph,
        use_qwen=False,
    )
    assignments = list((allocation.get("allocation") or [{}])[0].get("unit") or [])
    report = dispatch_assignments(
        assignments,
        root_task=args.task,
        client=RelayTaskClient(args.relay_url, timeout=args.timeout),
        known_robot_ids=robot_ids,
        dry_run=args.dry_run,
        relay_strategy=args.relay_strategy,
        max_actions=args.max_actions,
    )
    result = {
        "receiver_scene": state.get("sceneName"),
        "subgraph": subgraph,
        "task_graph": task_graph,
        "allocation": allocation,
        "report": report,
    }
    save_json(result, args.output)
    print(json.dumps({
        "output": str(args.output),
        "planner_backend": task_graph.get("planner_backend"),
        "assigned_subtasks": [str((item.get("subtask") or {}).get("id")) for item in assignments],
        "completed_task_ids": report["execution"]["completed_task_ids"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
