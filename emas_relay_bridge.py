#!/usr/bin/env python3
"""Bridge EMAS task assignments to the relay task HTTP service.

EMAS owns task decomposition and assignment.  This module translates its
``{agent_id, subtask}`` assignments into serial calls to the existing
``relay_task_server.py`` and returns the execution-report shape consumed by
the EMAS communication adapter.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SUCCESS = "success"
WAIT_RETRY = "wait_retry"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def assignment_list(payload: Any) -> list[dict[str, Any]]:
    """Accept EMAS communication payloads, allocation plans, or a raw unit."""

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        raise ValueError("assignments JSON must be an object or a list")
    if isinstance(payload.get("assignments"), list):
        return [item for item in payload["assignments"] if isinstance(item, dict)]
    if isinstance(payload.get("unit"), list):
        return [item for item in payload["unit"] if isinstance(item, dict)]
    allocation = payload.get("allocation") or []
    if allocation and isinstance(allocation[0], dict) and isinstance(allocation[0].get("unit"), list):
        return [item for item in allocation[0]["unit"] if isinstance(item, dict)]
    raise ValueError("could not find assignments, unit, or allocation[0].unit in JSON")


def subtask_text(subtask: dict[str, Any], root_task: str | None) -> str:
    parts = [str(subtask.get(key) or "").strip() for key in ("description", "name")]
    detail = next((part for part in parts if part), str(subtask.get("action") or "execute").strip())
    if root_task:
        return f"Overall goal: {root_task}\nCurrent assigned subtask: {detail}"
    return detail


class RelayTaskClient:
    def __init__(self, relay_url: str, timeout: float = 180.0) -> None:
        self.execute_url = f"{relay_url.rstrip('/')}/execute_task"
        self.timeout = timeout

    def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(self.execute_url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"relay returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"relay is unreachable at {self.execute_url}: {exc.reason}") from exc
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"relay returned invalid JSON: {body[:500]}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("relay returned a non-object JSON response")
        return parsed


def dispatch_assignments(
    assignments: list[dict[str, Any]],
    *,
    root_task: str | None,
    client: RelayTaskClient,
    known_robot_ids: list[int] | None = None,
    dry_run: bool = False,
    relay_strategy: str = "agent",
    max_replan_steps: int = 10,
    max_actions: int = 8,
) -> dict[str, Any]:
    """Execute one EMAS allocation unit and build an EMAS adapter report."""

    if known_robot_ids is None:
        known_robot_ids = sorted(
            {
                int(item.get("agent_id", 0))
                for item in assignments
                if str(item.get("agent_id", "")).strip().lstrip("-").isdigit()
            }
            or {0}
        )

    started = time.perf_counter()
    task_statuses: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    completed_task_ids: list[str] = []
    feedback: list[dict[str, Any]] = []

    for assignment in assignments:
        subtask = assignment.get("subtask") or {}
        if not isinstance(subtask, dict):
            subtask = {}
        task_id = str(subtask.get("id") or f"emas-task-{len(traces) + 1}")
        try:
            agent_id = int(assignment.get("agent_id", 0))
        except (TypeError, ValueError):
            agent_id = 0
        request_payload = {
            "task_id": task_id,
            "task": subtask_text(subtask, root_task),
            "primary_robot_id": agent_id,
            "known_robot_ids": known_robot_ids,
            "dry_run": dry_run,
            "relay_strategy": relay_strategy,
            "max_replan_steps": max_replan_steps,
            "max_actions": max_actions,
        }
        try:
            relay_response = client.execute(request_payload)
            completed = relay_response.get("status") == SUCCESS
            status = SUCCESS if completed else WAIT_RETRY
            reason = "Relay closed-loop task completed." if completed else str(relay_response.get("error") or "Relay requested upstream planning.")
        except RuntimeError as exc:
            relay_response = {"status": "failed", "error": str(exc)}
            completed = False
            status = WAIT_RETRY
            reason = str(exc)

        if completed:
            completed_task_ids.append(task_id)
        task_statuses.append(
            {
                "subtask_id": task_id,
                "agent_id": str(agent_id),
                "status": status,
                "reason": reason,
                "subtask": subtask,
            }
        )
        traces.append(
            {
                "subtask_id": task_id,
                "agent_id": str(agent_id),
                "subtask": subtask,
                "executor": "multagentai2thor_relay_http",
                "completed": completed,
                "relay_request": request_payload,
                "relay_response": relay_response,
            }
        )
        if not completed:
            feedback.append({"type": "relay_not_completed", "subtask_id": task_id, "message": reason})

    return {
        "execution": {
            "completed_task_ids": completed_task_ids,
            "traces": traces,
            "execution_time_seconds": time.perf_counter() - started,
        },
        "task_statuses": task_statuses,
        "agent_states": [],
        "object_changes": [],
        "feedback": feedback,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dispatch an EMAS allocation unit through relay_task_server.py")
    parser.add_argument("--assignments-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default=None, help="Optional overall task text used as relay context.")
    parser.add_argument("--relay-url", default="http://127.0.0.1:18080")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--known-robot-ids", default=None, help="Comma-separated robot IDs. Defaults to IDs in the assignment unit.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--relay-strategy", choices=["agent", "rules"], default="agent")
    parser.add_argument("--max-replan-steps", type=int, default=10)
    parser.add_argument("--max-actions", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source = load_json(args.assignments_json)
    assignments = assignment_list(source)
    task = args.task or (source.get("task") if isinstance(source, dict) else None)
    robot_ids = None
    if args.known_robot_ids:
        robot_ids = [int(value.strip()) for value in args.known_robot_ids.split(",") if value.strip()]
    report = dispatch_assignments(
        assignments,
        root_task=task,
        client=RelayTaskClient(args.relay_url, timeout=args.timeout),
        known_robot_ids=robot_ids,
        dry_run=args.dry_run,
        relay_strategy=args.relay_strategy,
        max_replan_steps=args.max_replan_steps,
        max_actions=args.max_actions,
    )
    save_json(report, args.output)
    print(json.dumps({"output": str(args.output), "completed_task_ids": report["execution"]["completed_task_ids"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
