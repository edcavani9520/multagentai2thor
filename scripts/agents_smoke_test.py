#!/usr/bin/env python3
"""Smoke-test the EMAS agents runtime without requiring curl."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


AGENTS_ROOT = Path(__file__).resolve().parents[1]
EMBODIED_ROOT = AGENTS_ROOT / "EmbodiedGPT_Pytorch"
DEFAULT_EXECUTE_ACTIONS_URL = "http://10.20.18.3:19000/execute_actions"


def json_request(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 20.0) -> tuple[int, dict[str, Any]]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            status = response.status
            response_body = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        status = exc.code
        response_body = exc.read().decode("utf-8", errors="replace")
    except URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc
    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{method} {url} returned non-JSON body: {response_body[:500]}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{method} {url} returned non-object JSON")
    return status, parsed


def base_url_from_execute_actions_url(url: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def endpoint_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")


def print_check(name: str, ok: bool, detail: str) -> None:
    marker = "OK" if ok else "FAIL"
    print(f"[{marker}] {name}: {detail}")


def check_wrapper_path() -> bool:
    script = EMBODIED_ROOT / "auto_scene_actions.sh"
    if not script.exists():
        print_check("wrapper_path", False, f"missing {script}")
        return False
    proc = subprocess.run(
        ["bash", "-x", str(script), "--help"],
        cwd=EMBODIED_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )
    output = proc.stdout
    expected = f"+ cd {EMBODIED_ROOT}"
    old_path = "/225010231/mwl/Linhao/EmbodiedGPT_Pytorch"
    ok = expected in output and old_path not in output
    detail = "uses script directory" if ok else "wrapper may be using an unexpected directory"
    print_check("wrapper_path", ok, detail)
    if not ok:
        print(output[-2000:])
    return ok


def summarize_endpoint_response(data: dict[str, Any]) -> str:
    pieces = []
    for key in ("status", "service", "sceneName", "planner", "error_code", "error"):
        if key in data:
            pieces.append(f"{key}={data[key]}")
    if isinstance(data.get("robots"), list):
        pieces.append(f"robots={len(data['robots'])}")
    if isinstance(data.get("positions"), list):
        pieces.append(f"positions={len(data['positions'])}")
    if isinstance(data.get("actions"), list):
        pieces.append(f"actions={len(data['actions'])}")
    if isinstance(data.get("results"), list):
        pieces.append(f"results={len(data['results'])}")
    return ", ".join(pieces) or "JSON object"


def check_receiver(base_url: str, robot_id: int, target: str, timeout: float) -> bool:
    checks = [
        ("health", "GET", endpoint_url(base_url, "/health"), None, {200}),
        ("state", "GET", endpoint_url(base_url, "/state"), None, {200}),
        (
            "reachable_positions",
            "GET",
            endpoint_url(base_url, f"/reachable_positions?robot_id={robot_id}"),
            None,
            {200},
        ),
        (
            "execute_actions",
            "POST",
            endpoint_url(base_url, "/execute_actions"),
            {
                "task_id": f"agents-smoke-pass-{uuid.uuid4()}",
                "robot_id": robot_id,
                "stop_on_failure": False,
                "actions": [{"action": "Pass"}],
            },
            {200},
        ),
        (
            "goto_dry_run",
            "POST",
            endpoint_url(base_url, "/goto"),
            {
                "task_id": f"agents-smoke-goto-{uuid.uuid4()}",
                "robot_id": robot_id,
                "object_type": target,
                "execute": False,
            },
            {200, 400},
        ),
    ]
    all_ok = True
    for name, method, url, payload, expected_statuses in checks:
        try:
            status, data = json_request(method, url, payload, timeout=timeout)
            ok = status in expected_statuses
            if name == "health":
                ok = ok and data.get("status") == "ok"
            if name == "state":
                ok = ok and isinstance(data.get("robots"), list)
            if name == "reachable_positions":
                ok = ok and data.get("status") == "success" and isinstance(data.get("positions"), list)
            if name == "execute_actions":
                ok = ok and data.get("status") in {"success", "partial"}
            if name == "goto_dry_run":
                ok = ok and data.get("status") in {"success", "failed"}
            detail = f"HTTP {status}; {summarize_endpoint_response(data)}"
            if name == "health" and not ok and status == 404:
                detail += "; receiver is probably running older code, restart ai2thor_receiver_server.py"
            print_check(name, ok, detail)
        except Exception as exc:
            ok = False
            print_check(name, False, str(exc))
        all_ok = all_ok and ok
    return all_ok


def run_navigation_cli(execute_actions_url: str, task: str, robot_id: int, timeout: float) -> bool:
    cmd = [
        str(EMBODIED_ROOT / "auto_scene_actions.sh"),
        "--task",
        task,
        "--primary-robot-id",
        str(robot_id),
        "--relay-mode",
        "--closed-loop-replan",
        "--print-raw-output",
        "--dry-run",
    ]
    env = os.environ.copy()
    env["SEND_ACTIONS_URL"] = execute_actions_url
    proc = subprocess.run(
        cmd,
        cwd=EMBODIED_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    combined = proc.stdout + "\n" + proc.stderr
    ok = proc.returncode == 0 and '"task_intent_source": "navigation_goto_intent"' in proc.stdout
    print_check("navigation_cli_dry_run", ok, f"returncode={proc.returncode}")
    if not ok:
        print(combined[-3000:])
    return ok


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test the EMAS agents runtime.")
    parser.add_argument("--execute-actions-url", default=DEFAULT_EXECUTE_ACTIONS_URL)
    parser.add_argument("--robot-id", type=int, default=0)
    parser.add_argument("--goto-target", default="Fridge")
    parser.add_argument("--task", default="go to fridge.")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--skip-cli", action="store_true", help="Only check receiver HTTP endpoints.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    base_url = base_url_from_execute_actions_url(args.execute_actions_url)
    ok = True
    ok = check_wrapper_path() and ok
    ok = check_receiver(base_url, args.robot_id, args.goto_target, args.timeout) and ok
    if not args.skip_cli:
        ok = run_navigation_cli(args.execute_actions_url, args.task, args.robot_id, args.timeout) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
