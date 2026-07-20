#!/usr/bin/env python3
"""HTTP wrapper for EmbodiedGPT's multi-robot relay closed-loop runtime."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import threading
import traceback
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


MAX_REQUEST_BYTES = 128 * 1024
SERVICE_NAME = "relay_task_server"
SERVICE_REVISION = "2026-07-19-semantic-placeholder-v4"
REPO_ROOT = Path(__file__).resolve().parent
EMBODIED_ROOT = REPO_ROOT / "EmbodiedGPT_Pytorch"


def log(message: str) -> None:
    print(f"[RELAY TASK] {message}", flush=True)


def _last_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append((index, index + end, value))
    if not candidates:
        raise RuntimeError("relay runtime did not produce a JSON result")
    return max(candidates, key=lambda item: (item[1] - item[0], item[1]))[2]


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _robot_ids(value: Any) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ValueError("known_robot_ids must be a non-empty JSON array of integers")
    result: list[int] = []
    for robot_id in value:
        if not isinstance(robot_id, int) or isinstance(robot_id, bool) or robot_id < 0:
            raise ValueError("known_robot_ids must contain non-negative integers")
        if robot_id not in result:
            result.append(robot_id)
    return result


@dataclass(frozen=True)
class RelayRuntimeConfig:
    receiver_url: str
    model_path: str
    device: str
    device_map: str
    dtype: str
    max_new_tokens: int
    temperature: float
    send_timeout: float
    output_dir: Path
    max_replan_steps: int
    relay_agent_max_turns: int
    max_actions: int


class RelayTaskService:
    """Runs one task at a time because one Qwen backend is shared."""

    def __init__(self, engine: Any, backend: Any, config: RelayRuntimeConfig):
        self.engine = engine
        self.backend = backend
        self.config = config
        self.lock = threading.Lock()

    def health(self) -> dict[str, Any]:
        relay_agent_module = sys.modules.get("demo.relay_agent")
        return {
            "status": "ok",
            "service": SERVICE_NAME,
            "service_revision": SERVICE_REVISION,
            "receiver_url": self.config.receiver_url,
            "model_path": self.config.model_path,
            "device": self.config.device,
            "relay_mode": True,
            "closed_loop_replan": True,
            "model_loaded": bool(getattr(self.backend, "model", None)),
            "relay_agent_module": str(getattr(relay_agent_module, "__file__", "not-loaded")),
        }

    def execute_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        task = payload.get("task", payload.get("instruction", payload.get("prompt")))
        if not isinstance(task, str) or not task.strip():
            raise ValueError("missing non-empty task (or instruction/prompt)")

        task_id = str(payload.get("task_id") or uuid.uuid4())
        primary_robot_id = payload.get("primary_robot_id", payload.get("robot_id", 0))
        if not isinstance(primary_robot_id, int) or isinstance(primary_robot_id, bool) or primary_robot_id < 0:
            raise ValueError("primary_robot_id must be a non-negative integer")
        known_robot_ids = _robot_ids(payload.get("known_robot_ids"))
        dry_run = bool(payload.get("dry_run", False))
        max_replan_steps = _positive_int(
            payload.get("max_replan_steps", self.config.max_replan_steps), "max_replan_steps"
        )
        relay_agent_max_turns = _positive_int(
            payload.get("relay_agent_max_turns", self.config.relay_agent_max_turns),
            "relay_agent_max_turns",
        )
        max_actions = _nonnegative_int(payload.get("max_actions", self.config.max_actions), "max_actions")
        relay_strategy = payload.get("relay_strategy", "agent")
        if relay_strategy not in {"agent", "rules"}:
            raise ValueError("relay_strategy must be 'agent' or 'rules'")

        execute_actions_url = f"{self.config.receiver_url.rstrip('/')}/execute_actions"
        argv = [
            "--execute-actions-url", execute_actions_url,
            "--task", task.strip(),
            "--task-id", task_id,
            "--output-dir", str(self.config.output_dir),
            "--send-timeout", str(self.config.send_timeout),
            "--qwen-model", self.config.model_path,
            "--qwen-device-map", self.config.device_map,
            "--qwen-dtype", self.config.dtype,
            "--device", self.config.device,
            "--max-new-tokens", str(self.config.max_new_tokens),
            "--temperature", str(self.config.temperature),
            "--max-actions", str(max_actions),
            "--save-raw-output",
            "--primary-robot-id", str(primary_robot_id),
            "--relay-mode",
            "--relay-strategy", relay_strategy,
            "--relay-agent-max-turns", str(relay_agent_max_turns),
            "--closed-loop-replan",
            "--max-replan-steps", str(max_replan_steps),
        ]
        if known_robot_ids is not None:
            argv.extend(["--known-robot-ids", ",".join(str(robot_id) for robot_id in known_robot_ids)])
        if dry_run:
            argv.append("--dry-run")

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        args = self.engine.parse_args(argv)
        setattr(args, "_qwen_backend", self.backend)
        stdout = io.StringIO()
        stderr = io.StringIO()
        with self.lock, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = self.engine.run(args)
        runtime_stdout = stdout.getvalue().strip()
        runtime_stderr = stderr.getvalue().strip()
        try:
            result = _last_json_object(runtime_stdout)
        except RuntimeError as exc:
            detail = runtime_stderr or runtime_stdout or "no diagnostic output"
            raise RuntimeError(
                f"relay runtime exited with code {exit_code} before producing a JSON result: {detail}"
            ) from exc
        closed_loop = result.get("closed_loop_result")
        closed_loop_status = closed_loop.get("status") if isinstance(closed_loop, dict) else None
        status = "success" if exit_code == 0 and closed_loop_status == "success" else "needs_upstream_planning"
        response: dict[str, Any] = {
            "status": status,
            "task_id": task_id,
            "dry_run": dry_run,
            "result": result,
        }
        if runtime_stderr:
            response["runtime_log"] = runtime_stderr
        return response


task_service: RelayTaskService | None = None


class Handler(BaseHTTPRequestHandler):
    server_version = "RelayTaskServer/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        log(f"HTTP {self.address_string()} - {fmt % args}")

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status_code)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length > MAX_REQUEST_BYTES:
            raise ValueError(f"request is too large (max {MAX_REQUEST_BYTES} bytes)")
        body = self.rfile.read(length)
        if not body:
            return {}
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("request JSON must be an object")
        return payload

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        if urlparse(self.path).path in {"/", "/health"}:
            self._send_json(200, task_service.health())
        else:
            self._send_json(404, {"status": "failed", "error": "not_found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/execute_task":
            self._send_json(404, {"status": "failed", "error": "not_found"})
            return
        try:
            payload = self._read_json()
            log(f"POST /execute_task task_id={payload.get('task_id', 'generated')}")
            self._send_json(200, task_service.execute_task(payload))
        except ValueError as exc:
            self._send_json(400, {"status": "failed", "error": str(exc)})
        except Exception as exc:
            log(f"/execute_task failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            self._send_json(502, {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    global task_service
    parser = argparse.ArgumentParser(description="EmbodiedGPT relay closed-loop task service for AI2-THOR")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--receiver-url", default="http://127.0.0.1:19000")
    parser.add_argument("--model-path", default="models/Qwen3.5-4B")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--send-timeout", type=float, default=60.0)
    parser.add_argument("--output", type=Path, default=Path("output/relay_tasks"))
    parser.add_argument("--max-replan-steps", type=int, default=10)
    parser.add_argument("--relay-agent-max-turns", type=int, default=8)
    parser.add_argument("--max-actions", type=int, default=8)
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.temperature <= 0 or args.send_timeout <= 0:
        parser.error("--max-new-tokens, --temperature, and --send-timeout must be positive")
    if args.max_replan_steps <= 0 or args.relay_agent_max_turns <= 0 or args.max_actions < 0:
        parser.error("invalid relay/action limits")
    if not EMBODIED_ROOT.is_dir():
        parser.error(f"EmbodiedGPT runtime is missing: {EMBODIED_ROOT}")

    sys.path.insert(0, str(EMBODIED_ROOT))
    from demo import auto_scene_actions
    from demo.qwen35_backend import Qwen35Backend, Qwen35Config

    config = RelayRuntimeConfig(
        receiver_url=args.receiver_url,
        model_path=args.model_path,
        device=args.device,
        device_map=args.device_map,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        send_timeout=args.send_timeout,
        output_dir=args.output,
        max_replan_steps=args.max_replan_steps,
        relay_agent_max_turns=args.relay_agent_max_turns,
        max_actions=args.max_actions,
    )
    backend = Qwen35Backend(
        Qwen35Config(
            model_name=config.model_path,
            device=config.device,
            device_map=config.device_map,
            torch_dtype=config.dtype,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
        )
    )
    task_service = RelayTaskService(auto_scene_actions, backend, config)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    log(f"listening at http://{args.host}:{args.port}")
    log(f"revision: {SERVICE_REVISION}; relay agent: {sys.modules['demo.relay_agent'].__file__}")
    log(f"receiver: {config.receiver_url}; endpoint: POST /execute_task")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("KeyboardInterrupt: shutting down")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
