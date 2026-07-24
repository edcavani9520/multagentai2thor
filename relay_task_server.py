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
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
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

SUPPORTED_NORMALIZED_ACTIONS = {"GotoObject", "PickupObject", "PutObject", "OpenObject", "CloseObject", "Done"}
TASK_NORMALIZER_TOOL_NAME = "normalize_incoming_task"
TASK_NORMALIZER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": TASK_NORMALIZER_TOOL_NAME,
        "description": (
            "Normalize an upstream planning subtask into one concise robot task and structured task intent "
            "that the agents runtime can execute. Use only supported actions and choose object types from "
            "the provided AI2-THOR object types."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "normalized_task": {"type": "string"},
                "requestedAction": {"type": "string", "enum": sorted(SUPPORTED_NORMALIZED_ACTIONS)},
                "requestedObjectType": {"type": ["string", "null"]},
                "requestedTargetType": {"type": ["string", "null"]},
                "intentSteps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "order": {"type": "integer"},
                            "action": {"type": "string", "enum": sorted(SUPPORTED_NORMALIZED_ACTIONS)},
                            "objectType": {"type": ["string", "null"]},
                            "targetType": {"type": ["string", "null"]},
                        },
                        "required": ["order", "action", "objectType", "targetType"],
                    },
                },
                "action": {"type": "string", "enum": sorted(SUPPORTED_NORMALIZED_ACTIONS)},
                "object_type": {"type": ["string", "null"]},
                "target_type": {"type": ["string", "null"]},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "reason": {"type": "string"},
            },
            "required": [
                "normalized_task",
                "requestedAction",
                "requestedObjectType",
                "requestedTargetType",
                "intentSteps",
                "confidence",
                "reason",
            ],
        },
    },
}


def log(message: str) -> None:
    print(f"[RELAY TASK] {message}", flush=True)



def _json_values_from_text(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    values: list[Any] = []
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        values.append(value)
    return values


def _tool_call_from_value(value: Any) -> dict[str, Any] | None:
    if isinstance(value, list):
        for item in value:
            tool_call = _tool_call_from_value(item)
            if tool_call is not None:
                return tool_call
        return None
    if not isinstance(value, dict):
        return None
    if isinstance(value.get("tool_calls"), list):
        for item in value["tool_calls"]:
            tool_call = _tool_call_from_value(item)
            if tool_call is not None:
                return tool_call
    function_value = value.get("function")
    if isinstance(function_value, dict):
        name = function_value.get("name") or value.get("name")
        arguments = function_value.get("arguments", value.get("arguments", {}))
    else:
        name = value.get("name") or value.get("tool_name")
        arguments = value.get("arguments", value.get("parameters", {}))
    if not isinstance(name, str) or not name.strip():
        return None
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"raw": arguments}
    if not isinstance(arguments, dict):
        arguments = {}
    return {"name": name.strip(), "arguments": arguments}


def parse_task_normalizer_tool_call(output: str) -> dict[str, Any]:
    for value in _json_values_from_text(output):
        tool_call = _tool_call_from_value(value)
        if tool_call is not None:
            return tool_call
    raise ValueError("Qwen output did not contain a valid task normalization tool call")


def _object_type_from_object(item: Any) -> str | None:
    if not isinstance(item, dict):
        return None
    for key in ("objectType", "object_type", "type"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    object_id = item.get("objectId") or item.get("object_id")
    if isinstance(object_id, str) and "|" in object_id:
        head = object_id.split("|", 1)[0].strip()
        return head or None
    return None


def extract_state_object_types(state: dict[str, Any]) -> list[str]:
    object_types: list[str] = []
    def add(value: str | None) -> None:
        if value and value not in object_types:
            object_types.append(value)
    for item in state.get("objects") or []:
        add(_object_type_from_object(item))
    for robot in state.get("robots") or []:
        if not isinstance(robot, dict):
            continue
        for key in ("visible_objects", "objects", "inventory"):
            for item in robot.get(key) or []:
                add(_object_type_from_object(item))
    return sorted(object_types)


def fetch_receiver_state_object_types(receiver_url: str, timeout: float) -> tuple[list[str], str | None]:
    state_url = f"{receiver_url.rstrip('/')}/state"
    try:
        with urlopen(state_url, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
        parsed = json.loads(body)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return [], f"could not read receiver /state for task normalization: {type(exc).__name__}: {exc}"
    if not isinstance(parsed, dict):
        return [], "receiver /state did not return a JSON object"
    return extract_state_object_types(parsed), None


def task_normalizer_messages(task: str, object_types: list[str]) -> list[dict[str, Any]]:
    supported = ", ".join(sorted(SUPPORTED_NORMALIZED_ACTIONS))
    objects = ", ".join(object_types) if object_types else "(none)"
    prompt = (
        "You are the agents module boundary normalizer. Convert an upstream planning subtask into one concise "
        "natural-language robot task and structured task_intent for the agents runtime.\n"
        "Rules:\n"
        "- Call normalize_incoming_task exactly once.\n"
        "- Use only these actions: " + supported + ".\n"
        "- requestedObjectType, requestedTargetType, objectType, and targetType must be null or exactly one of the current AI2-THOR object types.\n"
        "- Fill intentSteps with every required high-level step in order. Multi-step tasks must not be collapsed.\n"
        "- If the planning subtask is find/search/inspect a target object, normalize it as navigation to that object using GotoObject.\n"
        "- Do not invent object types. If no suitable object type exists, use confidence low and keep the task close to the original.\n\n"
        f"Current AI2-THOR object types: {objects}\n\n"
        f"Incoming task: {task.strip()}"
    )
    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]



def task_normalizer_json_messages(task: str, object_types: list[str]) -> list[dict[str, Any]]:
    supported = ", ".join(sorted(SUPPORTED_NORMALIZED_ACTIONS))
    objects = ", ".join(object_types) if object_types else "(none)"
    prompt = (
        "Normalize this upstream planning subtask into one concise robot task and structured task_intent for the agents runtime. "
        "Return only one JSON object and no prose.\n"
        "JSON keys: normalized_task, requestedAction, requestedObjectType, requestedTargetType, intentSteps, confidence, reason.\n"
        f"Supported actions: {supported}.\n"
        "Object fields must be null or exactly one of the current AI2-THOR object types.\n"
        "intentSteps must contain every high-level step in order and use keys order, action, objectType, targetType.\n"
        "If the planning subtask is find/search/inspect a target object, normalize it as navigation to that object using GotoObject.\n"
        "Do not invent object types.\n\n"
        f"Current AI2-THOR object types: {objects}\n\n"
        f"Incoming task: {task.strip()}"
    )
    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]


def generate_normalizer_json_fallback(backend: Any, task: str, object_types: list[str]) -> dict[str, Any] | None:
    if not hasattr(backend, "generate_messages"):
        return None
    try:
        try:
            output = backend.generate_messages(task_normalizer_json_messages(task, object_types), deterministic=True)
        except TypeError:
            output = backend.generate_messages(task_normalizer_json_messages(task, object_types))
    except Exception:
        return None
    for value in _json_values_from_text(str(output)):
        if isinstance(value, dict) and any(key in value for key in ("normalized_task", "action", "object_type")):
            return value
    return None


def _first_present(arguments: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in arguments:
            return arguments.get(key)
    return None


def _normalize_confidence(value: Any) -> tuple[str, str | None]:
    if isinstance(value, str):
        confidence = value.strip().lower()
        if confidence in {"high", "medium", "low"}:
            return confidence, None
        try:
            value = float(confidence)
        except ValueError:
            return "low", f"normalizer returned invalid confidence {value!r}"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        score = float(value)
        if score >= 0.75:
            return "high", None
        if score >= 0.4:
            return "medium", None
        return "low", None
    return "low", f"normalizer returned invalid confidence {value!r}"


def _canonical_task_from_normalized(action: Any, object_type: Any, target_type: Any) -> str | None:
    if action == "Done":
        return "done."
    if not isinstance(object_type, str) or not object_type:
        return None
    if action == "GotoObject":
        return f"go to the {object_type}."
    if action == "PickupObject":
        return f"pick up the {object_type}."
    if action == "OpenObject":
        return f"open the {object_type}."
    if action == "CloseObject":
        return f"close the {object_type}."
    if action == "PutObject" and isinstance(target_type, str) and target_type:
        return f"put the {object_type} on the {target_type}."
    return None


def _object_type_is_valid(value: Any, object_types: list[str]) -> bool:
    return value is None or (isinstance(value, str) and value in set(object_types))


def _normalize_intent_steps(arguments: dict[str, Any], action: Any, object_type: Any, target_type: Any, object_types: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    raw_steps = _first_present(arguments, "intentSteps", "intent_steps")
    steps: list[dict[str, Any]] = []
    if isinstance(raw_steps, list):
        for index, raw_step in enumerate(raw_steps, start=1):
            if not isinstance(raw_step, dict):
                warnings.append(f"intentSteps[{index - 1}] must be an object")
                continue
            step_action = _first_present(raw_step, "action", "requestedAction")
            step_object = _first_present(raw_step, "objectType", "object_type", "requestedObjectType")
            step_target = _first_present(raw_step, "targetType", "target_type", "requestedTargetType")
            order = raw_step.get("order", index)
            if not isinstance(order, int) or isinstance(order, bool) or order <= 0:
                warnings.append(f"intentSteps[{index - 1}] returned invalid order {order!r}; using {index}")
                order = index
            if step_action not in SUPPORTED_NORMALIZED_ACTIONS:
                warnings.append(f"intentSteps[{index - 1}] returned unsupported action {step_action!r}")
            for field, value in (("objectType", step_object), ("targetType", step_target)):
                if not _object_type_is_valid(value, object_types):
                    warnings.append(f"intentSteps[{index - 1}] returned {field} not present in receiver state: {value!r}")
            steps.append({"order": order, "action": step_action, "objectType": step_object, "targetType": step_target})
    elif raw_steps is not None:
        warnings.append("intentSteps must be a list")

    if not steps and action in SUPPORTED_NORMALIZED_ACTIONS:
        steps.append({"order": 1, "action": action, "objectType": object_type, "targetType": target_type})
    if not steps:
        warnings.append("normalizer omitted usable intentSteps")
    return steps, warnings


def validate_task_normalization(arguments: dict[str, Any], object_types: list[str]) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    normalized_task = _first_present(arguments, "normalized_task", "normalizedTask", "task")
    requested_action = _first_present(arguments, "requestedAction", "requested_action", "action")
    requested_object = _first_present(arguments, "requestedObjectType", "requested_object_type", "object_type", "objectType")
    requested_target = _first_present(arguments, "requestedTargetType", "requested_target_type", "target_type", "targetType")
    confidence_value = arguments.get("confidence")
    reason = arguments.get("reason")
    if not isinstance(normalized_task, str) or not normalized_task.strip():
        warnings.append("normalizer omitted non-empty normalized_task")
        normalized_task = ""
    if requested_action not in SUPPORTED_NORMALIZED_ACTIONS:
        warnings.append(f"normalizer returned unsupported action {requested_action!r}")
    confidence, confidence_warning = _normalize_confidence(confidence_value)
    if confidence_warning:
        warnings.append(confidence_warning)
    for field, value in (("requestedObjectType", requested_object), ("requestedTargetType", requested_target)):
        if not _object_type_is_valid(value, object_types):
            warnings.append(f"normalizer returned {field} not present in receiver state: {value!r}")

    steps, step_warnings = _normalize_intent_steps(arguments, requested_action, requested_object, requested_target, object_types)
    warnings.extend(step_warnings)
    primary_step = next((step for step in steps if step.get("action") in SUPPORTED_NORMALIZED_ACTIONS), None)
    if primary_step is not None:
        if requested_action not in SUPPORTED_NORMALIZED_ACTIONS:
            requested_action = primary_step.get("action")
        if requested_object is None:
            requested_object = primary_step.get("objectType")
        if requested_target is None:
            requested_target = primary_step.get("targetType")

    canonical_task = _canonical_task_from_normalized(requested_action, requested_object, requested_target)
    if canonical_task and (not normalized_task or normalized_task.strip() == requested_action):
        normalized_task = canonical_task
    task_intent = {
        "requestedAction": requested_action,
        "requestedObjectType": requested_object,
        "requestedTargetType": requested_target,
        "intentSteps": steps,
    }
    normalized = {
        "normalized_task": normalized_task.strip() if isinstance(normalized_task, str) else "",
        "action": requested_action,
        "object_type": requested_object,
        "target_type": requested_target,
        "requestedAction": requested_action,
        "requestedObjectType": requested_object,
        "requestedTargetType": requested_target,
        "intentSteps": steps,
        "task_intent": task_intent,
        "confidence": confidence,
        "reason": reason if isinstance(reason, str) else "",
    }
    return normalized, warnings

def normalize_incoming_task_with_backend(backend: Any, task: str, object_types: list[str]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "original_task": task,
        "normalized_task": task,
        "used": False,
        "available_object_types": object_types,
        "warnings": [],
    }
    if not hasattr(backend, "generate_with_tools"):
        record["warnings"].append("backend does not support tool-based task normalization")
        return record
    try:
        output = backend.generate_with_tools(task_normalizer_messages(task, object_types), [TASK_NORMALIZER_TOOL_SCHEMA])
        tool_call = parse_task_normalizer_tool_call(str(output).strip())
    except Exception as exc:
        record["warnings"].append(f"task normalization failed: {type(exc).__name__}: {exc}")
        return record
    if tool_call.get("name") != TASK_NORMALIZER_TOOL_NAME:
        record["warnings"].append(f"normalizer called unexpected tool {tool_call.get('name')!r}")
        return record
    arguments = tool_call.get("arguments") if isinstance(tool_call.get("arguments"), dict) else {}
    normalized, warnings = validate_task_normalization(arguments, object_types)
    source = "qwen_tool_call"
    if warnings:
        fallback_arguments = generate_normalizer_json_fallback(backend, task, object_types)
        if fallback_arguments is not None:
            fallback_normalized, fallback_warnings = validate_task_normalization(fallback_arguments, object_types)
            if not fallback_warnings:
                normalized, warnings = fallback_normalized, []
                source = "qwen_json_fallback"
            else:
                warnings.extend(f"json fallback: {warning}" for warning in fallback_warnings)
        else:
            warnings.append("json fallback was unavailable or did not return a valid JSON object")
    record.update(normalized)
    record["source"] = source
    record["warnings"].extend(warnings)
    if not warnings and normalized.get("confidence") in {"high", "medium"}:
        record["normalized_task"] = normalized["normalized_task"]
        record["used"] = normalized["normalized_task"] != task
    else:
        record["candidate_normalized_task"] = normalized.get("normalized_task") or ""
        record["normalized_task"] = task
        record["used"] = False
    return record


def missing_model_shards(model_path: str) -> list[str]:
    path = Path(model_path).expanduser()
    if not path.is_dir():
        return []
    index_path = path / "model.safetensors.index.json"
    if not index_path.exists():
        return []
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        return []
    missing = []
    for shard in sorted({str(value) for value in weight_map.values() if isinstance(value, str)}):
        if not (path / shard).exists():
            missing.append(shard)
    return missing


def model_shard_error(model_path: str) -> str | None:
    missing = missing_model_shards(model_path)
    if not missing:
        return None
    return (
        f"model path {model_path!r} is incomplete; missing safetensors shard(s): {', '.join(missing)}. "
        "Use a complete Qwen3.5-4B directory, e.g. /225010231/mwl/Linhao/models/Qwen3.5-4B."
    )


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
        if hasattr(self.backend, "generate_with_tools"):
            state_object_types, state_warning = fetch_receiver_state_object_types(
                self.config.receiver_url,
                self.config.send_timeout,
            )
        else:
            state_object_types, state_warning = [], None
        task_normalization = normalize_incoming_task_with_backend(self.backend, task.strip(), state_object_types)
        if state_warning:
            task_normalization.setdefault("warnings", []).append(state_warning)
        executable_task = str(task_normalization.get("normalized_task") or task).strip()
        task_intent_json = None
        if (
            isinstance(task_normalization.get("task_intent"), dict)
            and not task_normalization.get("warnings")
            and task_normalization.get("confidence") in {"high", "medium"}
        ):
            task_intent_json = json.dumps(
                {
                    "task_intent": task_normalization["task_intent"],
                    "task_intent_source": "qwen_normalizer_tool_call",
                    "task_normalization": task_normalization,
                },
                ensure_ascii=False,
            )
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
            "--task", executable_task,
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
        if task_intent_json is not None:
            argv.extend(["--task-intent-json", task_intent_json])
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
            "task_normalization": task_normalization,
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
    shard_error = model_shard_error(args.model_path)
    if shard_error is not None:
        parser.error(shard_error)

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
