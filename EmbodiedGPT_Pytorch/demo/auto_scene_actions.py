from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOCAL_HF_CACHE = REPO_ROOT.parent / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(LOCAL_HF_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(LOCAL_HF_CACHE / "transformers"))

from demo.plan_media import (  # noqa: E402
    DEFAULT_HTTP_TIMEOUT,
    DEFAULT_QWEN35_MODEL,
    parse_semantic_planning_output,
    positive_float,
    positive_int,
    semantic_planning_prompt,
    send_actions,
)
from demo.relay_agent import (  # noqa: E402
    RelayAgentConfig,
    run_relay_agent,
)


DEFAULT_EXECUTE_ACTIONS_URL = os.environ.get(
    "SEND_ACTIONS_URL", "http://127.0.0.1:19001/execute_actions"
)
DEFAULT_OUTPUT_DIR = Path(os.environ.get("AUTO_SCENE_OUTPUT_DIR", "/tmp/embodiedgpt_auto_scene"))

NO_ARG_ACTIONS = {
    "MoveAhead",
    "MoveBack",
    "MoveLeft",
    "MoveRight",
    "RotateLeft",
    "RotateRight",
    "LookUp",
    "LookDown",
    "DropHandObject",
    "Pass",
    "Done",
}
OBJECT_ACTIONS = {
    "PickupObject",
    "OpenObject",
    "CloseObject",
    "PushObject",
    "PullObject",
    "SliceObject",
    "BreakObject",
    "CookObject",
    "CleanObject",
    "FillObjectWithLiquid",
    "SetObjectStates",
}
FORCE_ACTIONS = OBJECT_ACTIONS | {"PutObject"}
PICKUPABLE_ACTIONS = {
    "PickupObject",
    "PushObject",
    "PullObject",
    "SliceObject",
    "BreakObject",
    "CookObject",
    "CleanObject",
    "FillObjectWithLiquid",
    "SetObjectStates",
}
OPENABLE_ACTIONS = {"OpenObject", "CloseObject"}
ACTION_INTENT_PATTERNS = (
    (("turn right", "rotate right"), "RotateRight"),
    (("turn left", "rotate left"), "RotateLeft"),
    (("move right", "strafe right", "step right"), "MoveRight"),
    (("move left", "strafe left", "step left"), "MoveLeft"),
    (("move forward", "go forward", "move ahead"), "MoveAhead"),
    (("move back", "go back", "back up"), "MoveBack"),
    (("look up",), "LookUp"),
    (("look down",), "LookDown"),
    (("pick up", "pickup", "grab", "take"), "PickupObject"),
    (("open",), "OpenObject"),
    (("close", "shut"), "CloseObject"),
    (("put", "place"), "PutObject"),
    (("push",), "PushObject"),
    (("pull",), "PullObject"),
    (("slice", "cut"), "SliceObject"),
    (("break",), "BreakObject"),
    (("clean", "wash"), "CleanObject"),
    (("cook", "heat"), "CookObject"),
    (("fill",), "FillObjectWithLiquid"),
)
TYPE_ALIASES = {
    "fridgedoor": "fridge",
    "cabinetdoor": "cabinet",
    "microwavedoor": "microwave",
    "drawardoor": "drawer",
    "drawerdoor": "drawer",
    "counter": "countertop",
    "counter top": "countertop",
    "sinkbasin": "sink",
}
COMMON_OBJECT_TYPES = {
    "Apple",
    "Banana",
    "Bowl",
    "Book",
    "Bread",
    "Cabinet",
    "CellPhone",
    "CoffeeMachine",
    "CounterTop",
    "CreditCard",
    "Cup",
    "Drawer",
    "Egg",
    "Fork",
    "Floor",
    "Fridge",
    "GarbageCan",
    "Knife",
    "Laptop",
    "Microwave",
    "Mug",
    "Pan",
    "Plate",
    "Pot",
    "Potato",
    "RemoteControl",
    "Sink",
    "SinkBasin",
    "Spoon",
    "TableTop",
    "Tomato",
}

TASK_INTENT_TOOL_NAME = "extract_task_intent"
TASK_INTENT_SOURCE = "qwen_native_tool_call"
TASK_INTENT_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": TASK_INTENT_TOOL_NAME,
        "description": "Extract requested robot action and primary object from the user task text.",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
            },
            "required": ["task"],
        },
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe an AI2-THOR execute_actions endpoint, render the current scene, "
            "ask Qwen for a semantic plan, ground object types to objectIds, and send actions."
        )
    )
    parser.add_argument("--execute-actions-url", default=DEFAULT_EXECUTE_ACTIONS_URL)
    parser.add_argument("--task", default="Infer a useful embodied task from the visible scene.")
    parser.add_argument("--task-intent-json", help="Structured task intent JSON supplied by an upstream normalizer.")
    parser.add_argument("--task-id", help="Task id for the final payload. Defaults to a generated id.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--send-timeout", type=positive_float, default=DEFAULT_HTTP_TIMEOUT)
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN35_MODEL)
    parser.add_argument("--qwen-device-map", default="auto")
    parser.add_argument("--qwen-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-new-tokens", type=positive_int, default=512)
    parser.add_argument("--temperature", type=positive_float, default=0.2)
    parser.add_argument("--max-actions", type=int, default=20, help="Maximum grounded actions before Done; use 0 for no limit.")
    parser.add_argument(
        "--allow-invisible-object-ids",
        action="store_true",
        help="Allow grounding to objects that are in metadata but not currently visible.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Probe and plan, but do not send the grounded actions.")
    parser.add_argument("--print-raw-output", action="store_true", help="Print raw Qwen output to stderr.")
    parser.add_argument(
        "--include-execute-response",
        action="store_true",
        help="Include the full execute_actions response in stdout. Defaults to a compact summary only.",
    )
    parser.add_argument(
        "--save-execute-response",
        action="store_true",
        help="Save the full execute_actions response to the output directory.",
    )
    parser.add_argument(
        "--save-raw-output",
        action="store_true",
        help="Save the raw Qwen output to the output directory before parsing it.",
    )
    parser.add_argument(
        "--primary-agent-id",
        help="Legacy agent id to treat as the local executor when a probe response contains multiple agents.",
    )
    parser.add_argument(
        "--primary-robot-id",
        type=int,
        default=0,
        help="Robot id to treat as the local executor for /state and /observe.",
    )
    parser.add_argument(
        "--known-robot-ids",
        help="Comma-separated robot ids available for relay queries, for example 0,1,2.",
    )
    parser.add_argument(
        "--state-endpoint",
        default="/state",
        help="Remote state endpoint path, relative to the execute_actions base URL.",
    )
    parser.add_argument(
        "--observe-endpoint",
        default="/observe",
        help="Remote observe endpoint path, relative to the execute_actions base URL.",
    )
    parser.add_argument(
        "--goto-endpoint",
        default="/goto",
        help="Remote goto endpoint path, relative to the execute_actions base URL.",
    )
    parser.add_argument(
        "--goto-max-actions",
        type=int,
        help="Optional max_actions value forwarded to /goto for navigation-only tasks.",
    )
    parser.add_argument(
        "--goto-min-distance",
        type=positive_float,
        help="Optional min_distance value forwarded to /goto for navigation-only tasks.",
    )
    parser.add_argument(
        "--goto-max-distance",
        type=positive_float,
        help="Optional max_distance value forwarded to /goto for navigation-only tasks.",
    )
    parser.add_argument(
        "--include-object-visibility-map",
        action="store_true",
        help="Include the full object visibility map in stdout. Defaults to a compact summary only.",
    )
    parser.add_argument(
        "--save-object-visibility-map",
        action="store_true",
        help="Save the full object visibility map to the output directory.",
    )
    parser.add_argument(
        "--relay-mode",
        action="store_true",
        help="Coordinate execution through the best visible agent instead of only reporting peer visibility.",
    )
    parser.add_argument(
        "--relay-strategy",
        choices=["agent", "rules"],
        default="agent",
        help="Choose the executor with the tool-calling relay agent (default) or the legacy deterministic rules.",
    )
    parser.add_argument(
        "--relay-agent-max-turns",
        type=positive_int,
        default=8,
        help="Maximum tool-calling turns for one relay-agent decision (default: 8).",
    )
    parser.add_argument(
        "--executor-agent-id-field",
        default="robot_id",
        help="Top-level payload field used to tell execute_actions which robot should execute actions.",
    )
    parser.add_argument(
        "--closed-loop-replan",
        action="store_true",
        help="Execute multi-step intent one step at a time, observing and replanning after each step.",
    )
    parser.add_argument(
        "--max-replan-steps",
        type=positive_int,
        default=10,
        help="Maximum intent steps to execute in --closed-loop-replan mode.",
    )
    return parser.parse_args(argv)


HTTP_ERROR_BODY_PREVIEW_CHARS = 2000


def _json_object_from_response_body(response_body: str) -> dict[str, Any]:
    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"HTTP response was not JSON: {response_body[:HTTP_ERROR_BODY_PREVIEW_CHARS]}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("HTTP response JSON must be an object")
    return parsed


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = _json_object_from_response_body(error_body)
        except RuntimeError as parse_error:
            raise RuntimeError(
                f"HTTP POST failed with status {exc.code}: {error_body[:HTTP_ERROR_BODY_PREVIEW_CHARS]}"
            ) from parse_error
        parsed["_http_status"] = exc.code
        return parsed
    except URLError as exc:
        raise RuntimeError(f"HTTP POST failed: {exc.reason}") from exc

    return _json_object_from_response_body(response_body)


def get_json(url: str, timeout: float) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP GET failed with status {exc.code}: {error_body[:200]}") from exc
    except URLError as exc:
        raise RuntimeError(f"HTTP GET failed: {exc.reason}") from exc

    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"HTTP response was not JSON: {response_body[:200]}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("HTTP response JSON must be an object")
    return parsed


def base_url_from_execute_actions_url(url: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def endpoint_url(base_url: str, endpoint: str, query: dict[str, Any] | None = None) -> str:
    base = base_url.rstrip("/")
    path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    encoded = urlencode(query or {})
    return f"{base}{path}" + (f"?{encoded}" if encoded else "")


def goto_url_from_execute_actions_url(execute_actions_url: str, goto_endpoint: str = "/goto") -> str:
    return endpoint_url(base_url_from_execute_actions_url(execute_actions_url), goto_endpoint)


NAVIGATION_ONLY_PATTERN = re.compile(
    r"^\s*(?:please\s+)?(?:go|navigate|move|walk|head)(?:\s+(?:over|up))?\s+to\s+(?:the\s+|a\s+|an\s+)?(.+?)\s*[.!?]?\s*$",
    re.IGNORECASE,
)
NAVIGATION_TARGET_TRAILING_PATTERN = re.compile(
    r"\s+(?:please|now|nearby|near|area|location|place)$",
    re.IGNORECASE,
)


def _clean_navigation_target_text(value: str) -> str:
    target = value.strip().strip('"\'`.,!?;:')
    target = re.sub(r"\b(?:please|now)\b", "", target, flags=re.IGNORECASE)
    target = re.sub(r"\s+", " ", target).strip()
    while True:
        cleaned = NAVIGATION_TARGET_TRAILING_PATTERN.sub("", target).strip()
        if cleaned == target:
            return target
        target = cleaned


def navigation_object_type_for_task(task: str, available_object_types: list[str] | set[str]) -> str | None:
    if not isinstance(task, str):
        return None
    match = NAVIGATION_ONLY_PATTERN.match(task)
    if not match:
        return None
    target_text = _clean_navigation_target_text(match.group(1))
    if not target_text:
        return None
    lookup = _object_type_lookup(set(available_object_types) | COMMON_OBJECT_TYPES)
    normalized = normalize_type(target_text)
    if normalized in lookup:
        return lookup[normalized]
    words = target_text.split()
    for start in range(1, len(words)):
        suffix = normalize_type(" ".join(words[start:]))
        if suffix in lookup:
            return lookup[suffix]
    containing_matches = [
        (len(key), object_type)
        for key, object_type in lookup.items()
        if key and key in normalized
    ]
    if containing_matches:
        return max(containing_matches, key=lambda item: item[0])[1]
    return None


def goto_payload_for_navigation_task(args: argparse.Namespace, task_id: str, object_type: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task_id": task_id,
        "robot_id": args.primary_robot_id,
        "object_type": object_type,
        "execute": not bool(args.dry_run),
    }
    if args.goto_max_actions is not None:
        payload["max_actions"] = args.goto_max_actions
    if args.goto_min_distance is not None:
        payload["min_distance"] = args.goto_min_distance
    if args.goto_max_distance is not None:
        payload["max_distance"] = args.goto_max_distance
    return payload


def _failed_action_summary(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": item.get("index"),
        "action": item.get("action"),
        "error": item.get("error"),
    }


def failed_action_from_replan_trace(replan_trace: Any) -> dict[str, Any] | None:
    if not isinstance(replan_trace, list):
        return None
    for trace_item in reversed(replan_trace):
        if not isinstance(trace_item, dict):
            continue
        segments = trace_item.get("segments")
        if not isinstance(segments, list):
            continue
        for segment in reversed(segments):
            if not isinstance(segment, dict):
                continue
            failed_action = segment.get("failed_action")
            if isinstance(failed_action, dict):
                return _failed_action_summary(failed_action)
    return None


def summarize_dynamic_obstacles(dynamic_obstacles: Any) -> list[dict[str, Any]]:
    if not isinstance(dynamic_obstacles, list):
        return []
    summary = []
    for obstacle in dynamic_obstacles[:5]:
        if not isinstance(obstacle, dict):
            continue
        item = {
            "robot_id": obstacle.get("robot_id"),
            "name": obstacle.get("name"),
            "blocked_node_count": obstacle.get("blocked_node_count"),
        }
        if isinstance(obstacle.get("nearest_node"), dict):
            item["nearest_node"] = obstacle["nearest_node"]
        summary.append(item)
    return summary


def summarize_goto_result(response: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in (
        "status",
        "error_code",
        "error",
        "replan_error_code",
        "replan_error",
        "replan_count",
        "_http_status",
        "planner",
        "robot_id",
        "estimated_distance",
        "execute",
        "blocked_node_count",
        "avoid_other_robots",
        "executed_action_count",
    ):
        if key in response:
            summary[key] = response[key]
    actions = response.get("actions")
    if isinstance(actions, list):
        summary["action_count"] = len(actions)
    path = response.get("path")
    if isinstance(path, list):
        summary["path_length"] = len(path)
    for key in ("target", "target_position", "goal_position", "start_position"):
        if isinstance(response.get(key), dict):
            summary[key] = response[key]

    failed_action = response.get("failed_action")
    if isinstance(failed_action, dict):
        summary["failed_action"] = _failed_action_summary(failed_action)

    execute_result = response.get("execute_result")
    if isinstance(execute_result, dict):
        summary["execute_result_status"] = execute_result.get("status")
        results = execute_result.get("results")
        if isinstance(results, list):
            summary["execute_result_count"] = len(results)
            if "failed_action" not in summary:
                for item in results:
                    if isinstance(item, dict) and item.get("success") is False:
                        summary["failed_action"] = _failed_action_summary(item)
                        break

    if "failed_action" not in summary:
        failed_from_trace = failed_action_from_replan_trace(response.get("replan_trace"))
        if failed_from_trace is not None:
            summary["failed_action"] = failed_from_trace
    if isinstance(response.get("replan_trace"), list):
        summary["replan_trace_count"] = len(response["replan_trace"])

    dynamic_obstacles = summarize_dynamic_obstacles(response.get("dynamic_obstacles"))
    if dynamic_obstacles:
        summary["dynamic_obstacles"] = dynamic_obstacles
        summary["dynamic_obstacle_count"] = len(response.get("dynamic_obstacles", []))
    return summary


def goto_failure_reason(response: dict[str, Any], summary: dict[str, Any]) -> str:
    status = response.get("status")
    error_code = response.get("error_code") or status or "goto_failed"
    error = response.get("error") or response.get("reason") or f"/goto returned status {status!r}"
    pieces = [f"{error_code}: {error}"]
    replan_error_code = response.get("replan_error_code")
    replan_error = response.get("replan_error")
    if replan_error_code or replan_error:
        pieces.append(f"replan={replan_error_code or 'unknown'}: {replan_error or 'unknown error'}")
    failed_action = summary.get("failed_action")
    if isinstance(failed_action, dict):
        action = failed_action.get("action")
        action_error = failed_action.get("error")
        if action or action_error:
            pieces.append(f"failed_action={action or 'unknown'}; error={action_error or 'unknown error'}")
    if summary.get("replan_count") is not None:
        pieces.append(f"replans={summary.get('replan_count')}")
    if summary.get("dynamic_obstacle_count") is not None:
        pieces.append(f"dynamic_obstacles={summary.get('dynamic_obstacle_count')}")
    return "; ".join(pieces)


def navigation_goto_result(
    args: argparse.Namespace,
    *,
    task_id: str,
    base_url: str,
    probe: dict[str, Any],
    primary_observation: dict[str, Any],
    image_path: Path,
    object_type: str,
) -> dict[str, Any]:
    payload = goto_payload_for_navigation_task(args, task_id, object_type)
    goto_url = endpoint_url(base_url, args.goto_endpoint)
    print(
        f"[goto] robot_{args.primary_robot_id} navigating to {object_type} via {goto_url}...",
        file=sys.stderr,
    )
    goto_response = post_json(goto_url, payload, args.send_timeout)
    summary = summarize_goto_result(goto_response)
    status = goto_response.get("status")
    if status == "success":
        closed_loop_result = {"status": "success", "strategy": "goto"}
        print(
            f"[goto] success: {summary.get('action_count', 0)} actions, goal_position={summary.get('goal_position')}",
            file=sys.stderr,
        )
    else:
        reason = goto_failure_reason(goto_response, summary)
        closed_loop_result = {
            "status": "needs_upstream_planning",
            "strategy": "goto",
            "failure_code": goto_response.get("error_code") or status or "goto_failed",
            "reason": reason,
        }
        print(f"[goto] failed: {reason}", file=sys.stderr)
    result = {
        "task_id": task_id,
        "image_path": str(image_path),
        "sceneName": primary_observation.get("sceneName") or probe.get("sceneName"),
        "primary_agent_id": primary_observation.get("agent_id"),
        "primary_robot_id": primary_observation.get("robot_id", args.primary_robot_id),
        "visible_object_types": object_categories(primary_observation.get("objects", []), visible_only=True),
        "task_intent": {
            "requestedAction": "GotoObject",
            "requestedObjectType": object_type,
            "intentSteps": [
                {"order": 1, "action": "GotoObject", "objectType": object_type, "targetType": None}
            ],
        },
        "task_intent_source": "navigation_goto_intent",
        "goto_url": goto_url,
        "goto_payload": payload,
        "goto_result_summary": summary,
        "closed_loop_result": closed_loop_result,
    }
    if args.include_execute_response:
        result["goto_result"] = goto_response
    return result


def get_state(
    base_url: str,
    endpoint: str,
    *,
    robot_id: int,
    render_image: bool,
    timeout: float,
) -> dict[str, Any]:
    return get_json(
        endpoint_url(
            base_url,
            endpoint,
            {"robot_id": robot_id, "render_image": int(render_image)},
        ),
        timeout,
    )


def get_global_state(base_url: str, endpoint: str, *, timeout: float) -> dict[str, Any]:
    return get_json(endpoint_url(base_url, endpoint), timeout)


def observe_robot(
    base_url: str,
    endpoint: str,
    *,
    robot_id: int,
    render_image: bool,
    timeout: float,
) -> dict[str, Any]:
    return post_json(
        endpoint_url(base_url, endpoint),
        {"robot_id": robot_id, "render_image": bool(render_image)},
        timeout,
    )


def execute_actions_probe_scene(
    url: str,
    task_id: str,
    timeout: float,
    *,
    robot_id: int = 0,
) -> dict[str, Any]:
    payload = {
        "task_id": f"{task_id}_probe_robot_{robot_id}",
        "task": "return current scene image and object ids",
        "plan": [{"action": "Pass", "objectType": None, "targetType": None}],
        "robot_id": robot_id,
        "render_image": True,
        "actions": [{"action": "Pass"}],
    }
    response = post_json(url, payload, timeout)
    if response.get("status") not in (None, "success"):
        raise RuntimeError(f"probe failed with status: {response.get('status')}")
    if not extract_agent_observations(response, primary_robot_id=robot_id):
        raise RuntimeError("probe response did not include state.objects or agent observations")
    return response


def legacy_probe_scene(url: str, task_id: str, timeout: float) -> dict[str, Any]:
    return execute_actions_probe_scene(url, task_id, timeout, robot_id=0)


def probe_scene(
    url: str,
    task_id: str,
    timeout: float,
    *,
    primary_robot_id: int = 0,
    state_endpoint: str = "/state",
) -> dict[str, Any]:
    return execute_actions_probe_scene(url, task_id, timeout, robot_id=primary_robot_id)


def image_base64_from_probe(probe: dict[str, Any]) -> str:
    results = probe.get("results")
    if isinstance(results, list):
        for result in results:
            if isinstance(result, dict) and isinstance(result.get("image_base64"), str):
                return result["image_base64"]
    image_base64 = probe.get("image_base64")
    if isinstance(image_base64, str):
        return image_base64
    raise RuntimeError("probe response did not include image_base64; send render_image=true")


def write_base64_image(image_base64: str, image_path: Path) -> None:
    if "," in image_base64 and image_base64.lstrip().startswith("data:"):
        image_base64 = image_base64.split(",", 1)[1]
    image_bytes = base64.b64decode(image_base64)
    image_path.write_bytes(image_bytes)


def save_probe_image(probe: dict[str, Any], output_dir: Path, task_id: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_base64 = image_base64_from_probe(probe)
    image_path = output_dir / f"{task_id}_scene.jpg"
    write_base64_image(image_base64, image_path)
    return image_path


def normalize_type(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    normalized = "".join(ch for ch in value.strip().lower() if ch.isalnum())
    return TYPE_ALIASES.get(normalized, normalized)


def object_id_of(item: dict[str, Any]) -> str | None:
    value = item.get("id") or item.get("objectId")
    return value if isinstance(value, str) and value else None


def object_type_of(item: dict[str, Any]) -> str:
    value = item.get("type") or item.get("objectType")
    return value if isinstance(value, str) else ""


def _state_metadata_of(item: dict[str, Any]) -> dict[str, Any]:
    state = item.get("state")
    if isinstance(state, dict):
        return state
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    return item


def _objects_from_observation_source(item: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = _state_metadata_of(item)
    objects = metadata.get("objects")
    if not isinstance(objects, list):
        objects = item.get("objects")
    if not isinstance(objects, list):
        return []
    return [obj for obj in objects if isinstance(obj, dict) and object_id_of(obj)]


def _scene_name_from_observation_source(item: dict[str, Any]) -> Any:
    metadata = _state_metadata_of(item)
    return metadata.get("sceneName") or item.get("sceneName")


def _image_base64_from_observation_source(item: dict[str, Any]) -> str | None:
    image_base64 = item.get("image_base64") or item.get("imageBase64")
    if isinstance(image_base64, str):
        return image_base64
    results = item.get("results")
    if isinstance(results, list):
        for result in results:
            if isinstance(result, dict):
                image_base64 = result.get("image_base64") or result.get("imageBase64")
                if isinstance(image_base64, str):
                    return image_base64
    return None


def _robot_id_from_value(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
        match = re.fullmatch(r"(?:agent|robot)[_-]?(\d+)", stripped, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _robot_id_from_observation_source(item: dict[str, Any], default: int | None) -> int | None:
    for key in ("robot_id", "robotId", "selected_robot_id", "agent_id", "agentId", "id"):
        robot_id = _robot_id_from_value(item.get(key))
        if robot_id is not None:
            return robot_id
    return default


def _robot_name_from_observation_source(item: dict[str, Any], robot_id: int | None) -> str | None:
    robot = item.get("robot")
    if isinstance(robot, dict):
        name = robot.get("name") or robot.get("robot_name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    for key in ("robot_name", "name"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if robot_id is not None:
        return f"Robot{robot_id}"
    return None


def _robot_state_from_observation_source(item: dict[str, Any], robot_id: int | None) -> dict[str, Any] | None:
    robot = item.get("robot")
    if isinstance(robot, dict):
        return robot
    for container in (item, _state_metadata_of(item)):
        robots = container.get("robots") if isinstance(container, dict) else None
        if isinstance(robots, list):
            for candidate in robots:
                if not isinstance(candidate, dict):
                    continue
                candidate_id = _robot_id_from_observation_source(candidate, None)
                if robot_id is not None and candidate_id == robot_id:
                    return candidate
    return None


def _append_inventory_object(inventory: list[dict[str, Any]], value: Any) -> None:
    if not isinstance(value, dict) or not (object_id_of(value) or object_type_of(value)):
        return
    value_id = object_id_of(value)
    value_type = object_type_of(value)
    for item in inventory:
        if value_id is not None and object_id_of(item) == value_id:
            return
        if value_id is None and value_type is not None and object_type_of(item) == value_type:
            return
    inventory.append(value)


def _held_object_candidate(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict) and (object_id_of(value) or object_type_of(value)):
        return value
    return None


def _held_object_from_observation_source(
    item: dict[str, Any],
    robot_state: dict[str, Any] | None,
    robot_id: int | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    if robot_state is not None:
        for key in ("held_object", "heldObject"):
            candidate = _held_object_candidate(robot_state.get(key))
            if candidate is not None:
                return candidate, f"robot_state.{key}"

    for key in ("held_object", "heldObject"):
        candidate = _held_object_candidate(item.get(key))
        if candidate is not None:
            return candidate, key

    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        for key in ("held_object", "heldObject"):
            candidate = _held_object_candidate(metadata.get(key))
            if candidate is not None:
                return candidate, f"metadata.{key}"

    results = item.get("results")
    if isinstance(results, list):
        for result in reversed(results):
            if not isinstance(result, dict):
                continue
            result_robot_id = _robot_id_from_value(result.get("robot_id"))
            if robot_id is not None and result_robot_id != robot_id:
                continue
            for key in ("held_object", "heldObject"):
                candidate = _held_object_candidate(result.get(key))
                if candidate is not None:
                    return candidate, f"result.{key}"
            result_robot = result.get("robot")
            if isinstance(result_robot, dict):
                for key in ("held_object", "heldObject"):
                    candidate = _held_object_candidate(result_robot.get(key))
                    if candidate is not None:
                        return candidate, f"result.robot.{key}"

    return None, None


def _inventory_from_observation_source(
    item: dict[str, Any],
    robot_state: dict[str, Any] | None,
    robot_id: int | None = None,
) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []

    for values in (item.get("inventory"), item.get("inventoryObjects")):
        if isinstance(values, list):
            for value in values:
                _append_inventory_object(inventory, value)

    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        values = metadata.get("inventoryObjects") or metadata.get("inventory")
        if isinstance(values, list):
            for value in values:
                _append_inventory_object(inventory, value)

    results = item.get("results")
    if isinstance(results, list):
        for result in reversed(results):
            if not isinstance(result, dict):
                continue
            result_robot_id = _robot_id_from_value(result.get("robot_id"))
            if robot_id is not None and result_robot_id != robot_id:
                continue
            values = result.get("inventory") or result.get("inventoryObjects")
            if isinstance(values, list):
                for value in values:
                    _append_inventory_object(inventory, value)
            result_robot = result.get("robot")
            if isinstance(result_robot, dict):
                values = result_robot.get("inventory") or result_robot.get("inventoryObjects")
                if isinstance(values, list):
                    for value in values:
                        _append_inventory_object(inventory, value)
            if inventory:
                break

    if robot_state is not None:
        values = robot_state.get("inventory") or robot_state.get("inventoryObjects")
        if isinstance(values, list):
            for value in values:
                _append_inventory_object(inventory, value)

    return inventory


def _legacy_agent_id_from_observation_source(item: dict[str, Any], default: str) -> str:
    for key in ("agent_id", "agentId", "id", "name"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _observation_from_source(
    item: dict[str, Any],
    *,
    default_agent_id: str,
    primary_agent_id: str | None,
    default_robot_id: int | None = None,
    primary_robot_id: int | None = None,
) -> dict[str, Any] | None:
    objects = _objects_from_observation_source(item)
    if not objects:
        return None
    robot_id = _robot_id_from_observation_source(item, default_robot_id)
    agent_id = f"robot_{robot_id}" if robot_id is not None else _legacy_agent_id_from_observation_source(item, default_agent_id)
    is_primary = False
    if primary_robot_id is not None and robot_id is not None:
        is_primary = robot_id == primary_robot_id
    elif primary_agent_id is not None:
        is_primary = agent_id == primary_agent_id
    robot_state = _robot_state_from_observation_source(item, robot_id)
    held_object, held_object_source = _held_object_from_observation_source(item, robot_state, robot_id)
    return {
        "agent_id": agent_id,
        "robot_id": robot_id,
        "robot_name": _robot_name_from_observation_source(item, robot_id),
        "robot_state": robot_state,
        "held_object": held_object,
        "held_object_source": held_object_source,
        "inventory": _inventory_from_observation_source(item, robot_state, robot_id),
        "is_primary": is_primary,
        "sceneName": _scene_name_from_observation_source(item),
        "objects": objects,
        "visible_object_types": object_categories(objects, visible_only=True),
        "image_base64": _image_base64_from_observation_source(item),
        "image_path": None,
    }


def extract_agent_observations(
    probe: dict[str, Any],
    primary_agent_id: str | None = None,
    primary_robot_id: int | None = None,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for key in ("agent_observations", "agents", "events"):
        values = probe.get(key)
        if not isinstance(values, list):
            continue
        for index, value in enumerate(values):
            if not isinstance(value, dict):
                continue
            observation = _observation_from_source(
                value,
                default_agent_id=f"robot_{index}",
                primary_agent_id=primary_agent_id,
                default_robot_id=index,
                primary_robot_id=primary_robot_id,
            )
            if observation is not None:
                observations.append(observation)

    if not observations:
        selected_robot_id = _robot_id_from_value(probe.get("selected_robot_id"))
        default_robot_id = primary_robot_id if primary_robot_id is not None else selected_robot_id
        if default_robot_id is None and primary_agent_id is None:
            default_robot_id = 0
        observation = _observation_from_source(
            probe,
            default_agent_id=f"robot_{default_robot_id}" if default_robot_id is not None else "agent_0",
            primary_agent_id=primary_agent_id,
            default_robot_id=default_robot_id,
            primary_robot_id=primary_robot_id if primary_robot_id is not None else default_robot_id,
        )
        if observation is not None:
            observations.append(observation)

    if not observations:
        return observations

    if primary_robot_id is not None:
        if primary_robot_id not in {item.get("robot_id") for item in observations}:
            available = ", ".join(str(item.get("robot_id")) for item in observations)
            raise ValueError(f"primary robot {primary_robot_id!r} was not found; available robots: {available}")
        for item in observations:
            item["is_primary"] = item.get("robot_id") == primary_robot_id
    elif primary_agent_id is not None:
        if primary_agent_id not in {item["agent_id"] for item in observations}:
            available = ", ".join(item["agent_id"] for item in observations)
            raise ValueError(f"primary agent {primary_agent_id!r} was not found; available agents: {available}")
        for item in observations:
            item["is_primary"] = item["agent_id"] == primary_agent_id
    elif not any(item["is_primary"] for item in observations):
        observations[0]["is_primary"] = True

    return observations


def primary_agent_observation(agent_observations: list[dict[str, Any]]) -> dict[str, Any]:
    for observation in agent_observations:
        if observation.get("is_primary"):
            return observation
    if agent_observations:
        return agent_observations[0]
    raise ValueError("no agent observations available")


def agent_observation_by_id(
    agent_observations: list[dict[str, Any]],
    agent_id: str,
) -> dict[str, Any]:
    for observation in agent_observations:
        if observation.get("agent_id") == agent_id:
            return observation
    available = ", ".join(str(item.get("agent_id")) for item in agent_observations)
    raise ValueError(f"agent {agent_id!r} was not found; available agents: {available}")


def _safe_filename_fragment(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe or "agent"


def save_agent_observation_images(
    agent_observations: list[dict[str, Any]],
    output_dir: Path,
    task_id: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for observation in agent_observations:
        image_base64 = observation.get("image_base64")
        if not isinstance(image_base64, str):
            continue
        if observation.get("is_primary"):
            image_path = output_dir / f"{task_id}_scene.jpg"
        else:
            agent_id = _safe_filename_fragment(str(observation.get("agent_id", "agent")))
            image_path = output_dir / f"{task_id}_{agent_id}_scene.jpg"
        write_base64_image(image_base64, image_path)
        observation["image_path"] = str(image_path)


def scene_objects(probe: dict[str, Any]) -> list[dict[str, Any]]:
    objects = probe["state"]["objects"]
    return [item for item in objects if isinstance(item, dict) and object_id_of(item)]


def object_categories(objects: list[dict[str, Any]], visible_only: bool) -> list[str]:
    categories = {
        object_type_of(item)
        for item in objects
        if object_type_of(item) and (not visible_only or bool(item.get("visible")))
    }
    return sorted(categories)


def _object_type_lookup(types: list[str] | set[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for object_type in types:
        normalized = normalize_type(object_type)
        if normalized and normalized not in lookup:
            lookup[normalized] = object_type
    return lookup


OBJECT_VISIBILITY_AFFORDANCE_FIELDS = (
    "pickupable",
    "openable",
    "receptacle",
    "sliceable",
    "breakable",
    "cookable",
    "dirtyable",
    "canFillWithLiquid",
)


def agent_observations_summary(agent_observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "agent_id": observation.get("agent_id"),
            "robot_id": observation.get("robot_id"),
            "robot_name": observation.get("robot_name"),
            "is_primary": bool(observation.get("is_primary")),
            "sceneName": observation.get("sceneName"),
            "visible_object_types": observation.get("visible_object_types", []),
            "held_object_type": held_object_type_from_observation(observation),
            "image_path": observation.get("image_path"),
        }
        for observation in agent_observations
    ]


def build_object_visibility_map(agent_observations: list[dict[str, Any]]) -> dict[str, Any]:
    primary = primary_agent_observation(agent_observations)
    primary_agent_id = str(primary.get("agent_id"))
    visibility_map: dict[str, Any] = {
        "primary_agent_id": primary_agent_id,
        "agents": agent_observations_summary(agent_observations),
        "objects_by_type": {},
    }
    objects_by_type: dict[str, dict[str, Any]] = visibility_map["objects_by_type"]

    for observation in agent_observations:
        agent_id = str(observation.get("agent_id"))
        objects = observation.get("objects", [])
        if not isinstance(objects, list):
            continue
        for item in objects:
            if not isinstance(item, dict):
                continue
            object_type = object_type_of(item)
            normalized = normalize_type(object_type)
            if not object_type or not normalized:
                continue
            entry = objects_by_type.setdefault(
                normalized,
                {
                    "object_type": object_type,
                    "normalized_type": normalized,
                    "visible_by_agent_ids": [],
                    "count_by_agent": {},
                    "best_agent_id": None,
                    "affordances": {},
                },
            )
            if bool(item.get("visible")):
                entry["count_by_agent"][agent_id] = entry["count_by_agent"].get(agent_id, 0) + 1
                if agent_id not in entry["visible_by_agent_ids"]:
                    entry["visible_by_agent_ids"].append(agent_id)
            affordances = entry["affordances"]
            for field in OBJECT_VISIBILITY_AFFORDANCE_FIELDS:
                if field in item:
                    affordances[field] = bool(affordances.get(field)) or bool(item.get(field))

    for entry in objects_by_type.values():
        visible_by_agent_ids = entry["visible_by_agent_ids"]
        if primary_agent_id in visible_by_agent_ids:
            entry["best_agent_id"] = primary_agent_id
        elif visible_by_agent_ids:
            entry["best_agent_id"] = visible_by_agent_ids[0]

    return visibility_map


def object_visibility_summary(object_visibility_map: dict[str, Any]) -> dict[str, Any]:
    objects_by_type = object_visibility_map.get("objects_by_type", {})
    visible_object_types: list[dict[str, Any]] = []
    total_type_count = 0
    visible_type_count = 0
    if isinstance(objects_by_type, dict):
        total_type_count = len(objects_by_type)
        for entry in objects_by_type.values():
            if not isinstance(entry, dict):
                continue
            visible_by_agent_ids = entry.get("visible_by_agent_ids", [])
            if not visible_by_agent_ids:
                continue
            visible_type_count += 1
            visible_object_types.append(
                {
                    "object_type": entry.get("object_type"),
                    "visible_by_agent_ids": visible_by_agent_ids,
                    "best_agent_id": entry.get("best_agent_id"),
                }
            )
    return {
        "primary_agent_id": object_visibility_map.get("primary_agent_id"),
        "agent_count": len(object_visibility_map.get("agents", [])),
        "total_object_type_count": total_type_count,
        "visible_object_type_count": visible_type_count,
        "hidden_object_type_count": max(total_type_count - visible_type_count, 0),
        "visible_object_types": sorted(visible_object_types, key=lambda item: str(item.get("object_type"))),
    }


def relay_agent_observation_summary(observation: dict[str, Any]) -> dict[str, Any]:
    visible_objects: dict[str, dict[str, Any]] = {}
    objects = observation.get("objects", [])
    if isinstance(objects, list):
        for item in objects:
            if not isinstance(item, dict) or not bool(item.get("visible")):
                continue
            object_type = object_type_of(item)
            if not object_type:
                continue
            entry = visible_objects.setdefault(
                object_type,
                {"object_type": object_type, "affordances": {}, "sample_positions": []},
            )
            affordances = entry["affordances"]
            for field in OBJECT_VISIBILITY_AFFORDANCE_FIELDS:
                if field in item:
                    affordances[field] = bool(affordances.get(field)) or bool(item.get(field))
            position = _position_from_value(item)
            if position is not None and position not in entry["sample_positions"] and len(entry["sample_positions"]) < 3:
                entry["sample_positions"].append(position)
    return {
        "agent_id": observation.get("agent_id"),
        "robot_id": observation.get("robot_id"),
        "robot_name": observation.get("robot_name"),
        "is_primary": bool(observation.get("is_primary")),
        "sceneName": observation.get("sceneName"),
        "image_path": observation.get("image_path"),
        "robot_position": _robot_position_from_observation(observation),
        "held_object_type": held_object_type_from_observation(observation),
        "inventory_object_types": sorted(
            {object_type_of(item) for item in observation.get("inventory", []) if isinstance(item, dict) and object_type_of(item)}
        ),
        "visible_objects": sorted(visible_objects.values(), key=lambda item: item["object_type"]),
    }


def requested_object_type_for_plan(
    task: str,
    semantic_plan: dict[str, Any],
    available_types: list[str],
    task_intent: dict[str, Any] | None = None,
) -> str | None:
    if isinstance(task_intent, dict):
        requested_object = task_intent.get("requestedObjectType")
        if isinstance(requested_object, str) and requested_object.strip():
            return requested_object.strip()
    target_object_type = semantic_plan.get("targetObjectType")
    if isinstance(target_object_type, str) and target_object_type.strip():
        return target_object_type.strip()
    return extract_requested_object_type(task, available_types)


def extract_requested_object_type(task: str, available_types: list[str]) -> str | None:
    candidates = _object_type_lookup(set(available_types) | COMMON_OBJECT_TYPES)
    words = re.findall(r"[a-z0-9]+", task.lower())
    for start in range(len(words)):
        for end in range(min(len(words), start + 3), start, -1):
            phrase = " ".join(words[start:end])
            normalized = normalize_type(phrase)
            if normalized in candidates:
                return candidates[normalized]
    return None


def split_task_clauses(task: str) -> list[str]:
    normalized = re.sub(r"\b(?:and then|then)\b", ",", task, flags=re.IGNORECASE)
    normalized = re.sub(r"\band\b", ",", normalized, flags=re.IGNORECASE)
    return [part.strip(" .") for part in re.split(r"[,;]", normalized) if part.strip(" .")]


def _contains_pronoun_reference(text: str) -> bool:
    return bool(re.search(r"\b(?:it|them|that object|the object)\b", text.lower()))


def _target_phrase_for_put(clause: str) -> str | None:
    match = re.search(r"\b(?:on|onto|in|into|inside|to)\b\s+(.+)$", clause, flags=re.IGNORECASE)
    if not match:
        return None
    phrase = re.sub(r"^(?:the|a|an)\s+", "", match.group(1).strip(), flags=re.IGNORECASE)
    return phrase.strip(" .") or None


def _object_phrase_before_put_target(clause: str) -> str:
    return re.split(r"\b(?:on|onto|in|into|inside|to)\b", clause, maxsplit=1, flags=re.IGNORECASE)[0]


def intent_step_for_clause(
    clause: str,
    available_types: list[str],
    previous_object_type: str | None,
    order: int,
) -> dict[str, Any] | None:
    action_name = extract_requested_action(clause)
    if action_name is None:
        return None

    object_type: str | None = None
    target_type: str | None = None
    if action_name in NO_ARG_ACTIONS:
        pass
    elif action_name == "PutObject":
        object_phrase = _object_phrase_before_put_target(clause)
        if _contains_pronoun_reference(object_phrase) and previous_object_type is not None:
            object_type = previous_object_type
        else:
            object_type = extract_requested_object_type(object_phrase, available_types)
        target_phrase = _target_phrase_for_put(clause)
        if target_phrase is not None:
            target_type = extract_requested_object_type(target_phrase, available_types)
    else:
        if _contains_pronoun_reference(clause) and previous_object_type is not None:
            object_type = previous_object_type
        else:
            object_type = extract_requested_object_type(clause, available_types)

    return {
        "order": order,
        "action": action_name,
        "objectType": object_type,
        "targetType": target_type,
    }


def extract_intent_steps(task: str, available_types: list[str]) -> list[dict[str, Any]]:
    candidates = sorted(set(available_types) | COMMON_OBJECT_TYPES)
    steps: list[dict[str, Any]] = []
    previous_object_type: str | None = None
    for clause in split_task_clauses(task):
        step = intent_step_for_clause(clause, candidates, previous_object_type, len(steps) + 1)
        if step is None:
            continue
        if isinstance(step.get("objectType"), str) and step["objectType"]:
            previous_object_type = step["objectType"]
        steps.append(step)
    if not steps:
        action_name = extract_requested_action(task)
        object_type = extract_requested_object_type(task, candidates)
        if action_name is not None or object_type is not None:
            steps.append({"order": 1, "action": action_name, "objectType": object_type, "targetType": None})
    return steps


def primary_intent_step(intent_steps: list[dict[str, Any]]) -> dict[str, Any] | None:
    for step in intent_steps:
        if isinstance(step, dict) and isinstance(step.get("action"), str):
            return step
    return None


def intent_steps(task_intent: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(task_intent, dict):
        return []
    steps = task_intent.get("intentSteps")
    if not isinstance(steps, list):
        return []
    return [step for step in steps if isinstance(step, dict)]


def has_multi_step_intent(task_intent: dict[str, Any] | None) -> bool:
    return len(intent_steps(task_intent)) > 1


def execute_extract_task_intent_tool(task: str, available_types: list[str]) -> dict[str, Any]:
    steps = extract_intent_steps(task, available_types)
    primary_step = primary_intent_step(steps)
    requested_action = primary_step.get("action") if primary_step else extract_requested_action(task)
    requested_object = primary_step.get("objectType") if primary_step else extract_requested_object_type(task, available_types)
    return {
        "requestedAction": requested_action,
        "requestedObjectType": requested_object,
        "intentSteps": steps,
    }


def task_intent_tool_messages(task: str) -> list[dict[str, Any]]:
    prompt = (
        "Call the extract_task_intent tool exactly once for this robot task. "
        "Do not answer directly. Do not inspect or infer from images. "
        "Use the original task text as the tool input.\n\n"
        f"Task: {task.strip()}"
    )
    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]


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
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError:
            parsed_arguments = {"raw": arguments}
        arguments = parsed_arguments
    if not isinstance(arguments, dict):
        arguments = {}
    return {"name": name.strip(), "arguments": arguments}


def parse_qwen_tool_call(output: str) -> dict[str, Any]:
    tool_blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", output, flags=re.DOTALL | re.IGNORECASE)
    for block in tool_blocks:
        for value in _json_values_from_text(block):
            tool_call = _tool_call_from_value(value)
            if tool_call is not None:
                return tool_call

    for value in _json_values_from_text(output):
        tool_call = _tool_call_from_value(value)
        if tool_call is not None:
            return tool_call

    raise ValueError("Qwen output did not contain a valid tool call")


def validate_task_intent_tool_call(tool_call: dict[str, Any], original_task: str) -> dict[str, Any]:
    if tool_call.get("name") != TASK_INTENT_TOOL_NAME:
        raise RuntimeError(f"Qwen called unexpected tool {tool_call.get('name')!r}; expected {TASK_INTENT_TOOL_NAME!r}")

    warnings: list[str] = []
    arguments = tool_call.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {}
        warnings.append("tool call arguments must be an object; using original CLI task")

    task_argument = arguments.get("task")
    if not isinstance(task_argument, str):
        warnings.append("tool call omitted required argument 'task'; using original CLI task")
    elif task_argument.strip() != original_task.strip():
        warnings.append("tool call task argument differs from original CLI task; using original CLI task")

    return {"status": "warning" if warnings else "ok", "warnings": warnings}


def generate_task_intent_tool_call(
    args: argparse.Namespace,
    available_types: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    qwen = qwen_backend_for_args(args)

    output = qwen.generate_with_tools(task_intent_tool_messages(args.task), [TASK_INTENT_TOOL_SCHEMA]).strip()
    tool_call = parse_qwen_tool_call(output)
    tool_call_validation = validate_task_intent_tool_call(tool_call, args.task)
    task_intent = execute_extract_task_intent_tool(args.task, available_types)
    return tool_call, task_intent, tool_call_validation

def task_intent_source_for_args(args: argparse.Namespace) -> str:
    source = getattr(args, "_task_intent_source", None)
    return source if isinstance(source, str) and source else TASK_INTENT_SOURCE


def _normalize_external_intent_step(step: dict[str, Any], index: int) -> dict[str, Any]:
    order = step.get("order", index)
    if not isinstance(order, int) or isinstance(order, bool) or order <= 0:
        order = index
    return {
        "order": order,
        "action": step.get("action"),
        "objectType": step.get("objectType", step.get("object_type")),
        "targetType": step.get("targetType", step.get("target_type")),
    }


def parse_external_task_intent_json(value: str, original_task: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str, dict[str, Any] | None]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"--task-intent-json must be a JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("--task-intent-json must be a JSON object")

    raw_intent = parsed.get("task_intent") if isinstance(parsed.get("task_intent"), dict) else parsed
    if not isinstance(raw_intent, dict):
        raise RuntimeError("--task-intent-json must contain a task_intent object")

    raw_steps = raw_intent.get("intentSteps")
    steps = []
    if isinstance(raw_steps, list):
        steps = [_normalize_external_intent_step(step, index) for index, step in enumerate(raw_steps, start=1) if isinstance(step, dict)]
    if not steps and raw_intent.get("requestedAction") is not None:
        steps = [
            {
                "order": 1,
                "action": raw_intent.get("requestedAction"),
                "objectType": raw_intent.get("requestedObjectType"),
                "targetType": raw_intent.get("requestedTargetType", raw_intent.get("targetType")),
            }
        ]
    if not steps:
        raise RuntimeError("--task-intent-json did not include usable intentSteps")

    primary_step = primary_intent_step(steps) or steps[0]
    requested_action = raw_intent.get("requestedAction", primary_step.get("action"))
    requested_object = raw_intent.get("requestedObjectType", primary_step.get("objectType"))
    requested_target = raw_intent.get("requestedTargetType", raw_intent.get("targetType", primary_step.get("targetType")))
    task_intent = {
        "requestedAction": requested_action,
        "requestedObjectType": requested_object,
        "requestedTargetType": requested_target,
        "intentSteps": steps,
    }
    tool_call = {
        "name": "external_task_intent",
        "arguments": {
            "task": original_task,
            "requestedAction": requested_action,
            "requestedObjectType": requested_object,
            "requestedTargetType": requested_target,
            "intentSteps": steps,
        },
    }
    validation = {"status": "ok", "warnings": []}
    source = parsed.get("task_intent_source")
    if not isinstance(source, str) or not source:
        source = "external_task_intent"
    metadata = parsed.get("task_normalization") if isinstance(parsed.get("task_normalization"), dict) else None
    return tool_call, task_intent, validation, source, metadata


def task_intent_from_args(
    args: argparse.Namespace,
    available_types: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if getattr(args, "task_intent_json", None):
        tool_call, task_intent, validation, source, metadata = parse_external_task_intent_json(args.task_intent_json, args.task)
        setattr(args, "_task_intent_source", source)
        if metadata is not None:
            setattr(args, "_task_normalization", metadata)
        return tool_call, task_intent, validation
    setattr(args, "_task_intent_source", TASK_INTENT_SOURCE)
    return generate_task_intent_tool_call(args, available_types)


def semantic_normalization_warnings(semantic_plan: dict[str, Any]) -> list[str]:
    warnings = semantic_plan.get("semanticNormalizationWarnings")
    if not isinstance(warnings, list):
        return []
    return [str(warning) for warning in warnings]


def _semantic_step_matches_intent(plan_step: dict[str, Any], intent_step: dict[str, Any]) -> tuple[bool, str | None]:
    required_action = intent_step.get("action")
    planned_action = plan_step.get("action")
    if required_action is not None and planned_action != required_action:
        return False, "action"

    required_object = intent_step.get("objectType")
    if isinstance(required_object, str) and required_object.strip():
        planned_object = plan_step.get("objectType")
        if not isinstance(planned_object, str) or normalize_type(planned_object) != normalize_type(required_object):
            return False, "objectType"

    required_target = intent_step.get("targetType")
    if isinstance(required_target, str) and required_target.strip():
        planned_target = plan_step.get("targetType")
        if not isinstance(planned_target, str) or normalize_type(planned_target) != normalize_type(required_target):
            return False, "targetType"

    return True, None


def validate_intent_steps_consistency(task_intent: dict[str, Any], semantic_plan: dict[str, Any]) -> None:
    required_steps = intent_steps(task_intent)
    if not required_steps:
        return
    plan_steps = semantic_plan.get("plan")
    if not isinstance(plan_steps, list):
        raise ValueError("semantic plan must contain a plan list")

    cursor = 0
    for required_index, required_step in enumerate(required_steps, start=1):
        required_action = required_step.get("action")
        if not isinstance(required_action, str) or not required_action.strip():
            continue
        matched = False
        saw_action_match = None
        saw_object_match = None
        for plan_index in range(cursor, len(plan_steps)):
            plan_step = plan_steps[plan_index]
            if not isinstance(plan_step, dict):
                continue
            planned_action = plan_step.get("action")
            if planned_action in {"Pass", "Done"}:
                continue
            if planned_action == required_action:
                saw_action_match = plan_step
            if (
                isinstance(required_step.get("objectType"), str)
                and isinstance(plan_step.get("objectType"), str)
                and normalize_type(plan_step.get("objectType")) == normalize_type(required_step.get("objectType"))
            ):
                saw_object_match = plan_step
            step_matches, mismatch_field = _semantic_step_matches_intent(plan_step, required_step)
            if step_matches:
                cursor = plan_index + 1
                matched = True
                break
            if planned_action == required_action and mismatch_field == "objectType":
                raise ValueError(
                    f"planned object {plan_step.get('objectType')!r} does not match intent step "
                    f"{required_index} object {required_step.get('objectType')!r}; refusing to execute"
                )
            if planned_action == required_action and mismatch_field == "targetType":
                raise ValueError(
                    f"planned target {plan_step.get('targetType')!r} does not match intent step "
                    f"{required_index} target {required_step.get('targetType')!r}; refusing to execute"
                )
        if matched:
            continue
        if saw_object_match is not None and saw_object_match.get("action") != required_action:
            raise ValueError(
                f"planned action {saw_object_match.get('action')!r} does not match intent step "
                f"{required_index} action {required_action!r}; refusing to execute"
            )
        planned_action = "none"
        for plan_step in plan_steps[cursor:]:
            if isinstance(plan_step, dict) and isinstance(plan_step.get("action"), str) and plan_step.get("action") not in {"Pass", "Done"}:
                planned_action = plan_step.get("action")
                break
        raise ValueError(
            f"semantic plan is missing intent step {required_index}: {required_action}"
            f"({required_step.get('objectType')!r}, {required_step.get('targetType')!r}); "
            f"next planned action was {planned_action!r}; refusing to execute"
        )


def validate_task_intent_consistency(
    task_intent: dict[str, Any],
    semantic_plan: dict[str, Any],
    *,
    check_action: bool = True,
) -> None:
    steps = intent_steps(task_intent)
    if len(steps) > 1:
        if check_action:
            validate_intent_steps_consistency(task_intent, semantic_plan)
        return
    if len(steps) == 1 and check_action:
        validate_intent_steps_consistency(task_intent, semantic_plan)
        return

    requested_object = task_intent.get("requestedObjectType")
    if isinstance(requested_object, str) and requested_object.strip():
        model_object = semantic_plan.get("targetObjectType")
        if not isinstance(model_object, str) or normalize_type(model_object) != normalize_type(requested_object):
            raise ValueError(
                f"model targetObjectType {model_object!r} does not match task intent object "
                f"{requested_object!r}; refusing to execute"
            )

    if not check_action:
        return

    requested_action = task_intent.get("requestedAction")
    if isinstance(requested_action, str) and requested_action.strip():
        planned_actions = semantic_plan_actions(semantic_plan)
        if requested_action not in planned_actions:
            planned_action = planned_actions[0] if planned_actions else "none"
            raise ValueError(
                f"planned action {planned_action!r} does not match task intent action "
                f"{requested_action!r}; refusing to execute"
            )



def _intent_step(action: str, object_type: str | None, target_type: str | None) -> dict[str, Any]:
    return {
        "order": 0,
        "action": action,
        "objectType": object_type,
        "targetType": target_type,
    }


def _object_for_type(objects: list[dict[str, Any]], object_type: Any) -> dict[str, Any] | None:
    normalized = normalize_type(object_type)
    if not normalized:
        return None
    for item in objects:
        if isinstance(item, dict) and normalize_type(object_type_of(item)) == normalized:
            return item
    return None


def _has_unmatched_pickup(expanded_steps: list[dict[str, Any]], object_type: str | None) -> bool:
    held_type: str | None = None
    for step in expanded_steps:
        action = step.get("action")
        if action == "PickupObject":
            step_object = step.get("objectType")
            held_type = step_object if isinstance(step_object, str) and step_object.strip() else None
        elif action in {"PutObject", "DropHandObject"}:
            held_type = None
    return bool(held_type and normalize_type(held_type) == normalize_type(object_type))


def expand_put_object_intent_preconditions(
    task_intent: dict[str, Any],
    observation: dict[str, Any] | None,
    known_held_object_types: list[str] | None = None,
) -> list[str]:
    """Expand implicit preconditions for PutObject while keeping user intent as the source of truth."""
    steps = intent_steps(task_intent)
    if not steps:
        return []

    objects = observation.get("objects", []) if isinstance(observation, dict) else []
    if not isinstance(objects, list):
        objects = []

    held_type = held_object_type_from_observation(observation)
    known_held_normalized = {normalize_type(item) for item in (known_held_object_types or []) if normalize_type(item)}
    expanded_steps: list[dict[str, Any]] = []
    warnings: list[str] = []

    for step in steps:
        action = step.get("action")
        object_type = step.get("objectType") if isinstance(step.get("objectType"), str) else None
        target_type = step.get("targetType") if isinstance(step.get("targetType"), str) else None

        if action == "PutObject":
            target = _object_for_type(objects, target_type)
            if target is not None and bool(target.get("openable")) and target.get("isOpen") is not True:
                already_opened = any(
                    prior.get("action") == "OpenObject"
                    and normalize_type(prior.get("objectType")) == normalize_type(target_type)
                    for prior in expanded_steps
                )
                already_closed_after_open = any(
                    prior.get("action") == "CloseObject"
                    and normalize_type(prior.get("objectType")) == normalize_type(target_type)
                    for prior in expanded_steps
                )
                if not already_opened or already_closed_after_open:
                    expanded_steps.append(_intent_step("OpenObject", target_type, None))
                    warnings.append(
                        f"inserted OpenObject({target_type}) before PutObject because target receptacle is closed or not known open"
                    )

            if object_type and not _has_unmatched_pickup(expanded_steps, object_type):
                object_is_known_held = normalize_type(object_type) in known_held_normalized
                if held_type is None and not object_is_known_held:
                    expanded_steps.append(_intent_step("PickupObject", object_type, None))
                    held_type = object_type
                    warnings.append(
                        f"inserted PickupObject({object_type}) before PutObject because no known robot is holding it"
                    )
                elif normalize_type(held_type) == normalize_type(object_type) or object_is_known_held:
                    pass
                else:
                    # Leave the original PutObject in place so the existing state verifier reports
                    # the clearer "holding X, not Y" failure.
                    pass

            expanded_steps.append(dict(step))
            held_type = None
            continue

        expanded_steps.append(dict(step))
        if action == "PickupObject":
            held_type = object_type
        elif action in {"PutObject", "DropHandObject"}:
            held_type = None
        elif action == "OpenObject":
            pass

    for order, step in enumerate(expanded_steps, start=1):
        step["order"] = order

    if expanded_steps != steps:
        task_intent["intentSteps"] = expanded_steps
        primary_step = primary_intent_step(expanded_steps)
        if primary_step is not None:
            original_requested_action = task_intent.get("requestedAction")
            original_requested_object = task_intent.get("requestedObjectType")
            if not isinstance(original_requested_action, str) or not original_requested_action.strip():
                task_intent["requestedAction"] = primary_step.get("action")
            if not isinstance(original_requested_object, str) or not original_requested_object.strip():
                task_intent["requestedObjectType"] = primary_step.get("objectType")
        if warnings:
            task_intent["intentExpansionWarnings"] = warnings
    return warnings


def repair_redundant_pickup_for_held_put(
    semantic_plan: dict[str, Any],
    task_intent: dict[str, Any],
    observation: dict[str, Any] | None,
) -> list[str]:
    steps = intent_steps(task_intent)
    if len(steps) != 1:
        return []
    intent_step = steps[0]
    if intent_step.get("action") != "PutObject":
        return []
    intent_object = intent_step.get("objectType")
    if not isinstance(intent_object, str) or not intent_object.strip():
        return []
    held_object_type = held_object_type_from_observation(observation)
    if normalize_type(held_object_type) != normalize_type(intent_object):
        return []

    plan_steps = semantic_plan.get("plan")
    if not isinstance(plan_steps, list):
        return []

    put_index = None
    for index, step in enumerate(plan_steps):
        if not isinstance(step, dict):
            continue
        if (
            step.get("action") == "PutObject"
            and normalize_type(step.get("objectType")) == normalize_type(intent_object)
        ):
            put_index = index
            break
    if put_index is None:
        return []

    remove_indices: set[int] = set()
    for index, step in enumerate(plan_steps[:put_index]):
        if not isinstance(step, dict):
            continue
        if (
            step.get("action") == "PickupObject"
            and normalize_type(step.get("objectType")) == normalize_type(intent_object)
        ):
            remove_indices.add(index)

    if not remove_indices:
        return []

    semantic_plan["plan"] = [
        step for index, step in enumerate(plan_steps) if index not in remove_indices
    ]
    return [
        f"removed redundant PickupObject({intent_object}) before PutObject because "
        f"{_robot_label(observation)} is already holding {held_object_type}"
    ]


def first_executable_plan_action(semantic_plan: dict[str, Any]) -> str | None:
    plan_steps = semantic_plan.get("plan")
    if not isinstance(plan_steps, list):
        return None
    for plan_step in plan_steps:
        if not isinstance(plan_step, dict):
            continue
        action = plan_step.get("action")
        if isinstance(action, str) and action not in {"Pass", "Done"}:
            return action
    return None


def put_object_target_type_for_plan(
    semantic_plan: dict[str, Any],
    task_intent: dict[str, Any] | None = None,
) -> str | None:
    plan_steps = semantic_plan.get("plan")
    if isinstance(plan_steps, list):
        for plan_step in plan_steps:
            if not isinstance(plan_step, dict):
                continue
            if plan_step.get("action") == "PutObject":
                target_type = plan_step.get("targetType")
                if isinstance(target_type, str) and target_type.strip():
                    return target_type
    for intent_step in intent_steps(task_intent):
        if intent_step.get("action") == "PutObject":
            target_type = intent_step.get("targetType")
            if isinstance(target_type, str) and target_type.strip():
                return target_type
    return None


def agent_ids_holding_object_from_visibility_map(
    object_visibility_map: dict[str, Any],
    object_type: str | None,
) -> list[str]:
    normalized = normalize_type(object_type)
    if not normalized:
        return []
    agent_ids: list[str] = []
    agents = object_visibility_map.get("agents")
    if not isinstance(agents, list):
        return agent_ids
    for agent in agents:
        if not isinstance(agent, dict):
            continue
        if normalize_type(agent.get("held_object_type")) == normalized:
            agent_id = agent.get("agent_id")
            if isinstance(agent_id, str) and agent_id not in agent_ids:
                agent_ids.append(agent_id)
    return agent_ids


def put_target_failure_for_agent(
    observation: dict[str, Any],
    target_type: str | None,
) -> str | None:
    if not isinstance(target_type, str) or not target_type.strip():
        return f"missing target receptacle for PutObject on {_robot_label(observation)}"
    objects = observation.get("objects", [])
    if not isinstance(objects, list):
        objects = []
    try:
        select_object(
            objects,
            target_type,
            role="receptacle",
            allow_invisible=False,
            action_name="PutObject",
        )
    except ValueError:
        return f"target receptacle {target_type!r} is not visible to {_robot_label(observation)}"
    return None


def coordination_result_for_plan(
    task: str,
    semantic_plan: dict[str, Any],
    object_visibility_map: dict[str, Any],
    task_intent: dict[str, Any] | None = None,
    *,
    relay_mode: bool = False,
) -> dict[str, Any]:
    objects_by_type = object_visibility_map.get("objects_by_type", {})
    available_types = [
        entry.get("object_type")
        for entry in objects_by_type.values()
        if isinstance(entry, dict) and isinstance(entry.get("object_type"), str)
    ] if isinstance(objects_by_type, dict) else []
    requested_type = requested_object_type_for_plan(task, semantic_plan, available_types, task_intent)
    primary_agent_id = str(object_visibility_map.get("primary_agent_id"))
    if requested_type is None:
        return {
            "status": "no_requested_object",
            "requested_object_type": None,
            "primary_agent_id": primary_agent_id,
            "visible_peer_agent_ids": [],
        }

    entry = objects_by_type.get(normalize_type(requested_type)) if isinstance(objects_by_type, dict) else None
    visible_by_agent_ids = []
    if isinstance(entry, dict):
        visible_by_agent_ids = [str(agent_id) for agent_id in entry.get("visible_by_agent_ids", [])]
    visible_peer_agent_ids = [agent_id for agent_id in visible_by_agent_ids if agent_id != primary_agent_id]

    first_action = first_executable_plan_action(semantic_plan)
    held_by_agent_ids = agent_ids_holding_object_from_visibility_map(object_visibility_map, requested_type)
    held_by_peer_agent_ids = [agent_id for agent_id in held_by_agent_ids if agent_id != primary_agent_id]
    if first_action == "PutObject":
        if primary_agent_id in held_by_agent_ids:
            return {
                "status": "primary_visible",
                "requested_object_type": requested_type,
                "primary_agent_id": primary_agent_id,
                "visible_peer_agent_ids": visible_peer_agent_ids,
                "held_by_agent_ids": held_by_agent_ids,
                "message": f"requested object {requested_type!r} is already held by primary agent {primary_agent_id!r}",
            }
        if held_by_peer_agent_ids:
            peer_text = ", ".join(held_by_peer_agent_ids)
            return {
                "status": "target_visible_by_peer",
                "requested_object_type": requested_type,
                "primary_agent_id": primary_agent_id,
                "visible_peer_agent_ids": visible_peer_agent_ids,
                "held_by_agent_ids": held_by_agent_ids,
                "message": (
                    f"requested object {requested_type!r} is already held by peer robot "
                    f"{peer_text!r}; selecting holder as executor"
                ),
            }

    if primary_agent_id in visible_by_agent_ids:
        return {
            "status": "primary_visible",
            "requested_object_type": requested_type,
            "primary_agent_id": primary_agent_id,
            "visible_peer_agent_ids": visible_peer_agent_ids,
        }
    if visible_peer_agent_ids:
        peer_text = ", ".join(visible_peer_agent_ids)
        message = (
            f"requested object {requested_type!r} is not visible to primary agent "
            f"{primary_agent_id!r}, but is visible to peer agent {peer_text!r}; refusing to execute locally"
        )
        if relay_mode:
            best_peer = visible_peer_agent_ids[0]
            message = (
                f"requested object {requested_type!r} is not visible to primary robot "
                f"{primary_agent_id!r}, but is visible to peer robot {best_peer!r}; "
                f"selecting peer robot {best_peer!r} as executor"
            )
        return {
            "status": "target_visible_by_peer",
            "requested_object_type": requested_type,
            "primary_agent_id": primary_agent_id,
            "visible_peer_agent_ids": visible_peer_agent_ids,
            "message": message,
        }
    return {
        "status": "target_not_visible",
        "requested_object_type": requested_type,
        "primary_agent_id": primary_agent_id,
        "visible_peer_agent_ids": [],
    }


def known_agent_ids(agent_observations: list[dict[str, Any]]) -> list[str]:
    return [str(observation.get("agent_id")) for observation in agent_observations]


def known_robot_ids_from_observations(agent_observations: list[dict[str, Any]]) -> list[int]:
    robot_ids = {
        observation.get("robot_id")
        for observation in agent_observations
        if isinstance(observation.get("robot_id"), int)
    }
    return sorted(robot_ids)


def robot_id_for_agent_id(agent_observations: list[dict[str, Any]], agent_id: str) -> int | None:
    for observation in agent_observations:
        if observation.get("agent_id") == agent_id and isinstance(observation.get("robot_id"), int):
            return observation["robot_id"]
    return None



def agent_ids_holding_object(agent_observations: list[dict[str, Any]], object_type: str | None) -> list[str]:
    normalized = normalize_type(object_type)
    if not normalized:
        return []
    agent_ids: list[str] = []
    for observation in agent_observations:
        held_type = held_object_type_from_observation(observation)
        if normalize_type(held_type) == normalized:
            agent_id = observation.get("agent_id")
            if isinstance(agent_id, str) and agent_id not in agent_ids:
                agent_ids.append(agent_id)
    return agent_ids


def held_object_types_from_observations(agent_observations: list[dict[str, Any]]) -> list[str]:
    held_types: list[str] = []
    for observation in agent_observations:
        held_type = held_object_type_from_observation(observation)
        if not isinstance(held_type, str) or not held_type.strip():
            continue
        if normalize_type(held_type) not in {normalize_type(item) for item in held_types}:
            held_types.append(held_type)
    return held_types


def pickup_step_already_satisfied(step: dict[str, Any], observation: dict[str, Any] | None) -> bool:
    if step.get("action") != "PickupObject":
        return False
    object_type = step.get("objectType")
    return bool(normalize_type(object_type) and normalize_type(held_object_type_from_observation(observation)) == normalize_type(object_type))


def object_state_step_already_satisfied(step: dict[str, Any], observation: dict[str, Any] | None) -> bool:
    action = step.get("action")
    if action not in {"OpenObject", "CloseObject"}:
        return False
    try:
        validate_action_state_preconditions({"plan": [step]}, observation, allow_invisible=False)
    except ValueError as exc:
        message = str(exc)
        return (action == "OpenObject" and "already open" in message) or (
            action == "CloseObject" and "already closed" in message
        )
    except Exception:
        return False
    return False


def step_already_satisfied(step: dict[str, Any], observation: dict[str, Any] | None) -> bool:
    return pickup_step_already_satisfied(step, observation) or object_state_step_already_satisfied(step, observation)


def step_already_satisfied_reason(step: dict[str, Any], observation: dict[str, Any] | None) -> str:
    action = step.get("action")
    if action == "PickupObject":
        return (
            f"{_robot_label(observation)} is already holding "
            f"{held_object_type_from_observation(observation)}; skipping PickupObject"
        )
    if action == "OpenObject":
        return f"object {step.get('objectType')!r} is already open; skipping OpenObject"
    if action == "CloseObject":
        return f"object {step.get('objectType')!r} is already closed; skipping CloseObject"
    return f"step {action!r} is already satisfied"


def already_satisfied_observation_for_step(
    step: dict[str, Any],
    agent_observations: list[dict[str, Any]],
    primary_robot_id: int,
) -> dict[str, Any] | None:
    ordered = sorted(
        agent_observations,
        key=lambda observation: 0 if observation.get("robot_id") == primary_robot_id else 1,
    )
    for observation in ordered:
        if step_already_satisfied(step, observation):
            return observation
    return None


def held_owner_agent_id(
    agent_observations: list[dict[str, Any]],
    simulated_held_by_agent_id: dict[str, str | None],
    object_type: str | None,
) -> str | None:
    normalized = normalize_type(object_type)
    if not normalized:
        return None
    for observation in agent_observations:
        agent_id = observation.get("agent_id")
        if not isinstance(agent_id, str):
            continue
        held_type = held_object_type_from_observation(observation) or simulated_held_by_agent_id.get(agent_id)
        if normalize_type(held_type) == normalized:
            return agent_id
    for agent_id, held_type in simulated_held_by_agent_id.items():
        if normalize_type(held_type) == normalized:
            return agent_id
    return None


def relay_result_for_held_put_step(
    step: dict[str, Any],
    object_visibility_map: dict[str, Any],
    agent_observations: list[dict[str, Any]],
    simulated_held_by_agent_id: dict[str, str | None],
) -> dict[str, Any] | None:
    if step.get("action") != "PutObject":
        return None
    object_type = step.get("objectType")
    if not isinstance(object_type, str) or not object_type.strip():
        return None
    holder_agent_id = held_owner_agent_id(agent_observations, simulated_held_by_agent_id, object_type)
    if holder_agent_id is None:
        return None
    primary_agent_id = str(object_visibility_map.get("primary_agent_id"))
    holder_observation = agent_observation_by_id(agent_observations, holder_agent_id)
    target_failure = put_target_failure_for_agent(holder_observation, step.get("targetType"))
    if target_failure is not None:
        return relay_failure_result(
            target_failure,
            requested_object_type=object_type,
            primary_agent_id=primary_agent_id,
            agent_observations=agent_observations,
        )
    owner_kind = "primary agent" if holder_agent_id == primary_agent_id else "relay executor"
    return {
        "status": "executor_selected",
        "requested_object_type": object_type,
        "primary_agent_id": primary_agent_id,
        "executor_agent_id": holder_agent_id,
        "executor_robot_id": robot_id_for_agent_id(agent_observations, holder_agent_id),
        "primary_robot_id": robot_id_for_agent_id(agent_observations, primary_agent_id),
        "reason": f"requested object {object_type!r} is already held by {owner_kind} {holder_agent_id!r}",
    }


def relay_failure_result(
    reason: str,
    *,
    requested_object_type: str | None,
    primary_agent_id: str,
    agent_observations: list[dict[str, Any]],
    known_robot_ids: list[int] | None = None,
) -> dict[str, Any]:
    primary_robot_id = robot_id_for_agent_id(agent_observations, primary_agent_id)
    return {
        "status": "needs_upstream_planning",
        "reason": reason,
        "requested_object_type": requested_object_type,
        "primary_agent_id": primary_agent_id,
        "primary_robot_id": primary_robot_id,
        "known_agent_ids": known_agent_ids(agent_observations),
        "known_robot_ids": known_robot_ids if known_robot_ids is not None else known_robot_ids_from_observations(agent_observations),
    }


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _point3_from_value(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    point: dict[str, float] = {}
    for axis in ("x", "y", "z"):
        number = _numeric(value.get(axis))
        if number is not None:
            point[axis] = number
    if "x" in point and "z" in point:
        point.setdefault("y", 0.0)
        return point
    return None


def _position_from_value(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    for key in ("position", "agentPosition", "robot_position", "center"):
        point = _point3_from_value(value.get(key))
        if point is not None:
            return point
    for box_key in ("axisAlignedBoundingBox", "objectOrientedBoundingBox"):
        box = value.get(box_key)
        if isinstance(box, dict):
            point = _point3_from_value(box.get("center"))
            if point is not None:
                return point
    return _point3_from_value(value)


def _robot_position_from_observation(observation: dict[str, Any]) -> dict[str, float] | None:
    robot_state = observation.get("robot_state")
    point = _position_from_value(robot_state)
    if point is not None:
        return point
    return _position_from_value(observation)


def _distance_between_points(a: dict[str, float] | None, b: dict[str, float] | None) -> float | None:
    if a is None or b is None:
        return None
    return sum((a.get(axis, 0.0) - b.get(axis, 0.0)) ** 2 for axis in ("x", "y", "z")) ** 0.5


def _visible_objects_of_type(observation: dict[str, Any], object_type: str | None) -> list[dict[str, Any]]:
    normalized = normalize_type(object_type)
    if not normalized:
        return []
    objects = observation.get("objects", [])
    if not isinstance(objects, list):
        return []
    return [
        item
        for item in objects
        if isinstance(item, dict)
        and bool(item.get("visible"))
        and normalize_type(object_type_of(item)) == normalized
    ]


def _distance_to_visible_object(observation: dict[str, Any], object_type: str | None) -> float | None:
    robot_position = _robot_position_from_observation(observation)
    distances = [
        distance
        for item in _visible_objects_of_type(observation, object_type)
        for distance in [_distance_between_points(robot_position, _position_from_value(item))]
        if distance is not None
    ]
    return min(distances) if distances else None


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[int, float, int, int]:
    distance = candidate.get("distance_to_target")
    robot_id = candidate.get("robot_id")
    return (
        1 if distance is None else 0,
        float("inf") if distance is None else float(distance),
        0 if candidate.get("is_primary") else 1,
        robot_id if isinstance(robot_id, int) else 1_000_000,
    )


def _selection_metadata(
    candidates: list[dict[str, Any]],
    selected: dict[str, Any] | None,
) -> dict[str, Any]:
    executable = sorted(
        [candidate for candidate in candidates if candidate.get("executable")],
        key=_candidate_sort_key,
    )
    return {
        "selection_policy": "executable > nearest_target_distance > primary > robot_id",
        "candidate_executor_robot_ids": [
            candidate["robot_id"]
            for candidate in executable
            if isinstance(candidate.get("robot_id"), int)
        ],
        "candidate_scores": sorted(
            candidates,
            key=lambda candidate: (0 if candidate.get("executable") else 1, *_candidate_sort_key(candidate)),
        ),
        "selected_distance_to_target": selected.get("distance_to_target") if selected is not None else None,
    }


def _holds_object_type(observation: dict[str, Any], object_type: str | None) -> bool:
    normalized = normalize_type(object_type)
    if not normalized:
        return False
    held_type = held_object_type_from_observation(observation)
    if normalize_type(held_type) == normalized:
        return True
    inventory = observation.get("inventory", [])
    if not isinstance(inventory, list):
        return False
    return any(
        isinstance(item, dict) and normalize_type(object_type_of(item)) == normalized
        for item in inventory
    )


def _agent_visibility_evidence(observation: dict[str, Any], object_type: str | None) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for item in _visible_objects_of_type(observation, object_type):
        evidence.append(
            {
                "object_type": object_type_of(item),
                "object_id": item.get("objectId") or item.get("object_id") or item.get("id"),
                "position": _position_from_value(item),
                "affordances": {
                    field: bool(item.get(field))
                    for field in OBJECT_VISIBILITY_AFFORDANCE_FIELDS
                    if field in item
                },
            }
        )
    return evidence


def relay_global_scene_summary(
    task: str,
    semantic_plan: dict[str, Any],
    task_intent: dict[str, Any] | None,
    object_visibility_map: dict[str, Any],
    agent_observations: list[dict[str, Any]],
    known_robot_ids: list[int],
    primary_robot_id: int,
    observation_errors: dict[int, str] | None = None,
) -> dict[str, Any]:
    objects_by_type = object_visibility_map.get("objects_by_type", {})
    available_types = [
        entry.get("object_type")
        for entry in objects_by_type.values()
        if isinstance(entry, dict) and isinstance(entry.get("object_type"), str)
    ] if isinstance(objects_by_type, dict) else []
    step = relay_routing_step(semantic_plan, task_intent)
    requested_type = requested_object_type_for_plan(task, semantic_plan, available_types, task_intent)
    observed_robot_ids = known_robot_ids_from_observations(agent_observations)
    return {
        "task": task,
        "task_intent": task_intent or {},
        "routing_step": step,
        "requested_object_type": requested_type,
        "target_receptacle_type": put_object_target_type_for_plan(semantic_plan, task_intent) if step.get("action") == "PutObject" else None,
        "primary_robot_id": primary_robot_id,
        "known_robot_ids": known_robot_ids,
        "observed_robot_ids": observed_robot_ids,
        "visibility_unknown_robot_ids": [robot_id for robot_id in known_robot_ids if robot_id not in observed_robot_ids],
        "observation_errors": observation_errors or {},
        "object_visibility_summary": object_visibility_summary(object_visibility_map),
        "agent_summaries": [relay_agent_observation_summary(item) for item in agent_observations],
    }


def evaluate_relay_executor_candidates(
    task: str,
    semantic_plan: dict[str, Any],
    task_intent: dict[str, Any] | None,
    object_visibility_map: dict[str, Any],
    agent_observations: list[dict[str, Any]],
    known_robot_ids: list[int],
    primary_robot_id: int,
    simulated_held_by_agent_id: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    objects_by_type = object_visibility_map.get("objects_by_type", {})
    available_types = [
        entry.get("object_type")
        for entry in objects_by_type.values()
        if isinstance(entry, dict) and isinstance(entry.get("object_type"), str)
    ] if isinstance(objects_by_type, dict) else []
    requested_type = requested_object_type_for_plan(task, semantic_plan, available_types, task_intent)
    step = relay_routing_step(semantic_plan, task_intent)
    action = step.get("action") or first_executable_plan_action(semantic_plan)
    target_type = put_object_target_type_for_plan(semantic_plan, task_intent) if action == "PutObject" else None
    distance_type = target_type if action == "PutObject" else requested_type
    observed_robot_ids = known_robot_ids_from_observations(agent_observations)

    candidates: list[dict[str, Any]] = []
    for observation in agent_observations:
        robot_id = observation.get("robot_id")
        if not isinstance(robot_id, int):
            continue
        accepted, validation_reason = validate_relay_agent_executor(
            robot_id,
            semantic_plan,
            task_intent,
            agent_observations,
            primary_robot_id,
            simulated_held_by_agent_id,
        )
        visible_requested = _agent_visibility_evidence(observation, requested_type)
        visible_receptacle = _agent_visibility_evidence(observation, target_type) if target_type else []
        candidates.append(
            {
                "agent_id": observation.get("agent_id"),
                "robot_id": robot_id,
                "is_primary": robot_id == primary_robot_id or bool(observation.get("is_primary")),
                "action": action,
                "target_object_type": requested_type,
                "target_receptacle_type": target_type,
                "robot_position": _robot_position_from_observation(observation),
                "distance_to_target": _distance_to_visible_object(observation, distance_type),
                "can_see_requested_object": bool(visible_requested),
                "can_see_target_receptacle": bool(visible_receptacle) if target_type else None,
                "holds_requested_object": _holds_object_type(observation, requested_type),
                "visible_requested_objects": visible_requested,
                "visible_target_receptacles": visible_receptacle,
                "held_object_type": held_object_type_from_observation(observation),
                "inventory_object_types": sorted(
                    {
                        object_type_of(item)
                        for item in observation.get("inventory", [])
                        if isinstance(item, dict) and object_type_of(item)
                    }
                ),
                "executable": accepted,
                "validation": validation_reason,
            }
        )

    executable = sorted(
        [candidate for candidate in candidates if candidate.get("executable")],
        key=_candidate_sort_key,
    )
    return {
        "selection_policy": "llm_tool_calling_with_hard_validation",
        "evidence_policy": "candidate evidence only; relay agent makes the final executor choice",
        "requested_object_type": requested_type,
        "target_receptacle_type": target_type,
        "action": action,
        "known_robot_ids": known_robot_ids,
        "observed_robot_ids": observed_robot_ids,
        "visibility_unknown_robot_ids": [robot_id for robot_id in known_robot_ids if robot_id not in observed_robot_ids],
        "candidate_executor_robot_ids": [candidate["robot_id"] for candidate in executable],
        "candidate_scores": sorted(
            candidates,
            key=lambda candidate: (0 if candidate.get("executable") else 1, *_candidate_sort_key(candidate)),
        ),
    }


def choose_relay_executor(
    task: str,
    semantic_plan: dict[str, Any],
    object_visibility_map: dict[str, Any],
    agent_observations: list[dict[str, Any]],
    task_intent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    primary_agent_id = str(object_visibility_map.get("primary_agent_id"))
    primary_robot_id = robot_id_for_agent_id(agent_observations, primary_agent_id)
    objects_by_type = object_visibility_map.get("objects_by_type", {})
    available_types = [
        entry.get("object_type")
        for entry in objects_by_type.values()
        if isinstance(entry, dict) and isinstance(entry.get("object_type"), str)
    ] if isinstance(objects_by_type, dict) else []
    requested_type = requested_object_type_for_plan(task, semantic_plan, available_types, task_intent)
    if requested_type is None:
        return {
            "status": "executor_selected",
            "requested_object_type": None,
            "primary_agent_id": primary_agent_id,
            "executor_agent_id": primary_agent_id,
            "executor_robot_id": primary_robot_id,
            "primary_robot_id": primary_robot_id,
            "selection_policy": "object-free action stays on primary robot",
            "candidate_executor_robot_ids": [primary_robot_id] if isinstance(primary_robot_id, int) else [],
            "candidate_scores": [],
            "reason": "task has no explicit requested object; using primary agent",
        }

    step = relay_routing_step(semantic_plan, task_intent)
    first_action = step.get("action") or first_executable_plan_action(semantic_plan)
    target_type = put_object_target_type_for_plan(semantic_plan, task_intent) if first_action == "PutObject" else None
    candidate_agent_ids: list[str] = []

    if first_action == "PutObject":
        for agent_id in agent_ids_holding_object(agent_observations, requested_type):
            if agent_id not in candidate_agent_ids:
                candidate_agent_ids.append(agent_id)
    else:
        entry = objects_by_type.get(normalize_type(requested_type)) if isinstance(objects_by_type, dict) else None
        if isinstance(entry, dict):
            for agent_id in entry.get("visible_by_agent_ids", []):
                agent_id = str(agent_id)
                if agent_id not in candidate_agent_ids:
                    candidate_agent_ids.append(agent_id)
        if not candidate_agent_ids:
            for observation in agent_observations:
                agent_id = observation.get("agent_id")
                if isinstance(agent_id, str) and _visible_objects_of_type(observation, requested_type):
                    candidate_agent_ids.append(agent_id)

    candidates: list[dict[str, Any]] = []
    rejected_reasons: list[str] = []
    for agent_id in candidate_agent_ids:
        try:
            observation = agent_observation_by_id(agent_observations, agent_id)
        except ValueError as exc:
            rejected_reasons.append(f"{agent_id}: {exc}")
            continue
        robot_id = observation.get("robot_id")
        if not isinstance(robot_id, int):
            rejected_reasons.append(f"{agent_id}: missing robot_id")
            continue
        accepted, validation_reason = validate_relay_agent_executor(
            robot_id,
            semantic_plan,
            task_intent,
            agent_observations,
            primary_robot_id if isinstance(primary_robot_id, int) else robot_id,
        )
        distance_type = target_type if first_action == "PutObject" else requested_type
        distance = _distance_to_visible_object(observation, distance_type)
        candidate = {
            "agent_id": agent_id,
            "robot_id": robot_id,
            "is_primary": agent_id == primary_agent_id,
            "action": first_action,
            "target_object_type": requested_type,
            "distance_to_target": distance,
            "executable": accepted,
            "validation": validation_reason,
        }
        candidates.append(candidate)
        if not accepted:
            rejected_reasons.append(f"robot {robot_id}: {validation_reason}")

    executable_candidates = sorted(
        [candidate for candidate in candidates if candidate.get("executable")],
        key=_candidate_sort_key,
    )
    if executable_candidates:
        selected = executable_candidates[0]
        selected_agent_id = str(selected["agent_id"])
        selected_robot_id = selected["robot_id"]
        distance = selected.get("distance_to_target")
        if first_action == "PutObject":
            held_text = f"requested object {requested_type!r} is already held by executor robot {selected_robot_id}"
            if distance is None:
                reason = f"{held_text}; no target distance was available, so fallback ordering selected it"
            else:
                reason = (
                    f"{held_text} and it is closest to the target receptacle among executable candidates "
                    f"(distance {distance:.3f})"
                )
        elif distance is None:
            reason = (
                f"robot {selected_robot_id} can execute {first_action} {requested_type}; "
                "no target distance was available, so fallback ordering selected it"
            )
        else:
            reason = (
                f"robot {selected_robot_id} can execute {first_action} {requested_type} "
                f"and is closest among executable candidates (distance {distance:.3f})"
            )
        return {
            "status": "executor_selected",
            "requested_object_type": requested_type,
            "primary_agent_id": primary_agent_id,
            "executor_agent_id": selected_agent_id,
            "executor_robot_id": selected_robot_id,
            "primary_robot_id": primary_robot_id,
            "reason": reason,
            **_selection_metadata(candidates, selected),
        }

    if candidates:
        detail = "; ".join(rejected_reasons) or "no candidate passed hard validation"
        return {
            **relay_failure_result(
                f"requested object {requested_type!r} has visible or holding candidates, but none can execute: {detail}",
                requested_object_type=requested_type,
                primary_agent_id=primary_agent_id,
                agent_observations=agent_observations,
            ),
            **_selection_metadata(candidates, None),
        }

    return {
        **relay_failure_result(
            f"requested object {requested_type!r} is not visible to any known robot",
            requested_object_type=requested_type,
            primary_agent_id=primary_agent_id,
            agent_observations=agent_observations,
        ),
        **_selection_metadata([], None),
    }


def agent_observation_by_robot_id(
    agent_observations: list[dict[str, Any]],
    robot_id: int,
) -> dict[str, Any]:
    for observation in agent_observations:
        if observation.get("robot_id") == robot_id:
            return observation
    available = ", ".join(str(item.get("robot_id")) for item in agent_observations)
    raise ValueError(f"robot {robot_id!r} was not observed; available robot ids: {available}")


def relay_routing_step(
    semantic_plan: dict[str, Any],
    task_intent: dict[str, Any] | None,
) -> dict[str, Any]:
    steps = intent_steps(task_intent)
    if steps:
        return steps[0]
    plan = semantic_plan.get("plan")
    if isinstance(plan, list):
        for step in plan:
            if isinstance(step, dict) and isinstance(step.get("action"), str):
                return step
    return {
        "action": task_intent.get("requestedAction") if isinstance(task_intent, dict) else None,
        "objectType": task_intent.get("requestedObjectType") if isinstance(task_intent, dict) else None,
        "targetType": None,
    }


def validate_relay_agent_executor(
    robot_id: int,
    semantic_plan: dict[str, Any],
    task_intent: dict[str, Any] | None,
    agent_observations: list[dict[str, Any]],
    primary_robot_id: int,
    simulated_held_by_agent_id: dict[str, str | None] | None = None,
) -> tuple[bool, str]:
    try:
        observation = agent_observation_by_robot_id(agent_observations, robot_id)
    except ValueError as exc:
        return False, str(exc)
    step = relay_routing_step(semantic_plan, task_intent)
    action = step.get("action")
    object_type = step.get("objectType")
    if not isinstance(action, str) or not action:
        return False, "task does not contain a supported executable action"
    if action in NO_ARG_ACTIONS:
        if robot_id != primary_robot_id:
            return False, f"{action} has no object target and must remain on primary robot {primary_robot_id}"
        return True, f"primary robot {robot_id} can execute object-free action {action}"

    simulated_held_type = None
    if simulated_held_by_agent_id is not None:
        simulated_held_type = simulated_held_by_agent_id.get(str(observation.get("agent_id")))
    routing_plan = {
        "task": task_text_for_intent_step(step),
        "targetObjectType": object_type,
        "needsGrounding": True,
        "observations": [],
        "plan": [step],
    }
    try:
        if action == "PutObject":
            validate_put_object_goal_consistency(
                routing_plan,
                observation,
                simulated_held_object_type=simulated_held_type,
            )
        else:
            validate_action_affordances(routing_plan, observation.get("objects", []), allow_invisible=False)
            validate_action_state_preconditions(
                routing_plan,
                observation,
                allow_invisible=False,
                simulated_held_object_type=simulated_held_type,
            )
    except ValueError as exc:
        return False, str(exc)
    target_text = f" {object_type}" if isinstance(object_type, str) and object_type else ""
    return True, f"robot {robot_id} has verified state and affordances for {action}{target_text}"



def primary_fast_path_relay_result(
    semantic_plan: dict[str, Any],
    task_intent: dict[str, Any] | None,
    agent_observations: list[dict[str, Any]],
    primary_robot_id: int,
    simulated_held_by_agent_id: dict[str, str | None] | None = None,
) -> dict[str, Any] | None:
    accepted, reason = validate_relay_agent_executor(
        primary_robot_id,
        semantic_plan,
        task_intent,
        agent_observations,
        primary_robot_id,
        simulated_held_by_agent_id,
    )
    if not accepted:
        return None
    primary_observation = agent_observation_by_robot_id(agent_observations, primary_robot_id)
    step = relay_routing_step(semantic_plan, task_intent)
    primary_agent_id = primary_observation.get("agent_id")
    return {
        "status": "executor_selected",
        "strategy": "primary_fast_path",
        "executor_robot_id": primary_robot_id,
        "executor_agent_id": primary_agent_id,
        "primary_robot_id": primary_robot_id,
        "primary_agent_id": primary_agent_id,
        "requested_object_type": step.get("objectType"),
        "reason": reason,
    }



def relay_step_label(step: dict[str, Any]) -> str:
    action = step.get("action")
    object_type = step.get("objectType")
    target_type = step.get("targetType")
    parts = [str(action)] if isinstance(action, str) and action else ["task"]
    if isinstance(object_type, str) and object_type:
        parts.append(object_type)
    if isinstance(target_type, str) and target_type:
        parts.append(f"-> {target_type}")
    return " ".join(parts)


def primary_inability_reason_for_step(
    semantic_plan: dict[str, Any],
    task_intent: dict[str, Any] | None,
    agent_observations: list[dict[str, Any]],
    primary_robot_id: int,
    simulated_held_by_agent_id: dict[str, str | None] | None = None,
) -> str:
    accepted, reason = validate_relay_agent_executor(
        primary_robot_id,
        semantic_plan,
        task_intent,
        agent_observations,
        primary_robot_id,
        simulated_held_by_agent_id,
    )
    step = relay_routing_step(semantic_plan, task_intent)
    if accepted:
        return f"robot {primary_robot_id} can execute {relay_step_label(step)}: {reason}"
    return f"robot {primary_robot_id} cannot execute {relay_step_label(step)}: {reason}"


def relay_candidate_explanation_summary(relay_result: dict[str, Any], *, limit: int = 5) -> str:
    candidate_scores = relay_result.get("candidate_scores")
    if not isinstance(candidate_scores, list):
        candidate_scores = []
    parts: list[str] = []
    seen_robot_ids: set[int] = set()
    for candidate in candidate_scores:
        if not isinstance(candidate, dict):
            continue
        robot_id = candidate.get("robot_id")
        if not isinstance(robot_id, int) or isinstance(robot_id, bool):
            continue
        seen_robot_ids.add(robot_id)
        status = "executable" if candidate.get("executable") else "rejected"
        details: list[str] = []
        validation = candidate.get("validation")
        if validation is not None:
            details.append(_truncated_text(validation, 140))
        distance = candidate.get("distance_to_target")
        if isinstance(distance, (int, float)) and not isinstance(distance, bool):
            details.append(f"distance={distance:.3f}")
        detail_text = "; ".join(details) if details else "no validation detail"
        parts.append(f"robot_{robot_id} {status} ({detail_text})")
        if len(parts) >= limit:
            return "; ".join(parts)

    observation_errors = relay_result.get("observation_errors")
    if isinstance(observation_errors, dict):
        for raw_robot_id, error in observation_errors.items():
            try:
                robot_id = int(raw_robot_id)
            except (TypeError, ValueError):
                continue
            if robot_id in seen_robot_ids:
                continue
            seen_robot_ids.add(robot_id)
            parts.append(f"robot_{robot_id} rejected ({_truncated_text(error, 140)})")
            if len(parts) >= limit:
                return "; ".join(parts)

    unknown_ids = relay_result.get("visibility_unknown_robot_ids")
    if isinstance(unknown_ids, list):
        for robot_id in unknown_ids:
            if not isinstance(robot_id, int) or isinstance(robot_id, bool) or robot_id in seen_robot_ids:
                continue
            seen_robot_ids.add(robot_id)
            parts.append(f"robot_{robot_id} unknown (visibility unknown)")
            if len(parts) >= limit:
                return "; ".join(parts)
    return "; ".join(parts)


def relay_coordination_explanation(relay_result: dict[str, Any]) -> str:
    status = relay_result.get("status")
    executor_robot_id = relay_result.get("executor_robot_id")
    if status in {"executor_selected", "executor_ready"} and isinstance(executor_robot_id, int):
        agent_reason = relay_result.get("agent_reason") or relay_result.get("reason")
        validation_reason = relay_result.get("validation_reason")
        details: list[str] = []
        if agent_reason:
            details.append(_truncated_text(agent_reason, 180))
        if validation_reason:
            details.append(f"validated: {_truncated_text(validation_reason, 180)}")
        detail_text = "; ".join(details) if details else "executor passed relay validation"
        return f"coordination succeeded: robot_{executor_robot_id} selected; {detail_text}"

    failure_code = relay_result.get("failure_code")
    reason = relay_result.get("reason")
    code_text = str(failure_code) if failure_code else "no_executor_selected"
    reason_text = _truncated_text(reason, 220) if reason else "relay did not provide a reason"
    return f"coordination failed: {code_text}; {reason_text}"


def relay_explanation_for_trace(relay_result: dict[str, Any]) -> dict[str, str]:
    explanation: dict[str, str] = {}
    for key in ("primary_inability_reason", "coordination_explanation", "candidate_explanation_summary"):
        value = relay_result.get(key)
        if isinstance(value, str) and value:
            explanation[key] = value
    return explanation


def print_relay_handoff_explanation(
    *,
    primary_inability_reason: str | None = None,
    relay_result: dict[str, Any] | None = None,
) -> None:
    if primary_inability_reason:
        print(f"[relay] primary cannot execute: {primary_inability_reason}", file=sys.stderr)
    if relay_result is None:
        return
    coordination = relay_result.get("coordination_explanation")
    if isinstance(coordination, str) and coordination:
        print(f"[relay] {coordination}", file=sys.stderr)
    candidates = relay_result.get("candidate_explanation_summary")
    if isinstance(candidates, str) and candidates:
        print(f"[relay] candidates: {candidates}", file=sys.stderr)


def parse_known_robot_ids(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    robot_ids: list[int] = []
    for part in value.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        try:
            robot_id = int(stripped)
        except ValueError as exc:
            raise ValueError(f"invalid robot id in --known-robot-ids: {stripped!r}") from exc
        if robot_id not in robot_ids:
            robot_ids.append(robot_id)
    return robot_ids


def robot_ids_from_state_response(probe: dict[str, Any]) -> list[int]:
    robot_ids: list[int] = []
    containers: list[dict[str, Any]] = [probe]
    state = probe.get("state")
    if isinstance(state, dict):
        containers.append(state)
    metadata = probe.get("metadata")
    if isinstance(metadata, dict):
        containers.append(metadata)
    for container in containers:
        robots = container.get("robots")
        if not isinstance(robots, list):
            continue
        for robot in robots:
            if not isinstance(robot, dict):
                continue
            robot_id = None
            for key in ("robot_id", "robotId", "id"):
                robot_id = _robot_id_from_value(robot.get(key))
                if robot_id is not None:
                    break
            if robot_id is not None and robot_id not in robot_ids:
                robot_ids.append(robot_id)
    return robot_ids


def merge_robot_ids(*robot_id_lists: list[int]) -> list[int]:
    merged: list[int] = []
    for robot_ids in robot_id_lists:
        for robot_id in robot_ids:
            if robot_id not in merged:
                merged.append(robot_id)
    return merged


def known_robot_ids_for_run(
    args: argparse.Namespace,
    probe: dict[str, Any],
    agent_observations: list[dict[str, Any]],
) -> list[int]:
    explicit = parse_known_robot_ids(args.known_robot_ids)
    if explicit is not None:
        return explicit
    discovered = robot_ids_from_state_response(probe)
    if discovered:
        return discovered
    observed = known_robot_ids_from_observations(agent_observations)
    if observed:
        return observed
    return [args.primary_robot_id]


def robot_discovery_source_for_run(
    args: argparse.Namespace,
    probe: dict[str, Any],
    agent_observations: list[dict[str, Any]],
) -> str:
    if parse_known_robot_ids(args.known_robot_ids) is not None:
        return "--known-robot-ids"
    if robot_ids_from_state_response(probe):
        return "execute_actions_state"
    if known_robot_ids_from_observations(agent_observations):
        return "observed"
    return "primary_robot_id"


def discover_global_robot_ids_if_needed(
    args: argparse.Namespace,
    base_url: str,
    known_robot_ids: list[int],
) -> tuple[list[int], str | None]:
    if parse_known_robot_ids(args.known_robot_ids) is not None:
        return known_robot_ids, None
    try:
        global_state = get_global_state(base_url, args.state_endpoint, timeout=args.send_timeout)
    except RuntimeError as exc:
        print(f"warning: failed to discover robots from global state: {exc}", file=sys.stderr)
        return known_robot_ids, None
    global_robot_ids = robot_ids_from_state_response(global_state)
    if not global_robot_ids:
        return known_robot_ids, None
    return merge_robot_ids(global_robot_ids, known_robot_ids, [args.primary_robot_id]), "global_state_fallback"


def target_visible_to_primary(coordination_result: dict[str, Any]) -> bool:
    return coordination_result.get("status") in {"primary_visible", "no_requested_object"}


def append_observation_if_new(
    agent_observations: list[dict[str, Any]],
    observation: dict[str, Any],
) -> None:
    robot_id = observation.get("robot_id")
    agent_id = observation.get("agent_id")
    for index, existing in enumerate(agent_observations):
        if robot_id is not None and existing.get("robot_id") == robot_id:
            agent_observations[index] = observation
            return
        if robot_id is None and existing.get("agent_id") == agent_id:
            agent_observations[index] = observation
            return
    agent_observations.append(observation)


def put_object_types_requiring_ownership(task_intent: dict[str, Any]) -> list[str]:
    object_types: list[str] = []
    for step in intent_steps(task_intent):
        if step.get("action") != "PutObject":
            continue
        object_type = step.get("objectType")
        if not isinstance(object_type, str) or not object_type.strip():
            continue
        if normalize_type(object_type) not in {normalize_type(item) for item in object_types}:
            object_types.append(object_type)
    return object_types


def query_relay_observations_for_ownership(
    args: argparse.Namespace,
    task_id: str,
    base_url: str,
    probe: dict[str, Any],
    agent_observations: list[dict[str, Any]],
    object_visibility_map: dict[str, Any],
    known_robot_ids: list[int],
    queried_robot_ids: list[int],
    robot_discovery_source: str,
    object_types: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[int], list[int], str]:
    if not object_types:
        return agent_observations, object_visibility_map, known_robot_ids, queried_robot_ids, robot_discovery_source

    if robot_discovery_source != "execute_actions_state":
        known_robot_ids, global_source = discover_global_robot_ids_if_needed(args, base_url, known_robot_ids)
        if global_source is not None:
            robot_discovery_source = global_source

    for robot_id in known_robot_ids:
        if robot_id in queried_robot_ids:
            continue
        try:
            response = execute_actions_probe_scene(
                args.execute_actions_url,
                task_id,
                args.send_timeout,
                robot_id=robot_id,
            )
            observations = extract_agent_observations(response, primary_robot_id=robot_id)
        except RuntimeError as exc:
            print(f"warning: failed to probe robot {robot_id} via execute_actions: {exc}", file=sys.stderr)
            queried_robot_ids.append(robot_id)
            continue
        if not observations:
            queried_robot_ids.append(robot_id)
            continue
        observation = observations[0]
        observation["is_primary"] = observation.get("robot_id") == args.primary_robot_id
        append_observation_if_new(agent_observations, observation)
        queried_robot_ids.append(robot_id)
        save_agent_observation_images(agent_observations, args.output_dir.expanduser().resolve(), task_id)
        object_visibility_map = build_object_visibility_map(agent_observations)

    return agent_observations, object_visibility_map, known_robot_ids, queried_robot_ids, robot_discovery_source


def query_relay_observations_if_needed(
    args: argparse.Namespace,
    task_id: str,
    base_url: str,
    probe: dict[str, Any],
    primary_semantic_plan: dict[str, Any],
    agent_observations: list[dict[str, Any]],
    object_visibility_map: dict[str, Any],
    coordination_result: dict[str, Any],
    task_intent: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[int], list[int], str]:
    known_robot_ids = known_robot_ids_for_run(args, probe, agent_observations)
    robot_discovery_source = robot_discovery_source_for_run(args, probe, agent_observations)
    queried_robot_ids = known_robot_ids_from_observations(agent_observations)
    first_action = first_executable_plan_action(primary_semantic_plan)
    requested_type_for_primary = requested_object_type_for_plan(
        args.task,
        primary_semantic_plan,
        object_categories(primary_agent_observation(agent_observations)["objects"], visible_only=False),
        task_intent,
    )
    primary_visible_lookup = _object_type_lookup(
        object_categories(primary_agent_observation(agent_observations).get("objects", []), visible_only=True)
    )
    pickup_requires_peer_query = (
        first_action == "PickupObject"
        and normalize_type(requested_type_for_primary) not in primary_visible_lookup
        and not agent_ids_holding_object(agent_observations, requested_type_for_primary)
    )
    if target_visible_to_primary(coordination_result) and not pickup_requires_peer_query:
        return (
            agent_observations,
            object_visibility_map,
            coordination_result,
            known_robot_ids,
            queried_robot_ids,
            robot_discovery_source,
        )

    if robot_discovery_source != "execute_actions_state":
        known_robot_ids, global_source = discover_global_robot_ids_if_needed(args, base_url, known_robot_ids)
        if global_source is not None:
            robot_discovery_source = global_source

    requested_type = requested_object_type_for_plan(
        args.task,
        primary_semantic_plan,
        object_categories(primary_agent_observation(agent_observations)["objects"], visible_only=False),
        task_intent,
    )
    for robot_id in known_robot_ids:
        if robot_id in queried_robot_ids:
            continue
        try:
            response = execute_actions_probe_scene(
                args.execute_actions_url,
                task_id,
                args.send_timeout,
                robot_id=robot_id,
            )
            observations = extract_agent_observations(response, primary_robot_id=robot_id)
        except RuntimeError as exc:
            print(f"warning: failed to probe robot {robot_id} via execute_actions: {exc}", file=sys.stderr)
            queried_robot_ids.append(robot_id)
            continue
        if not observations:
            queried_robot_ids.append(robot_id)
            continue
        observation = observations[0]
        observation["is_primary"] = observation.get("robot_id") == args.primary_robot_id
        append_observation_if_new(agent_observations, observation)
        queried_robot_ids.append(robot_id)
        save_agent_observation_images(agent_observations, args.output_dir.expanduser().resolve(), task_id)
        object_visibility_map = build_object_visibility_map(agent_observations)
        coordination_result = coordination_result_for_plan(
            args.task,
            primary_semantic_plan,
            object_visibility_map,
            task_intent,
            relay_mode=args.relay_mode,
        )
        if requested_type is None or coordination_result.get("status") in {"primary_visible", "target_visible_by_peer"}:
            break

    return (
        agent_observations,
        object_visibility_map,
        coordination_result,
        known_robot_ids,
        queried_robot_ids,
        robot_discovery_source,
    )


def validate_relay_agent_failure(
    requested_code: str,
    agent_reason: str,
    semantic_plan: dict[str, Any],
    task_intent: dict[str, Any] | None,
    agent_observations: list[dict[str, Any]],
    primary_robot_id: int,
    observation_errors: dict[int, str],
    simulated_held_by_agent_id: dict[str, str | None] | None = None,
) -> tuple[bool, str, str]:
    valid_robot_ids: list[int] = []
    rejected_reasons: list[str] = []
    for observation in agent_observations:
        robot_id = observation.get("robot_id")
        if not isinstance(robot_id, int):
            continue
        accepted, reason = validate_relay_agent_executor(
            robot_id,
            semantic_plan,
            task_intent,
            agent_observations,
            primary_robot_id,
            simulated_held_by_agent_id,
        )
        if accepted:
            valid_robot_ids.append(robot_id)
        else:
            rejected_reasons.append(f"robot {robot_id}: {reason}")
    if valid_robot_ids:
        return (
            False,
            "executor_available",
            f"failure rejected because robot(s) {valid_robot_ids} satisfy the hard executor constraints",
        )

    step = relay_routing_step(semantic_plan, task_intent)
    action = step.get("action")
    object_type = step.get("objectType")
    if not isinstance(action, str) or action not in (NO_ARG_ACTIONS | OBJECT_ACTIONS | {"PutObject"}):
        return True, "unsupported_task", f"task action {action!r} is not supported"
    if observation_errors:
        failed = ", ".join(f"{robot_id}: {reason}" for robot_id, reason in sorted(observation_errors.items()))
        return True, "observation_failed", f"could not verify every robot observation ({failed})"
    if action == "PutObject":
        holders = agent_ids_holding_object(agent_observations, object_type)
        if not holders:
            return True, "missing_required_state", f"no known robot is holding {object_type!r}"
    normalized_type = normalize_type(object_type)
    visible_robot_ids: list[int] = []
    if normalized_type:
        for observation in agent_observations:
            if any(
                isinstance(item, dict)
                and bool(item.get("visible"))
                and normalize_type(object_type_of(item)) == normalized_type
                for item in observation.get("objects", [])
            ):
                robot_id = observation.get("robot_id")
                if isinstance(robot_id, int):
                    visible_robot_ids.append(robot_id)
    if visible_robot_ids:
        detail = "; ".join(rejected_reasons) or agent_reason
        return (
            True,
            "object_not_actionable",
            f"{object_type!r} is visible to robot(s) {visible_robot_ids}, but no robot passes the action constraints: {detail}",
        )
    return (
        True,
        "target_not_visible",
        f"{object_type!r} is not visible to any successfully observed robot",
    )


def route_with_relay_agent(
    args: argparse.Namespace,
    *,
    task_id: str,
    base_url: str,
    probe: dict[str, Any],
    semantic_plan: dict[str, Any],
    task_intent: dict[str, Any] | None,
    agent_observations: list[dict[str, Any]],
    object_visibility_map: dict[str, Any],
    task: str | None = None,
    simulated_held_by_agent_id: dict[str, str | None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], list[int], list[int], str]:
    known_robot_ids = known_robot_ids_for_run(args, probe, agent_observations)
    known_robot_ids = merge_robot_ids(known_robot_ids, [args.primary_robot_id])
    robot_discovery_source = robot_discovery_source_for_run(args, probe, agent_observations)
    if robot_discovery_source != "execute_actions_state":
        known_robot_ids, global_source = discover_global_robot_ids_if_needed(args, base_url, known_robot_ids)
        if global_source is not None:
            robot_discovery_source = global_source
    queried_robot_ids = known_robot_ids_from_observations(agent_observations)
    observation_errors: dict[int, str] = {}

    def observe_for_agent(robot_id: int) -> dict[str, Any]:
        nonlocal object_visibility_map
        if robot_id not in queried_robot_ids:
            queried_robot_ids.append(robot_id)
        try:
            response = execute_actions_probe_scene(
                args.execute_actions_url,
                task_id,
                args.send_timeout,
                robot_id=robot_id,
            )
            observations = extract_agent_observations(response, primary_robot_id=robot_id)
            if not observations:
                raise RuntimeError(f"robot {robot_id} observation did not include objects")
            observation = next(
                (item for item in observations if item.get("robot_id") == robot_id),
                observations[0],
            )
            if observation.get("robot_id") != robot_id:
                raise RuntimeError(
                    f"requested robot {robot_id}, but observation identified robot {observation.get('robot_id')}"
                )
            observation["is_primary"] = robot_id == args.primary_robot_id
            append_observation_if_new(agent_observations, observation)
            save_agent_observation_images(agent_observations, args.output_dir.expanduser().resolve(), task_id)
            object_visibility_map = build_object_visibility_map(agent_observations)
            return relay_agent_observation_summary(observation)
        except Exception as exc:
            observation_errors[robot_id] = str(exc)
            raise

    def validate_executor_for_agent(robot_id: int) -> tuple[bool, str]:
        return validate_relay_agent_executor(
            robot_id,
            semantic_plan,
            task_intent,
            agent_observations,
            args.primary_robot_id,
            simulated_held_by_agent_id,
        )

    def validate_failure_for_agent(code: str, reason: str) -> tuple[bool, str, str]:
        return validate_relay_agent_failure(
            code,
            reason,
            semantic_plan,
            task_intent,
            agent_observations,
            args.primary_robot_id,
            observation_errors,
            simulated_held_by_agent_id,
        )

    def inspect_global_scene_for_agent() -> dict[str, Any]:
        return relay_global_scene_summary(
            task or args.task,
            semantic_plan,
            task_intent,
            object_visibility_map,
            agent_observations,
            known_robot_ids,
            args.primary_robot_id,
            observation_errors,
        )

    def evaluate_candidates_for_agent() -> dict[str, Any]:
        return evaluate_relay_executor_candidates(
            task or args.task,
            semantic_plan,
            task_intent,
            object_visibility_map,
            agent_observations,
            known_robot_ids,
            args.primary_robot_id,
            simulated_held_by_agent_id,
        )

    primary = primary_agent_observation(agent_observations)
    primary_inability_reason = primary_inability_reason_for_step(
        semantic_plan,
        task_intent,
        agent_observations,
        args.primary_robot_id,
        simulated_held_by_agent_id,
    )
    print_relay_handoff_explanation(primary_inability_reason=primary_inability_reason)

    # Relay decisions use one complete evidence snapshot. The primary probe is
    # already current; collect every missing peer before asking the model to
    # select an executor or report a failure.
    observed_before_relay = set(known_robot_ids_from_observations(agent_observations))
    print(
        f"[relay] evidence precollection start: known={known_robot_ids}, "
        f"already_observed={sorted(observed_before_relay)}",
        file=sys.stderr,
    )
    for robot_id in known_robot_ids:
        if robot_id in observed_before_relay:
            continue
        try:
            observe_for_agent(robot_id)
            observed_before_relay.add(robot_id)
        except Exception as exc:
            print(f"[relay] failed to precollect robot {robot_id}: {exc}", file=sys.stderr)

    object_visibility_map = build_object_visibility_map(agent_observations)
    print(
        f"[relay] evidence precollection complete: "
        f"observed={known_robot_ids_from_observations(agent_observations)}, "
        f"errors={sorted(observation_errors)}",
        file=sys.stderr,
    )
    initial_summaries = [relay_agent_observation_summary(item) for item in agent_observations]
    initial_global_scene_summary = inspect_global_scene_for_agent()
    initial_candidate_evaluation = evaluate_candidates_for_agent()
    relay_result = run_relay_agent(
        qwen_backend_for_args(args),
        task=task or args.task,
        task_intent=task_intent or {},
        known_robot_ids=known_robot_ids,
        primary_robot_id=args.primary_robot_id,
        initial_summaries=initial_summaries,
        global_scene_summary=initial_global_scene_summary,
        initial_candidate_evaluation=initial_candidate_evaluation,
        initial_observation_errors=observation_errors,
        evidence_precollected=True,
        observe_robot=observe_for_agent,
        inspect_global_scene=inspect_global_scene_for_agent,
        evaluate_executor_candidates=evaluate_candidates_for_agent,
        validate_executor=validate_executor_for_agent,
        validate_failure=validate_failure_for_agent,
        config=RelayAgentConfig(max_turns=args.relay_agent_max_turns),
    )
    relay_result["known_robot_ids"] = known_robot_ids
    relay_result["queried_robot_ids"] = queried_robot_ids
    relay_result["robot_discovery_source"] = robot_discovery_source
    relay_result["primary_robot_id"] = args.primary_robot_id
    relay_result["primary_agent_id"] = primary.get("agent_id")
    relay_result["primary_inability_reason"] = primary_inability_reason
    requested_type = relay_routing_step(semantic_plan, task_intent).get("objectType")
    relay_result["requested_object_type"] = requested_type
    if not relay_result.get("candidate_scores"):
        candidate_evaluation = evaluate_candidates_for_agent()
        relay_result["candidate_evaluation"] = candidate_evaluation
        relay_result["candidate_executor_robot_ids"] = candidate_evaluation.get("candidate_executor_robot_ids", [])
        relay_result["candidate_scores"] = candidate_evaluation.get("candidate_scores", [])
        relay_result["selection_policy"] = candidate_evaluation.get(
            "selection_policy",
            relay_result.get("selection_policy", "llm_tool_calling_with_hard_validation"),
        )
    relay_result["candidate_explanation_summary"] = relay_candidate_explanation_summary(relay_result)
    relay_result["coordination_explanation"] = relay_coordination_explanation(relay_result)
    print_relay_handoff_explanation(relay_result=relay_result)
    executor_robot_id = relay_result.get("executor_robot_id")
    if isinstance(executor_robot_id, int):
        executor_observation = agent_observation_by_robot_id(agent_observations, executor_robot_id)
        relay_result["executor_agent_id"] = executor_observation.get("agent_id")
    return (
        relay_result,
        agent_observations,
        object_visibility_map,
        known_robot_ids,
        queried_robot_ids,
        robot_discovery_source,
    )



def semantic_plan_primary_action(semantic_plan: dict[str, Any]) -> str | None:
    return first_executable_plan_action(semantic_plan)


def validate_put_object_goal_consistency(
    semantic_plan: dict[str, Any],
    executor_observation: dict[str, Any] | None,
    simulated_held_object_type: str | None = None,
) -> None:
    plan_steps = semantic_plan.get("plan")
    if not isinstance(plan_steps, list):
        raise ValueError("semantic plan must contain a plan list")
    put_step = None
    for step in plan_steps:
        if isinstance(step, dict) and step.get("action") == "PutObject":
            put_step = step
            break
    if put_step is None:
        return

    object_type = put_step.get("objectType")
    target_type = put_step.get("targetType")
    robot_label = _robot_label(executor_observation)
    held_type = held_object_type_from_observation(executor_observation) or simulated_held_object_type
    if not isinstance(object_type, str) or not object_type.strip():
        raise ValueError("PutObject plan is missing objectType")
    if normalize_type(held_type) != normalize_type(object_type):
        if held_type:
            raise ValueError(f"{robot_label} is holding {held_type}, not {object_type}; cannot execute PutObject")
        raise ValueError(f"{robot_label} is not holding {object_type}; cannot execute PutObject")

    target_failure = put_target_failure_for_agent(executor_observation or {}, target_type)
    if target_failure is not None:
        raise ValueError(target_failure)


def validate_executor_plan_or_failure(
    task: str,
    semantic_plan: dict[str, Any],
    executor_objects: list[dict[str, Any]],
    *,
    allow_invisible: bool,
    relay_result: dict[str, Any],
    agent_observations: list[dict[str, Any]],
    task_intent: dict[str, Any] | None = None,
    executor_observation: dict[str, Any] | None = None,
    simulated_held_object_type: str | None = None,
) -> dict[str, Any] | None:
    try:
        multi_step = has_multi_step_intent(task_intent)
        if task_intent is not None:
            validate_task_intent_consistency(task_intent, semantic_plan)
        if executor_observation is None:
            executor_observation = {"objects": executor_objects}
        plan_actions = semantic_plan_actions(semantic_plan)
        requested_action = task_intent.get("requestedAction") if isinstance(task_intent, dict) else None
        is_put_goal = "PutObject" in plan_actions and requested_action == "PutObject"
        if not multi_step:
            if is_put_goal:
                validate_put_object_goal_consistency(
                    semantic_plan,
                    executor_observation,
                    simulated_held_object_type=simulated_held_object_type,
                )
            else:
                validate_goal_consistency(task, semantic_plan, executor_objects)
            validate_action_intent_consistency(task, semantic_plan)
        validate_action_affordances(semantic_plan, executor_objects, allow_invisible=allow_invisible)
        validate_action_state_preconditions(
            semantic_plan,
            executor_observation,
            allow_invisible=allow_invisible,
            simulated_held_object_type=simulated_held_object_type,
        )
    except ValueError as exc:
        return relay_failure_result(
            str(exc),
            requested_object_type=relay_result.get("requested_object_type"),
            primary_agent_id=str(relay_result.get("primary_agent_id")),
            agent_observations=agent_observations,
        )
    return None


def extract_requested_action(task: str) -> str | None:
    normalized_task = " ".join(re.findall(r"[a-z0-9]+", task.lower()))
    for phrases, action_name in ACTION_INTENT_PATTERNS:
        for phrase in phrases:
            normalized_phrase = " ".join(re.findall(r"[a-z0-9]+", phrase.lower()))
            if re.search(rf"\b{re.escape(normalized_phrase)}\b", normalized_task):
                return action_name
    return None


def semantic_plan_actions(semantic_plan: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    plan_steps = semantic_plan.get("plan")
    if not isinstance(plan_steps, list):
        return actions
    for step in plan_steps:
        if not isinstance(step, dict):
            continue
        action_name = step.get("action")
        if isinstance(action_name, str) and action_name not in {"Pass", "Done"}:
            actions.append(action_name)
    return actions


def semantic_plan_object_types(semantic_plan: dict[str, Any]) -> list[str]:
    object_types: list[str] = []
    plan_steps = semantic_plan.get("plan")
    if not isinstance(plan_steps, list):
        return object_types
    for step in plan_steps:
        if not isinstance(step, dict):
            continue
        action_name = step.get("action")
        if action_name not in OBJECT_ACTIONS and action_name != "PutObject":
            continue
        value = step.get("objectType")
        if isinstance(value, str) and value.strip():
            object_types.append(value)
    return object_types


def validate_goal_consistency(task: str, semantic_plan: dict[str, Any], objects: list[dict[str, Any]]) -> None:
    target_object_type = semantic_plan.get("targetObjectType")
    if isinstance(target_object_type, str) and target_object_type.strip():
        requested_type = target_object_type.strip()
    else:
        all_types = object_categories(objects, visible_only=False)
        requested_type = extract_requested_object_type(task, all_types)
    if requested_type is None:
        return

    normalized_requested = normalize_type(requested_type)
    visible_types = object_categories(objects, visible_only=True)
    visible_lookup = _object_type_lookup(visible_types)
    if normalized_requested not in visible_lookup:
        visible_text = ", ".join(visible_types) or "(none)"
        raise ValueError(
            f"requested object {requested_type!r} is not visible; visible categories: {visible_text}"
        )

    planned_types = semantic_plan_object_types(semantic_plan)
    if not planned_types:
        raise ValueError(
            f"plan does not operate on requested object {requested_type!r}; refusing to execute"
        )
    for planned_type in planned_types:
        if normalize_type(planned_type) != normalized_requested:
            raise ValueError(
                f"planned object {planned_type!r} does not match requested object {requested_type!r}; refusing to execute"
            )


def visible_object_prompt(objects: list[dict[str, Any]]) -> str:
    visible = object_categories(objects, visible_only=True)
    if not visible:
        return (
            "\n\nSimulator metadata reports no currently visible object categories. "
            "Return Pass or Done unless the image clearly shows a usable object."
        )
    lines = "\n".join(f"- {name}" for name in visible)
    return (
        "\n\nSimulator-visible object categories, without objectIds:\n"
        f"{lines}\n"
        "Use these categories for objectType and targetType when possible. "
        "Do not output objectIds, coordinates, pipe-delimited strings, or UNKNOWN placeholders."
    )


def build_grounded_prompt(
    task: str,
    objects: list[dict[str, Any]],
    task_intent: dict[str, Any] | None = None,
) -> str:
    return semantic_planning_prompt("image", task, task_intent=task_intent) + visible_object_prompt(objects)


def role_property(role: str) -> str | None:
    if role in {"pickupable", "openable", "receptacle"}:
        return role
    return None


def action_object_role(action_name: str) -> str | None:
    if action_name in PICKUPABLE_ACTIONS:
        return "pickupable"
    if action_name in OPENABLE_ACTIONS:
        return "openable"
    return None


def select_object(
    objects: list[dict[str, Any]],
    wanted_type: Any,
    *,
    role: str,
    allow_invisible: bool,
    action_name: str | None = None,
) -> dict[str, Any]:
    normalized = normalize_type(wanted_type)
    if not normalized:
        raise ValueError(f"missing object type for {role}")

    candidates = [
        item for item in objects if normalize_type(object_type_of(item)) == normalized
    ]
    if not candidates:
        available = ", ".join(object_categories(objects, visible_only=False))
        raise ValueError(f"no objectId found for {wanted_type!r}; available types: {available}")

    visible_candidates = [item for item in candidates if bool(item.get("visible"))]
    if visible_candidates:
        candidates = visible_candidates
    elif not allow_invisible:
        raise ValueError(
            f"{wanted_type!r} exists in scene metadata but is not visible; "
            "rerun with --allow-invisible-object-ids to use it anyway"
        )

    required_property = role_property(role)
    if required_property is not None:
        matching_candidates = [item for item in candidates if bool(item.get(required_property))]
        if not matching_candidates:
            action_text = f"; cannot execute {action_name}" if action_name else ""
            raise ValueError(f"object {wanted_type!r} is not {required_property}{action_text}")
        candidates = matching_candidates

    def score(item: dict[str, Any]) -> tuple[int, int, str]:
        return (
            int(bool(item.get("visible"))),
            int(bool(item.get("receptacle"))),
            object_id_of(item) or "",
        )

    return sorted(candidates, key=score, reverse=True)[0]


def validate_action_intent_consistency(task: str, semantic_plan: dict[str, Any]) -> None:
    requested_action = extract_requested_action(task)
    if requested_action is None:
        return
    planned_actions = semantic_plan_actions(semantic_plan)
    if requested_action in planned_actions:
        return
    planned_action = planned_actions[0] if planned_actions else "none"
    raise ValueError(
        f"planned action {planned_action!r} does not match requested action {requested_action!r}; refusing to execute"
    )


def held_objects_from_observation(observation: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(observation, dict):
        return []
    held_object = _held_object_candidate(observation.get("held_object") or observation.get("heldObject"))
    if held_object is not None:
        return [held_object]

    robot_state = observation.get("robot_state")
    if isinstance(robot_state, dict):
        for key in ("held_object", "heldObject"):
            held_object = _held_object_candidate(robot_state.get(key))
            if held_object is not None:
                return [held_object]

    return []


def held_object_source_from_observation(observation: dict[str, Any] | None) -> str | None:
    if not isinstance(observation, dict):
        return None
    if _held_object_candidate(observation.get("held_object") or observation.get("heldObject")) is not None:
        source = observation.get("held_object_source")
        return source if isinstance(source, str) and source else "held_object"
    robot_state = observation.get("robot_state")
    if isinstance(robot_state, dict):
        for key in ("held_object", "heldObject"):
            if _held_object_candidate(robot_state.get(key)) is not None:
                return f"robot_state.{key}"
    return None


def held_object_type_from_observation(observation: dict[str, Any] | None) -> str | None:
    for item in held_objects_from_observation(observation):
        object_type = object_type_of(item)
        if object_type:
            return object_type
        object_id = object_id_of(item)
        if isinstance(object_id, str) and "|" in object_id:
            prefix = object_id.split("|", 1)[0]
            if prefix:
                return prefix
    return None


def held_object_debug_from_observation(observation: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(observation, dict):
        return {"held_object_type": None, "held_object_source": None, "held_objects": []}
    robot_state = observation.get("robot_state")
    if not isinstance(robot_state, dict):
        robot_state = {}
    return {
        "agent_id": observation.get("agent_id"),
        "robot_id": observation.get("robot_id"),
        "held_object_type": held_object_type_from_observation(observation),
        "held_object_source": held_object_source_from_observation(observation),
        "held_object": observation.get("held_object") or observation.get("heldObject"),
        "held_objects": held_objects_from_observation(observation),
        "inventory": observation.get("inventory", []),
        "robot_state_held_object": robot_state.get("held_object") or robot_state.get("heldObject"),
        "robot_state_inventory": robot_state.get("inventory") or robot_state.get("inventoryObjects") or [],
        "robot_state_proxy": robot_state.get("proxy"),
    }


def _robot_label(observation: dict[str, Any] | None) -> str:
    if not isinstance(observation, dict):
        return "robot"
    robot_id = observation.get("robot_id")
    if isinstance(robot_id, int):
        return f"robot {robot_id}"
    agent_id = observation.get("agent_id")
    if isinstance(agent_id, str) and agent_id:
        return agent_id
    return "robot"


def _object_open_state(item: dict[str, Any]) -> bool | None:
    value = item.get("isOpen")
    return value if isinstance(value, bool) else None


def validate_action_state_preconditions(
    semantic_plan: dict[str, Any],
    observation: dict[str, Any] | None,
    *,
    allow_invisible: bool,
    simulated_held_object_type: str | None = None,
) -> str | None:
    plan_steps = semantic_plan.get("plan")
    if not isinstance(plan_steps, list):
        raise ValueError("semantic plan must contain a plan list")

    objects = observation.get("objects", []) if isinstance(observation, dict) else []
    if not isinstance(objects, list):
        objects = []
    robot_label = _robot_label(observation)
    current_held_type = held_object_type_from_observation(observation) or simulated_held_object_type
    open_state_by_id: dict[str, bool] = {}

    for step in plan_steps:
        if not isinstance(step, dict):
            continue
        action_name = step.get("action")
        if not isinstance(action_name, str):
            continue

        if action_name == "PickupObject":
            if current_held_type:
                raise ValueError(f"{robot_label} is already holding {current_held_type}; cannot execute PickupObject")
            object_type = step.get("objectType")
            current_held_type = object_type if isinstance(object_type, str) and object_type.strip() else None
            continue

        if action_name in {"PutObject", "DropHandObject"}:
            if not current_held_type:
                raise ValueError(f"{robot_label} is not holding any object; cannot execute {action_name}")
            object_type = step.get("objectType")
            if isinstance(object_type, str) and object_type.strip() and normalize_type(object_type) != normalize_type(current_held_type):
                raise ValueError(f"{robot_label} is holding {current_held_type}, not {object_type}; cannot execute {action_name}")
            current_held_type = None
            continue

        if action_name in {"OpenObject", "CloseObject"}:
            target = select_object(
                objects,
                step.get("objectType"),
                role="openable",
                allow_invisible=allow_invisible,
                action_name=action_name,
            )
            target_id = object_id_of(target) or object_type_of(target)
            known_state = open_state_by_id.get(target_id) if target_id else None
            if known_state is None:
                known_state = _object_open_state(target)
            target_type = object_type_of(target) or str(step.get("objectType"))
            if action_name == "OpenObject":
                if known_state is True:
                    raise ValueError(f"object {target_type!r} is already open; cannot execute OpenObject")
                if target_id:
                    open_state_by_id[target_id] = True
            else:
                if known_state is False:
                    raise ValueError(f"object {target_type!r} is already closed; cannot execute CloseObject")
                if target_id:
                    open_state_by_id[target_id] = False

    return current_held_type


def validate_action_affordances(
    semantic_plan: dict[str, Any],
    objects: list[dict[str, Any]],
    *,
    allow_invisible: bool,
) -> None:
    plan_steps = semantic_plan.get("plan")
    if not isinstance(plan_steps, list):
        raise ValueError("semantic plan must contain a plan list")
    for step in plan_steps:
        if not isinstance(step, dict):
            continue
        action_name = step.get("action")
        if not isinstance(action_name, str) or action_name in NO_ARG_ACTIONS:
            continue
        if action_name == "PutObject":
            target_type = step.get("targetType") or step.get("objectType")
            select_object(
                objects,
                target_type,
                role="receptacle",
                allow_invisible=allow_invisible,
                action_name=action_name,
            )
            continue
        role = action_object_role(action_name)
        if role is None:
            continue
        select_object(
            objects,
            step.get("objectType"),
            role=role,
            allow_invisible=allow_invisible,
            action_name=action_name,
        )


def grounded_action_for_step(
    step: dict[str, Any],
    objects: list[dict[str, Any]],
    *,
    allow_invisible: bool,
) -> dict[str, Any] | None:
    action_name = step.get("action")
    if not isinstance(action_name, str) or not action_name:
        raise ValueError("semantic plan step is missing action")

    if action_name in NO_ARG_ACTIONS:
        return {"action": action_name}

    if action_name == "PutObject":
        target_type = step.get("targetType") or step.get("objectType")
        target = select_object(
            objects,
            target_type,
            role="receptacle",
            allow_invisible=allow_invisible,
            action_name=action_name,
        )
        return {
            "action": "PutObject",
            "objectId": object_id_of(target),
            "forceAction": True,
        }

    if action_name in OBJECT_ACTIONS:
        role = action_object_role(action_name) or "openable"
        target = select_object(
            objects,
            step.get("objectType"),
            role=role,
            allow_invisible=allow_invisible,
            action_name=action_name,
        )
        action = {
            "action": action_name,
            "objectId": object_id_of(target),
        }
        if action_name in FORCE_ACTIONS:
            action["forceAction"] = True
        return action

    raise ValueError(f"unsupported grounded semantic action: {action_name}")


def ground_semantic_plan(
    semantic_plan: dict[str, Any],
    objects: list[dict[str, Any]],
    *,
    allow_invisible: bool,
    max_actions: int,
    include_done: bool = True,
) -> list[dict[str, Any]]:
    plan_steps = semantic_plan.get("plan")
    if not isinstance(plan_steps, list):
        raise ValueError("semantic plan must contain a plan list")

    actions = [
        action
        for action in (
            grounded_action_for_step(step, objects, allow_invisible=allow_invisible)
            for step in plan_steps
            if isinstance(step, dict)
        )
        if action is not None
    ]
    if not actions:
        actions = [{"action": "Pass"}]
    if include_done and actions[-1].get("action") != "Done":
        actions.append({"action": "Done"})

    executable_count = len([action for action in actions if action.get("action") != "Done"])
    if max_actions < 0:
        raise ValueError("--max-actions must be 0 or greater")
    if max_actions and executable_count > max_actions:
        raise ValueError(f"grounded {executable_count} actions, exceeding --max-actions {max_actions}")
    return actions


def _truncated_text(value: Any, max_chars: int = 500) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def save_text_artifact(output_dir: Path, task_id: str, suffix: str, content: str) -> str:
    output_path = output_dir.expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_path = output_path / f"{task_id}_{suffix}"
    artifact_path.write_text(content, encoding="utf-8")
    return str(artifact_path)


def save_qwen_raw_output(output_dir: Path, task_id: str, raw_output: str) -> str:
    return save_text_artifact(output_dir, task_id, "qwen_raw.txt", raw_output)


def parse_execute_response_text(response_text: str) -> Any:
    stripped = response_text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return response_text
    return response_text


def _latest_result_for_robot(execute_response: Any, robot_id: int | None) -> dict[str, Any] | None:
    if not isinstance(execute_response, dict):
        return None
    results = execute_response.get("results")
    if not isinstance(results, list):
        return None
    for result in reversed(results):
        if not isinstance(result, dict):
            continue
        result_robot_id = _robot_id_from_value(result.get("robot_id"))
        if robot_id is None or result_robot_id == robot_id:
            return result
    return None


def merge_execute_result_into_observation(
    observation: dict[str, Any],
    execute_response: Any,
) -> None:
    robot_id = _robot_id_from_value(observation.get("robot_id"))
    result = _latest_result_for_robot(execute_response, robot_id)
    if result is None:
        return

    robot_state = observation.get("robot_state")
    if not isinstance(robot_state, dict):
        robot_state = {}
        observation["robot_state"] = robot_state

    result_robot = result.get("robot")
    if isinstance(result_robot, dict):
        robot_state.update(result_robot)
    else:
        for key in ("robot_id", "robot_name", "agent"):
            if key in result:
                robot_state[key] = result[key]

    held_object, held_object_source = _held_object_from_observation_source(result, robot_state, robot_id)
    observation["held_object"] = held_object
    observation["held_object_source"] = held_object_source
    observation["inventory"] = _inventory_from_observation_source(result, robot_state, robot_id)


def execute_response_failure_reason(response: Any) -> str | None:
    if not isinstance(response, dict):
        return None
    status = response.get("status")
    if isinstance(status, str) and status.lower() not in {"success", "ok"}:
        reason = response.get("error") or response.get("errorMessage") or response.get("message")
        return str(reason) if reason else f"execute_actions returned status {status!r}"

    results = response.get("results")
    if isinstance(results, list):
        for index, item in enumerate(results):
            if not isinstance(item, dict):
                continue
            success_value = None
            for key in ("success", "last_success", "lastActionSuccess"):
                if key in item:
                    success_value = item.get(key)
                    break
            if success_value is False:
                action = item.get("action") or item.get("last_action") or item.get("lastAction")
                error = item.get("error") or item.get("last_error") or item.get("errorMessage") or item.get("message")
                action_text = f" for action {action!r}" if action else ""
                error_text = f": {error}" if error else ""
                return f"execute_actions result {index} failed{action_text}{error_text}"
    return None


def summarize_execute_response(response: Any, action_count: int) -> dict[str, Any]:
    summary: dict[str, Any] = {"action_count": action_count}
    if not isinstance(response, dict):
        summary["response_type"] = type(response).__name__
        summary["text_preview"] = _truncated_text(response)
        return summary

    summary["response_type"] = "dict"
    state = response.get("state")
    if isinstance(state, dict):
        scene_name = state.get("sceneName")
        if scene_name is not None:
            summary["sceneName"] = scene_name
        objects = state.get("objects")
        if isinstance(objects, list):
            summary["object_count"] = len(objects)

    for key in ("success", "status", "lastActionSuccess", "error", "errorMessage", "message"):
        if key in response:
            value = response[key]
            if isinstance(value, (str, int, float, bool)) or value is None:
                summary[key] = _truncated_text(value) if isinstance(value, str) else value

    summary["top_level_keys"] = sorted(str(key) for key in response.keys())
    return summary


def add_execute_response_result(
    result: dict[str, Any],
    response_text: str,
    *,
    include_response: bool,
    save_response: bool,
    output_dir: Path,
    task_id: str,
    action_count: int,
) -> None:
    execute_response = parse_execute_response_text(response_text)
    result["execute_response_summary"] = summarize_execute_response(execute_response, action_count)
    if include_response:
        result["execute_response"] = execute_response
    if save_response:
        result["execute_response_path"] = save_text_artifact(
            output_dir,
            task_id,
            "execute_response.json",
            response_text,
        )


def resolved_device(args: argparse.Namespace) -> str:
    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def qwen_backend_for_args(args: argparse.Namespace) -> Any:
    qwen = getattr(args, "_qwen_backend", None)
    if qwen is None:
        from demo.qwen35_backend import Qwen35Backend, Qwen35Config

        qwen = Qwen35Backend(
            Qwen35Config(
                model_name=args.qwen_model,
                device=resolved_device(args),
                device_map=args.qwen_device_map,
                torch_dtype=args.qwen_dtype,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        )
        setattr(args, "_qwen_backend", qwen)
    return qwen


def generate_semantic_plan(args: argparse.Namespace, image_path: Path, objects: list[dict[str, Any]], task_id: str) -> tuple[str, dict[str, Any], str | None]:
    qwen = qwen_backend_for_args(args)
    prompt = build_grounded_prompt(args.task, objects, getattr(args, "_task_intent", None))
    output = qwen.generate(str(image_path), "image", prompt).strip()
    if args.print_raw_output:
        print("Raw model output:", file=sys.stderr)
        print(output, file=sys.stderr)
    raw_output_path = None
    if args.save_raw_output:
        raw_output_path = save_qwen_raw_output(args.output_dir, task_id, output)
        print(f"Saved raw Qwen output: {raw_output_path}", file=sys.stderr)
    semantic_plan = parse_semantic_planning_output(output)
    return output, semantic_plan, raw_output_path


def task_text_for_intent_step(step: dict[str, Any]) -> str:
    action = step.get("action")
    obj = step.get("objectType")
    target = step.get("targetType")
    object_text = str(obj).lower() if isinstance(obj, str) else ""
    target_text = str(target).lower() if isinstance(target, str) else ""
    if action == "PickupObject":
        return f"pick up the {object_text}."
    if action == "OpenObject":
        return f"open the {object_text}."
    if action == "CloseObject":
        return f"close the {object_text}."
    if action == "PutObject":
        return f"put the {object_text} on the {target_text}." if target_text else f"put the {object_text}."
    if action == "RotateRight":
        return "turn right."
    if action == "RotateLeft":
        return "turn left."
    if action == "MoveRight":
        return "move right."
    if action == "MoveLeft":
        return "move left."
    if action == "MoveAhead":
        return "move ahead."
    if action == "MoveBack":
        return "move back."
    if action == "LookUp":
        return "look up."
    if action == "LookDown":
        return "look down."
    if isinstance(action, str) and object_text:
        return f"{action} the {object_text}."
    if isinstance(action, str):
        return f"{action}."
    return "perform the next task step."


def task_intent_for_step(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "requestedAction": step.get("action"),
        "requestedObjectType": step.get("objectType"),
        "intentSteps": [step],
    }


def semantic_plan_for_intent_step(step: dict[str, Any], task: str) -> dict[str, Any]:
    return {
        "task": task,
        "targetObjectType": step.get("objectType"),
        "needsGrounding": True,
        "observations": [],
        "plan": [
            {
                "action": step.get("action"),
                "objectType": step.get("objectType"),
                "targetType": step.get("targetType"),
            }
        ],
    }


def generate_step_semantic_plan(
    args: argparse.Namespace,
    image_path: Path,
    objects: list[dict[str, Any]],
    task_id: str,
    step_task: str,
    step_intent: dict[str, Any],
) -> tuple[str, dict[str, Any], str | None]:
    original_task = args.task
    original_intent = getattr(args, "_task_intent", None)
    args.task = step_task
    setattr(args, "_task_intent", step_intent)
    try:
        raw_output, semantic_plan, raw_output_path = generate_semantic_plan(
            args, image_path, objects, task_id
        )
        repair_semantic_placeholders_from_step_intent(semantic_plan, step_intent)
        return raw_output, semantic_plan, raw_output_path
    finally:
        args.task = original_task
        setattr(args, "_task_intent", original_intent)


SEMANTIC_OBJECT_PLACEHOLDERS = {
    "objecttype",
    "requestedobject",
    "requestedobjecttype",
}
SEMANTIC_TARGET_PLACEHOLDERS = {"targettype", "requestedtarget", "requestedtargettype"}


def repair_semantic_placeholders_from_step_intent(
    semantic_plan: dict[str, Any],
    step_intent: dict[str, Any],
) -> list[str]:
    steps = intent_steps(step_intent)
    if not steps:
        return []
    expected_step = steps[0]
    expected_object = expected_step.get("objectType")
    expected_target = expected_step.get("targetType")
    warnings = semantic_plan.setdefault("semanticNormalizationWarnings", [])
    if not isinstance(warnings, list):
        warnings = []
        semantic_plan["semanticNormalizationWarnings"] = warnings

    def repair_field(container: dict[str, Any], field: str, expected: Any, placeholders: set[str]) -> None:
        value = container.get(field)
        if not isinstance(value, str) or normalize_type(value) not in placeholders:
            return
        container[field] = expected
        warnings.append(f"{field} repaired from placeholder {value!r} to {expected!r} using step intent")

    repair_field(
        semantic_plan,
        "targetObjectType",
        expected_object,
        SEMANTIC_OBJECT_PLACEHOLDERS,
    )
    plan_steps = semantic_plan.get("plan")
    if isinstance(plan_steps, list):
        for plan_step in plan_steps:
            if not isinstance(plan_step, dict):
                continue
            repair_field(plan_step, "objectType", expected_object, SEMANTIC_OBJECT_PLACEHOLDERS)
            repair_field(plan_step, "targetType", expected_target, SEMANTIC_TARGET_PLACEHOLDERS)
    if not warnings:
        semantic_plan.pop("semanticNormalizationWarnings", None)
    return warnings


def payload_for_actions(
    args: argparse.Namespace,
    task_id: str,
    task: str,
    plan: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    executor_observation: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"task_id": task_id, "task": task, "plan": plan, "stop_on_failure": False, "actions": actions}
    if args.relay_mode and executor_observation is not None:
        executor_robot_id = executor_observation.get("robot_id")
        executor_agent_id = executor_observation.get("agent_id")
        if args.executor_agent_id_field == "robot_id" and isinstance(executor_robot_id, int):
            payload[args.executor_agent_id_field] = executor_robot_id
        elif isinstance(executor_agent_id, str):
            payload[args.executor_agent_id_field] = executor_agent_id
        payload["render_image"] = True
    return payload


def closed_loop_failure_result(
    reason: str,
    *,
    failed_step_index: int,
    failed_step: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "status": "needs_upstream_planning",
        "failed_step_index": failed_step_index,
        "failed_step": failed_step,
        "reason": reason,
    }


def run_closed_loop_replan(
    args: argparse.Namespace,
    *,
    task_id: str,
    base_url: str,
    probe: dict[str, Any],
    agent_observations: list[dict[str, Any]],
    primary_observation: dict[str, Any],
    image_path: Path,
    object_visibility_map: dict[str, Any],
    known_robot_ids: list[int],
    queried_robot_ids: list[int],
    robot_discovery_source: str,
    task_intent: dict[str, Any],
    task_intent_tool_call: dict[str, Any],
    task_intent_tool_call_validation: dict[str, Any],
) -> int:
    intent_expansion_warnings: list[str] = []
    steps = intent_steps(task_intent)
    if not steps:
        steps = [{"order": 1, "action": task_intent.get("requestedAction"), "objectType": task_intent.get("requestedObjectType"), "targetType": None}]
    simulated_held_by_agent_id: dict[str, str | None] = {
        str(observation.get("agent_id")): held_object_type_from_observation(observation)
        for observation in agent_observations
    }
    if args.relay_mode:
        ownership_object_types: list[str] = []
        for step in steps:
            if step.get("action") != "PutObject":
                continue
            object_type = step.get("objectType")
            if not isinstance(object_type, str) or not object_type.strip():
                continue
            step_task = task_text_for_intent_step(step)
            step_intent = task_intent_for_step(step)
            step_plan_hint = semantic_plan_for_intent_step(step, step_task)
            primary_relay_result = primary_fast_path_relay_result(
                step_plan_hint,
                step_intent,
                agent_observations,
                args.primary_robot_id,
                simulated_held_by_agent_id,
            )
            if primary_relay_result is not None and primary_relay_result.get("executor_robot_id") == args.primary_robot_id:
                continue
            if normalize_type(object_type) not in {normalize_type(item) for item in ownership_object_types}:
                ownership_object_types.append(object_type)
        (
            agent_observations,
            object_visibility_map,
            known_robot_ids,
            queried_robot_ids,
            robot_discovery_source,
        ) = query_relay_observations_for_ownership(
            args,
            task_id,
            base_url,
            probe,
            agent_observations,
            object_visibility_map,
            known_robot_ids,
            queried_robot_ids,
            robot_discovery_source,
            ownership_object_types,
        )
        intent_expansion_warnings = expand_put_object_intent_preconditions(
            task_intent,
            primary_observation,
            held_object_types_from_observations(agent_observations),
        )
        steps = intent_steps(task_intent)
        if not steps:
            steps = [{"order": 1, "action": task_intent.get("requestedAction"), "objectType": task_intent.get("requestedObjectType"), "targetType": None}]
        simulated_held_by_agent_id = {
            str(observation.get("agent_id")): held_object_type_from_observation(observation)
            for observation in agent_observations
        }
    if len(steps) > args.max_replan_steps:
        result = {
            "task_id": task_id,
            "image_path": str(image_path),
            "sceneName": primary_observation.get("sceneName"),
            "primary_agent_id": primary_observation.get("agent_id"),
            "primary_robot_id": primary_observation.get("robot_id"),
            "known_robot_ids": known_robot_ids,
            "queried_robot_ids": queried_robot_ids,
            "robot_discovery_source": robot_discovery_source,
            "task_intent": task_intent,
            "intent_steps": steps,
            "task_intent_tool_call": task_intent_tool_call,
            "task_intent_tool_call_validation": task_intent_tool_call_validation,
            "task_intent_source": task_intent_source_for_args(args),
            "closed_loop_result": closed_loop_failure_result(
                f"intent has {len(steps)} steps, exceeding --max-replan-steps {args.max_replan_steps}",
                failed_step_index=args.max_replan_steps + 1,
                failed_step=None,
            ),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    result: dict[str, Any] = {
        "task_id": task_id,
        "image_path": str(image_path),
        "sceneName": primary_observation.get("sceneName"),
        "primary_agent_id": primary_observation.get("agent_id"),
        "primary_robot_id": primary_observation.get("robot_id"),
        "known_robot_ids": known_robot_ids,
        "queried_robot_ids": queried_robot_ids,
        "robot_discovery_source": robot_discovery_source,
        "visible_object_types": object_categories(primary_observation.get("objects", []), visible_only=True),
        "agent_observations_summary": agent_observations_summary(agent_observations),
        "object_visibility_summary": object_visibility_summary(object_visibility_map),
        "task_intent": task_intent,
        "intent_steps": steps,
        "task_intent_tool_call": task_intent_tool_call,
        "task_intent_tool_call_validation": task_intent_tool_call_validation,
        "task_intent_source": task_intent_source_for_args(args),
        "closed_loop_trace": [],
        "step_payloads": [],
        "step_execute_response_summaries": [],
    }

    last_executor_observation: dict[str, Any] | None = primary_observation
    for step_index, step in enumerate(steps, start=1):
        step_task = task_text_for_intent_step(step)
        step_intent = task_intent_for_step(step)
        step_plan_hint = semantic_plan_for_intent_step(step, step_task)
        already_satisfied_observation = already_satisfied_observation_for_step(
            step,
            agent_observations,
            args.primary_robot_id,
        )
        if already_satisfied_observation is not None:
            executor_agent_id = str(already_satisfied_observation.get("agent_id"))
            trace_entry = {
                "step_index": step_index,
                "intent_step": step,
                "executor_agent_id": executor_agent_id,
                "executor_robot_id": already_satisfied_observation.get("robot_id"),
                "relay_result": {
                    "status": "already_satisfied",
                    "strategy": "pre_relay_state",
                    "executor_robot_id": already_satisfied_observation.get("robot_id"),
                    "executor_agent_id": executor_agent_id,
                    "reason": step_already_satisfied_reason(step, already_satisfied_observation),
                    "known_robot_ids": known_robot_ids,
                },
                "actions": [],
                "skipped_reason": step_already_satisfied_reason(step, already_satisfied_observation),
                "held_object_debug": held_object_debug_from_observation(already_satisfied_observation),
            }
            result["closed_loop_trace"].append(trace_entry)
            if isinstance(executor_agent_id, str):
                simulated_held_by_agent_id[executor_agent_id] = held_object_type_from_observation(already_satisfied_observation)
            continue
        coordination_result = coordination_result_for_plan(
            step_task,
            step_plan_hint,
            object_visibility_map,
            step_intent,
            relay_mode=args.relay_mode,
        )
        agent_step_relay_result: dict[str, Any] | None = None
        if args.relay_mode:
            if args.relay_strategy == "agent":
                agent_step_relay_result = primary_fast_path_relay_result(
                    step_plan_hint,
                    step_intent,
                    agent_observations,
                    args.primary_robot_id,
                    simulated_held_by_agent_id,
                )
                if agent_step_relay_result is None:
                    (
                        agent_step_relay_result,
                        agent_observations,
                        object_visibility_map,
                        known_robot_ids,
                        queried_robot_ids,
                        robot_discovery_source,
                    ) = route_with_relay_agent(
                        args,
                        task_id=task_id,
                        base_url=base_url,
                        probe=probe,
                        semantic_plan=step_plan_hint,
                        task_intent=step_intent,
                        agent_observations=agent_observations,
                        object_visibility_map=object_visibility_map,
                        task=step_task,
                        simulated_held_by_agent_id=simulated_held_by_agent_id,
                    )
            else:
                (
                    agent_observations,
                    object_visibility_map,
                    coordination_result,
                    known_robot_ids,
                    queried_robot_ids,
                    robot_discovery_source,
                ) = query_relay_observations_if_needed(
                    args,
                    task_id,
                    base_url,
                    probe,
                    step_plan_hint,
                    agent_observations,
                    object_visibility_map,
                    coordination_result,
                    step_intent,
                )
        elif coordination_result.get("status") == "target_visible_by_peer":
            failure = closed_loop_failure_result(coordination_result.get("message", "target visible only to peer"), failed_step_index=step_index, failed_step=step)
            result["closed_loop_result"] = failure
            print(failure["reason"], file=sys.stderr)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

        relay_result = agent_step_relay_result
        if relay_result is None:
            relay_result = relay_result_for_held_put_step(
                step,
                object_visibility_map,
                agent_observations,
                simulated_held_by_agent_id,
            )
        if relay_result is None:
            relay_result = choose_relay_executor(
                step_task,
                step_plan_hint,
                object_visibility_map,
                agent_observations,
                step_intent,
            )
            relay_result["strategy"] = "rules"
        relay_result["known_robot_ids"] = known_robot_ids
        if relay_result.get("status") == "needs_upstream_planning":
            if args.dry_run and step_index > 1:
                relay_result = {
                    "status": "dry_run_requires_execution_feedback",
                    "reason": "later closed-loop step requires execution feedback before selecting an executor",
                    "failed_step_index": step_index,
                    "failed_step": step,
                    "known_robot_ids": known_robot_ids,
                }
            relay_explanation = relay_explanation_for_trace(relay_result)
            if relay_explanation:
                result["closed_loop_trace"].append(
                    {
                        "step_index": step_index,
                        "intent_step": step,
                        "relay_result": relay_result,
                        "relay_explanation": relay_explanation,
                        "actions": [],
                    }
                )
            result["closed_loop_result"] = relay_result
            result["known_robot_ids"] = known_robot_ids
            result["queried_robot_ids"] = queried_robot_ids
            result["final_object_visibility_summary"] = object_visibility_summary(object_visibility_map)
            print(relay_result.get("reason"), file=sys.stderr)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

        executor_agent_id = str(relay_result["executor_agent_id"])
        executor_observation = agent_observation_by_id(agent_observations, executor_agent_id)
        if step_already_satisfied(step, executor_observation):
            trace_entry = {
                "step_index": step_index,
                "intent_step": step,
                "executor_agent_id": executor_agent_id,
                "executor_robot_id": executor_observation.get("robot_id"),
                "relay_result": {**relay_result, "status": "already_satisfied"},
                "actions": [],
                "skipped_reason": step_already_satisfied_reason(step, executor_observation),
                "held_object_debug": held_object_debug_from_observation(executor_observation),
            }
            result["closed_loop_trace"].append(trace_entry)
            simulated_held_by_agent_id[executor_agent_id] = held_object_type_from_observation(executor_observation)
            continue

        executor_objects = executor_observation["objects"]
        executor_image_path_value = executor_observation.get("image_path")
        if not isinstance(executor_image_path_value, str):
            failure = closed_loop_failure_result(
                f"executor agent {executor_agent_id!r} did not include image_base64; cannot replan safely",
                failed_step_index=step_index,
                failed_step=step,
            )
            result["closed_loop_result"] = failure
            print(failure["reason"], file=sys.stderr)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

        raw_output, step_semantic_plan, raw_output_path = generate_step_semantic_plan(
            args,
            Path(executor_image_path_value),
            executor_objects,
            task_id,
            step_task,
            step_intent,
        )
        validation_failure = validate_executor_plan_or_failure(
            step_task,
            step_semantic_plan,
            executor_objects,
            allow_invisible=args.allow_invisible_object_ids,
            relay_result=relay_result,
            agent_observations=agent_observations,
            task_intent=step_intent,
            executor_observation=executor_observation,
            simulated_held_object_type=simulated_held_by_agent_id.get(executor_agent_id) if args.dry_run else None,
        )
        if validation_failure is not None:
            validation_failure["failed_step_index"] = step_index
            validation_failure["failed_step"] = step
            validation_failure["known_robot_ids"] = known_robot_ids
            for key in ("primary_inability_reason", "coordination_explanation", "candidate_explanation_summary"):
                if isinstance(relay_result.get(key), str):
                    validation_failure[key] = relay_result[key]
            validation_failure["executor_held_object_debug"] = held_object_debug_from_observation(executor_observation)
            result["closed_loop_result"] = validation_failure
            print(validation_failure.get("reason"), file=sys.stderr)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

        actions = ground_semantic_plan(
            step_semantic_plan,
            executor_objects,
            allow_invisible=args.allow_invisible_object_ids,
            max_actions=args.max_actions,
            include_done=False,
        )
        payload = payload_for_actions(
            args,
            task_id,
            step_semantic_plan.get("task", step_task),
            step_semantic_plan.get("plan", []),
            actions,
            executor_observation,
        )
        result["step_payloads"].append(payload)
        trace_entry = {
            "step_index": step_index,
            "intent_step": step,
            "executor_agent_id": executor_agent_id,
            "executor_robot_id": executor_observation.get("robot_id"),
            "semantic_plan": step_semantic_plan,
            "relay_result": {**relay_result, "status": "executor_ready"},
            "actions": actions,
        }
        relay_explanation = relay_explanation_for_trace(relay_result)
        if relay_explanation:
            trace_entry["relay_explanation"] = relay_explanation
        semantic_warnings = semantic_normalization_warnings(step_semantic_plan)
        if semantic_warnings:
            trace_entry["semantic_normalization_warnings"] = semantic_warnings
        if raw_output_path is not None:
            trace_entry["raw_output_path"] = raw_output_path
        result["closed_loop_trace"].append(trace_entry)

        last_executor_observation = executor_observation
        if args.dry_run:
            simulated_held_by_agent_id[executor_agent_id] = validate_action_state_preconditions(
                step_semantic_plan,
                executor_observation,
                allow_invisible=args.allow_invisible_object_ids,
                simulated_held_object_type=simulated_held_by_agent_id.get(executor_agent_id),
            )
            print(f"Dry run enabled; not sending closed-loop step {step_index}.", file=sys.stderr)
            continue

        print(f"Sending closed-loop step {step_index} with {len(actions)} grounded actions...", file=sys.stderr)
        response_text = send_actions(args.execute_actions_url, payload, args.send_timeout)
        execute_response = parse_execute_response_text(response_text)
        result["step_execute_response_summaries"].append(summarize_execute_response(execute_response, len(actions)))
        failure_reason = execute_response_failure_reason(execute_response)
        if failure_reason is not None:
            failure = closed_loop_failure_result(
                failure_reason,
                failed_step_index=step_index,
                failed_step=step,
            )
            result["closed_loop_result"] = failure
            result["known_robot_ids"] = known_robot_ids
            result["queried_robot_ids"] = queried_robot_ids
            result["final_object_visibility_summary"] = object_visibility_summary(object_visibility_map)
            print(failure["reason"], file=sys.stderr)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

        merge_execute_result_into_observation(executor_observation, execute_response)
        simulated_held_by_agent_id[executor_agent_id] = held_object_type_from_observation(executor_observation)

        executor_robot_id = executor_observation.get("robot_id")
        observations = extract_agent_observations(execute_response)
        if observations:
            observation = observations[0]
            if held_object_type_from_observation(observation) is None and simulated_held_by_agent_id.get(executor_agent_id):
                observation["inventory"] = list(executor_observation.get("inventory", []))
                if isinstance(executor_observation.get("robot_state"), dict):
                    observation["robot_state"] = dict(executor_observation["robot_state"])
            observation["is_primary"] = observation.get("robot_id") == args.primary_robot_id
            append_observation_if_new(agent_observations, observation)
            if isinstance(executor_robot_id, int) and executor_robot_id not in queried_robot_ids:
                queried_robot_ids.append(executor_robot_id)
            save_agent_observation_images(agent_observations, args.output_dir.expanduser().resolve(), task_id)
            object_visibility_map = build_object_visibility_map(agent_observations)
            result["agent_observations_summary"] = agent_observations_summary(agent_observations)
            result["object_visibility_summary"] = object_visibility_summary(object_visibility_map)

    done_payload = payload_for_actions(
        args,
        task_id,
        args.task,
        [{"action": "Done", "objectType": None, "targetType": None}],
        [{"action": "Done"}],
        last_executor_observation,
    )
    result["step_payloads"].append(done_payload)
    if args.dry_run:
        print("Dry run enabled; not sending final Done.", file=sys.stderr)
    else:
        print("Sending final Done for closed-loop task...", file=sys.stderr)
        response_text = send_actions(args.execute_actions_url, done_payload, args.send_timeout)
        execute_response = parse_execute_response_text(response_text)
        result["step_execute_response_summaries"].append(summarize_execute_response(execute_response, 1))
        failure_reason = execute_response_failure_reason(execute_response)
        if failure_reason is not None:
            result["closed_loop_result"] = {
                "status": "needs_upstream_planning",
                "failed_step_index": "Done",
                "failed_step": {"action": "Done", "objectType": None, "targetType": None},
                "reason": failure_reason,
            }
            result["known_robot_ids"] = known_robot_ids
            result["queried_robot_ids"] = queried_robot_ids
            result["robot_discovery_source"] = robot_discovery_source
            result["final_object_visibility_summary"] = object_visibility_summary(object_visibility_map)
            print(failure_reason, file=sys.stderr)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
    result["closed_loop_result"] = {"status": "success", "step_count": len(steps)}
    result["known_robot_ids"] = known_robot_ids
    result["queried_robot_ids"] = queried_robot_ids
    result["robot_discovery_source"] = robot_discovery_source
    result["final_object_visibility_summary"] = object_visibility_summary(object_visibility_map)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def run(args: argparse.Namespace) -> int:
    task_id = args.task_id or str(uuid.uuid4())
    try:
        print(f"Probing scene from {args.execute_actions_url}...", file=sys.stderr)
        base_url = base_url_from_execute_actions_url(args.execute_actions_url)
        probe = probe_scene(
            args.execute_actions_url,
            task_id,
            args.send_timeout,
            primary_robot_id=args.primary_robot_id,
            state_endpoint=args.state_endpoint,
        )
        agent_observations = extract_agent_observations(
            probe,
            args.primary_agent_id,
            primary_robot_id=args.primary_robot_id,
        )
        save_agent_observation_images(agent_observations, args.output_dir.expanduser().resolve(), task_id)
        primary_observation = primary_agent_observation(agent_observations)
        objects = primary_observation["objects"]
        image_path_value = primary_observation.get("image_path")
        if not isinstance(image_path_value, str):
            raise RuntimeError("primary agent observation did not include image_base64; send render_image=true")
        image_path = Path(image_path_value)
        object_visibility_map = build_object_visibility_map(agent_observations)
        known_robot_ids = known_robot_ids_for_run(args, probe, agent_observations)
        robot_discovery_source = robot_discovery_source_for_run(args, probe, agent_observations)
        queried_robot_ids = known_robot_ids_from_observations(agent_observations)
        print(f"Saved rendered scene image: {image_path}", file=sys.stderr)
        print(
            f"Primary agent: {primary_observation.get('agent_id')}; visible object categories: "
            + (", ".join(object_categories(objects, visible_only=True)) or "(none)"),
            file=sys.stderr,
        )

        navigation_object_type = navigation_object_type_for_task(
            args.task,
            object_categories(objects, visible_only=False),
        )
        if navigation_object_type is not None:
            result = navigation_goto_result(
                args,
                task_id=task_id,
                base_url=base_url,
                probe=probe,
                primary_observation=primary_observation,
                image_path=image_path,
                object_type=navigation_object_type,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

        if getattr(args, "task_intent_json", None):
            print("Using upstream task intent JSON...", file=sys.stderr)
        else:
            print("Generating task intent tool call with Qwen...", file=sys.stderr)
        task_intent_tool_call, task_intent, task_intent_tool_call_validation = task_intent_from_args(
            args,
            object_categories(objects, visible_only=False),
        )
        upstream_navigation_step = primary_intent_step(intent_steps(task_intent))
        if (
            upstream_navigation_step is not None
            and upstream_navigation_step.get("action") == "GotoObject"
            and isinstance(upstream_navigation_step.get("objectType"), str)
            and upstream_navigation_step.get("objectType").strip()
        ):
            result = navigation_goto_result(
                args,
                task_id=task_id,
                base_url=base_url,
                probe=probe,
                primary_observation=primary_observation,
                image_path=image_path,
                object_type=upstream_navigation_step["objectType"],
            )
            result["task_intent_source"] = task_intent_source_for_args(args)
            result["task_intent_tool_call"] = task_intent_tool_call
            result["task_intent_tool_call_validation"] = task_intent_tool_call_validation
            metadata = getattr(args, "_task_normalization", None)
            if isinstance(metadata, dict):
                result["task_normalization"] = metadata
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

        intent_expansion_warnings: list[str] = []
        if not (args.relay_mode and args.closed_loop_replan):
            intent_expansion_warnings = expand_put_object_intent_preconditions(task_intent, primary_observation)
        setattr(args, "_task_intent", task_intent)

        if args.closed_loop_replan:
            return run_closed_loop_replan(
                args,
                task_id=task_id,
                base_url=base_url,
                probe=probe,
                agent_observations=agent_observations,
                primary_observation=primary_observation,
                image_path=image_path,
                object_visibility_map=object_visibility_map,
                known_robot_ids=known_robot_ids,
                queried_robot_ids=queried_robot_ids,
                robot_discovery_source=robot_discovery_source,
                task_intent=task_intent,
                task_intent_tool_call=task_intent_tool_call,
                task_intent_tool_call_validation=task_intent_tool_call_validation,
            )

        print("Generating semantic plan with Qwen...", file=sys.stderr)
        raw_output, primary_semantic_plan, raw_output_path = generate_semantic_plan(args, image_path, objects, task_id)
        try:
            validate_task_intent_consistency(task_intent, primary_semantic_plan, check_action=False)
        except ValueError as exc:
            result = {
                "task_id": task_id,
                "image_path": str(image_path),
                "sceneName": primary_observation.get("sceneName"),
                "primary_agent_id": primary_observation.get("agent_id"),
                "primary_robot_id": primary_observation.get("robot_id"),
                "known_robot_ids": known_robot_ids,
                "queried_robot_ids": queried_robot_ids,
                "robot_discovery_source": robot_discovery_source,
                "visible_object_types": object_categories(objects, visible_only=True),
                "agent_observations_summary": agent_observations_summary(agent_observations),
                "object_visibility_summary": object_visibility_summary(object_visibility_map),
                "coordination_result": {
                    "status": "task_intent_mismatch",
                    "reason": str(exc),
                    "primary_agent_id": primary_observation.get("agent_id"),
                },
                "semantic_plan": primary_semantic_plan,
                "task_intent": task_intent,
                "task_intent_tool_call": task_intent_tool_call,
                "task_intent_tool_call_validation": task_intent_tool_call_validation,
                "task_intent_source": task_intent_source_for_args(args),
            }
            semantic_warnings = semantic_normalization_warnings(primary_semantic_plan)
            if semantic_warnings:
                result["semantic_normalization_warnings"] = semantic_warnings
            if raw_output_path is not None:
                result["raw_output_path"] = raw_output_path
            print(str(exc), file=sys.stderr)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        coordination_result = coordination_result_for_plan(
            args.task,
            primary_semantic_plan,
            object_visibility_map,
            task_intent,
            relay_mode=args.relay_mode,
        )
        agent_relay_result: dict[str, Any] | None = None
        if args.relay_mode:
            if args.relay_strategy == "agent":
                agent_relay_result = primary_fast_path_relay_result(
                    primary_semantic_plan,
                    task_intent,
                    agent_observations,
                    args.primary_robot_id,
                )
                if agent_relay_result is None:
                    (
                        agent_relay_result,
                        agent_observations,
                        object_visibility_map,
                        known_robot_ids,
                        queried_robot_ids,
                        robot_discovery_source,
                    ) = route_with_relay_agent(
                        args,
                        task_id=task_id,
                        base_url=base_url,
                        probe=probe,
                        semantic_plan=primary_semantic_plan,
                        task_intent=task_intent,
                        agent_observations=agent_observations,
                        object_visibility_map=object_visibility_map,
                    )
                    coordination_result = coordination_result_for_plan(
                        args.task,
                        primary_semantic_plan,
                        object_visibility_map,
                        task_intent,
                        relay_mode=True,
                    )
            else:
                (
                    agent_observations,
                    object_visibility_map,
                    coordination_result,
                    known_robot_ids,
                    queried_robot_ids,
                    robot_discovery_source,
                ) = query_relay_observations_if_needed(
                    args,
                    task_id,
                    base_url,
                    probe,
                    primary_semantic_plan,
                    agent_observations,
                    object_visibility_map,
                    coordination_result,
                    task_intent,
                )
            primary_observation = primary_agent_observation(agent_observations)
            objects = primary_observation["objects"]

        result: dict[str, Any] = {
            "task_id": task_id,
            "image_path": str(image_path),
            "sceneName": primary_observation.get("sceneName"),
            "primary_agent_id": primary_observation.get("agent_id"),
            "primary_robot_id": primary_observation.get("robot_id"),
            "known_robot_ids": known_robot_ids,
            "queried_robot_ids": queried_robot_ids,
            "robot_discovery_source": robot_discovery_source,
            "visible_object_types": object_categories(objects, visible_only=True),
            "agent_observations_summary": agent_observations_summary(agent_observations),
            "object_visibility_summary": object_visibility_summary(object_visibility_map),
            "coordination_result": coordination_result,
            "semantic_plan": primary_semantic_plan,
            "task_intent": task_intent,
            "task_intent_tool_call": task_intent_tool_call,
            "task_intent_tool_call_validation": task_intent_tool_call_validation,
            "task_intent_source": task_intent_source_for_args(args),
        }
        if intent_expansion_warnings:
            result["intent_expansion_warnings"] = intent_expansion_warnings
        semantic_warnings = semantic_normalization_warnings(primary_semantic_plan)
        if semantic_warnings:
            result["semantic_normalization_warnings"] = semantic_warnings
        if raw_output_path is not None:
            result["raw_output_path"] = raw_output_path
        if args.include_object_visibility_map:
            result["object_visibility_map"] = object_visibility_map
        if args.save_object_visibility_map:
            result["object_visibility_map_path"] = save_text_artifact(
                args.output_dir,
                task_id,
                "object_visibility_map.json",
                json.dumps(object_visibility_map, ensure_ascii=False, indent=2),
            )

        if not args.relay_mode:
            if coordination_result.get("status") == "target_visible_by_peer":
                print(coordination_result.get("message"), file=sys.stderr)
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 0

            repair_warnings = repair_redundant_pickup_for_held_put(
                primary_semantic_plan,
                task_intent,
                primary_observation,
            )
            if repair_warnings:
                result["semantic_plan_repair_warnings"] = repair_warnings

            validate_task_intent_consistency(task_intent, primary_semantic_plan)
            if not has_multi_step_intent(task_intent):
                validate_goal_consistency(args.task, primary_semantic_plan, objects)
                validate_action_intent_consistency(args.task, primary_semantic_plan)
            validate_action_affordances(
                primary_semantic_plan,
                objects,
                allow_invisible=args.allow_invisible_object_ids,
            )
            validate_action_state_preconditions(
                primary_semantic_plan,
                primary_observation,
                allow_invisible=args.allow_invisible_object_ids,
            )
            executor_agent_id = None
            executor_observation = primary_observation
            executor_semantic_plan = primary_semantic_plan
        else:
            relay_result = agent_relay_result
            if relay_result is None:
                relay_result = choose_relay_executor(
                    args.task,
                    primary_semantic_plan,
                    object_visibility_map,
                    agent_observations,
                    task_intent,
                )
                relay_result["strategy"] = "rules"
                relay_result["known_robot_ids"] = known_robot_ids
            relay_result.setdefault("primary_robot_id", primary_observation.get("robot_id"))
            relay_result.setdefault("known_robot_ids", known_robot_ids)
            result["relay_result"] = relay_result
            if relay_result.get("status") == "needs_upstream_planning":
                print(relay_result.get("reason"), file=sys.stderr)
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 0

            executor_agent_id = str(relay_result["executor_agent_id"])
            executor_observation = agent_observation_by_id(agent_observations, executor_agent_id)
            executor_objects = executor_observation["objects"]
            executor_image_path_value = executor_observation.get("image_path")
            if not isinstance(executor_image_path_value, str):
                result["relay_result"] = relay_failure_result(
                    f"executor agent {executor_agent_id!r} did not include image_base64; cannot replan safely",
                    requested_object_type=relay_result.get("requested_object_type"),
                    primary_agent_id=str(relay_result.get("primary_agent_id")),
                    agent_observations=agent_observations,
                    known_robot_ids=known_robot_ids,
                )
                print(result["relay_result"].get("reason"), file=sys.stderr)
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 0
            executor_image_path = Path(executor_image_path_value)

            if executor_agent_id == primary_observation.get("agent_id"):
                executor_semantic_plan = primary_semantic_plan
            else:
                print(f"Relay executor {executor_agent_id} selected; replanning from executor view...", file=sys.stderr)
                raw_output, executor_semantic_plan, raw_output_path = generate_semantic_plan(
                    args,
                    executor_image_path,
                    executor_objects,
                    task_id,
                )
                result["executor_semantic_plan"] = executor_semantic_plan
                result["semantic_plan"] = executor_semantic_plan
                semantic_warnings = semantic_normalization_warnings(executor_semantic_plan)
                if semantic_warnings:
                    result["semantic_normalization_warnings"] = semantic_warnings
                else:
                    result.pop("semantic_normalization_warnings", None)
                if raw_output_path is not None:
                    result["raw_output_path"] = raw_output_path

            repair_warnings = repair_redundant_pickup_for_held_put(
                executor_semantic_plan,
                task_intent,
                executor_observation,
            )
            if repair_warnings:
                result["semantic_plan_repair_warnings"] = repair_warnings

            validation_failure = validate_executor_plan_or_failure(
                args.task,
                executor_semantic_plan,
                executor_objects,
                allow_invisible=args.allow_invisible_object_ids,
                relay_result=relay_result,
                agent_observations=agent_observations,
                task_intent=task_intent,
                executor_observation=executor_observation,
            )
            if validation_failure is not None:
                validation_failure["known_robot_ids"] = known_robot_ids
                for key in ("primary_inability_reason", "coordination_explanation", "candidate_explanation_summary"):
                    if isinstance(relay_result.get(key), str):
                        validation_failure[key] = relay_result[key]
                result["relay_result"] = validation_failure
                print(validation_failure.get("reason"), file=sys.stderr)
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 0

            result["relay_result"] = {
                **relay_result,
                "status": "executor_ready",
            }
            executor_robot_id = executor_observation.get("robot_id")
            result["executor_agent_id"] = executor_agent_id
            result["executor_robot_id"] = executor_robot_id
            result["executor_image_path"] = str(executor_image_path)
            result["executor_visible_object_types"] = object_categories(executor_objects, visible_only=True)
            result["primary_semantic_plan"] = primary_semantic_plan
            objects = executor_objects

        actions = ground_semantic_plan(
            executor_semantic_plan,
            objects,
            allow_invisible=args.allow_invisible_object_ids,
            max_actions=args.max_actions,
        )
        payload = {
            "task_id": task_id,
            "task": executor_semantic_plan.get("task", args.task),
            "plan": executor_semantic_plan.get("plan", []),
            "stop_on_failure": False,
            "actions": actions,
        }
        if args.relay_mode and executor_agent_id is not None:
            executor_robot_id = executor_observation.get("robot_id") if isinstance(executor_observation, dict) else None
            if args.executor_agent_id_field == "robot_id" and isinstance(executor_robot_id, int):
                payload[args.executor_agent_id_field] = executor_robot_id
            else:
                payload[args.executor_agent_id_field] = executor_agent_id
            payload["render_image"] = True
        result["payload"] = payload

        if args.dry_run:
            print("Dry run enabled; not sending grounded actions.", file=sys.stderr)
        else:
            print(f"Sending {len(actions)} grounded actions...", file=sys.stderr)
            response_text = send_actions(args.execute_actions_url, payload, args.send_timeout)
            add_execute_response_result(
                result,
                response_text,
                include_response=args.include_execute_response,
                save_response=args.save_execute_response,
                output_dir=args.output_dir,
                task_id=task_id,
                action_count=len(actions),
            )

        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
