from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOCAL_HF_CACHE = REPO_ROOT.parent / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(LOCAL_HF_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(LOCAL_HF_CACHE / "transformers"))

IMAGE_EXTENSIONS = {
    ".bmp",
    ".dib",
    ".jpeg",
    ".jpg",
    ".pbm",
    ".pgm",
    ".png",
    ".ppm",
    ".tif",
    ".tiff",
}
VIDEO_EXTENSIONS = {".avi", ".iso", ".mkv", ".mp4", ".webm", ".wmv"}
QWEN35_HUB_MODEL = "Qwen/Qwen3.5-4B"
LOCAL_QWEN35_MODEL = REPO_ROOT.parent / "models" / "Qwen3.5-4B"
DEFAULT_QWEN35_MODEL = str(LOCAL_QWEN35_MODEL) if LOCAL_QWEN35_MODEL.is_dir() else QWEN35_HUB_MODEL
DEFAULT_HTTP_TIMEOUT = 10.0


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a one-shot EmbodiedGPT image/video capability check."
    )
    parser.add_argument("--media", required=True, type=Path, help="Path to an image or video file.")
    intent = parser.add_mutually_exclusive_group()
    intent.add_argument("--task", help="Task for which the model should produce an embodied plan.")
    intent.add_argument("--question", help="Question to answer about a video.")
    parser.add_argument("--plan-mode", choices=["semantic", "native"], default="semantic", help="Planning output mode: semantic drafts avoid objectId and require later grounding; native keeps executable AI2-THOR actions.")
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN35_MODEL)
    parser.add_argument("--qwen-device-map", default="auto")
    parser.add_argument("--qwen-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-new-tokens", type=positive_int, default=256)
    parser.add_argument("--temperature", type=positive_float, default=0.2)
    parser.add_argument(
        "--send-actions-url",
        help="HTTP endpoint that receives the executable plan mapped to {'actions': [...]}.",
    )
    parser.add_argument(
        "--task-id",
        help="Optional task id included in the HTTP payload sent with --send-actions-url.",
    )
    parser.add_argument(
        "--send-timeout",
        type=positive_float,
        default=DEFAULT_HTTP_TIMEOUT,
        help=f"HTTP timeout in seconds for --send-actions-url (default: {DEFAULT_HTTP_TIMEOUT}).",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print only the parsed executable plan JSON instead of the full model output.",
    )
    return parser.parse_args(argv)


def media_type(path: Path) -> str | None:
    extension = path.suffix.lower()
    if extension in IMAGE_EXTENSIONS:
        return "image"
    if extension in VIDEO_EXTENSIONS:
        return "video"
    return None


SEMANTIC_ACTIONS = {
    "MoveAhead",
    "MoveBack",
    "MoveLeft",
    "MoveRight",
    "RotateLeft",
    "RotateRight",
    "LookUp",
    "LookDown",
    "PickupObject",
    "PutObject",
    "OpenObject",
    "CloseObject",
    "DropHandObject",
    "PushObject",
    "PullObject",
    "MoveHeldObject",
    "SliceObject",
    "BreakObject",
    "CookObject",
    "CleanObject",
    "FillObjectWithLiquid",
    "SetObjectStates",
    "Pass",
    "Done",
}


def semantic_planning_prompt(kind: str, task: str | None, task_intent: dict[str, Any] | None = None) -> str:
    grounding = (
        "Base the draft on objects and spatial relationships visible in the image."
        if kind == "image"
        else "Respect the observed action order and visible state changes in the video."
    )
    if task:
        objective = f'The task is: "{task.strip()}".'
    else:
        objective = "Infer the most plausible embodied task from the visual input."

    allowed_actions = ", ".join(sorted(SEMANTIC_ACTIONS))
    task_intent_text = ""
    example_action = "Pass"
    example_object: str | None = None
    example_target: str | None = None
    if task_intent is not None:
        requested_action = task_intent.get("requestedAction")
        requested_object = task_intent.get("requestedObjectType")
        intent_steps = task_intent.get("intentSteps") if isinstance(task_intent.get("intentSteps"), list) else []
        requested_action_text = requested_action if requested_action is not None else "null"
        requested_object_text = requested_object if requested_object is not None else "null"
        intent_steps_text = json.dumps(intent_steps, ensure_ascii=False)
        task_intent_text = (
            "\n\nTask intent tool result: "
            f"requestedAction={requested_action_text}, requestedObjectType={requested_object_text}, "
            f"intentSteps={intent_steps_text}. "
            "Treat requestedAction, requestedObjectType, and every intentSteps item as hard constraints. "
            "For multi-step tasks, semantic plan[] must contain matching intentSteps in the same order; "
            "extra navigation or look actions may appear between required steps. "
            "targetObjectType should equal the current or primary requested object from the tool result. "
            "The plan must preserve requestedAction when it is not null. "
            "Do not infer, replace, or rewrite task intent from the image."
        )
        if intent_steps and isinstance(intent_steps[0], dict):
            example_step = intent_steps[0]
            if isinstance(example_step.get("action"), str):
                example_action = example_step["action"]
            example_object = (
                example_step.get("objectType")
                if isinstance(example_step.get("objectType"), str)
                else None
            )
            example_target = (
                example_step.get("targetType")
                if isinstance(example_step.get("targetType"), str)
                else None
            )
        elif isinstance(requested_action, str):
            example_action = requested_action
            example_object = requested_object if isinstance(requested_object, str) else None

    example_event = {
        "PickupObject": "picked up",
        "PutObject": "placed",
        "OpenObject": "opened",
        "CloseObject": "closed",
    }.get(example_action)
    example_observations = []
    if example_object is not None and example_event is not None:
        example_observations.append(
            {
                "order": 1,
                "eventType": _expected_event_type_for_event(example_event) or "moved_object",
                "objectType": example_object,
                "event": example_event,
                "targetType": example_target,
            }
        )
    example_json = json.dumps(
        {
            "task": task.strip() if isinstance(task, str) and task.strip() else "perform the requested task",
            "targetObjectType": example_object,
            "needsGrounding": True,
            "observations": example_observations,
            "plan": [
                {
                    "action": example_action,
                    "objectType": example_object,
                    "targetType": example_target,
                }
            ],
        },
        ensure_ascii=False,
        indent=2,
    )
    return f"""Return one JSON object only. The first character must be {{. Do not output <think>, reasoning, markdown, comments, or text outside JSON.

Visual input: {kind}. {objective} {grounding}{task_intent_text}

Required top-level keys: task, targetObjectType, needsGrounding, observations, plan. Set needsGrounding to true.

targetObjectType: the primary object category requested by the user's task, or null when the task has no explicit object. It must come from the task text, not from a visible substitute.

observations: list every visible robot-relevant event from the whole input in order. Each observation object must have:
- order: positive integer
- eventType: moved_object or state_changed_object
- objectType: concrete visible object category
- event: short past-tense phrase such as picked up, placed, pushed, pulled, opened, closed
- targetType: target category or null
Use moved_object for pickup/place/put down/push/pull/drop events. Use state_changed_object only for open/close state changes.

plan: semantic action objects derived from observations and task intent. Each plan item must have action, objectType, targetType.
For multi-step task intent, include every required intent step in plan[] in the same order.
Use only these action names: {allowed_actions}.

Minimal valid JSON shape:
{example_json}

Never omit needsGrounding, targetObjectType, or targetType. Use targetObjectType: null when the task has no explicit object. Use targetType: null when there is no target category.
Do not replace targetObjectType or the requested object with another visible object. If the requested object is unavailable, do not invent a substitute action.
Use PickupObject only for pickupable small objects. Do not use PickupObject for large fixtures or openable receptacles such as Fridge, Cabinet, Drawer, or Microwave; use OpenObject or CloseObject for those when the task asks to open or close them.
Do not replace the user's requested action with another more feasible action. For example, if the task says pick up the Fridge, do not output OpenObject; preserve the requested action intent and let validation reject impossible actions.
For navigation tasks with no object target, set targetObjectType: null, objectType: null, and targetType: null in each navigation plan item.
Use RotateRight/RotateLeft for turn or rotate right/left. Use MoveRight/MoveLeft only for move, strafe, or step right/left.

Mapping rules:
- turn right or rotate right -> RotateRight; turn left or rotate left -> RotateLeft
- move right, strafe right, or step right -> MoveRight; move left, strafe left, or step left -> MoveLeft
- move forward, go forward, or move ahead -> MoveAhead; move back, go back, or back up -> MoveBack
- look up -> LookUp; look down -> LookDown
- picked up -> PickupObject
- placed or put down -> PutObject
- pushed -> PushObject; pulled -> PullObject
- opened -> OpenObject; closed -> CloseObject
- sliced -> SliceObject; broken -> BreakObject; cleaned -> CleanObject; cooked -> CookObject; filled -> FillObjectWithLiquid
- Door, drawer, fridge, cabinet, and microwave opening/closing are state_changed_object events. Use whole object types such as Fridge, Drawer, Cabinet, Microwave; never FridgeDoor.

Forbidden: objectId, receptacleObjectId, coordinates, exact positions, UNKNOWN, pipe-delimited ids, NavTo, NavigateTo, ToggleObject, MoveRelative, MediaPlayer, WatchVideo, Wait."""


def native_planning_prompt(kind: str, task: str | None) -> str:
    grounding = (
        "Base every step on objects and spatial relationships visible in the image."
        if kind == "image"
        else "Respect the observed action order and state changes in the video."
    )
    if task:
        objective = f'The task is: "{task.strip()}".'
    else:
        objective = "Infer the most plausible embodied task from the visual input."

    return f"""You are an AI2-THOR embodied agent planner. Analyze the {kind} carefully.
{objective}
Generate a concise, executable AI2-THOR action sequence. {grounding}

Return ONLY valid JSON. Do not include markdown, comments, explanations, or extra text.

Schema:
{{
  "task": "<short task description>",
  "plan": [
    {{"action": "<AI2-THOR action name>", "...": "<required arguments>"}}
  ]
}}

Rules:
- Every item in "plan" must be a valid AI2-THOR controller.step action dictionary.
- Use navigation actions when needed: MoveAhead, MoveBack, MoveLeft, MoveRight, RotateLeft, RotateRight, LookUp, LookDown, Crouch, Stand.
- Use interaction actions when appropriate: PickupObject, PutObject, OpenObject, CloseObject, ToggleObjectOn, ToggleObjectOff, SliceObject, BreakObject, DirtyObject, CleanObject, FillObjectWithLiquid, EmptyLiquidFromObject.
- For PickupObject, OpenObject, CloseObject, ToggleObjectOn, ToggleObjectOff, SliceObject, BreakObject, DirtyObject, CleanObject, FillObjectWithLiquid, and EmptyLiquidFromObject, include "objectId".
- For PutObject, include both "objectId" and "receptacleObjectId".
- If the exact AI2-THOR objectId is not provided, use a placeholder derived from the visible object type, such as "Apple|UNKNOWN" or "CounterTop|UNKNOWN".
- Do not invent exact coordinates or exact object instance IDs.
- Prefer short, executable action sequences.
- If the task cannot be completed from visible evidence, still output the best approximate AI2-THOR action sequence."""


def planning_prompt(kind: str, task: str | None, plan_mode: str = "semantic") -> str:
    if plan_mode == "native":
        return native_planning_prompt(kind, task)
    return semantic_planning_prompt(kind, task)


def question_prompt(question: str) -> str:
    return f"""Watch the video carefully and answer the question using only events visible in the video.
Pay attention to action order and state changes. Answer concisely in English.
Question: {question.strip()}"""


def extract_json_object(output: str) -> dict[str, Any]:
    stripped = output.strip()
    decoder = json.JSONDecoder()
    parsed_objects: list[dict[str, Any]] = []
    last_error = None

    fenced_blocks = re.findall(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
    scan_sources = fenced_blocks + [stripped]

    for source in scan_sources:
        for match in re.finditer(r"\{", source):
            try:
                parsed, _ = decoder.raw_decode(source[match.start():])
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            if isinstance(parsed, dict):
                parsed_objects.append(parsed)

    if parsed_objects:
        for parsed in reversed(parsed_objects):
            if "task" in parsed and "plan" in parsed:
                return parsed
        for parsed in reversed(parsed_objects):
            if "task" in parsed:
                return parsed
        return parsed_objects[0]

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        last_error = exc
    else:
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("model JSON must be an object")

    if last_error is not None:
        raise ValueError(f"model output is not valid JSON: {last_error.msg}") from last_error
    raise ValueError("model output does not contain a JSON object")


def _contains_forbidden_semantic_value(value: Any) -> bool:
    if isinstance(value, str):
        return "UNKNOWN" in value.upper() or "|" in value or "<" in value or ">" in value
    if isinstance(value, list):
        return any(_contains_forbidden_semantic_value(item) for item in value)
    if isinstance(value, dict):
        forbidden_keys = {"objectId", "receptacleObjectId", "position", "rotation", "x", "y", "z"}
        return any(key in forbidden_keys or _contains_forbidden_semantic_value(item) for key, item in value.items())
    return False


SEMANTIC_EVENT_TYPES = {"moved_object", "state_changed_object"}
FORBIDDEN_SEMANTIC_OBJECT_TYPES = {"FridgeDoor", "CabinetDoor", "MicrowaveDoor", "DrawerDoor"}
MOVED_OBJECT_EVENT_PHRASES = ("picked up", "placed", "put down", "pushed", "pulled", "dropped")
STATE_CHANGED_EVENT_PHRASES = ("opened", "closed")


def _expected_event_type_for_event(event: Any) -> str | None:
    if not isinstance(event, str):
        return None
    normalized = " ".join(re.findall(r"[a-z0-9]+", event.lower()))
    if any(phrase in normalized for phrase in MOVED_OBJECT_EVENT_PHRASES):
        return "moved_object"
    if any(phrase in normalized for phrase in STATE_CHANGED_EVENT_PHRASES):
        return "state_changed_object"
    return None


def _semantic_normalization_warnings(parsed: dict[str, Any]) -> list[str]:
    warnings = parsed.get("semanticNormalizationWarnings")
    if not isinstance(warnings, list):
        warnings = []
        parsed["semanticNormalizationWarnings"] = warnings
    return warnings


def _normalize_semantic_planning_output(parsed: dict[str, Any]) -> dict[str, Any]:
    if "needsGrounding" not in parsed:
        parsed["needsGrounding"] = True
    if "targetObjectType" not in parsed:
        parsed["targetObjectType"] = None

    observations = parsed.get("observations")
    if isinstance(observations, list):
        for index, observation in enumerate(observations, start=1):
            if not isinstance(observation, dict):
                continue
            if "targetType" not in observation:
                observation["targetType"] = None
            expected_event_type = _expected_event_type_for_event(observation.get("event"))
            current_event_type = observation.get("eventType")
            if (
                expected_event_type is not None
                and current_event_type in SEMANTIC_EVENT_TYPES
                and current_event_type != expected_event_type
            ):
                observation["eventType"] = expected_event_type
                _semantic_normalization_warnings(parsed).append(
                    f"observations[{index}].eventType normalized from {current_event_type!r} "
                    f"to {expected_event_type!r} for event {observation.get('event')!r}"
                )

    plan = parsed.get("plan")
    if isinstance(plan, list):
        for step in plan:
            if isinstance(step, dict) and "targetType" not in step:
                step["targetType"] = None

    return parsed


def _validate_semantic_nullable_text(value: Any, field_name: str) -> None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or null")
    if isinstance(value, str) and value in FORBIDDEN_SEMANTIC_OBJECT_TYPES:
        raise ValueError(f"{field_name} must use the whole openable object type, not {value}")


def _validate_semantic_observation(observation: Any, index: int) -> None:
    if not isinstance(observation, dict):
        raise ValueError(f"observations[{index}] must be an object")

    required = ("order", "eventType", "objectType", "event", "targetType")
    for field in required:
        if field not in observation:
            raise ValueError(f"observations[{index}] must contain {field}")

    order = observation.get("order")
    if not isinstance(order, int) or order <= 0:
        raise ValueError(f"observations[{index}].order must be a positive integer")

    event_type = observation.get("eventType")
    if event_type not in SEMANTIC_EVENT_TYPES:
        raise ValueError(f"observations[{index}].eventType must be moved_object or state_changed_object")

    if not isinstance(observation.get("objectType"), str) or not observation["objectType"].strip():
        raise ValueError(f"observations[{index}].objectType must be a non-empty string")
    if observation["objectType"] in FORBIDDEN_SEMANTIC_OBJECT_TYPES:
        raise ValueError(f"observations[{index}].objectType must use the whole openable object type, not {observation['objectType']}")
    if not isinstance(observation.get("event"), str) or not observation["event"].strip():
        raise ValueError(f"observations[{index}].event must be a non-empty string")
    _validate_semantic_nullable_text(observation.get("targetType"), f"observations[{index}].targetType")


def _validate_semantic_plan_step(step: Any, index: int) -> None:
    if not isinstance(step, dict):
        raise ValueError(f"plan[{index}] must be an object")
    action = step.get("action")
    if not isinstance(action, str) or action not in SEMANTIC_ACTIONS:
        raise ValueError(f"plan[{index}] has unsupported action: {action}")
    if "objectType" not in step or "targetType" not in step:
        raise ValueError(f"plan[{index}] must contain objectType and targetType fields")
    _validate_semantic_nullable_text(step.get("objectType"), f"plan[{index}].objectType")
    _validate_semantic_nullable_text(step.get("targetType"), f"plan[{index}].targetType")


def parse_semantic_planning_output(output: str) -> dict[str, Any]:
    parsed = extract_json_object(output)

    if "actions" in parsed:
        raise ValueError('semantic output must not contain top-level "actions"; put semantic actions in "plan"')
    parsed = _normalize_semantic_planning_output(parsed)
    if parsed.get("needsGrounding") is not True:
        raise ValueError('semantic output must contain "needsGrounding": true')
    if not isinstance(parsed.get("task"), str) or not parsed["task"].strip():
        raise ValueError('semantic output must contain a non-empty string "task"')
    _validate_semantic_nullable_text(parsed.get("targetObjectType"), "targetObjectType")

    observations = parsed.get("observations")
    if not isinstance(observations, list):
        raise ValueError('semantic output must contain an "observations" list')
    plan = parsed.get("plan")
    if not isinstance(plan, list):
        raise ValueError('semantic output must contain a "plan" list')
    if _contains_forbidden_semantic_value(parsed):
        raise ValueError("semantic output must not contain objectId, receptacleObjectId, coordinates, UNKNOWN placeholders, angle-bracket placeholders, or pipe-delimited ids")

    for index, observation in enumerate(observations, start=1):
        _validate_semantic_observation(observation, index)
    for index, step in enumerate(plan, start=1):
        _validate_semantic_plan_step(step, index)

    if parsed.get("semanticNormalizationWarnings") == []:
        parsed.pop("semanticNormalizationWarnings", None)

    return parsed


def parse_native_planning_output(output: str) -> dict[str, Any]:
    parsed = extract_json_object(output)

    if not isinstance(parsed.get("task"), str) or not parsed["task"].strip():
        raise ValueError('native output must contain a non-empty string "task"')
    if "actions" in parsed:
        raise ValueError('native output must not contain top-level "actions"; put executable actions in "plan"')

    plan = parsed.get("plan")
    if not isinstance(plan, list) or not plan:
        raise ValueError('native output must contain a non-empty "plan" list')

    for index, action in enumerate(plan, start=1):
        if not isinstance(action, dict):
            raise ValueError(f"plan[{index}] must be an object")
        if not isinstance(action.get("action"), str) or not action["action"]:
            raise ValueError(f'plan[{index}] must contain a non-empty "action" string')

    return parsed


def build_execution_payload(parsed: dict[str, Any], task_id: str | None = None) -> dict[str, Any]:
    executable_plan = parsed["plan"]
    return {
        "task_id": task_id or str(uuid.uuid4()),
        "task": parsed["task"],
        "plan": executable_plan,
        "stop_on_failure": False,
        "actions": executable_plan,
    }


def plan_only_document(executable_plan: list[dict[str, Any]]) -> dict[str, Any]:
    return {"plan": executable_plan}


def send_actions(url: str, payload: dict[str, Any], timeout: float) -> str:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP POST failed with status {exc.code}: {error_body[:200]}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"HTTP POST failed: {exc.reason}") from exc


def validate_inputs(args: argparse.Namespace) -> tuple[Path, str]:
    media = args.media.expanduser().resolve()

    if not media.is_file():
        raise ValueError(f"media file does not exist: {media}")
    kind = media_type(media)
    if kind is None:
        supported = ", ".join(sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS))
        raise ValueError(f"unsupported media extension '{media.suffix}'; supported: {supported}")
    if args.question and kind != "video":
        raise ValueError("--question is supported only for video input; use --task for an image")
    if args.question and (args.plan_only or args.send_actions_url):
        raise ValueError("--plan-only and --send-actions-url require planning mode, not --question")
    if args.plan_mode == "semantic" and (args.plan_only or args.send_actions_url):
        raise ValueError("--plan-only and --send-actions-url require --plan-mode native")
    return media, kind


def run(args: argparse.Namespace) -> int:
    try:
        media, kind = validate_inputs(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        print("error: CUDA was requested but is not available", file=sys.stderr)
        return 2
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    if args.question:
        prompt = question_prompt(args.question)
        planning = False
    else:
        prompt = planning_prompt(kind, args.task, args.plan_mode)
        planning = True

    print(f"Processing {kind}: {media} with Qwen3.5...", file=sys.stderr)
    try:
        from demo.qwen35_backend import Qwen35Backend, Qwen35Config

        qwen = Qwen35Backend(
            Qwen35Config(
                model_name=args.qwen_model,
                device=device,
                device_map=args.qwen_device_map,
                torch_dtype=args.qwen_dtype,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        )
        output = qwen.generate(str(media), kind, prompt).strip()
    except Exception as exc:
        print(f"error: inference failed: {exc}", file=sys.stderr)
        return 1

    if planning and args.plan_mode == "semantic":
        try:
            semantic_output = parse_semantic_planning_output(output)
        except ValueError as exc:
            print(f"error: failed to parse semantic plan: {exc}", file=sys.stderr)
            print("Raw model output:", file=sys.stderr)
            print(output, file=sys.stderr)
            return 1
        print(json.dumps(semantic_output, ensure_ascii=False))
        return 0

    parsed_output = None
    executable_plan = None
    if planning:
        try:
            parsed_output = parse_native_planning_output(output)
            executable_plan = parsed_output["plan"]
        except ValueError as exc:
            print(f"error: failed to parse native plan: {exc}", file=sys.stderr)
            return 1

    if args.send_actions_url:
        assert executable_plan is not None
        assert parsed_output is not None
        payload = build_execution_payload(parsed_output, args.task_id)
        try:
            response_text = send_actions(args.send_actions_url, payload, args.send_timeout)
        except Exception as exc:
            print(f"error: failed to send actions: {exc}", file=sys.stderr)
            return 1
        print(f"Sent {len(executable_plan)} actions to {args.send_actions_url}", file=sys.stderr)
        if response_text:
            print(f"Receiver response: {response_text}", file=sys.stderr)

    if args.plan_only:
        assert executable_plan is not None
        print(json.dumps(plan_only_document(executable_plan), ensure_ascii=False))
    elif parsed_output is not None:
        print(json.dumps(parsed_output, ensure_ascii=False))
    else:
        print(output)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
