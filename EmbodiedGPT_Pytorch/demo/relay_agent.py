from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass
from typing import Any, Callable


OBSERVE_ROBOT_TOOL = {
    "type": "function",
    "function": {
        "name": "observe_robot",
        "description": "Observe one known robot and return a compact summary of what it can currently act on.",
        "parameters": {
            "type": "object",
            "properties": {"robot_id": {"type": "integer"}},
            "required": ["robot_id"],
        },
    },
}

INSPECT_GLOBAL_SCENE_TOOL = {
    "type": "function",
    "function": {
        "name": "inspect_global_scene",
        "description": "Inspect the current global scene summary available to the relay agent.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

EVALUATE_EXECUTOR_CANDIDATES_TOOL = {
    "type": "function",
    "function": {
        "name": "evaluate_executor_candidates",
        "description": (
            "Evaluate known robots against the current task and return evidence such as visibility, held state, "
            "affordances, validation failures, and distance to the target. This tool does not choose the executor."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

SELECT_EXECUTOR_TOOL = {
    "type": "function",
    "function": {
        "name": "select_executor",
        "description": "Select a robot that has enough observed evidence to execute the requested task.",
        "parameters": {
            "type": "object",
            "properties": {
                "robot_id": {"type": "integer"},
                "reason": {"type": "string"},
            },
            "required": ["robot_id", "reason"],
        },
    },
}

REPORT_FAILURE_TOOL = {
    "type": "function",
    "function": {
        "name": "report_failure",
        "description": "Report that no known robot can execute the task after the required observations are complete.",
        "parameters": {
            "type": "object",
            "properties": {
                "failure_code": {
                    "type": "string",
                    "enum": [
                        "target_not_visible",
                        "object_not_actionable",
                        "missing_required_state",
                        "observation_failed",
                        "unsupported_task",
                        "other",
                    ],
                },
                "reason": {"type": "string"},
            },
            "required": ["failure_code", "reason"],
        },
    },
}

RELAY_TOOLS = [
    INSPECT_GLOBAL_SCENE_TOOL,
    EVALUATE_EXECUTOR_CANDIDATES_TOOL,
    OBSERVE_ROBOT_TOOL,
    SELECT_EXECUTOR_TOOL,
    REPORT_FAILURE_TOOL,
]
TERMINAL_TOOL_NAMES = {"select_executor", "report_failure"}
ALLOWED_FAILURE_CODES = {
    "target_not_visible",
    "object_not_actionable",
    "missing_required_state",
    "observation_failed",
    "unsupported_task",
    "other",
}
SCHEMA_ARGUMENT_KEYS = {"type", "properties", "required", "items", "enum", "$schema"}


@dataclass
class RelayAgentConfig:
    max_turns: int = 8


def _tool_call_from_json(value: Any) -> dict[str, Any] | None:
    if isinstance(value, list):
        for item in value:
            call = _tool_call_from_json(item)
            if call is not None:
                return call
        return None
    if not isinstance(value, dict):
        return None
    if isinstance(value.get("tool_calls"), list):
        return _tool_call_from_json(value["tool_calls"])
    function = value.get("function")
    if isinstance(function, dict):
        name = function.get("name") or value.get("name")
        arguments = function.get("arguments", value.get("arguments", {}))
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


def parse_relay_tool_call(output: str) -> dict[str, Any]:
    """Parse JSON, JSON inside <tool_call>, and OpenAI-style JSON tool calls."""

    decoder = json.JSONDecoder()
    tool_blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", output, flags=re.DOTALL | re.IGNORECASE)
    candidates = [*tool_blocks, output]
    for candidate in candidates:
        for index, char in enumerate(candidate):
            if char not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            call = _tool_call_from_json(value)
            if call is not None:
                return call
    raise ValueError("relay agent output did not contain a valid JSON tool call")


def relay_system_prompt(evidence_precollected: bool = False) -> str:
    prompt = (
        "You are a global robot task coordination agent. Your job is to reason over the task, global scene "
        "evidence, robot visibility, held objects, affordances, state preconditions, and distances, then select "
        "the best executor robot or report why the task cannot be completed. You are the decision maker: "
        "candidate evidence tools provide facts and validation results, but they do not choose for you. First use "
        "global_scene_summary, already_observed, known_visibility, and candidate evidence from the input. Prefer "
        "inspect_global_scene and evaluate_executor_candidates when you need structured global evidence. Treat "
        "robots listed in visibility_unknown_robot_ids as known robots whose visibility is unknown, not as robots "
        "that can see the target. Use observe_robot only when global evidence is missing, stale, inconsistent, "
        "visibility_unknown, or validation feedback asks for fresher evidence. You may re-observe a known robot "
        "when fresh evidence is needed, but if repeated observations add no useful evidence, choose a proven "
        "executor, inspect/evaluate candidates, observe a more relevant unknown robot, or report failure. Never "
        "invent visibility, capabilities, held objects, positions, distances, or robot ids. Call select_executor "
        "only when the observed or globally inspected evidence proves that the robot can perform the requested "
        "action. If several robots can execute, choose the one that best fits the task; distance to the target is "
        "important, but you may also consider held objects, receptacle visibility, primary robot preference, and "
        "validation feedback. Before report_failure, make sure there is enough observed or global evidence for "
        "the failure unless the task itself is unsupported. Return exactly one JSON object per turn with keys name "
        "and arguments. "
        "Do not output XML, markdown, explanations, example_function_name, or text outside the JSON object. "
        "Allowed calls: inspect_global_scene(), evaluate_executor_candidates(), observe_robot(robot_id: integer), "
        "select_executor(robot_id: integer, reason: non-empty string), and "
        "report_failure(failure_code: allowed string, reason: non-empty string). "
        "arguments must contain actual values. Never put schema fields such as type, properties, required, "
        "items, or enum inside arguments. "
        "Allowed failure_code values: target_not_visible, object_not_actionable, missing_required_state, "
        "observation_failed, unsupported_task, other. "
        "Valid examples: "
        '{"name":"inspect_global_scene","arguments":{}}, '
        '{"name":"evaluate_executor_candidates","arguments":{}}, '
        '{"name":"observe_robot","arguments":{"robot_id":2}}, '
        '{"name":"select_executor","arguments":{"robot_id":2,"reason":"target is visible and actionable"}}, '
        '{"name":"report_failure","arguments":{"failure_code":"target_not_visible","reason":"no robot sees it"}}.'
    )
    if evidence_precollected:
        prompt += (
            " The collection phase for every known robot has already completed before this decision. Successful "
            "observations, relevant metadata, candidate validation evidence, and any collection errors are in the "
            "input. Do not request any additional observation or evidence tools. "
            "Decide now by calling select_executor for a verified candidate or report_failure when none can execute."
        )
    return prompt


def relay_user_message(
    task: str,
    task_intent: dict[str, Any],
    known_robot_ids: list[int],
    primary_robot_id: int,
    initial_summaries: list[dict[str, Any]],
    global_scene_summary: dict[str, Any] | None = None,
    candidate_evaluation: dict[str, Any] | None = None,
    observation_errors: dict[int, str] | None = None,
    evidence_precollected: bool = False,
) -> dict[str, Any]:
    observed_robot_ids = {
        summary["robot_id"]
        for summary in initial_summaries
        if isinstance(summary, dict)
        and isinstance(summary.get("robot_id"), int)
        and not isinstance(summary.get("robot_id"), bool)
        and summary["robot_id"] in known_robot_ids
    }

    content = {
        "task": task,
        "task_intent": task_intent,
        "known_robot_ids": known_robot_ids,
        "primary_robot_id": primary_robot_id,
        "already_observed": initial_summaries,
        "known_visibility": _known_visibility(initial_summaries),
        "global_scene_summary": global_scene_summary or {},
        "candidate_evaluation": candidate_evaluation or {},
        "evidence_collection_status": (
            "all_known_agents_collection_attempted_with_errors"
            if evidence_precollected and observation_errors
            else "all_known_agents_collected"
            if evidence_precollected
            else "on_demand"
        ),
        "available_evidence_tools": (
            [] if evidence_precollected else ["inspect_global_scene", "evaluate_executor_candidates"]
        ),
        "available_decision_tools": ["select_executor", "report_failure"],
        **_relay_state(
            known_robot_ids,
            observed_robot_ids,
            observation_attempt_counts=_initial_observation_attempt_counts(initial_summaries, known_robot_ids),
            observation_errors=observation_errors,
        ),
    }
    message_content: list[dict[str, Any]] = [
        {"type": "text", "text": json.dumps(content, ensure_ascii=False)}
    ]
    if evidence_precollected:
        for summary in initial_summaries:
            image_path = summary.get("image_path") if isinstance(summary, dict) else None
            if not isinstance(image_path, str) or not image_path.strip():
                continue
            message_content.extend(
                [
                    {
                        "type": "text",
                        "text": f"Precollected first-person view for robot_id={summary.get('robot_id')}",
                    },
                    {"type": "image", "image": image_path},
                ]
            )
    return {
        "role": "user",
        "content": message_content,
    }


def _assistant_json_message(call: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": json.dumps(call, ensure_ascii=False),
    }


def _initial_observation_attempt_counts(
    summaries: list[dict[str, Any]],
    known_robot_ids: list[int],
) -> dict[int, int]:
    counts: dict[int, int] = {}
    known = set(known_robot_ids)
    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        robot_id = summary.get("robot_id")
        if isinstance(robot_id, int) and not isinstance(robot_id, bool) and robot_id in known:
            counts[robot_id] = counts.get(robot_id, 0) + 1
    return counts


def _known_visibility(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    visibility: list[dict[str, Any]] = []
    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        visibility.append(
            {
                "robot_id": summary.get("robot_id"),
                "robot_name": summary.get("robot_name"),
                "is_primary": bool(summary.get("is_primary")),
                "visible_objects": summary.get("visible_objects", []),
                "held_object_type": summary.get("held_object_type"),
                "inventory_object_types": summary.get("inventory_object_types", []),
            }
        )
    return visibility


def _relay_state(
    known_robot_ids: list[int],
    observed_robot_ids: set[int],
    *,
    observation_attempt_counts: dict[int, int] | None = None,
    observation_errors: dict[int, str] | None = None,
) -> dict[str, Any]:
    known = list(dict.fromkeys(known_robot_ids))
    observed = [robot_id for robot_id in known if robot_id in observed_robot_ids]
    unobserved = [robot_id for robot_id in known if robot_id not in observed_robot_ids]
    counts = observation_attempt_counts or {}
    return {
        "known_robot_ids": known,
        "observed_robot_ids": observed,
        "successfully_observed_robot_ids": observed,
        "unobserved_robot_ids": unobserved,
        "visibility_unknown_robot_ids": unobserved,
        "observation_attempt_counts": {str(robot_id): counts.get(robot_id, 0) for robot_id in known},
        "last_observation_errors": {
            str(robot_id): error
            for robot_id, error in sorted((observation_errors or {}).items())
            if robot_id in known
        },
    }


def _schema_as_arguments(value: Any) -> bool:
    if isinstance(value, dict):
        if SCHEMA_ARGUMENT_KEYS.intersection(value):
            return True
        return any(_schema_as_arguments(item) for item in value.values())
    if isinstance(value, list):
        return any(_schema_as_arguments(item) for item in value)
    return False


def _valid_observe_calls(
    known_robot_ids: list[int], observed_robot_ids: set[int]
) -> list[dict[str, Any]]:
    return [
        {"name": "observe_robot", "arguments": {"robot_id": robot_id}}
        for robot_id in known_robot_ids
    ]


def _recommended_observe_calls(
    known_robot_ids: list[int],
    observed_robot_ids: set[int],
    observation_errors: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    errors = observation_errors or {}
    recommended_robot_ids = [
        robot_id
        for robot_id in known_robot_ids
        if robot_id not in observed_robot_ids and robot_id not in errors
    ]
    return [
        {"name": "observe_robot", "arguments": {"robot_id": robot_id}}
        for robot_id in recommended_robot_ids
    ]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _observation_fingerprint(summary: dict[str, Any]) -> str:
    visible_objects = summary.get("visible_objects", [])
    if isinstance(visible_objects, list):
        visible_objects = sorted(_canonical_json(item) for item in visible_objects)
    else:
        visible_objects = []
    inventory = summary.get("inventory_object_types", [])
    if isinstance(inventory, list):
        inventory = sorted(str(item) for item in inventory)
    else:
        inventory = []
    return _canonical_json(
        {
            "held_object_type": summary.get("held_object_type"),
            "inventory_object_types": inventory,
            "visible_objects": visible_objects,
        }
    )


def _initial_observation_fingerprints(
    summaries: list[dict[str, Any]],
    known_robot_ids: list[int],
) -> dict[int, str]:
    fingerprints: dict[int, str] = {}
    known = set(known_robot_ids)
    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        robot_id = summary.get("robot_id")
        if isinstance(robot_id, int) and not isinstance(robot_id, bool) and robot_id in known:
            fingerprints[robot_id] = _observation_fingerprint(summary)
    return fingerprints


def _response_with_state(
    payload: dict[str, Any],
    known_robot_ids: list[int],
    observed_robot_ids: set[int],
    *,
    observation_attempt_counts: dict[int, int] | None = None,
    observation_errors: dict[int, str] | None = None,
) -> dict[str, Any]:
    return {
        **payload,
        **_relay_state(
            known_robot_ids,
            observed_robot_ids,
            observation_attempt_counts=observation_attempt_counts,
            observation_errors=observation_errors,
        ),
        "recommended_observe_calls": _recommended_observe_calls(
            known_robot_ids,
            observed_robot_ids,
            observation_errors,
        ),
    }


def _relay_tools_without(suppressed_tool_names: set[str]) -> list[dict[str, Any]]:
    if not suppressed_tool_names:
        return RELAY_TOOLS
    return [
        tool
        for tool in RELAY_TOOLS
        if tool.get("function", {}).get("name") not in suppressed_tool_names
    ]


def _generate_relay_message(
    backend: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> str:
    parameters = inspect.signature(backend.generate_messages).parameters
    if "tools" in parameters:
        return backend.generate_messages(messages, tools=tools, deterministic=True).strip()
    return backend.generate_messages(messages, deterministic=True).strip()


def _tool_feedback_message(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    instruction = "Return the next allowed tool call as one JSON object with name and arguments."
    if tool_name in {"inspect_global_scene", "evaluate_executor_candidates"}:
        instruction = (
            "Use this evidence to select a verified executor, request fresher observations only when needed, "
            "or report failure if no executor can satisfy the task. Return one JSON object with name and arguments."
        )
    if payload.get("error_code") == "no_new_evidence_for_repeated_global_inspect":
        instruction = (
            "Do not call inspect_global_scene again until new observations are gathered. Use "
            "evaluate_executor_candidates, observe one of recommended_observe_calls, select a proven executor, "
            "or report failure if enough evidence has been gathered. Return one JSON object with name and arguments."
        )
    if payload.get("error_code") == "no_new_evidence_for_repeated_candidate_evaluation":
        instruction = (
            "Do not call evaluate_executor_candidates again until new observations are gathered. Observe one of "
            "recommended_observe_calls, select a proven executor, or report failure if enough evidence has been "
            "gathered. Return one JSON object with name and arguments."
        )
    if payload.get("error_code") == "no_new_evidence_for_repeated_observe":
        instruction = (
            "Do not observe the same robot again unless validation explicitly requires that refresh. "
            "Use recommended_observe_calls for visibility-unknown robots, select a proven executor, "
            "or report failure if enough evidence has been gathered. Return one JSON object with name and arguments."
        )
    feedback = {
        "tool_response": {
            "name": tool_name,
            **payload,
        },
        "instruction": instruction,
    }
    return {
        "role": "user",
        "content": [{"type": "text", "text": json.dumps(feedback, ensure_ascii=False)}],
    }


def run_relay_agent(
    backend: Any,
    *,
    task: str,
    task_intent: dict[str, Any],
    known_robot_ids: list[int],
    primary_robot_id: int,
    initial_summaries: list[dict[str, Any]],
    global_scene_summary: dict[str, Any] | None = None,
    initial_candidate_evaluation: dict[str, Any] | None = None,
    initial_observation_errors: dict[int, str] | None = None,
    evidence_precollected: bool = False,
    observe_robot: Callable[[int], dict[str, Any]],
    inspect_global_scene: Callable[[], dict[str, Any]] | None = None,
    evaluate_executor_candidates: Callable[[], dict[str, Any]] | None = None,
    validate_executor: Callable[[int], tuple[bool, str]],
    validate_failure: Callable[[str, str], tuple[bool, str, str]],
    config: RelayAgentConfig | None = None,
) -> dict[str, Any]:
    config = config or RelayAgentConfig()
    observed_robot_ids = {
        summary["robot_id"]
        for summary in initial_summaries
        if isinstance(summary, dict)
        and isinstance(summary.get("robot_id"), int)
        and not isinstance(summary.get("robot_id"), bool)
        and summary["robot_id"] in known_robot_ids
    }
    observation_errors: dict[int, str] = dict(initial_observation_errors or {})
    observation_attempt_counts = _initial_observation_attempt_counts(initial_summaries, known_robot_ids)
    last_observation_fingerprints = _initial_observation_fingerprints(initial_summaries, known_robot_ids)
    last_global_scene_summary = global_scene_summary or {}
    last_global_scene_fingerprint = _canonical_json(last_global_scene_summary)
    global_scene_inspect_count = 0
    last_candidate_evaluation: dict[str, Any] = dict(initial_candidate_evaluation or {})
    last_candidate_evaluation_fingerprint = (
        _canonical_json(last_candidate_evaluation) if last_candidate_evaluation else ""
    )
    candidate_evaluation_count = 1 if last_candidate_evaluation else 0
    suppressed_tool_names: set[str] = set()
    if evidence_precollected:
        suppressed_tool_names.update(
            {"inspect_global_scene", "evaluate_executor_candidates", "observe_robot"}
        )
    trace: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": relay_system_prompt(evidence_precollected)},
        relay_user_message(
            task,
            task_intent,
            known_robot_ids,
            primary_robot_id,
            initial_summaries,
            last_global_scene_summary,
            last_candidate_evaluation,
            observation_errors,
            evidence_precollected,
        ),
    ]

    def response_with_state(payload: dict[str, Any]) -> dict[str, Any]:
        return _response_with_state(
            payload,
            known_robot_ids,
            observed_robot_ids,
            observation_attempt_counts=observation_attempt_counts,
            observation_errors=observation_errors,
        )

    def final_state() -> dict[str, Any]:
        return _relay_state(
            known_robot_ids,
            observed_robot_ids,
            observation_attempt_counts=observation_attempt_counts,
            observation_errors=observation_errors,
        )

    def result_metadata() -> dict[str, Any]:
        candidate_scores = last_candidate_evaluation.get("candidate_scores")
        candidate_executor_robot_ids = last_candidate_evaluation.get("candidate_executor_robot_ids")
        metadata = {
            "global_scene_summary": last_global_scene_summary,
            "candidate_executor_robot_ids": (
                candidate_executor_robot_ids if isinstance(candidate_executor_robot_ids, list) else []
            ),
            "candidate_scores": candidate_scores if isinstance(candidate_scores, list) else [],
            "selection_policy": "llm_tool_calling_with_hard_validation",
            "evidence_collection_status": (
                "all_known_agents_collection_attempted_with_errors"
                if evidence_precollected and observation_errors
                else "all_known_agents_collected"
                if evidence_precollected
                else "on_demand"
            ),
            "agent_tool_trace": trace,
        }
        if last_candidate_evaluation:
            metadata["candidate_evaluation"] = last_candidate_evaluation
        return metadata

    def apply_candidate_evaluation(refreshed_evaluation: dict[str, Any]) -> dict[str, Any]:
        nonlocal last_candidate_evaluation, last_candidate_evaluation_fingerprint
        last_candidate_evaluation = refreshed_evaluation
        last_candidate_evaluation_fingerprint = _canonical_json(refreshed_evaluation)
        executable_robot_ids = refreshed_evaluation.get("candidate_executor_robot_ids")
        if isinstance(executable_robot_ids, list) and executable_robot_ids:
            suppressed_tool_names.update({
                "inspect_global_scene",
                "evaluate_executor_candidates",
                "observe_robot",
            })
        return {
            "candidate_evaluation": refreshed_evaluation,
            "candidate_executor_robot_ids": executable_robot_ids if isinstance(executable_robot_ids, list) else [],
            "candidate_scores": refreshed_evaluation.get("candidate_scores", []),
            "selection_policy": "llm_tool_calling_with_hard_validation",
            "suppressed_tool_names": sorted(suppressed_tool_names),
        }

    def terminal_failure_result(agent_reason: str) -> dict[str, Any] | None:
        unresolved_robot_ids = [
            robot_id
            for robot_id in known_robot_ids
            if robot_id not in observed_robot_ids and robot_id not in observation_errors
        ]
        candidate_executor_robot_ids = last_candidate_evaluation.get("candidate_executor_robot_ids")
        if (
            unresolved_robot_ids
            or not last_candidate_evaluation
            or not isinstance(candidate_executor_robot_ids, list)
            or candidate_executor_robot_ids
        ):
            return None

        accepted, failure_code, reason = validate_failure("target_not_visible", agent_reason)
        if not accepted:
            return None
        return {
            "status": "needs_upstream_planning",
            "strategy": "deterministic_terminal_fallback",
            "failure_code": failure_code,
            "reason": reason,
            "agent_reason": agent_reason,
            **final_state(),
            "observation_errors": observation_errors,
            "trace": trace,
            **result_metadata(),
        }

    for turn in range(1, config.max_turns + 1):
        raw_output = _generate_relay_message(backend, messages, _relay_tools_without(suppressed_tool_names))
        try:
            call = parse_relay_tool_call(raw_output)
        except ValueError as exc:
            response = response_with_state(
                {
                    "ok": False,
                    "error_code": "invalid_json_tool_call",
                    "error": str(exc),
                    "expected_format": {"name": "observe_robot", "arguments": {"robot_id": 2}},
                    "valid_calls": _valid_observe_calls(known_robot_ids, observed_robot_ids),
                }
            )
            trace.append(
                {
                    "turn": turn,
                    "status": "protocol_error",
                    "reason": str(exc),
                    "raw_output": raw_output[:500],
                    "response": response,
                }
            )
            messages.append({"role": "assistant", "content": raw_output})
            messages.append(_tool_feedback_message("protocol_error", response))
            continue

        name = call["name"]
        arguments = call.get("arguments", {})
        trace_entry: dict[str, Any] = {"turn": turn, "tool": name, "arguments": arguments}
        messages.append(_assistant_json_message(call))

        if _schema_as_arguments(arguments):
            response = response_with_state(
                {
                    "ok": False,
                    "error_code": "arguments_are_schema",
                    "error": "arguments contains a JSON Schema; provide concrete argument values instead",
                    "valid_calls": _valid_observe_calls(known_robot_ids, observed_robot_ids),
                }
            )
            trace_entry["response"] = response
            trace.append(trace_entry)
            messages.append(_tool_feedback_message(name, response))
            continue

        if name in suppressed_tool_names:
            response = response_with_state(
                {
                    "ok": False,
                    "error_code": "tool_temporarily_suppressed_after_no_progress",
                    "error": f"{name} is temporarily suppressed because it repeated without new evidence",
                    "suppressed_tool_names": sorted(suppressed_tool_names),
                    "recommended_next_calls": [
                        {"name": "evaluate_executor_candidates", "arguments": {}}
                    ] if name == "inspect_global_scene" else [],
                }
            )
            trace_entry["response"] = response
            trace.append(trace_entry)
            messages.append(_tool_feedback_message(name, response))
            continue

        if name == "inspect_global_scene":
            if inspect_global_scene is None:
                response = response_with_state(
                    {
                        "ok": False,
                        "error_code": "tool_unavailable",
                        "error": "inspect_global_scene is not available in this run",
                    }
                )
            else:
                global_scene_inspect_count += 1
                refreshed_summary = inspect_global_scene()
                fingerprint = _canonical_json(refreshed_summary)
                scene_changed = fingerprint != last_global_scene_fingerprint
                last_global_scene_summary = refreshed_summary
                last_global_scene_fingerprint = fingerprint
                response = {
                    "ok": True,
                    "global_scene_summary": last_global_scene_summary,
                    "global_scene_changed": scene_changed,
                    "global_scene_inspect_count": global_scene_inspect_count,
                }
                if global_scene_inspect_count > 1 and not scene_changed:
                    suppressed_tool_names.add("inspect_global_scene")
                    response = {
                        **response,
                        "ok": False,
                        "error_code": "no_new_evidence_for_repeated_global_inspect",
                        "error": (
                            "inspect_global_scene returned no new evidence; evaluate executor candidates, "
                            "observe only if fresh evidence is needed, select a proven executor, or report failure"
                        ),
                        "recommended_next_calls": [
                            {"name": "evaluate_executor_candidates", "arguments": {}}
                        ],
                        "suppressed_tool_names": sorted(suppressed_tool_names),
                    }
                response = response_with_state(response)
            trace_entry["response"] = response
            trace.append(trace_entry)
            messages.append(_tool_feedback_message(name, response))
            continue

        if name == "evaluate_executor_candidates":
            if evaluate_executor_candidates is None:
                response = response_with_state(
                    {
                        "ok": False,
                        "error_code": "tool_unavailable",
                        "error": "evaluate_executor_candidates is not available in this run",
                    }
                )
            else:
                candidate_evaluation_count += 1
                refreshed_evaluation = evaluate_executor_candidates()
                fingerprint = _canonical_json(refreshed_evaluation)
                evaluation_changed = fingerprint != last_candidate_evaluation_fingerprint
                metadata = apply_candidate_evaluation(refreshed_evaluation)
                response = {
                    "ok": True,
                    **last_candidate_evaluation,
                    "candidate_evaluation_changed": evaluation_changed,
                    "candidate_evaluation_count": candidate_evaluation_count,
                    **metadata,
                }
                unknown_robot_ids = [
                    robot_id
                    for robot_id in known_robot_ids
                    if robot_id not in observed_robot_ids and robot_id not in observation_errors
                ]
                executable_robot_ids = refreshed_evaluation.get("candidate_executor_robot_ids")
                if (
                    candidate_evaluation_count > 1
                    and not evaluation_changed
                    and unknown_robot_ids
                    and not executable_robot_ids
                ):
                    suppressed_tool_names.add("evaluate_executor_candidates")
                    response = {
                        **response,
                        "ok": False,
                        "error_code": "no_new_evidence_for_repeated_candidate_evaluation",
                        "error": (
                            "candidate evaluation returned no executable robot and no new evidence while robot "
                            "visibility is still unknown; observe unknown robots before evaluating again"
                        ),
                        "suppressed_tool_names": sorted(suppressed_tool_names),
                    }
                response = response_with_state(response)
            trace_entry["response"] = response
            trace.append(trace_entry)
            terminal_result = terminal_failure_result(
                "all known robots have resolved observations and candidate evaluation found no valid executor"
            )
            if terminal_result is not None:
                return terminal_result
            messages.append(_tool_feedback_message(name, response))
            continue

        if name == "observe_robot":
            robot_id = arguments.get("robot_id")
            if (not isinstance(robot_id, int) or isinstance(robot_id, bool)) and suppressed_tool_names:
                repair_candidates = [
                    candidate_id
                    for candidate_id in known_robot_ids
                    if candidate_id not in observed_robot_ids and candidate_id not in observation_errors
                ]
                if repair_candidates:
                    robot_id = repair_candidates[0]
                    arguments = {**arguments, "robot_id": robot_id}
                    trace_entry["arguments"] = arguments
                    trace_entry["argument_repair"] = {
                        "robot_id": robot_id,
                        "reason": "observe_robot omitted robot_id after no-progress tools were suppressed",
                    }
            if not isinstance(robot_id, int) or isinstance(robot_id, bool):
                response = {
                    "ok": False,
                    "error_code": "invalid_robot_id",
                    "error": "observe_robot.robot_id must be an integer",
                    "valid_calls": _valid_observe_calls(known_robot_ids, observed_robot_ids),
                }
            elif robot_id not in known_robot_ids:
                response = {
                    "ok": False,
                    "error_code": "unknown_robot_id",
                    "error": f"unknown robot_id {robot_id!r}",
                    "valid_calls": _valid_observe_calls(known_robot_ids, observed_robot_ids),
                }
            else:
                observation_attempt_counts[robot_id] = observation_attempt_counts.get(robot_id, 0) + 1
                try:
                    summary = observe_robot(robot_id)
                    observed_robot_ids.add(robot_id)
                    observation_errors.pop(robot_id, None)
                    fingerprint = _observation_fingerprint(summary)
                    previous_fingerprint = last_observation_fingerprints.get(robot_id)
                    observation_changed = previous_fingerprint is None or previous_fingerprint != fingerprint
                    last_observation_fingerprints[robot_id] = fingerprint
                    other_unknown_robot_ids = [
                        candidate_id
                        for candidate_id in known_robot_ids
                        if candidate_id != robot_id
                        and candidate_id not in observed_robot_ids
                        and candidate_id not in observation_errors
                    ]
                    response = {
                        "ok": True,
                        "observation": summary,
                        "observation_attempt_count": observation_attempt_counts[robot_id],
                        "observation_changed": observation_changed,
                    }
                    if observation_changed:
                        suppressed_tool_names.clear()
                    if evaluate_executor_candidates is not None:
                        response.update(apply_candidate_evaluation(evaluate_executor_candidates()))
                    if not observation_changed and other_unknown_robot_ids:
                        response = {
                            **response,
                            "ok": False,
                            "error_code": "no_new_evidence_for_repeated_observe",
                            "error": (
                                f"robot {robot_id} observation did not change; observe visibility-unknown "
                                "robots before re-observing it again"
                            ),
                            "repeated_robot_id": robot_id,
                            "recommended_observe_calls": [
                                {"name": "observe_robot", "arguments": {"robot_id": candidate_id}}
                                for candidate_id in other_unknown_robot_ids
                            ],
                            "valid_calls": _valid_observe_calls(known_robot_ids, observed_robot_ids),
                        }
                except Exception as exc:  # The HTTP adapter supplies the actionable error text.
                    observation_errors[robot_id] = str(exc)
                    response = {
                        "ok": False,
                        "error_code": "observation_failed",
                        "error": str(exc),
                        "robot_id": robot_id,
                        "observation_attempt_count": observation_attempt_counts[robot_id],
                        "valid_calls": _valid_observe_calls(known_robot_ids, observed_robot_ids),
                    }
            response = response_with_state(response)
            trace_entry["response"] = response
            trace.append(trace_entry)
            terminal_result = terminal_failure_result(
                "all known robots have resolved observations and candidate evaluation found no valid executor"
            )
            if terminal_result is not None:
                return terminal_result
            messages.append(_tool_feedback_message(name, response))
            continue

        if name == "select_executor":
            robot_id = arguments.get("robot_id")
            agent_reason = arguments.get("reason")
            candidate_robot_ids = last_candidate_evaluation.get("candidate_executor_robot_ids")
            if (
                (not isinstance(robot_id, int) or isinstance(robot_id, bool))
                and isinstance(candidate_robot_ids, list)
                and len(candidate_robot_ids) == 1
                and isinstance(candidate_robot_ids[0], int)
            ):
                robot_id = candidate_robot_ids[0]
                if not isinstance(agent_reason, str) or not agent_reason.strip():
                    agent_reason = "only executable candidate from relay evidence"
                arguments = {**arguments, "robot_id": robot_id, "reason": agent_reason}
                trace_entry["arguments"] = arguments
                trace_entry["argument_repair"] = {
                    "robot_id": robot_id,
                    "reason": "select_executor omitted robot_id when exactly one executable candidate was available",
                }
            if not isinstance(robot_id, int) or isinstance(robot_id, bool):
                response = {
                    "ok": False,
                    "error_code": "invalid_robot_id",
                    "error": "select_executor.robot_id must be an integer",
                }
            elif robot_id not in known_robot_ids or robot_id not in observed_robot_ids:
                response = {
                    "ok": False,
                    "error_code": "executor_not_observed",
                    "error": "executor must be a known, observed robot",
                }
            elif not isinstance(agent_reason, str) or not agent_reason.strip():
                response = {
                    "ok": False,
                    "error_code": "invalid_reason",
                    "error": "select_executor.reason must be a non-empty string",
                }
            else:
                accepted, reason = validate_executor(robot_id)
                response = {"ok": accepted, "validation": reason}
                if accepted:
                    response = response_with_state(response)
                    trace_entry["response"] = response
                    trace.append(trace_entry)
                    agent_reason = agent_reason.strip()
                    return {
                        "status": "executor_selected",
                        "strategy": "agent",
                        "executor_robot_id": robot_id,
                        "reason": f"{agent_reason} (validated: {reason})",
                        "validation_reason": reason,
                        "agent_reason": agent_reason,
                        **final_state(),
                        "observation_errors": observation_errors,
                        "trace": trace,
                        **result_metadata(),
                    }
            response = response_with_state(response)
            trace_entry["response"] = response
            trace.append(trace_entry)
            messages.append(_tool_feedback_message(name, response))
            continue

        if name == "report_failure":
            failure_code = arguments.get("failure_code")
            agent_reason = arguments.get("reason")
            unobserved = [
                robot_id
                for robot_id in known_robot_ids
                if robot_id not in observed_robot_ids and robot_id not in observation_errors
            ]
            if not isinstance(failure_code, str) or failure_code not in ALLOWED_FAILURE_CODES:
                response = {
                    "ok": False,
                    "error_code": "invalid_failure_code",
                    "error": "report_failure.failure_code must be an allowed value",
                    "allowed_failure_codes": sorted(ALLOWED_FAILURE_CODES),
                }
                response = response_with_state(response)
                trace_entry["response"] = response
                trace.append(trace_entry)
                messages.append(_tool_feedback_message(name, response))
                continue

            if not isinstance(agent_reason, str) or not agent_reason.strip():
                response = {
                    "ok": False,
                    "error_code": "invalid_reason",
                    "error": "report_failure.reason must be a non-empty string",
                }
                response = response_with_state(response)
                trace_entry["response"] = response
                trace.append(trace_entry)
                messages.append(_tool_feedback_message(name, response))
                continue

            if failure_code != "unsupported_task" and unobserved:
                response = {
                    "ok": False,
                    "error_code": "robots_not_observed",
                    "error": "observe robots whose visibility is still unknown before reporting failure",
                    "valid_calls": _valid_observe_calls(known_robot_ids, observed_robot_ids),
                }
                response = response_with_state(response)
                trace_entry["response"] = response
                trace.append(trace_entry)
                messages.append(_tool_feedback_message(name, response))
                continue
            accepted, normalized_code, normalized_reason = validate_failure(failure_code, agent_reason.strip())
            response = {"ok": accepted, "failure_code": normalized_code, "reason": normalized_reason}
            response = response_with_state(response)
            if not accepted:
                trace_entry["response"] = response
                trace.append(trace_entry)
                messages.append(_tool_feedback_message(name, response))
                continue
            trace_entry["response"] = response
            trace.append(trace_entry)
            return {
                "status": "needs_upstream_planning",
                "strategy": "agent",
                "failure_code": normalized_code,
                "reason": normalized_reason,
                "agent_reason": agent_reason.strip(),
                **final_state(),
                "observation_errors": observation_errors,
                "trace": trace,
                **result_metadata(),
            }

        response = response_with_state(
            {
                "ok": False,
                "error_code": "unsupported_relay_tool",
                "error": f"unsupported relay tool {name!r}",
            }
        )
        trace_entry["response"] = response
        trace.append(trace_entry)
        messages.append(_tool_feedback_message(name, response))

    terminal_result = terminal_failure_result(
        "relay agent exhausted its turn budget after terminal candidate evidence was available"
    )
    if terminal_result is not None:
        return terminal_result

    return {
        "status": "needs_upstream_planning",
        "strategy": "agent",
        "failure_code": "agent_max_turns_exceeded",
        "reason": f"relay agent did not reach a valid decision within {config.max_turns} turns",
        **final_state(),
        "observation_errors": observation_errors,
        "trace": trace,
        **result_metadata(),
    }
