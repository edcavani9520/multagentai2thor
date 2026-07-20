from __future__ import annotations

import contextlib
import io
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo import auto_scene_actions as auto_scene_actions_module
from demo.auto_scene_actions import (
    add_execute_response_result,
    agent_observations_summary,
    build_object_visibility_map,
    choose_relay_executor,
    coordination_result_for_plan,
    execute_extract_task_intent_tool,
    execute_response_failure_reason,
    evaluate_relay_executor_candidates,
    expand_put_object_intent_preconditions,
    extract_agent_observations,
    extract_requested_action,
    extract_requested_object_type,
    ground_semantic_plan,
    held_object_debug_from_observation,
    held_object_type_from_observation,
    merge_execute_result_into_observation,
    parse_qwen_tool_call,
    object_visibility_summary,
    pickup_step_already_satisfied,
    object_state_step_already_satisfied,
    relay_result_for_held_put_step,
    repair_semantic_placeholders_from_step_intent,
    repair_redundant_pickup_for_held_put,
    save_qwen_raw_output,
    summarize_execute_response,
    validate_action_affordances,
    validate_action_intent_consistency,
    validate_action_state_preconditions,
    validate_goal_consistency,
    validate_intent_steps_consistency,
    validate_put_object_goal_consistency,
    validate_executor_plan_or_failure,
    validate_task_intent_consistency,
    validate_task_intent_tool_call,
)
from demo.plan_media import (
    build_execution_payload,
    parse_args,
    parse_native_planning_output,
    parse_semantic_planning_output,
    question_prompt,
    native_planning_prompt,
    semantic_planning_prompt,
    plan_only_document,
    send_actions,
)


def fake_task_intent_tool_call(args, available_types):
    tool_call = {"name": "extract_task_intent", "arguments": {"task": args.task}}
    return (
        tool_call,
        execute_extract_task_intent_tool(args.task, available_types),
        validate_task_intent_tool_call(tool_call, args.task),
    )


class ParseNativePlanTest(unittest.TestCase):
    def test_returns_executable_plan(self) -> None:
        plan = [
            {"action": "PickupObject", "objectId": "Apple|UNKNOWN"},
            {"action": "RotateRight"},
        ]
        output = json.dumps({"task": "move an apple", "plan": plan})

        self.assertEqual(parse_native_planning_output(output)["plan"], plan)

    def test_rejects_legacy_top_level_actions(self) -> None:
        output = json.dumps({"task": "move", "actions": [{"action": "MoveAhead"}]})
        with self.assertRaisesRegex(ValueError, "top-level"):
            parse_native_planning_output(output)

    def test_rejects_missing_plan(self) -> None:
        with self.assertRaisesRegex(ValueError, "plan"):
            parse_native_planning_output(json.dumps({"task": "x"}))

    def test_rejects_empty_plan(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            parse_native_planning_output(json.dumps({"task": "x", "plan": []}))

    def test_rejects_non_list_plan(self) -> None:
        with self.assertRaisesRegex(ValueError, "plan"):
            parse_native_planning_output(json.dumps({"task": "x", "plan": {"action": "MoveAhead"}}))

    def test_rejects_non_object_plan_item(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be an object"):
            parse_native_planning_output(json.dumps({"task": "x", "plan": ["MoveAhead"]}))

    def test_rejects_missing_action_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "action"):
            parse_native_planning_output(json.dumps({"task": "x", "plan": [{"objectId": "Apple|UNKNOWN"}]}))

    def test_rejects_invalid_json(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid JSON"):
            parse_native_planning_output("Plan: 1. MoveAhead")

    def test_plan_only_document_contains_no_actions_key(self) -> None:
        plan = [{"action": "MoveAhead"}]
        self.assertEqual(plan_only_document(plan), {"plan": plan})

    def test_plan_only_replaces_actions_only_cli(self) -> None:
        args = parse_args(["--media", "scene.jpg", "--plan-only"])
        self.assertTrue(args.plan_only)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--media", "scene.jpg", "--actions-only"])


class SendActionsTest(unittest.TestCase):
    def test_maps_plan_to_actions_at_http_boundary(self) -> None:
        captured = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                captured["content_type"] = self.headers.get("Content-Type")
                captured["body"] = self.rfile.read(length).decode("utf-8")
                self.send_response(204)
                self.end_headers()

            def log_message(self, format: str, *args: object) -> None:
                return

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.handle_request)
        thread.start()
        try:
            plan = [
                {"action": "MoveAhead"},
                {"action": "PickupObject", "objectId": "Apple|UNKNOWN"},
            ]
            parsed = {"task": "move an apple", "plan": plan}
            payload = build_execution_payload(parsed, "task-1")
            send_actions(f"http://127.0.0.1:{server.server_port}/actions", payload, 2.0)
        finally:
            thread.join(timeout=5)
            server.server_close()

        self.assertIn("application/json", captured["content_type"])
        body = json.loads(captured["body"])
        self.assertEqual(body["task_id"], "task-1")
        self.assertEqual(body["task"], "move an apple")
        self.assertEqual(body["plan"], plan)
        self.assertIs(body["stop_on_failure"], False)
        self.assertEqual(body["actions"], plan)


class AutoSceneActionsOutputTest(unittest.TestCase):
    def test_default_execute_response_is_summarized_not_included(self) -> None:
        result: dict[str, object] = {}
        response_text = json.dumps(
            {
                "success": True,
                "state": {
                    "sceneName": "FloorPlan1",
                    "objects": [{"id": "Egg|1"}, {"id": "Fridge|1"}],
                },
                "message": "ok",
            }
        )

        add_execute_response_result(
            result,
            response_text,
            include_response=False,
            save_response=False,
            output_dir=Path("/tmp"),
            task_id="task-1",
            action_count=2,
        )

        self.assertNotIn("execute_response", result)
        self.assertEqual(result["execute_response_summary"]["success"], True)
        self.assertEqual(result["execute_response_summary"]["sceneName"], "FloorPlan1")
        self.assertEqual(result["execute_response_summary"]["object_count"], 2)
        self.assertNotIn('"objects"', json.dumps(result["execute_response_summary"]))

    def test_include_execute_response_keeps_full_response(self) -> None:
        result: dict[str, object] = {}
        response = {"success": True, "state": {"objects": [{"id": "Egg|1"}]}}

        add_execute_response_result(
            result,
            json.dumps(response),
            include_response=True,
            save_response=False,
            output_dir=Path("/tmp"),
            task_id="task-1",
            action_count=1,
        )

        self.assertEqual(result["execute_response"], response)
        self.assertIn("execute_response_summary", result)

    def test_save_response_and_raw_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result: dict[str, object] = {}
            response_text = json.dumps({"success": True, "state": {"objects": []}})

            raw_path = save_qwen_raw_output(output_dir, "task-1", "raw qwen json")
            add_execute_response_result(
                result,
                response_text,
                include_response=False,
                save_response=True,
                output_dir=output_dir,
                task_id="task-1",
                action_count=1,
            )

            self.assertEqual(Path(raw_path).read_text(encoding="utf-8"), "raw qwen json")
            response_path = Path(result["execute_response_path"])
            self.assertEqual(response_path.read_text(encoding="utf-8"), response_text)
            self.assertTrue(response_path.name.endswith("_execute_response.json"))

    def test_execute_response_summary_truncates_text_response(self) -> None:
        summary = summarize_execute_response("x" * 600, action_count=3)

        self.assertEqual(summary["response_type"], "str")
        self.assertEqual(summary["action_count"], 3)
        self.assertLess(len(summary["text_preview"]), 510)

    def test_execute_response_failure_reason_detects_failed_status(self) -> None:
        self.assertIn("failed", execute_response_failure_reason({"status": "failed"}))

    def test_execute_response_failure_reason_detects_failed_result(self) -> None:
        reason = execute_response_failure_reason(
            {"status": "success", "results": [{"action": "PickupObject", "success": False, "error": "not reachable"}]}
        )

        self.assertIn("PickupObject", reason)
        self.assertIn("not reachable", reason)


class TaskIntentToolTest(unittest.TestCase):
    def test_parses_qwen_tool_call_block(self) -> None:
        output = '<tool_call>{"name":"extract_task_intent","arguments":{"task":"Pick up the apple."}}</tool_call>'

        self.assertEqual(
            parse_qwen_tool_call(output),
            {"name": "extract_task_intent", "arguments": {"task": "Pick up the apple."}},
        )

    def test_parses_openai_style_tool_call(self) -> None:
        output = json.dumps(
            {
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "extract_task_intent",
                            "arguments": json.dumps({"task": "Turn right."}),
                        },
                    }
                ]
            }
        )

        self.assertEqual(
            parse_qwen_tool_call(output),
            {"name": "extract_task_intent", "arguments": {"task": "Turn right."}},
        )

    def test_rejects_missing_tool_call(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid tool call"):
            parse_qwen_tool_call('{"answer":"no tool"}')

    def test_rejects_qwen_xml_placeholder_tool_call(self) -> None:
        output = (
            "<tool_call><function=example_function_name>"
            "<parameter=example_parameter_1>value_1</parameter>"
            "</function></tool_call>"
        )

        with self.assertRaisesRegex(ValueError, "valid tool call"):
            parse_qwen_tool_call(output)


    def test_tool_call_validation_warns_when_task_argument_missing(self) -> None:
        validation = validate_task_intent_tool_call(
            {"name": "extract_task_intent", "arguments": {}},
            "Pick up the apple.",
        )

        self.assertEqual(validation["status"], "warning")
        self.assertIn("omitted required argument", validation["warnings"][0])

    def test_tool_call_validation_warns_when_task_argument_differs(self) -> None:
        validation = validate_task_intent_tool_call(
            {"name": "extract_task_intent", "arguments": {"task": "Pick up the tomato."}},
            "Pick up the apple.",
        )

        self.assertEqual(validation["status"], "warning")
        self.assertIn("differs from original", validation["warnings"][0])

    def test_tool_call_validation_rejects_wrong_tool(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unexpected tool"):
            validate_task_intent_tool_call({"name": "other_tool", "arguments": {"task": "x"}}, "x")

    def test_local_tool_uses_original_task_intent(self) -> None:
        intent = execute_extract_task_intent_tool("Pick up the apple.", ["Tomato"])

        self.assertEqual(intent["requestedAction"], "PickupObject")
        self.assertEqual(intent["requestedObjectType"], "Apple")
        self.assertEqual(
            intent["intentSteps"],
            [{"order": 1, "action": "PickupObject", "objectType": "Apple", "targetType": None}],
        )

    def test_local_tool_extracts_multi_step_intent(self) -> None:
        intent = execute_extract_task_intent_tool("Open the fridge and pick up the apple.", ["Fridge", "Apple"])

        self.assertEqual(
            intent["intentSteps"],
            [
                {"order": 1, "action": "OpenObject", "objectType": "Fridge", "targetType": None},
                {"order": 2, "action": "PickupObject", "objectType": "Apple", "targetType": None},
            ],
        )

    def test_local_tool_resolves_it_for_put_object(self) -> None:
        intent = execute_extract_task_intent_tool("Pick up the apple and put it on the counter.", ["Apple", "CounterTop"])

        self.assertEqual(
            intent["intentSteps"],
            [
                {"order": 1, "action": "PickupObject", "objectType": "Apple", "targetType": None},
                {"order": 2, "action": "PutObject", "objectType": "Apple", "targetType": "CounterTop"},
            ],
        )

    def test_local_tool_extracts_navigation_sequence(self) -> None:
        intent = execute_extract_task_intent_tool("Turn right, move ahead, then open the cabinet.", ["Cabinet"])

        self.assertEqual(
            intent["intentSteps"],
            [
                {"order": 1, "action": "RotateRight", "objectType": None, "targetType": None},
                {"order": 2, "action": "MoveAhead", "objectType": None, "targetType": None},
                {"order": 3, "action": "OpenObject", "objectType": "Cabinet", "targetType": None},
            ],
        )

    def test_rejects_model_target_that_conflicts_with_task_intent(self) -> None:
        task_intent = {"requestedAction": "PickupObject", "requestedObjectType": "Apple"}
        semantic_plan = {"targetObjectType": "Tomato", "plan": [{"action": "PickupObject", "objectType": "Tomato"}]}

        with self.assertRaisesRegex(ValueError, "Tomato.*Apple"):
            validate_task_intent_consistency(task_intent, semantic_plan, check_action=False)

    def test_multi_step_verifier_allows_ordered_steps_with_navigation_between(self) -> None:
        task_intent = {
            "intentSteps": [
                {"order": 1, "action": "OpenObject", "objectType": "Fridge", "targetType": None},
                {"order": 2, "action": "PickupObject", "objectType": "Apple", "targetType": None},
            ]
        }
        semantic_plan = {
            "plan": [
                {"action": "OpenObject", "objectType": "Fridge", "targetType": None},
                {"action": "LookDown", "objectType": None, "targetType": None},
                {"action": "PickupObject", "objectType": "Apple", "targetType": None},
            ]
        }

        validate_intent_steps_consistency(task_intent, semantic_plan)

    def test_multi_step_verifier_rejects_missing_step(self) -> None:
        task_intent = {
            "intentSteps": [
                {"order": 1, "action": "OpenObject", "objectType": "Fridge", "targetType": None},
                {"order": 2, "action": "PickupObject", "objectType": "Apple", "targetType": None},
            ]
        }
        semantic_plan = {"plan": [{"action": "OpenObject", "objectType": "Fridge", "targetType": None}]}

        with self.assertRaisesRegex(ValueError, "missing intent step 2"):
            validate_intent_steps_consistency(task_intent, semantic_plan)

    def test_multi_step_verifier_rejects_wrong_object(self) -> None:
        task_intent = {"intentSteps": [{"order": 1, "action": "PickupObject", "objectType": "Apple", "targetType": None}]}
        semantic_plan = {"plan": [{"action": "PickupObject", "objectType": "Tomato", "targetType": None}]}

        with self.assertRaisesRegex(ValueError, "Tomato.*Apple"):
            validate_intent_steps_consistency(task_intent, semantic_plan)

    def test_multi_step_verifier_rejects_wrong_order(self) -> None:
        task_intent = {
            "intentSteps": [
                {"order": 1, "action": "OpenObject", "objectType": "Fridge", "targetType": None},
                {"order": 2, "action": "PickupObject", "objectType": "Apple", "targetType": None},
            ]
        }
        semantic_plan = {
            "plan": [
                {"action": "PickupObject", "objectType": "Apple", "targetType": None},
                {"action": "OpenObject", "objectType": "Fridge", "targetType": None},
            ]
        }

        with self.assertRaisesRegex(ValueError, "missing intent step 2"):
            validate_intent_steps_consistency(task_intent, semantic_plan)

    def test_run_outputs_tool_call_validation_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            probe = {
                "state": {
                    "sceneName": "FloorPlan1",
                    "objects": [{"id": "Apple|1", "type": "Apple", "visible": True, "pickupable": True}],
                },
                "image_base64": "eA==",
            }
            semantic_plan = {
                "task": "Pick up the apple.",
                "targetObjectType": "Apple",
                "needsGrounding": True,
                "observations": [],
                "plan": [{"action": "PickupObject", "objectType": "Apple", "targetType": None}],
            }

            def missing_argument_tool_call(args, available_types):
                tool_call = {"name": "extract_task_intent", "arguments": {}}
                return (
                    tool_call,
                    execute_extract_task_intent_tool(args.task, available_types),
                    validate_task_intent_tool_call(tool_call, args.task),
                )

            old_probe = auto_scene_actions_module.probe_scene
            old_generate = auto_scene_actions_module.generate_semantic_plan
            old_send = auto_scene_actions_module.send_actions
            old_tool_call = auto_scene_actions_module.generate_task_intent_tool_call
            auto_scene_actions_module.probe_scene = lambda *args, **kwargs: probe
            auto_scene_actions_module.generate_semantic_plan = lambda *args, **kwargs: ("{}", semantic_plan, None)
            auto_scene_actions_module.generate_task_intent_tool_call = missing_argument_tool_call
            auto_scene_actions_module.send_actions = lambda *args, **kwargs: json.dumps({"status": "success"})
            try:
                args = auto_scene_actions_module.parse_args(
                    [
                        "--execute-actions-url",
                        "http://127.0.0.1:1/execute_actions",
                        "--task",
                        "Pick up the apple.",
                        "--task-id",
                        "task-1",
                        "--output-dir",
                        temp_dir,
                        "--dry-run",
                    ]
                )
                stdout = io.StringIO()
                stderr = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    exit_code = auto_scene_actions_module.run(args)
            finally:
                auto_scene_actions_module.probe_scene = old_probe
                auto_scene_actions_module.generate_semantic_plan = old_generate
                auto_scene_actions_module.generate_task_intent_tool_call = old_tool_call
                auto_scene_actions_module.send_actions = old_send

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["task_intent_tool_call_validation"]["status"], "warning")
        self.assertIn("omitted required argument", output["task_intent_tool_call_validation"]["warnings"][0])


class ObjectVisibilityMapTest(unittest.TestCase):
    def setUp(self) -> None:
        self.old_task_intent_tool_call = auto_scene_actions_module.generate_task_intent_tool_call
        auto_scene_actions_module.generate_task_intent_tool_call = fake_task_intent_tool_call

    def tearDown(self) -> None:
        auto_scene_actions_module.generate_task_intent_tool_call = self.old_task_intent_tool_call

    def test_peer_visible_message_mentions_executor_in_relay_mode(self) -> None:
        visibility_map = build_object_visibility_map(
            [
                {
                    "agent_id": "robot_0",
                    "robot_id": 0,
                    "is_primary": True,
                    "objects": [{"id": "Egg|1", "type": "Egg", "visible": True, "pickupable": True}],
                },
                {
                    "agent_id": "robot_1",
                    "robot_id": 1,
                    "is_primary": False,
                    "objects": [{"id": "Apple|1", "type": "Apple", "visible": True, "pickupable": True}],
                },
            ]
        )
        semantic_plan = {"targetObjectType": "Apple", "plan": [{"action": "PickupObject", "objectType": "Apple"}]}

        local_result = coordination_result_for_plan("Pick up the apple.", semantic_plan, visibility_map)
        relay_result = coordination_result_for_plan(
            "Pick up the apple.",
            semantic_plan,
            visibility_map,
            relay_mode=True,
        )

        self.assertIn("refusing to execute locally", local_result["message"])
        self.assertIn("selecting peer robot 'robot_1' as executor", relay_result["message"])

    def test_single_agent_probe_becomes_agent_zero(self) -> None:
        probe = {
            "state": {
                "sceneName": "FloorPlan1",
                "objects": [{"id": "Egg|1", "type": "Egg", "visible": True, "pickupable": True}],
            },
            "image_base64": "eA==",
        }

        observations = extract_agent_observations(probe)
        visibility_map = build_object_visibility_map(observations)

        self.assertEqual(observations[0]["agent_id"], "robot_0")
        self.assertTrue(observations[0]["is_primary"])
        self.assertEqual(visibility_map["primary_agent_id"], "robot_0")
        self.assertEqual(visibility_map["objects_by_type"]["egg"]["visible_by_agent_ids"], ["robot_0"])

    def test_probe_response_with_state_selected_robot_id_uses_requested_primary_robot_id(self) -> None:
        probe = {
            "status": "success",
            "results": [
                {
                    "robot_id": 2,
                    "robot_name": "Robot2",
                    "image_base64": "eA==",
                }
            ],
            "state": {
                "sceneName": "FloorPlan1",
                "selected_robot_id": 2,
                "objects": [{"id": "Apple|2", "type": "Apple", "visible": True, "pickupable": True}],
            },
        }

        observations = extract_agent_observations(probe, primary_robot_id=2)

        self.assertEqual(observations[0]["robot_id"], 2)
        self.assertEqual(observations[0]["agent_id"], "robot_2")
        self.assertTrue(observations[0]["is_primary"])

    def test_peer_visible_target_is_reported(self) -> None:
        probe = {
            "events": [
                {
                    "agentId": "agent_0",
                    "metadata": {
                        "sceneName": "FloorPlan1",
                        "objects": [{"id": "Egg|1", "type": "Egg", "visible": True, "pickupable": True}],
                    },
                },
                {
                    "agentId": "agent_1",
                    "metadata": {
                        "sceneName": "FloorPlan1",
                        "objects": [{"id": "Apple|1", "type": "Apple", "visible": True, "pickupable": True}],
                    },
                },
            ]
        }
        semantic_plan = {"targetObjectType": "Apple", "plan": [{"action": "PickupObject", "objectType": "Apple"}]}

        observations = extract_agent_observations(probe)
        visibility_map = build_object_visibility_map(observations)
        result = coordination_result_for_plan("Pick up the apple.", semantic_plan, visibility_map)

        self.assertEqual(result["status"], "target_visible_by_peer")
        self.assertEqual(result["visible_peer_agent_ids"], ["robot_1"])

    def test_primary_agent_id_can_be_selected(self) -> None:
        probe = {
            "agents": [
                {"id": "agent_0", "objects": [{"id": "Egg|1", "type": "Egg", "visible": True}]},
                {"id": "agent_1", "objects": [{"id": "Apple|1", "type": "Apple", "visible": True}]},
            ]
        }

        observations = extract_agent_observations(probe, primary_robot_id=1)
        visibility_map = build_object_visibility_map(observations)

        self.assertEqual(visibility_map["primary_agent_id"], "robot_1")
        self.assertEqual(visibility_map["objects_by_type"]["apple"]["best_agent_id"], "robot_1")

    def test_visibility_summary_does_not_expand_objects(self) -> None:
        observations = extract_agent_observations(
            {
                "state": {
                    "objects": [
                        {"id": "Apple|1", "type": "Apple", "visible": True, "pickupable": True},
                        {"id": "Fridge|1", "type": "Fridge", "visible": False, "openable": True},
                    ],
                },
                "image_base64": "eA==",
            }
        )

        summary = object_visibility_summary(build_object_visibility_map(observations))
        summary_text = json.dumps(summary)

        self.assertEqual(summary["total_object_type_count"], 2)
        self.assertEqual(summary["visible_object_type_count"], 1)
        self.assertEqual(summary["hidden_object_type_count"], 1)
        self.assertEqual(summary["visible_object_types"][0]["object_type"], "Apple")
        self.assertNotIn("Apple|1", summary_text)
        self.assertNotIn("Fridge", summary_text)
        self.assertNotIn('"objects"', summary_text)
        self.assertNotIn('"affordances"', summary_text)
        self.assertNotIn('"object_types"', summary_text)

    def test_run_rejects_model_target_that_conflicts_with_task_intent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            probe = {
                "state": {
                    "sceneName": "FloorPlan1",
                    "objects": [{"id": "Tomato|1", "type": "Tomato", "visible": True, "pickupable": True}],
                },
                "image_base64": "eA==",
            }
            semantic_plan = {
                "task": "Pick up the apple.",
                "targetObjectType": "Tomato",
                "needsGrounding": True,
                "observations": [],
                "plan": [{"action": "PickupObject", "objectType": "Tomato", "targetType": None}],
            }
            old_probe = auto_scene_actions_module.probe_scene
            old_generate = auto_scene_actions_module.generate_semantic_plan
            old_send = auto_scene_actions_module.send_actions
            auto_scene_actions_module.probe_scene = lambda *args, **kwargs: probe
            auto_scene_actions_module.generate_semantic_plan = lambda *args, **kwargs: ("{}", semantic_plan, None)
            auto_scene_actions_module.send_actions = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not send"))
            try:
                args = auto_scene_actions_module.parse_args(
                    [
                        "--execute-actions-url",
                        "http://127.0.0.1:1/execute_actions",
                        "--task",
                        "Pick up the apple.",
                        "--task-id",
                        "task-1",
                        "--output-dir",
                        temp_dir,
                    ]
                )
                stdout = io.StringIO()
                stderr = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    exit_code = auto_scene_actions_module.run(args)
            finally:
                auto_scene_actions_module.probe_scene = old_probe
                auto_scene_actions_module.generate_semantic_plan = old_generate
                auto_scene_actions_module.send_actions = old_send

        self.assertEqual(exit_code, 0)
        self.assertIn("Tomato", stderr.getvalue())
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["coordination_result"]["status"], "task_intent_mismatch")
        self.assertEqual(output["task_intent"]["requestedAction"], "PickupObject")
        self.assertEqual(output["task_intent"]["requestedObjectType"], "Apple")
        self.assertEqual(output["task_intent_source"], "qwen_native_tool_call")
        self.assertNotIn("payload", output)

    def test_run_reports_peer_visible_target_without_payload_or_send(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            probe = {
                "events": [
                    {
                        "agentId": "agent_0",
                        "image_base64": "eA==",
                        "metadata": {"objects": [{"id": "Egg|1", "type": "Egg", "visible": True, "pickupable": True}]},
                    },
                    {
                        "agentId": "agent_1",
                        "image_base64": "eA==",
                        "metadata": {"objects": [{"id": "Apple|1", "type": "Apple", "visible": True, "pickupable": True}]},
                    },
                ]
            }
            semantic_plan = {
                "task": "Pick up the apple.",
                "targetObjectType": "Apple",
                "needsGrounding": True,
                "observations": [],
                "plan": [{"action": "PickupObject", "objectType": "Apple", "targetType": None}],
            }
            old_probe = auto_scene_actions_module.probe_scene
            old_generate = auto_scene_actions_module.generate_semantic_plan
            old_send = auto_scene_actions_module.send_actions
            auto_scene_actions_module.probe_scene = lambda *args, **kwargs: probe
            auto_scene_actions_module.generate_semantic_plan = lambda *args, **kwargs: ("{}", semantic_plan, None)
            auto_scene_actions_module.send_actions = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not send"))
            try:
                args = auto_scene_actions_module.parse_args(
                    [
                        "--execute-actions-url",
                        "http://127.0.0.1:1/execute_actions",
                        "--task",
                        "Pick up the apple.",
                        "--task-id",
                        "task-1",
                        "--output-dir",
                        temp_dir,
                    ]
                )
                stdout = io.StringIO()
                stderr = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    exit_code = auto_scene_actions_module.run(args)
            finally:
                auto_scene_actions_module.probe_scene = old_probe
                auto_scene_actions_module.generate_semantic_plan = old_generate
                auto_scene_actions_module.send_actions = old_send

        self.assertEqual(exit_code, 0)
        self.assertNotIn("Sending", stderr.getvalue())
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["coordination_result"]["status"], "target_visible_by_peer")
        self.assertNotIn("payload", output)

    def test_closed_loop_dry_run_builds_step_payloads_without_done_until_final(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            probe = {
                "sceneName": "FloorPlan1",
                "selected_robot_id": 0,
                "robots": [{"robot_id": 0}],
                "objects": [
                    {"id": "Fridge|1", "type": "Fridge", "visible": True, "openable": True, "receptacle": True},
                    {"id": "Apple|1", "type": "Apple", "visible": True, "pickupable": True},
                ],
                "image_base64": "eA==",
            }

            def generate(args, image_path, objects, task_id):
                if "open" in args.task.lower():
                    plan = {
                        "task": args.task,
                        "targetObjectType": "Fridge",
                        "needsGrounding": True,
                        "observations": [],
                        "plan": [{"action": "OpenObject", "objectType": "Fridge", "targetType": None}],
                    }
                else:
                    plan = {
                        "task": args.task,
                        "targetObjectType": "Apple",
                        "needsGrounding": True,
                        "observations": [],
                        "plan": [{"action": "PickupObject", "objectType": "Apple", "targetType": None}],
                    }
                return "{}", plan, None

            old_probe = auto_scene_actions_module.probe_scene
            old_generate = auto_scene_actions_module.generate_semantic_plan
            old_send = auto_scene_actions_module.send_actions
            old_observe = auto_scene_actions_module.observe_robot
            auto_scene_actions_module.probe_scene = lambda *args, **kwargs: probe
            auto_scene_actions_module.generate_semantic_plan = generate
            auto_scene_actions_module.send_actions = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("dry-run should not send"))
            auto_scene_actions_module.observe_robot = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("dry-run should not observe"))
            try:
                args = auto_scene_actions_module.parse_args(
                    [
                        "--execute-actions-url",
                        "http://127.0.0.1:1/execute_actions",
                        "--task",
                        "Open the fridge and pick up the apple.",
                        "--task-id",
                        "task-1",
                        "--output-dir",
                        temp_dir,
                        "--relay-mode",
                        "--relay-strategy",
                        "rules",
                        "--closed-loop-replan",
                        "--dry-run",
                    ]
                )
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                    exit_code = auto_scene_actions_module.run(args)
            finally:
                auto_scene_actions_module.probe_scene = old_probe
                auto_scene_actions_module.generate_semantic_plan = old_generate
                auto_scene_actions_module.send_actions = old_send
                auto_scene_actions_module.observe_robot = old_observe

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["closed_loop_result"]["status"], "success")
        self.assertEqual(len(output["intent_steps"]), 2)
        self.assertEqual(output["step_payloads"][0]["actions"], [{"action": "OpenObject", "objectId": "Fridge|1", "forceAction": True}])
        self.assertEqual(output["step_payloads"][1]["actions"], [{"action": "PickupObject", "objectId": "Apple|1", "forceAction": True}])
        self.assertEqual(output["step_payloads"][2]["actions"], [{"action": "Done"}])
        self.assertEqual([payload["stop_on_failure"] for payload in output["step_payloads"]], [False, False, False])

    def test_closed_loop_relay_dry_run_selects_peer_executor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            probe = {
                "sceneName": "FloorPlan1",
                "selected_robot_id": 1,
                "robots": [{"robot_id": 0}, {"robot_id": 1}],
                "objects": [{"id": "Fridge|1", "type": "Fridge", "visible": True, "openable": True}],
                "image_base64": "eA==",
            }
            peer_observe = {
                "status": "success",
                "robot_id": 0,
                "robot": {"name": "Robot0"},
                "objects": [{"id": "Apple|1", "type": "Apple", "visible": True, "pickupable": True}],
                "image_base64": "eA==",
            }
            semantic_plan = {
                "task": "pick up the apple.",
                "targetObjectType": "Apple",
                "needsGrounding": True,
                "observations": [],
                "plan": [{"action": "PickupObject", "objectType": "Apple", "targetType": None}],
            }
            old_probe = auto_scene_actions_module.probe_scene
            old_generate = auto_scene_actions_module.generate_semantic_plan
            old_execute_probe = auto_scene_actions_module.execute_actions_probe_scene
            old_send = auto_scene_actions_module.send_actions
            auto_scene_actions_module.probe_scene = lambda *args, **kwargs: probe
            auto_scene_actions_module.generate_semantic_plan = lambda *args, **kwargs: ("{}", semantic_plan, None)
            auto_scene_actions_module.execute_actions_probe_scene = lambda *args, **kwargs: peer_observe
            auto_scene_actions_module.send_actions = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("dry-run should not send"))
            try:
                args = auto_scene_actions_module.parse_args(
                    [
                        "--execute-actions-url",
                        "http://127.0.0.1:1/execute_actions",
                        "--task",
                        "Pick up the apple.",
                        "--task-id",
                        "task-1",
                        "--output-dir",
                        temp_dir,
                        "--relay-mode",
                        "--relay-strategy",
                        "rules",
                        "--primary-robot-id",
                        "1",
                        "--closed-loop-replan",
                        "--dry-run",
                    ]
                )
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                    exit_code = auto_scene_actions_module.run(args)
            finally:
                auto_scene_actions_module.probe_scene = old_probe
                auto_scene_actions_module.generate_semantic_plan = old_generate
                auto_scene_actions_module.execute_actions_probe_scene = old_execute_probe
                auto_scene_actions_module.send_actions = old_send

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["closed_loop_result"]["status"], "success")
        self.assertEqual(output["closed_loop_trace"][0]["executor_robot_id"], 0)
        self.assertEqual(output["step_payloads"][0]["robot_id"], 0)
        self.assertEqual(output["step_payloads"][0]["actions"][0]["objectId"], "Apple|1")

    def test_closed_loop_pickup_step_skips_when_peer_already_holds_object(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            probe = {
                "sceneName": "FloorPlan1",
                "selected_robot_id": 0,
                "state": {"robots": [{"robot_id": 0}, {"robot_id": 1}]},
                "objects": [
                    {"id": "CounterTop|0", "type": "CounterTop", "visible": True, "receptacle": True},
                    {"id": "Pan|1", "type": "Pan", "visible": False, "pickupable": True},
                ],
                "image_base64": "eA==",
            }
            peer_observe = {
                "status": "success",
                "robot_id": 1,
                "robot": {
                    "robot_id": 1,
                    "name": "Robot1",
                    "held_object": {"objectId": "Pan|1", "objectType": "Pan"},
                    "inventory": [{"objectId": "Pan|1", "objectType": "Pan"}],
                },
                "objects": [
                    {"id": "CounterTop|1", "type": "CounterTop", "visible": True, "receptacle": True},
                    {"id": "Pan|1", "type": "Pan", "visible": False, "pickupable": True},
                ],
                "image_base64": "eA==",
            }
            put_plan = {
                "task": "put the pan on the countertop.",
                "targetObjectType": "Pan",
                "needsGrounding": True,
                "observations": [],
                "plan": [{"action": "PutObject", "objectType": "Pan", "targetType": "CounterTop"}],
            }
            generated = [put_plan]
            observed_robot_ids = []
            sent_payloads = []
            old_probe = auto_scene_actions_module.probe_scene
            old_generate = auto_scene_actions_module.generate_semantic_plan
            old_execute_probe = auto_scene_actions_module.execute_actions_probe_scene
            old_send = auto_scene_actions_module.send_actions
            auto_scene_actions_module.probe_scene = lambda *args, **kwargs: probe
            auto_scene_actions_module.generate_semantic_plan = lambda *args, **kwargs: ("{}", generated.pop(0), None)
            auto_scene_actions_module.execute_actions_probe_scene = lambda url, task_id, timeout, robot_id=0: observed_robot_ids.append(robot_id) or peer_observe

            def fake_send(url, payload, timeout):
                sent_payloads.append(payload)
                return json.dumps({"status": "success", "results": [{"robot_id": payload.get("robot_id"), "action": payload["actions"][0]["action"], "success": True}], "state": {"objects": []}})

            auto_scene_actions_module.send_actions = fake_send
            try:
                args = auto_scene_actions_module.parse_args(
                    [
                        "--execute-actions-url",
                        "http://127.0.0.1:1/execute_actions",
                        "--task",
                        "put the pan on the counter",
                        "--task-id",
                        "task-1",
                        "--output-dir",
                        temp_dir,
                        "--relay-mode",
                        "--relay-strategy",
                        "rules",
                        "--primary-robot-id",
                        "0",
                        "--closed-loop-replan",
                    ]
                )
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                    exit_code = auto_scene_actions_module.run(args)
            finally:
                auto_scene_actions_module.probe_scene = old_probe
                auto_scene_actions_module.generate_semantic_plan = old_generate
                auto_scene_actions_module.execute_actions_probe_scene = old_execute_probe
                auto_scene_actions_module.send_actions = old_send

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue())
        self.assertEqual(observed_robot_ids, [1])
        self.assertEqual(
            output["intent_steps"],
            [{"order": 1, "action": "PutObject", "objectType": "Pan", "targetType": "CounterTop"}],
        )
        self.assertNotIn("intentExpansionWarnings", output["task_intent"])
        self.assertEqual(output["closed_loop_trace"][0]["executor_robot_id"], 1)
        self.assertEqual(output["closed_loop_trace"][0]["actions"], [{"action": "PutObject", "objectId": "CounterTop|1", "forceAction": True}])
        self.assertEqual(sent_payloads[0]["robot_id"], 1)
        self.assertIs(sent_payloads[0]["stop_on_failure"], False)
        self.assertEqual(sent_payloads[0]["actions"], [{"action": "PutObject", "objectId": "CounterTop|1", "forceAction": True}])

    def test_closed_loop_put_inserts_pickup_when_no_robot_holds_object_but_peer_sees_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            probe = {
                "sceneName": "FloorPlan1",
                "selected_robot_id": 0,
                "state": {"robots": [{"robot_id": 0}, {"robot_id": 1}]},
                "objects": [
                    {"id": "CounterTop|0", "type": "CounterTop", "visible": True, "receptacle": True},
                    {"id": "Pan|1", "type": "Pan", "visible": False, "pickupable": True},
                ],
                "image_base64": "eA==",
            }
            peer_observe = {
                "status": "success",
                "robot_id": 1,
                "robot": {"robot_id": 1, "name": "Robot1", "inventory": []},
                "objects": [
                    {"id": "CounterTop|1", "type": "CounterTop", "visible": True, "receptacle": True},
                    {"id": "Pan|1", "type": "Pan", "visible": True, "pickupable": True},
                ],
                "image_base64": "eA==",
            }
            pickup_plan = {
                "task": "pick up the pan",
                "targetObjectType": "Pan",
                "needsGrounding": True,
                "observations": [],
                "plan": [{"action": "PickupObject", "objectType": "Pan", "targetType": None}],
            }
            put_plan = {
                "task": "put the pan on the countertop.",
                "targetObjectType": "Pan",
                "needsGrounding": True,
                "observations": [],
                "plan": [{"action": "PutObject", "objectType": "Pan", "targetType": "CounterTop"}],
            }
            generated = [pickup_plan, put_plan]
            observed_robot_ids = []
            sent_payloads = []
            old_probe = auto_scene_actions_module.probe_scene
            old_generate = auto_scene_actions_module.generate_semantic_plan
            old_execute_probe = auto_scene_actions_module.execute_actions_probe_scene
            old_send = auto_scene_actions_module.send_actions
            auto_scene_actions_module.probe_scene = lambda *args, **kwargs: probe
            auto_scene_actions_module.generate_semantic_plan = lambda *args, **kwargs: ("{}", generated.pop(0), None)
            auto_scene_actions_module.execute_actions_probe_scene = lambda url, task_id, timeout, robot_id=0: observed_robot_ids.append(robot_id) or peer_observe

            def fake_send(url, payload, timeout):
                sent_payloads.append(payload)
                held = [{"objectId": "Pan|1", "objectType": "Pan"}] if payload["actions"][0]["action"] == "PickupObject" else []
                return json.dumps({
                    "status": "success",
                    "results": [
                        {
                            "robot_id": payload.get("robot_id"),
                            "action": payload["actions"][0]["action"],
                            "success": True,
                            "inventory": held,
                            "held_object": held[0] if held else None,
                        }
                    ],
                    "state": {"objects": []},
                })

            auto_scene_actions_module.send_actions = fake_send
            try:
                args = auto_scene_actions_module.parse_args(
                    [
                        "--execute-actions-url",
                        "http://127.0.0.1:1/execute_actions",
                        "--task",
                        "put the pan on the counter",
                        "--task-id",
                        "task-1",
                        "--output-dir",
                        temp_dir,
                        "--relay-mode",
                        "--relay-strategy",
                        "rules",
                        "--primary-robot-id",
                        "0",
                        "--closed-loop-replan",
                    ]
                )
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                    exit_code = auto_scene_actions_module.run(args)
            finally:
                auto_scene_actions_module.probe_scene = old_probe
                auto_scene_actions_module.generate_semantic_plan = old_generate
                auto_scene_actions_module.execute_actions_probe_scene = old_execute_probe
                auto_scene_actions_module.send_actions = old_send

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue())
        self.assertEqual(observed_robot_ids, [1])
        self.assertEqual(output["intent_steps"][0]["action"], "PickupObject")
        self.assertEqual(output["intent_steps"][1]["action"], "PutObject")
        self.assertIn("inserted PickupObject", output["task_intent"]["intentExpansionWarnings"][0])
        self.assertEqual(sent_payloads[0]["robot_id"], 1)
        self.assertEqual(sent_payloads[0]["actions"], [{"action": "PickupObject", "objectId": "Pan|1", "forceAction": True}])
        self.assertEqual(sent_payloads[1]["robot_id"], 1)
        self.assertEqual(sent_payloads[1]["actions"], [{"action": "PutObject", "objectId": "CounterTop|1", "forceAction": True}])

    def test_closed_loop_real_step_failure_reports_upstream(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            probe = {
                "sceneName": "FloorPlan1",
                "selected_robot_id": 0,
                "robots": [{"robot_id": 0}],
                "objects": [{"id": "Tomato|1", "type": "Tomato", "visible": True, "pickupable": True}],
                "image_base64": "eA==",
            }
            semantic_plan = {
                "task": "pick up the tomato.",
                "targetObjectType": "Tomato",
                "needsGrounding": True,
                "observations": [],
                "plan": [{"action": "PickupObject", "objectType": "Tomato", "targetType": None}],
            }
            old_probe = auto_scene_actions_module.probe_scene
            old_generate = auto_scene_actions_module.generate_semantic_plan
            old_send = auto_scene_actions_module.send_actions
            old_observe = auto_scene_actions_module.observe_robot
            auto_scene_actions_module.probe_scene = lambda *args, **kwargs: probe
            auto_scene_actions_module.generate_semantic_plan = lambda *args, **kwargs: ("{}", semantic_plan, None)
            auto_scene_actions_module.send_actions = lambda *args, **kwargs: json.dumps(
                {"status": "failed", "results": [{"action": "PickupObject", "success": False, "error": "not reachable"}]}
            )
            auto_scene_actions_module.observe_robot = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not observe after failed execute"))
            try:
                args = auto_scene_actions_module.parse_args(
                    [
                        "--execute-actions-url",
                        "http://127.0.0.1:1/execute_actions",
                        "--task",
                        "Pick up the tomato.",
                        "--task-id",
                        "task-1",
                        "--output-dir",
                        temp_dir,
                        "--relay-mode",
                        "--relay-strategy",
                        "rules",
                        "--closed-loop-replan",
                    ]
                )
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                    exit_code = auto_scene_actions_module.run(args)
            finally:
                auto_scene_actions_module.probe_scene = old_probe
                auto_scene_actions_module.generate_semantic_plan = old_generate
                auto_scene_actions_module.send_actions = old_send
                auto_scene_actions_module.observe_robot = old_observe

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["closed_loop_result"]["status"], "needs_upstream_planning")
        self.assertEqual(output["closed_loop_result"]["failed_step_index"], 1)
        self.assertIn("failed", output["closed_loop_result"]["reason"])

    def test_run_saves_object_visibility_map(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            probe = {
                "state": {
                    "sceneName": "FloorPlan1",
                    "objects": [{"id": "Apple|1", "type": "Apple", "visible": True, "pickupable": True}],
                },
                "image_base64": "eA==",
            }
            semantic_plan = {
                "task": "Pick up the apple.",
                "targetObjectType": "Apple",
                "needsGrounding": True,
                "observations": [],
                "plan": [{"action": "PickupObject", "objectType": "Apple", "targetType": None}],
            }
            old_probe = auto_scene_actions_module.probe_scene
            old_generate = auto_scene_actions_module.generate_semantic_plan
            auto_scene_actions_module.probe_scene = lambda *args, **kwargs: probe
            auto_scene_actions_module.generate_semantic_plan = lambda *args, **kwargs: ("{}", semantic_plan, None)
            try:
                args = auto_scene_actions_module.parse_args(
                    [
                        "--execute-actions-url",
                        "http://127.0.0.1:1/execute_actions",
                        "--task",
                        "Pick up the apple.",
                        "--task-id",
                        "task-1",
                        "--output-dir",
                        temp_dir,
                        "--dry-run",
                        "--save-object-visibility-map",
                    ]
                )
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                    exit_code = auto_scene_actions_module.run(args)
            finally:
                auto_scene_actions_module.probe_scene = old_probe
                auto_scene_actions_module.generate_semantic_plan = old_generate

            self.assertEqual(exit_code, 0)
            output = json.loads(stdout.getvalue())
            visibility_path = Path(output["object_visibility_map_path"])
            self.assertTrue(visibility_path.exists())
            self.assertEqual(json.loads(visibility_path.read_text(encoding="utf-8"))["primary_agent_id"], "robot_0")


    def test_relay_mode_primary_visible_adds_executor_agent_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            probe = {
                "state": {
                    "sceneName": "FloorPlan1",
                    "objects": [{"id": "Apple|1", "type": "Apple", "visible": True, "pickupable": True}],
                },
                "image_base64": "eA==",
            }
            semantic_plan = {
                "task": "Pick up the apple.",
                "targetObjectType": "Apple",
                "needsGrounding": True,
                "observations": [],
                "plan": [{"action": "PickupObject", "objectType": "Apple", "targetType": None}],
            }
            old_probe = auto_scene_actions_module.probe_scene
            old_generate = auto_scene_actions_module.generate_semantic_plan
            auto_scene_actions_module.probe_scene = lambda *args, **kwargs: probe
            auto_scene_actions_module.generate_semantic_plan = lambda *args, **kwargs: ("{}", semantic_plan, None)
            try:
                args = auto_scene_actions_module.parse_args(
                    [
                        "--execute-actions-url",
                        "http://127.0.0.1:1/execute_actions",
                        "--task",
                        "Pick up the apple.",
                        "--task-id",
                        "task-1",
                        "--output-dir",
                        temp_dir,
                        "--dry-run",
                        "--relay-mode",
                        "--relay-strategy",
                        "rules",
                    ]
                )
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                    exit_code = auto_scene_actions_module.run(args)
            finally:
                auto_scene_actions_module.probe_scene = old_probe
                auto_scene_actions_module.generate_semantic_plan = old_generate

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["relay_result"]["status"], "executor_ready")
        self.assertEqual(output["executor_agent_id"], "robot_0")
        self.assertEqual(output["payload"]["robot_id"], 0)

    def test_relay_mode_peer_visible_replans_and_sends_to_peer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            probe = {
                "events": [
                    {
                        "agentId": "agent_0",
                        "image_base64": "eA==",
                        "metadata": {"objects": [{"id": "Egg|1", "type": "Egg", "visible": True, "pickupable": True}]},
                    },
                    {
                        "agentId": "agent_1",
                        "image_base64": "eA==",
                        "metadata": {"objects": [{"id": "Apple|1", "type": "Apple", "visible": True, "pickupable": True}]},
                    },
                ]
            }
            semantic_plan = {
                "task": "Pick up the apple.",
                "targetObjectType": "Apple",
                "needsGrounding": True,
                "observations": [],
                "plan": [{"action": "PickupObject", "objectType": "Apple", "targetType": None}],
            }
            calls = []
            captured = {}
            old_probe = auto_scene_actions_module.probe_scene
            old_generate = auto_scene_actions_module.generate_semantic_plan
            old_send = auto_scene_actions_module.send_actions
            auto_scene_actions_module.probe_scene = lambda *args, **kwargs: probe
            auto_scene_actions_module.generate_semantic_plan = lambda args, image_path, objects, task_id: calls.append(str(image_path)) or ("{}", semantic_plan, None)
            def fake_send(url, payload, timeout):
                captured["payload"] = payload
                return json.dumps({"status": "success", "state": {"objects": []}})

            auto_scene_actions_module.send_actions = fake_send
            try:
                args = auto_scene_actions_module.parse_args(
                    [
                        "--execute-actions-url",
                        "http://127.0.0.1:1/execute_actions",
                        "--task",
                        "Pick up the apple.",
                        "--task-id",
                        "task-1",
                        "--output-dir",
                        temp_dir,
                        "--relay-mode",
                        "--relay-strategy",
                        "rules",
                    ]
                )
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                    exit_code = auto_scene_actions_module.run(args)
            finally:
                auto_scene_actions_module.probe_scene = old_probe
                auto_scene_actions_module.generate_semantic_plan = old_generate
                auto_scene_actions_module.send_actions = old_send

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue())
        self.assertEqual(len(calls), 2)
        self.assertEqual(output["relay_result"]["status"], "executor_ready")
        self.assertEqual(output["executor_agent_id"], "robot_1")
        self.assertEqual(captured["payload"]["robot_id"], 1)
        self.assertEqual(captured["payload"]["actions"][0]["objectId"], "Apple|1")
        self.assertIn("executor_semantic_plan", output)

    def test_relay_mode_peer_visible_but_action_not_affordable_reports_upstream(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            probe = {
                "events": [
                    {
                        "agentId": "agent_0",
                        "image_base64": "eA==",
                        "metadata": {"objects": [{"id": "Egg|1", "type": "Egg", "visible": True, "pickupable": True}]},
                    },
                    {
                        "agentId": "agent_1",
                        "image_base64": "eA==",
                        "metadata": {"objects": [{"id": "Fridge|1", "type": "Fridge", "visible": True, "openable": True, "pickupable": False}]},
                    },
                ]
            }
            semantic_plan = {
                "task": "Pick up the fridge.",
                "targetObjectType": "Fridge",
                "needsGrounding": True,
                "observations": [],
                "plan": [{"action": "PickupObject", "objectType": "Fridge", "targetType": None}],
            }
            old_probe = auto_scene_actions_module.probe_scene
            old_generate = auto_scene_actions_module.generate_semantic_plan
            old_send = auto_scene_actions_module.send_actions
            auto_scene_actions_module.probe_scene = lambda *args, **kwargs: probe
            auto_scene_actions_module.generate_semantic_plan = lambda *args, **kwargs: ("{}", semantic_plan, None)
            auto_scene_actions_module.send_actions = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not send"))
            try:
                args = auto_scene_actions_module.parse_args(
                    [
                        "--execute-actions-url",
                        "http://127.0.0.1:1/execute_actions",
                        "--task",
                        "Pick up the fridge.",
                        "--task-id",
                        "task-1",
                        "--output-dir",
                        temp_dir,
                        "--relay-mode",
                        "--relay-strategy",
                        "rules",
                    ]
                )
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                    exit_code = auto_scene_actions_module.run(args)
            finally:
                auto_scene_actions_module.probe_scene = old_probe
                auto_scene_actions_module.generate_semantic_plan = old_generate
                auto_scene_actions_module.send_actions = old_send

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["relay_result"]["status"], "needs_upstream_planning")
        self.assertIn("not pickupable", output["relay_result"]["reason"])
        self.assertNotIn("payload", output)

    def test_relay_mode_target_not_visible_reports_upstream(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            probe = {
                "events": [
                    {"agentId": "agent_0", "image_base64": "eA==", "metadata": {"objects": [{"id": "Egg|1", "type": "Egg", "visible": True, "pickupable": True}]}},
                    {"agentId": "agent_1", "image_base64": "eA==", "metadata": {"objects": [{"id": "Mug|1", "type": "Mug", "visible": True, "pickupable": True}]}},
                ]
            }
            semantic_plan = {
                "task": "Pick up the banana.",
                "targetObjectType": "Banana",
                "needsGrounding": True,
                "observations": [],
                "plan": [{"action": "PickupObject", "objectType": "Banana", "targetType": None}],
            }
            old_probe = auto_scene_actions_module.probe_scene
            old_generate = auto_scene_actions_module.generate_semantic_plan
            old_send = auto_scene_actions_module.send_actions
            auto_scene_actions_module.probe_scene = lambda *args, **kwargs: probe
            auto_scene_actions_module.generate_semantic_plan = lambda *args, **kwargs: ("{}", semantic_plan, None)
            auto_scene_actions_module.send_actions = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not send"))
            try:
                args = auto_scene_actions_module.parse_args(
                    [
                        "--execute-actions-url",
                        "http://127.0.0.1:1/execute_actions",
                        "--task",
                        "Pick up the banana.",
                        "--task-id",
                        "task-1",
                        "--output-dir",
                        temp_dir,
                        "--relay-mode",
                        "--relay-strategy",
                        "rules",
                    ]
                )
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                    exit_code = auto_scene_actions_module.run(args)
            finally:
                auto_scene_actions_module.probe_scene = old_probe
                auto_scene_actions_module.generate_semantic_plan = old_generate
                auto_scene_actions_module.send_actions = old_send

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["relay_result"]["status"], "needs_upstream_planning")
        self.assertIn("not visible", output["relay_result"]["reason"])
        self.assertNotIn("payload", output)

    def test_relay_mode_dry_run_builds_payload_without_send(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            probe = {
                "state": {
                    "objects": [{"id": "Apple|1", "type": "Apple", "visible": True, "pickupable": True}],
                },
                "image_base64": "eA==",
            }
            semantic_plan = {
                "task": "Pick up the apple.",
                "targetObjectType": "Apple",
                "needsGrounding": True,
                "observations": [],
                "plan": [{"action": "PickupObject", "objectType": "Apple", "targetType": None}],
            }
            old_probe = auto_scene_actions_module.probe_scene
            old_generate = auto_scene_actions_module.generate_semantic_plan
            old_send = auto_scene_actions_module.send_actions
            auto_scene_actions_module.probe_scene = lambda *args, **kwargs: probe
            auto_scene_actions_module.generate_semantic_plan = lambda *args, **kwargs: ("{}", semantic_plan, None)
            auto_scene_actions_module.send_actions = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not send"))
            try:
                args = auto_scene_actions_module.parse_args(
                    [
                        "--execute-actions-url",
                        "http://127.0.0.1:1/execute_actions",
                        "--task",
                        "Pick up the apple.",
                        "--task-id",
                        "task-1",
                        "--output-dir",
                        temp_dir,
                        "--relay-mode",
                        "--relay-strategy",
                        "rules",
                        "--dry-run",
                    ]
                )
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                    exit_code = auto_scene_actions_module.run(args)
            finally:
                auto_scene_actions_module.probe_scene = old_probe
                auto_scene_actions_module.generate_semantic_plan = old_generate
                auto_scene_actions_module.send_actions = old_send

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["payload"]["robot_id"], 0)



    def test_closed_loop_put_primary_holder_skips_ownership_probe_and_relay_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            probe = {
                "sceneName": "FloorPlan1",
                "selected_robot_id": 0,
                "robots": [
                    {
                        "robot_id": 0,
                        "held_object": {"objectId": "Tomato|1", "objectType": "Tomato"},
                        "inventory": [{"objectId": "Tomato|1", "objectType": "Tomato"}],
                    },
                    {"robot_id": 2},
                ],
                "objects": [
                    {"id": "CounterTop|0", "type": "CounterTop", "visible": True, "receptacle": True},
                    {"id": "Tomato|1", "type": "Tomato", "visible": False, "pickupable": True},
                ],
                "image_base64": "eA==",
            }
            put_plan = {
                "task": "put the tomato on the countertop.",
                "targetObjectType": "Tomato",
                "needsGrounding": True,
                "observations": [],
                "plan": [{"action": "PutObject", "objectType": "Tomato", "targetType": "CounterTop"}],
            }

            class FailingRelayBackend:
                def generate_messages(self, *args, **kwargs):
                    raise AssertionError("primary PutObject fast path should not call relay backend")

            def fake_put_intent(args, available_types):
                tool_call = {"name": "extract_task_intent", "arguments": {"task": args.task}}
                intent = {
                    "requestedAction": "PutObject",
                    "requestedObjectType": "Tomato",
                    "intentSteps": [
                        {"order": 1, "action": "PutObject", "objectType": "Tomato", "targetType": "CounterTop"}
                    ],
                }
                return tool_call, intent, {"status": "ok", "warnings": []}

            old_probe = auto_scene_actions_module.probe_scene
            old_execute_probe = auto_scene_actions_module.execute_actions_probe_scene
            old_generate = auto_scene_actions_module.generate_semantic_plan
            old_intent = auto_scene_actions_module.generate_task_intent_tool_call
            old_send = auto_scene_actions_module.send_actions
            auto_scene_actions_module.probe_scene = lambda *args, **kwargs: probe
            auto_scene_actions_module.execute_actions_probe_scene = lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("primary PutObject fast path should not probe peer robots")
            )
            auto_scene_actions_module.generate_semantic_plan = lambda *args, **kwargs: ("{}", put_plan, None)
            auto_scene_actions_module.generate_task_intent_tool_call = fake_put_intent
            auto_scene_actions_module.send_actions = lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("dry run should not send actions")
            )
            try:
                args = auto_scene_actions_module.parse_args(
                    [
                        "--execute-actions-url",
                        "http://127.0.0.1:1/execute_actions",
                        "--task",
                        "put the tomato on the counter.",
                        "--task-id",
                        "task-1",
                        "--output-dir",
                        temp_dir,
                        "--relay-mode",
                        "--relay-strategy",
                        "agent",
                        "--primary-robot-id",
                        "0",
                        "--closed-loop-replan",
                        "--dry-run",
                    ]
                )
                args._qwen_backend = FailingRelayBackend()
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                    exit_code = auto_scene_actions_module.run(args)
            finally:
                auto_scene_actions_module.probe_scene = old_probe
                auto_scene_actions_module.execute_actions_probe_scene = old_execute_probe
                auto_scene_actions_module.generate_semantic_plan = old_generate
                auto_scene_actions_module.generate_task_intent_tool_call = old_intent
                auto_scene_actions_module.send_actions = old_send

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["queried_robot_ids"], [0])
        self.assertEqual(output["closed_loop_result"]["status"], "success")
        self.assertEqual(output["closed_loop_trace"][0]["relay_result"]["strategy"], "primary_fast_path")
        self.assertEqual(output["closed_loop_trace"][0]["executor_robot_id"], 0)
        self.assertEqual(
            output["step_payloads"][0]["actions"],
            [{"action": "PutObject", "objectId": "CounterTop|0", "forceAction": True}],
        )
        self.assertEqual(output["step_payloads"][0]["robot_id"], 0)

    def test_closed_loop_agent_strategy_primary_fast_path_skips_relay_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            probe = {
                "sceneName": "FloorPlan1",
                "selected_robot_id": 0,
                "robots": [{"robot_id": 0}, {"robot_id": 1}, {"robot_id": 2}],
                "objects": [{"id": "Tomato|1", "type": "Tomato", "visible": True, "pickupable": True}],
                "image_base64": "eA==",
            }
            semantic_plan = {
                "task": "Pick up the tomato.",
                "targetObjectType": "Tomato",
                "needsGrounding": True,
                "observations": [],
                "plan": [{"action": "PickupObject", "objectType": "Tomato", "targetType": None}],
            }

            class FailingRelayBackend:
                def generate_messages(self, *args, **kwargs):
                    raise AssertionError("primary fast path should not call relay backend")

            old_probe = auto_scene_actions_module.probe_scene
            old_execute_probe = auto_scene_actions_module.execute_actions_probe_scene
            old_generate = auto_scene_actions_module.generate_semantic_plan
            auto_scene_actions_module.probe_scene = lambda *args, **kwargs: probe
            auto_scene_actions_module.execute_actions_probe_scene = lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("primary fast path should not probe peer robots")
            )
            auto_scene_actions_module.generate_semantic_plan = lambda *args, **kwargs: ("{}", semantic_plan, None)
            try:
                args = auto_scene_actions_module.parse_args(
                    [
                        "--execute-actions-url",
                        "http://127.0.0.1:1/execute_actions",
                        "--task",
                        "Pick up the tomato.",
                        "--task-id",
                        "task-1",
                        "--output-dir",
                        temp_dir,
                        "--relay-mode",
                        "--relay-strategy",
                        "agent",
                        "--closed-loop-replan",
                        "--dry-run",
                    ]
                )
                args._qwen_backend = FailingRelayBackend()
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                    exit_code = auto_scene_actions_module.run(args)
            finally:
                auto_scene_actions_module.probe_scene = old_probe
                auto_scene_actions_module.execute_actions_probe_scene = old_execute_probe
                auto_scene_actions_module.generate_semantic_plan = old_generate

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["closed_loop_result"]["status"], "success")
        self.assertEqual(output["closed_loop_trace"][0]["relay_result"]["strategy"], "primary_fast_path")
        self.assertEqual(output["closed_loop_trace"][0]["executor_robot_id"], 0)
        self.assertEqual(output["step_payloads"][0]["robot_id"], 0)
        self.assertEqual(output["queried_robot_ids"], [0])

    def test_probe_scene_uses_execute_actions_pass_with_primary_robot_id(self) -> None:
        captured: dict[str, object] = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                captured["path"] = self.path
                length = int(self.headers.get("Content-Length", "0"))
                captured["payload"] = json.loads(self.rfile.read(length).decode("utf-8"))
                body = json.dumps(
                    {
                        "status": "success",
                        "results": [
                            {
                                "robot_id": 0,
                                "robot_name": "Robot0",
                                "action": "Pass",
                                "success": True,
                                "image_base64": "eA==",
                            }
                        ],
                        "state": {
                            "sceneName": "FloorPlan1",
                            "selected_robot_id": 0,
                            "robots": [{"robot_id": 0, "name": "Robot0"}],
                            "objects": [{"id": "Apple|1", "type": "Apple", "visible": True, "pickupable": True}],
                        },
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.handle_request)
        thread.start()
        try:
            probe = auto_scene_actions_module.probe_scene(
                f"http://127.0.0.1:{server.server_port}/execute_actions",
                "task-1",
                2.0,
                primary_robot_id=0,
                state_endpoint="/state",
            )
        finally:
            thread.join(timeout=5)
            server.server_close()

        observations = extract_agent_observations(probe, primary_robot_id=0)
        self.assertEqual(captured["path"], "/execute_actions")
        self.assertEqual(captured["payload"]["robot_id"], 0)
        self.assertEqual(captured["payload"]["actions"], [{"action": "Pass"}])
        self.assertEqual(observations[0]["robot_id"], 0)
        self.assertEqual(observations[0]["agent_id"], "robot_0")

    def test_legacy_probe_scene_uses_execute_actions_pass(self) -> None:
        old_post_json = auto_scene_actions_module.post_json
        captured: dict[str, object] = {}

        def fake_post_json(url, payload, timeout):
            captured["url"] = url
            captured["payload"] = payload
            return {
                "status": "success",
                "state": {"objects": [{"id": "Egg|1", "type": "Egg", "visible": True, "pickupable": True}]},
                "results": [{"robot_id": 0, "image_base64": "eA=="}],
            }

        auto_scene_actions_module.post_json = fake_post_json
        try:
            probe = auto_scene_actions_module.legacy_probe_scene(
                "http://127.0.0.1:1/execute_actions",
                "task-1",
                2.0,
            )
        finally:
            auto_scene_actions_module.post_json = old_post_json

        self.assertEqual(captured["payload"]["actions"], [{"action": "Pass"}])
        self.assertEqual(captured["payload"]["robot_id"], 0)
        self.assertEqual(probe["status"], "success")

    def test_relay_mode_primary_visible_does_not_observe_other_robots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            probe = {
                "sceneName": "FloorPlan1",
                "selected_robot_id": 0,
                "robots": [{"robot_id": 0}, {"robot_id": 1}],
                "objects": [{"id": "Apple|1", "type": "Apple", "visible": True, "pickupable": True}],
                "image_base64": "eA==",
            }
            semantic_plan = {
                "task": "Pick up the apple.",
                "targetObjectType": "Apple",
                "needsGrounding": True,
                "observations": [],
                "plan": [{"action": "PickupObject", "objectType": "Apple", "targetType": None}],
            }
            old_probe = auto_scene_actions_module.probe_scene
            old_generate = auto_scene_actions_module.generate_semantic_plan
            old_observe = auto_scene_actions_module.observe_robot
            auto_scene_actions_module.probe_scene = lambda *args, **kwargs: probe
            auto_scene_actions_module.generate_semantic_plan = lambda *args, **kwargs: ("{}", semantic_plan, None)
            auto_scene_actions_module.observe_robot = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not observe peer"))
            try:
                args = auto_scene_actions_module.parse_args(
                    [
                        "--execute-actions-url",
                        "http://127.0.0.1:1/execute_actions",
                        "--task",
                        "Pick up the apple.",
                        "--task-id",
                        "task-1",
                        "--output-dir",
                        temp_dir,
                        "--relay-mode",
                        "--relay-strategy",
                        "rules",
                        "--dry-run",
                    ]
                )
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                    exit_code = auto_scene_actions_module.run(args)
            finally:
                auto_scene_actions_module.probe_scene = old_probe
                auto_scene_actions_module.generate_semantic_plan = old_generate
                auto_scene_actions_module.observe_robot = old_observe

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["payload"]["robot_id"], 0)
        self.assertEqual(output["queried_robot_ids"], [0])

    def test_relay_mode_lazy_observe_selects_peer_robot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            probe = {
                "sceneName": "FloorPlan1",
                "selected_robot_id": 0,
                "robots": [{"robot_id": 0}, {"robot_id": 1}],
                "objects": [{"id": "Egg|1", "type": "Egg", "visible": True, "pickupable": True}],
                "image_base64": "eA==",
            }
            peer_observation = {
                "status": "success",
                "robot_id": 1,
                "robot": {"name": "Robot1"},
                "objects": [{"id": "Apple|1", "type": "Apple", "visible": True, "pickupable": True}],
                "image_base64": "eA==",
            }
            semantic_plan = {
                "task": "Pick up the apple.",
                "targetObjectType": "Apple",
                "needsGrounding": True,
                "observations": [],
                "plan": [{"action": "PickupObject", "objectType": "Apple", "targetType": None}],
            }
            observed = []
            calls = []
            captured = {}
            old_probe = auto_scene_actions_module.probe_scene
            old_execute_probe = auto_scene_actions_module.execute_actions_probe_scene
            old_generate = auto_scene_actions_module.generate_semantic_plan
            old_send = auto_scene_actions_module.send_actions
            auto_scene_actions_module.probe_scene = lambda *args, **kwargs: probe
            auto_scene_actions_module.execute_actions_probe_scene = lambda url, task_id, timeout, robot_id=0: observed.append(robot_id) or peer_observation
            auto_scene_actions_module.generate_semantic_plan = lambda args, image_path, objects, task_id: calls.append(str(image_path)) or ("{}", semantic_plan, None)
            def fake_send(url, payload, timeout):
                captured["payload"] = payload
                return json.dumps({"status": "success", "state": {"objects": []}})
            auto_scene_actions_module.send_actions = fake_send
            try:
                args = auto_scene_actions_module.parse_args(
                    [
                        "--execute-actions-url",
                        "http://127.0.0.1:1/execute_actions",
                        "--task",
                        "Pick up the apple.",
                        "--task-id",
                        "task-1",
                        "--output-dir",
                        temp_dir,
                        "--relay-mode",
                        "--relay-strategy",
                        "rules",
                    ]
                )
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                    exit_code = auto_scene_actions_module.run(args)
            finally:
                auto_scene_actions_module.probe_scene = old_probe
                auto_scene_actions_module.execute_actions_probe_scene = old_execute_probe
                auto_scene_actions_module.generate_semantic_plan = old_generate
                auto_scene_actions_module.send_actions = old_send

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue())
        self.assertEqual(observed, [1])
        self.assertEqual(len(calls), 2)
        self.assertEqual(output["executor_robot_id"], 1)
        self.assertEqual(output["executor_agent_id"], "robot_1")
        self.assertEqual(captured["payload"]["robot_id"], 1)
        self.assertEqual(captured["payload"]["actions"][0]["objectId"], "Apple|1")

    def test_relay_mode_discovers_global_robot_ids_when_primary_state_is_local_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            probe = {
                "sceneName": "FloorPlan1",
                "selected_robot_id": 1,
                "objects": [{"id": "Fridge|1", "type": "Fridge", "visible": False, "openable": True}],
                "image_base64": "eA==",
            }
            global_state = {"robots": [{"robot_id": 0}, {"robot_id": 1}]}
            robot0_observation = {
                "status": "success",
                "robot_id": 0,
                "robot": {"name": "Robot0"},
                "objects": [{"id": "Fridge|1", "type": "Fridge", "visible": True, "openable": True}],
                "image_base64": "eA==",
            }
            primary_plan = {
                "task": "Close the fridge.",
                "targetObjectType": "Fridge",
                "needsGrounding": True,
                "observations": [],
                "plan": [{"action": "CloseObject", "objectType": "Fridge", "targetType": None}],
            }
            observed = []
            calls = []
            captured = {}
            old_probe = auto_scene_actions_module.probe_scene
            old_global_state = auto_scene_actions_module.get_global_state
            old_execute_probe = auto_scene_actions_module.execute_actions_probe_scene
            old_generate = auto_scene_actions_module.generate_semantic_plan
            old_send = auto_scene_actions_module.send_actions
            auto_scene_actions_module.probe_scene = lambda *args, **kwargs: probe
            auto_scene_actions_module.get_global_state = lambda *args, **kwargs: global_state
            auto_scene_actions_module.execute_actions_probe_scene = lambda url, task_id, timeout, robot_id=0: observed.append(robot_id) or robot0_observation
            auto_scene_actions_module.generate_semantic_plan = lambda args, image_path, objects, task_id: calls.append(str(image_path)) or ("{}", primary_plan, None)
            def fake_send(url, payload, timeout):
                captured["payload"] = payload
                return json.dumps({"status": "success", "state": {"objects": []}})
            auto_scene_actions_module.send_actions = fake_send
            try:
                args = auto_scene_actions_module.parse_args(
                    [
                        "--execute-actions-url",
                        "http://127.0.0.1:1/execute_actions",
                        "--task",
                        "Close the fridge.",
                        "--task-id",
                        "task-1",
                        "--output-dir",
                        temp_dir,
                        "--relay-mode",
                        "--relay-strategy",
                        "rules",
                        "--primary-robot-id",
                        "1",
                    ]
                )
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                    exit_code = auto_scene_actions_module.run(args)
            finally:
                auto_scene_actions_module.probe_scene = old_probe
                auto_scene_actions_module.get_global_state = old_global_state
                auto_scene_actions_module.execute_actions_probe_scene = old_execute_probe
                auto_scene_actions_module.generate_semantic_plan = old_generate
                auto_scene_actions_module.send_actions = old_send

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["known_robot_ids"], [0, 1])
        self.assertEqual(output["queried_robot_ids"], [1, 0])
        self.assertEqual(output["robot_discovery_source"], "global_state_fallback")
        self.assertEqual(observed, [0])
        self.assertEqual(len(calls), 2)
        self.assertEqual(output["executor_robot_id"], 0)
        self.assertEqual(captured["payload"]["robot_id"], 0)
        self.assertEqual(captured["payload"]["actions"][0]["objectId"], "Fridge|1")

    def test_relay_mode_global_discovery_failure_keeps_observed_robot_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            probe = {
                "selected_robot_id": 1,
                "robots": [{"robot_id": 1}],
                "objects": [{"id": "Fridge|1", "type": "Fridge", "visible": False, "openable": True}],
                "image_base64": "eA==",
            }
            semantic_plan = {
                "task": "Close the fridge.",
                "targetObjectType": "Fridge",
                "needsGrounding": True,
                "observations": [],
                "plan": [{"action": "CloseObject", "objectType": "Fridge", "targetType": None}],
            }
            observed = []
            old_probe = auto_scene_actions_module.probe_scene
            old_global_state = auto_scene_actions_module.get_global_state
            old_execute_probe = auto_scene_actions_module.execute_actions_probe_scene
            old_generate = auto_scene_actions_module.generate_semantic_plan
            old_send = auto_scene_actions_module.send_actions
            auto_scene_actions_module.probe_scene = lambda *args, **kwargs: probe
            auto_scene_actions_module.get_global_state = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no global state"))
            auto_scene_actions_module.execute_actions_probe_scene = lambda *args, **kwargs: observed.append(kwargs.get("robot_id"))
            auto_scene_actions_module.generate_semantic_plan = lambda *args, **kwargs: ("{}", semantic_plan, None)
            auto_scene_actions_module.send_actions = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not send"))
            try:
                args = auto_scene_actions_module.parse_args(
                    [
                        "--execute-actions-url",
                        "http://127.0.0.1:1/execute_actions",
                        "--task",
                        "Close the fridge.",
                        "--task-id",
                        "task-1",
                        "--output-dir",
                        temp_dir,
                        "--relay-mode",
                        "--relay-strategy",
                        "rules",
                        "--primary-robot-id",
                        "1",
                    ]
                )
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                    exit_code = auto_scene_actions_module.run(args)
            finally:
                auto_scene_actions_module.probe_scene = old_probe
                auto_scene_actions_module.get_global_state = old_global_state
                auto_scene_actions_module.execute_actions_probe_scene = old_execute_probe
                auto_scene_actions_module.generate_semantic_plan = old_generate
                auto_scene_actions_module.send_actions = old_send

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["known_robot_ids"], [1])
        self.assertEqual(output["queried_robot_ids"], [1])
        self.assertEqual(output["robot_discovery_source"], "execute_actions_state")
        self.assertEqual(output["relay_result"]["status"], "needs_upstream_planning")
        self.assertEqual(observed, [])

    def test_known_robot_ids_limits_lazy_observe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            probe = {
                "selected_robot_id": 0,
                "robots": [{"robot_id": 0}, {"robot_id": 1}, {"robot_id": 2}],
                "objects": [{"id": "Egg|1", "type": "Egg", "visible": True, "pickupable": True}],
                "image_base64": "eA==",
            }
            semantic_plan = {
                "task": "Pick up the banana.",
                "targetObjectType": "Banana",
                "needsGrounding": True,
                "observations": [],
                "plan": [{"action": "PickupObject", "objectType": "Banana", "targetType": None}],
            }
            observed = []
            old_probe = auto_scene_actions_module.probe_scene
            old_execute_probe = auto_scene_actions_module.execute_actions_probe_scene
            old_generate = auto_scene_actions_module.generate_semantic_plan
            old_send = auto_scene_actions_module.send_actions
            auto_scene_actions_module.probe_scene = lambda *args, **kwargs: probe
            auto_scene_actions_module.execute_actions_probe_scene = lambda url, task_id, timeout, robot_id=0: observed.append(robot_id) or {
                "robot_id": robot_id,
                "objects": [{"id": f"Mug|{robot_id}", "type": "Mug", "visible": True, "pickupable": True}],
                "image_base64": "eA==",
            }
            auto_scene_actions_module.generate_semantic_plan = lambda *args, **kwargs: ("{}", semantic_plan, None)
            auto_scene_actions_module.send_actions = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not send"))
            try:
                args = auto_scene_actions_module.parse_args(
                    [
                        "--execute-actions-url",
                        "http://127.0.0.1:1/execute_actions",
                        "--task",
                        "Pick up the banana.",
                        "--task-id",
                        "task-1",
                        "--output-dir",
                        temp_dir,
                        "--relay-mode",
                        "--relay-strategy",
                        "rules",
                        "--known-robot-ids",
                        "0,2",
                    ]
                )
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                    exit_code = auto_scene_actions_module.run(args)
            finally:
                auto_scene_actions_module.probe_scene = old_probe
                auto_scene_actions_module.execute_actions_probe_scene = old_execute_probe
                auto_scene_actions_module.generate_semantic_plan = old_generate
                auto_scene_actions_module.send_actions = old_send

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue())
        self.assertEqual(observed, [2])
        self.assertEqual(output["known_robot_ids"], [0, 2])
        self.assertEqual(output["relay_result"]["status"], "needs_upstream_planning")



class GoalConsistencyVerifierTest(unittest.TestCase):
    def test_extracts_requested_object_from_common_types(self) -> None:
        self.assertEqual(extract_requested_object_type("Pick up the apple.", ["Egg", "Fridge"]), "Apple")

    def test_rejects_unseen_model_target_object(self) -> None:
        semantic_plan = {"targetObjectType": "Banana", "plan": [{"action": "Done", "objectType": "Egg", "targetType": None}]}
        objects = [
            {"id": "Egg|1", "type": "Egg", "visible": True, "pickupable": True},
            {"id": "Fridge|1", "type": "Fridge", "visible": True, "openable": True},
        ]

        with self.assertRaisesRegex(ValueError, "Banana.*visible categories: Egg, Fridge"):
            validate_goal_consistency("Pick up the banana.", semantic_plan, objects)

    def test_rejects_substituted_plan_object(self) -> None:
        semantic_plan = {"targetObjectType": "Apple", "plan": [{"action": "PickupObject", "objectType": "Egg", "targetType": None}]}
        objects = [
            {"id": "Apple|1", "type": "Apple", "visible": True, "pickupable": True},
            {"id": "Egg|1", "type": "Egg", "visible": True, "pickupable": True},
        ]

        with self.assertRaisesRegex(ValueError, "does not match requested object"):
            validate_goal_consistency("Pick up the apple.", semantic_plan, objects)

    def test_rejects_requested_object_when_not_visible(self) -> None:
        semantic_plan = {"targetObjectType": "Apple", "plan": [{"action": "PickupObject", "objectType": "Egg", "targetType": None}]}
        objects = [
            {"id": "Egg|1", "type": "Egg", "visible": True, "pickupable": True},
            {"id": "Fridge|1", "type": "Fridge", "visible": True, "openable": True},
        ]

        with self.assertRaisesRegex(ValueError, "Apple.*visible categories: Egg, Fridge"):
            validate_goal_consistency("Pick up the apple.", semantic_plan, objects)

    def test_allows_matching_requested_object(self) -> None:
        semantic_plan = {"targetObjectType": "Apple", "plan": [{"action": "PickupObject", "objectType": "Apple", "targetType": None}]}
        objects = [{"id": "Apple|1", "type": "Apple", "visible": True, "pickupable": True}]

        validate_goal_consistency("Pick up the apple.", semantic_plan, objects)

    def test_allows_matching_put_object_with_different_receptacle(self) -> None:
        semantic_plan = {"targetObjectType": "Apple", "plan": [{"action": "PutObject", "objectType": "Apple", "targetType": "CounterTop"}]}
        objects = [
            {"id": "Apple|1", "type": "Apple", "visible": True, "pickupable": True},
            {"id": "CounterTop|1", "type": "CounterTop", "visible": True, "receptacle": True},
        ]

        validate_goal_consistency("Put the apple on the counter.", semantic_plan, objects)

    def test_rejects_done_only_plan_for_visible_requested_object(self) -> None:
        semantic_plan = {"targetObjectType": "Apple", "plan": [{"action": "Done", "objectType": "Apple", "targetType": None}]}
        objects = [{"id": "Apple|1", "type": "Apple", "visible": True, "pickupable": True}]

        with self.assertRaisesRegex(ValueError, "does not operate on requested object"):
            validate_goal_consistency("Pick up the apple.", semantic_plan, objects)

    def test_ignores_tasks_without_explicit_object(self) -> None:
        semantic_plan = {"targetObjectType": None, "plan": [{"action": "PickupObject", "objectType": "Egg", "targetType": None}]}
        objects = [{"id": "Egg|1", "type": "Egg", "visible": True, "pickupable": True}]

        validate_goal_consistency("Clean up the scene.", semantic_plan, objects)


class ActionIntentVerifierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.old_task_intent_tool_call = auto_scene_actions_module.generate_task_intent_tool_call
        auto_scene_actions_module.generate_task_intent_tool_call = fake_task_intent_tool_call

    def tearDown(self) -> None:
        auto_scene_actions_module.generate_task_intent_tool_call = self.old_task_intent_tool_call

    def test_extracts_requested_action(self) -> None:
        self.assertEqual(extract_requested_action("Pick up the fridge."), "PickupObject")
        self.assertEqual(extract_requested_action("Open the fridge."), "OpenObject")
        self.assertEqual(extract_requested_action("Turn right."), "RotateRight")
        self.assertEqual(extract_requested_action("Move right."), "MoveRight")
        self.assertEqual(extract_requested_action("Look down."), "LookDown")

    def test_rejects_turn_right_rewritten_as_move_right(self) -> None:
        semantic_plan = {"plan": [{"action": "MoveRight", "objectType": None, "targetType": None}]}

        with self.assertRaisesRegex(ValueError, "MoveRight.*RotateRight"):
            validate_action_intent_consistency("Turn right.", semantic_plan)

    def test_allows_matching_turn_right_action(self) -> None:
        semantic_plan = {"plan": [{"action": "RotateRight", "objectType": None, "targetType": None}]}

        validate_action_intent_consistency("Turn right.", semantic_plan)

    def test_allows_matching_move_right_action(self) -> None:
        semantic_plan = {"plan": [{"action": "MoveRight", "objectType": None, "targetType": None}]}

        validate_action_intent_consistency("Move right.", semantic_plan)

    def test_rejects_move_right_rewritten_as_rotate_right(self) -> None:
        semantic_plan = {"plan": [{"action": "RotateRight", "objectType": None, "targetType": None}]}

        with self.assertRaisesRegex(ValueError, "RotateRight.*MoveRight"):
            validate_action_intent_consistency("Move right.", semantic_plan)

    def test_allows_matching_look_down_action(self) -> None:
        semantic_plan = {"plan": [{"action": "LookDown", "objectType": None, "targetType": None}]}

        validate_action_intent_consistency("Look down.", semantic_plan)

    def test_relay_dry_run_turn_right_payload_uses_rotate_right(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            probe = {
                "sceneName": "FloorPlan1",
                "selected_robot_id": 1,
                "robots": [{"robot_id": 0}, {"robot_id": 1}],
                "objects": [{"id": "Fridge|1", "type": "Fridge", "visible": True, "openable": True}],
                "image_base64": "eA==",
            }
            semantic_plan = {
                "task": "Turn right.",
                "targetObjectType": None,
                "needsGrounding": True,
                "observations": [],
                "plan": [{"action": "RotateRight", "objectType": None, "targetType": None}],
            }
            old_probe = auto_scene_actions_module.probe_scene
            old_generate = auto_scene_actions_module.generate_semantic_plan
            old_observe = auto_scene_actions_module.observe_robot
            old_send = auto_scene_actions_module.send_actions
            auto_scene_actions_module.probe_scene = lambda *args, **kwargs: probe
            auto_scene_actions_module.generate_semantic_plan = lambda *args, **kwargs: ("{}", semantic_plan, None)
            auto_scene_actions_module.observe_robot = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not observe peer"))
            auto_scene_actions_module.send_actions = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("dry-run should not send"))
            try:
                args = auto_scene_actions_module.parse_args(
                    [
                        "--execute-actions-url",
                        "http://127.0.0.1:1/execute_actions",
                        "--task",
                        "Turn right.",
                        "--task-id",
                        "task-1",
                        "--output-dir",
                        temp_dir,
                        "--relay-mode",
                        "--relay-strategy",
                        "rules",
                        "--primary-robot-id",
                        "1",
                        "--dry-run",
                    ]
                )
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                    exit_code = auto_scene_actions_module.run(args)
            finally:
                auto_scene_actions_module.probe_scene = old_probe
                auto_scene_actions_module.generate_semantic_plan = old_generate
                auto_scene_actions_module.observe_robot = old_observe
                auto_scene_actions_module.send_actions = old_send

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["payload"]["robot_id"], 1)
        self.assertEqual(output["queried_robot_ids"], [1])
        self.assertIs(output["payload"]["stop_on_failure"], False)
        self.assertEqual(output["payload"]["actions"], [{"action": "RotateRight"}, {"action": "Done"}])

    def test_rejects_pickup_rewritten_as_open(self) -> None:
        semantic_plan = {"plan": [{"action": "OpenObject", "objectType": "Fridge", "targetType": None}]}

        with self.assertRaisesRegex(ValueError, "OpenObject.*PickupObject"):
            validate_action_intent_consistency("Pick up the fridge.", semantic_plan)

    def test_allows_matching_pickup_action(self) -> None:
        semantic_plan = {"plan": [{"action": "PickupObject", "objectType": "Egg", "targetType": None}]}

        validate_action_intent_consistency("Pick up the egg.", semantic_plan)

    def test_allows_matching_open_action(self) -> None:
        semantic_plan = {"plan": [{"action": "OpenObject", "objectType": "Fridge", "targetType": None}]}

        validate_action_intent_consistency("Open the fridge.", semantic_plan)

    def test_rejects_open_rewritten_as_pickup(self) -> None:
        semantic_plan = {"plan": [{"action": "PickupObject", "objectType": "Fridge", "targetType": None}]}

        with self.assertRaisesRegex(ValueError, "PickupObject.*OpenObject"):
            validate_action_intent_consistency("Open the fridge.", semantic_plan)

    def test_ignores_tasks_without_explicit_action(self) -> None:
        semantic_plan = {"plan": [{"action": "PickupObject", "objectType": "Egg", "targetType": None}]}

        validate_action_intent_consistency("The egg on the counter.", semantic_plan)


class ActionAffordanceVerifierTest(unittest.TestCase):
    def test_rejects_pickup_object_for_non_pickupable_fridge(self) -> None:
        semantic_plan = {"plan": [{"action": "PickupObject", "objectType": "Fridge", "targetType": None}]}
        objects = [{"id": "Fridge|1", "type": "Fridge", "visible": True, "openable": True, "pickupable": False}]

        with self.assertRaisesRegex(ValueError, "not pickupable.*PickupObject"):
            validate_action_affordances(semantic_plan, objects, allow_invisible=False)

    def test_allows_open_object_for_openable_fridge(self) -> None:
        semantic_plan = {"plan": [{"action": "OpenObject", "objectType": "Fridge", "targetType": None}]}
        objects = [{"id": "Fridge|1", "type": "Fridge", "visible": True, "openable": True}]

        validate_action_affordances(semantic_plan, objects, allow_invisible=False)
        self.assertEqual(
            ground_semantic_plan(semantic_plan, objects, allow_invisible=False, max_actions=20),
            [{"action": "OpenObject", "objectId": "Fridge|1", "forceAction": True}, {"action": "Done"}],
        )

    def test_allows_pickup_object_for_pickupable_egg(self) -> None:
        semantic_plan = {"plan": [{"action": "PickupObject", "objectType": "Egg", "targetType": None}]}
        objects = [{"id": "Egg|1", "type": "Egg", "visible": True, "pickupable": True}]

        validate_action_affordances(semantic_plan, objects, allow_invisible=False)

    def test_allows_put_object_into_receptacle(self) -> None:
        semantic_plan = {"plan": [{"action": "PutObject", "objectType": "Apple", "targetType": "CounterTop"}]}
        objects = [
            {"id": "Apple|1", "type": "Apple", "visible": True, "pickupable": True},
            {"id": "CounterTop|1", "type": "CounterTop", "visible": True, "receptacle": True},
        ]

        validate_action_affordances(semantic_plan, objects, allow_invisible=False)

    def test_rejects_put_object_into_non_receptacle(self) -> None:
        semantic_plan = {"plan": [{"action": "PutObject", "objectType": "Apple", "targetType": "Fridge"}]}
        objects = [
            {"id": "Apple|1", "type": "Apple", "visible": True, "pickupable": True},
            {"id": "Fridge|1", "type": "Fridge", "visible": True, "openable": True, "receptacle": False},
        ]

        with self.assertRaisesRegex(ValueError, "not receptacle.*PutObject"):
            validate_action_affordances(semantic_plan, objects, allow_invisible=False)


class ActionStatePreconditionVerifierTest(unittest.TestCase):
    def test_inventory_alone_does_not_define_held_object(self) -> None:
        observations = extract_agent_observations(
            {
                "selected_robot_id": 0,
                "objects": [{"id": "Apple|1", "type": "Apple", "visible": True}],
                "inventory": [{"objectId": "Tomato|1", "objectType": "Tomato"}],
            },
            primary_robot_id=0,
        )

        self.assertIsNone(held_object_type_from_observation(observations[0]))
        self.assertIsNone(agent_observations_summary(observations)[0]["held_object_type"])
        self.assertEqual(observations[0]["inventory"], [{"objectId": "Tomato|1", "objectType": "Tomato"}])

    def test_robot_proxy_alone_does_not_define_held_object(self) -> None:
        observations = extract_agent_observations(
            {
                "selected_robot_id": 0,
                "robots": [{"robot_id": 0, "proxy": {"objectId": "Mug|1", "objectType": "Mug"}}],
                "objects": [{"id": "Mug|1", "type": "Mug", "visible": True, "pickupable": True}],
            },
            primary_robot_id=0,
        )

        debug = held_object_debug_from_observation(observations[0])

        self.assertIsNone(held_object_type_from_observation(observations[0]))
        self.assertIsNone(debug["held_object_source"])
        self.assertEqual(debug["robot_state_proxy"], {"objectId": "Mug|1", "objectType": "Mug"})

    def test_extracts_held_object_from_new_observe_fields(self) -> None:
        observations = extract_agent_observations(
            {
                "robot_id": 0,
                "robot": {
                    "robot_id": 0,
                    "held_object": {"objectId": "Tomato|1", "objectType": "Tomato"},
                    "inventory": [{"objectId": "Tomato|1", "objectType": "Tomato"}],
                },
                "metadata": {
                    "inventoryObjects": [{"objectId": "Tomato|1", "objectType": "Tomato"}],
                },
                "objects": [{"id": "Tomato|1", "type": "Tomato", "visible": True, "pickupable": True}],
            },
            primary_robot_id=0,
        )

        self.assertEqual(held_object_type_from_observation(observations[0]), "Tomato")
        self.assertEqual(observations[0]["held_object"]["objectType"], "Tomato")
        self.assertEqual(observations[0]["held_object_source"], "robot_state.held_object")
        self.assertEqual(agent_observations_summary(observations)[0]["held_object_type"], "Tomato")

    def test_held_object_priority_uses_robot_state_before_inventory(self) -> None:
        observation = {
            "agent_id": "robot_0",
            "robot_id": 0,
            "inventory": [{"objectId": "Pan|1", "objectType": "Pan"}],
            "robot_state": {
                "robot_id": 0,
                "held_object": {"objectId": "TomatoSliced|1", "objectType": "TomatoSliced"},
                "inventory": [{"objectId": "Pan|1", "objectType": "Pan"}],
            },
        }

        debug = held_object_debug_from_observation(observation)

        self.assertEqual(held_object_type_from_observation(observation), "TomatoSliced")
        self.assertEqual(debug["held_object_source"], "robot_state.held_object")
        self.assertEqual(debug["held_objects"], [{"objectId": "TomatoSliced|1", "objectType": "TomatoSliced"}])

    def test_normalized_held_object_priority_ignores_conflicting_inventory(self) -> None:
        observation = {
            "agent_id": "robot_0",
            "robot_id": 0,
            "held_object": {"objectId": "TomatoSliced|1", "objectType": "TomatoSliced"},
            "held_object_source": "robot_state.held_object",
            "inventory": [{"objectId": "Pan|1", "objectType": "Pan"}],
            "robot_state": {
                "robot_id": 0,
                "held_object": {"objectId": "TomatoSliced|1", "objectType": "TomatoSliced"},
                "inventory": [{"objectId": "Pan|1", "objectType": "Pan"}],
            },
        }

        debug = held_object_debug_from_observation(observation)

        self.assertEqual(held_object_type_from_observation(observation), "TomatoSliced")
        self.assertEqual(debug["held_object_source"], "robot_state.held_object")
        self.assertEqual(debug["inventory"], [{"objectId": "Pan|1", "objectType": "Pan"}])

    def test_merges_execute_result_held_object_into_observation(self) -> None:
        observation = {
            "agent_id": "robot_0",
            "robot_id": 0,
            "objects": [{"id": "Tomato|1", "type": "Tomato", "visible": True, "pickupable": True}],
            "inventory": [],
            "robot_state": {"robot_id": 0},
        }
        execute_response = {
            "status": "success",
            "results": [
                {
                    "robot_id": 0,
                    "action": "PickupObject",
                    "success": True,
                    "inventory": [{"objectId": "Tomato|1", "objectType": "Tomato"}],
                    "held_object": {"objectId": "Tomato|1", "objectType": "Tomato"},
                    "robot": {
                        "robot_id": 0,
                        "held_object": {"objectId": "Tomato|1", "objectType": "Tomato"},
                        "inventory": [{"objectId": "Tomato|1", "objectType": "Tomato"}],
                    },
                }
            ],
        }

        merge_execute_result_into_observation(observation, execute_response)

        self.assertEqual(held_object_type_from_observation(observation), "Tomato")
        self.assertEqual(observation["held_object"]["objectType"], "Tomato")
        self.assertEqual(observation["held_object_source"], "robot_state.held_object")

    def test_extracts_observation_from_execute_actions_response(self) -> None:
        observations = extract_agent_observations(
            {
                "status": "success",
                "results": [
                    {
                        "robot_id": 0,
                        "image_base64": "eA==",
                        "inventory": [{"objectId": "Tomato|1", "objectType": "Tomato"}],
                        "held_object": {"objectId": "Tomato|1", "objectType": "Tomato"},
                    }
                ],
                "state": {
                    "selected_robot_id": 0,
                    "robots": [
                        {"robot_id": 0, "held_object": {"objectId": "Tomato|1", "objectType": "Tomato"}}
                    ],
                    "objects": [{"id": "Tomato|1", "type": "Tomato", "visible": True, "pickupable": True}],
                },
            },
            primary_robot_id=0,
        )

        self.assertEqual(observations[0]["robot_id"], 0)
        self.assertEqual(held_object_type_from_observation(observations[0]), "Tomato")
        self.assertEqual(observations[0]["image_base64"], "eA==")

    def test_execute_actions_inventory_is_filtered_by_robot_id(self) -> None:
        observations = extract_agent_observations(
            {
                "status": "success",
                "results": [
                    {
                        "robot_id": 0,
                        "inventory": [{"objectId": "TomatoSliced|1", "objectType": "TomatoSliced"}],
                        "held_object": {"objectId": "TomatoSliced|1", "objectType": "TomatoSliced"},
                    },
                    {
                        "robot_id": 1,
                        "inventory": [{"objectId": "Pan|1", "objectType": "Pan"}],
                        "held_object": {"objectId": "Pan|1", "objectType": "Pan"},
                    },
                ],
                "state": {
                    "selected_robot_id": 0,
                    "robots": [
                        {"robot_id": 0, "held_object": {"objectId": "TomatoSliced|1", "objectType": "TomatoSliced"}},
                        {"robot_id": 1, "held_object": {"objectId": "Pan|1", "objectType": "Pan"}},
                    ],
                    "objects": [
                        {"id": "TomatoSliced|1", "type": "TomatoSliced", "visible": True, "pickupable": True},
                        {"id": "Pan|1", "type": "Pan", "visible": True, "pickupable": True},
                    ],
                },
            },
            primary_robot_id=0,
        )

        self.assertEqual(observations[0]["robot_id"], 0)
        self.assertEqual(held_object_type_from_observation(observations[0]), "TomatoSliced")

    def test_merge_execute_result_ignores_other_robot_inventory(self) -> None:
        observation = {
            "agent_id": "robot_0",
            "robot_id": 0,
            "objects": [{"id": "CounterTop|1", "type": "CounterTop", "visible": True, "receptacle": True}],
            "inventory": [],
            "robot_state": {"robot_id": 0},
        }
        execute_response = {
            "status": "success",
            "results": [
                {
                    "robot_id": 1,
                    "inventory": [{"objectId": "Pan|1", "objectType": "Pan"}],
                    "held_object": {"objectId": "Pan|1", "objectType": "Pan"},
                },
                {
                    "robot_id": 0,
                    "inventory": [{"objectId": "TomatoSliced|1", "objectType": "TomatoSliced"}],
                    "held_object": {"objectId": "TomatoSliced|1", "objectType": "TomatoSliced"},
                    "robot": {
                        "robot_id": 0,
                        "held_object": {"objectId": "TomatoSliced|1", "objectType": "TomatoSliced"},
                        "inventory": [{"objectId": "TomatoSliced|1", "objectType": "TomatoSliced"}],
                    },
                },
            ],
        }

        merge_execute_result_into_observation(observation, execute_response)

        self.assertEqual(held_object_type_from_observation(observation), "TomatoSliced")

    def test_held_object_debug_reports_sources(self) -> None:
        observation = {
            "agent_id": "robot_0",
            "robot_id": 0,
            "inventory": [{"objectId": "TomatoSliced|1", "objectType": "TomatoSliced"}],
            "robot_state": {
                "robot_id": 0,
                "held_object": {"objectId": "TomatoSliced|1", "objectType": "TomatoSliced"},
                "inventory": [{"objectId": "TomatoSliced|1", "objectType": "TomatoSliced"}],
                "proxy": None,
            },
        }

        debug = held_object_debug_from_observation(observation)

        self.assertEqual(debug["agent_id"], "robot_0")
        self.assertEqual(debug["robot_id"], 0)
        self.assertEqual(debug["held_object_type"], "TomatoSliced")
        self.assertEqual(debug["held_object_source"], "robot_state.held_object")
        self.assertEqual(debug["robot_state_held_object"]["objectType"], "TomatoSliced")

    def test_rejects_pickup_when_robot_already_holding_object(self) -> None:
        semantic_plan = {"plan": [{"action": "PickupObject", "objectType": "Apple", "targetType": None}]}
        observation = {
            "robot_id": 0,
            "objects": [{"id": "Apple|1", "type": "Apple", "visible": True, "pickupable": True}],
            "inventory": [{"objectId": "Tomato|1", "objectType": "Tomato"}],
            "held_object": {"objectId": "Tomato|1", "objectType": "Tomato"},
        }

        with self.assertRaisesRegex(ValueError, "already holding Tomato.*PickupObject"):
            validate_action_state_preconditions(semantic_plan, observation, allow_invisible=False)

    def test_rejects_put_when_robot_hand_is_empty(self) -> None:
        semantic_plan = {"plan": [{"action": "PutObject", "objectType": "Tomato", "targetType": "CounterTop"}]}
        observation = {
            "robot_id": 0,
            "objects": [{"id": "CounterTop|1", "type": "CounterTop", "visible": True, "receptacle": True}],
            "inventory": [],
        }

        with self.assertRaisesRegex(ValueError, "not holding any object.*PutObject"):
            validate_action_state_preconditions(semantic_plan, observation, allow_invisible=False)


    def test_allows_put_when_robot_holds_matching_object(self) -> None:
        semantic_plan = {"plan": [{"action": "PutObject", "objectType": "Tomato", "targetType": "CounterTop"}]}
        observation = {
            "robot_id": 0,
            "objects": [{"id": "CounterTop|1", "type": "CounterTop", "visible": True, "receptacle": True}],
            "inventory": [{"objectId": "Tomato|1", "objectType": "Tomato"}],
            "held_object": {"objectId": "Tomato|1", "objectType": "Tomato"},
        }

        self.assertIsNone(validate_action_state_preconditions(semantic_plan, observation, allow_invisible=False))

    def test_repairs_redundant_pickup_before_put_when_already_holding_object(self) -> None:
        task_intent = {
            "intentSteps": [
                {"order": 1, "action": "PutObject", "objectType": "Tomato", "targetType": "CounterTop"}
            ]
        }
        semantic_plan = {
            "plan": [
                {"action": "PickupObject", "objectType": "Tomato", "targetType": None},
                {"action": "PutObject", "objectType": "Tomato", "targetType": "CounterTop"},
            ]
        }
        observation = {
            "robot_id": 0,
            "objects": [
                {"id": "Tomato|1", "type": "Tomato", "visible": True, "pickupable": True},
                {"id": "CounterTop|1", "type": "CounterTop", "visible": True, "receptacle": True},
            ],
            "inventory": [{"objectId": "Tomato|1", "objectType": "Tomato"}],
            "held_object": {"objectId": "Tomato|1", "objectType": "Tomato"},
        }

        warnings = repair_redundant_pickup_for_held_put(semantic_plan, task_intent, observation)

        self.assertEqual(
            semantic_plan["plan"],
            [{"action": "PutObject", "objectType": "Tomato", "targetType": "CounterTop"}],
        )
        self.assertIn("removed redundant PickupObject", warnings[0])
        self.assertIsNone(validate_action_state_preconditions(semantic_plan, observation, allow_invisible=False))

    def test_rejects_put_when_robot_holds_different_object(self) -> None:
        semantic_plan = {"plan": [{"action": "PutObject", "objectType": "Apple", "targetType": "CounterTop"}]}
        observation = {
            "robot_id": 0,
            "objects": [{"id": "CounterTop|1", "type": "CounterTop", "visible": True, "receptacle": True}],
            "inventory": [{"objectId": "Tomato|1", "objectType": "Tomato"}],
            "held_object": {"objectId": "Tomato|1", "objectType": "Tomato"},
        }

        with self.assertRaisesRegex(ValueError, "holding Tomato, not Apple.*PutObject"):
            validate_action_state_preconditions(semantic_plan, observation, allow_invisible=False)

    def test_expands_put_intent_with_pickup_when_robot_hand_is_empty(self) -> None:
        task_intent = {
            "requestedAction": "PutObject",
            "requestedObjectType": "TomatoSliced",
            "intentSteps": [
                {"order": 1, "action": "PutObject", "objectType": "TomatoSliced", "targetType": "CounterTop"}
            ],
        }
        observation = {
            "robot_id": 0,
            "objects": [
                {"id": "TomatoSliced|1", "type": "TomatoSliced", "visible": True, "pickupable": True},
                {"id": "CounterTop|1", "type": "CounterTop", "visible": True, "receptacle": True},
            ],
            "inventory": [],
        }

        warnings = expand_put_object_intent_preconditions(task_intent, observation)

        self.assertEqual(
            task_intent["intentSteps"],
            [
                {"order": 1, "action": "PickupObject", "objectType": "TomatoSliced", "targetType": None},
                {"order": 2, "action": "PutObject", "objectType": "TomatoSliced", "targetType": "CounterTop"},
            ],
        )
        self.assertIn("inserted PickupObject", warnings[0])

    def test_expands_put_intent_with_open_for_closed_receptacle(self) -> None:
        task_intent = {
            "requestedAction": "PutObject",
            "requestedObjectType": "TomatoSliced",
            "intentSteps": [
                {"order": 1, "action": "PutObject", "objectType": "TomatoSliced", "targetType": "Drawer"}
            ],
        }
        observation = {
            "robot_id": 0,
            "objects": [
                {"id": "TomatoSliced|1", "type": "TomatoSliced", "visible": True, "pickupable": True},
                {"id": "Drawer|1", "type": "Drawer", "visible": True, "receptacle": True, "openable": True, "isOpen": False},
            ],
            "inventory": [],
        }

        expand_put_object_intent_preconditions(task_intent, observation)

        self.assertEqual(
            task_intent["intentSteps"],
            [
                {"order": 1, "action": "OpenObject", "objectType": "Drawer", "targetType": None},
                {"order": 2, "action": "PickupObject", "objectType": "TomatoSliced", "targetType": None},
                {"order": 3, "action": "PutObject", "objectType": "TomatoSliced", "targetType": "Drawer"},
            ],
        )

    def test_put_intent_does_not_pickup_when_robot_already_holds_object(self) -> None:
        task_intent = {
            "requestedAction": "PutObject",
            "requestedObjectType": "TomatoSliced",
            "intentSteps": [
                {"order": 1, "action": "PutObject", "objectType": "TomatoSliced", "targetType": "CounterTop"}
            ],
        }
        observation = {
            "robot_id": 0,
            "objects": [{"id": "CounterTop|1", "type": "CounterTop", "visible": True, "receptacle": True}],
            "inventory": [{"objectId": "TomatoSliced|1", "objectType": "TomatoSliced"}],
            "held_object": {"objectId": "TomatoSliced|1", "objectType": "TomatoSliced"},
        }

        warnings = expand_put_object_intent_preconditions(task_intent, observation)

        self.assertEqual(
            task_intent["intentSteps"],
            [{"order": 1, "action": "PutObject", "objectType": "TomatoSliced", "targetType": "CounterTop"}],
        )
        self.assertEqual(warnings, [])

    def test_put_intent_does_not_open_receptacle_that_is_already_open(self) -> None:
        task_intent = {
            "requestedAction": "PutObject",
            "requestedObjectType": "TomatoSliced",
            "intentSteps": [
                {"order": 1, "action": "PutObject", "objectType": "TomatoSliced", "targetType": "Drawer"}
            ],
        }
        observation = {
            "robot_id": 0,
            "objects": [
                {"id": "Drawer|1", "type": "Drawer", "visible": True, "receptacle": True, "openable": True, "isOpen": True}
            ],
            "inventory": [{"objectId": "TomatoSliced|1", "objectType": "TomatoSliced"}],
            "held_object": {"objectId": "TomatoSliced|1", "objectType": "TomatoSliced"},
        }

        expand_put_object_intent_preconditions(task_intent, observation)

        self.assertEqual(
            task_intent["intentSteps"],
            [{"order": 1, "action": "PutObject", "objectType": "TomatoSliced", "targetType": "Drawer"}],
        )

    def test_pickup_step_is_satisfied_when_executor_already_holds_object(self) -> None:
        step = {"order": 1, "action": "PickupObject", "objectType": "TomatoSliced", "targetType": None}
        observation = {
            "agent_id": "robot_0",
            "robot_id": 0,
            "objects": [],
            "inventory": [{"objectId": "TomatoSliced|1", "objectType": "TomatoSliced"}],
            "held_object": {"objectId": "TomatoSliced|1", "objectType": "TomatoSliced"},
        }

        self.assertTrue(pickup_step_already_satisfied(step, observation))

    def test_open_close_step_is_satisfied_when_object_state_already_matches(self) -> None:
        observation = {
            "agent_id": "robot_0",
            "robot_id": 0,
            "objects": [
                {"id": "Cabinet|1", "type": "Cabinet", "visible": True, "openable": True, "isOpen": True},
                {"id": "Drawer|1", "type": "Drawer", "visible": True, "openable": True, "isOpen": False},
            ],
        }

        self.assertTrue(object_state_step_already_satisfied({"action": "OpenObject", "objectType": "Cabinet"}, observation))
        self.assertTrue(object_state_step_already_satisfied({"action": "CloseObject", "objectType": "Drawer"}, observation))
        self.assertFalse(object_state_step_already_satisfied({"action": "OpenObject", "objectType": "Drawer"}, observation))

    def test_put_executor_can_be_selected_by_held_object_when_not_visible(self) -> None:
        observations = [
            {"agent_id": "robot_2", "robot_id": 2, "is_primary": True, "objects": [], "inventory": []},
            {
                "agent_id": "robot_0",
                "robot_id": 0,
                "is_primary": False,
                "objects": [
                    {"id": "CounterTop|1", "type": "CounterTop", "visible": True, "receptacle": True}
                ],
                "inventory": [{"objectId": "TomatoSliced|1", "objectType": "TomatoSliced"}],
                "held_object": {"objectId": "TomatoSliced|1", "objectType": "TomatoSliced"},
            },
        ]
        visibility_map = build_object_visibility_map(observations)
        semantic_plan = {
            "targetObjectType": "TomatoSliced",
            "plan": [{"action": "PutObject", "objectType": "TomatoSliced", "targetType": "CounterTop"}],
        }
        task_intent = {
            "requestedAction": "PutObject",
            "requestedObjectType": "TomatoSliced",
            "intentSteps": [
                {"order": 1, "action": "PutObject", "objectType": "TomatoSliced", "targetType": "CounterTop"}
            ],
        }

        result = choose_relay_executor(
            "put the tomatosliced on the counter",
            semantic_plan,
            visibility_map,
            observations,
            task_intent,
        )

        self.assertEqual(result["status"], "executor_selected")
        self.assertEqual(result["executor_agent_id"], "robot_0")
        self.assertIn("already held", result["reason"])

    def test_put_executor_reports_target_receptacle_not_visible_for_holder(self) -> None:
        observations = [
            {"agent_id": "robot_2", "robot_id": 2, "is_primary": True, "objects": [], "inventory": []},
            {
                "agent_id": "robot_0",
                "robot_id": 0,
                "is_primary": False,
                "objects": [],
                "inventory": [{"objectId": "TomatoSliced|1", "objectType": "TomatoSliced"}],
                "held_object": {"objectId": "TomatoSliced|1", "objectType": "TomatoSliced"},
            },
        ]
        visibility_map = build_object_visibility_map(observations)
        semantic_plan = {
            "targetObjectType": "TomatoSliced",
            "plan": [{"action": "PutObject", "objectType": "TomatoSliced", "targetType": "CounterTop"}],
        }
        task_intent = {
            "requestedAction": "PutObject",
            "requestedObjectType": "TomatoSliced",
            "intentSteps": [
                {"order": 1, "action": "PutObject", "objectType": "TomatoSliced", "targetType": "CounterTop"}
            ],
        }

        result = choose_relay_executor(
            "put the tomatosliced on the counter",
            semantic_plan,
            visibility_map,
            observations,
            task_intent,
        )

        self.assertEqual(result["status"], "needs_upstream_planning")
        self.assertIn("target receptacle 'CounterTop' is not visible to robot 0", result["reason"])

    def test_choose_relay_executor_selects_nearest_visible_candidate(self) -> None:
        observations = [
            {
                "agent_id": "robot_0",
                "robot_id": 0,
                "is_primary": True,
                "robot_state": {"position": {"x": 10, "y": 0, "z": 0}},
                "objects": [{"id": "Apple|1", "type": "Apple", "visible": True, "pickupable": True, "position": {"x": 0, "y": 0, "z": 0}}],
            },
            {
                "agent_id": "robot_1",
                "robot_id": 1,
                "is_primary": False,
                "robot_state": {"position": {"x": 1, "y": 0, "z": 0}},
                "objects": [{"id": "Apple|1", "type": "Apple", "visible": True, "pickupable": True, "position": {"x": 0, "y": 0, "z": 0}}],
            },
            {
                "agent_id": "robot_2",
                "robot_id": 2,
                "is_primary": False,
                "robot_state": {"position": {"x": 3, "y": 0, "z": 0}},
                "objects": [{"id": "Apple|1", "type": "Apple", "visible": True, "pickupable": True, "position": {"x": 0, "y": 0, "z": 0}}],
            },
        ]
        semantic_plan = {"targetObjectType": "Apple", "plan": [{"action": "PickupObject", "objectType": "Apple"}]}

        result = choose_relay_executor("pick up the apple", semantic_plan, build_object_visibility_map(observations), observations)

        self.assertEqual(result["status"], "executor_selected")
        self.assertEqual(result["executor_robot_id"], 1)
        self.assertEqual(result["candidate_executor_robot_ids"], [1, 2, 0])
        self.assertEqual(result["selected_distance_to_target"], 1.0)
        self.assertIn("closest", result["reason"])

    def test_choose_relay_executor_skips_nearest_candidate_that_fails_validation(self) -> None:
        observations = [
            {
                "agent_id": "robot_0",
                "robot_id": 0,
                "is_primary": True,
                "robot_state": {"position": {"x": 10, "y": 0, "z": 0}},
                "objects": [],
            },
            {
                "agent_id": "robot_1",
                "robot_id": 1,
                "is_primary": False,
                "robot_state": {"position": {"x": 1, "y": 0, "z": 0}},
                "objects": [{"id": "Apple|1", "type": "Apple", "visible": True, "pickupable": False, "position": {"x": 0, "y": 0, "z": 0}}],
            },
            {
                "agent_id": "robot_2",
                "robot_id": 2,
                "is_primary": False,
                "robot_state": {"position": {"x": 3, "y": 0, "z": 0}},
                "objects": [{"id": "Apple|1", "type": "Apple", "visible": True, "pickupable": True, "position": {"x": 0, "y": 0, "z": 0}}],
            },
        ]
        semantic_plan = {"targetObjectType": "Apple", "plan": [{"action": "PickupObject", "objectType": "Apple"}]}

        result = choose_relay_executor("pick up the apple", semantic_plan, build_object_visibility_map(observations), observations)

        self.assertEqual(result["status"], "executor_selected")
        self.assertEqual(result["executor_robot_id"], 2)
        self.assertFalse(next(item for item in result["candidate_scores"] if item["robot_id"] == 1)["executable"])
        self.assertEqual(result["candidate_executor_robot_ids"], [2])

    def test_choose_relay_executor_falls_back_to_primary_when_distance_missing(self) -> None:
        observations = [
            {
                "agent_id": "robot_0",
                "robot_id": 0,
                "is_primary": True,
                "objects": [{"id": "Apple|0", "type": "Apple", "visible": True, "pickupable": True}],
            },
            {
                "agent_id": "robot_1",
                "robot_id": 1,
                "is_primary": False,
                "objects": [{"id": "Apple|1", "type": "Apple", "visible": True, "pickupable": True}],
            },
        ]
        semantic_plan = {"targetObjectType": "Apple", "plan": [{"action": "PickupObject", "objectType": "Apple"}]}

        result = choose_relay_executor("pick up the apple", semantic_plan, build_object_visibility_map(observations), observations)

        self.assertEqual(result["status"], "executor_selected")
        self.assertEqual(result["executor_robot_id"], 0)
        self.assertIsNone(result["selected_distance_to_target"])
        self.assertEqual(result["candidate_executor_robot_ids"], [0, 1])

    def test_choose_relay_executor_put_selects_holder_nearest_receptacle(self) -> None:
        observations = [
            {
                "agent_id": "robot_0",
                "robot_id": 0,
                "is_primary": True,
                "robot_state": {"position": {"x": 5, "y": 0, "z": 0}},
                "objects": [{"id": "CounterTop|1", "type": "CounterTop", "visible": True, "receptacle": True, "position": {"x": 0, "y": 0, "z": 0}}],
                "inventory": [{"objectId": "Tomato|1", "objectType": "Tomato"}],
                "held_object": {"objectId": "Tomato|1", "objectType": "Tomato"},
            },
            {
                "agent_id": "robot_2",
                "robot_id": 2,
                "is_primary": False,
                "robot_state": {"position": {"x": 1, "y": 0, "z": 0}},
                "objects": [{"id": "CounterTop|1", "type": "CounterTop", "visible": True, "receptacle": True, "position": {"x": 0, "y": 0, "z": 0}}],
                "inventory": [{"objectId": "Tomato|1", "objectType": "Tomato"}],
                "held_object": {"objectId": "Tomato|1", "objectType": "Tomato"},
            },
        ]
        semantic_plan = {"targetObjectType": "Tomato", "plan": [{"action": "PutObject", "objectType": "Tomato", "targetType": "CounterTop"}]}
        task_intent = {"requestedAction": "PutObject", "requestedObjectType": "Tomato", "intentSteps": [{"order": 1, "action": "PutObject", "objectType": "Tomato", "targetType": "CounterTop"}]}

        result = choose_relay_executor("put the tomato on the counter", semantic_plan, build_object_visibility_map(observations), observations, task_intent)

        self.assertEqual(result["status"], "executor_selected")
        self.assertEqual(result["executor_robot_id"], 2)
        self.assertEqual(result["candidate_executor_robot_ids"], [2, 0])
        self.assertIn("already held", result["reason"])
        self.assertIn("closest", result["reason"])

    def test_choose_relay_executor_reports_when_visible_candidates_are_not_executable(self) -> None:
        observations = [
            {
                "agent_id": "robot_0",
                "robot_id": 0,
                "is_primary": True,
                "objects": [{"id": "Apple|1", "type": "Apple", "visible": True, "pickupable": False}],
            },
            {
                "agent_id": "robot_1",
                "robot_id": 1,
                "is_primary": False,
                "objects": [{"id": "Apple|2", "type": "Apple", "visible": True, "pickupable": False}],
            },
        ]
        semantic_plan = {"targetObjectType": "Apple", "plan": [{"action": "PickupObject", "objectType": "Apple"}]}

        result = choose_relay_executor("pick up the apple", semantic_plan, build_object_visibility_map(observations), observations)

        self.assertEqual(result["status"], "needs_upstream_planning")
        self.assertEqual(result["candidate_executor_robot_ids"], [])
        self.assertEqual(len(result["candidate_scores"]), 2)
        self.assertIn("none can execute", result["reason"])

    def test_evaluate_relay_executor_candidates_exposes_nearest_without_selecting(self) -> None:
        observations = [
            {
                "agent_id": "robot_0",
                "robot_id": 0,
                "is_primary": True,
                "robot_state": {"position": {"x": 5, "y": 0, "z": 0}},
                "objects": [{"id": "Apple|0", "type": "Apple", "visible": True, "pickupable": True, "position": {"x": 0, "y": 0, "z": 0}}],
            },
            {
                "agent_id": "robot_2",
                "robot_id": 2,
                "is_primary": False,
                "robot_state": {"position": {"x": 1, "y": 0, "z": 0}},
                "objects": [{"id": "Apple|0", "type": "Apple", "visible": True, "pickupable": True, "position": {"x": 0, "y": 0, "z": 0}}],
            },
        ]
        semantic_plan = {"targetObjectType": "Apple", "plan": [{"action": "PickupObject", "objectType": "Apple"}]}

        result = evaluate_relay_executor_candidates(
            "pick up the apple",
            semantic_plan,
            None,
            build_object_visibility_map(observations),
            observations,
            [0, 2],
            0,
        )

        self.assertEqual(result["selection_policy"], "llm_tool_calling_with_hard_validation")
        self.assertEqual(result["candidate_executor_robot_ids"], [2, 0])
        self.assertEqual(result["candidate_scores"][0]["robot_id"], 2)
        self.assertEqual(result["candidate_scores"][0]["distance_to_target"], 1.0)
        self.assertIn("relay agent makes the final executor choice", result["evidence_policy"])

    def test_evaluate_relay_executor_candidates_keeps_failed_nearest_as_evidence(self) -> None:
        observations = [
            {
                "agent_id": "robot_1",
                "robot_id": 1,
                "is_primary": False,
                "robot_state": {"position": {"x": 1, "y": 0, "z": 0}},
                "objects": [{"id": "Apple|0", "type": "Apple", "visible": True, "pickupable": False, "position": {"x": 0, "y": 0, "z": 0}}],
            },
            {
                "agent_id": "robot_2",
                "robot_id": 2,
                "is_primary": False,
                "robot_state": {"position": {"x": 3, "y": 0, "z": 0}},
                "objects": [{"id": "Apple|0", "type": "Apple", "visible": True, "pickupable": True, "position": {"x": 0, "y": 0, "z": 0}}],
            },
        ]
        semantic_plan = {"targetObjectType": "Apple", "plan": [{"action": "PickupObject", "objectType": "Apple"}]}

        result = evaluate_relay_executor_candidates(
            "pick up the apple",
            semantic_plan,
            None,
            build_object_visibility_map(observations),
            observations,
            [1, 2],
            1,
        )

        self.assertEqual(result["candidate_executor_robot_ids"], [2])
        failed_nearest = next(item for item in result["candidate_scores"] if item["robot_id"] == 1)
        self.assertFalse(failed_nearest["executable"])
        self.assertEqual(failed_nearest["distance_to_target"], 1.0)
        self.assertIn("pickupable", failed_nearest["validation"])

    def test_put_goal_consistency_allows_held_object_that_is_not_visible(self) -> None:
        semantic_plan = {
            "targetObjectType": "TomatoSliced",
            "plan": [{"action": "PutObject", "objectType": "TomatoSliced", "targetType": "CounterTop"}],
        }
        observation = {
            "agent_id": "robot_0",
            "robot_id": 0,
            "objects": [{"id": "CounterTop|1", "type": "CounterTop", "visible": True, "receptacle": True}],
            "inventory": [{"objectId": "TomatoSliced|1", "objectType": "TomatoSliced"}],
            "held_object": {"objectId": "TomatoSliced|1", "objectType": "TomatoSliced"},
        }

        validate_put_object_goal_consistency(semantic_plan, observation)

    def test_put_goal_consistency_rejects_when_object_not_held(self) -> None:
        semantic_plan = {
            "targetObjectType": "TomatoSliced",
            "plan": [{"action": "PutObject", "objectType": "TomatoSliced", "targetType": "CounterTop"}],
        }
        observation = {
            "agent_id": "robot_0",
            "robot_id": 0,
            "objects": [{"id": "CounterTop|1", "type": "CounterTop", "visible": True, "receptacle": True}],
            "inventory": [],
        }

        with self.assertRaisesRegex(ValueError, "robot 0 is not holding TomatoSliced.*PutObject"):
            validate_put_object_goal_consistency(semantic_plan, observation)

    def test_executor_validation_allows_put_when_held_object_is_not_visible(self) -> None:
        semantic_plan = {
            "targetObjectType": "TomatoSliced",
            "plan": [{"action": "PutObject", "objectType": "TomatoSliced", "targetType": "CounterTop"}],
        }
        observation = {
            "agent_id": "robot_0",
            "robot_id": 0,
            "objects": [{"id": "CounterTop|1", "type": "CounterTop", "visible": True, "receptacle": True}],
            "inventory": [{"objectId": "TomatoSliced|1", "objectType": "TomatoSliced"}],
            "held_object": {"objectId": "TomatoSliced|1", "objectType": "TomatoSliced"},
        }
        relay_result = {"requested_object_type": "TomatoSliced", "primary_agent_id": "robot_2"}

        failure = validate_executor_plan_or_failure(
            "put the tomatosliced on the counter",
            semantic_plan,
            observation["objects"],
            allow_invisible=False,
            relay_result=relay_result,
            agent_observations=[{"agent_id": "robot_2", "robot_id": 2, "is_primary": True}, observation],
            task_intent={
                "requestedAction": "PutObject",
                "requestedObjectType": "TomatoSliced",
                "intentSteps": [
                    {"order": 1, "action": "PutObject", "objectType": "TomatoSliced", "targetType": "CounterTop"}
                ],
            },
            executor_observation=observation,
        )

        self.assertIsNone(failure)

    def test_executor_validation_allows_put_with_navigation_when_held_object_is_not_visible(self) -> None:
        semantic_plan = {
            "targetObjectType": "TomatoSliced",
            "plan": [
                {"action": "MoveRight", "objectType": None, "targetType": None},
                {"action": "PutObject", "objectType": "TomatoSliced", "targetType": "CounterTop"},
            ],
        }
        observation = {
            "agent_id": "robot_0",
            "robot_id": 0,
            "objects": [{"id": "CounterTop|1", "type": "CounterTop", "visible": True, "receptacle": True}],
            "inventory": [{"objectId": "TomatoSliced|1", "objectType": "TomatoSliced"}],
            "held_object": {"objectId": "TomatoSliced|1", "objectType": "TomatoSliced"},
        }
        relay_result = {"requested_object_type": "TomatoSliced", "primary_agent_id": "robot_2"}

        failure = validate_executor_plan_or_failure(
            "put the tomatosliced on the counter",
            semantic_plan,
            observation["objects"],
            allow_invisible=False,
            relay_result=relay_result,
            agent_observations=[{"agent_id": "robot_2", "robot_id": 2, "is_primary": True}, observation],
            task_intent={
                "requestedAction": "PutObject",
                "requestedObjectType": "TomatoSliced",
                "intentSteps": [
                    {"order": 1, "action": "PutObject", "objectType": "TomatoSliced", "targetType": "CounterTop"}
                ],
            },
            executor_observation=observation,
        )

        self.assertIsNone(failure)

    def test_put_coordination_uses_held_peer_even_when_object_not_visible(self) -> None:
        observations = [
            {"agent_id": "robot_2", "robot_id": 2, "is_primary": True, "objects": [], "inventory": []},
            {
                "agent_id": "robot_0",
                "robot_id": 0,
                "is_primary": False,
                "objects": [{"id": "CounterTop|1", "type": "CounterTop", "visible": True, "receptacle": True}],
                "inventory": [{"objectId": "TomatoSliced|1", "objectType": "TomatoSliced"}],
                "held_object": {"objectId": "TomatoSliced|1", "objectType": "TomatoSliced"},
            },
        ]
        visibility_map = build_object_visibility_map(observations)
        semantic_plan = {
            "targetObjectType": "TomatoSliced",
            "plan": [{"action": "PutObject", "objectType": "TomatoSliced", "targetType": "CounterTop"}],
        }
        task_intent = {
            "requestedAction": "PutObject",
            "requestedObjectType": "TomatoSliced",
            "intentSteps": [
                {"order": 1, "action": "PutObject", "objectType": "TomatoSliced", "targetType": "CounterTop"}
            ],
        }

        result = coordination_result_for_plan(
            "put the tomatosliced on the counter",
            semantic_plan,
            visibility_map,
            task_intent,
            relay_mode=True,
        )

        self.assertEqual(result["status"], "target_visible_by_peer")
        self.assertEqual(result["held_by_agent_ids"], ["robot_0"])

    def test_closed_loop_put_uses_simulated_held_owner(self) -> None:
        observations = [
            {"agent_id": "robot_2", "robot_id": 2, "is_primary": True, "objects": [], "inventory": []},
            {
                "agent_id": "robot_0",
                "robot_id": 0,
                "is_primary": False,
                "objects": [{"id": "CounterTop|1", "type": "CounterTop", "visible": True, "receptacle": True}],
                "inventory": [],
            },
        ]
        visibility_map = build_object_visibility_map(observations)
        step = {"order": 2, "action": "PutObject", "objectType": "TomatoSliced", "targetType": "CounterTop"}

        result = relay_result_for_held_put_step(
            step,
            visibility_map,
            observations,
            {"robot_0": "TomatoSliced"},
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["status"], "executor_selected")
        self.assertEqual(result["executor_agent_id"], "robot_0")

    def test_closed_loop_put_simulated_holder_reports_missing_target_receptacle(self) -> None:
        observations = [
            {"agent_id": "robot_2", "robot_id": 2, "is_primary": True, "objects": [], "inventory": []},
            {"agent_id": "robot_0", "robot_id": 0, "is_primary": False, "objects": [], "inventory": []},
        ]
        visibility_map = build_object_visibility_map(observations)
        step = {"order": 2, "action": "PutObject", "objectType": "TomatoSliced", "targetType": "CounterTop"}

        result = relay_result_for_held_put_step(
            step,
            visibility_map,
            observations,
            {"robot_0": "TomatoSliced"},
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["status"], "needs_upstream_planning")
        self.assertIn("target receptacle 'CounterTop' is not visible to robot 0", result["reason"])

    def test_rejects_open_when_object_is_already_open(self) -> None:
        semantic_plan = {"plan": [{"action": "OpenObject", "objectType": "Fridge", "targetType": None}]}
        observation = {
            "objects": [{"id": "Fridge|1", "type": "Fridge", "visible": True, "openable": True, "isOpen": True}],
            "inventory": [],
        }

        with self.assertRaisesRegex(ValueError, "already open.*OpenObject"):
            validate_action_state_preconditions(semantic_plan, observation, allow_invisible=False)

    def test_rejects_close_when_object_is_already_closed(self) -> None:
        semantic_plan = {"plan": [{"action": "CloseObject", "objectType": "Fridge", "targetType": None}]}
        observation = {
            "objects": [{"id": "Fridge|1", "type": "Fridge", "visible": True, "openable": True, "isOpen": False}],
            "inventory": [],
        }

        with self.assertRaisesRegex(ValueError, "already closed.*CloseObject"):
            validate_action_state_preconditions(semantic_plan, observation, allow_invisible=False)

    def test_simulates_state_across_multi_action_plan(self) -> None:
        semantic_plan = {
            "plan": [
                {"action": "PickupObject", "objectType": "Tomato", "targetType": None},
                {"action": "PutObject", "objectType": "Tomato", "targetType": "CounterTop"},
            ]
        }
        observation = {
            "objects": [
                {"id": "Tomato|1", "type": "Tomato", "visible": True, "pickupable": True},
                {"id": "CounterTop|1", "type": "CounterTop", "visible": True, "receptacle": True},
            ],
            "inventory": [],
        }

        self.assertIsNone(validate_action_state_preconditions(semantic_plan, observation, allow_invisible=False))



class UnifiedPlanningTest(unittest.TestCase):
    def test_semantic_prompt_uses_concrete_intent_in_minimal_example(self) -> None:
        task_intent = {
            "requestedAction": "PickupObject",
            "requestedObjectType": "CreditCard",
            "intentSteps": [
                {
                    "order": 1,
                    "action": "PickupObject",
                    "objectType": "CreditCard",
                    "targetType": None,
                }
            ],
        }

        prompt = semantic_planning_prompt(
            "image",
            "pick up CreditCard",
            task_intent=task_intent,
        )

        self.assertIn('"targetObjectType": "CreditCard"', prompt)
        self.assertIn('"objectType": "CreditCard"', prompt)
        self.assertNotIn('"ObjectType"', prompt)

    def test_repairs_only_explicit_semantic_object_placeholder(self) -> None:
        semantic_plan = {
            "targetObjectType": "ObjectType",
            "plan": [
                {"action": "PickupObject", "objectType": "ObjectType", "targetType": None}
            ],
        }
        step_intent = {
            "requestedAction": "PickupObject",
            "requestedObjectType": "CreditCard",
            "intentSteps": [
                {
                    "order": 1,
                    "action": "PickupObject",
                    "objectType": "CreditCard",
                    "targetType": None,
                }
            ],
        }

        warnings = repair_semantic_placeholders_from_step_intent(semantic_plan, step_intent)

        self.assertEqual(semantic_plan["targetObjectType"], "CreditCard")
        self.assertEqual(semantic_plan["plan"][0]["objectType"], "CreditCard")
        self.assertEqual(len(warnings), 2)

        wrong_real_type = {
            "targetObjectType": "Apple",
            "plan": [{"action": "PickupObject", "objectType": "Apple", "targetType": None}],
        }
        self.assertEqual(
            repair_semantic_placeholders_from_step_intent(wrong_real_type, step_intent),
            [],
        )
        self.assertEqual(wrong_real_type["plan"][0]["objectType"], "Apple")

    def test_native_prompt_declares_plan_without_top_level_actions(self) -> None:
        prompt = native_planning_prompt("image", "Move the apple")
        self.assertIn('"plan": [', prompt)
        self.assertNotIn('"actions":', prompt)

    def test_semantic_rejects_top_level_actions(self) -> None:
        document = {
            "task": "move",
            "needsGrounding": True,
            "observations": [],
            "plan": [],
            "actions": [{"action": "MoveAhead"}],
        }
        with self.assertRaisesRegex(ValueError, "top-level"):
            parse_semantic_planning_output(json.dumps(document))

    def test_semantic_plan_still_validates(self) -> None:
        document = {
            "task": "move the apple",
            "targetObjectType": "Apple",
            "needsGrounding": True,
            "observations": [
                {
                    "order": 1,
                    "eventType": "moved_object",
                    "objectType": "Apple",
                    "event": "picked up",
                    "targetType": None,
                }
            ],
            "plan": [
                {"action": "PickupObject", "objectType": "Apple", "targetType": None}
            ],
        }
        self.assertEqual(parse_semantic_planning_output(json.dumps(document)), document)

    def test_semantic_accepts_target_object_type(self) -> None:
        document = {
            "task": "pick up the banana",
            "targetObjectType": "Banana",
            "needsGrounding": True,
            "observations": [],
            "plan": [{"action": "Done", "objectType": None, "targetType": None}],
        }

        self.assertEqual(parse_semantic_planning_output(json.dumps(document)), document)

    def test_semantic_normalizes_missing_safe_fields(self) -> None:
        document = {
            "task": "pick up the egg",
            "observations": [
                {
                    "order": 1,
                    "eventType": "moved_object",
                    "objectType": "Egg",
                    "event": "picked up",
                }
            ],
            "plan": [{"action": "PickupObject", "objectType": "Egg"}],
        }

        parsed = parse_semantic_planning_output(json.dumps(document))

        self.assertIs(parsed["needsGrounding"], True)
        self.assertIsNone(parsed["targetObjectType"])
        self.assertIsNone(parsed["observations"][0]["targetType"])
        self.assertIsNone(parsed["plan"][0]["targetType"])

    def test_semantic_normalizes_picked_up_event_type(self) -> None:
        document = {
            "task": "pick up the tomato",
            "targetObjectType": "Tomato",
            "needsGrounding": True,
            "observations": [
                {
                    "order": 1,
                    "eventType": "state_changed_object",
                    "objectType": "Tomato",
                    "event": "picked up",
                    "targetType": None,
                }
            ],
            "plan": [{"action": "PickupObject", "objectType": "Tomato", "targetType": None}],
        }

        parsed = parse_semantic_planning_output(json.dumps(document))

        self.assertEqual(parsed["observations"][0]["eventType"], "moved_object")
        self.assertIn("semanticNormalizationWarnings", parsed)

    def test_semantic_normalizes_opened_event_type(self) -> None:
        document = {
            "task": "open the fridge",
            "targetObjectType": "Fridge",
            "needsGrounding": True,
            "observations": [
                {
                    "order": 1,
                    "eventType": "moved_object",
                    "objectType": "Fridge",
                    "event": "opened",
                    "targetType": None,
                }
            ],
            "plan": [{"action": "OpenObject", "objectType": "Fridge", "targetType": None}],
        }

        parsed = parse_semantic_planning_output(json.dumps(document))

        self.assertEqual(parsed["observations"][0]["eventType"], "state_changed_object")
        self.assertIn("semanticNormalizationWarnings", parsed)

    def test_semantic_still_rejects_missing_required_observation_fields(self) -> None:
        document = {
            "task": "pick up the egg",
            "observations": [
                {
                    "order": 1,
                    "eventType": "moved_object",
                    "event": "picked up",
                }
            ],
            "plan": [{"action": "PickupObject", "objectType": "Egg"}],
        }
        with self.assertRaisesRegex(ValueError, "objectType"):
            parse_semantic_planning_output(json.dumps(document))

    def test_semantic_still_rejects_false_needs_grounding(self) -> None:
        document = {
            "task": "pick up the egg",
            "needsGrounding": False,
            "observations": [],
            "plan": [],
        }
        with self.assertRaisesRegex(ValueError, "needsGrounding"):
            parse_semantic_planning_output(json.dumps(document))

    def test_semantic_prompt_shows_required_safe_fields(self) -> None:
        prompt = semantic_planning_prompt("image", "Pick up the egg")
        self.assertIn('"needsGrounding": true', prompt)
        self.assertIn('"targetObjectType": null', prompt)
        self.assertIn('"targetType": null', prompt)
        self.assertNotIn('"ObjectType"', prompt)
        self.assertIn("Never omit needsGrounding, targetObjectType, or targetType", prompt)
        self.assertIn("Do not replace targetObjectType", prompt)
        self.assertIn("Use PickupObject only for pickupable small objects", prompt)
        self.assertIn("Do not replace the user's requested action", prompt)
        self.assertIn("Use moved_object for pickup/place/put down/push/pull/drop events", prompt)
        self.assertNotIn('"objectType": "Egg"', prompt)

    def test_semantic_plan_grounds_to_http_actions(self) -> None:
        semantic_plan = {
            "plan": [
                {"action": "PickupObject", "objectType": "Apple", "targetType": None},
                {"action": "PutObject", "objectType": "Apple", "targetType": "CounterTop"},
            ]
        }
        objects = [
            {"id": "Apple|1", "type": "Apple", "visible": True, "pickupable": True},
            {"id": "CounterTop|1", "type": "CounterTop", "visible": True, "receptacle": True},
        ]
        self.assertEqual(
            ground_semantic_plan(semantic_plan, objects, allow_invisible=False, max_actions=20),
            [
                {"action": "PickupObject", "objectId": "Apple|1", "forceAction": True},
                {"action": "PutObject", "objectId": "CounterTop|1", "forceAction": True},
                {"action": "Done"},
            ],
        )

    def test_video_question_remains_natural_language(self) -> None:
        prompt = question_prompt("What happens next?")
        self.assertIn("What happens next?", prompt)
        self.assertNotIn('"plan"', prompt)


if __name__ == "__main__":
    unittest.main()
