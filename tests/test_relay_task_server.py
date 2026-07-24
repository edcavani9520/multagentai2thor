import contextlib
import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import relay_task_server
from relay_task_server import RelayRuntimeConfig, RelayTaskService, model_shard_error


class FakeEngine:
    def __init__(self):
        self.argv = None

    def parse_args(self, argv):
        self.argv = argv
        return Namespace()

    def run(self, args):
        assert getattr(args, "_qwen_backend") is not None
        print(json.dumps({"task_id": "task-1", "closed_loop_result": {"status": "success", "step_count": 2}}))
        return 0


class FakeNormalizerBackend:
    def __init__(self, arguments):
        self.arguments = arguments
        self.calls = []

    def generate_with_tools(self, messages, tools):
        self.calls.append({"messages": messages, "tools": tools})
        return json.dumps(
            {
                "name": "normalize_incoming_task",
                "arguments": self.arguments,
            }
        )


class SequentialToolBackend:
    def __init__(self, *arguments):
        self.arguments = list(arguments)
        self.calls = []

    def generate_with_tools(self, messages, tools):
        self.calls.append({"messages": messages, "tools": tools})
        index = min(len(self.calls) - 1, len(self.arguments) - 1)
        arguments = self.arguments[index] if self.arguments else {}
        return json.dumps({"name": "normalize_incoming_task", "arguments": arguments})


def service_config(temp_dir: str | Path = "/tmp") -> RelayRuntimeConfig:
    return RelayRuntimeConfig(
        receiver_url="http://127.0.0.1:19000",
        model_path="models/Qwen3.5-4B",
        device="cuda",
        device_map="auto",
        dtype="float16",
        max_new_tokens=64,
        temperature=0.1,
        send_timeout=60.0,
        output_dir=Path(temp_dir),
        max_replan_steps=10,
        relay_agent_max_turns=8,
        max_actions=8,
    )


class RelayTaskServiceTest(unittest.TestCase):
    def test_parse_task_normalizer_ignores_tool_schema_block(self):
        output = (
            'system\n<tools>{"type":"function","function":{"name":"normalize_incoming_task",'
            '"parameters":{"type":"object"}}}</tools>\n'
            '<tool_call>{"name":"normalize_incoming_task","arguments":'
            '{"normalized_task":"look down.","intentSteps":[{"order":1,"action":"LookDown","objectType":null,"targetType":null}],"confidence":"high","reason":"preserve requested look action"}}}</tool_call>'
        )
        tool_call = relay_task_server.parse_task_normalizer_tool_call(output)
        self.assertEqual(tool_call["name"], "normalize_incoming_task")
        self.assertEqual(tool_call["arguments"]["normalized_task"], "look down.")
        self.assertEqual(tool_call["arguments"]["intentSteps"][0]["action"], "LookDown")

    def test_parse_task_normalizer_accepts_qwen_parameter_tool_call(self):
        output = (
            '<tool_call>\n<function=normalize_incoming_task>\n'
            '<parameter=normalized_task>\nOpen the fridge and look down.\n</parameter>\n'
            '<parameter=intentSteps>\n'
            '[{"order": 1, "action": "OpenObject", "objectType": "Fridge", "targetType": null}, '
            '{"order": 2, "action": "LookDown", "objectType": null, "targetType": null}]\n'
            '</parameter>\n'
            '<parameter=confidence>\nhigh\n</parameter>\n'
            '<parameter=reason>\nThe task explicitly requests these two actions.'
        )
        tool_call = relay_task_server.parse_task_normalizer_tool_call(output)
        self.assertEqual(tool_call["name"], "normalize_incoming_task")
        self.assertEqual(tool_call["arguments"]["normalized_task"], "Open the fridge and look down.")
        self.assertEqual(tool_call["arguments"]["confidence"], "high")
        self.assertEqual(tool_call["arguments"]["intentSteps"][1]["action"], "LookDown")

    def test_builds_closed_loop_relay_request(self):
        old_fetch = relay_task_server.fetch_receiver_state_object_types
        relay_task_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Bread", "CounterTop"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                backend = FakeNormalizerBackend(
                    {
                        "normalized_task": "put the Bread on the CounterTop.",
                        "intentSteps": [
                            {"order": 1, "action": "PutObject", "objectType": "Bread", "targetType": "CounterTop"},
                        ],
                        "confidence": "high",
                        "reason": "the task asks to put Bread on CounterTop",
                    }
                )
                service = RelayTaskService(engine, backend, service_config(temp_dir))
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task(
                        {
                            "task_id": "task-1",
                            "task": "put the bread on the countertop",
                            "primary_robot_id": 0,
                            "known_robot_ids": [0, 1],
                            "dry_run": True,
                        }
                    )
        finally:
            relay_task_server.fetch_receiver_state_object_types = old_fetch

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["result"]["closed_loop_result"]["step_count"], 2)
        self.assertIn("--relay-mode", engine.argv)
        self.assertIn("--closed-loop-replan", engine.argv)
        self.assertIn("--save-raw-output", engine.argv)
        self.assertIn("http://127.0.0.1:19000/execute_actions", engine.argv)
        self.assertIn("0,1", engine.argv)
        self.assertIn("--dry-run", engine.argv)


    def test_normalizer_prompt_includes_recognized_action_sequence_constraint(self):
        messages = relay_task_server.task_normalizer_messages("close the fridge.", ["Fridge"])
        prompt = messages[0]["content"][0]["text"]
        self.assertIn("exactly these action names in this order", prompt)
        self.assertIn("CloseObject", prompt)
        self.assertIn("no extra actions", prompt)

    def test_normalizes_planning_find_subtask_with_llm_tool(self):
        old_fetch = relay_task_server.fetch_receiver_state_object_types
        relay_task_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Fridge", "Cabinet"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                backend = FakeNormalizerBackend(
                    {
                        "normalized_task": "go to the Fridge.",
                        "intentSteps": [
                            {"order": 1, "action": "GotoObject", "objectType": "Fridge", "targetType": None},
                        ],
                        "confidence": "high",
                        "reason": "planning requested finding a fridge; execute as navigation",
                    }
                )
                service = RelayTaskService(engine, backend, service_config(temp_dir))
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task(
                        {
                            "task_id": "find-fridge",
                            "task": "Find target for T1 Search/inspect the environment for unresolved task-relevant objects before executing T1: fridge find fridge",
                            "primary_robot_id": 0,
                            "known_robot_ids": [0, 1],
                        }
                    )
        finally:
            relay_task_server.fetch_receiver_state_object_types = old_fetch

        task_index = engine.argv.index("--task") + 1
        intent_index = engine.argv.index("--task-intent-json") + 1
        task_intent_payload = json.loads(engine.argv[intent_index])
        self.assertEqual(engine.argv[task_index], "go to the Fridge.")
        self.assertEqual(task_intent_payload["task_intent_source"], "qwen_normalizer_tool_call")
        self.assertEqual(task_intent_payload["task_intent"]["requestedAction"], "GotoObject")
        self.assertEqual(task_intent_payload["task_intent"]["intentSteps"][0]["action"], "GotoObject")
        self.assertTrue(response["task_normalization"]["used"])
        self.assertEqual(response["task_normalization"]["object_type"], "Fridge")
        self.assertEqual(response["status"], "success")

    def test_normalizer_preserves_multi_step_task_intent(self):
        old_fetch = relay_task_server.fetch_receiver_state_object_types
        relay_task_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Tomato", "CounterTop"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                backend = FakeNormalizerBackend(
                    {
                        "normalized_task": "pick up the Tomato and put it on the CounterTop.",
                        "requestedAction": "PickupObject",
                        "requestedObjectType": "Tomato",
                        "requestedTargetType": "CounterTop",
                        "intentSteps": [
                            {"order": 1, "action": "PickupObject", "objectType": "Tomato", "targetType": None},
                            {"order": 2, "action": "PutObject", "objectType": "Tomato", "targetType": "CounterTop"},
                        ],
                        "confidence": "high",
                        "reason": "the task asks for pickup followed by placement",
                    }
                )
                service = RelayTaskService(engine, backend, service_config(temp_dir))
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task(
                        {"task_id": "tomato-put", "task": "pick up the tomato and put it on the counter", "primary_robot_id": 0}
                    )
        finally:
            relay_task_server.fetch_receiver_state_object_types = old_fetch

        intent_index = engine.argv.index("--task-intent-json") + 1
        task_intent = json.loads(engine.argv[intent_index])["task_intent"]
        self.assertEqual(task_intent["requestedAction"], "PickupObject")
        self.assertEqual(task_intent["requestedTargetType"], "CounterTop")
        self.assertEqual(
            task_intent["intentSteps"],
            [
                {"order": 1, "action": "PickupObject", "objectType": "Tomato", "targetType": None},
                {"order": 2, "action": "PutObject", "objectType": "Tomato", "targetType": "CounterTop"},
            ],
        )
        self.assertEqual(response["task_normalization"]["intentSteps"][1]["action"], "PutObject")

    def test_normalizer_uses_tool_retry_when_tool_arguments_are_empty(self):
        old_fetch = relay_task_server.fetch_receiver_state_object_types
        relay_task_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Fridge"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                service = RelayTaskService(
                    engine,
                    SequentialToolBackend(
                        {},
                        {
                            "normalized_task": "go to the Fridge.",
                            "intentSteps": [
                                {"order": 1, "action": "GotoObject", "objectType": "Fridge", "targetType": None},
                            ],
                            "confidence": "high",
                            "reason": "retry normalized search to navigation",
                        },
                    ),
                    service_config(temp_dir),
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task({"task_id": "retry", "task": "find fridge"})
        finally:
            relay_task_server.fetch_receiver_state_object_types = old_fetch

        task_index = engine.argv.index("--task") + 1
        self.assertEqual(engine.argv[task_index], "go to the Fridge.")
        self.assertTrue(response["task_normalization"]["used"])
        self.assertEqual(response["task_normalization"]["source"], "qwen_tool_call_retry")
        self.assertIn("raw_tool_output_preview", response["task_normalization"])
        self.assertIn("raw_retry_tool_output_preview", response["task_normalization"])
        self.assertNotIn("raw_json_fallback_preview", response["task_normalization"])
        intent_index = engine.argv.index("--task-intent-json") + 1
        task_intent = json.loads(engine.argv[intent_index])["task_intent"]
        self.assertEqual(task_intent["requestedAction"], "GotoObject")
        self.assertEqual(task_intent["requestedObjectType"], "Fridge")


    def test_normalizer_failure_reason_includes_low_confidence_reason(self):
        response = relay_task_server.task_normalization_failure_response(
            "task-low-confidence",
            False,
            {
                "warnings": [],
                "confidence": "low",
                "reason": "The object 'Moon' is not present in the current AI2-THOR environment object types.",
            },
        )

        self.assertEqual(response["failure_code"], "task_normalization_failed")
        self.assertIn("confidence", response["reason"])
        self.assertIn("Moon", response["reason"])
        self.assertEqual(response["result"]["closed_loop_result"]["reason"], response["reason"])

    def test_normalizer_failure_does_not_call_runtime_or_legacy_parser(self):
        old_fetch = relay_task_server.fetch_receiver_state_object_types
        relay_task_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Fridge"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                service = RelayTaskService(engine, SequentialToolBackend({}, {}), service_config(temp_dir))
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task({"task_id": "normalizer-fail", "task": "find fridge", "dry_run": True})
        finally:
            relay_task_server.fetch_receiver_state_object_types = old_fetch

        self.assertIsNone(engine.argv)
        self.assertEqual(response["status"], "needs_upstream_planning")
        self.assertEqual(response["failure_code"], "task_normalization_failed")
        self.assertEqual(response["dry_run"], True)
        self.assertEqual(response["result"]["closed_loop_result"]["failure_code"], "task_normalization_failed")
        self.assertIn("normalizer omitted usable intentSteps", response["reason"])
        self.assertNotEqual(response["task_normalization"].get("source"), "qwen_json_fallback")


    def test_normalizer_rejects_exploratory_actions_for_find_target_and_retries_goto(self):
        old_fetch = relay_task_server.fetch_receiver_state_object_types
        relay_task_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Fridge"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                service = RelayTaskService(
                    engine,
                    SequentialToolBackend(
                        {
                            "normalized_task": "search for the Fridge.",
                            "intentSteps": [
                                {"order": 1, "action": "LookDown", "objectType": None, "targetType": None},
                                {"order": 2, "action": "MoveAhead", "objectType": None, "targetType": None},
                            ],
                            "confidence": "high",
                            "reason": "bad exploratory search plan",
                        },
                        {
                            "normalized_task": "go to the Fridge.",
                            "intentSteps": [
                                {"order": 1, "action": "GotoObject", "objectType": "Fridge", "targetType": None},
                            ],
                            "confidence": "high",
                            "reason": "find target is executable as navigation to Fridge",
                        },
                    ),
                    service_config(temp_dir),
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task({"task_id": "find-fridge", "task": "find fridge"})
        finally:
            relay_task_server.fetch_receiver_state_object_types = old_fetch

        intent_index = engine.argv.index("--task-intent-json") + 1
        task_intent = json.loads(engine.argv[intent_index])["task_intent"]
        task_index = engine.argv.index("--task") + 1
        self.assertEqual(engine.argv[task_index], "go to the Fridge.")
        self.assertEqual(response["task_normalization"]["source"], "qwen_tool_call_retry")
        self.assertEqual(task_intent["requestedAction"], "GotoObject")
        self.assertEqual(task_intent["requestedObjectType"], "Fridge")
        self.assertEqual(task_intent["intentSteps"], [
            {"order": 1, "action": "GotoObject", "objectType": "Fridge", "targetType": None}
        ])

    def test_normalizer_tool_retry_extracts_multi_step_open_close(self):
        old_fetch = relay_task_server.fetch_receiver_state_object_types
        relay_task_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Fridge"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                service = RelayTaskService(
                    engine,
                    SequentialToolBackend(
                        {},
                        {
                            "normalized_task": "open the Fridge and close the Fridge.",
                            "intentSteps": [
                                {"order": 1, "action": "OpenObject", "objectType": "Fridge", "targetType": None},
                                {"order": 2, "action": "CloseObject", "objectType": "Fridge", "targetType": None},
                            ],
                            "confidence": "high",
                            "reason": "the task requests opening then closing the same object",
                        },
                    ),
                    service_config(temp_dir),
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task({"task_id": "open-close", "task": "open the fridge and close the fridge"})
        finally:
            relay_task_server.fetch_receiver_state_object_types = old_fetch

        intent_index = engine.argv.index("--task-intent-json") + 1
        task_intent = json.loads(engine.argv[intent_index])["task_intent"]
        self.assertEqual(response["task_normalization"]["source"], "qwen_tool_call_retry")
        self.assertEqual(task_intent["requestedAction"], "OpenObject")
        self.assertEqual(task_intent["intentSteps"][1]["action"], "CloseObject")

    def test_normalizer_tool_retry_extracts_multi_step_pickup_put(self):
        old_fetch = relay_task_server.fetch_receiver_state_object_types
        relay_task_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Tomato", "CounterTop"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                service = RelayTaskService(
                    engine,
                    SequentialToolBackend(
                        {},
                        {
                            "normalized_task": "pick up the Tomato and put it on the CounterTop.",
                            "intentSteps": [
                                {"order": 1, "action": "PickupObject", "objectType": "Tomato", "targetType": None},
                                {"order": 2, "action": "PutObject", "objectType": "Tomato", "targetType": "CounterTop"},
                            ],
                            "confidence": "high",
                            "reason": "the task requests pickup followed by placement",
                        },
                    ),
                    service_config(temp_dir),
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task({"task_id": "pickup-put", "task": "pick up the tomato and put it on the counter"})
        finally:
            relay_task_server.fetch_receiver_state_object_types = old_fetch

        intent_index = engine.argv.index("--task-intent-json") + 1
        task_intent = json.loads(engine.argv[intent_index])["task_intent"]
        self.assertEqual(response["task_normalization"]["source"], "qwen_tool_call_retry")
        self.assertEqual(task_intent["requestedTargetType"], "CounterTop")
        self.assertEqual(task_intent["intentSteps"][0]["action"], "PickupObject")
        self.assertEqual(task_intent["intentSteps"][1]["action"], "PutObject")

    def test_normalizer_canonicalizes_action_only_task_text(self):
        old_fetch = relay_task_server.fetch_receiver_state_object_types
        relay_task_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Fridge"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                backend = FakeNormalizerBackend(
                    {
                        "normalized_task": "GotoObject",
                        "action": "GotoObject",
                        "object_type": "Fridge",
                        "target_type": "Fridge",
                        "confidence": "high",
                        "reason": "planning find request can be executed as navigation",
                    }
                )
                service = RelayTaskService(engine, backend, service_config(temp_dir))
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task({"task_id": "action-only", "task": "find fridge"})
        finally:
            relay_task_server.fetch_receiver_state_object_types = old_fetch

        task_index = engine.argv.index("--task") + 1
        self.assertEqual(engine.argv[task_index], "go to the Fridge.")
        self.assertTrue(response["task_normalization"]["used"])
        self.assertEqual(response["task_normalization"]["normalized_task"], "go to the Fridge.")

    def test_normalizer_accepts_camel_case_and_numeric_confidence(self):
        old_fetch = relay_task_server.fetch_receiver_state_object_types
        relay_task_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Fridge"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                backend = FakeNormalizerBackend(
                    {
                        "normalizedTask": "go to the Fridge.",
                        "action": "GotoObject",
                        "objectType": "Fridge",
                        "targetType": None,
                        "confidence": 0.95,
                        "reason": "planning find request can be executed as navigation",
                    }
                )
                service = RelayTaskService(engine, backend, service_config(temp_dir))
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task({"task_id": "numeric-confidence", "task": "find fridge"})
        finally:
            relay_task_server.fetch_receiver_state_object_types = old_fetch

        task_index = engine.argv.index("--task") + 1
        self.assertEqual(engine.argv[task_index], "go to the Fridge.")
        self.assertTrue(response["task_normalization"]["used"])
        self.assertEqual(response["task_normalization"]["confidence"], "high")
        self.assertEqual(response["task_normalization"]["object_type"], "Fridge")

    def test_normalizer_preserves_no_arg_look_down_step(self):
        old_fetch = relay_task_server.fetch_receiver_state_object_types
        relay_task_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Fridge"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                backend = FakeNormalizerBackend(
                    {
                        "normalized_task": "open the Fridge and look down.",
                        "intentSteps": [
                            {"order": 1, "action": "OpenObject", "objectType": "Fridge", "targetType": None},
                            {"order": 2, "action": "LookDown", "objectType": None, "targetType": None},
                        ],
                        "confidence": "high",
                        "reason": "the task requests opening the fridge and looking down",
                    }
                )
                service = RelayTaskService(engine, backend, service_config(temp_dir))
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task({"task_id": "open-look-down", "task": "open the fridge and look down"})
        finally:
            relay_task_server.fetch_receiver_state_object_types = old_fetch

        intent_index = engine.argv.index("--task-intent-json") + 1
        task_intent = json.loads(engine.argv[intent_index])["task_intent"]
        self.assertTrue(response["task_normalization"]["used"])
        self.assertEqual(task_intent["intentSteps"][0]["action"], "OpenObject")
        self.assertEqual(task_intent["intentSteps"][1]["action"], "LookDown")
        self.assertIsNone(task_intent["intentSteps"][1]["objectType"])

    def test_normalizer_preserves_single_look_down_task(self):
        old_fetch = relay_task_server.fetch_receiver_state_object_types
        relay_task_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Fridge"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                backend = FakeNormalizerBackend(
                    {
                        "normalized_task": "look down.",
                        "intentSteps": [
                            {"order": 1, "action": "LookDown", "objectType": None, "targetType": None},
                        ],
                        "confidence": "high",
                        "reason": "the task requests looking down",
                    }
                )
                service = RelayTaskService(engine, backend, service_config(temp_dir))
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task({"task_id": "look-down", "task": "look down"})
        finally:
            relay_task_server.fetch_receiver_state_object_types = old_fetch

        intent_index = engine.argv.index("--task-intent-json") + 1
        task_intent = json.loads(engine.argv[intent_index])["task_intent"]
        self.assertTrue(response["task_normalization"]["used"])
        self.assertEqual(task_intent["requestedAction"], "LookDown")
        self.assertEqual(task_intent["intentSteps"][0]["action"], "LookDown")

    def test_rejects_normalizer_that_adds_unrequested_goto_to_pickup_put(self):
        old_fetch = relay_task_server.fetch_receiver_state_object_types
        relay_task_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Tomato", "CounterTop"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                backend = FakeNormalizerBackend(
                    {
                        "normalized_task": "pick up the Tomato and put it on the CounterTop.",
                        "intentSteps": [
                            {"order": 1, "action": "GotoObject", "objectType": "Tomato", "targetType": None},
                            {"order": 2, "action": "PickupObject", "objectType": "Tomato", "targetType": None},
                            {"order": 3, "action": "PutObject", "objectType": "Tomato", "targetType": "CounterTop"},
                        ],
                        "confidence": "high",
                        "reason": "badly inserted navigation",
                    }
                )
                service = RelayTaskService(engine, backend, service_config(temp_dir))
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task(
                        {"task_id": "bad-extra-goto", "task": "pick up the tomato and put it on the counter"}
                    )
        finally:
            relay_task_server.fetch_receiver_state_object_types = old_fetch

        self.assertIsNone(engine.argv)
        self.assertEqual(response["failure_code"], "task_normalization_failed")
        self.assertIn("added unrequested action", " ".join(response["task_normalization"]["warnings"]))

    def test_rejects_normalizer_that_changes_recognized_look_down_action(self):
        old_fetch = relay_task_server.fetch_receiver_state_object_types
        relay_task_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Fridge"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                backend = FakeNormalizerBackend(
                    {
                        "normalized_task": "open the Fridge and close the Fridge.",
                        "intentSteps": [
                            {"order": 1, "action": "OpenObject", "objectType": "Fridge", "targetType": None},
                            {"order": 2, "action": "CloseObject", "objectType": "Fridge", "targetType": None},
                        ],
                        "confidence": "high",
                        "reason": "bad rewrite",
                    }
                )
                service = RelayTaskService(engine, backend, service_config(temp_dir))
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task({"task_id": "bad-look-down", "task": "open the fridge and look down"})
        finally:
            relay_task_server.fetch_receiver_state_object_types = old_fetch

        self.assertIsNone(engine.argv)
        self.assertEqual(response["status"], "needs_upstream_planning")
        self.assertEqual(response["failure_code"], "task_normalization_failed")
        self.assertFalse(response["task_normalization"]["used"])
        self.assertIn("changed recognized action LookDown to CloseObject", " ".join(response["task_normalization"]["warnings"]))

    def test_rejects_normalized_task_with_unknown_object_type(self):
        old_fetch = relay_task_server.fetch_receiver_state_object_types
        relay_task_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Fridge"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                backend = FakeNormalizerBackend(
                    {
                        "normalized_task": "go to the Moon.",
                        "action": "GotoObject",
                        "object_type": "Moon",
                        "target_type": None,
                        "confidence": "high",
                        "reason": "bad object",
                    }
                )
                service = RelayTaskService(engine, backend, service_config(temp_dir))
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task({"task_id": "bad-target", "task": "find the moon"})
        finally:
            relay_task_server.fetch_receiver_state_object_types = old_fetch

        self.assertIsNone(engine.argv)
        self.assertEqual(response["status"], "needs_upstream_planning")
        self.assertEqual(response["failure_code"], "task_normalization_failed")
        self.assertFalse(response["task_normalization"]["used"])
        self.assertIn("requestedObjectType", " ".join(response["task_normalization"]["warnings"]))

    def test_model_shard_error_reports_missing_safetensors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"a": "model.safetensors-00001-of-00002.safetensors"}}),
                encoding="utf-8",
            )
            error = model_shard_error(str(model_dir))
        self.assertIsNotNone(error)
        self.assertIn("missing safetensors shard", error)
        self.assertIn("/225010231/mwl/Linhao/models/Qwen3.5-4B", error)

    def test_rejects_invalid_robot_list(self):
        old_fetch = relay_task_server.fetch_receiver_state_object_types
        relay_task_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Bread"], None)
        try:
            service = RelayTaskService(FakeEngine(), FakeNormalizerBackend({
                    "normalized_task": "pick up the Bread.",
                    "intentSteps": [
                        {"order": 1, "action": "PickupObject", "objectType": "Bread", "targetType": None},
                    ],
                    "confidence": "high",
                    "reason": "the task asks to pick up Bread",
                }), service_config())
            with self.assertRaisesRegex(ValueError, "known_robot_ids"):
                service.execute_task({"task": "pick up bread", "known_robot_ids": ["Robot0"]})
        finally:
            relay_task_server.fetch_receiver_state_object_types = old_fetch

    def test_surfaces_engine_error_when_no_json_is_produced(self):
        class FailingEngine(FakeEngine):
            def run(self, args):
                print("error: receiver unavailable", file=__import__("sys").stderr)
                return 1

        old_fetch = relay_task_server.fetch_receiver_state_object_types
        relay_task_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Bread"], None)
        try:
            service = RelayTaskService(FailingEngine(), FakeNormalizerBackend({
                    "normalized_task": "pick up the Bread.",
                    "intentSteps": [
                        {"order": 1, "action": "PickupObject", "objectType": "Bread", "targetType": None},
                    ],
                    "confidence": "high",
                    "reason": "the task asks to pick up Bread",
                }), service_config())
            with self.assertRaisesRegex(RuntimeError, "receiver unavailable"):
                service.execute_task({"task": "pick up bread"})
        finally:
            relay_task_server.fetch_receiver_state_object_types = old_fetch
