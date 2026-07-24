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


class EmptyToolThenJsonBackend:
    def __init__(self, fallback_arguments=None):
        self.fallback_arguments = fallback_arguments or {
            "normalized_task": "go to the Fridge.",
            "intentSteps": [
                {"order": 1, "action": "GotoObject", "objectType": "Fridge", "targetType": None},
            ],
            "confidence": "high",
            "reason": "fallback normalized search to navigation",
        }

    def generate_with_tools(self, messages, tools):
        return json.dumps({"name": "normalize_incoming_task", "arguments": {}})

    def generate_messages(self, messages, deterministic=False):
        return json.dumps(self.fallback_arguments)


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
    def test_builds_closed_loop_relay_request(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = FakeEngine()
            service = RelayTaskService(
                engine,
                object(),
                RelayRuntimeConfig(
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
                ),
            )
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

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["result"]["closed_loop_result"]["step_count"], 2)
        self.assertIn("--relay-mode", engine.argv)
        self.assertIn("--closed-loop-replan", engine.argv)
        self.assertIn("--save-raw-output", engine.argv)
        self.assertIn("http://127.0.0.1:19000/execute_actions", engine.argv)
        self.assertIn("0,1", engine.argv)
        self.assertIn("--dry-run", engine.argv)


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

    def test_normalizer_uses_json_fallback_when_tool_arguments_are_empty(self):
        old_fetch = relay_task_server.fetch_receiver_state_object_types
        relay_task_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Fridge"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                service = RelayTaskService(engine, EmptyToolThenJsonBackend(), service_config(temp_dir))
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task({"task_id": "fallback", "task": "find fridge"})
        finally:
            relay_task_server.fetch_receiver_state_object_types = old_fetch

        task_index = engine.argv.index("--task") + 1
        self.assertEqual(engine.argv[task_index], "go to the Fridge.")
        self.assertTrue(response["task_normalization"]["used"])
        self.assertEqual(response["task_normalization"]["source"], "qwen_json_fallback")
        self.assertIn("raw_tool_output_preview", response["task_normalization"])
        self.assertIn("raw_json_fallback_preview", response["task_normalization"])
        intent_index = engine.argv.index("--task-intent-json") + 1
        task_intent = json.loads(engine.argv[intent_index])["task_intent"]
        self.assertEqual(task_intent["requestedAction"], "GotoObject")
        self.assertEqual(task_intent["requestedObjectType"], "Fridge")

    def test_normalizer_json_fallback_extracts_multi_step_open_close(self):
        old_fetch = relay_task_server.fetch_receiver_state_object_types
        relay_task_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Fridge"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                service = RelayTaskService(
                    engine,
                    EmptyToolThenJsonBackend(
                        {
                            "normalized_task": "open the Fridge and close the Fridge.",
                            "intentSteps": [
                                {"order": 1, "action": "OpenObject", "objectType": "Fridge", "targetType": None},
                                {"order": 2, "action": "CloseObject", "objectType": "Fridge", "targetType": None},
                            ],
                            "confidence": "high",
                            "reason": "the task requests opening then closing the same object",
                        }
                    ),
                    service_config(temp_dir),
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task({"task_id": "open-close", "task": "open the fridge and close the fridge"})
        finally:
            relay_task_server.fetch_receiver_state_object_types = old_fetch

        intent_index = engine.argv.index("--task-intent-json") + 1
        task_intent = json.loads(engine.argv[intent_index])["task_intent"]
        self.assertEqual(response["task_normalization"]["source"], "qwen_json_fallback")
        self.assertEqual(task_intent["requestedAction"], "OpenObject")
        self.assertEqual(task_intent["intentSteps"][1]["action"], "CloseObject")

    def test_normalizer_json_fallback_extracts_multi_step_pickup_put(self):
        old_fetch = relay_task_server.fetch_receiver_state_object_types
        relay_task_server.fetch_receiver_state_object_types = lambda receiver_url, timeout: (["Tomato", "CounterTop"], None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = FakeEngine()
                service = RelayTaskService(
                    engine,
                    EmptyToolThenJsonBackend(
                        {
                            "normalized_task": "pick up the Tomato and put it on the CounterTop.",
                            "intentSteps": [
                                {"order": 1, "action": "PickupObject", "objectType": "Tomato", "targetType": None},
                                {"order": 2, "action": "PutObject", "objectType": "Tomato", "targetType": "CounterTop"},
                            ],
                            "confidence": "high",
                            "reason": "the task requests pickup followed by placement",
                        }
                    ),
                    service_config(temp_dir),
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    response = service.execute_task({"task_id": "pickup-put", "task": "pick up the tomato and put it on the counter"})
        finally:
            relay_task_server.fetch_receiver_state_object_types = old_fetch

        intent_index = engine.argv.index("--task-intent-json") + 1
        task_intent = json.loads(engine.argv[intent_index])["task_intent"]
        self.assertEqual(response["task_normalization"]["source"], "qwen_json_fallback")
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

        task_index = engine.argv.index("--task") + 1
        self.assertEqual(engine.argv[task_index], "find the moon")
        self.assertNotIn("--task-intent-json", engine.argv)
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
        service = RelayTaskService(
            FakeEngine(),
            object(),
            RelayRuntimeConfig(
                receiver_url="http://127.0.0.1:19000",
                model_path="model",
                device="cuda",
                device_map="auto",
                dtype="float16",
                max_new_tokens=64,
                temperature=0.1,
                send_timeout=60.0,
                output_dir=Path("/tmp"),
                max_replan_steps=10,
                relay_agent_max_turns=8,
                max_actions=8,
            ),
        )
        with self.assertRaisesRegex(ValueError, "known_robot_ids"):
            service.execute_task({"task": "pick up bread", "known_robot_ids": ["Robot0"]})

    def test_surfaces_engine_error_when_no_json_is_produced(self):
        class FailingEngine(FakeEngine):
            def run(self, args):
                print("error: receiver unavailable", file=__import__("sys").stderr)
                return 1

        service = RelayTaskService(
            FailingEngine(),
            object(),
            RelayRuntimeConfig(
                receiver_url="http://127.0.0.1:19000",
                model_path="model",
                device="cuda",
                device_map="auto",
                dtype="float16",
                max_new_tokens=64,
                temperature=0.1,
                send_timeout=60.0,
                output_dir=Path("/tmp"),
                max_replan_steps=10,
                relay_agent_max_turns=8,
                max_actions=8,
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "receiver unavailable"):
            service.execute_task({"task": "pick up bread"})
