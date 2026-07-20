from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest

from demo import auto_scene_actions as module


class FakeRelayBackend:
    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.calls: list[dict] = []

    def generate_messages(self, messages, *, tools=None, deterministic=False):
        if not deterministic:
            raise AssertionError("relay integration must use deterministic JSON generation")
        if not self.outputs:
            raise AssertionError("unexpected relay-agent turn")
        self.calls.append({"messages": messages, "tools": tools})
        return self.outputs.pop(0)


def call(name: str, **arguments) -> str:
    return json.dumps({"name": name, "arguments": arguments})


def fake_intent(args, available_types):
    tool_call = {"name": "extract_task_intent", "arguments": {"task": args.task}}
    intent = {
        "requestedAction": "PickupObject",
        "requestedObjectType": "Apple",
        "intentSteps": [
            {"order": 1, "action": "PickupObject", "objectType": "Apple", "targetType": None}
        ],
    }
    return tool_call, intent, {"status": "ok", "warnings": []}


APPLE_PLAN = {
    "task": "Pick up the apple.",
    "targetObjectType": "Apple",
    "needsGrounding": True,
    "observations": [],
    "plan": [{"action": "PickupObject", "objectType": "Apple", "targetType": None}],
}


class RelayAgentIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.originals = {
            "probe_scene": module.probe_scene,
            "execute_actions_probe_scene": module.execute_actions_probe_scene,
            "generate_semantic_plan": module.generate_semantic_plan,
            "generate_task_intent_tool_call": module.generate_task_intent_tool_call,
            "send_actions": module.send_actions,
        }

    def tearDown(self) -> None:
        for name, value in self.originals.items():
            setattr(module, name, value)

    def base_probe(self):
        return {
            "selected_robot_id": 0,
            "robots": [{"robot_id": 0}, {"robot_id": 2}],
            "objects": [{"id": "Egg|1", "type": "Egg", "visible": True, "pickupable": True}],
            "image_base64": "eA==",
        }

    def test_default_agent_strategy_uses_primary_fast_path_when_primary_can_execute(self) -> None:
        primary_probe = {
            "selected_robot_id": 0,
            "robots": [{"robot_id": 0}, {"robot_id": 2}],
            "objects": [{"id": "Apple|1", "type": "Apple", "visible": True, "pickupable": True}],
            "image_base64": "eA==",
        }
        sent: list[dict] = []
        module.probe_scene = lambda *args, **kwargs: primary_probe
        module.execute_actions_probe_scene = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("primary fast path should not probe peer robots")
        )
        module.generate_semantic_plan = lambda *args, **kwargs: ("{}", APPLE_PLAN, None)
        module.generate_task_intent_tool_call = fake_intent
        module.send_actions = lambda url, payload, timeout: sent.append(payload) or json.dumps(
            {"status": "success", "state": {"objects": []}}
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            args = module.parse_args(
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
                ]
            )
            args._qwen_backend = FakeRelayBackend([])
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exit_code = module.run(args)

        output = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(output["relay_result"]["strategy"], "primary_fast_path")
        self.assertEqual(output["relay_result"]["status"], "executor_ready")
        self.assertEqual(output["executor_robot_id"], 0)
        self.assertEqual(output["queried_robot_ids"], [0])
        self.assertEqual(sent[0]["robot_id"], 0)
        self.assertEqual(sent[0]["actions"][0]["objectId"], "Apple|1")
        self.assertNotIn("[relay] primary cannot execute", stderr.getvalue())

    def test_default_agent_strategy_observes_and_routes_to_robot_two(self) -> None:
        peer_probe = {
            "status": "success",
            "results": [{"robot_id": 2, "robot_name": "Robot2", "image_base64": "eA=="}],
            "state": {
                "selected_robot_id": 2,
                "objects": [{"id": "Apple|2", "type": "Apple", "visible": True, "pickupable": True}],
            },
        }
        observed: list[int] = []
        sent: list[dict] = []
        module.probe_scene = lambda *args, **kwargs: self.base_probe()
        module.execute_actions_probe_scene = (
            lambda url, task_id, timeout, robot_id=0: observed.append(robot_id) or peer_probe
        )
        module.generate_semantic_plan = lambda *args, **kwargs: ("{}", APPLE_PLAN, None)
        module.generate_task_intent_tool_call = fake_intent
        module.send_actions = lambda url, payload, timeout: sent.append(payload) or json.dumps(
            {"status": "success", "state": {"objects": []}}
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            args = module.parse_args(
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
                ]
            )
            backend = FakeRelayBackend(
                [call("select_executor", robot_id=2, reason="robot 2 sees a pickupable apple")]
            )
            args._qwen_backend = backend
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exit_code = module.run(args)

        output = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(observed, [2])
        self.assertEqual(output["relay_result"]["strategy"], "agent")
        self.assertEqual(output["relay_result"]["status"], "executor_ready")
        self.assertEqual(output["executor_robot_id"], 2)
        self.assertEqual(sent[0]["robot_id"], 2)
        self.assertEqual(sent[0]["actions"][0]["objectId"], "Apple|2")
        relay_input = json.loads(backend.calls[0]["messages"][1]["content"][0]["text"])
        self.assertEqual(relay_input["evidence_collection_status"], "all_known_agents_collected")
        self.assertEqual(relay_input["observed_robot_ids"], [0, 2])
        self.assertEqual(relay_input["candidate_evaluation"]["candidate_executor_robot_ids"], [2])
        self.assertEqual(
            [tool["function"]["name"] for tool in backend.calls[0]["tools"]],
            ["select_executor", "report_failure"],
        )
        self.assertEqual(
            len([item for item in backend.calls[0]["messages"][1]["content"] if item.get("type") == "image"]),
            2,
        )
        relay_result = output["relay_result"]
        self.assertIn("robot 0 cannot execute PickupObject Apple", relay_result["primary_inability_reason"])
        self.assertIn("coordination succeeded: robot_2 selected", relay_result["coordination_explanation"])
        self.assertIn("robot_2 executable", relay_result["candidate_explanation_summary"])
        stderr_text = stderr.getvalue()
        self.assertIn("[relay] primary cannot execute", stderr_text)
        self.assertIn("[relay] coordination succeeded: robot_2 selected", stderr_text)
        self.assertIn("[relay] candidates: robot_2 executable", stderr_text)

    def test_agent_returns_reason_when_no_robot_can_see_target(self) -> None:
        peer_probe = {
            "robot_id": 2,
            "objects": [{"id": "Mug|2", "type": "Mug", "visible": True, "pickupable": True}],
            "image_base64": "eA==",
        }
        module.probe_scene = lambda *args, **kwargs: self.base_probe()
        module.execute_actions_probe_scene = lambda *args, **kwargs: peer_probe
        module.generate_semantic_plan = lambda *args, **kwargs: ("{}", APPLE_PLAN, None)
        module.generate_task_intent_tool_call = fake_intent
        module.send_actions = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("failure must not send actions")
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            args = module.parse_args(
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
                ]
            )
            args._qwen_backend = FakeRelayBackend(
                [
                    call(
                        "report_failure",
                        failure_code="target_not_visible",
                        reason="neither robot can see an apple",
                    )
                ]
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exit_code = module.run(args)

        output = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(output["relay_result"]["status"], "needs_upstream_planning")
        self.assertEqual(output["relay_result"]["failure_code"], "target_not_visible")
        self.assertIn("not visible", output["relay_result"]["reason"])
        self.assertIn("coordination failed: target_not_visible", output["relay_result"]["coordination_explanation"])
        self.assertIn("robot_2 rejected", output["relay_result"]["candidate_explanation_summary"])
        stderr_text = stderr.getvalue()
        self.assertIn("[relay] primary cannot execute", stderr_text)
        self.assertIn("[relay] coordination failed: target_not_visible", stderr_text)
        self.assertIn("[relay] candidates:", stderr_text)
        self.assertNotIn("payload", output)

    def test_precollection_error_is_given_to_agent_without_retry_polling(self) -> None:
        observed: list[int] = []
        module.probe_scene = lambda *args, **kwargs: self.base_probe()

        def fail_peer_observation(url, task_id, timeout, robot_id=0):
            observed.append(robot_id)
            raise RuntimeError("robot camera unavailable")

        module.execute_actions_probe_scene = fail_peer_observation
        module.generate_semantic_plan = lambda *args, **kwargs: ("{}", APPLE_PLAN, None)
        module.generate_task_intent_tool_call = fake_intent
        module.send_actions = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("observation failure must not send actions")
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            args = module.parse_args(
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
                ]
            )
            backend = FakeRelayBackend(
                [
                    call(
                        "report_failure",
                        failure_code="observation_failed",
                        reason="robot 2 observation failed",
                    )
                ]
            )
            args._qwen_backend = backend
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exit_code = module.run(args)

        output = json.loads(stdout.getvalue())
        relay_result = output["relay_result"]
        relay_input = json.loads(backend.calls[0]["messages"][1]["content"][0]["text"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(observed, [2])
        self.assertEqual(relay_result["failure_code"], "observation_failed")
        self.assertEqual(relay_result["observation_errors"], {"2": "robot camera unavailable"})
        self.assertEqual(
            relay_input["evidence_collection_status"],
            "all_known_agents_collection_attempted_with_errors",
        )
        self.assertEqual(relay_input["visibility_unknown_robot_ids"], [2])
        self.assertEqual(relay_input["last_observation_errors"], {"2": "robot camera unavailable"})
        self.assertEqual(
            [tool["function"]["name"] for tool in backend.calls[0]["tools"]],
            ["select_executor", "report_failure"],
        )


if __name__ == "__main__":
    unittest.main()
