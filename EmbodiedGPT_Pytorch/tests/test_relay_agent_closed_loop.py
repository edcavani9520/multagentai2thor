from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest

from demo import auto_scene_actions as module


class FakeJsonRelayBackend:
    def __init__(self, outputs: list[dict]):
        self.outputs = [json.dumps(output) for output in outputs]
        self.calls: list[dict] = []

    def generate_messages(self, messages, *, tools=None, deterministic=False):
        if not deterministic:
            raise AssertionError("closed-loop relay must use deterministic JSON generation")
        if not self.outputs:
            raise AssertionError("unexpected relay-agent turn")
        self.calls.append({"messages": messages, "tools": tools})
        return self.outputs.pop(0)


class RelayAgentClosedLoopTest(unittest.TestCase):
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

    def test_pickup_and_put_remain_on_selected_peer(self) -> None:
        counter = {
            "id": "CounterTop|1",
            "type": "CounterTop",
            "visible": True,
            "receptacle": True,
        }
        tomato = {
            "id": "Tomato|2",
            "type": "Tomato",
            "visible": True,
            "pickupable": True,
        }
        primary_probe = {
            "selected_robot_id": 0,
            "robots": [{"robot_id": 0}, {"robot_id": 2}],
            "objects": [counter],
            "image_base64": "eA==",
        }
        peer_probe = {
            "robot_id": 2,
            "objects": [tomato, counter],
            "image_base64": "eA==",
        }
        task_intent = {
            "requestedAction": "PutObject",
            "requestedObjectType": "Tomato",
            "intentSteps": [
                {"order": 1, "action": "PickupObject", "objectType": "Tomato", "targetType": None},
                {
                    "order": 2,
                    "action": "PutObject",
                    "objectType": "Tomato",
                    "targetType": "CounterTop",
                },
            ],
        }
        tool_call = {"name": "extract_task_intent", "arguments": {"task": "put the tomato on the counter."}}
        sent: list[dict] = []
        held_state: dict[str, dict | None] = {"object": None}

        module.probe_scene = lambda *args, **kwargs: primary_probe
        module.execute_actions_probe_scene = lambda *args, **kwargs: {
            **peer_probe,
            "robot": {"held_object": held_state["object"]},
        }
        module.generate_task_intent_tool_call = lambda *args, **kwargs: (
            tool_call,
            task_intent,
            {"status": "ok", "warnings": []},
        )

        def generate_step_plan(args, image_path, objects, task_id):
            intent = args._task_intent
            step = intent["intentSteps"][0]
            return (
                "{}",
                {
                    "task": args.task,
                    "targetObjectType": step["objectType"],
                    "needsGrounding": True,
                    "observations": [],
                    "plan": [step],
                },
                None,
            )

        module.generate_semantic_plan = generate_step_plan

        def send_actions(url, payload, timeout):
            sent.append(payload)
            action = payload["actions"][0]["action"]
            held_object = (
                {"objectId": "Tomato|2", "objectType": "Tomato"}
                if action == "PickupObject"
                else None
            )
            held_state["object"] = held_object
            return json.dumps(
                {
                    "status": "success",
                    "robot_id": 2,
                    "robot": {"held_object": held_object},
                    "objects": [tomato, counter],
                    "image_base64": "eA==",
                    "results": [
                        {
                            "robot_id": 2,
                            "success": True,
                            "robot": {"held_object": held_object},
                        }
                    ],
                }
            )

        module.send_actions = send_actions

        relay_outputs = [
            {
                "name": "select_executor",
                "arguments": {"robot_id": 2, "reason": "robot 2 sees a pickupable tomato"},
            },
            {
                "name": "select_executor",
                "arguments": {"robot_id": 2, "reason": "robot 2 holds the tomato and sees the counter"},
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            args = module.parse_args(
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
                    "--closed-loop-replan",
                ]
            )
            backend = FakeJsonRelayBackend(relay_outputs)
            args._qwen_backend = backend
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exit_code = module.run(args)

        output = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(output["closed_loop_result"], {"status": "success", "step_count": 2}, json.dumps(output, indent=2))
        self.assertEqual([payload["robot_id"] for payload in sent], [2, 2, 2])
        self.assertEqual(
            [payload["actions"][0]["action"] for payload in sent],
            ["PickupObject", "PutObject", "Done"],
        )
        self.assertIn("relay_explanation", output["closed_loop_trace"][0])
        self.assertIn("robot 0 cannot execute PickupObject Tomato", output["closed_loop_trace"][0]["relay_explanation"]["primary_inability_reason"])
        self.assertIn("coordination succeeded: robot_2 selected", output["closed_loop_trace"][0]["relay_explanation"]["coordination_explanation"])
        self.assertIn("relay_explanation", output["closed_loop_trace"][1])
        self.assertIn("robot 0 cannot execute PutObject Tomato -> CounterTop", output["closed_loop_trace"][1]["relay_explanation"]["primary_inability_reason"])
        stderr_text = stderr.getvalue()
        self.assertIn("[relay] primary cannot execute", stderr_text)
        self.assertIn("[relay] coordination succeeded: robot_2 selected", stderr_text)
        self.assertIn("[relay] candidates:", stderr_text)
        self.assertTrue(
            all(
                [tool["function"]["name"] for tool in call["tools"]]
                == ["select_executor", "report_failure"]
                for call in backend.calls
            )
        )


if __name__ == "__main__":
    unittest.main()
