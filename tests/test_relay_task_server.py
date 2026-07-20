import contextlib
import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from relay_task_server import RelayRuntimeConfig, RelayTaskService


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
