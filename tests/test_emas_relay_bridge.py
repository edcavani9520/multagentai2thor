from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch

from emas_relay_bridge import RelayTaskClient, assignment_list, dispatch_assignments


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class EmasRelayBridgeTest(unittest.TestCase):
    def test_reads_first_allocation_unit(self):
        assignments = assignment_list(
            {"allocation": [{"time_step": 1, "unit": [{"agent_id": "1", "subtask": {"id": "T1"}}]}]}
        )
        self.assertEqual(assignments[0]["agent_id"], "1")

    @patch("emas_relay_bridge.urlopen")
    def test_dispatches_each_assignment_and_returns_emas_statuses(self, mocked_urlopen):
        mocked_urlopen.side_effect = [
            FakeResponse({"status": "success", "result": {"closed_loop_result": {"status": "success"}}}),
            FakeResponse({"status": "needs_upstream_planning", "error": "target not visible"}),
        ]
        report = dispatch_assignments(
            [
                {"agent_id": "0", "subtask": {"id": "T1", "description": "pick up the bread"}},
                {"agent_id": "1", "subtask": {"id": "T2", "description": "put the bread on the counter"}},
            ],
            root_task="put bread on counter",
            client=RelayTaskClient("http://127.0.0.1:18080"),
            known_robot_ids=[0, 1],
            dry_run=True,
            relay_strategy="rules",
        )

        self.assertEqual(report["execution"]["completed_task_ids"], ["T1"])
        self.assertEqual([item["status"] for item in report["task_statuses"]], ["success", "wait_retry"])
        self.assertEqual(mocked_urlopen.call_count, 2)
        first_request = mocked_urlopen.call_args_list[0].args[0]
        first_payload = json.loads(first_request.data.decode("utf-8"))
        self.assertEqual(first_payload["primary_robot_id"], 0)
        self.assertEqual(first_payload["known_robot_ids"], [0, 1])
        self.assertIn("pick up the bread", first_payload["task"])


if __name__ == "__main__":
    unittest.main()
