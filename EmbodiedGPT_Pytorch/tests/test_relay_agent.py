from __future__ import annotations

import json
import unittest

from demo.relay_agent import RelayAgentConfig, parse_relay_tool_call, run_relay_agent


class FakeBackend:
    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.messages: list[list[dict]] = []

    def generate_messages(self, messages, *, deterministic=False):
        if not deterministic:
            raise AssertionError("relay agent must request deterministic JSON generation")
        self.messages.append(list(messages))
        if not self.outputs:
            raise AssertionError("relay agent requested an unexpected extra turn")
        return self.outputs.pop(0)


def json_call(name: str, **arguments) -> str:
    return json.dumps({"name": name, "arguments": arguments})


class RelayAgentTest(unittest.TestCase):
    def test_parses_json_tool_call(self) -> None:
        self.assertEqual(
            parse_relay_tool_call(json_call("select_executor", robot_id=2, reason="apple is visible")),
            {
                "name": "select_executor",
                "arguments": {"robot_id": 2, "reason": "apple is visible"},
            },
        )

    def test_parses_json_inside_tool_call_wrapper(self) -> None:
        wrapped = f"<tool_call>{json_call('observe_robot', robot_id=2)}</tool_call>"

        self.assertEqual(
            parse_relay_tool_call(wrapped),
            {"name": "observe_robot", "arguments": {"robot_id": 2}},
        )

    def test_parses_openai_style_json_tool_call(self) -> None:
        output = json.dumps(
            {
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "observe_robot",
                            "arguments": json.dumps({"robot_id": 2}),
                        },
                    }
                ]
            }
        )

        self.assertEqual(
            parse_relay_tool_call(output),
            {"name": "observe_robot", "arguments": {"robot_id": 2}},
        )

    def test_xml_placeholder_is_rejected_then_corrected_with_json(self) -> None:
        xml_placeholder = (
            "<tool_call><function=example_function_name>"
            "<parameter=example_parameter_1>value_1</parameter>"
            "</function></tool_call>"
        )
        backend = FakeBackend(
            [
                xml_placeholder,
                json_call("select_executor", robot_id=0, reason="primary is valid"),
            ]
        )

        result = run_relay_agent(
            backend,
            task="Turn right.",
            task_intent={"requestedAction": "RotateRight", "requestedObjectType": None},
            known_robot_ids=[0],
            primary_robot_id=0,
            initial_summaries=[{"robot_id": 0, "visible_objects": []}],
            observe_robot=lambda robot_id: {"robot_id": robot_id},
            validate_executor=lambda robot_id: (True, "primary can turn right"),
            validate_failure=lambda code, reason: (True, code, reason),
        )

        self.assertEqual(result["status"], "executor_selected")
        self.assertEqual(result["trace"][0]["status"], "protocol_error")
        self.assertIn("valid JSON tool call", result["trace"][0]["reason"])
        self.assertNotIn("tool", {message["role"] for message in backend.messages[1]})


    def test_observes_peer_then_selects_it(self) -> None:
        backend = FakeBackend(
            [
                json_call("observe_robot", robot_id=2),
                json_call("select_executor", robot_id=2, reason="robot 2 sees a pickupable apple"),
            ]
        )
        observed: list[int] = []

        result = run_relay_agent(
            backend,
            task="Pick up the apple.",
            task_intent={"requestedAction": "PickupObject", "requestedObjectType": "Apple"},
            known_robot_ids=[0, 2],
            primary_robot_id=0,
            initial_summaries=[{"robot_id": 0, "visible_objects": []}],
            observe_robot=lambda robot_id: observed.append(robot_id) or {
                "robot_id": robot_id,
                "visible_objects": [{"object_type": "Apple", "affordances": {"pickupable": True}}],
            },
            validate_executor=lambda robot_id: (
                robot_id == 2,
                "robot 2 has verified state and affordances for PickupObject Apple",
            ),
            validate_failure=lambda code, reason: (True, code, reason),
        )

        self.assertEqual(observed, [2])
        self.assertEqual(result["status"], "executor_selected")
        self.assertEqual(result["executor_robot_id"], 2)
        self.assertEqual([entry["tool"] for entry in result["trace"]], ["observe_robot", "select_executor"])

    def test_selects_already_observed_visible_peer_without_observe(self) -> None:
        backend = FakeBackend(
            [json_call("select_executor", robot_id=2, reason="robot 2 already sees a pickupable apple")]
        )

        result = run_relay_agent(
            backend,
            task="Pick up the apple.",
            task_intent={"requestedAction": "PickupObject", "requestedObjectType": "Apple"},
            known_robot_ids=[0, 2],
            primary_robot_id=0,
            initial_summaries=[
                {"robot_id": 0, "visible_objects": []},
                {
                    "robot_id": 2,
                    "robot_name": "Robot2",
                    "visible_objects": [
                        {"object_type": "Apple", "affordances": {"pickupable": True}}
                    ],
                },
            ],
            observe_robot=lambda robot_id: self.fail("known visible peer should not be re-observed"),
            validate_executor=lambda robot_id: (robot_id == 2, "robot 2 can pick up Apple"),
            validate_failure=lambda code, reason: (True, code, reason),
        )

        self.assertEqual(result["status"], "executor_selected")
        self.assertEqual(result["executor_robot_id"], 2)
        self.assertEqual([entry["tool"] for entry in result["trace"]], ["select_executor"])
        initial_state = json.loads(backend.messages[0][1]["content"][0]["text"])
        self.assertEqual(initial_state["visibility_unknown_robot_ids"], [])
        self.assertEqual(initial_state["known_visibility"][1]["robot_id"], 2)
        self.assertEqual(initial_state["known_visibility"][1]["visible_objects"][0]["object_type"], "Apple")

    def test_rejects_premature_failure_until_all_robots_are_observed(self) -> None:
        backend = FakeBackend(
            [
                json_call("report_failure", failure_code="target_not_visible", reason="not visible"),
                json_call("observe_robot", robot_id=2),
                json_call("report_failure", failure_code="target_not_visible", reason="not visible anywhere"),
            ]
        )
        result = run_relay_agent(
            backend,
            task="Pick up the apple.",
            task_intent={"requestedAction": "PickupObject", "requestedObjectType": "Apple"},
            known_robot_ids=[0, 2],
            primary_robot_id=0,
            initial_summaries=[{"robot_id": 0, "visible_objects": []}],
            observe_robot=lambda robot_id: {"robot_id": robot_id, "visible_objects": []},
            validate_executor=lambda robot_id: (False, "apple is not visible"),
            validate_failure=lambda code, reason: (
                True,
                "target_not_visible",
                "'Apple' is not visible to any successfully observed robot",
            ),
        )

        self.assertEqual(result["status"], "needs_upstream_planning")
        self.assertEqual(result["failure_code"], "target_not_visible")
        self.assertEqual(result["observed_robot_ids"], [0, 2])
        self.assertFalse(result["trace"][0]["response"]["ok"])

    def test_rejected_executor_can_be_corrected(self) -> None:
        backend = FakeBackend(
            [
                json_call("select_executor", robot_id=0, reason="try primary"),
                json_call("observe_robot", robot_id=2),
                json_call("select_executor", robot_id=2, reason="peer has target"),
            ]
        )
        result = run_relay_agent(
            backend,
            task="Pick up the apple.",
            task_intent={"requestedAction": "PickupObject", "requestedObjectType": "Apple"},
            known_robot_ids=[0, 2],
            primary_robot_id=0,
            initial_summaries=[{"robot_id": 0, "visible_objects": []}],
            observe_robot=lambda robot_id: {"robot_id": robot_id, "visible_objects": []},
            validate_executor=lambda robot_id: (robot_id == 2, "valid" if robot_id == 2 else "target not visible"),
            validate_failure=lambda code, reason: (True, code, reason),
            config=RelayAgentConfig(max_turns=4),
        )

        self.assertEqual(result["executor_robot_id"], 2)
        self.assertFalse(result["trace"][0]["response"]["ok"])

    def test_schema_arguments_are_rejected_with_known_robot_calls_and_visibility_state(self) -> None:
        schema_call = json.dumps(
            {
                "name": "observe_robot",
                "arguments": {
                    "type": "object",
                    "properties": {"robot_id": {"type": "integer"}},
                    "required": ["robot_id"],
                },
            }
        )
        backend = FakeBackend(
            [
                schema_call,
                json_call("observe_robot", robot_id=2),
                json_call("select_executor", robot_id=2, reason="robot 2 sees the target"),
            ]
        )
        observed: list[int] = []

        result = run_relay_agent(
            backend,
            task="Pick up the tomato.",
            task_intent={"requestedAction": "PickupObject", "requestedObjectType": "Tomato"},
            known_robot_ids=[0, 1, 2],
            primary_robot_id=1,
            initial_summaries=[
                {"robot_id": 1, "visible_objects": []},
                {"robot_id": 0, "visible_objects": []},
            ],
            observe_robot=lambda robot_id: observed.append(robot_id) or {
                "robot_id": robot_id,
                "visible_objects": [{"object_type": "Tomato"}],
            },
            validate_executor=lambda robot_id: (robot_id == 2, "robot 2 can pick up Tomato"),
            validate_failure=lambda code, reason: (True, code, reason),
        )

        correction = result["trace"][0]["response"]
        self.assertEqual(correction["error_code"], "arguments_are_schema")
        self.assertEqual(correction["observed_robot_ids"], [0, 1])
        self.assertEqual(correction["unobserved_robot_ids"], [2])
        self.assertEqual(
            correction["valid_calls"],
            [
                {"name": "observe_robot", "arguments": {"robot_id": 0}},
                {"name": "observe_robot", "arguments": {"robot_id": 1}},
                {"name": "observe_robot", "arguments": {"robot_id": 2}},
            ],
        )
        self.assertEqual(correction["visibility_unknown_robot_ids"], [2])
        self.assertEqual(correction["observation_attempt_counts"], {"0": 1, "1": 1, "2": 0})
        self.assertEqual(observed, [2])
        self.assertEqual(result["executor_robot_id"], 2)
        initial_state = json.loads(backend.messages[0][1]["content"][0]["text"])
        self.assertEqual(initial_state["observed_robot_ids"], [0, 1])
        self.assertEqual(initial_state["unobserved_robot_ids"], [2])
        self.assertEqual(initial_state["visibility_unknown_robot_ids"], [2])
        self.assertEqual([item["robot_id"] for item in initial_state["known_visibility"]], [1, 0])
        self.assertNotIn("Allowed tools and argument schemas", backend.messages[0][0]["content"])

    def test_repeated_failed_observe_is_not_rejected_before_max_turns(self) -> None:
        backend = FakeBackend([json_call("observe_robot", robot_id=2)] * 4)
        observed: list[int] = []

        result = run_relay_agent(
            backend,
            task="Pick up the apple.",
            task_intent={"requestedAction": "PickupObject", "requestedObjectType": "Apple"},
            known_robot_ids=[0, 1, 2],
            primary_robot_id=0,
            initial_summaries=[{"robot_id": 0, "visible_objects": []}],
            observe_robot=lambda robot_id: observed.append(robot_id) or (_ for _ in ()).throw(
                RuntimeError("requested robot 2, but observation identified robot 0")
            ),
            validate_executor=lambda robot_id: (False, "target not visible"),
            validate_failure=lambda code, reason: (True, code, reason),
            config=RelayAgentConfig(max_turns=4),
        )

        self.assertEqual(observed, [2, 2, 2, 2])
        self.assertEqual(result["status"], "needs_upstream_planning")
        self.assertEqual(result["failure_code"], "agent_max_turns_exceeded")
        self.assertEqual(result["observed_robot_ids"], [0])
        self.assertEqual(result["visibility_unknown_robot_ids"], [1, 2])
        self.assertEqual(result["observation_attempt_counts"], {"0": 1, "1": 0, "2": 4})
        self.assertEqual(result["last_observation_errors"]["2"], "requested robot 2, but observation identified robot 0")
        self.assertTrue(all(entry["response"]["error_code"] == "observation_failed" for entry in result["trace"]))

    def test_terminal_empty_candidates_return_validated_failure_after_turn_budget(self) -> None:
        backend = FakeBackend(
            [
                json_call("observe_robot", robot_id=1),
                json_call("inspect_global_scene"),
            ]
        )

        result = run_relay_agent(
            backend,
            task="Pick up the bread.",
            task_intent={"requestedAction": "PickupObject", "requestedObjectType": "Bread"},
            known_robot_ids=[0, 1],
            primary_robot_id=0,
            initial_summaries=[{"robot_id": 0, "visible_objects": []}],
            observe_robot=lambda robot_id: {"robot_id": robot_id, "visible_objects": []},
            inspect_global_scene=lambda: {"visible_object_types": []},
            evaluate_executor_candidates=lambda: {
                "candidate_executor_robot_ids": [],
                "candidate_scores": [
                    {"robot_id": 0, "executable": False},
                    {"robot_id": 1, "executable": False},
                ],
            },
            validate_executor=lambda robot_id: (False, "Bread is not visible"),
            validate_failure=lambda code, reason: (
                True,
                "target_not_visible",
                "'Bread' is not visible to any successfully observed robot",
            ),
            config=RelayAgentConfig(max_turns=2),
        )

        self.assertEqual(result["status"], "needs_upstream_planning")
        self.assertEqual(result["strategy"], "deterministic_terminal_fallback")
        self.assertEqual(result["failure_code"], "target_not_visible")
        self.assertEqual(result["observed_robot_ids"], [0, 1])
        self.assertEqual(len(result["trace"]), 1)

    def test_unchanged_repeated_observe_recommends_unknown_robot(self) -> None:
        backend = FakeBackend(
            [
                json_call("observe_robot", robot_id=2),
                json_call("observe_robot", robot_id=2),
                json_call("observe_robot", robot_id=0),
                json_call("select_executor", robot_id=0, reason="robot 0 sees a pickupable tomato"),
            ]
        )
        observed: list[int] = []

        def observe(robot_id: int) -> dict:
            observed.append(robot_id)
            if robot_id == 0:
                return {
                    "robot_id": 0,
                    "visible_objects": [{"object_type": "Tomato", "affordances": {"pickupable": True}}],
                }
            return {"robot_id": robot_id, "visible_objects": []}

        result = run_relay_agent(
            backend,
            task="Pick up the tomato.",
            task_intent={"requestedAction": "PickupObject", "requestedObjectType": "Tomato"},
            known_robot_ids=[0, 1, 2],
            primary_robot_id=1,
            initial_summaries=[{"robot_id": 1, "visible_objects": []}],
            observe_robot=observe,
            validate_executor=lambda robot_id: (robot_id == 0, "robot 0 can pick up Tomato"),
            validate_failure=lambda code, reason: (True, code, reason),
            config=RelayAgentConfig(max_turns=5),
        )

        repeated_response = result["trace"][1]["response"]
        self.assertEqual(observed, [2, 2, 0])
        self.assertEqual(repeated_response["error_code"], "no_new_evidence_for_repeated_observe")
        self.assertFalse(repeated_response["observation_changed"])
        self.assertEqual(
            repeated_response["recommended_observe_calls"],
            [{"name": "observe_robot", "arguments": {"robot_id": 0}}],
        )
        self.assertIn("Do not observe the same robot again", backend.messages[2][-1]["content"][0]["text"])
        self.assertEqual(result["status"], "executor_selected")
        self.assertEqual(result["executor_robot_id"], 0)

    def test_report_failure_allowed_after_all_known_robots_observed_without_target(self) -> None:
        backend = FakeBackend(
            [json_call("report_failure", failure_code="target_not_visible", reason="no robot sees tomato")]
        )

        result = run_relay_agent(
            backend,
            task="Pick up the tomato.",
            task_intent={"requestedAction": "PickupObject", "requestedObjectType": "Tomato"},
            known_robot_ids=[0, 1, 2],
            primary_robot_id=1,
            initial_summaries=[
                {"robot_id": 0, "visible_objects": []},
                {"robot_id": 1, "visible_objects": []},
                {"robot_id": 2, "visible_objects": []},
            ],
            observe_robot=lambda robot_id: self.fail("all robots were already observed"),
            validate_executor=lambda robot_id: (False, "target not visible"),
            validate_failure=lambda code, reason: (True, "target_not_visible", "Tomato is not visible anywhere"),
        )

        self.assertEqual(result["status"], "needs_upstream_planning")
        self.assertEqual(result["failure_code"], "target_not_visible")
        self.assertEqual(result["observed_robot_ids"], [0, 1, 2])
        self.assertEqual(result["visibility_unknown_robot_ids"], [])
        self.assertEqual([entry["tool"] for entry in result["trace"]], ["report_failure"])

    def test_reobserves_same_robot_after_validation_feedback(self) -> None:
        backend = FakeBackend(
            [
                json_call("select_executor", robot_id=2, reason="robot 2 had stale evidence"),
                json_call("observe_robot", robot_id=2),
                json_call("select_executor", robot_id=2, reason="fresh observation confirms apple"),
            ]
        )
        observed: list[int] = []
        fresh = {"value": False}

        def observe(robot_id: int) -> dict:
            observed.append(robot_id)
            fresh["value"] = True
            return {
                "robot_id": robot_id,
                "visible_objects": [{"object_type": "Apple", "affordances": {"pickupable": True}}],
            }

        result = run_relay_agent(
            backend,
            task="Pick up the apple.",
            task_intent={"requestedAction": "PickupObject", "requestedObjectType": "Apple"},
            known_robot_ids=[0, 2],
            primary_robot_id=0,
            initial_summaries=[
                {"robot_id": 0, "visible_objects": []},
                {"robot_id": 2, "visible_objects": [{"object_type": "Apple"}]},
            ],
            observe_robot=observe,
            validate_executor=lambda robot_id: (
                fresh["value"],
                "fresh observation is actionable" if fresh["value"] else "stale evidence needs refresh",
            ),
            validate_failure=lambda code, reason: (True, code, reason),
            config=RelayAgentConfig(max_turns=4),
        )

        self.assertEqual(observed, [2])
        self.assertEqual(result["status"], "executor_selected")
        self.assertEqual(result["executor_robot_id"], 2)
        self.assertFalse(result["trace"][0]["response"]["ok"])
        self.assertTrue(result["trace"][1]["response"]["ok"])
        self.assertEqual(result["trace"][1]["response"]["observation_attempt_counts"], {"0": 1, "2": 2})

    def test_invalid_tool_arguments_are_rejected_before_hard_validation(self) -> None:
        backend = FakeBackend(
            [
                json_call("observe_robot"),
                json_call("observe_robot", robot_id="2"),
                json_call("select_executor", robot_id=0, reason=" "),
                json_call("report_failure", failure_code="made_up", reason="cannot do it"),
                json_call("select_executor", robot_id=0, reason="primary can execute"),
            ]
        )

        result = run_relay_agent(
            backend,
            task="Turn right.",
            task_intent={"requestedAction": "RotateRight", "requestedObjectType": None},
            known_robot_ids=[0, 1, 2],
            primary_robot_id=1,
            initial_summaries=[
                {"robot_id": 0, "visible_objects": []},
                {"robot_id": 1, "visible_objects": []},
            ],
            observe_robot=lambda robot_id: self.fail("invalid observe arguments must not probe a robot"),
            validate_executor=lambda robot_id: (True, "robot can turn right"),
            validate_failure=lambda code, reason: self.fail("invalid failure must not reach hard validation"),
            config=RelayAgentConfig(max_turns=6),
        )

        self.assertEqual(
            [entry["response"]["error_code"] for entry in result["trace"][:4]],
            ["invalid_robot_id", "invalid_robot_id", "invalid_reason", "invalid_failure_code"],
        )
        self.assertEqual(result["executor_robot_id"], 0)

    def test_passes_native_tool_schemas_when_backend_supports_tools(self) -> None:
        class ToolAwareBackend:
            def __init__(self):
                self.tools = None
                self.messages = []

            def generate_messages(self, messages, *, tools=None, deterministic=False):
                self.tools = tools
                self.messages.append(list(messages))
                return json_call("select_executor", robot_id=0, reason="primary can execute")

        backend = ToolAwareBackend()

        result = run_relay_agent(
            backend,
            task="Turn right.",
            task_intent={"requestedAction": "RotateRight", "requestedObjectType": None},
            known_robot_ids=[0],
            primary_robot_id=0,
            initial_summaries=[{"robot_id": 0, "visible_objects": []}],
            observe_robot=lambda robot_id: self.fail("should not observe"),
            validate_executor=lambda robot_id: (True, "primary can execute"),
            validate_failure=lambda code, reason: (True, code, reason),
        )

        self.assertEqual(result["status"], "executor_selected")
        self.assertIsNotNone(backend.tools)
        self.assertIn("inspect_global_scene", [item["function"]["name"] for item in backend.tools])
        self.assertIn("evaluate_executor_candidates", [item["function"]["name"] for item in backend.tools])

    def test_evaluates_global_candidates_then_selects_executor(self) -> None:
        backend = FakeBackend(
            [
                json_call("evaluate_executor_candidates"),
                json_call("select_executor", robot_id=2, reason="robot 2 is closest among executable apple candidates"),
            ]
        )
        candidate_evaluation = {
            "candidate_executor_robot_ids": [2, 0],
            "candidate_scores": [
                {"robot_id": 2, "executable": True, "distance_to_target": 1.0},
                {"robot_id": 0, "executable": True, "distance_to_target": 4.0},
            ],
        }

        result = run_relay_agent(
            backend,
            task="Pick up the apple.",
            task_intent={"requestedAction": "PickupObject", "requestedObjectType": "Apple"},
            known_robot_ids=[0, 2],
            primary_robot_id=0,
            initial_summaries=[
                {"robot_id": 0, "visible_objects": [{"object_type": "Apple"}]},
                {"robot_id": 2, "visible_objects": [{"object_type": "Apple"}]},
            ],
            global_scene_summary={"known_robot_ids": [0, 2], "requested_object_type": "Apple"},
            observe_robot=lambda robot_id: self.fail("global evidence should be enough"),
            inspect_global_scene=lambda: {"known_robot_ids": [0, 2]},
            evaluate_executor_candidates=lambda: candidate_evaluation,
            validate_executor=lambda robot_id: (robot_id == 2, "robot 2 can pick up Apple"),
            validate_failure=lambda code, reason: (True, code, reason),
        )

        self.assertEqual(result["status"], "executor_selected")
        self.assertEqual(result["executor_robot_id"], 2)
        self.assertEqual(result["candidate_executor_robot_ids"], [2, 0])
        self.assertEqual(result["candidate_scores"], candidate_evaluation["candidate_scores"])
        self.assertEqual(result["selection_policy"], "llm_tool_calling_with_hard_validation")
        self.assertEqual([entry["tool"] for entry in result["trace"]], ["evaluate_executor_candidates", "select_executor"])
        self.assertIn("closest", result["agent_reason"])
        self.assertIn("validated", result["reason"])

    def test_repairs_select_executor_when_only_one_candidate_is_available(self) -> None:
        backend = FakeBackend(
            [
                json_call("evaluate_executor_candidates"),
                json_call("select_executor"),
            ]
        )

        result = run_relay_agent(
            backend,
            task="Pick up the bread.",
            task_intent={"requestedAction": "PickupObject", "requestedObjectType": "Bread"},
            known_robot_ids=[0, 1],
            primary_robot_id=1,
            initial_summaries=[
                {"robot_id": 0, "visible_objects": [{"object_type": "Bread"}]},
                {"robot_id": 1, "visible_objects": []},
            ],
            observe_robot=lambda robot_id: self.fail("candidate evidence should be enough"),
            evaluate_executor_candidates=lambda: {
                "candidate_executor_robot_ids": [0],
                "candidate_scores": [
                    {"robot_id": 0, "executable": True, "distance_to_target": 1.0},
                    {"robot_id": 1, "executable": False, "distance_to_target": None},
                ],
            },
            validate_executor=lambda robot_id: (robot_id == 0, "robot 0 can pick up Bread"),
            validate_failure=lambda code, reason: (True, code, reason),
        )

        self.assertEqual(result["status"], "executor_selected")
        self.assertEqual(result["executor_robot_id"], 0)
        self.assertEqual(result["trace"][1]["argument_repair"]["robot_id"], 0)
        self.assertEqual(result["agent_reason"], "only executable candidate from relay evidence")

    def test_agent_can_choose_non_top_candidate_if_hard_validation_passes(self) -> None:
        backend = FakeBackend(
            [
                json_call("evaluate_executor_candidates"),
                json_call("select_executor", robot_id=0, reason="primary is better for this context even though robot 2 is nearer"),
            ]
        )

        result = run_relay_agent(
            backend,
            task="Pick up the apple.",
            task_intent={"requestedAction": "PickupObject", "requestedObjectType": "Apple"},
            known_robot_ids=[0, 2],
            primary_robot_id=0,
            initial_summaries=[
                {"robot_id": 0, "visible_objects": [{"object_type": "Apple"}]},
                {"robot_id": 2, "visible_objects": [{"object_type": "Apple"}]},
            ],
            observe_robot=lambda robot_id: self.fail("candidate evidence should be enough"),
            evaluate_executor_candidates=lambda: {
                "candidate_executor_robot_ids": [2, 0],
                "candidate_scores": [
                    {"robot_id": 2, "executable": True, "distance_to_target": 1.0},
                    {"robot_id": 0, "executable": True, "distance_to_target": 4.0},
                ],
            },
            validate_executor=lambda robot_id: (True, f"robot {robot_id} passes hard validation"),
            validate_failure=lambda code, reason: (True, code, reason),
        )

        self.assertEqual(result["status"], "executor_selected")
        self.assertEqual(result["executor_robot_id"], 0)
        self.assertEqual(result["candidate_executor_robot_ids"], [2, 0])
        self.assertEqual(result["selection_policy"], "llm_tool_calling_with_hard_validation")



if __name__ == "__main__":
    unittest.main()
