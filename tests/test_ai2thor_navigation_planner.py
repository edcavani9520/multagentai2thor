from __future__ import annotations

import unittest
import threading
from types import MethodType

import ai2thor_receiver_server as module
from ai2thor_receiver_server import (
    NativeControllerReceiverHandler,
    NativeControllerThorServer,
    RobotState,
    astar_path,
    build_reachable_graph,
    choose_reachable_goal,
    path_to_actions,
    remove_graph_nodes,
    yaw_rotation_actions,
)


class NavigationPlannerPureFunctionTest(unittest.TestCase):
    def test_reachable_graph_connects_only_four_neighbors(self) -> None:
        positions = [
            {"x": 0.0, "y": 0.9, "z": 0.0},
            {"x": 0.25, "y": 0.9, "z": 0.0},
            {"x": 0.0, "y": 0.9, "z": 0.25},
            {"x": 0.25, "y": 0.9, "z": 0.25},
        ]
        graph, _ = build_reachable_graph(positions, grid_size=0.25, epsilon=0.01)

        self.assertIn(((0.25, 0.0), 0.25), graph[(0.0, 0.0)])
        self.assertIn(((0.0, 0.25), 0.25), graph[(0.0, 0.0)])
        self.assertNotIn((0.25, 0.25), [node for node, _ in graph[(0.0, 0.0)]])

    def test_astar_returns_shortest_grid_path(self) -> None:
        positions = [
            {"x": 0.0, "y": 0.9, "z": 0.0},
            {"x": 0.25, "y": 0.9, "z": 0.0},
            {"x": 0.5, "y": 0.9, "z": 0.0},
        ]
        graph, _ = build_reachable_graph(positions, grid_size=0.25, epsilon=0.01)

        self.assertEqual(astar_path(graph, (0.0, 0.0), (0.5, 0.0)), [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0)])

    def test_astar_returns_none_when_disconnected(self) -> None:
        positions = [
            {"x": 0.0, "y": 0.9, "z": 0.0},
            {"x": 1.0, "y": 0.9, "z": 0.0},
        ]
        graph, _ = build_reachable_graph(positions, grid_size=0.25, epsilon=0.01)

        self.assertIsNone(astar_path(graph, (0.0, 0.0), (1.0, 0.0)))

    def test_remove_graph_nodes_blocks_dynamic_obstacle_but_preserves_start(self) -> None:
        positions = [
            {"x": 0.0, "y": 0.9, "z": 0.0},
            {"x": 0.25, "y": 0.9, "z": 0.0},
            {"x": 0.5, "y": 0.9, "z": 0.0},
        ]
        graph, _ = build_reachable_graph(positions, grid_size=0.25, epsilon=0.01)

        pruned = remove_graph_nodes(graph, {(0.25, 0.0), (0.0, 0.0)}, preserve_nodes={(0.0, 0.0)})

        self.assertIn((0.0, 0.0), pruned)
        self.assertNotIn((0.25, 0.0), pruned)
        self.assertNotIn((0.25, 0.0), [node for node, _ in pruned[(0.0, 0.0)]])

    def test_path_to_actions_uses_yaw_and_moveahead(self) -> None:
        actions = path_to_actions(
            [
                {"x": 0.0, "y": 0.9, "z": 0.0},
                {"x": 0.25, "y": 0.9, "z": 0.0},
                {"x": 0.25, "y": 0.9, "z": 0.25},
            ],
            current_yaw=0.0,
            rotate_step_degrees=90.0,
        )

        self.assertEqual(
            actions,
            [
                {"action": "RotateRight"},
                {"action": "MoveAhead"},
                {"action": "RotateLeft"},
                {"action": "MoveAhead"},
            ],
        )

    def test_yaw_rotation_actions_uses_precise_degrees_for_non_grid_yaw(self) -> None:
        actions = yaw_rotation_actions(current_yaw=90.0, target_yaw=41.1859)

        self.assertEqual(actions[0]["action"], "RotateLeft")
        self.assertAlmostEqual(actions[0]["degrees"], 48.8141)

    def test_path_to_actions_corrects_non_grid_initial_yaw(self) -> None:
        actions = path_to_actions(
            [
                {"x": 0.0, "y": 0.9, "z": 0.0},
                {"x": 0.0, "y": 0.9, "z": 0.25},
            ],
            current_yaw=41.1859,
            rotate_step_degrees=90.0,
        )

        self.assertEqual(actions[0]["action"], "RotateLeft")
        self.assertAlmostEqual(actions[0]["degrees"], 41.1859)
        self.assertEqual(actions[1], {"action": "MoveAhead"})

    def test_choose_reachable_goal_uses_standing_point_not_object_center(self) -> None:
        reachable = [
            {"x": 0.0, "y": 0.9, "z": 0.0},
            {"x": 0.75, "y": 0.9, "z": 0.0},
            {"x": 1.5, "y": 0.9, "z": 0.0},
        ]

        goal = choose_reachable_goal(
            reachable,
            {"x": 0.0, "y": 0.9, "z": 0.0},
            min_distance=0.5,
            max_distance=1.0,
        )

        self.assertEqual(goal, {"x": 0.75, "y": 0.9, "z": 0.0})


class NavigationReceiverMethodTest(unittest.TestCase):
    def fake_server(self, *, yaw: float = 0.0) -> NativeControllerThorServer:
        server = NativeControllerThorServer.__new__(NativeControllerThorServer)
        server.lock = threading.RLock()
        server.robots = [
            RobotState(
                robot_id=0,
                name="Robot0",
                position={"x": 0.0, "y": 0.9, "z": 0.0},
                rotation={"x": 0.0, "y": yaw, "z": 0.0},
            )
        ]
        return server

    def test_reachable_positions_response(self) -> None:
        server = self.fake_server()
        server._get_reachable_positions = MethodType(
            lambda self, robot_ref=None: [{"x": 0.0, "y": 0.9, "z": 0.0}],
            server,
        )

        self.assertEqual(
            server.reachable_positions_response(0),
            {"status": "success", "robot_id": 0, "positions": [{"x": 0.0, "y": 0.9, "z": 0.0}]},
        )

    def test_goto_dry_run_returns_plan_without_executing(self) -> None:
        server = self.fake_server()
        server.capture_state = MethodType(lambda self, robot_ref=None, render_image=False: {"objects": []}, server)
        server._get_reachable_positions = MethodType(
            lambda self, robot_ref=None: [
                {"x": 0.0, "y": 0.9, "z": 0.0},
                {"x": 0.25, "y": 0.9, "z": 0.0},
            ],
            server,
        )
        server.execute_batch = MethodType(lambda *args, **kwargs: self.fail("dry-run /goto must not execute"), server)

        result = server.goto({"task_id": "goto-1", "robot_id": 0, "target_position": {"x": 0.25, "z": 0.0}})

        self.assertEqual(result["status"], "success")
        self.assertFalse(result["execute"])
        self.assertEqual(result["actions"], [{"action": "RotateRight"}, {"action": "MoveAhead"}])
        self.assertNotIn("execute_result", result)

    def test_goto_dry_run_corrects_non_grid_initial_yaw(self) -> None:
        server = self.fake_server(yaw=41.1859)
        server.capture_state = MethodType(lambda self, robot_ref=None, render_image=False: {"objects": []}, server)
        server._get_reachable_positions = MethodType(
            lambda self, robot_ref=None: [
                {"x": 0.0, "y": 0.9, "z": 0.0},
                {"x": 0.0, "y": 0.9, "z": 0.25},
            ],
            server,
        )
        server.execute_batch = MethodType(lambda *args, **kwargs: self.fail("dry-run /goto must not execute"), server)

        result = server.goto({"task_id": "goto-1", "robot_id": 0, "target_position": {"x": 0.0, "z": 0.25}})

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["actions"][0]["action"], "RotateLeft")
        self.assertAlmostEqual(result["actions"][0]["degrees"], 41.1859)
        self.assertEqual(result["actions"][1], {"action": "MoveAhead"})

    def test_goto_dry_run_faces_object_target_after_reaching_goal(self) -> None:
        server = self.fake_server(yaw=0.0)
        server.capture_state = MethodType(
            lambda self, robot_ref=None, render_image=False: {
                "objects": [
                    {
                        "id": "Target|1",
                        "type": "Target",
                        "position": {"x": 0.25, "y": 0.0, "z": 0.25},
                    }
                ]
            },
            server,
        )
        server._get_reachable_positions = MethodType(
            lambda self, robot_ref=None: [
                {"x": 0.0, "y": 0.9, "z": 0.0},
                {"x": 0.0, "y": 0.9, "z": 0.25},
            ],
            server,
        )
        server.execute_batch = MethodType(lambda *args, **kwargs: self.fail("dry-run /goto must not execute"), server)

        result = server.goto({"task_id": "goto-1", "robot_id": 0, "object_type": "Target", "min_distance": 0.0})

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["actions"], [{"action": "MoveAhead"}, {"action": "RotateRight"}])
        self.assertTrue(result["face_target"])
        self.assertEqual(result["face_target_yaw"], 90.0)

    def test_goto_dry_run_can_disable_face_target(self) -> None:
        server = self.fake_server(yaw=0.0)
        server.capture_state = MethodType(
            lambda self, robot_ref=None, render_image=False: {
                "objects": [
                    {
                        "id": "Target|1",
                        "type": "Target",
                        "position": {"x": 0.25, "y": 0.0, "z": 0.25},
                    }
                ]
            },
            server,
        )
        server._get_reachable_positions = MethodType(
            lambda self, robot_ref=None: [
                {"x": 0.0, "y": 0.9, "z": 0.0},
                {"x": 0.0, "y": 0.9, "z": 0.25},
            ],
            server,
        )
        server.execute_batch = MethodType(lambda *args, **kwargs: self.fail("dry-run /goto must not execute"), server)

        result = server.goto(
            {"task_id": "goto-1", "robot_id": 0, "object_type": "Target", "min_distance": 0.0, "face_target": False}
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["actions"], [{"action": "MoveAhead"}])
        self.assertFalse(result["face_target"])

    def test_plan_goto_avoids_other_robot_dynamic_obstacle(self) -> None:
        server = self.fake_server()
        server.robots.append(
            RobotState(
                robot_id=1,
                name="Robot1",
                position={"x": 0.25, "y": 0.9, "z": 0.0},
                rotation={"x": 0.0, "y": 0.0, "z": 0.0},
            )
        )
        server.capture_state = MethodType(lambda self, robot_ref=None, render_image=False: {"objects": []}, server)
        server._get_reachable_positions = MethodType(
            lambda self, robot_ref=None: [
                {"x": 0.0, "y": 0.9, "z": 0.0},
                {"x": 0.25, "y": 0.9, "z": 0.0},
                {"x": 0.5, "y": 0.9, "z": 0.0},
                {"x": 0.0, "y": 0.9, "z": 0.25},
                {"x": 0.25, "y": 0.9, "z": 0.25},
                {"x": 0.5, "y": 0.9, "z": 0.25},
            ],
            server,
        )

        result = server.plan_goto(
            {
                "task_id": "goto-1",
                "robot_id": 0,
                "target_position": {"x": 0.5, "z": 0.0},
                "dynamic_obstacle_radius": 0.01,
            },
            dynamic_obstacles=server._dynamic_obstacles_for_robot(0),
        )

        self.assertNotIn({"x": 0.25, "y": 0.9, "z": 0.0}, result["path"])
        self.assertEqual(result["goal_position"], {"x": 0.5, "y": 0.9, "z": 0.0})
        self.assertEqual(result["dynamic_obstacles"][0]["robot_id"], 1)
        self.assertGreater(result["blocked_node_count"], 0)

    def test_goto_execute_calls_execute_batch_with_stop_on_failure_true(self) -> None:
        server = self.fake_server()
        calls = []
        server.capture_state = MethodType(lambda self, robot_ref=None, render_image=False: {"objects": []}, server)
        server._get_reachable_positions = MethodType(
            lambda self, robot_ref=None: [
                {"x": 0.0, "y": 0.9, "z": 0.0},
                {"x": 0.0, "y": 0.9, "z": 0.25},
            ],
            server,
        )

        def execute_batch(self, actions, default_robot_ref=None, render_image=False, stop_on_failure=True):
            calls.append(
                {
                    "actions": actions,
                    "default_robot_ref": default_robot_ref,
                    "render_image": render_image,
                    "stop_on_failure": stop_on_failure,
                }
            )
            return {"status": "success", "results": []}

        server.execute_batch = MethodType(execute_batch, server)

        result = server.goto(
            {
                "task_id": "goto-1",
                "robot_id": 0,
                "target_position": {"x": 0.0, "z": 0.25},
                "execute": True,
            }
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(calls[0]["actions"], [{"action": "MoveAhead"}])
        self.assertEqual(calls[0]["default_robot_ref"], 0)
        self.assertTrue(calls[0]["stop_on_failure"])

    def test_goto_replans_after_blocking_failure_and_succeeds(self) -> None:
        server = self.fake_server()
        server.robots.append(
            RobotState(
                robot_id=1,
                name="Robot1",
                position={"x": 0.25, "y": 0.9, "z": 0.0},
                rotation={"x": 0.0, "y": 0.0, "z": 0.0},
            )
        )
        server.capture_state = MethodType(lambda self, robot_ref=None, render_image=False: {"objects": []}, server)
        plan_calls = []
        plans = [
            {
                "status": "success",
                "robot_id": 0,
                "target": {"kind": "position"},
                "target_position": {"x": 0.5, "y": 0.9, "z": 0.0},
                "start_position": {"x": 0.0, "y": 0.9, "z": 0.0},
                "goal_position": {"x": 0.5, "y": 0.9, "z": 0.0},
                "path": [{"x": 0.0, "y": 0.9, "z": 0.0}, {"x": 0.5, "y": 0.9, "z": 0.0}],
                "actions": [{"action": "MoveAhead"}],
                "estimated_distance": 0.5,
                "face_target": False,
                "face_target_yaw": 0.0,
                "planner": "reachable_positions_astar",
                "grid_size": 0.25,
                "rotate_step_degrees": 90.0,
                "dynamic_obstacles": [],
                "blocked_node_count": 0,
            },
            {
                "status": "success",
                "robot_id": 0,
                "target": {"kind": "position"},
                "target_position": {"x": 0.5, "y": 0.9, "z": 0.0},
                "start_position": {"x": 0.0, "y": 0.9, "z": 0.0},
                "goal_position": {"x": 0.5, "y": 0.9, "z": 0.0},
                "path": [{"x": 0.0, "y": 0.9, "z": 0.0}, {"x": 0.0, "y": 0.9, "z": 0.25}, {"x": 0.5, "y": 0.9, "z": 0.25}],
                "actions": [{"action": "RotateRight"}, {"action": "MoveAhead"}],
                "estimated_distance": 0.75,
                "face_target": False,
                "face_target_yaw": 0.0,
                "planner": "reachable_positions_astar",
                "grid_size": 0.25,
                "rotate_step_degrees": 90.0,
                "dynamic_obstacles": [{"robot_id": 1}],
                "blocked_node_count": 1,
            },
        ]

        def plan_goto(self, payload, dynamic_obstacles=None):
            plan_calls.append(dynamic_obstacles)
            return plans.pop(0)

        def execute_batch(self, actions, default_robot_ref=None, render_image=False, stop_on_failure=True):
            if len(plan_calls) == 1:
                return {
                    "status": "failed",
                    "results": [
                        {
                            "robot_id": 0,
                            "action": "MoveAhead",
                            "success": False,
                            "error": "Agent 1 is blocking Agent 0",
                        }
                    ],
                }
            return {"status": "success", "results": [{"robot_id": 0, "action": actions[0]["action"], "success": True}]}

        server.plan_goto = MethodType(plan_goto, server)
        server.execute_batch = MethodType(execute_batch, server)

        result = server.goto({"task_id": "goto-1", "robot_id": 0, "target_position": {"x": 0.5, "z": 0.0}, "execute": True})

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["replan_count"], 1)
        self.assertEqual(len(result["replan_trace"]), 2)
        self.assertEqual(plan_calls[0][0]["robot_id"], 1)
        self.assertEqual(result["execute_result"]["status"], "success")

    def test_goto_fails_after_replan_limit(self) -> None:
        server = self.fake_server()
        server.capture_state = MethodType(lambda self, robot_ref=None, render_image=False: {"objects": []}, server)

        def plan_goto(self, payload, dynamic_obstacles=None):
            return {
                "status": "success",
                "robot_id": 0,
                "target": {"kind": "position"},
                "target_position": {"x": 0.25, "y": 0.9, "z": 0.0},
                "start_position": {"x": 0.0, "y": 0.9, "z": 0.0},
                "goal_position": {"x": 0.25, "y": 0.9, "z": 0.0},
                "path": [{"x": 0.0, "y": 0.9, "z": 0.0}, {"x": 0.25, "y": 0.9, "z": 0.0}],
                "actions": [{"action": "MoveAhead"}],
                "estimated_distance": 0.25,
                "face_target": False,
                "face_target_yaw": 0.0,
                "planner": "reachable_positions_astar",
                "grid_size": 0.25,
                "rotate_step_degrees": 90.0,
                "dynamic_obstacles": [],
                "blocked_node_count": 0,
            }

        def execute_batch(self, actions, default_robot_ref=None, render_image=False, stop_on_failure=True):
            return {
                "status": "failed",
                "results": [
                    {
                        "robot_id": 0,
                        "action": "MoveAhead",
                        "success": False,
                        "error": "Agent 1 is blocking Agent 0",
                    }
                ],
            }

        server.plan_goto = MethodType(plan_goto, server)
        server.execute_batch = MethodType(execute_batch, server)

        result = server.goto(
            {
                "task_id": "goto-1",
                "robot_id": 0,
                "target_position": {"x": 0.25, "z": 0.0},
                "execute": True,
                "max_replans": 1,
            }
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_code"], "execution_failed_after_replans")
        self.assertEqual(result["replan_count"], 1)
        self.assertEqual(len(result["replan_trace"]), 2)
        self.assertIn("blocking", result["failed_action"]["error"])

    def test_goto_replan_failure_survives_broken_capture_state(self) -> None:
        server = self.fake_server()
        plan_calls = 0

        def plan_goto(self, payload, dynamic_obstacles=None):
            nonlocal plan_calls
            plan_calls += 1
            if plan_calls > 1:
                raise module.NavigationPlanningError(
                    "no_reachable_positions",
                    "GetReachablePositions returned no usable positions",
                )
            return {
                "status": "success",
                "robot_id": 0,
                "target": {"kind": "position"},
                "target_position": {"x": 0.25, "y": 0.9, "z": 0.0},
                "start_position": {"x": 0.0, "y": 0.9, "z": 0.0},
                "goal_position": {"x": 0.25, "y": 0.9, "z": 0.0},
                "path": [{"x": 0.0, "y": 0.9, "z": 0.0}, {"x": 0.25, "y": 0.9, "z": 0.0}],
                "actions": [{"action": "MoveAhead"}],
                "estimated_distance": 0.25,
                "face_target": False,
                "face_target_yaw": 0.0,
                "planner": "reachable_positions_astar",
                "grid_size": 0.25,
                "rotate_step_degrees": 90.0,
                "dynamic_obstacles": [],
                "blocked_node_count": 0,
            }

        def execute_batch(self, actions, default_robot_ref=None, render_image=False, stop_on_failure=True):
            return {
                "status": "failed",
                "results": [
                    {
                        "robot_id": 0,
                        "action": "MoveAhead",
                        "success": False,
                        "error": "Agent 1 is blocking Agent 0",
                    }
                ],
            }

        def capture_state(self, robot_ref=None, render_image=False):
            raise ValueError("write to closed file")

        server.plan_goto = MethodType(plan_goto, server)
        server.execute_batch = MethodType(execute_batch, server)
        server.capture_state = MethodType(capture_state, server)

        result = server.goto(
            {
                "task_id": "goto-1",
                "robot_id": 0,
                "target_position": {"x": 0.25, "z": 0.0},
                "execute": True,
            }
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_code"], "execution_failed_after_replans")
        self.assertEqual(result["replan_error_code"], "no_reachable_positions")
        self.assertEqual(result["execute_result"]["state"]["status"], "unavailable")
        self.assertEqual(result["execute_result"]["state"]["error_type"], "ValueError")

    def test_execute_batch_uses_safe_capture_state(self) -> None:
        server = self.fake_server()

        def execute(self, robot_ref, action, render_image=False, **kwargs):
            return {
                "robot_id": robot_ref,
                "action": action,
                "success": True,
                "error": None,
            }

        def capture_state(self, robot_ref=None, render_image=False):
            raise RuntimeError("controller pipe closed")

        server.execute = MethodType(execute, server)
        server.capture_state = MethodType(capture_state, server)

        result = server.execute_batch([{"action": "MoveAhead"}], default_robot_ref=0)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["state"]["status"], "unavailable")
        self.assertEqual(result["state"]["error_type"], "RuntimeError")

    def test_controller_step_is_serialized(self) -> None:
        server = self.fake_server()
        entered_first = threading.Event()
        release_first = threading.Event()
        second_entered = threading.Event()
        order: list[str] = []
        active = 0
        max_active = 0
        guard = threading.Lock()

        class FakeController:
            def step(self, action):
                nonlocal active, max_active
                with guard:
                    active += 1
                    max_active = max(max_active, active)
                    order.append(action["action"])
                if action["action"] == "first":
                    entered_first.set()
                    release_first.wait(timeout=2.0)
                else:
                    second_entered.set()
                with guard:
                    active -= 1
                return {"action": action["action"]}

        server.controller = FakeController()

        first_result = {}
        second_result = {}

        def run_first() -> None:
            first_result["value"] = server._controller_step({"action": "first"})

        def run_second() -> None:
            second_result["value"] = server._controller_step({"action": "second"})

        t1 = threading.Thread(target=run_first)
        t2 = threading.Thread(target=run_second)
        t1.start()
        self.assertTrue(entered_first.wait(timeout=2.0))
        t2.start()
        self.assertFalse(second_entered.wait(timeout=0.2))
        self.assertEqual(order, ["first"])
        self.assertEqual(max_active, 1)
        release_first.set()
        t1.join(timeout=2.0)
        t2.join(timeout=2.0)

        self.assertFalse(t1.is_alive())
        self.assertFalse(t2.is_alive())
        self.assertEqual(order, ["first", "second"])
        self.assertEqual(max_active, 1)
        self.assertEqual(first_result["value"], {"action": "first"})
        self.assertEqual(second_result["value"], {"action": "second"})

    def test_goto_object_type_not_found(self) -> None:
        server = self.fake_server()
        server.capture_state = MethodType(lambda self, robot_ref=None, render_image=False: {"objects": []}, server)
        server._get_reachable_positions = MethodType(lambda self, robot_ref=None: [], server)

        result = server.goto({"task_id": "goto-1", "robot_id": 0, "object_type": "Fridge"})

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_code"], "target_not_found")

    def test_handler_health_reports_service_without_controller_ready_check(self) -> None:
        class FakeThor:
            robots = [object(), object()]

        fake = FakeThor()
        original = module.thor_instance
        module.thor_instance = fake
        try:
            handler = NativeControllerReceiverHandler.__new__(NativeControllerReceiverHandler)
            handler.path = "/health"
            sent = []
            handler._send_json = MethodType(lambda self, code, payload: sent.append((code, payload)), handler)

            handler.do_GET()
        finally:
            module.thor_instance = original

        self.assertEqual(sent, [(200, {"status": "ok", "service": "ai2thor_receiver_server", "robots": 2})])

    def test_handler_goto_dispatches_to_thor_instance(self) -> None:
        class FakeThor:
            controller = object()

            def goto(self, payload):
                self.payload = payload
                return {"status": "success", "actions": [{"action": "MoveAhead"}]}

        fake = FakeThor()
        original = module.thor_instance
        module.thor_instance = fake
        try:
            handler = NativeControllerReceiverHandler.__new__(NativeControllerReceiverHandler)
            handler._controller_ready = MethodType(lambda self: True, handler)
            handler._read_json = MethodType(lambda self: {"task_id": "goto-1", "execute": False}, handler)
            sent = []
            handler._send_json = MethodType(lambda self, code, payload: sent.append((code, payload)), handler)

            handler._handle_goto()
        finally:
            module.thor_instance = original

        self.assertEqual(sent, [(200, {"status": "success", "actions": [{"action": "MoveAhead"}]})])
        self.assertEqual(fake.payload, {"task_id": "goto-1", "execute": False})

    def test_handler_goto_returns_json_for_receiver_exception(self) -> None:
        class FakeThor:
            controller = object()

            def goto(self, payload):
                raise RuntimeError("controller pipe closed")

        fake = FakeThor()
        original = module.thor_instance
        module.thor_instance = fake
        try:
            handler = NativeControllerReceiverHandler.__new__(NativeControllerReceiverHandler)
            handler._controller_ready = MethodType(lambda self: True, handler)
            handler._read_json = MethodType(lambda self: {"task_id": "goto-1", "execute": True}, handler)
            sent = []
            handler._send_json = MethodType(lambda self, code, payload: sent.append((code, payload)), handler)

            handler._handle_goto()
        finally:
            module.thor_instance = original

        self.assertEqual(sent[0][0], 500)
        self.assertEqual(sent[0][1]["status"], "failed")
        self.assertEqual(sent[0][1]["task_id"], "goto-1")
        self.assertEqual(sent[0][1]["error_code"], "receiver_exception")
        self.assertEqual(sent[0][1]["error_type"], "RuntimeError")
        self.assertIn("controller pipe closed", sent[0][1]["error"])


if __name__ == "__main__":
    unittest.main()
