#!/usr/bin/env python3
"""
AI2-THOR Multi-Agent Receiver Server
====================================
HTTP receiver using AI2-THOR's native multi-agent support through the
standard Controller:

    Controller(agentCount=N)
    controller.step(action=..., agentId=robot_id)

This keeps one shared Unity world and avoids RobotController/WSGI, which is
useful when WSGI hangs on Initialize(agentCount=N). Remote agents still talk to
this script over HTTP, so they do not need direct Unity access.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from typing import Optional
from urllib.parse import parse_qs, urlparse

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from ai2thor.server import MultiAgentEvent


DEFAULT_AGENT_Y = 0.900999128818512
_CV2 = None
_CV2_IMPORT_ERROR = None


def get_cv2():
    global _CV2, _CV2_IMPORT_ERROR
    if _CV2 is not None:
        return _CV2
    if _CV2_IMPORT_ERROR is not None:
        return None
    try:
        import cv2
    except Exception as exc:
        _CV2_IMPORT_ERROR = exc
        return None
    _CV2 = cv2
    return _CV2


def log_event(channel: str, message: str = "", *, file=None, blank_before: bool = False):
    stream = file or sys.stdout
    if blank_before:
        print("", file=stream, flush=True)
    for line in (str(message).splitlines() or [""]):
        print(f"[{channel}] {line}", file=stream, flush=True)


@dataclass
class RobotState:
    robot_id: int
    name: str
    position: dict
    rotation: dict
    horizon: float = 0.0
    task: str = ""
    last_action: str = "Pass"
    last_success: bool = True
    last_error: Optional[str] = None
    last_event: object = None
    inventory: list[dict] = field(default_factory=list)
    action_count: int = 0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        held_object = self.inventory[0] if self.inventory else None
        return {
            "robot_id": self.robot_id,
            "name": self.name,
            "position": self.position,
            "rotation": self.rotation,
            "horizon": self.horizon,
            "inventory": self.inventory,
            "held_object": held_object,
            "task": self.task,
            "last_action": self.last_action,
            "last_success": self.last_success,
            "last_error": self.last_error,
            "action_count": self.action_count,
            "controller": "native_standard_controller",
        }


class NativeControllerThorServer:
    def __init__(
        self,
        scene: str = "FloorPlan1",
        robot_count: int = 2,
        headless: bool = True,
        width: int = 600,
        height: int = 600,
        robot0_dx: float = 0.0,
        robot0_dz: float = 0.0,
        robot0_left: float = 0.0,
        robot0_right: float = 0.0,
        robot0_dyaw: float = 0.0,
        robot0_at_fridge: bool = False,
        robot0_fridge_distance: float = 1.0,
    ):
        self.scene = scene
        self.robot_count = max(1, robot_count)
        self.headless = headless
        self.width = width
        self.height = height
        self.robot0_dx = robot0_dx
        self.robot0_dz = robot0_dz
        self.robot0_left = robot0_left
        self.robot0_right = robot0_right
        self.robot0_dyaw = robot0_dyaw
        self.robot0_at_fridge = robot0_at_fridge
        self.robot0_fridge_distance = robot0_fridge_distance

        self.controller = None
        self.robots: list[RobotState] = []
        self.lock = threading.RLock()
        self._step_count = 0
        self._window_names: dict[int, str] = {}

    def start(self):
        kwargs = dict(
            scene=self.scene,
            width=self.width,
            height=self.height,
            agentCount=self.robot_count,
            port=0,
        )
        if self.headless:
            kwargs["platform"] = CloudRendering

        log_event(
            "SIM IN",
            f"Start standard Controller(scene={self.scene}, robots={self.robot_count}, headless={self.headless})",
        )
        self.controller = Controller(**kwargs)
        self._validate_agent_count(self.controller.last_event)
        self._setup_robots()
        log_event("SIM OUT", "Native standard Controller multi-agent ready")

    def stop(self):
        with self.lock:
            if self.controller is not None:
                log_event("SIM IN", "Stop standard Controller")
                self.controller.stop()
                self.controller = None

    def reset(self, scene: Optional[str] = None, robot_count: Optional[int] = None):
        with self.lock:
            if scene:
                self.scene = scene
            if robot_count is not None:
                self.robot_count = max(1, int(robot_count))
            self.stop()
            self._step_count = 0
            self.start()

    def _validate_agent_count(self, event):
        events = self._events_from(event)
        meta_agents = event.metadata.get("agents", []) if hasattr(event, "metadata") else []
        count = max(len(events), len(meta_agents), 1)
        log_event("SIM OUT", f"Controller agents={count} requested={self.robot_count}")
        if count < self.robot_count:
            raise RuntimeError(
                f"Controller created {count} agents, requested {self.robot_count}. "
                "This AI2-THOR build may not support agentCount with standard Controller."
            )

    def _setup_robots(self):
        positions = self._select_spawn_positions(self.robot_count)
        if self.robot0_at_fridge and positions:
            positions[0] = self._select_position_in_front_of_object("Fridge", positions[0])
            positions = self._avoid_spawn_overlap(positions)

        center = self._position_center(positions)
        self.robots = []
        log_event("SIM OUT", "Initial native Controller agent positions:")

        for robot_id in range(self.robot_count):
            pos = dict(positions[robot_id])
            yaw = self._yaw_to_target(pos, center)
            if robot_id == 0 and self.robot0_at_fridge:
                fridge_pos = self._find_object_position("Fridge")
                if fridge_pos:
                    yaw = self._yaw_to_target(pos, fridge_pos)
            if robot_id == 0:
                pos, yaw = self._apply_robot0_initial_offset(pos, yaw)

            robot = RobotState(
                robot_id=robot_id,
                name=f"Robot{robot_id}",
                position=pos,
                rotation={"x": 0, "y": yaw, "z": 0},
                task=f"Native standard Controller agent slot {robot_id}",
            )
            self.robots.append(robot)
            event = self._controller_step(
                {
                    "action": "TeleportFull",
                    "agentId": robot_id,
                    "position": robot.position,
                    "rotation": robot.rotation,
                    "horizon": robot.horizon,
                    "standing": True,
                }
            )
            agent_event = self._event_for_robot(event, robot_id)
            robot.last_event = agent_event
            self._update_robot_pose_from_metadata(robot, agent_event.metadata)
            log_event(
                "SIM OUT",
                f"  {robot.name}: pos=({robot.position['x']:.2f}, "
                f"{robot.position.get('y', DEFAULT_AGENT_Y):.2f}, {robot.position['z']:.2f}), "
                f"yaw={robot.rotation.get('y', 0):.1f}",
            )

    def resolve_robot(self, robot_ref=None) -> RobotState:
        if robot_ref is None or robot_ref == "":
            return self.robots[0]
        if isinstance(robot_ref, str):
            robot_ref = robot_ref.strip()
            for robot in self.robots:
                if robot.name.lower() == robot_ref.lower():
                    return robot
            if robot_ref.lower().startswith("robot") and robot_ref[5:].isdigit():
                robot_ref = int(robot_ref[5:])
        try:
            robot_id = int(robot_ref)
        except (TypeError, ValueError):
            raise ValueError(f"unknown robot_id: {robot_ref!r}")
        if robot_id < 0 or robot_id >= len(self.robots):
            raise ValueError(f"robot_id out of range: {robot_id} (0..{len(self.robots)-1})")
        return self.robots[robot_id]

    def observe(self, robot_ref=None, render_image: bool = True) -> dict:
        with self.lock:
            robot = self.resolve_robot(robot_ref)
            event = self._controller_step({"action": "Pass", "agentId": robot.robot_id, "renderImage": render_image})
            self._refresh_all_robot_states(event)
            agent_event = self._event_for_robot(event, robot.robot_id)
            data = {
                "status": "success",
                "robot_id": robot.robot_id,
                "robot": robot.to_dict(),
                "objects": self._objects_from_metadata(agent_event.metadata),
                "metadata": agent_event.metadata,
            }
            if render_image:
                image_b64 = self._encode_image(agent_event)
                if image_b64:
                    data["image_base64"] = image_b64
            return data

    def execute(self, robot_ref, action: str, render_image: bool = False, **kwargs) -> dict:
        with self.lock:
            robot = self.resolve_robot(robot_ref)
            before_pose = self._robot_pose_snapshot(robot)
            before_meta = getattr(robot.last_event, "metadata", {}) if robot.last_event is not None else {}
            result = {
                "robot_id": robot.robot_id,
                "robot_name": robot.name,
                "action": action,
                "success": False,
                "error": None,
                "agent": None,
                "inventory": [],
                "controller": "native_standard_controller",
                "robot_pose": before_pose,
                "robot_pose_changed": False,
                "interacted_objects": [],
            }
            try:
                action_dict = dict(kwargs)
                action_dict["action"] = action
                action_dict["agentId"] = robot.robot_id
                action_dict["renderImage"] = render_image
                event = self._controller_step(action_dict)
                self._refresh_all_robot_states(event)
                agent_event = self._event_for_robot(event, robot.robot_id)
                meta = agent_event.metadata
                success = meta.get("lastActionSuccess", False)
                result["success"] = success
                if not success:
                    result["error"] = meta.get("errorMessage", "unknown error")
                result["agent"] = meta.get("agent", {})
                result["inventory"] = self._inventory_from_metadata(meta)
                robot.last_event = agent_event
                robot.last_action = action
                robot.last_success = success
                robot.last_error = result["error"]
                robot.action_count += 1
                self._step_count += 1
                self._update_robot_pose_from_metadata(robot, meta)
                result["robot"] = robot.to_dict()
                result["robot_pose"] = self._robot_pose_snapshot(robot)
                result["robot_pose_changed"] = self._pose_changed(before_pose, result["robot_pose"])
                if result["robot_pose_changed"]:
                    result["robot_pose_delta"] = {
                        "before": before_pose,
                        "after": result["robot_pose"],
                    }
                result["interacted_objects"] = self._interacted_objects_after_action(
                    action,
                    kwargs,
                    before_meta,
                    meta,
                )
                result["held_object"] = robot.inventory[0] if robot.inventory else None
                if render_image:
                    image_b64 = self._encode_image(agent_event)
                    if image_b64:
                        result["image_base64"] = image_b64
            except ValueError as exc:
                result["error"] = str(exc)[:200]
                robot.last_success = False
                robot.last_error = result["error"]
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
                robot.last_success = False
                robot.last_error = result["error"]
            return result

    def execute_batch(
        self,
        actions: list[dict],
        default_robot_ref=None,
        render_image: bool = False,
        stop_on_failure: bool = True,
    ) -> dict:
        results = []
        all_succeeded = True
        last_robot_ref = default_robot_ref
        for i, act in enumerate(actions):
            if not isinstance(act, dict):
                results.append({"index": i, "action": str(act), "success": False, "error": "action must be a dict"})
                all_succeeded = False
                if stop_on_failure:
                    break
                continue
            act = dict(act)
            robot_ref = default_robot_ref
            for key in ("robot_id", "robot", "agent_id"):
                if key in act:
                    robot_ref = act.pop(key)
                    break
            action_name = act.pop("action", None)
            if action_name is None:
                results.append({"index": i, "robot_id": robot_ref, "success": False, "error": "missing action"})
                all_succeeded = False
                if stop_on_failure:
                    break
                continue
            try:
                item = self.execute(robot_ref, action_name, render_image=render_image, **act)
            except ValueError as exc:
                item = {"robot_id": robot_ref, "action": action_name, "success": False, "error": str(exc)}
            item["index"] = i
            results.append(item)
            if "robot_id" in item:
                last_robot_ref = item["robot_id"]
            if not item.get("success"):
                all_succeeded = False
                if stop_on_failure:
                    break
        if all_succeeded and len(results) == len(actions):
            status = "success"
        elif any(item.get("success") for item in results):
            status = "partial"
        else:
            status = "failed"
        return {"status": status, "results": results, "state": self.capture_state(last_robot_ref)}

    def capture_state(self, robot_ref=None, render_image: bool = False) -> dict:
        with self.lock:
            selected = self.resolve_robot(robot_ref) if robot_ref is not None else self.robots[0]
            event = self._controller_step({"action": "Pass", "agentId": selected.robot_id, "renderImage": render_image})
            self._refresh_all_robot_states(event)
            agent_event = self._event_for_robot(event, selected.robot_id)
            meta = agent_event.metadata
            state = {
                "sceneName": meta.get("sceneName", self.scene),
                "step": self._step_count,
                "selected_robot_id": selected.robot_id,
                "controller_mode": "native_standard_controller_shared_unity",
                "agent": {
                    "position": selected.position,
                    "rotation": selected.rotation,
                    "horizon": selected.horizon,
                },
                "robots": [robot.to_dict() for robot in self.robots],
                "inventory": self._inventory_from_metadata(meta),
                "objects": self._objects_from_metadata(meta),
            }
            state["num_objects"] = len(state["objects"])
            if render_image:
                image_b64 = self._encode_image(agent_event)
                if image_b64:
                    state["image_base64"] = image_b64
            return state

    def _controller_step(self, action: dict):
        return self.controller.step(action)

    @staticmethod
    def _events_from(event):
        return event.events if isinstance(event, MultiAgentEvent) else [event]

    def _event_for_robot(self, event, robot_id: int):
        events = self._events_from(event)
        if robot_id >= len(events):
            return events[0]
        return events[robot_id]

    def _refresh_all_robot_states(self, event):
        events = self._events_from(event)
        for robot_id, agent_event in enumerate(events[: len(self.robots)]):
            robot = self.robots[robot_id]
            robot.last_event = agent_event
            self._update_robot_pose_from_metadata(robot, agent_event.metadata)

    def _update_robot_pose_from_metadata(self, robot: RobotState, meta: dict):
        agent = meta.get("agent", {})
        if agent.get("position"):
            robot.position = agent["position"]
        if agent.get("rotation"):
            robot.rotation = agent["rotation"]
        if "cameraHorizon" in agent:
            robot.horizon = agent["cameraHorizon"]
        robot.inventory = self._inventory_from_metadata(meta)

    @staticmethod
    def _inventory_from_metadata(meta: dict) -> list[dict]:
        return [
            {"objectId": item["objectId"], "objectType": item.get("objectType", "")}
            for item in meta.get("inventoryObjects", [])
        ]

    @staticmethod
    def _robot_pose_snapshot(robot: RobotState) -> dict:
        return {
            "position": dict(robot.position or {}),
            "rotation": dict(robot.rotation or {}),
            "horizon": robot.horizon,
        }

    @classmethod
    def _pose_changed(cls, before: dict, after: dict, eps: float = 1e-5) -> bool:
        return (
            cls._dict_changed(before.get("position", {}), after.get("position", {}), eps)
            or cls._dict_changed(before.get("rotation", {}), after.get("rotation", {}), eps)
            or cls._value_changed(before.get("horizon"), after.get("horizon"), eps)
        )

    @classmethod
    def _dict_changed(cls, before: dict, after: dict, eps: float) -> bool:
        keys = set(before) | set(after)
        return any(cls._value_changed(before.get(key), after.get(key), eps) for key in keys)

    @staticmethod
    def _value_changed(before, after, eps: float) -> bool:
        if before is None and after is None:
            return False
        try:
            return abs(float(before) - float(after)) > eps
        except (TypeError, ValueError):
            return before != after

    def _interacted_objects_after_action(
        self,
        action: str,
        action_kwargs: dict,
        before_meta: dict,
        after_meta: dict,
    ) -> list[dict]:
        object_ids = self._interaction_object_ids(action, action_kwargs, before_meta, after_meta)
        objects = []
        for object_id in object_ids:
            after_state = self._object_state_from_metadata(after_meta, object_id)
            before_state = self._object_state_from_metadata(before_meta, object_id)
            inventory_state = self._inventory_object_state(after_meta, object_id)
            state = after_state or inventory_state or before_state
            if not state:
                state = {"id": object_id, "objectId": object_id, "missing": True}
            state["before"] = before_state
            state["after"] = after_state or inventory_state
            state["state_changed"] = before_state != (after_state or inventory_state)
            objects.append(state)
        return objects

    def _interaction_object_ids(
        self,
        action: str,
        action_kwargs: dict,
        before_meta: dict,
        after_meta: dict,
    ) -> list[str]:
        object_ids: list[str] = []

        def add(value):
            if isinstance(value, str) and value and value not in object_ids:
                object_ids.append(value)
            elif isinstance(value, list):
                for item in value:
                    add(item)

        for key, value in action_kwargs.items():
            key_lower = key.lower()
            if key_lower == "objectid" or key_lower.endswith("objectid") or key_lower.endswith("objectids"):
                add(value)

        before_inventory = [item.get("objectId") for item in before_meta.get("inventoryObjects", [])]
        after_inventory = [item.get("objectId") for item in after_meta.get("inventoryObjects", [])]
        action_lower = (action or "").lower()
        if action_lower in {"putobject", "dropobject", "throwobject"}:
            add(before_inventory)
        if action_lower in {"pickupobject", "putobject", "dropobject", "throwobject"}:
            add(after_inventory)
        if before_inventory != after_inventory:
            add(before_inventory)
            add(after_inventory)

        return object_ids

    @staticmethod
    def _object_state_from_metadata(meta: dict, object_id: str) -> Optional[dict]:
        for obj in meta.get("objects", []):
            if obj.get("objectId") == object_id:
                return {
                    "id": object_id,
                    "objectId": object_id,
                    "type": obj.get("objectType", object_id.split("|")[0]),
                    "name": obj.get("name", ""),
                    "position": obj.get("position", {}),
                    "rotation": obj.get("rotation", {}),
                    "distance": obj.get("distance", -1),
                    "visible": obj.get("visible", False),
                    "pickupable": obj.get("pickupable", False),
                    "moveable": obj.get("moveable", False),
                    "receptacle": obj.get("receptacle", False),
                    "openable": obj.get("openable", False),
                    "isOpen": obj.get("isOpen", False),
                    "isPickedUp": obj.get("isPickedUp", False),
                    "parentReceptacles": obj.get("parentReceptacles"),
                    "receptacleObjectIds": obj.get("receptacleObjectIds", []),
                }
        return None

    @staticmethod
    def _inventory_object_state(meta: dict, object_id: str) -> Optional[dict]:
        for item in meta.get("inventoryObjects", []):
            if item.get("objectId") == object_id:
                return {
                    "id": object_id,
                    "objectId": object_id,
                    "type": item.get("objectType", object_id.split("|")[0]),
                    "inInventory": True,
                }
        return None

    def _select_spawn_positions(self, count: int) -> list[dict]:
        reachable = self._get_reachable_positions()
        if not reachable:
            reachable = self._fallback_positions(count)
        if len(reachable) <= count:
            return [dict(pos) for pos in reachable[:count]]
        center = self._position_center(reachable)
        first = max(reachable, key=lambda pos: self._dist2(pos, center))
        selected = [first]
        remaining = [pos for pos in reachable if pos is not first]
        while len(selected) < count and remaining:
            best = max(remaining, key=lambda pos: min(self._dist2(pos, item) for item in selected))
            selected.append(best)
            remaining.remove(best)
        return [dict(pos) for pos in selected]

    def _get_reachable_positions(self) -> list[dict]:
        try:
            event = self._controller_step({"action": "GetReachablePositions", "agentId": 0})
            meta = self._event_for_robot(event, 0).metadata
            positions = meta.get("actionReturn") or []
        except Exception as exc:
            log_event("SIM OUT", f"WARNING GetReachablePositions failed: {exc}")
            return []
        default_y = DEFAULT_AGENT_Y
        cleaned = []
        for pos in positions:
            if "x" not in pos or "z" not in pos:
                continue
            cleaned.append({"x": float(pos["x"]), "y": float(pos.get("y", default_y)), "z": float(pos["z"])})
        return cleaned

    def _select_position_in_front_of_object(self, object_type: str, fallback: dict) -> dict:
        target_pos = self._find_object_position(object_type)
        if not target_pos:
            log_event("SIM OUT", f"WARNING cannot place Robot0 at {object_type}: object not found")
            return fallback
        reachable = self._get_reachable_positions()
        if not reachable:
            return fallback
        min_distance = max(0.1, self.robot0_fridge_distance)
        candidates = [pos for pos in reachable if self._dist2(pos, target_pos) >= min_distance * min_distance]
        if not candidates:
            candidates = reachable
        selected = min(candidates, key=lambda pos: self._dist2(pos, target_pos))
        log_event(
            "SIM OUT",
            f"Robot0 placed near {object_type}: object=({target_pos.get('x', 0):.2f}, "
            f"{target_pos.get('z', 0):.2f}), stand=({selected['x']:.2f}, {selected['z']:.2f})",
        )
        return dict(selected)

    def _find_object_position(self, object_type: str) -> Optional[dict]:
        event = self._controller_step({"action": "Pass", "agentId": 0, "renderImage": False})
        meta = self._event_for_robot(event, 0).metadata
        for obj in meta.get("objects", []):
            if obj.get("objectType") == object_type and obj.get("position"):
                return obj["position"]
        return None

    def _avoid_spawn_overlap(self, positions: list[dict], min_distance: float = 0.75) -> list[dict]:
        reachable = self._get_reachable_positions()
        if not reachable:
            return positions
        adjusted = [dict(positions[0])]
        min_dist2 = min_distance * min_distance
        for pos in positions[1:]:
            if all(self._dist2(pos, existing) >= min_dist2 for existing in adjusted):
                adjusted.append(dict(pos))
                continue
            replacement = max(reachable, key=lambda candidate: min(self._dist2(candidate, existing) for existing in adjusted))
            adjusted.append(dict(replacement))
            log_event(
                "SIM OUT",
                f"Adjusted spawn to avoid overlap: ({pos['x']:.2f}, {pos['z']:.2f}) -> "
                f"({replacement['x']:.2f}, {replacement['z']:.2f})",
            )
        return adjusted

    def _apply_robot0_initial_offset(self, pos: dict, yaw: float) -> tuple[dict, float]:
        if not any((self.robot0_dx, self.robot0_dz, self.robot0_left, self.robot0_right, self.robot0_dyaw)):
            return pos, yaw
        adjusted = dict(pos)
        adjusted["x"] += self.robot0_dx
        adjusted["z"] += self.robot0_dz
        lateral = self.robot0_left - self.robot0_right
        if lateral:
            yaw_rad = math.radians(yaw)
            adjusted["x"] += lateral * math.cos(yaw_rad)
            adjusted["z"] += lateral * math.sin(yaw_rad)
        adjusted_yaw = yaw + self.robot0_dyaw
        log_event(
            "SIM IN",
            f"Apply Robot0 initial offset: dx={self.robot0_dx:.2f}, dz={self.robot0_dz:.2f}, "
            f"left={self.robot0_left:.2f}, right={self.robot0_right:.2f}, dyaw={self.robot0_dyaw:.1f}",
        )
        return adjusted, adjusted_yaw

    def initialize_windows(self):
        cv2 = get_cv2()
        if cv2 is None:
            raise RuntimeError(f"OpenCV import failed: {_CV2_IMPORT_ERROR}")
        self._window_names = {}
        for robot in self.robots:
            name = f"{robot.name} Native Controller View"
            self._window_names[robot.robot_id] = name
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name, self.width, self.height)
            col = robot.robot_id % 3
            row = robot.robot_id // 3
            cv2.moveWindow(name, 40 + col * (self.width + 30), 60 + row * (self.height + 60))
        log_event("LOCAL OUT", f"Opened {len(self._window_names)} robot view windows")

    def display_loop(self, fps: float = 5.0):
        cv2 = get_cv2()
        if cv2 is None:
            log_event("LOCAL OUT", "OpenCV unavailable; running without local windows")
            log_event("LOCAL OUT", f"OpenCV import error: {_CV2_IMPORT_ERROR}")
            return False
        try:
            self.initialize_windows()
        except Exception as exc:
            log_event("LOCAL OUT", "OpenCV cannot create windows; running without local windows")
            log_event("LOCAL OUT", f"OpenCV error: {exc}")
            return False
        delay_ms = max(1, int(1000 / max(0.1, fps)))
        try:
            while True:
                with self.lock:
                    event = self._controller_step({"action": "Pass", "agentId": 0, "renderImage": True})
                    self._refresh_all_robot_states(event)
                    events = self._events_from(event)
                    for robot in self.robots:
                        if robot.robot_id >= len(events):
                            continue
                        img = self._event_bgr_image(events[robot.robot_id])
                        if img is None:
                            continue
                        img = img.copy()
                        self._draw_overlay(img, robot)
                        cv2.imshow(self._window_names[robot.robot_id], img)
                key = cv2.waitKey(delay_ms) & 0xFF
                if key in (ord("q"), 27):
                    log_event("LOCAL IN", "GUI quit requested")
                    break
        finally:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        return True

    @staticmethod
    def _draw_overlay(img, robot: RobotState):
        cv2 = get_cv2()
        if cv2 is None:
            return
        pos = robot.position
        color = (0, 255, 0) if robot.last_success else (0, 0, 255)
        cv2.putText(img, f"{robot.name} | native controller | action #{robot.action_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(img, f"Pos: ({pos['x']:.2f}, {pos.get('y', DEFAULT_AGENT_Y):.2f}, {pos['z']:.2f})", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        cv2.putText(img, f"{robot.last_action}: {'OK' if robot.last_success else 'FAIL'}", (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    @staticmethod
    def _objects_from_metadata(meta: dict) -> list[dict]:
        objects = []
        for obj in meta.get("objects", []):
            object_id = obj.get("objectId", "")
            if object_id.startswith("Floor|"):
                continue
            objects.append(
                {
                    "id": object_id,
                    "type": obj.get("objectType", object_id.split("|")[0]),
                    "position": obj.get("position", {}),
                    "distance": obj.get("distance", -1),
                    "pickupable": obj.get("pickupable", False),
                    "receptacle": obj.get("receptacle", False),
                    "openable": obj.get("openable", False),
                    "isOpen": obj.get("isOpen", False),
                    "visible": obj.get("visible", False),
                }
            )
        objects.sort(key=lambda item: item["distance"] if item["distance"] >= 0 else float("inf"))
        return objects

    @staticmethod
    def _event_bgr_image(event):
        if event is None:
            return None
        if hasattr(event, "cv2img") and event.cv2img is not None:
            return event.cv2img
        cv2 = get_cv2()
        if cv2 is not None and hasattr(event, "frame") and event.frame is not None:
            return event.frame[:, :, ::-1]
        return None

    @classmethod
    def _encode_image(cls, event) -> Optional[str]:
        if hasattr(event, "frame") and event.frame is not None:
            try:
                from PIL import Image
            except Exception:
                pass
            else:
                image = Image.fromarray(event.frame)
                buf = BytesIO()
                image.save(buf, format="JPEG", quality=85)
                return base64.b64encode(buf.getvalue()).decode("utf-8")
        cv2 = get_cv2()
        if cv2 is not None:
            img = cls._event_bgr_image(event)
            if img is not None:
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    return base64.b64encode(buf.tobytes()).decode("utf-8")
        return None

    @staticmethod
    def _fallback_positions(count: int) -> list[dict]:
        if count == 1:
            return [{"x": 0.0, "y": DEFAULT_AGENT_Y, "z": 0.0}]
        radius = 1.25
        return [
            {"x": radius * math.cos(2 * math.pi * i / count), "y": DEFAULT_AGENT_Y, "z": radius * math.sin(2 * math.pi * i / count)}
            for i in range(count)
        ]

    @staticmethod
    def _position_center(positions: list[dict]) -> dict:
        if not positions:
            return {"x": 0.0, "y": DEFAULT_AGENT_Y, "z": 0.0}
        return {
            "x": sum(pos["x"] for pos in positions) / len(positions),
            "y": sum(pos.get("y", DEFAULT_AGENT_Y) for pos in positions) / len(positions),
            "z": sum(pos["z"] for pos in positions) / len(positions),
        }

    @staticmethod
    def _dist2(a: dict, b: dict) -> float:
        dx = a["x"] - b["x"]
        dz = a["z"] - b["z"]
        return dx * dx + dz * dz

    @staticmethod
    def _yaw_to_target(pos: dict, target: dict) -> float:
        dx = target["x"] - pos["x"]
        dz = target["z"] - pos["z"]
        if abs(dx) < 1e-6 and abs(dz) < 1e-6:
            return 0.0
        return math.degrees(math.atan2(-dx, dz))


thor_instance: Optional[NativeControllerThorServer] = None


class NativeControllerReceiverHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log_event("HTTP ACCESS", f"{self.address_string()} - {fmt % args}")

    def _set_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        log_event("REMOTE IN", f"GET {self.path}")
        if parsed.path == "/":
            self._send_json(200, {"status": "ok", "service": "ai2thor_receiver_server", "robots": len(thor_instance.robots) if thor_instance else 0})
        elif parsed.path == "/robots":
            if not self._controller_ready():
                return
            self._send_json(200, {"robots": [robot.to_dict() for robot in thor_instance.robots]})
        elif parsed.path == "/state":
            if not self._controller_ready():
                return
            robot_ref = self._query_value(query, "robot_id")
            render_image = self._query_bool(query, "render_image")
            try:
                self._send_json(200, thor_instance.capture_state(robot_ref, render_image=render_image))
            except ValueError as exc:
                self._send_json(400, {"status": "failed", "error": str(exc)})
        else:
            self._send_json(404, {"status": "failed", "error": "not_found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        log_event("REMOTE IN", f"POST {parsed.path}")
        if parsed.path == "/execute_actions":
            self._handle_execute()
        elif parsed.path == "/observe":
            self._handle_observe()
        elif parsed.path == "/reset":
            self._handle_reset()
        else:
            self._send_json(404, {"status": "failed", "error": "not_found"})

    def _controller_ready(self) -> bool:
        if thor_instance is None or thor_instance.controller is None:
            self._send_json(503, {"status": "failed", "error": "controller not started"})
            return False
        return True

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        return json.loads(body.decode("utf-8")) if body else {}

    def _handle_execute(self):
        if not self._controller_ready():
            return
        try:
            payload = self._read_json()
        except Exception as exc:
            self._send_json(400, {"status": "failed", "error": f"invalid json: {exc}"})
            return
        task_id = payload.get("task_id", "unknown")
        default_robot_ref = payload.get("robot_id", payload.get("robot", payload.get("agent_id")))
        actions = payload.get("actions", [])
        render_image = bool(payload.get("render_image", False))
        stop_on_failure = bool(payload.get("stop_on_failure", True))
        if not actions:
            self._send_json(400, {"status": "failed", "error": "empty actions"})
            return
        log_event("REMOTE IN", f"POST /execute_actions task_id={task_id}, default_robot={default_robot_ref}, actions={len(actions)}, render_image={render_image}", blank_before=True)
        for action in actions:
            log_event("REMOTE IN", f"action {json.dumps(action, ensure_ascii=False)}")
        result = thor_instance.execute_batch(actions, default_robot_ref=default_robot_ref, render_image=render_image, stop_on_failure=stop_on_failure)
        result["task_id"] = task_id
        n_ok = sum(1 for item in result["results"] if item.get("success"))
        log_event("REMOTE OUT", f"POST /execute_actions status={result['status']} ({n_ok}/{len(result['results'])} ok)")
        for item in result["results"]:
            if not item.get("success"):
                log_event(
                    "REMOTE OUT",
                    "failed action "
                    f"index={item.get('index')} robot_id={item.get('robot_id')} "
                    f"action={item.get('action')} error={item.get('error')}",
                )
        self._send_json(200, result)

    def _handle_observe(self):
        if not self._controller_ready():
            return
        try:
            payload = self._read_json()
        except Exception as exc:
            self._send_json(400, {"status": "failed", "error": f"invalid json: {exc}"})
            return
        robot_ref = payload.get("robot_id", payload.get("robot", payload.get("agent_id")))
        render_image = bool(payload.get("render_image", True))
        try:
            self._send_json(200, thor_instance.observe(robot_ref, render_image=render_image))
        except ValueError as exc:
            self._send_json(400, {"status": "failed", "error": str(exc)})

    def _handle_reset(self):
        if not self._controller_ready():
            return
        try:
            payload = self._read_json()
        except Exception:
            payload = {}
        scene = payload.get("scene", thor_instance.scene)
        robot_count = payload.get("robots", payload.get("robot_count"))
        try:
            thor_instance.reset(scene=scene, robot_count=robot_count)
            self._send_json(200, {"status": "success", "scene": thor_instance.scene, "robots": len(thor_instance.robots), "state": thor_instance.capture_state()})
        except Exception as exc:
            traceback.print_exc()
            self._send_json(500, {"status": "failed", "error": str(exc)})

    @staticmethod
    def _query_value(query: dict, key: str):
        values = query.get(key)
        return values[0] if values else None

    @staticmethod
    def _query_bool(query: dict, key: str) -> bool:
        value = NativeControllerReceiverHandler._query_value(query, key)
        if value is None:
            return False
        return value.lower() in {"1", "true", "yes", "on"}

    def _send_json(self, status_code: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status_code)
        self._set_cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()


def main():
    global thor_instance
    parser = argparse.ArgumentParser(description="AI2-THOR Native Controller Multi-Agent Receiver HTTP Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=19000)
    parser.add_argument("--scene", default="FloorPlan1")
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", action="store_false", dest="headless")
    parser.add_argument("--width", type=int, default=600)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--show", action="store_true", default=True)
    parser.add_argument("--no-show", action="store_false", dest="show")
    parser.add_argument("--display-fps", type=float, default=5.0)
    parser.add_argument("--robot0-dx", type=float, default=0.0)
    parser.add_argument("--robot0-dz", type=float, default=0.0)
    parser.add_argument("--robot0-left", type=float, default=0.0)
    parser.add_argument("--robot0-right", type=float, default=0.0)
    parser.add_argument("--robot0-dyaw", type=float, default=0.0)
    parser.add_argument("--robot0-at-fridge", action="store_true")
    parser.add_argument("--robot0-fridge-distance", type=float, default=1.0)
    args = parser.parse_args()

    thor_instance = NativeControllerThorServer(
        scene=args.scene,
        robot_count=args.robots,
        headless=args.headless,
        width=args.width,
        height=args.height,
        robot0_dx=args.robot0_dx,
        robot0_dz=args.robot0_dz,
        robot0_left=args.robot0_left,
        robot0_right=args.robot0_right,
        robot0_dyaw=args.robot0_dyaw,
        robot0_at_fridge=args.robot0_at_fridge,
        robot0_fridge_distance=args.robot0_fridge_distance,
    )
    try:
        thor_instance.start()
    except Exception as exc:
        log_event("SERVER ERROR", f"Failed to start native Controller multi-agent AI2-THOR: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    server = ThreadingHTTPServer((args.host, args.port), NativeControllerReceiverHandler)
    log_event("SERVER OUT", f"http://{args.host}:{args.port}", blank_before=True)
    log_event("SERVER OUT", "mode: native_standard_controller_shared_unity")
    log_event("SERVER OUT", "GET  /robots")
    log_event("SERVER OUT", "GET  /state")
    log_event("SERVER OUT", "POST /observe")
    log_event("SERVER OUT", "POST /execute_actions")
    log_event("SERVER OUT", "POST /reset")
    try:
        if args.show:
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            gui_started = thor_instance.display_loop(fps=args.display_fps)
            if not gui_started:
                log_event("SERVER OUT", "Running in --no-show fallback mode. Press Ctrl+C to stop.")
                while True:
                    time.sleep(1)
        else:
            server.serve_forever()
    except KeyboardInterrupt:
        log_event("SERVER IN", "KeyboardInterrupt: shutting down", blank_before=True)
    finally:
        server.shutdown()
        server.server_close()
        thor_instance.stop()


if __name__ == "__main__":
    main()
