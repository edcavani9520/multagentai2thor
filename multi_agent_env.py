"""
Multi-Agent Environment for AI2-THOR
=====================================
Central coordinator that manages multiple agents in a shared AI2-THOR scene.

Architecture:
    MultiAgentEnv (coordinator)
        ├── Agent 1 (obs → action)
        ├── Agent 2 (obs → action)
        ├── Agent N (obs → action)
        └── AI2-THOR Controller (single, shared)

Each agent receives observations and outputs actions.
The coordinator steps agents round-robin and executes their actions.
"""

import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Callable

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering


class AgentState:
    """Represents a single agent's state in the environment."""

    def __init__(self, agent_id: int, name: str, position: dict, rotation: dict, horizon: float = 0.0):
        self.agent_id = agent_id
        self.name = name
        self.position = position       # {"x": ..., "y": ..., "z": ...}
        self.rotation = rotation       # {"x": ..., "y": ..., "z": ...}
        self.horizon = horizon
        self.last_event = None         # Most recent event from this agent's viewpoint
        self.action_history = []       # List of actions taken
        self.task = ""                 # Current task assignment

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "position": self.position,
            "rotation": self.rotation,
            "task": self.task,
        }

    def __repr__(self):
        return f"Agent({self.name}, pos=({self.position['x']:.2f}, {self.position['z']:.2f}))"


class MultiAgentEnv:
    """
    Multi-agent coordinator for AI2-THOR.

    Uses a single Controller; agents' viewpoints are rendered by
    teleporting the camera. Each agent gets round-robin execution.

    Ready to accept real agent functions (obs → action).
    """

    def __init__(self, scene: str = "FloorPlan1", headless: bool = True,
                 save_dir: str = "output", width: int = 600, height: int = 600):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Start AI2-THOR
        kwargs = dict(scene=scene, width=width, height=height)
        if headless:
            kwargs["platform"] = CloudRendering
        self.controller = Controller(**kwargs)

        self.scene = scene
        self.agents: List[AgentState] = []
        self._step_count = 0

        # Metadata about the scene
        self._scene_bounds = self.controller.last_event.metadata.get("sceneBounds", {})

        print(f"[MultiAgentEnv] Scene '{scene}' loaded")
        print(f"[MultiAgentEnv] Scene bounds: {self._scene_bounds}")

    def add_agent(self, name: str, position: dict, rotation: dict = None,
                  horizon: float = 0.0, task: str = "") -> AgentState:
        """Register a new agent at the given position."""
        if rotation is None:
            rotation = {"x": 0, "y": 0, "z": 0}

        agent_id = len(self.agents)
        agent = AgentState(agent_id, name, position, rotation, horizon)
        agent.task = task
        self.agents.append(agent)

        print(f"[MultiAgentEnv] Added {agent}")
        return agent

    def place_proxy_object(self, object_type: str, position: dict,
                           rotation: dict = None) -> bool:
        """
        Place a visible object in the scene to represent an agent's body.
        
        Uses TeleportObject to reposition an existing scene object.
        """
        if rotation is None:
            rotation = {"x": 0, "y": 0, "z": 0}

        objects = self.controller.last_event.metadata.get("objects", [])
        matching = [o for o in objects if o["objectType"] == object_type]

        if matching:
            obj_id = matching[0]["objectId"]
            event = self.controller.step("TeleportObject",
                                          objectId=obj_id,
                                          position=position,
                                          rotation=rotation)
            success = event.metadata["lastActionSuccess"]
            if success:
                print(f"[MultiAgentEnv] Placed proxy '{object_type}' at ({position['x']:.2f}, {position['z']:.2f})")
            else:
                print(f"[MultiAgentEnv] WARNING: Failed to place proxy '{object_type}'")
            return success
        else:
            print(f"[MultiAgentEnv] WARNING: Object type '{object_type}' not found in scene")
            return False

    def _teleport_camera(self, agent: AgentState) -> bool:
        """Move the camera to the agent's position and orientation."""
        event = self.controller.step("TeleportFull",
                                      position=agent.position,
                                      rotation=agent.rotation,
                                      horizon=agent.horizon,
                                      standing=True)
        success = event.metadata["lastActionSuccess"]
        if not success:
            err = event.metadata.get("errorMessage", "")
            if "Collided" in err:
                print(f"  [Teleport] WARNING: {agent.name} collided at {agent.position}")
            else:
                print(f"  [Teleport] WARNING: {agent.name} teleport failed: {err[:80]}")
            return False
        agent.last_event = event
        return success

    def observe_agent(self, agent: AgentState) -> dict:
        """
        Get the observation from this agent's perspective.
        
        Returns a dict with keys:
            - image: np.ndarray (H, W, 3) RGB
            - objects: list of visible objects
            - metadata: full AI2-THOR metadata
        """
        if not self._teleport_camera(agent):
            print(f"[MultiAgentEnv] WARNING: Teleport failed for {agent.name}")
            return {}

        event = agent.last_event
        img = event.cv2img.copy() if event.cv2img is not None else None

        observation = {
            "agent_name": agent.name,
            "agent_id": agent.agent_id,
            "image": img,
            "objects": event.metadata.get("objects", []),
            "camera_position": event.metadata.get("cameraPosition", {}),
            "agent_position": agent.position,
            "metadata": event.metadata,
        }
        return observation

    def _visible_objects_from_view(self, event) -> List[dict]:
        """Extract visible objects from an event's metadata."""
        objects = event.metadata.get("objects", [])
        # In AI2-THOR, all scene objects are returned in metadata,
        # but we can filter by visibility if available
        visible = []
        for obj in objects:
            try:
                visible.append({
                    "name": obj["objectType"],
                    "objectId": obj.get("objectId", ""),
                    "distance": obj.get("distance", -1),
                    "position": obj.get("position", {}),
                })
            except (KeyError, TypeError):
                pass
        return visible

    def step_agent(self, agent: AgentState, action: dict) -> dict:
        """
        Execute one action for an agent and return the result.
        
        Args:
            agent: The agent to step
            action: AI2-THOR action dict, e.g. {"action": "MoveAhead"}
            
        Returns:
            Result event metadata
        """
        self._teleport_camera(agent)  # Move to agent's viewpoint first
        event = self.controller.step(action)
        agent.last_event = event
        agent.action_history.append(action)

        self._step_count += 1
        return event.metadata

    def run_agents_round_robin(self, agent_fns: Dict[int, Callable],
                               max_steps: int = 100, save_images: bool = True):
        """
        Main loop: run agents round-robin.
        
        Args:
            agent_fns: dict {agent_id: function(observation) -> action}
                       function takes observation dict, returns AI2-THOR action dict
            max_steps: total steps across all agents
            save_images: save observations to disk
        """
        print(f"\n{'='*60}")
        print(f"[Run] Starting round-robin multi-agent loop")
        print(f"[Run] Agents: {[a.name for a in self.agents]}")
        print(f"[Run] Max steps: {max_steps}")
        print(f"{'='*60}\n")

        for step in range(max_steps):
            for agent in self.agents:
                if agent.agent_id not in agent_fns:
                    continue

                # 1. Get observation from agent's viewpoint
                obs = self.observe_agent(agent)
                if not obs:
                    print(f"[Run] WARNING: Empty observation for {agent.name}, skipping")
                    continue

                # 2. Call the agent's function
                agent_fn = agent_fns[agent.agent_id]
                try:
                    action = agent_fn(obs)
                except Exception as e:
                    print(f"[Run] ERROR: Agent {agent.name} function failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

                # 3. Execute the action
                result = self.step_agent(agent, action)
                success = result.get("lastActionSuccess", False)
                print(f"[Step {step}] {agent.name} → {action.get('action', '?')} "
                      f"[{'OK' if success else 'FAIL'}]")

                # 4. Save image
                if save_images and obs.get("image") is not None:
                    img_path = self.save_dir / f"step_{step:04d}_{agent.name}.png"
                    import cv2
                    cv2.imwrite(str(img_path), obs["image"][:, :, ::-1])  # RGB -> BGR

        print(f"\n{'='*60}")
        print(f"[Run] Completed {max_steps} steps")
        print(f"{'='*60}")

    def get_state(self) -> dict:
        """Get full environment state."""
        return {
            "scene": self.scene,
            "step": self._step_count,
            "agents": [a.to_dict() for a in self.agents],
            "controller_state": {
                "scene_name": self.controller.last_event.metadata.get("sceneName", ""),
            }
        }

    def close(self):
        """Shutdown the environment."""
        self.controller.stop()
        print("[MultiAgentEnv] Environment closed")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
