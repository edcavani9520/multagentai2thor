"""
Base Agent Interface
=====================
Defines the contract for AI2-THOR agents.

Your colleague's agent should implement:
    def agent_fn(observation: dict) -> dict

Where:
    observation = {
        "agent_name": str,
        "agent_id": int,
        "image": np.ndarray (H, W, 3 RGB),
        "objects": list of object metadata,
        "camera_position": {"x", "y", "z"},
        "agent_position": {"x", "y", "z"},
        "metadata": full AI2-THOR metadata,
    }

    action = {
        "action": str,           # AI2-THOR action name
        # plus action-specific params like:
        # "objectId": str,
        # "position": dict,
        # "rotation": dict,
        ...
    }
"""

from typing import Callable, Dict, Any

# Type aliases for clarity
Observation = Dict[str, Any]
AI2THORAction = Dict[str, Any]


class AgentInterface:
    """
    Reference interface. Agents don't need to inherit this class;
    they just need to be a callable with the right signature.
    """

    @staticmethod
    def agent_fn(observation: Observation) -> AI2THORAction:
        """
        Take an observation, return an AI2-THOR action.

        Override this with your agent's logic.
        """
        raise NotImplementedError
