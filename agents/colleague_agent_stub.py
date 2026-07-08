"""
Colleague Agent Stub
=====================
Placeholder for the colleague's real agent.

When their agent is ready, replace this file with the actual
agent function that takes (observation) → (action).

Usage in the coordinator:
    from agents.colleague_agent_stub import colleague_agent

    env.run_agents_round_robin({
        0: colleague_agent,  # Agent 0 uses colleague's agent
        1: make_dummy_agent("robot_B"),
    })
"""

from .base_agent import Observation, AI2THORAction


def colleague_agent(obs: Observation) -> AI2THORAction:
    """
    TODO: Replace this with the actual agent implementation.

    The agent receives:
        obs = {
            "agent_name": str,
            "agent_id": int,
            "image": np.ndarray (H, W, 3),    # RGB camera view
            "objects": [                        # scene objects
                {"objectType": ..., "objectId": ..., "distance": ..., ...},
            ],
            "camera_position": {"x", "y", "z"},
            "agent_position": {"x", "y", "z"},
            "metadata": {...},                  # full AI2-THOR metadata
        }

    Must return:
        action = {
            "action": "PickupObject",           # AI2-THOR action name
            "objectId": "...",                  # action-specific params
        }
    """

    # ---- Placeholder: just observe, don't move ----
    print(f"  [ColleagueAgent] Observing from {obs.get('agent_position')}")
    print(f"  [ColleagueAgent] Sees {len(obs.get('objects', []))} objects")

    # --- Example of what a real action would look like: ---
    # return {"action": "PickupObject", "objectId": "Apple|...|..."}
    # return {"action": "RotateRight"}
    # return {"action": "MoveAhead"}

    return {"action": "Pass"}
