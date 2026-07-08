"""
Dummy Agent - placeholder for development/testing.
Does nothing (stays in place) and just reports what it sees.
"""

import json
from pathlib import Path
from .base_agent import Observation, AI2THORAction


def make_dummy_agent(name: str, log_dir: str = "output") -> callable:
    """
    Create a dummy agent that:
    - Does NOT move (always returns "Pass" / no-op)
    - Logs what it sees
    - Reports visible objects (proxy "other robot")

    This is a placeholder until the real agent is ready.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}_observations.jsonl"

    def agent_fn(obs: Observation) -> AI2THORAction:
        # ---- Report what the agent sees ----
        visible_objects = []
        for obj in obs.get("objects", []):
            try:
                visible_objects.append({
                    "type": obj["objectType"],
                    "id": obj.get("objectId", ""),
                    "distance": obj.get("distance", -1),
                    "position": obj.get("position", {}),
                })
            except (KeyError, TypeError):
                pass

        # Rank by distance
        visible_objects.sort(key=lambda o: o["distance"])

        report = {
            "agent": obs.get("agent_name"),
            "step": 0,  # caller should overwrite
            "position": obs.get("agent_position"),
            "num_visible_objects": len(visible_objects),
            "closest_objects": visible_objects[:10],
            "action_taken": "Pass",
        }

        # Log to file
        with open(log_file, "a") as f:
            f.write(json.dumps(report) + "\n")

        print(f"  [{obs.get('agent_name')}] sees {len(visible_objects)} objects. "
              f"Closest: {visible_objects[0]['type'] if visible_objects else 'none'} "
              f"at {visible_objects[0]['distance']:.2f}m" if visible_objects else "")

        # ---- Return no-op (stay in place) ----
        return {"action": "Pass"}

    return agent_fn


def make_logging_agent(name: str, log_dir: str = "output") -> callable:
    """
    Slightly more useful demo agent: logs its view.
    Same as dummy but with richer logging.
    """
    return make_dummy_agent(name, log_dir)
