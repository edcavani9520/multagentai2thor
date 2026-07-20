# Relay Task Service

`relay_task_server.py` is the task-level entry point for this repository. It
wraps the existing `EmbodiedGPT_Pytorch` relay closed-loop runtime and sends
all simulator actions to `ai2thor_receiver_server.py`.

```text
client
  | POST /execute_task
  v
relay_task_server.py
  | task intent -> relay tool calls -> executor-view plan -> grounding
  v
ai2thor_receiver_server.py
  | /observe, /state, /execute_actions
  v
AI2-THOR shared multi-robot scene
```

The service always enables `--relay-mode --closed-loop-replan`. It does not
use the removed one-step Qwen action service.

When the primary robot cannot execute the current step, the runtime first
precollects observations from every known robot. The relay model receives all
first-person images, relevant metadata, and the complete candidate validation
table in its first decision input; it then selects an executor or reports a
validated failure without polling robots one by one.

## Start

Start the receiver first:

```bash
python ai2thor_receiver_server.py \
  --scene FloorPlan1 --robots 2 --port 19000 --no-show
```

Then start the relay task service. One process owns one Qwen model instance.

```bash
bash run_relay_task_server.sh \
  --receiver-url http://127.0.0.1:19000 \
  --model-path models/Qwen3.5-4B \
  --port 18080 \
  --device cuda \
  --device-map auto \
  --dtype float16 \
  --max-new-tokens 128
```

Use `CUDA_VISIBLE_DEVICES=0` or `CUDA_VISIBLE_DEVICES=1` before this command
when the model must be pinned to one physical GPU.

The launcher clears ROS `PYTHONPATH` and gives Conda's `libstdc++` precedence.
This is required on this workstation because the ROS library path otherwise
breaks the Transformers import chain through `sklearn` and `pyarrow`.

## HTTP API

`GET /health` returns service configuration and whether the model has been
loaded. The model is loaded lazily by the first task.

The service saves each semantic-planning model response as
`output/relay_tasks/<task_id>_qwen_raw.txt` before validation and grounding.

`POST /execute_task` accepts:

```json
{
  "task_id": "bread-001",
  "task": "pick up the bread",
  "primary_robot_id": 0,
  "known_robot_ids": [0, 1],
  "dry_run": false,
  "max_replan_steps": 10,
  "relay_agent_max_turns": 8,
  "max_actions": 8,
  "relay_strategy": "agent"
}
```

`task` may also be supplied as `instruction` or `prompt`. `relay_strategy`
can be `agent` for Qwen tool-calling coordination or `rules` for the
deterministic candidate selector. `dry_run=true` performs observation,
planning, grounding, and relay selection without changing the scene.

Example:

```bash
curl -X POST http://127.0.0.1:18080/execute_task \
  -H 'Content-Type: application/json' \
  -d '{
    "task_id": "bread-001",
    "task": "pick up the bread",
    "primary_robot_id": 0,
    "known_robot_ids": [0, 1],
    "dry_run": false
  }'
```

The response contains `result.closed_loop_trace`, each grounded action payload,
executor-selection evidence, and `closed_loop_result`. A logical inability to
complete the task returns `status: "needs_upstream_planning"` with a structured
reason; HTTP `4xx` and `5xx` are reserved for malformed requests and runtime
failures.

For the complete relay decision model, see `relay_closed_loop_design_cn.md`.
