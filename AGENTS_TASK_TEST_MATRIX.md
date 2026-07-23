# Agents Task Test Matrix

This table is the smoke/regression checklist for the EMAS agents runtime. Use the CLI from `EmbodiedGPT_Pytorch` unless the test explicitly targets an HTTP endpoint.

## Receiver Health

| Case | Command / Endpoint | Expected result |
| --- | --- | --- |
| Receiver health | `GET /health` | `status=ok`, `service=ai2thor_receiver_server`. If 404, the running receiver process is old and must be restarted. |
| Scene state | `GET /state` | Returns `sceneName`, `robots`, robot `position`, `rotation`, `inventory` / `held_object`. |
| Reachable points | `GET /reachable_positions?robot_id=0` | Returns `status=success` and non-empty `positions`. |
| Action dispatch | `POST /execute_actions` with `Pass` and top-level `stop_on_failure=false` | Returns `status=success` or structured failure JSON. |
| Goto dry-run | `POST /goto` with `execute=false` | Returns path/actions on success, or structured failed JSON. |

## CLI Navigation-Only Tasks

| Task | Expected route | Expected JSON signal |
| --- | --- | --- |
| `go to fridge.` | `/goto`, no Qwen, no relay | `task_intent_source=navigation_goto_intent`, `closed_loop_result.strategy=goto` |
| `go to cabinet.` | `/goto`, no Qwen, no relay | Same as above |
| `go to counter.` | `/goto`, no Qwen, no relay | Same as above; target usually maps to `CounterTop` |
| `navigate to tomato.` | `/goto`, no object interaction | Same as above |
| `go to nonexistent_object.` | `/goto` structured failure | `closed_loop_result.status=needs_upstream_planning`, meaningful `failure_code` |

## Primary Fast Path Tasks

Run these only when the primary robot currently has the required visible/held state.

| Task | Expected route | Expected behavior |
| --- | --- | --- |
| `pick up the bread.` | Primary fast path | No `[relay] primary cannot execute` stderr block. Payload includes top-level `stop_on_failure=false`. |
| `open the fridge.` | Primary fast path | Primary generates executor-view semantic plan and sends grounded actions. |
| `put the mug on the counter.` | Primary fast path if primary holds mug and sees counter | No relay handoff; structured success or action-level failure. |

## Relay Handoff Tasks

Run these when primary cannot execute but another known robot can.

| Task / Setup | Expected route | Expected behavior |
| --- | --- | --- |
| robot1 cannot see Bread, robot0 can: `pick up the bread.` with `--primary-robot-id 1` | Relay handoff | stderr prints primary inability, coordination result, candidate summary. |
| Bread is held by robot0, task sent with primary robot1: `put the bread on the counter.` | Relay handoff to holder if valid | Relay selects a validated executor or reports structured failure. |
| All robots cannot see target: `pick up the moon.` | Relay failure | No crash; `closed_loop_result.status=needs_upstream_planning`. |

## Multi-Step Closed Loop

| Task | Expected route | Expected behavior |
| --- | --- | --- |
| `put the tomato on the counter.` | task intent -> per-step primary/relay routing | `closed_loop_trace`, `step_payloads`, and `step_execute_response_summaries` are populated. |
| `pick up the bread and put it on the counter.` | multiple intent steps | Already-satisfied, fast path, and relay decisions are recorded per step. |
| `open the fridge and put the tomato in the fridge.` | multiple intent steps | Failure stops with explicit `needs_upstream_planning` reason. |

## Recommended Smoke Command

```bash
cd /225010231/mwl/EMAS/agents
python scripts/agents_smoke_test.py \
  --execute-actions-url http://10.20.18.3:19000/execute_actions \
  --robot-id 0 \
  --goto-target Fridge
```
