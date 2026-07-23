# EMAS Agents Runtime

当前主线是“EmbodiedGPT relay 闭环任务服务 + 原生 AI2-THOR 多 robot receiver”。

```text
client / coordinator -> relay_task_server.py -> ai2thor_receiver_server.py -> AI2-THOR
```

## 当前稳定入口

你日常测试 agents 模块时，优先使用 EmbodiedGPT runtime 的 CLI：

```bash
cd /225010231/mwl/EMAS/agents/EmbodiedGPT_Pytorch

SEND_ACTIONS_URL="http://10.20.18.3:19000/execute_actions" \
./auto_scene_actions.sh \
  --task "go to fridge." \
  --primary-robot-id 0 \
  --relay-mode \
  --closed-loop-replan \
  --print-raw-output
```

行为约定：

- `go to ...` / `navigate to ...` / `move to ...` / `walk to ...` 这类 navigation-only 任务直接调用 receiver 的 `/goto`，不加载 Qwen，不进入 relay。
- `PickupObject` / `PutObject` / `OpenObject` / `CloseObject` 等交互任务走 EmbodiedGPT semantic planning + object grounding + closed-loop relay。
- primary robot 通过硬验证即可执行时不进入 relay。
- primary robot 不能执行时，relay 会打印 primary 失败原因、候选摘要、协调成功或失败原因。
- 所有普通 `/execute_actions` payload 都包含顶层 `"stop_on_failure": false`。
- `/goto` 失败、relay 失败和执行失败都应返回结构化 JSON，不能只抛截断异常。

## 启动

先启动共享场景 receiver：

```bash
python ai2thor_receiver_server.py \
  --scene FloorPlan1 --robots 2 --port 19000 --no-show
```

receiver 对外提供：

```text
GET  /health
GET  /robots
GET  /state
GET  /reachable_positions?robot_id=0
POST /observe
POST /execute_actions
POST /goto
POST /reset
```

再启动 relay 闭环任务服务：

```bash
bash run_relay_task_server.sh \
  --model-path models/Qwen3.5-4B \
  --receiver-url http://127.0.0.1:19000 \
  --port 18080 --device cuda --dtype float16
```

任务服务对外提供：

```text
POST /execute_task
```

这个 HTTP 入口用于后续 planning/memory 联调。planning 模块不需要知道 `auto_scene_actions.py` 的内部细节，只需要把 task、primary robot、known robots 等字段传给 `/execute_task`。

## Smoke Test

没有 `curl` 时可直接用 Python 进行健康检查：

```bash
cd /225010231/mwl/EMAS/agents

python scripts/agents_smoke_test.py \
  --execute-actions-url http://10.20.18.3:19000/execute_actions \
  --robot-id 0 \
  --goto-target Fridge \
  --skip-cli
```

如果还想检查 CLI wrapper 是否仍指向当前 EMAS 目录，并验证 navigation-only dry-run 路径：

```bash
python scripts/agents_smoke_test.py \
  --execute-actions-url http://10.20.18.3:19000/execute_actions \
  --robot-id 0 \
  --goto-target Fridge
```

关键验收点：

- `wrapper_path` 必须显示 OK，表示没有跳回旧的 `/225010231/mwl/Linhao/EmbodiedGPT_Pytorch`。
- `/health`、`/state`、`/reachable_positions`、`/execute_actions` 都应 OK。
- `/goto` dry-run 可以成功，也可以返回结构化 failed，例如目标不存在；但不能是连接错误或非 JSON 响应。

## 回归测试

```bash
cd /225010231/mwl/EMAS/agents
python -m py_compile ai2thor_receiver_server.py relay_task_server.py
python -m unittest tests/test_ai2thor_navigation_planner.py tests/test_relay_task_server.py

cd /225010231/mwl/EMAS/agents/EmbodiedGPT_Pytorch
python -m py_compile demo/auto_scene_actions.py demo/relay_agent.py demo/plan_media.py
python -m unittest tests/test_plan_media.py
python -m unittest tests/test_relay_agent.py
python -m unittest tests/test_relay_agent_integration.py tests/test_relay_agent_closed_loop.py
```

任务测试表见 [AGENTS_TASK_TEST_MATRIX.md](AGENTS_TASK_TEST_MATRIX.md)。

任务 API 见 [RELAY_TASK_SERVICE.md](RELAY_TASK_SERVICE.md)，relay 闭环设计见 [relay_closed_loop_design_cn.md](relay_closed_loop_design_cn.md)。AI2-THOR HTTP 接口和项目架构见 [PROJECT_ARCHITECTURE_DRAFT.md](PROJECT_ARCHITECTURE_DRAFT.md)。

## 文件状态

完整文件分层与保留状态见 [PROJECT_FILE_MAP.md](PROJECT_FILE_MAP.md)。

当前仓库已移除单步 `qwen_action_service.py`、本地 Qwen loop、早期 `MultiAgentEnv`、custom agent 和 OpenVLA 实验代码，只保留 Qwen3.5-4B 的 relay 闭环路线。
