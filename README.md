# multagentai2thor

当前主线是“EmbodiedGPT relay 闭环任务服务 + 原生 AI2-THOR 多 robot receiver”。

```text
client / coordinator -> relay_task_server.py -> ai2thor_receiver_server.py -> AI2-THOR
```

## 启动

先启动共享场景 receiver：

```bash
python ai2thor_receiver_server.py \
  --scene FloorPlan1 --robots 2 --port 19000 --no-show
```

再启动 relay 闭环任务服务：

```bash
bash run_relay_task_server.sh \
  --model-path models/Qwen3.5-4B \
  --receiver-url http://127.0.0.1:19000 \
  --port 18080 --device cuda --dtype float16
```

任务 API 见 [RELAY_TASK_SERVICE.md](RELAY_TASK_SERVICE.md)，relay 闭环设计见 [relay_closed_loop_design_cn.md](relay_closed_loop_design_cn.md)。AI2-THOR HTTP 接口和项目架构见 [PROJECT_ARCHITECTURE_DRAFT.md](PROJECT_ARCHITECTURE_DRAFT.md)。

## 文件状态

完整文件分层与保留状态见 [PROJECT_FILE_MAP.md](PROJECT_FILE_MAP.md)。

当前仓库已移除单步 `qwen_action_service.py`、本地 Qwen loop、早期 `MultiAgentEnv`、custom agent 和 OpenVLA 实验代码，只保留 Qwen3.5-4B 的 relay 闭环路线。
