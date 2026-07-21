# 项目文件地图

当前项目只保留“EmbodiedGPT relay 闭环任务服务 + 原生 AI2-THOR receiver”这一条服务主线。单步 Qwen service、早期 receiver、`MultiAgentEnv` 原型、custom agent 与 OpenVLA 实验代码均已移除，避免多个入口和不同仿真语义造成混淆。

## 当前主线

```text
client / coordinator
        |
        | POST /execute_task
        v
relay_task_server.py (:18080)
        |
        | task intent, relay, grounding, POST /observe and /execute_actions
        v
ai2thor_receiver_server.py (:19000)
        |
        v
AI2-THOR shared Unity scene, Controller(agentCount=N)
```

| 路径 | 状态 | 作用 |
| --- | --- | --- |
| `ai2thor_receiver_server.py` | 主线 | 唯一保留的仿真 HTTP receiver。维护共享 Unity 场景，提供 `/robots`、`/state`、`/observe`、`/execute_actions`、`/reachable_positions`、`/goto`、`/reset`。 |
| `relay_task_server.py` | 主线 | 常驻 HTTP 任务服务。复用 EmbodiedGPT 的 relay 和闭环重规划代码执行一个完整自然语言任务。 |
| `EmbodiedGPT_Pytorch/demo/auto_scene_actions.py` | 主线引擎 | 任务意图、语义规划、grounding、动作执行反馈合并和闭环控制器。 |
| `EmbodiedGPT_Pytorch/demo/relay_agent.py` | 主线引擎 | 受控工具调用的多 robot executor 选择与失败报告。 |
| `EmbodiedGPT_Pytorch/demo/qwen35_backend.py` | 主线引擎 | Qwen3.5 模型加载和图像/工具调用推理。 |
| `models/Qwen3.5-4B/` | 主线资源 | post-trained Qwen 模型权重、processor 和模型说明。 |
| `RELAY_TASK_SERVICE.md` | 主线文档 | 任务服务的启动、HTTP 契约和运行方式。 |
| `relay_closed_loop_design_cn.md` | 主线设计 | relay、硬验证、executor 选择和闭环重规划机制。 |
| `PROJECT_ARCHITECTURE_DRAFT.md` | 主线文档 | 项目架构、AI2-THOR 接口和 Qwen 接入说明。 |
| `README.md` | 入口文档 | 当前服务启动命令和文档索引。 |
| `tests/test_relay_task_server.py` | 测试 | 不加载模型地验证 HTTP 包装层的 relay/closed-loop 参数映射。 |

## 生成文件

`__pycache__/` 与 `output/` 是 Python 或运行时生成内容，不属于源代码结构。它们可以忽略；需要彻底清空运行产物时再单独删除。
