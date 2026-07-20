# multagentai2thor 与 Qwen3.5-4B 接入说明

本文档说明当前 `multagentai2thor` 仓库在整个多机器人协作项目中的位置，以及 AI2-THOR 仿真端、HTTP receiver、Python agent 接口和 post-trained `Qwen3.5-4B` 模型之间如何连接。

桌面 `6.29ppt.pdf` 中关于 Scene Graph、任务图、候选方案评估和协商机制的内容可作为上层系统设计参考；本文不展开论文式高层规划，重点记录本仓库实际提供的接口、输入输出对象、运行方式和模型接入方式。

---

## 1. 项目定位

`multagentai2thor` 当前主要承担“仿真执行端”和“模型接入验证端”两类职责。

它负责：

- 启动 AI2-THOR 场景；
- 在同一个 Unity 世界中创建多个 robot / agent slot；
- 接收上层服务端或本地模型产生的 AI2-THOR action；
- 按 `robot_id` 把动作路由到对应 robot；
- 返回 observation、scene state、object metadata、inventory、执行成功/失败原因；
- 支持本地视觉语言模型，例如 post-trained `Qwen3.5-4B`，直接根据第一视角图像和任务指令生成动作。

它不负责：

- 长期记忆数据库；
- 完整自然语言任务图规划；
- 多 robot 全局协商策略；
- 场景图世界模型训练；
- 任务完成度的长期判断。

这些职责应该由模型服务端、中转/协商 agent 和记忆系统承担。本仓库提供可靠的环境闭环：观察、执行、反馈。

---

## 2. 当前推荐架构

```text
Natural Language Instruction
        |
        v
Model Server / Middleware Agent
  - task decomposition
  - robot assignment
  - failure handling
  - optional memory query
        |
        | HTTP JSON
        v
multagentai2thor
  - relay_task_server.py: task -> relay closed-loop execution
  - ai2thor_receiver_server.py: action -> Unity execution
        |
        v
AI2-THOR Controller(agentCount=N)
  - shared Unity scene
  - agentId = robot_id
  - object metadata / RGB image / inventory
```

当前正式实验建议使用：

```text
ai2thor_receiver_server.py
```

原因是它使用 AI2-THOR 标准 `Controller(agentCount=N)` 和 `controller.step(..., agentId=robot_id)`，可以在一个共享 Unity 世界中维护多个 robot，并避免旧版单 camera teleport 方案的 inventory 串台问题。

---

## 3. 代码结构

| 文件 / 目录 | 作用 |
| --- | --- |
| `ai2thor_receiver_server.py` | 推荐主线。原生多 agent AI2-THOR HTTP receiver |
| `relay_task_server.py` | 无界面任务服务。复用 EmbodiedGPT 的 relay、规划、grounding 和执行反馈闭环 |
| `EmbodiedGPT_Pytorch/demo/` | Qwen backend、任务意图、语义规划、中转智能体和闭环控制器实现 |
| `RELAY_TASK_SERVICE.md` | relay 任务服务启动和 HTTP 契约 |
| `relay_closed_loop_design_cn.md` | 多机器人 relay 闭环设计说明 |
| `models/Qwen3.5-4B/` | 已解压的 post-trained Qwen3.5-4B 模型权重和配置 |
| `PROJECT_FILE_MAP.md` | 当前项目文件分层、入口和保留状态 |

---

## 4. AI2-THOR Receiver

### 4.1 启动方式

推荐启动：

```bash
python ai2thor_receiver_server.py \
  --scene FloorPlan1 \
  --robots 2 \
  --port 19000 \
  --no-show
```

如果要本地窗口显示：

```bash
python ai2thor_receiver_server.py \
  --scene FloorPlan1 \
  --robots 2 \
  --port 19000 \
  --show
```

当前环境中 OpenCV 窗口有可能卡住。服务端实验优先使用 `--no-show`，通过 `/observe` 或 `/state?render_image=1` 获取图像。

### 4.2 核心实现

`NativeControllerThorServer.start()` 中创建 AI2-THOR Controller：

```python
Controller(
    scene="FloorPlan1",
    width=600,
    height=600,
    agentCount=robot_count,
    port=0,
)
```

执行动作时：

```python
controller.step(
    action="MoveAhead",
    agentId=robot_id,
    renderImage=True,
)
```

`robot_id` 和 AI2-THOR 的 `agentId` 对应。也就是说：

- `robot_id=0` 控制 `agentId=0`；
- `robot_id=1` 控制 `agentId=1`；
- 所有 agent 在同一个 Unity 物理世界中交互。

### 4.3 RobotState

receiver 内部用 `RobotState` 记录每个 robot 的状态。HTTP 返回时常见字段如下：

```json
{
  "robot_id": 0,
  "name": "Robot0",
  "position": {"x": -1.50, "y": 0.90, "z": 2.00},
  "rotation": {"x": 0, "y": 221.2, "z": 0},
  "horizon": 0.0,
  "inventory": [],
  "held_object": null,
  "task": "Native standard Controller agent slot 0",
  "last_action": "Pass",
  "last_success": true,
  "last_error": null,
  "action_count": 0,
  "controller": "native_standard_controller"
}
```

含义：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `robot_id` | `int` | robot 编号，也是 AI2-THOR `agentId` |
| `position` | `dict` | 当前世界坐标 |
| `rotation` | `dict` | 当前朝向，常用 `rotation.y` 表示 yaw |
| `horizon` | `float` | 相机俯仰角 |
| `inventory` | `list` | 当前手持物列表 |
| `held_object` | `dict/null` | 当前手持物，通常是 `inventory[0]` |
| `last_action` | `str` | 上一步动作 |
| `last_success` | `bool` | 上一步是否成功 |
| `last_error` | `str/null` | 上一步失败原因 |

---

## 5. HTTP 接口

所有 HTTP 输入输出都是 JSON。图像使用 JPEG base64 字符串，字段名为 `image_base64`。

### 5.1 `GET /robots`

获取所有 robot 状态。

```bash
curl http://localhost:19000/robots
```

返回：

```json
{
  "robots": [
    {"robot_id": 0, "name": "Robot0", "position": {}, "inventory": []},
    {"robot_id": 1, "name": "Robot1", "position": {}, "inventory": []}
  ]
}
```

用途：

- 上层 task allocator 查询有哪些 robot；
- 检查 robot 是否手持物体；
- 判断 robot 上一步是否失败。

### 5.2 `GET /state`

获取场景状态。

```bash
curl "http://localhost:19000/state?robot_id=0&render_image=1"
```

返回核心字段：

```json
{
  "sceneName": "FloorPlan1",
  "step": 12,
  "selected_robot_id": 0,
  "controller_mode": "native_standard_controller_shared_unity",
  "agent": {
    "position": {"x": -1.5, "y": 0.9, "z": 2.0},
    "rotation": {"x": 0, "y": 221.2, "z": 0},
    "horizon": 0.0
  },
  "robots": [],
  "inventory": [],
  "objects": [],
  "num_objects": 93,
  "image_base64": "..."
}
```

`objects` 是从 AI2-THOR metadata 中整理出的轻量物体列表：

```json
{
  "id": "Bread|-00.52|+01.17|-00.03",
  "type": "Bread",
  "position": {"x": -0.52, "y": 1.17, "z": -0.03},
  "distance": 1.42,
  "pickupable": true,
  "receptacle": false,
  "openable": false,
  "isOpen": false,
  "visible": true
}
```

用途：

- 记忆系统更新 Scene Graph；
- 中转 agent 判断目标物体是否可见/可达；
- robot policy 获取 objectId。

### 5.3 `POST /observe`

获取某个 robot 的单帧观察，适合给视觉模型作为输入。

请求：

```json
{
  "robot_id": 0,
  "render_image": true
}
```

返回：

```json
{
  "status": "success",
  "robot_id": 0,
  "robot": {},
  "objects": [],
  "metadata": {},
  "image_base64": "..."
}
```

和 `/state` 的区别：

- `/observe` 更偏当前 robot 的第一视角输入；
- `/state` 更偏全局状态快照；
- 两者都可以带图像，模型输入通常优先用 `/observe`。

### 5.4 `POST /execute_actions`

执行一批 AI2-THOR 动作。

请求：

```json
{
  "task_id": "task_001",
  "robot_id": 0,
  "render_image": true,
  "stop_on_failure": true,
  "actions": [
    {"action": "MoveAhead"},
    {"action": "PutObject", "objectId": "CounterTop|-01.87|+00.95|-01.21", "forceAction": true}
  ]
}
```

字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `task_id` | `str` | 上层任务 id，会原样带回 |
| `robot_id` | `int/str` | 默认执行动作的 robot |
| `render_image` | `bool` | 是否在结果中返回图像 |
| `stop_on_failure` | `bool` | 某步失败后是否停止后续动作，默认 `true` |
| `actions` | `list[dict]` | AI2-THOR action 字典列表 |

返回：

```json
{
  "status": "failed",
  "task_id": "task_001",
  "results": [
    {
      "index": 0,
      "robot_id": 0,
      "robot_name": "Robot0",
      "action": "MoveAhead",
      "success": false,
      "error": "Wall is blocking Agent 0 from moving ...",
      "agent": {},
      "inventory": [],
      "held_object": null,
      "robot": {},
      "robot_pose": {},
      "robot_pose_changed": false,
      "interacted_objects": []
    }
  ],
  "state": {}
}
```

`status` 取值：

| 值 | 含义 |
| --- | --- |
| `success` | 所有动作成功 |
| `partial` | 部分动作成功 |
| `failed` | 没有动作成功 |

注意：

- 如果 `stop_on_failure=true`，前一个动作失败，后续动作不会执行。
- 例如 batch 里先 `MoveRight` 再 `PutObject`，如果 `MoveRight` 被挡住，`PutObject` 会被跳过。
- 服务端已经打印失败细节：`failed action index=... robot_id=... action=... error=...`。

### 5.5 `POST /reset`

重置场景或 robot 数量。

```json
{
  "scene": "FloorPlan2",
  "robots": 2
}
```

返回：

```json
{
  "status": "success",
  "scene": "FloorPlan2",
  "robots": 2,
  "state": {}
}
```

---

## 6. AI2-THOR 动作与 objectId

### 6.1 常见动作

| 类别 | 动作 | 重要参数 |
| --- | --- | --- |
| 空动作 | `Pass` | 无 |
| 导航 | `MoveAhead`、`MoveBack`、`MoveLeft`、`MoveRight` | 可选移动幅度 |
| 旋转 | `RotateLeft`、`RotateRight` | 可选旋转角度 |
| 视角 | `LookUp`、`LookDown` | 无 |
| 拾取 | `PickupObject` | `objectId` |
| 放置 | `PutObject` | receptacle 的 `objectId` |
| 丢下 | `DropHandObject` | 无 |
| 开关容器 | `OpenObject`、`CloseObject` | `objectId` |
| 切换状态 | `ToggleObjectOn`、`ToggleObjectOff` | `objectId` |
| 传送 | `TeleportFull` | `position`、`rotation`、`horizon`、`standing` |

### 6.2 objectId 的意义

AI2-THOR 的 `objectId` 是场景中某个具体物体实例的唯一标识。它通常包含物体类型和位置编码。

例如：

```json
"Bread|-00.52|+01.17|-00.03"
```

可以理解为：

```text
Bread | -00.52 | +01.17 | -00.03
类型      x        y        z
```

但不要把它只当作坐标。对 AI2-THOR 来说，它是“这一个 Bread 实例”的 id。执行动作时必须使用当前 metadata 中最新的 `objectId`。

推荐做法：

- 每次动作前先 `/observe` 或 `/state`；
- 从返回的 `objects` 列表中取目标物体的最新 `id`；
- 不长期硬编码 objectId；
- 物体移动、拾取、放置、切片或状态变化后，重新读取 metadata。

### 6.3 PutObject 的常见失败原因

`PutObject` 不只是“把东西放到某坐标”。它要求当前状态满足多个条件：

- robot 手里有东西，即 `inventory` 非空；
- 目标 `objectId` 是 receptacle，例如 `CounterTop`、`Cabinet`、`Sink`；
- 目标可达或允许 `forceAction`；
- 如果目标是 `Cabinet` 这类可打开容器，通常需要先 `OpenObject`；
- 目标附近空间可放置，且物理状态允许。

典型序列：

```json
[
  {"action": "OpenObject", "objectId": "Cabinet|-01.85|+02.02|+00.38", "forceAction": true},
  {"action": "PutObject", "objectId": "Cabinet|-01.85|+02.02|+00.38", "forceAction": true}
]
```

如果失败，应优先查看 `results[0].error`，不要只看顶层 `status=failed`。

---

## 7. Python Agent 接口

本地调试路线使用 Python callable：

```python
def agent_fn(observation: dict) -> dict:
    return {"action": "MoveAhead"}
```

输入 observation：

```python
{
    "agent_name": "RobotA",
    "agent_id": 0,
    "image": np.ndarray,       # H x W x 3 RGB
    "objects": list[dict],
    "camera_position": dict,
    "agent_position": dict,
    "metadata": dict,
}
```

输出 action：

```python
{"action": "MoveAhead"}
{"action": "PickupObject", "objectId": "Bread|-00.52|+01.17|-00.03"}
{"action": "Pass"}
```

当前模型接入由 `EmbodiedGPT_Pytorch/demo/qwen35_backend.py` 提供。它接收图像或工具调用消息，输出任务意图 tool call 或语义计划 JSON；物体实例 id 由后续 grounding 模块根据 receiver 的当前 observation 填充。

---

## 8. Qwen3.5-4B Post-trained 模型

### 8.1 模型目录

当前模型位于：

```text
models/Qwen3.5-4B
```

主要文件：

| 文件 | 作用 |
| --- | --- |
| `config.json` | 模型结构配置 |
| `preprocessor_config.json` | 图像预处理配置 |
| `tokenizer_config.json` / `tokenizer.json` | tokenizer 和 chat template 配置 |
| `chat_template.jinja` | 多模态对话模板 |
| `model.safetensors-00001-of-00002.safetensors` | 权重分片 |
| `model.safetensors-00002-of-00002.safetensors` | 权重分片 |
| `README.md` | 模型说明和部署建议 |

### 8.2 从模型配置读出的关键信息

| 项目 | 内容 |
| --- | --- |
| pipeline | `image-text-to-text` |
| architecture | `Qwen3_5ForConditionalGeneration` |
| model_type | `qwen3_5` |
| processor | `Qwen3VLProcessor` |
| image processor | `Qwen2VLImageProcessorFast` |
| 模型阶段 | post-trained |
| 模型类型 | Causal Language Model with Vision Encoder |
| text hidden size | 2560 |
| text layers | 32 |
| vision depth | 24 |
| vision patch size | 16 |
| context length | 262,144 tokens |

对本项目的意义：

- 它能同时看 AI2-THOR 第一视角图像和文本状态；
- 输出是文本，因此需要 prompt 约束为 JSON；
- 适合做 robot policy，即“当前观察下下一步做什么”；
- 不建议直接让它承担完整多机器人长期规划；
- 图像分辨率和输出 token 数会明显影响速度和显存。

### 8.3 在 relay 闭环中的输入输出

`EmbodiedGPT_Pytorch/demo/qwen35_backend.py` 的输入是 executor 的第一视角图像或工具调用 messages，加上当前单步任务意图。模型输出不是立即执行的单个 action，而是受约束的 JSON：任务意图 tool call，或不含 `objectId` 的 semantic plan。

```json
{
  "task": "pick up the bread",
  "targetObjectType": "Bread",
  "needsGrounding": true,
  "observations": [],
  "plan": [{"action": "PickupObject", "objectType": "Bread", "targetType": null}]
}
```

`auto_scene_actions.py` 再使用 receiver 当前 `objects`、affordance、inventory 和可见性进行硬验证，并将 object type grounding 为精确 `objectId`。因此模型不会直接持有或长期复用旧的 objectId。

### 8.4 Thinking mode

Qwen3.5 默认可能输出：

```text
<think>
...
</think>

{"action": "MoveAhead"}
```

relay runtime 的 Qwen backend 在 chat template 中传入 `enable_thinking=False`，要求模型输出结构化 JSON/tool call。中转智能体的多轮行为来自受控工具调用，不是暴露模型思维链。

---

## 9. Relay 闭环任务服务

当前服务化入口是 `relay_task_server.py`。它不输出单个动作，而是将完整自然语言任务交给 `EmbodiedGPT_Pytorch` 的闭环引擎：任务意图解析、primary fast path、relay 工具调用、executor 视角规划、objectId grounding、执行反馈合并和下一步骤重规划均在一次 `POST /execute_task` 中完成。

```text
Client
        |
        | POST /execute_task
        v
relay_task_server.py (:18080)
        | --relay-mode --closed-loop-replan
        v
EmbodiedGPT planner and relay agent
        |
        | POST /observe, /state, /execute_actions
        v
ai2thor_receiver_server.py (:19000)
```

每个任务请求必须包含 `task`，可选指定 `primary_robot_id`、`known_robot_ids`、`dry_run` 和重规划限制。若 primary robot 满足硬前置条件，系统直接执行；否则程序先收集所有已知 robot 的第一视角图像和相关 metadata，统一计算候选证据，再由 relay agent 选择 executor 或以 `needs_upstream_planning` 返回可解释失败。主线决策阶段不再由模型逐个轮询 robot。完整设计见 `relay_closed_loop_design_cn.md`，HTTP 契约见 `RELAY_TASK_SERVICE.md`。

模型推理仍可能较慢。一个 relay 任务会产生任务意图、relay 决策和每个 executor 步骤的多次 Qwen 调用，应让一个服务进程独占或固定使用一张有足够空闲显存的 GPU。

---

## 10. 运行与排错

### 10.1 确认 GPU

```bash
nvidia-smi
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'
```

期望输出：

```text
True 2
```

如果 GPU0 已被其他进程占用，需要先释放，否则两个 4B VL 模型实例可能 OOM。

### 10.2 无 GUI 运行

当前主线不使用 OpenCV 窗口。`relay_task_server.py` 将 receiver 返回的 observation 图像保存到 `output/relay_tasks/`，供每个 executor 步骤的 Qwen 规划和后续排错使用。

### 10.3 动作失败

AI2-THOR 动作失败是正常现象，原因包括：

- 前方被墙或物体挡住；
- 目标物体不可见；
- 目标物体太远；
- 没有手持物却执行 `PutObject`；
- 目标 receptacle 关闭；
- batch 中前一步失败，后一步被 `stop_on_failure=true` 跳过。

排查顺序：

1. 看服务端 `failed action ... error=...`；
2. 看 HTTP response 中 `results[i].error`；
3. 调 `/state?robot_id=0&render_image=1` 检查当前状态；
4. 如有 batch，确认是否被前一步失败提前停止；
5. 对物体操作，确认 `objectId` 来自最新 `objects` 列表。

### 10.4 Relay 任务返回 `needs_upstream_planning`

这表示闭环引擎无法在已有 observation 与硬验证条件下继续执行，不是 HTTP 通信异常。检查响应中的 `result.closed_loop_trace`、`relay_explanation`、`known_robot_ids` 和 `final_object_visibility_summary`，可区分目标不可见、前置状态缺失、grounding 失败或 AI2-THOR 动作失败。

---

## 11. 与上层系统的接口边界

上层系统给本仓库的输入：

- `task`；
- `primary_robot_id`；
- 可选 `known_robot_ids`、`dry_run` 和闭环限制。

本仓库给上层系统的输出：

- `closed_loop_result` 与失败原因；
- `closed_loop_trace`、executor 选择证据和每步 action payload；
- receiver 的 pose、inventory、objects 和执行结果摘要。

推荐闭环：

```text
relay_task_server.py 接收完整任务
        |
        v
任务意图、relay 选择、executor 规划和 grounding
        |
        v
ai2thor_receiver_server.py 执行动作
        |
        v
返回 success / error / state
        |
        v
闭环控制器合并反馈并进入下一意图步骤
```

---

## 12. 后续建议

短期：

- 保持 `ai2thor_receiver_server.py` 作为主线；
- 保持 `relay_task_server.py` 作为唯一任务级入口；
- 用 `dry_run=true` 检查 relay、grounding 和候选执行者选择；
- 在服务端日志中持续保留 action error 细节；
- 每次物体操作前重新 observe，使用最新 objectId。

中期：

- 把 Qwen3.5-4B 作为独立推理服务部署；
- 上层服务端通过 OpenAI-compatible API 调 Qwen；
- 将 AI2-THOR 动作封装成 tool calling schema；
- 将 `/state` 转换为 Scene Graph，供记忆系统存储；
- 用失败日志训练或微调更稳的 action policy。

长期：

- 由中转 agent 统一处理任务分配、失败转交和重规划；
- 引入 Scene Graph World Model 评估候选计划；
- 将 Qwen policy、记忆系统、AI2-THOR receiver 整合成完整多机器人协作闭环。
