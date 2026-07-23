1,所有 HTTP 输入输出都是 JSON。图像使用 JPEG base64 字符串，字段名为 `image_base64`。

###  `GET /robots`

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

###  `GET /state`

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

### `POST /observe`

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

###  `POST /execute_actions`

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

###  `POST /reset`

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

## 2 AI2-THOR 动作与 objectId

###  常见动作

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

### objectId 的意义

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

###  PutObject 的常见失败原因

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
