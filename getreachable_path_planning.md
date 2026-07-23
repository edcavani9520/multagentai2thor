# AI2-THOR 可达位置与自动路径规划整理

本文档整理当前项目中 `GetReachablePositions` 的实现方式，并给出在此基础上实现自动路径规划的推荐方案。

## 1. 结论

当前项目已经使用 AI2-THOR 的 `GetReachablePositions`，但它的用途主要是：

- 初始化多个 robot 的出生位置；
- 避免多个 robot 出生位置重叠；
- 可选地把 `Robot0` 放到某个物体附近。

当前项目还没有实现完整的 `goto(target)` 接口，也没有实现 A*、Dijkstra 或其他几何路径规划器。

当前任务闭环中的导航，主要由 Qwen 根据第一视角图像生成动作序列，例如：

```json
[
  {"action": "MoveAhead"},
  {"action": "RotateRight"},
  {"action": "MoveAhead"}
]
```

这些动作再通过 `POST /execute_actions` 发送给 AI2-THOR 执行。

## 2. 当前仿真结构

项目的核心链路如下：

```text
客户端
  |
  | POST /execute_task
  v
relay_task_server.py
  |
  | 任务意图、视觉规划、object grounding、闭环重规划
  v
EmbodiedGPT_Pytorch/demo/auto_scene_actions.py
  |
  | POST /observe、/state、/execute_actions
  v
ai2thor_receiver_server.py
  |
  v
AI2-THOR Controller(agentCount=N)
  |
  v
共享 Unity 场景
```

每个 robot 对应一个 AI2-THOR `agentId`，例如 `robot_id=0` 对应 `agentId=0`。

## 3. `GetReachablePositions` 的现有实现

### 3.1 调用方式

文件：`ai2thor_receiver_server.py`

当前代码通过 AI2-THOR Controller 发送以下 action：

```python
event = self._controller_step({
    "action": "GetReachablePositions",
    "agentId": 0,
})
```

AI2-THOR 将结果放在事件 metadata 的 `actionReturn` 字段中：

```python
meta = self._event_for_robot(event, 0).metadata
positions = meta.get("actionReturn") or []
```

该逻辑位于：

```text
ai2thor_receiver_server.py:590
```

### 3.2 返回数据清洗

项目只保留位置中的 `x`、`y`、`z` 三个坐标，并且将它们转换为浮点数：

```python
cleaned.append({
    "x": float(pos["x"]),
    "y": float(pos.get("y", DEFAULT_AGENT_Y)),
    "z": float(pos["z"]),
})
```

如果 AI2-THOR 返回的位置缺少 `x` 或 `z`，该位置会被忽略。缺少 `y` 时使用：

```python
DEFAULT_AGENT_Y = 0.900999128818512
```

因此，当前 `_get_reachable_positions()` 的返回格式类似：

```python
[
    {"x": -1.50, "y": 0.90, "z": 2.00},
    {"x": -1.25, "y": 0.90, "z": 2.00},
    {"x": -1.00, "y": 0.90, "z": 2.00},
]
```

### 3.3 当前用途一：选择出生点

启动 Controller 后，项目会执行：

```python
positions = self._select_spawn_positions(self.robot_count)
```

`_select_spawn_positions()` 的流程是：

1. 调用 `_get_reachable_positions()`；
2. 如果查询失败，则使用 `_fallback_positions()`；
3. 如果可达点数量不足，直接取前几个点；
4. 否则先选择离可达点中心最远的点；
5. 之后每次选择距离已选点集合最远的点，以尽量分散多个 robot。

这不是路径规划，只是出生点选择。

### 3.4 当前用途二：把 Robot0 放到物体附近

如果启动参数启用了 `--robot0-at-fridge`，项目会：

1. 查询 `Fridge` 的物体坐标；
2. 再次获取所有可达位置；
3. 过滤掉距离冰箱小于 `robot0_fridge_distance` 的位置；
4. 选择距离冰箱最近的可达位置；
5. 使用 `TeleportFull` 将 Robot0 放到该位置。

对应代码位于：

```text
ai2thor_receiver_server.py:606
```

这里是“从可达点中选一个合适点并传送”，仍然不是从当前位置走到目标位置的路线规划。

### 3.5 当前用途三：避免多机器人出生重叠

`_avoid_spawn_overlap()` 会检查出生点之间的距离。如果两个点过近，就从 reachable positions 中重新选择一个距离已选位置尽可能远的点。

默认最小距离为：

```python
min_distance = 0.75
```

对应代码位于：

```text
ai2thor_receiver_server.py:634
```

## 4. 当前对外动作接口

当前 receiver 对外暴露的是：

```text
POST /execute_actions
```

请求示例：

```json
{
  "task_id": "navigation-001",
  "robot_id": 0,
  "stop_on_failure": true,
  "actions": [
    {"action": "MoveAhead"},
    {"action": "RotateRight"},
    {"action": "MoveAhead"}
  ]
}
```

服务端会遍历 `actions`，取出每个动作的 `action` 字段，然后调用：

```python
self.execute(robot_ref, action_name, render_image=render_image, **act)
```

最终由 `execute()` 组装 AI2-THOR action：

```python
action_dict["action"] = action
action_dict["agentId"] = robot.robot_id
action_dict["renderImage"] = render_image
event = self._controller_step(action_dict)
```

因此，目前可以直接执行：

```text
MoveAhead
MoveBack
MoveLeft
MoveRight
RotateLeft
RotateRight
LookUp
LookDown
TeleportFull
```

但是接口本身不会自动根据目标坐标生成路径。

## 5. 基于 reachable positions 实现自动规划

推荐新增一个独立的路径规划层，基本流程如下：

```text
读取 robot 当前位姿
  |
  v
读取场景目标坐标
  |
  v
调用 GetReachablePositions
  |
  v
将可达点构造成图
  |
  v
A* 或 Dijkstra 搜索起点到目标点
  |
  v
将点路径转换为旋转和移动动作
  |
  v
POST /execute_actions
  |
  v
重新读取位姿，检查动作结果
  |
  +--> 失败：重新规划
```

### 5.1 起点

起点来自 `/state` 或 `/robots` 返回的 robot 位姿：

```json
{
  "position": {"x": -1.5, "y": 0.9, "z": 2.0},
  "rotation": {"x": 0, "y": 90.0, "z": 0},
  "horizon": 0.0
}
```

路径规划通常只在二维平面使用：

```text
节点坐标 = (x, z)
```

`y` 需要保留，用于最后的 `TeleportFull` 或进行楼层过滤，但不应直接当成二维路径坐标。

### 5.2 目标点

目标可以来自：

- 用户直接提供的世界坐标；
- `/state` 中物体的 `position`；
- 物体周围的一个可交互站立点；
- AI2-THOR 返回的目标对象 metadata。

如果任务是“走到冰箱旁边”，不能把冰箱中心点直接作为机器人落点。应该从 reachable positions 中选一个距离冰箱合适、且最后朝向冰箱的位置。

可使用类似以下条件选择终点：

```text
距离目标大于最小交互距离
距离目标小于最大交互距离
该点属于 reachable positions
该点尽量能看到目标
```

### 5.3 图节点

每个 reachable position 是一个图节点：

```python
node = (round(position["x"], 3), round(position["z"], 3))
```

使用 `round` 的原因是浮点坐标不能直接作为稳定的字典 key。实际项目中建议统一使用一个小的坐标量化精度，例如 `0.001` 或 `0.01`。

### 5.4 图边

最简单的做法是连接相邻网格点。

假设 AI2-THOR 的移动网格尺寸约为 `grid_size`，对于两个点 `a` 和 `b`，可以使用：

```text
abs(dx) <= grid_size + epsilon
abs(dz) <= grid_size + epsilon
dx + dz > 0
```

为了避免斜向移动，可以只连接以下四种邻居：

```text
(+grid_size, 0)
(-grid_size, 0)
(0, +grid_size)
(0, -grid_size)
```

更稳妥的实现不是只依赖距离，而是对每个候选邻居执行一次模拟器移动验证。原因是 reachable positions 是可达点集合，但仅凭点间距离不一定能准确表达每一条动作边对应的碰撞行为。

推荐的工程折中：

1. 先用坐标距离建立候选边；
2. 规划得到动作后，实际执行时检查 `lastActionSuccess`；
3. 如果某条边失败，将该边加入临时黑名单；
4. 从最新状态重新运行 A*。

### 5.5 A* 搜索

在二维平面中，A* 的启发式函数可以使用欧氏距离：

```python
def heuristic(a, b):
    dx = a[0] - b[0]
    dz = a[1] - b[1]
    return (dx * dx + dz * dz) ** 0.5
```

如果图只允许横向和纵向移动，也可以使用曼哈顿距离：

```python
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
```

A* 的核心逻辑是：

```python
open_set = PriorityQueue()
open_set.push(start, priority=0)

g_score = {start: 0.0}
came_from = {}

while open_set:
    current = open_set.pop()
    if current == goal:
        return reconstruct_path(came_from, current)

    for neighbor, edge_cost in graph[current]:
        tentative = g_score[current] + edge_cost
        if tentative < g_score.get(neighbor, float("inf")):
            came_from[neighbor] = current
            g_score[neighbor] = tentative
            priority = tentative + heuristic(neighbor, goal)
            open_set.push(neighbor, priority)

return None
```

路径不存在时应返回结构化错误，例如：

```json
{
  "status": "failed",
  "error": "no path between start and goal"
}
```

## 6. 将点路径转换成 AI2-THOR 动作

点路径只描述机器人应该经过哪些坐标，例如：

```text
P0 -> P1 -> P2 -> P3
```

但 `/execute_actions` 需要的是动作序列。因此需要对每一段路径做以下处理：

1. 计算下一点相对于当前点的方向；
2. 计算当前 yaw 与目标方向的差值；
3. 生成若干次 `RotateLeft` 或 `RotateRight`；
4. 生成一次 `MoveAhead`；
5. 更新当前 yaw，继续处理下一段。

二维方向角可以定义为：

```python
import math

target_yaw = math.degrees(math.atan2(dx, dz)) % 360
```

这里使用 `atan2(dx, dz)`，因为 AI2-THOR 的前方通常与世界坐标的 `+z` 方向对应，具体方向仍应通过实际场景验证。

角度差应归一化到 `[-180, 180)`：

```python
def signed_angle_delta(target_yaw, current_yaw):
    return (target_yaw - current_yaw + 180.0) % 360.0 - 180.0
```

生成动作时，应使用项目和 AI2-THOR 当前配置中的旋转步长。例如旋转步长是 90 度时：

```python
delta = signed_angle_delta(target_yaw, current_yaw)

if delta > 0:
    turns = round(delta / 90.0)
    actions.extend({"action": "RotateRight"} for _ in range(turns))
else:
    turns = round(-delta / 90.0)
    actions.extend({"action": "RotateLeft"} for _ in range(turns))

actions.append({"action": "MoveAhead"})
```

注意：左转还是右转的符号必须用一次实际动作确认。不同版本或坐标约定可能导致 yaw 增长方向与直觉不同。

如果路径段不是机器人当前朝向的前方，也可以使用：

```text
RotateLeft / RotateRight
MoveAhead
```

不建议默认将每个点直接转换成 `TeleportFull`，因为那会跳过真实导航中的碰撞和移动过程。

## 7. 推荐的 `/goto` 接口设计

如果希望对外提供真正的 `goto`，可以在 `ai2thor_receiver_server.py` 增加：

```text
POST /goto
```

请求示例：

```json
{
  "task_id": "goto-001",
  "robot_id": 0,
  "target": {"x": -1.0, "y": 0.9, "z": 1.5},
  "planner": "astar",
  "execute": true,
  "render_image": false,
  "stop_on_failure": true
}
```

服务端内部流程建议为：

```python
current = get_robot_pose(robot_id)
reachable = get_reachable_positions(robot_id)
goal = choose_nearest_reachable_goal(reachable, target)
graph = build_reachable_graph(reachable)
point_path = astar(graph, current, goal)
actions = path_to_actions(point_path, current_yaw=current_yaw)
result = execute_batch(
    actions,
    default_robot_ref=robot_id,
    render_image=render_image,
    stop_on_failure=stop_on_failure,
)
```

返回建议包含规划和执行两部分：

```json
{
  "status": "success",
  "robot_id": 0,
  "target": {"x": -1.0, "y": 0.9, "z": 1.5},
  "goal_position": {"x": -1.0, "y": 0.9, "z": 1.5},
  "path": [
    {"x": -1.5, "y": 0.9, "z": 2.0},
    {"x": -1.25, "y": 0.9, "z": 2.0},
    {"x": -1.0, "y": 0.9, "z": 1.5}
  ],
  "actions": [
    {"action": "RotateLeft"},
    {"action": "MoveAhead"}
  ],
  "results": []
}
```

如果 `execute=false`，接口只返回规划结果，不改变仿真状态，便于调试和 dry-run。

## 8. 目标物体附近的导航

物体导航通常不是“到达物体中心”，而是“到达一个可以交互的位置”。建议使用以下策略：

1. 从物体 metadata 获取目标位置；
2. 从 reachable positions 中筛选与物体距离在交互范围内的点；
3. 如果没有合适点，返回 `target_not_reachable`；
4. 对候选点按距离、可见性和朝向评分；
5. 选择评分最高的候选点作为 A* 终点；
6. 到达终点后再执行 `PickupObject`、`OpenObject` 或 `PutObject`。

可使用如下评分思路：

```text
总分 = 路径长度惩罚
     + 与目标距离惩罚
     + 不可见惩罚
     + 朝向偏差惩罚
```

这比直接把物体坐标作为机器人终点更可靠。

## 9. 执行中的闭环修正

一次性规划出的动作序列仍可能因为以下原因失败：

- 其他 robot 阻挡；
- 场景状态发生变化；
- 坐标量化误差；
- 旋转步长与假设不一致；
- 机器人没有完全到达预期位置；
- 目标物体移动或被其他动作改变。

因此建议不要一次执行过长的动作列表。更稳妥的方式是分段执行：

```text
规划 3～8 个动作
  -> 执行
  -> 检查 success 和 robot_pose
  -> 更新起点
  -> 重新规划剩余路径
```

`/execute_actions` 的每个结果包含：

```text
success
error
robot_pose
robot_pose_changed
```

如果某步失败，可以把失败动作对应的边暂时加入黑名单，然后重新调用 A*。如果多次规划都失败，应返回：

```json
{
  "status": "needs_upstream_planning",
  "reason": "navigation failed after replanning"
}
```

## 10. 多机器人场景的注意事项

当前所有 robot 共用同一个 Unity 场景。`GetReachablePositions` 得到的是场景中的静态可达点，不代表这些点在执行时一定没有其他 robot 占用。

因此多机器人规划至少需要考虑：

- 将其他 robot 的当前位置作为动态障碍；
- 执行前再次查询 `/robots` 或 `/state`；
- 不让两个 robot 同时规划到同一个终点；
- 对关键动作设置 `stop_on_failure=true`；
- 一个 robot 移动后，其他 robot 的旧路径可能需要重新规划。

当前项目的 relay 逻辑主要解决“哪个 robot 执行任务”和“哪个 robot 能看到目标”，并不负责多机器人几何路径避碰。

## 11. 推荐实现顺序

建议按以下顺序实现：

### 阶段一：暴露可达点

新增：

```text
GET /reachable_positions?robot_id=0
```

返回：

```json
{
  "robot_id": 0,
  "positions": [
    {"x": -1.5, "y": 0.9, "z": 2.0}
  ]
}
```

### 阶段二：实现纯规划函数

先实现不执行动作的函数：

```text
build_reachable_graph(positions)
find_nearest_node(position)
astar(graph, start, goal)
path_to_actions(path, current_pose)
```

使用 `execute=false` 或单元测试验证路径和动作序列。

### 阶段三：实现 `/goto`

将上述函数串接起来，但先支持 dry-run：

```text
POST /goto
  -> 获取起点
  -> 获取 reachable positions
  -> 规划
  -> 返回 path 和 actions
```

### 阶段四：加入分段闭环执行

规划少量动作，执行后检查结果，再从实际 pose 重新规划。

### 阶段五：加入物体交互终点和多机器人避碰

最后再将“物体附近站立点”“可见性”“动态 robot 障碍”纳入代价函数和终点选择。

## 12. 相关代码位置

| 功能 | 文件和位置 |
| --- | --- |
| 创建 AI2-THOR Controller | `ai2thor_receiver_server.py:135` |
| 多 robot 初始化 | `ai2thor_receiver_server.py:183` |
| 初始位置选择 | `ai2thor_receiver_server.py:574` |
| `GetReachablePositions` | `ai2thor_receiver_server.py:590` |
| 目标物体附近位置选择 | `ai2thor_receiver_server.py:606` |
| 避免出生点重叠 | `ai2thor_receiver_server.py:634` |
| 执行单个动作 | `ai2thor_receiver_server.py:268` |
| 执行动作批次 | `ai2thor_receiver_server.py:337` |
| HTTP `/execute_actions` | `ai2thor_receiver_server.py:866`、`889` |
| 导航动作集合 | `EmbodiedGPT_Pytorch/demo/auto_scene_actions.py:45` |
| 闭环任务重规划 | `EmbodiedGPT_Pytorch/demo/auto_scene_actions.py:3968` |
| 初始场景观察和任务规划 | `EmbodiedGPT_Pytorch/demo/auto_scene_actions.py:4420` |

## 13. 最终判断

当前项目已经具备自动路径规划所需的基础数据来源：

- 当前 robot 位置和朝向；
- AI2-THOR 的 reachable positions；
- 目标物体位置；
- 动作执行成功/失败反馈；
- 多 robot 的共享场景状态。

但现有代码还缺少以下核心组件：

- reachable positions 对外接口；
- 可达点图构建；
- A* 或 Dijkstra 搜索；
- 点路径到旋转/移动动作的转换；
- `/goto` 接口；
- 动态障碍和失败后的局部重规划。

因此，推荐的实现路线是：

```text
GetReachablePositions
  -> reachable graph
  -> A*
  -> Rotate + MoveAhead
  -> /execute_actions
  -> observe and replan
```
