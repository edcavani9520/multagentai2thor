# LaMMA-P Multi-Agent Framework — 教程

在 AI2-THOR 仿真环境中运行多智能体任务的框架，支持异构机器人轮询式协同。

---

## 目录

1. [架构总览](#1-架构总览)
2. [目录结构](#2-目录结构)
3. [核心组件](#3-核心组件)
4. [运行逻辑](#4-运行逻辑)
5. [CLI 使用](#5-cli-使用)
6. [接入真实 Agent](#6-接入真实-agent)
7. [GUI 模式](#7-gui-模式)
8. [FAQ / 踩坑记录](#8-faq--踩坑记录)

---

## 1. 架构总览

```
                    ┌──────────────────────────┐
                    │     MultiAgentEnv        │
                    │     (中央协调器)           │
                    └─────────┬────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
         Agent A          Agent B        Agent N
         (obs→action)     (obs→action)   (obs→action)
              │               │               │
              └───────┬───────┴───────┬───────┘
                      │               │
              ┌───────┴───────┐ ┌─────┴────────┐
              │  Teleport to  │ │  Execute      │
              │  agent A's    │ │  agent A's    │
              │  viewpoint    │ │  action       │
              └───────────────┘ └──────────────┘
                      │
              ┌───────┴───────┐
              │ ai2thor.Controller  │
              │ (共享单个 Unity)    │
              └───────────────────┘
```

**核心设计决策：** 不用 AI2-THOR 自带的 RobotController（多 HTTP client 无同步），改用**单一 Controller + Python 层状态管理 + 轮询**。

每个 agent 的"视角"通过 `TeleportFull` 移动相机到该 agent 的位置来模拟，agent 的动作通过同一个 Controller 执行。

---

## 2. 目录结构

```
lamma-p-framework/
├── run.py                     # 入口，CLI 参数处理
├── multi_agent_env.py         # 核心：MultiAgentEnv 协调器
├── agents/
│   ├── __init__.py
│   ├── base_agent.py          # Agent 接口定义
│   ├── dummy_agent.py         # 占位 agent（不动，只写日志）
│   └── colleague_agent_stub.py # 同学的 agent 模板
├── output/                    # 运行输出截图 + 日志
├── TUTORIAL.md                # 本教程
└── README.md                  # 快速入门
```

---

## 3. 核心组件

### 3.1 `MultiAgentEnv` — 中央协调器

路径: `multi_agent_env.py`

**职责：**
- 启动/管理 AI2-THOR Controller
- 管理所有 agent 的状态（位置、朝向、任务）
- 提供 agent 视角的观察（`observe_agent`）
- 执行 agent 的动作（`step_agent`）
- 运行轮询主循环（`run_agents_round_robin`）

**关键类：**

```python
class AgentState:
    """单个 agent 的状态"""
    agent_id: int
    name: str
    position: dict      # {"x": ..., "y": ..., "z": ...}
    rotation: dict      # {"x": ..., "y": ..., "z": ...}
    horizon: float
    last_event: Event   # 最近一次观察的 event
    action_history: list # 动作历史
    task: str            # 当前任务
```

```python
class MultiAgentEnv:
    def observe_agent(agent) -> dict     # 获取 agent 视角的观察
    def step_agent(agent, action) -> dict # 在 agent 视角下执行动作
    def place_proxy_object(...) -> bool   # 放置可见的代理物体
    def run_agents_round_robin(...)       # 主循环
```

### 3.2 Agent 接口

路径: `agents/base_agent.py`

每个 agent 就是一个函数：

```python
def my_agent(observation: dict) -> dict:
    """
    Args:
        observation = {
            "agent_name":     str,            # 当前 agent 名字
            "agent_id":       int,            # ID
            "image":          np.ndarray,     # (H, W, 3) RGB 相机画面
            "objects":        list,           # 场景内所有物体
            "camera_position": dict,          # 当前相机位置
            "agent_position":  dict,          # 本 agent 预设位置
            "metadata":       dict,           # 完整 AI2-THOR metadata
        }
    Returns:
        action = {
            "action": str,        # AI2-THOR 动作名
            # 以及动作相关参数，如:
            # "objectId": str,
            # "position": dict,
            # "rotation": dict,
            # "receptacleObjectId": str,
            # ...
        }
    """
```

### 3.3 Dummy Agent

路径: `agents/dummy_agent.py`

占位用，**不移动**（始终返回 `{"action": "Pass"}`），只做三件事：
1. 打印当前可见物体列表
2. 将观察数据写入 JSONL 日志文件
3. 返回 no-op 动作

当真实 agent 未就绪时，用 dummy agent 测试框架完整性。

### 3.4 Colleague Agent Stub

路径: `agents/colleague_agent_stub.py`

你同学 agent 的接入模板。等他的 agent 写好了，替换此文件中的 `colleague_agent` 函数即可。

---
(vla_env) kinova-1@kinova1-HP-Z8-G4-Workstation:~/mult agent ai2thor$ conda activate vla_env
python -c "import transformers; print('ok')"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'transformers'
(vla_env) kinova-1@kinova1-HP-Z8-G4-Workstation:~/mult agent ai2thor$ which python
# 应该输出: /home/kinova-1/anaconda3/envs/vla_env/bin/python
/home/kinova-1/anaconda3/envs/vla_env/bin/python

## 4. 运行逻辑

### 4.1 启动阶段

```
1. 创建 MultiAgentEnv
   ├── 启动 ai2thor.Controller (CloudRendering / 窗口)
   ├── 加载场景 (FloorPlan1-4)
   └── 读取场景物体列表

2. 注册 Agent
   ├── 分配 agent_id、名字、位置、朝向
   ├── 放置 proxy 物体（让 robot A 的"身体"能被 B 看到）
   └── 添加到 env.agents 列表

3. 验证初始观察
   ├── TeleportFull 到每个 agent 的位置
   ├── 拍摄初始截图 → output/initial_view_*.png
   └── 打印最近物体
```

### 4.2 轮询主循环

```
for step in range(max_steps):
    for each agent:
        1. observe_agent(agent)
           └── TeleportFull 到 agent 位置 → 获取 event
           └── event.cv2img → RGB 图像
           └── event.metadata → 全部场景状态

        2. agent_fn(observation) → action
           └── 调用 agent 函数，传入观察
           └── 拿到 AI2-THOR 动作

        3. step_agent(agent, action)
           └── TeleportFull 确保位置正确
           └── controller.step(action)
           └── 记录 action_history
```

### 4.3 "看到彼此" 的机制

AI2-THOR 的默认 agent 是**不可见的**（只有相机，没有身体模型）。为了让两个 robot "看到彼此"，框架会：

1. 在场景中选取两个物体作为 robot 的 body proxy（如 Bowl、Vase）
2. 用 `TeleportObject` 把它们放在两个 agent 之间的位置
3. 各 agent 从自己的位置可以看到对方的 proxy 物体

如果 proxy 物体和 agent 位置重叠，`TeleportFull` 会因碰撞检测失败。所以 proxy 放在两个 agent 之间的中点附近，不重叠。

---

## 5. CLI 使用

### 5.1 基本用法

```bash
cd ~/lamma-p-framework

# 默认运行（headless，5步）
python3 run.py

# 换场景
python3 run.py --scene FloorPlan3

# 更多步数
python3 run.py --steps 20

# 指定输出目录
python3 run.py --output demo_results

# 显示 Unity 窗口（需要图形界面）
python3 run.py --no-headless
```

### 5.2 完整参数

| 参数                           | 默认值     | 说明                            |
| ------------------------------ | ---------- | ------------------------------- |
| `--scene`                      | FloorPlan1 | AI2-THOR 场景名                 |
| `--headless` / `--no-headless` | headless   | CloudRendering / Unity 窗口     |
| `--output`                     | output     | 截图和日志输出目录              |
| `--steps`                      | 5          | 每 agent 执行的步数             |
| `--show`                       | off        | 打开两个 OpenCV 窗口显示视角    |
| `--keep-open`                  | off        | 运行完后不关闭窗口              |
| `--real-agent`                 | off        | 使用 colleague_agent 替代 dummy |

### 5.3 组合示例

```bash
# 开发调试：终端输出日志
python3 run.py --steps 3

# 看两个 robot 的视角（OpenCV 窗口）
python3 run.py --show --keep-open

# 看 Unity 窗口 + OpenCV 双窗口
python3 run.py --no-headless --show --keep-open --steps 10

# 接入真实 agent 运行
python3 run.py --real-agent --steps 100

# 全部功能
python3 run.py --scene FloorPlan2 --no-headless --show --keep-open \
               --steps 50 --output my_exp --real-agent
```

---

## 6. 接入真实 Agent

### 6.1 编写 Agent 函数

文件: `agents/colleague_agent_stub.py`

```python
def my_agent(obs):
    """
    你的 agent 逻辑写在这里。
    
    obs 包含:
      - image:    (600,600,3) RGB numpy 数组
      - objects:  场景所有物体列表，每个包含:
                  objectType, objectId, position, distance, mass, ...
      - camera_position: 当前相机位置
      - agent_position:  本 agent 的位置
      - metadata: 完整 AI2-THOR metadata
      
    返回:
      {"action": "MoveAhead"}   # 前进一步
      {"action": "RotateRight"}  # 右转
      {"action": "PickupObject", "objectId": "Apple|..."}
      {"action": "PutObject", "objectId": "...", "receptacleObjectId": "..."}
      等等——所有 AI2-THOR 标准动作都支持
    """
    
    # ==== 你的代码从这里开始 ====
    
    # 示例：查找最近的苹果
    for obj in obs["objects"]:
        if obj["objectType"] == "Apple":
            return {
                "action": "PickupObject",
                "objectId": obj["objectId"]
            }
    
    return {"action": "Pass"}
```

### 6.2 注册 Agent

在 `run.py` 的 `main()` 中找到 `agent_fns` 字典，替换：

```python
# 修改前
agent_fns[agent_b.agent_id] = make_dummy_agent("RobotB", ...)

# 修改后
from agents.colleague_agent_stub import colleague_agent
agent_fns[agent_b.agent_id] = colleague_agent
```

或者直接用 `--real-agent` flag（如果 stub 已实现）：

```bash
python3 run.py --real-agent
```

### 6.3 多 Agent 混合

```python
agent_fns = {
    agent_a.agent_id: my_custom_agent_1,    # 你的 agent
    agent_b.agent_id: make_dummy_agent("RobotB"),  # 同学还没好，先用 dummy
    agent_c.agent_id: my_custom_agent_2,    # 第三个 agent
}
```

---

## 7. GUI 模式

### 7.1 OpenCV 双窗口 (`--show`)

```
┌─────────────────────┐  ┌─────────────────────┐
│    RobotA View      │  │    RobotB View      │
│                     │  │                     │
│  [场景厨房左侧视角]   │  │  [场景厨房右侧视角]   │
│                     │  │                     │
│  Step: 3            │  │  Step: 3            │
└─────────────────────┘  └─────────────────────┘
```

- 每个窗口 600×600
- 左上角显示当前 step 编号
- 步进阶段：每 100ms 更新一次
- 空闲阶段：每 500ms 刷新，显示 "IDLE — press q to exit"

### 7.2 Unity 窗口 (`--no-headless`)

AI2-THOR 自带的 Unity 游戏窗口，显示 agent 的第一人称视角。在步进过程中会随 agent 切换而变化。

### 7.3 组合模式

同时显示 Unity 窗口和两个 OpenCV 窗口：

```bash
python3 run.py --no-headless --show --keep-open
```

Unity 显示当前 active agent 的视角，两个 OpenCV 窗口固定显示 RobotA 和 RobotB 的视角。

---

## 8. FAQ / 踩坑记录

### Q: TeleportFull 失败？

**原因：** 目标位置有物体阻挡（碰撞检测），最常见的是 proxy 物体放在了 agent 同一位置。

**解决：** 确保 proxy 物体和 agent 位置不重叠。框架已自动处理。

### Q: 没有图形界面怎么办？

框架默认 `--headless`（CloudRendering），不需要 DISPLAY。截图保存到 `output/` 目录。

如果是 SSH 远程，可以用 `ssh -X` 转发图形界面。

### Q: 怎么加更多 agent？

在 `run.py` 中添加：

```python
agent_c = env.add_agent(
    name="RobotC",
    position={"x": 0.0, "y": 0.900999, "z": 0.0},
    rotation={"x": 0, "y": 90, "z": 0},
    task="Explore"
)
agent_fns[agent_c.agent_id] = make_dummy_agent("RobotC")
```

### Q: 如何让 agent 真的移动？

把 dummy agent 换成你的 agent 函数，返回真实的 AI2-THOR 动作：

```python
def my_agent(obs):
    return {"action": "MoveAhead"}  # 或者 RotateRight, PickupObject 等
```

### Q: 保存的图片在哪？

```
output/
├── initial_view_RobotA.png    # 初始视角
├── initial_view_RobotB.png
├── step_0000_RobotA.png       # 每步视角
├── step_0000_RobotB.png
├── ...
├── RobotA_observations.jsonl  # 结构化日志
└── RobotB_observations.jsonl
```

JSONL 每行一个 JSON 对象，包含 agent 名、位置、可见物体列表、动作等。

### Q: 支持什么版本的 AI2-THOR？

测试于 AI2-THOR 5.0.0。`TeleportFull`、`TeleportObject` 是核心依赖动作。

---

## 参考

- LaMMA-P: [arxiv:2409.20560](https://arxiv.org/abs/2409.20560)
- AI2-THOR: [github.com/allenai/ai2thor](https://github.com/allenai/ai2thor)
