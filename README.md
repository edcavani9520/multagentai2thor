# LaMMA-P Multi-Agent AI2-THOR Framework

基于 AI2-THOR 的多智能体框架，支持异构机器人任务分配与协同。

```
python run.py --show --keep-open
```

## 快速开始

```bash
pip install ai2thor
cd ~/lamma-p-framework

# 基本运行
python3 run.py

# 看两个 robot 视角
python3 run.py --show

# 完整功能
python3 run.py --no-headless --show --keep-open --steps 20
```

## 文档

详细教程：[TUTORIAL.md](./TUTORIAL.md)

涵盖：
- 架构与运行逻辑
- 所有 CLI 参数
- Agent 接口定义
- 接入真实 Agent 的方法
- GUI 模式与窗口管理
- FAQ / 踩坑记录

## 项目结构

```
├── run.py                 # 入口
├── multi_agent_env.py     # 核心协调器
├── agents/
│   ├── base_agent.py      # Agent 接口
│   ├── dummy_agent.py     # 占位 agent
│   └── colleague_agent_stub.py  # 接入模板
├── output/                # 截图 + 日志
├── TUTORIAL.md            # 完整教程
└── README.md
```
