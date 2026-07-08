"""
Custom Discrete Agent Adapter
==============================
接入输出为离散动作的外部模型，格式: [base_action, modifier]

常用的 base_action × modifier 组合到 AI2-THOR 动作的映射。

用法:
    agent = CustomDiscreteAgent("RobotA", model, instruction="pick up the bowl")
    action = agent(observation)   # → AI2-THOR action dict
"""

from typing import List, Optional, Callable
import numpy as np


# ── 动作映射表 ──
#
# 你的模型输出 [base_action, modifier] 格式，
# 映射到 AI2-THOR 的离散动作。

ACTION_MAP = {
    # ── 导航 ──
    ("moveahead",  "none"):      {"action": "MoveAhead"},
    ("moveahead",  "left"):      {"action": "MoveAhead"},       # 左/右移用 MoveLeft/Right
    ("moveahead",  "right"):     {"action": "MoveAhead"},
    ("moveback",   "none"):      {"action": "MoveBack"},
    ("moveleft",   "none"):      {"action": "MoveLeft"},
    ("moveright",  "none"):      {"action": "MoveRight"},
    ("rotateleft", "none"):      {"action": "RotateLeft"},
    ("rotateright","none"):      {"action": "RotateRight"},
    ("lookup",     "none"):      {"action": "LookUp"},
    ("lookdown",   "none"):      {"action": "LookDown"},
    ("stop",       "none"):      {"action": "Pass"},

    # ── 操作 (gripper/arm) ──
    # 拾取：需要在运行时找到最近的物体填充 objectId
    ("pickup",     "none"):      {"action": "PickupObject"},    # objectId 由外部填充
    ("place",      "none"):      {"action": "PutObject"},
    ("drop",       "none"):      {"action": "DropHandObject"},
    ("open",       "none"):      {"action": "OpenObject"},
    ("close",      "none"):      {"action": "CloseObject"},
    ("toggleon",   "none"):      {"action": "ToggleObjectOn"},
    ("toggleoff",  "none"):      {"action": "ToggleObjectOff"},

    # ── 手臂特殊命令 ──
    ("moveahead",  "lefthand"):  {"action": "MoveAhead"},
    ("moveback",   "righthand"): {"action": "MoveBack"},
    ("rotateleft", "arm"):       {"action": "RotateLeft"},
}


class CustomDiscreteAgent:
    """
    离散动作模型适配器。

    用法:
        >>> agent = CustomDiscreteAgent(
        ...     name="RobotA",
        ...     model=my_model,
        ...     instruction="pick up the bowl",
        ... )

        >>> action = agent(observation)

    如果是纯规则（无模型），直接传 model=None，
    然后用 PromptedDiscreteAgent（见下）。
    """

    def __init__(
        self,
        name: str,
        model: Optional[Callable] = None,
        instruction: str = "",
        device: str = "cuda:0",
    ):
        """
        Args:
            name: Agent 名称
            model: 你的模型 callable，签名 (image, text) → [base_action, modifier]
                   例如 output = ["moveahead", "none"]
            instruction: 文本指令
            device: 推理设备
        """
        self.name = name
        self.model = model
        self.instruction = instruction
        self.device = device

        print(f"[{name}] CustomDiscreteAgent 初始化")
        print(f"  ├─ 模型: {'有' if model else '无 (规则模式)'}")
        print(f"  └─ 指令: \"{instruction}\"")

    def __call__(self, obs: dict) -> dict:
        """
        输入 observation → 输出 AI2-THOR action dict
        """
        img = obs.get("image")
        objects = obs.get("objects", [])

        if img is None:
            return {"action": "Pass"}

        # ── 调用模型 ──
        if self.model is not None:
            try:
                output = self.model(img, self.instruction)
            except Exception as e:
                print(f"[{self.name}] 模型推理失败: {e}")
                return {"action": "Pass"}
        else:
            # 没有模型 → 返回 Pass
            return {"action": "Pass"}

        # ── 解析输出 ──
        if isinstance(output, (list, tuple)) and len(output) == 2:
            base_action = str(output[0]).lower().strip()
            modifier = str(output[1]).lower().strip()
        elif isinstance(output, str):
            # 如果模型直接输出 "moveahead" 或 "moveahead,left"
            parts = output.replace(",", " ").split()
            base_action = parts[0].lower().strip() if parts else "pass"
            modifier = parts[1].lower().strip() if len(parts) > 1 else "none"
        else:
            print(f"[{self.name}] 无法解析模型输出: {output}")
            return {"action": "Pass"}

        # ── 查表映射 ──
        key = (base_action, modifier)
        if key in ACTION_MAP:
            action = dict(ACTION_MAP[key])  # 拷贝，避免修改全局表
        else:
            # 尝试只用 base_action（忽略 modifier）
            fallback_key = (base_action, "none")
            if fallback_key in ACTION_MAP:
                action = dict(ACTION_MAP[fallback_key])
                print(f"[{self.name}] 未找到 ({base_action}, {modifier}) 的精确映射，"
                      f"回退到 ({base_action}, none)")
            else:
                print(f"[{self.name}] 无法映射: base={base_action}, mod={modifier}")
                return {"action": "Pass"}

        # ── 对需要 objectId 的动作，自动填补最近物体 ──
        if action["action"] in ("PickupObject", "OpenObject", "CloseObject",
                                "ToggleObjectOn", "ToggleObjectOff"):
            target = self._find_nearest_object(objects, action["action"])
            if target:
                action["objectId"] = target
            else:
                print(f"[{self.name}] 未找到可操作的物体，跳过 {action['action']}")
                return {"action": "Pass"}

        if action["action"] == "PutObject":
            target = self._find_nearest_receptacle(objects)
            if target:
                action["objectId"] = target
            else:
                print(f"[{self.name}] 未找到可放置的容器，跳过 PutObject")
                return {"action": "Pass"}

        # 附加调试信息
        action["_raw_output"] = output

        return action

    def _find_nearest_object(self, objects: list, action_type: str) -> str:
        """找最近的、可交互的物体 objectId"""
        SKIP_TYPES = {"Floor", "Wall", "Window", "Door", "Ceiling"}

        filter_key = None
        if action_type == "PickupObject":
            filter_key = "pickupable"
        elif action_type == "OpenObject":
            filter_key = "openable"
        elif action_type == "CloseObject":
            filter_key = "openable"

        nearest = None
        min_dist = float("inf")

        for obj in objects:
            obj_type = obj.get("objectType", "")
            if obj_type in SKIP_TYPES:
                continue

            dist = obj.get("distance", float("inf"))

            if filter_key and not obj.get(filter_key, False):
                continue

            if dist < min_dist:
                min_dist = dist
                nearest = obj.get("objectId")

        if nearest and min_dist < 3.0:
            return nearest
        return ""

    def _find_nearest_receptacle(self, objects: list) -> str:
        """找最近的容器"""
        nearest = None
        min_dist = float("inf")

        for obj in objects:
            if obj.get("receptacle", False):
                dist = obj.get("distance", float("inf"))
                if dist < min_dist:
                    min_dist = dist
                    nearest = obj.get("objectId")

        if nearest and min_dist < 3.0:
            return nearest
        return ""


# ──────────────────────────────────────────────────────────
# 规则版：用 VLM 输出文本 → 规则解析动作
# ──────────────────────────────────────────────────────────

class PromptedDiscreteAgent(CustomDiscreteAgent):
    """
    升级版：用任意 VLM（LLaVA, Qwen-VL 等）看图像+指令 → 输出动作文本 → 映射。

    适合快速验证，不需要微调。

    用法:
        agent = PromptedDiscreteAgent(
            name="RobotA",
            vlm=my_vlm,          # 任意 (image, text) → str 的模型
            system_prompt="Output format: [action, modifier]",
            instruction="go to the table and pick up the bowl",
        )
    """

    def __init__(
        self,
        name: str,
        vlm: Callable,
        system_prompt: str = "",
        instruction: str = "",
        device: str = "cuda:0",
    ):
        # vlm 就是模型，传给父类的 model 参数
        super().__init__(name=name, model=vlm, instruction=instruction, device=device)
        self.system_prompt = system_prompt or (
            "You are a robot in a house. "
            "Output exactly 2 words: [base_action] [modifier]. "
            "base_action: moveahead, moveback, rotateleft, rotateright, "
            "pickup, place, open, close. "
            "modifier: none, lefthand, righthand. "
            "Example: moveahead none"
        )

    def __call__(self, obs: dict) -> dict:
        if self.model is None:
            return {"action": "Pass"}

        img = obs.get("image")
        if img is None:
            return {"action": "Pass"}

        # VLM 调用：传入图像 + 组合提示
        prompt = f"{self.system_prompt}\nTask: {self.instruction}\nOutput:"
        try:
            # 假设 vlm 是 (image, text) → str
            text_output = self.model(img, prompt)
            # 解析成 [base, modifier]
            parts = text_output.strip().lower().replace(",", " ").split()
            base_action = parts[0] if parts else "pass"
            modifier = parts[1] if len(parts) > 1 else "none"
            output = [base_action, modifier]
        except Exception as e:
            print(f"[{self.name}] VLM 推理失败: {e}")
            return {"action": "Pass"}

        # 复用父类的映射逻辑
        return self._map_output(obs, output)

    def _map_output(self, obs: dict, output: list) -> dict:
        """解析输出并查表（复用父类逻辑）"""
        base_action = str(output[0]).lower().strip()
        modifier = str(output[1]).lower().strip() if len(output) > 1 else "none"
        objects = obs.get("objects", [])

        key = (base_action, modifier)
        if key in ACTION_MAP:
            action = dict(ACTION_MAP[key])
        else:
            fallback_key = (base_action, "none")
            if fallback_key in ACTION_MAP:
                action = dict(ACTION_MAP[fallback_key])
            else:
                print(f"[{self.name}] 无法映射: {output}")
                return {"action": "Pass"}

        # 自动填补 objectId
        if action["action"] in ("PickupObject", "OpenObject", "CloseObject",
                                "ToggleObjectOn", "ToggleObjectOff"):
            target = self._find_nearest_object(objects, action["action"])
            if target:
                action["objectId"] = target
            else:
                return {"action": "Pass"}

        action["_raw_output"] = output
        return action
