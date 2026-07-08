"""
VLA Agent for AI2-THOR
=======================
用 OpenVLA 驱动 AI2-THOR 机器人。

加载 openvla-7b（4-bit 量化，一张 3080 Ti 可跑），
输入 agent 视野 RGB 图像 + 自然语言指令，
输出映射为 AI2-THOR 动作并执行。

两种映射模式：
  - discrete (默认): VLA 7-DoF 连续动作 → MoveAhead / RotateLeft 等导航动作
  - continuous:       VLA 输出 → TeleportFull 直接设置绝对位姿
"""

import numpy as np
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
)
from PIL import Image


# ──────────────────────────────────────────────────────────
# Action Mapper — VLA 连续动作 → AI2-THOR 动作
# ──────────────────────────────────────────────────────────

class DiscreteActionMapper:
    """
    把 VLA 7-DoF 连续动作 (dx, dy, dz, roll, pitch, yaw, gripper)
    映射为 AI2-THOR 离散导航/操作动作。
    """

    # AI2-THOR 每步移动/旋转的物理量（用于计算连续位置增量模式）
    STEP_MOVE = 0.25       # MoveAhead 一步约 0.25 米
    STEP_ROTATE = 45.0     # RotateLeft/Right 一步 45 度
    STEP_LOOK = 15.0       # LookUp/Down 一步 15 度

    def __init__(self, move_threshold: float = 0.25, rot_threshold: float = 0.25):
        self.move_threshold = move_threshold
        self.rot_threshold = rot_threshold

    def map(self, vla_action: np.ndarray, current_pos: dict = None,
            current_rot: dict = None) -> dict:
        """
        将 VLA 输出的 7-DoF 动作映射为 AI2-THOR 动作字典。

        Args:
            vla_action: (7,) ndarray — (dx, dy, dz, roll, pitch, yaw, gripper)
            current_pos: 当前位置 {"x", "y", "z"}，用于 Teleport 模式
            current_rot: 当前旋转 {"x", "y", "z"}，用于 Teleport 模式

        Returns:
            AI2-THOR action dict
        """
        dx, dy, dz, roll, pitch, yaw, gripper = vla_action

        # ── 决策树 ──

        # 1) 横向移动（dx > 阈值）
        if abs(dx) > self.move_threshold:
            return {"action": "MoveRight" if dx > 0 else "MoveLeft"}

        # 2) 前后移动（dz > 阈值, 前正后负）
        if abs(dz) > self.move_threshold:
            return {"action": "MoveAhead" if dz > 0 else "MoveBack"}

        # 3) 水平旋转（yaw > 阈值）
        if abs(yaw) > self.rot_threshold:
            return {"action": "RotateRight" if yaw > 0 else "RotateLeft"}

        # 4) 俯仰（pitch > 阈值）
        if abs(pitch) > self.rot_threshold:
            return {"action": "LookDown" if pitch > 0 else "LookUp"}

        # 5) 夹爪关闭 → 尝试抓取
        if gripper < 0.5:
            return {"action": "Pass"}  # 此处由外部逻辑补充 PickupObject

        # 6) 什么都不做
        return {"action": "Pass"}


class ContinuousActionMapper:
    """
    更激进的映射模式:
    将 VLA 输出的连续增量直接通过 TeleportFull 施加到机器人位姿上。
    """

    # 缩放系数：VLA 输出的 [-1,1] 范围 → 实际物理量
    POS_SCALE = 0.5     # 位置缩放（米）
    ROT_SCALE = 30.0    # 旋转缩放（度）

    def map(self, vla_action: np.ndarray, current_pos: dict,
            current_rot: dict) -> dict:
        """
        计算目标位姿并用 TeleportFull 直接设置。
        """
        dx, dy, dz, roll, pitch, yaw, gripper = vla_action

        new_pos = {
            "x": current_pos["x"] + dx * self.POS_SCALE,
            "y": current_pos["y"] + dy * self.POS_SCALE * 0.5,  # 垂直移动减半
            "z": current_pos["z"] + dz * self.POS_SCALE,
        }
        new_rot = {
            "x": current_rot["x"] + roll * self.ROT_SCALE,
            "y": current_rot["y"] + yaw * self.ROT_SCALE,
            "z": current_rot["z"] + pitch * self.ROT_SCALE,
        }

        return {
            "action": "TeleportFull",
            "position": new_pos,
            "rotation": new_rot,
            "horizon": 0.0,
            "standing": True,
        }


# ──────────────────────────────────────────────────────────
# VLA Agent 类
# ──────────────────────────────────────────────────────────

class VLAgent:
    """
    VLA Agent — 用 OpenVLA 驱动一个 AI2-THOR agent。

    Usage:
        agent = VLAgent("RobotA", instruction="go to the table")
        action = agent(observation)   # → AI2-THOR action dict
    """

    def __init__(
        self,
        name: str,
        instruction: str,
        model_id: str = "openvla/openvla-7b",
        load_in_4bit: bool = True,
        device: str = "cuda:0",
        mode: str = "discrete",
        unnorm_key: Optional[str] = None,
    ):
        """
        Args:
            name: Agent 名称
            instruction: 自然语言指令，如 "go to the table and pick up the bowl"
            model_id: HuggingFace 模型 ID（默认 openvla/openvla-7b）
            load_in_4bit: 是否加载 4-bit 量化（3080 Ti 12GB 建议开启）
            device: 推理设备
            mode: 动作映射模式 — "discrete" 或 "continuous"
            unnorm_key: 反归一化 key（OpenVLA 训练数据集）
        """
        self.name = name
        self.instruction = instruction
        self.device = device
        self.unnorm_key = unnorm_key
        self.mode = mode

        # 选择映射器
        if mode == "continuous":
            self.mapper = ContinuousActionMapper()
        else:
            self.mapper = DiscreteActionMapper()

        # ── 加载模型 ──
        print(f"[{name}] Loading OpenVLA ({model_id}) on {device} "
              f"(4-bit={load_in_4bit}) ...")

        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True,
        )
        self.vla = AutoModelForVision2Seq.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=device if not load_in_4bit else None,
        )

        if not load_in_4bit:
            self.vla = self.vla.to(device)

        print(f"[{name}] OpenVLA loaded ✓  |  Mode: {mode}  |  "
              f"Instr: \"{instruction}\"")

    def __call__(self, obs: dict) -> dict:
        """
        Agent 接口：observation → AI2-THOR action dict

        Args:
            obs: MultiAgentEnv.observe_agent() 返回的观测

        Returns:
            AI2-THOR action dict
        """
        # ── 获取图像 ──
        img = obs.get("image")
        if img is None:
            print(f"[{self.name}] WARNING: No image in observation")
            return {"action": "Pass"}

        # ── 当前位姿（用于 continuous 模式） ──
        current_pos = obs.get("agent_position")
        current_rot = obs.get("camera_position")

        # ── VLA 推理 ──
        try:
            pil_img = Image.fromarray(img)
            prompt = f"In: What action should the robot take to {self.instruction}?\nOut:"

            inputs = self.processor(prompt, pil_img).to(
                self.device, dtype=torch.bfloat16)

            with torch.inference_mode():
                vla_action = self.vla.predict_action(
                    **inputs,
                    unnorm_key=self.unnorm_key,
                    do_sample=False,
                )

            # vla_action shape: (7,)
            action = self.mapper.map(
                vla_action,
                current_pos=current_pos,
                current_rot=current_rot,
            )

            # 附加原始 VLA 输出（调试用）
            action["_vla_raw"] = vla_action.tolist()
            action["_vla_instr"] = self.instruction

            return action

        except torch.cuda.OutOfMemoryError:
            print(f"[{self.name}] CUDA OOM — 清空缓存并返回 Pass")
            torch.cuda.empty_cache()
            return {"action": "Pass"}
        except Exception as e:
            print(f"[{self.name}] VLA inference error: {e}")
            import traceback
            traceback.print_exc()
            return {"action": "Pass"}
