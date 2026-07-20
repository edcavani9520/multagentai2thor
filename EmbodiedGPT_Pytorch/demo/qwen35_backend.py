from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import torch


QWEN35_HUB_MODEL = "Qwen/Qwen3.5-4B"
LOCAL_QWEN35_MODEL = Path(__file__).resolve().parents[2] / "models" / "Qwen3.5-4B"
DEFAULT_QWEN35_MODEL = str(LOCAL_QWEN35_MODEL) if LOCAL_QWEN35_MODEL.is_dir() else QWEN35_HUB_MODEL


def _as_media_source(path_or_url: Union[str, Path]) -> str:
    value = str(path_or_url)
    if value.startswith(("http://", "https://")):
        return value
    return str(Path(value).expanduser().resolve())


def _move_to_device(value: Any, device: str) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


@dataclass
class Qwen35Config:
    model_name: str = DEFAULT_QWEN35_MODEL
    device: str = "auto"
    device_map: str = "auto"
    torch_dtype: str = "auto"
    max_new_tokens: int = 512
    temperature: float = 0.0
    trust_remote_code: bool = True


class Qwen35Backend:
    """Qwen3.5 vision-language backend for semantic robot perception.

    It exposes a small, stable interface shared by the command-line demos.
    """

    def __init__(self, config: Optional[Qwen35Config] = None):
        self.config = config or Qwen35Config()
        self.processor = None
        self.model = None

    def load(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        try:
            from transformers import AutoProcessor
            import transformers
        except ImportError as exc:
            raise RuntimeError(
                "Qwen backend requires transformers in the active Python environment. "
                "Create a separate Qwen environment and install the versions requested "
                "by the Qwen/Qwen3.5-4B model card."
            ) from exc

        model_class = self._select_model_class(transformers)
        dtype = self._torch_dtype()
        kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
        }
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        if self.config.device_map:
            kwargs["device_map"] = self.config.device_map

        self.processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        self.model = model_class.from_pretrained(self.config.model_name, **kwargs).eval()
        if self.config.device != "auto" and not self.config.device_map:
            self.model.to(self.config.device)

    def generate(self, media_path: Union[str, Path], media_type: str, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    self._media_item(media_path, media_type),
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self.generate_messages(messages)

    def generate_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        return self.generate_messages(messages, tools=tools, require_tool_support=True)

    def generate_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        require_tool_support: bool = False,
        deterministic: bool = False,
    ) -> str:
        self.load()
        assert self.model is not None
        assert self.processor is not None

        inputs = self._build_inputs(messages, tools=tools, require_tool_support=require_tool_support)
        inputs = _move_to_device(inputs, self._input_device())

        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
        }
        if deterministic:
            generation_kwargs["do_sample"] = False
        elif self.config.temperature > 0:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.config.temperature,
                }
            )
        else:
            generation_kwargs["do_sample"] = False

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)

        input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else None
        if input_ids is not None:
            generated_ids = generated_ids[:, input_ids.shape[-1] :]

        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def _media_item(self, media_path: Union[str, Path], media_type: str) -> dict[str, Any]:
        media_source = _as_media_source(media_path)
        if media_type == "image":
            return {"type": "image", "image": media_source}
        if media_type != "video":
            raise ValueError(f"Qwen backend supports image/video media, got: {media_type}")
        return {"type": "video", "video": media_source}

    def _build_inputs(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        require_tool_support: bool = False,
    ) -> dict[str, Any]:
        processor = self.processor
        assert processor is not None

        has_video = False
        for message in messages:
            content = message.get("content", [])
            if not isinstance(content, list):
                continue
            if any(isinstance(item, dict) and item.get("type") == "video" for item in content):
                has_video = True
                break

        if not has_video:
            chat_template_kwargs: dict[str, Any] = {
                "add_generation_prompt": True,
                "enable_thinking": False,
                "tokenize": True,
                "return_dict": True,
                "return_tensors": "pt",
            }
            if tools is not None:
                chat_template_kwargs["tools"] = tools
            try:
                return processor.apply_chat_template(messages, **chat_template_kwargs)
            except TypeError as exc:
                if require_tool_support or tools is not None:
                    raise RuntimeError(
                        "Qwen processor.apply_chat_template does not support native tool calling; "
                        "upgrade the Qwen/transformers environment."
                    ) from exc

        text_template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        if tools is not None:
            text_template_kwargs["tools"] = tools
        try:
            text = processor.apply_chat_template(messages, **text_template_kwargs)
        except TypeError as exc:
            if require_tool_support or tools is not None:
                raise RuntimeError(
                    "Qwen processor.apply_chat_template does not support native tool calling; "
                    "upgrade the Qwen/transformers environment."
                ) from exc
            raise
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise RuntimeError(
                "Qwen backend could not process local visual inputs with the current processor. "
                "Install qwen-vl-utils in the Qwen environment or use a transformers version "
                "whose processor.apply_chat_template can tokenize visual messages directly."
            ) from exc

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
        )
        fps = video_kwargs.get("fps")
        if isinstance(fps, list) and len(fps) == 1:
            video_kwargs["fps"] = fps[0]

        return processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )

    def _input_device(self) -> str:
        if self.config.device != "auto":
            return self.config.device
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _torch_dtype(self) -> Optional[torch.dtype]:
        if self.config.torch_dtype == "auto":
            return None
        if self.config.torch_dtype == "bfloat16":
            return torch.bfloat16
        if self.config.torch_dtype == "float16":
            return torch.float16
        if self.config.torch_dtype == "float32":
            return torch.float32
        raise ValueError(f"unsupported torch dtype: {self.config.torch_dtype}")

    @staticmethod
    def _select_model_class(transformers_module: Any) -> Any:
        for name in (
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForCausalLM",
        ):
            model_class = getattr(transformers_module, name, None)
            if model_class is not None:
                return model_class
        raise RuntimeError(
            "No compatible AutoModel class is available. Upgrade transformers in the Qwen environment."
        )
