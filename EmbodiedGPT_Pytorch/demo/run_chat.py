from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOCAL_HF_CACHE = REPO_ROOT.parent / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(LOCAL_HF_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(LOCAL_HF_CACHE / "transformers"))

LOCAL_QWEN35_MODEL = REPO_ROOT.parent / "models" / "Qwen3.5-4B"
DEFAULT_QWEN35_MODEL = str(LOCAL_QWEN35_MODEL) if LOCAL_QWEN35_MODEL.is_dir() else "Qwen/Qwen3.5-4B"
from demo.plan_media import parse_semantic_planning_output, semantic_planning_prompt

IMAGE_EXTENSIONS = (".bmp", ".dib", ".png", ".jpg", ".jpeg", ".pbm", ".pgm", ".ppm", ".tif", ".tiff")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".wmv", ".iso", ".webm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EmbodiedGPT interactive chat.")
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN35_MODEL)
    parser.add_argument("--qwen-device-map", default="auto")
    parser.add_argument("--qwen-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--semantic-plan", action="store_true", help="Wrap visual questions as semantic action draft prompts that avoid objectId and require later grounding.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import torch
    from demo.qwen35_backend import Qwen35Backend, Qwen35Config
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    qwen = Qwen35Backend(
        Qwen35Config(
            model_name=args.qwen_model,
            device=device,
            device_map=args.qwen_device_map,
            torch_dtype=args.qwen_dtype,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    )
    media_path = None
    image_state = False
    video_state = False
    print("Qwen3.5 visual chat ready. Type an image/video path, a prompt, clear, or stop.")

    while True:
        query = input("\n> ").strip()
        if query.lower() == "stop":
            break
        if query.lower() == "clear" or query == "":
            media_path = None
            image_state = False
            video_state = False
            print("Conversation cleared.")
            continue

        lower_query = query.lower()
        if lower_query.endswith(IMAGE_EXTENSIONS) and os.path.exists(query):
            print("Image received.")
            media_path = query
            image_state = True
            video_state = False
            continue

        if lower_query.endswith(VIDEO_EXTENSIONS) and os.path.exists(query):
            print("Video received.")
            media_path = query
            image_state = False
            video_state = True
            continue

        modal_type = "image" if image_state else "video" if video_state else "text"
        prompt = semantic_planning_prompt(modal_type, query) if args.semantic_plan and modal_type != "text" else query

        if modal_type == "text" or media_path is None:
            print("Agent:\nPlease provide an image or video path before asking a visual question.")
            continue
        try:
            outputs = qwen.generate(media_path, modal_type, prompt)
        except Exception as exc:
            print(f"Agent:\nQwen backend failed: {exc}")
            continue
        if args.semantic_plan:
            try:
                outputs = json.dumps(parse_semantic_planning_output(outputs), ensure_ascii=False)
            except ValueError as exc:
                outputs = f"Semantic plan failed validation: {exc}\nRaw model output:\n{outputs}"
        print(f"Agent:\n{outputs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
