# Installation

The active runtime requires Qwen3.5-compatible PyTorch and the latest Transformers implementation.

```bash
conda create -n qwen35 python=3.11 -y
conda activate qwen35

# Install the PyTorch build matching your CUDA driver first.
pip install torch torchvision

pip install -r requirements.txt
```

The repository automatically uses ../models/Qwen3.5-4B when that directory exists. Otherwise it falls back to Qwen/Qwen3.5-4B from Hugging Face. Override either choice with --qwen-model.

Verify the command-line entry points without loading the model:

```bash
python demo/plan_media.py --help
python demo/run_chat.py --help
python demo/auto_scene_actions.py --help
```

The robohusky package is retained only for historical training/data compatibility and is not used by the active Qwen inference path.
