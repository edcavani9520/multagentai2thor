# EmbodiedGPT Qwen3.5 Runtime

This repository uses Qwen3.5-4B as its only active inference backend. The runtime supports image and video understanding, semantic embodied planning, AI2-THOR action generation, simulator object grounding, and HTTP action dispatch.

## Installation

See [INSTALLATION.md](INSTALLATION.md). The local model is discovered automatically at ../models/Qwen3.5-4B; use --qwen-model to override it.

## Usage

Generate a semantic plan from an image:

```bash
python demo/plan_media.py --media scene.jpg --task "Open the drawer"
```

Analyze a video or answer a temporal question:

```bash
python demo/plan_media.py --media action.mp4 --task "Put the cup on the table"
python demo/plan_media.py --media action.mp4 --question "What happens after the cup is picked up?"
```

Generate an executable native plan:

    python demo/plan_media.py --media scene.jpg --task "Put the cup on the table" --plan-mode native --plan-only

All model planning outputs store action objects in the top-level plan array. Native output uses task plus plan; the top-level actions field exists only in the HTTP payload sent to the AI2-THOR execution endpoint.

Start visual chat:

```bash
python demo/run_chat.py
```

Run the closed-loop AI2-THOR flow (probe scene, plan, ground object IDs, and dispatch actions):

```bash
./auto_scene_actions.sh --task "Put the apple on the countertop"
```

Use --dry-run to inspect the grounded payload without executing it. Run each Python entry point with --help for model, device, dtype, generation, and action-dispatch options.

The runtime is a qualitative planning and simulator integration tool. It does not include a policy network, real-robot control, or the paper evaluation suite.

### Legacy training utilities

This repo can be used in conjunction with PyTorch's `Dataset` and `DataLoader` for training models on heterogeneous
data. Here's a brief overview of the classes and their functionalities:

### BaseDataset

The `BaseDataset` class extends PyTorch's `Dataset` and is designed to handle different media types (images, videos, and
text). It includes a transformation process to standardize the input data and a processor to handle the data specific to
the task.

#### Example

```python
from robohusky.base_dataset_uni import BaseDataset

# Initialize the dataset with the required parameters
dataset = BaseDataset(
    dataset,  # Your dataset here
    processor,  # Your processor here
    image_path="path/to/images",
    input_size=224,
    num_segments=8,
    norm_type="openai",
    media_type="image"
)

# Use the dataset with a PyTorch DataLoader
from torch.utils.data import DataLoader

data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### WeightedConcatDataset

The `WeightedConcatDataset` class extends PyTorch's `ConcatDataset` and allows for the creation of a unified dataset by
concatenating multiple datasets with specified weights.

#### Example

```python
from robohusky.base_dataset_uni import WeightedConcatDataset

# Assume we have multiple datasets for different tasks
dataset1 = BaseDataset(...)
dataset2 = BaseDataset(...)
dataset3 = BaseDataset(...)

# Define the weights for each dataset
weights = [0.5, 0.3, 0.2]

# Create a weighted concatenated dataset
weighted_dataset = WeightedConcatDataset([dataset1, dataset2, dataset3], weights=weights)

# Use the weighted dataset with a PyTorch DataLoader
data_loader = DataLoader(weighted_dataset, batch_size=32, shuffle=True)
```

## Customization

The package is designed to be flexible and customizable. You can implement your own transformation and processing logic
by subclassing `BaseDataset` and overriding the necessary methods.

## 🎫 License

This project is released under the [Apache 2.0 license](LICENSE).

## 🖊️ Citation

If you find this project useful in your research, please consider cite:
```bibtex
@article{mu2024embodiedgpt,
  title={Embodiedgpt: Vision-language pre-training via embodied chain of thought},
  author={Mu, Yao and Zhang, Qinglong and Hu, Mengkang and Wang, Wenhai and Ding, Mingyu and Jin, Jun and Wang, Bin and Dai, Jifeng and Qiao, Yu and Luo, Ping},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```
