# Classification Training Agent

An autonomous training workflow for **single-head and multi-head image classification**, built with **PyTorch** and **LangGraph**.

The agent validates datasets, trains and evaluates models, records experiments, and uses an LLM to analyze results and propose improved configurations for later attempts.

The workflow is not tied to a specific architecture. ResNet is one supported example, but the framework can be extended to other compatible classifiers.

> The first experiment always uses the configured baseline.  
> LLM-guided tuning starts after baseline results are available.

---

## Features

- Single-head and multi-head classification
- Configurable model architectures
- Dataset validation before training
- Single-GPU and multi-GPU training
- Automatic experiment tracking
- LLM-guided hyperparameter tuning
- Validation and test evaluation
- Best-model selection
- Best and last checkpoint saving
- Optional ONNX export
- Session-level experiment summaries

---

## Requirements

- Python 3.12+
- Linux
- NVIDIA GPU with CUDA-enabled PyTorch
- NCCL for distributed training
- CSV datasets and image files
- Compatible pretrained weights when required
- API key for the configured hosted LLM

Dataset validation can run without GPU or LLM credentials.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/NoraAlqasem/classification_agent.git
cd classification_agent
```

Create a Conda environment with Python 3.12:

```bash
conda create -n classification-agent python=3.12 -y
```

Activate the environment:

```bash
conda activate classification-agent
```

Upgrade pip and install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The current dependencies use:

```text
PyTorch      2.6.0
Torchvision  0.21.0
CUDA         12.4
```

If your system uses a different CUDA runtime, update the PyTorch/Torchvision installation accordingly.

---

## Configuration

Create your environment file:

```bash
cp .env.example .env
```

Configure your:

- API key
- dataset paths
- data root
- classification heads
- model settings

Example:

```env
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o

TRAIN_CSV=...
VAL_CSV=...
TEST_CSV=...

DATA_ROOT=...
HEADS="make vehicle_type"
```

For a single-head task:

```env
HEADS="make"
```

---

## Dataset Format

The dataset must provide `train.csv`, `val.csv`, and `test.csv`.

Each CSV must include:

- `root_folder`
- `image_name`
- one label column for each configured classification head

Additional columns are allowed.

---

## Validate the Dataset

```bash
python main.py \
  --validate-only \
  --arch resnet18 \
  --weights weights/pretrained/resnet18.pth
```

Validation checks dataset structure, labels, image paths, and class consistency before training begins.

---

## Start Training

### 1. Single-Head Classification

Example training a classifier with one prediction head:

```bash
python main.py \
  --arch resnet18 \
  --weights weights/pretrained/resnet18.pth \
  --heads make \
  --nproc-per-node 1 \
  --target-metric make_acc \
  --target-value 0.85 \
  --max-attempts 5 \
  --export onnx
```

---

### 2. Multi-Head Classification

Example training a classifier with multiple prediction heads:

```bash
python main.py \
  --arch resnet18 \
  --weights weights/pretrained/resnet18.pth \
  --heads make vehicle_type \
  --nproc-per-node 1 \
  --target-metric mean_acc \
  --target-value 0.85 \
  --max-attempts 5 \
  --export onnx
```

View all available options with:

```bash
python main.py --help
```

---

## Project Structure

| Path | Purpose |
|---|---|
| `main.py` | CLI and session execution |
| `agent/` | LangGraph workflow and LLM logic |
| `training/` | Model training, evaluation, and export |
| `tools/` | Configuration, metrics, and utility functions |
| `memory/` | Experiment and session storage |
| `config/defaults.yaml` | Baseline configuration |
| `docs/AGENT_PIPELINE.md` | Full workflow and implementation documentation |

---

## Documentation

For detailed information about the agent workflow, dataset validation, configuration, tuning logic, evaluation, outputs, ONNX export, and resume behavior, see:

[`docs/AGENT_PIPELINE.md`](docs/AGENT_PIPELINE.md)