# ResNet Training Agent

An autonomous training workflow for multi-head image classification, built with
PyTorch and LangGraph. A shared ResNet backbone predicts multiple labels per image;
a hosted LLM analyzes measured results and proposes settings for later experiments.

The agent validates CSV datasets, trains and evaluates checkpoints, records each
experiment, and optionally exports the best model to ONNX. The first experiment
uses the YAML baseline; LLM tuning begins after its results are available.

## Requirements

- Python 3.12 or newer.
- Linux with an NVIDIA GPU and a CUDA-enabled PyTorch build with NCCL for training,
  including a single-GPU run.
- CSV datasets, image files, and matching pretrained ResNet backbone weights.
- An API key for the configured hosted LLM for full training sessions.

Dataset validation and the test suite can run without a GPU or live API calls.
Datasets, pretrained weights, and training outputs are excluded from Git.

## Installation

```bash
git clone https://github.com/NoraAlqasem/classification_agent.git
cd classification_agent
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For the pinned CUDA 12.4 profile:

```bash
python -m pip install -r requirements-cu124.txt
```

For another CUDA runtime, install a compatible PyTorch/Torchvision pair using the
[PyTorch installation guide](https://pytorch.org/get-started/locally/), then install
`requirements.txt`. The CUDA 12.4 profile pins PyTorch 2.6.0 and Torchvision 0.21.0.
The existing `uv.lock` resolves a different runtime on Linux; use the pip profile
above when you need CUDA 12.4.

Check your training environment:

```bash
python - <<'PYTHON'
import torch
print("PyTorch:", torch.__version__, "Bundled CUDA:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.distributed.is_nccl_available(), "NCCL is unavailable"
print("GPU:", torch.cuda.get_device_name(0))
PYTHON
```

## Configuration

```bash
cp .env.example .env
```

Edit `.env` with your API key, dataset paths, and label columns. `OPENAI_MODEL`
defaults to `gpt-4o` in the code. For an OpenAI-compatible provider, set
`OPENAI_API_BASE` to its endpoint and `OPENAI_MODEL` to a model available to your
account. Exported shell variables take precedence over `.env`; explicit CLI dataset
arguments take precedence over environment settings.

Training and inference run locally. Dataset summaries, configurations, metrics,
class labels, and experiment evidence are sent to the configured LLM. Image pixels
and model tensors are not included in prompts. Keep `.env` private.

The baseline is in [config/defaults.yaml](config/defaults.yaml). Current defaults
are `resnet50`, two epochs, a batch size of 32 per GPU, a target mean accuracy of
`0.99`, and up to 40 attempts. Override the target and attempt limit for your task.
The LLM can tune supported hyperparameters and choose compatible local
`resnet18`, `resnet34`, or `resnet50` weights for later attempts.

## Dataset format

Provide `train.csv`, `val.csv`, and, for the default test decision mode, `test.csv`.
Each row needs an image path and a label for every requested head:

```csv
root_folder,image_name,make,vehicle_type
train,img001.jpg,Toyota,suv
train,img002.jpg,Ford,sedan
```

Relative `root_folder` values are resolved against `DATA_ROOT` or `--data-root`.
Absolute paths are used directly. Validation and test labels must belong to the
training class vocabulary. Missing labels or image files fail validation.

An `image_path` column is also supported, for example
`train/img001.jpg`. CompCars CSVs with a `type` label are exposed as
`vehicle_type`; use `HEADS="make vehicle_type"`.

Set the individual CSV paths in `.env`, or use `--dataset /path/to/dataset` for a
folder containing the required splits. See the
[pipeline guide](docs/AGENT_PIPELINE.md) for input precedence and label restrictions.

## Pretrained weights

Supply a raw ImageNet backbone `state_dict` matching your chosen architecture:

```text
weights/pretrained/resnet18.pth
weights/pretrained/resnet34.pth
weights/pretrained/resnet50.pth
```

Only the file for the selected architecture is required. Additional compatible
files in this directory become choices for later LLM tuning. Use `--weights` to
select a different local file. A multi-head training checkpoint such as
`best_model.pth` cannot be used as initial backbone weights.

## Validate and train

With dataset paths and heads configured in `.env`, validate inputs first:

```bash
python main.py --validate-only --arch resnet18 \
  --weights weights/pretrained/resnet18.pth
```

Validation requires the selected weights file to exist, but does not deserialize
it or require CUDA or LLM credentials.

Start a training session:

```bash
python main.py \
  --arch resnet18 \
  --weights weights/pretrained/resnet18.pth \
  --nproc-per-node 1 \
  --target-metric mean_acc \
  --target-value 0.85 \
  --max-attempts 5 \
  --export onnx
```

Use `python main.py --help` for all options. Dataset arguments can be supplied
explicitly with `--train-csv`, `--val-csv`, `--test-csv`, `--data-root`, and
`--heads make vehicle_type`.

Launch `main.py` once; the agent starts `torchrun` automatically. Set
`--nproc-per-node 2` for two visible GPUs. Batch size and DataLoader workers are
per GPU. Use `--export none` to keep PyTorch checkpoints without ONNX export.

`mean_acc` is the unweighted mean head accuracy; `HEAD_acc` targets one head.
Targets use fractions from 0 to 1. Validation selects the checkpoint within each
training attempt; the decision split controls best-attempt selection and stopping.

By default, test scores guide tuning. That split therefore serves as a tuning
set; keep a separate untouched split for final evaluation. Use
`--decision-split val` to guide tuning with validation scores instead.

## Outputs

Each invocation creates the next numbered session under `runs/resnet/`:

```text
runs/resnet/
├── knowledge_base.json
└── session_1/
    ├── session_summary.md
    ├── llm_responses.json
    ├── report_<timestamp>.md
    ├── report_summary.json
    └── exp_01/
        ├── exp_1.json
        ├── weights/           # Checkpoints, class mappings, optional ONNX + metadata
        ├── train_evaluation/  # Configuration, metrics, plots, and training log
        └── test_evaluation/   # Test summaries and per-head class diagnostics
```

Later attempts use `exp_02`, `exp_03`, and so on. Session summaries compare
experiments and identify the best measured result. Reaching the attempt limit can
complete a session without reaching the accuracy target; inspect the report's
target status.

ONNX exports provide one logits tensor per head with a dynamic batch dimension.
The adjacent metadata JSON describes class ordering and image preprocessing.
Apply softmax separately to each head when probabilities are needed.

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite covers input validation, graph execution, retry behavior, experiment
records, checkpoint evaluation, and ONNX export. API calls and training subprocesses
are mocked; model and export checks run on CPU. For numerical ONNX checks, install
`onnxruntime` or use the suite's ONNX ReferenceEvaluator fallback. These checks do
not replace a real CUDA/NCCL training run.

## Project layout

| Path | Purpose |
| --- | --- |
| [main.py](main.py) | CLI and session execution. |
| [agent/](agent/) | LangGraph workflow, nodes, and LLM handling. |
| [training/](training/) | DDP trainer, checkpoint evaluation, and ONNX export. |
| [tools/](tools/) | Configuration, metrics, weights, and artifact helpers. |
| [memory/](memory/) | Experiment storage and session paths. |
| [config/defaults.yaml](config/defaults.yaml) | Baseline agent and trainer settings. |
| [tests/](tests/) | Automated regression and integration checks. |
| [docs/AGENT_PIPELINE.md](docs/AGENT_PIPELINE.md) | Detailed workflow, tuning rules, output formats, and standalone resume. |
