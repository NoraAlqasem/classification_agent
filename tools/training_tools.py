from __future__ import annotations

import json
import math
from pathlib import Path

from langchain_core.tools import tool
from training.artifact_paths import find_training_file, train_evaluation_dir


@tool
def check_gpu_availability() -> str:
    """
    Check whether a CUDA GPU is available for training.
    Returns JSON with 'cuda_available', 'device_count', and device names.
    """
    try:
        import torch
        available = torch.cuda.is_available()
        count = torch.cuda.device_count() if available else 0
        devices = [torch.cuda.get_device_name(i) for i in range(count)]
        return json.dumps({
            "cuda_available": available,
            "device_count":   count,
            "devices":        devices,
        })
    except ImportError:
        return json.dumps({"cuda_available": False, "error": "torch not installed"})


@tool
def estimate_training_time(
    num_images: int,
    epochs: int,
    batch_size: int,
    img_size: list[int],
    arch: str = "resnet18",
    nproc_per_node: int = 1,
) -> str:
    """
    Rough estimate of training time based on dataset size and config.
    Returns JSON with 'estimated_minutes' and 'notes'.
    """
    if (
        num_images < 1 or epochs < 1 or batch_size < 1 or nproc_per_node < 1
        or len(img_size) != 2 or any(size < 1 for size in img_size)
        or arch not in {"resnet18", "resnet34", "resnet50"}
    ):
        return json.dumps({"error": "Use positive dataset/epoch/batch/GPU counts, image [H, W], and a supported ResNet architecture."})
    steps_per_epoch = math.ceil(num_images / (batch_size * nproc_per_node))
    resolution_scale = (img_size[0] * img_size[1]) / (224 * 224)
    ms_per_step = {"resnet18": 50, "resnet34": 80, "resnet50": 120}[arch] * resolution_scale
    total_minutes = round((steps_per_epoch * epochs * ms_per_step) / 60_000, 1)
    return json.dumps({
        "estimated_minutes": total_minutes,
        "steps_per_epoch":   steps_per_epoch,
        "notes": "Rough CUDA estimate only; storage, GPU model, validation, and DDP overhead can dominate. Batch size is per GPU.",
    })


@tool
def read_training_log(run_dir: str, last_n_lines: int = 50) -> str:
    """
    Read the last N lines of the torchrun training log if present.
    """
    if last_n_lines < 1:
        return json.dumps({"error": "last_n_lines must be positive"})
    preferred = find_training_file(run_dir, "training.log")
    log_candidates = [preferred] if preferred.is_file() else [
        *sorted(train_evaluation_dir(run_dir).glob("*.log")),
        *sorted(Path(run_dir).glob("*.log")),
    ]
    if not log_candidates:
        return json.dumps({"note": "No log file found in run directory.", "run_dir": run_dir})

    log_path = log_candidates[0]
    lines = log_path.read_text().splitlines()[-last_n_lines:]
    return json.dumps({"log_file": str(log_path), "lines": lines})
