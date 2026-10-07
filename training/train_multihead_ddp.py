import os
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Iterator

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, models

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__:
    from .data import read_dataset_csv
    from .artifact_paths import checkpoint_run_dir, prepare_new_run_directory, train_evaluation_dir, weights_dir
else:
    # torchrun executes this file directly, with training/ on sys.path.
    from data import read_dataset_csv
    from artifact_paths import checkpoint_run_dir, prepare_new_run_directory, train_evaluation_dir, weights_dir


DEFAULT_ALPHA = 1.0


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction="none")

        if alpha is None:
            self.alpha = None
        else:
            self.register_buffer("alpha", alpha.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        loss = (1.0 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            loss = self.alpha[targets] * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class MultiHeadVehicleDataset(Dataset):
    """
    Expected CSV columns:
        root_folder,image_name,<head_1>,<head_2>,...

    Example:
        root_folder,image_name,make,vehicle_type,color
        /data/train,img001.jpg,Toyota,suv,white
    """

    def __init__(
        self,
        csv_path: str,
        heads: List[str],
        class_to_idx_by_head: Dict[str, Dict[str, int]],
        transform=None,
        data_root: Optional[str] = None,
    ):
        self.csv_path = csv_path
        self.df = read_dataset_csv(csv_path, heads)
        self.heads = heads
        self.class_to_idx_by_head = class_to_idx_by_head
        self.transform = transform
        self.data_root = data_root

        required_columns = {"root_folder", "image_name", *heads}
        missing_columns = sorted(required_columns - set(self.df.columns))
        if missing_columns:
            raise ValueError(
                f"{csv_path} is missing required columns: {missing_columns}"
            )

        # First version intentionally requires a complete label for every head.
        for head in heads:
            missing_mask = self.df[head].isna() | (self.df[head].astype(str).str.strip() == "")
            if missing_mask.any():
                bad_rows = self.df.index[missing_mask].tolist()[:10]
                raise ValueError(
                    f"{csv_path}: head '{head}' has missing labels. "
                    f"Example row indices: {bad_rows}. "
                    "This script requires every image to have every requested head label."
                )

            self.df[head] = self.df[head].astype(str)

            unknown = sorted(
                set(self.df[head].unique())
                - set(self.class_to_idx_by_head[head].keys())
            )
            if unknown:
                raise ValueError(
                    f"{csv_path}: head '{head}' contains labels not present in the "
                    f"training mapping: {unknown}"
                )

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, root_folder: str, image_name: str) -> str:
        root_folder = str(root_folder)
        image_name = str(image_name)

        if os.path.isabs(root_folder):
            base = root_folder
        elif self.data_root:
            base = os.path.join(self.data_root, root_folder)
        else:
            base = root_folder

        return os.path.join(base, image_name)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = self._resolve_image_path(row["root_folder"], row["image_name"])

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to read image: {image_path}") from exc

        if self.transform is not None:
            image = self.transform(image)

        targets = {
            head: torch.tensor(
                self.class_to_idx_by_head[head][row[head]],
                dtype=torch.long,
            )
            for head in self.heads
        }

        return image, targets


class DistributedEvalSampler(Sampler[int]):
    """
    Evaluation sampler that does not pad/duplicate samples.

    torch.utils.data.DistributedSampler pads when dataset size is not divisible by
    world size. That is desirable for synchronized training steps, but it can
    slightly bias validation metrics. This sampler shards validation indices
    without duplication.
    """

    def __init__(self, dataset: Dataset, num_replicas: int, rank: int):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        if self.rank >= len(self.dataset):
            return 0
        return (len(self.dataset) - 1 - self.rank) // self.num_replicas + 1


class MultiHeadResNet(nn.Module):
    def __init__(
        self,
        arch: str,
        weights_path: Optional[str],
        num_classes_by_head: Dict[str, int],
    ):
        super().__init__()

        if arch not in {"resnet18", "resnet34", "resnet50"}:
            raise ValueError(f"Unsupported arch: {arch}")

        backbone_fn = getattr(models, arch)
        backbone = backbone_fn(weights=None)

        if weights_path is not None:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            backbone.load_state_dict(state, strict=True)

        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.heads = nn.ModuleDict(
            {
                head: nn.Linear(feature_dim, num_classes)
                for head, num_classes in num_classes_by_head.items()
            }
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        return {head: classifier(features) for head, classifier in self.heads.items()}


def build_class_mappings(
    train_csv: str,
    val_csv: str,
    heads: List[str],
) -> Tuple[
    Dict[str, List[str]],
    Dict[str, Dict[str, int]],
    Dict[str, int],
]:
    train_df = read_dataset_csv(train_csv, heads)
    val_df = read_dataset_csv(val_csv, heads)

    required = {"root_folder", "image_name", *heads}

    for split_name, df, path in [
        ("Train", train_df, train_csv),
        ("Validation", val_df, val_csv),
    ]:
        missing_columns = sorted(required - set(df.columns))
        if missing_columns:
            raise ValueError(
                f"{split_name} CSV {path} is missing required columns: {missing_columns}"
            )

    classes_by_head: Dict[str, List[str]] = {}
    class_to_idx_by_head: Dict[str, Dict[str, int]] = {}
    num_classes_by_head: Dict[str, int] = {}

    for head in heads:
        if train_df[head].isna().any():
            raise ValueError(f"Training CSV has missing values in head '{head}'.")
        if val_df[head].isna().any():
            raise ValueError(f"Validation CSV has missing values in head '{head}'.")

        train_labels = train_df[head].astype(str)
        val_labels = val_df[head].astype(str)

        train_classes = sorted(train_labels.unique().tolist())
        val_classes = set(val_labels.unique().tolist())

        unknown_val_classes = sorted(val_classes - set(train_classes))
        if unknown_val_classes:
            raise ValueError(
                f"Validation head '{head}' contains classes not present in Train: "
                f"{unknown_val_classes}"
            )

        mapping = {name: idx for idx, name in enumerate(train_classes)}

        classes_by_head[head] = train_classes
        class_to_idx_by_head[head] = mapping
        num_classes_by_head[head] = len(train_classes)

    return classes_by_head, class_to_idx_by_head, num_classes_by_head


def load_alpha_config(alpha_json: Optional[str]) -> Dict[str, Dict[str, float]]:
    if alpha_json is None:
        return {}

    with open(alpha_json, "r") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise ValueError("--alpha-json must contain a JSON object.")

    return cfg


def build_alpha_by_head(
    classes_by_head: Dict[str, List[str]],
    alpha_cfg: Dict[str, Dict[str, float]],
    rank: int,
) -> Dict[str, torch.Tensor]:
    alpha_by_head = {}

    for head, class_names in classes_by_head.items():
        head_cfg = alpha_cfg.get(head, {})
        values = [float(head_cfg.get(name, DEFAULT_ALPHA)) for name in class_names]
        alpha_by_head[head] = torch.tensor(values, dtype=torch.float32)

        if rank == 0:
            print(f"\nAlpha values for head '{head}':")
            for name, value in zip(class_names, values):
                print(f"  {name}: {value}")

            if head_cfg:
                unknown_config_names = sorted(set(head_cfg) - set(class_names))
                if unknown_config_names:
                    print(
                        f"[WARN] Alpha config for head '{head}' contains unused classes: "
                        f"{unknown_config_names}"
                    )

    return alpha_by_head


def parse_head_loss_weights(
    pairs: Optional[List[str]],
    heads: List[str],
) -> Dict[str, float]:
    weights = {head: 1.0 for head in heads}

    if not pairs:
        return weights

    for item in pairs:
        if "=" not in item:
            raise ValueError(
                f"Invalid --head-loss-weights value '{item}'. Expected HEAD=WEIGHT."
            )

        head, value = item.split("=", 1)
        head = head.strip()

        if head not in weights:
            raise ValueError(
                f"Unknown head '{head}' in --head-loss-weights. Valid heads: {heads}"
            )

        weight = float(value)
        if weight < 0:
            raise ValueError("Head loss weights must be >= 0.")

        weights[head] = weight

    if all(value == 0 for value in weights.values()):
        raise ValueError("At least one head loss weight must be > 0.")

    return weights


def get_dataloaders(
    train_csv: str,
    val_csv: str,
    data_root: Optional[str],
    heads: List[str],
    class_to_idx_by_head: Dict[str, Dict[str, int]],
    img_size: Tuple[int, int],
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
):
    # Same basic preprocessing as the original single-head script.
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_ds = MultiHeadVehicleDataset(
        csv_path=train_csv,
        heads=heads,
        class_to_idx_by_head=class_to_idx_by_head,
        transform=transform,
        data_root=data_root,
    )

    val_ds = MultiHeadVehicleDataset(
        csv_path=val_csv,
        heads=heads,
        class_to_idx_by_head=class_to_idx_by_head,
        transform=transform,
        data_root=data_root,
    )

    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False,
    )

    val_sampler = DistributedEvalSampler(
        val_ds,
        num_replicas=world_size,
        rank=rank,
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    train_loader = DataLoader(
        train_ds,
        shuffle=False,
        sampler=train_sampler,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        sampler=val_sampler,
        **loader_kwargs,
    )

    return train_loader, val_loader


def _new_local_metric_state(heads: List[str]):
    return {
        "total_loss_sum": 0.0,
        "sample_count": 0,
        "head_loss_sum": {head: 0.0 for head in heads},
        "head_correct": {head: 0 for head in heads},
    }


def _reduce_metrics(
    state,
    heads: List[str],
    device: torch.device,
):
    # [total_loss_sum, sample_count,
    #  head1_loss_sum, head1_correct, head2_loss_sum, head2_correct, ...]
    values = [
        float(state["total_loss_sum"]),
        float(state["sample_count"]),
    ]

    for head in heads:
        values.extend(
            [
                float(state["head_loss_sum"][head]),
                float(state["head_correct"][head]),
            ]
        )

    stats = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    reduced = stats.tolist()

    global_sample_count = int(reduced[1])
    if global_sample_count == 0:
        raise RuntimeError("No samples were processed.")

    metrics = {
        "total_loss": reduced[0] / global_sample_count,
        "sample_count": global_sample_count,
        "heads": {},
    }

    cursor = 2
    for head in heads:
        loss_sum = reduced[cursor]
        correct = reduced[cursor + 1]
        cursor += 2

        metrics["heads"][head] = {
            "loss": loss_sum / global_sample_count,
            "acc": correct / global_sample_count,
        }

    metrics["mean_acc"] = sum(
        metrics["heads"][head]["acc"] for head in heads
    ) / len(heads)

    return metrics


def train_one_epoch(
    model,
    loader,
    criteria,
    head_loss_weights,
    optimizer,
    device,
    epoch,
    heads,
    rank,
):
    model.train()
    loader.sampler.set_epoch(epoch)

    state = _new_local_metric_state(heads)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    epoch_start = time.perf_counter()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = {
            head: targets[head].to(device, non_blocking=True)
            for head in heads
        }

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)

        head_losses = {
            head: criteria[head](outputs[head], targets[head])
            for head in heads
        }

        total_loss = sum(
            head_loss_weights[head] * head_losses[head]
            for head in heads
        )

        total_loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        state["sample_count"] += batch_size
        state["total_loss_sum"] += total_loss.item() * batch_size

        for head in heads:
            state["head_loss_sum"][head] += head_losses[head].item() * batch_size
            preds = outputs[head].argmax(dim=1)
            state["head_correct"][head] += (
                preds == targets[head]
            ).sum().item()

    torch.cuda.synchronize(device)
    local_epoch_time = time.perf_counter() - epoch_start

    metrics = _reduce_metrics(state, heads, device)

    # Slowest rank determines distributed wall-clock epoch time.
    epoch_time_tensor = torch.tensor(
        local_epoch_time, dtype=torch.float64, device=device
    )
    dist.all_reduce(epoch_time_tensor, op=dist.ReduceOp.MAX)
    epoch_time = epoch_time_tensor.item()

    num_iterations = len(loader)
    avg_iter_time = epoch_time / num_iterations if num_iterations > 0 else 0.0

    local_peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    local_peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)

    peak_alloc_tensor = torch.tensor(
        local_peak_alloc, dtype=torch.float64, device=device
    )
    peak_reserved_tensor = torch.tensor(
        local_peak_reserved, dtype=torch.float64, device=device
    )

    dist.all_reduce(peak_alloc_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(peak_reserved_tensor, op=dist.ReduceOp.MAX)

    perf = {
        "epoch": epoch,
        "iterations_per_gpu": num_iterations,
        "epoch_time_sec": epoch_time,
        "avg_iter_time_sec": avg_iter_time,
        "max_peak_mem_allocated_gb": peak_alloc_tensor.item(),
        "max_peak_mem_reserved_gb": peak_reserved_tensor.item(),
    }

    if rank == 0:
        head_text = " | ".join(
            f"{head}: loss={metrics['heads'][head]['loss']:.4f}, "
            f"acc={metrics['heads'][head]['acc'] * 100:.2f}%"
            for head in heads
        )

        print(
            f"[Train] Epoch {epoch} | Total Loss: {metrics['total_loss']:.4f} "
            f"| Mean Acc: {metrics['mean_acc'] * 100:.2f}% | {head_text} "
            f"| Epoch: {epoch_time:.2f}s | Avg iter: {avg_iter_time:.4f}s "
            f"| Max alloc/GPU: {perf['max_peak_mem_allocated_gb']:.2f} GB "
            f"| Max reserved/GPU: {perf['max_peak_mem_reserved_gb']:.2f} GB"
        )

    return metrics, perf


@torch.no_grad()
def evaluate(
    model,
    loader,
    criteria,
    head_loss_weights,
    device,
    epoch,
    heads,
    rank,
):
    model.eval()
    # Validation shards have different lengths, sometimes zero. DDP's forward
    # broadcasts BatchNorm buffers on each batch and would deadlock when one
    # rank finishes early. Synchronize once, then evaluate the unwrapped model.
    if isinstance(model, DDP):
        for buffer in model.module.buffers():
            dist.broadcast(buffer, src=0)
        model = model.module
    state = _new_local_metric_state(heads)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = {
            head: targets[head].to(device, non_blocking=True)
            for head in heads
        }

        outputs = model(images)

        head_losses = {
            head: criteria[head](outputs[head], targets[head])
            for head in heads
        }

        total_loss = sum(
            head_loss_weights[head] * head_losses[head]
            for head in heads
        )

        batch_size = images.size(0)
        state["sample_count"] += batch_size
        state["total_loss_sum"] += total_loss.item() * batch_size

        for head in heads:
            state["head_loss_sum"][head] += head_losses[head].item() * batch_size
            preds = outputs[head].argmax(dim=1)
            state["head_correct"][head] += (
                preds == targets[head]
            ).sum().item()

    metrics = _reduce_metrics(state, heads, device)

    if rank == 0:
        head_text = " | ".join(
            f"{head}: loss={metrics['heads'][head]['loss']:.4f}, "
            f"acc={metrics['heads'][head]['acc'] * 100:.2f}%"
            for head in heads
        )

        print(
            f"[Val] Epoch {epoch} | Total Loss: {metrics['total_loss']:.4f} "
            f"| Mean Acc: {metrics['mean_acc'] * 100:.2f}% | {head_text}"
        )

    return metrics


def write_metrics_row(
    file_handle,
    epoch: int,
    split: str,
    metrics,
    heads: List[str],
):
    values = [
        str(epoch),
        split,
        str(metrics["total_loss"]),
        str(metrics["mean_acc"]),
    ]

    for head in heads:
        values.extend(
            [
                str(metrics["heads"][head]["loss"]),
                str(metrics["heads"][head]["acc"]),
            ]
        )

    file_handle.write(",".join(values) + "\n")
    file_handle.flush()


def selection_metric(metrics, monitor: str) -> float:
    if monitor == "mean":
        return metrics["mean_acc"]
    return metrics["heads"][monitor]["acc"]


def save_checkpoint_atomic(checkpoint: dict, path: str) -> None:
    """Keep the previous complete checkpoint if a write is interrupted."""
    temporary_path = path + ".tmp"
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def plot_logs(log_path: str, run_dir: str, heads: List[str]):
    logs = pd.read_csv(log_path)
    train_logs = logs[logs["Split"] == "train"]
    val_logs = logs[logs["Split"] == "val"]

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 1, 1)
    plt.plot(train_logs["Epoch"], train_logs["TotalLoss"], label="Train Total Loss")
    plt.plot(val_logs["Epoch"], val_logs["TotalLoss"], label="Val Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Total Loss")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(train_logs["Epoch"], train_logs["MeanAcc"], label="Train Mean Acc")
    plt.plot(val_logs["Epoch"], val_logs["MeanAcc"], label="Val Mean Acc")

    for head in heads:
        plt.plot(
            val_logs["Epoch"],
            val_logs[f"{head}_Acc"],
            label=f"Val {head} Acc",
            linestyle="--",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "training_plots.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Multi-head ResNet training with PyTorch DistributedDataParallel"
    )

    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--val-csv", type=str, required=True)
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help=(
            "Optional prefix for relative root_folder values in the CSV. "
            "Absolute root_folder values are used as-is."
        ),
    )
    parser.add_argument(
        "--heads",
        type=str,
        nargs="+",
        required=True,
        help="CSV columns to train as classification heads, e.g. make vehicle_type color",
    )

    parser.add_argument(
        "--arch",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50"],
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to ImageNet state_dict matching --arch (required for a new run)",
    )
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--resume", type=str, default=None,
        help="Resume from weights/last_model.pth; historical root checkpoints retain their flat layout",
    )
    run_mode.add_argument(
        "--run-dir", type=str, default=None,
        help="Explicit directory for a new agent attempt; existing training artifacts are refused",
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size PER GPU/process",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers PER GPU/process",
    )
    parser.add_argument("--img-size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=7)

    parser.add_argument(
        "--monitor",
        type=str,
        default="mean",
        help="Checkpoint/early-stop metric: 'mean' or one of the head names",
    )
    parser.add_argument(
        "--head-loss-weights",
        nargs="*",
        default=None,
        metavar="HEAD=WEIGHT",
        help=(
            "Optional per-head loss weights. Example: "
            "--head-loss-weights make=1.0 vehicle_type=0.5 color=0.75"
        ),
    )
    parser.add_argument(
        "--alpha-json",
        type=str,
        default=None,
        help=(
            "Optional focal-alpha JSON. Format: "
            "{\"make\": {\"Toyota\": 1.0}, \"color\": {\"white\": 0.5}}"
        ),
    )
    parser.add_argument("--out", type=str, default="outputs")

    args = parser.parse_args()

    if not args.resume and not args.weights:
        parser.error("--weights is required for a new run")
    if args.resume and os.path.basename(args.resume) != "last_model.pth":
        parser.error("--resume must point to last_model.pth")

    # Preserve CLI order while preventing duplicate heads.
    args.heads = list(dict.fromkeys(args.heads))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NCCL DDP.")

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    try:
        if args.monitor != "mean" and args.monitor not in args.heads:
            raise ValueError(
                f"--monitor must be 'mean' or one of {args.heads}; got '{args.monitor}'"
            )

        head_loss_weights = parse_head_loss_weights(
            args.head_loss_weights,
            args.heads,
        )

        classes_by_head, class_to_idx_by_head, num_classes_by_head = (
            build_class_mappings(
                args.train_csv,
                args.val_csv,
                args.heads,
            )
        )

        alpha_cfg = load_alpha_config(args.alpha_json)
        alpha_by_head = build_alpha_by_head(classes_by_head, alpha_cfg, rank)

        if rank == 0:
            print(f"DDP world size: {world_size}")
            print(f"Heads: {args.heads}")
            print(f"Classes per head: {num_classes_by_head}")
            print(f"Head loss weights: {head_loss_weights}")
            print(f"Monitor: {args.monitor}")
            print(f"Batch size per GPU: {args.batch_size}")
            print(f"Global batch size: {args.batch_size * world_size}")
            print(f"Workers per GPU: {args.num_workers}")
            print(f"Total DataLoader workers: {args.num_workers * world_size}")

        if args.resume:
            run_dir = str(checkpoint_run_dir(os.path.abspath(args.resume)))
        elif args.run_dir:
            run_dir = os.path.abspath(args.run_dir)
        else:
            # Rank 0 creates the timestamp, then broadcasts it.
            timestamp_holder = [
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S") if rank == 0 else ""
            ]
            dist.broadcast_object_list(timestamp_holder, src=0)
            run_dir = os.path.join(args.out, timestamp_holder[0])

        legacy_resume = bool(args.resume and os.path.basename(os.path.dirname(os.path.abspath(args.resume))) != "weights")
        model_dir = run_dir if legacy_resume else str(weights_dir(run_dir))
        evaluation_dir = run_dir if legacy_resume else str(train_evaluation_dir(run_dir))
        if rank == 0:
            if args.resume:
                os.makedirs(run_dir, exist_ok=True)
                os.makedirs(model_dir, exist_ok=True)
                os.makedirs(evaluation_dir, exist_ok=True)
            elif args.run_dir:
                prepare_new_run_directory(run_dir, allow_existing=True)
            else:
                prepare_new_run_directory(run_dir)
        dist.barrier()

        best_model_path = os.path.join(model_dir, "best_model.pth")
        last_model_path = os.path.join(model_dir, "last_model.pth")

        if rank == 0:
            print("Run dir:", run_dir)
            print("Best model path:", best_model_path)
            print("Last model path:", last_model_path)

            if not args.resume:
                with open(os.path.join(model_dir, "classes_by_head.json"), "w") as f:
                    json.dump(classes_by_head, f, indent=2)

                with open(os.path.join(model_dir, "class_to_idx_by_head.json"), "w") as f:
                    json.dump(class_to_idx_by_head, f, indent=2)

                with open(os.path.join(evaluation_dir, "training_config.json"), "w") as f:
                    json.dump(
                        {
                            "arch": args.arch,
                            "heads": args.heads,
                            "num_classes_by_head": num_classes_by_head,
                            "head_loss_weights": head_loss_weights,
                            "gamma": args.gamma,
                            "lr": args.lr,
                            "weight_decay": args.weight_decay,
                            "batch_size_per_gpu": args.batch_size,
                            "world_size": world_size,
                            "global_batch_size": args.batch_size * world_size,
                            "monitor": args.monitor,
                            "alpha_json": args.alpha_json,
                            "train_csv": args.train_csv,
                            "val_csv": args.val_csv,
                            "epochs": args.epochs,
                            "img_size": args.img_size,
                            "data_root": args.data_root,
                            "weights": args.weights,
                            "num_workers": args.num_workers,
                            "patience": args.patience,
                        },
                        f,
                        indent=2,
                    )

        train_loader, val_loader = get_dataloaders(
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            data_root=args.data_root,
            heads=args.heads,
            class_to_idx_by_head=class_to_idx_by_head,
            img_size=tuple(args.img_size),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            rank=rank,
            world_size=world_size,
        )

        model = MultiHeadResNet(
            arch=args.arch,
            weights_path=None if args.resume else args.weights,
            num_classes_by_head=num_classes_by_head,
        ).to(device)

        resume_state = None
        if args.resume:
            with open(os.path.join(evaluation_dir, "training_config.json")) as f:
                saved_config = json.load(f)
            if (
                saved_config["epochs"] != args.epochs
                or saved_config["gamma"] != args.gamma
                or saved_config["batch_size_per_gpu"] != args.batch_size
                or saved_config["world_size"] != world_size
                or saved_config["train_csv"] != args.train_csv
                or saved_config["val_csv"] != args.val_csv
                or saved_config["alpha_json"] != args.alpha_json
            ):
                raise ValueError("Resume with the original epochs, batch size, GPU count, CSVs, alpha config, and gamma")
            resume_state = torch.load(args.resume, map_location="cpu", weights_only=False)
            if (
                resume_state["arch"] != args.arch
                or resume_state["heads"] != args.heads
                or resume_state["classes_by_head"] != classes_by_head
                or resume_state["monitor"] != args.monitor
                or resume_state["head_loss_weights"] != head_loss_weights
            ):
                raise ValueError("Resume checkpoint does not match the model, classes, monitor, or loss weights")
            model.load_state_dict(resume_state["model_state"])

        # All heads are used in every forward pass, so find_unused_parameters=False.
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

        criteria = {
            head: FocalLoss(
                alpha=alpha_by_head[head],
                gamma=args.gamma,
            ).to(device)
            for head in args.heads
        }

        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )

        best_score = float("-inf")
        best_epoch = 0
        best_val_metrics = None
        patience_counter = 0
        start_epoch = 1

        if resume_state is not None:
            optimizer.load_state_dict(resume_state["optimizer_state"])
            scheduler.load_state_dict(resume_state["scheduler_state"])
            best_score = resume_state["best_score"]
            best_epoch = resume_state["best_epoch"]
            best_val_metrics = resume_state["best_val_metrics"]
            patience_counter = resume_state["patience_counter"]
            start_epoch = resume_state["epoch"] + 1
            if args.epochs < start_epoch:
                raise ValueError(f"--epochs must be at least {start_epoch} to resume")
            if rank == 0:
                print(f"Resuming after epoch {start_epoch - 1}")

        epochs_trained = start_epoch - 1

        metrics_file = None
        perf_file = None
        metrics_path = os.path.join(evaluation_dir, "training_metrics.csv")
        perf_path = os.path.join(evaluation_dir, "performance_metrics.csv")

        try:
            if rank == 0:
                metrics_file = open(metrics_path, "a" if args.resume else "w")
                perf_file = open(perf_path, "a" if args.resume else "w")

                if not args.resume:
                    metric_columns = ["Epoch", "Split", "TotalLoss", "MeanAcc"]
                    for head in args.heads:
                        metric_columns.extend([f"{head}_Loss", f"{head}_Acc"])
                    metrics_file.write(",".join(metric_columns) + "\n")

                    perf_file.write(
                        "Epoch,IterationsPerGPU,EpochTimeSec,AvgIterTimeSec,"
                        "MaxPeakMemAllocatedGB,MaxPeakMemReservedGB\n"
                    )

            for epoch in range(start_epoch, args.epochs + 1):
                epochs_trained = epoch

                train_metrics, perf = train_one_epoch(
                    model=model,
                    loader=train_loader,
                    criteria=criteria,
                    head_loss_weights=head_loss_weights,
                    optimizer=optimizer,
                    device=device,
                    epoch=epoch,
                    heads=args.heads,
                    rank=rank,
                )

                val_metrics = evaluate(
                    model=model,
                    loader=val_loader,
                    criteria=criteria,
                    head_loss_weights=head_loss_weights,
                    device=device,
                    epoch=epoch,
                    heads=args.heads,
                    rank=rank,
                )

                scheduler.step()

                stop_tensor = torch.zeros(1, dtype=torch.int32, device=device)

                if rank == 0:
                    write_metrics_row(
                        metrics_file, epoch, "train", train_metrics, args.heads
                    )
                    write_metrics_row(
                        metrics_file, epoch, "val", val_metrics, args.heads
                    )

                    perf_file.write(
                        f"{perf['epoch']},{perf['iterations_per_gpu']},"
                        f"{perf['epoch_time_sec']},{perf['avg_iter_time_sec']},"
                        f"{perf['max_peak_mem_allocated_gb']},"
                        f"{perf['max_peak_mem_reserved_gb']}\n"
                    )
                    perf_file.flush()
                    metrics_file.flush()

                    score = selection_metric(val_metrics, args.monitor)

                    if score > best_score:
                        best_score = score
                        best_epoch = epoch
                        patience_counter = 0
                        best_val_metrics = val_metrics

                        best_checkpoint = {
                            "model_state": model.module.state_dict(),
                            "arch": args.arch,
                            "heads": args.heads,
                            "classes_by_head": classes_by_head,
                            "class_to_idx_by_head": class_to_idx_by_head,
                            "num_classes_by_head": num_classes_by_head,
                            "head_loss_weights": head_loss_weights,
                            "monitor": args.monitor,
                            "best_score": best_score,
                            "best_epoch": best_epoch,
                            "best_val_metrics": best_val_metrics,
                            "img_size": args.img_size,
                        }
                        save_checkpoint_atomic(best_checkpoint, best_model_path)

                        print(
                            f"New best validation {args.monitor} score: "
                            f"{score * 100:.2f}% at epoch {epoch}. Saved {best_model_path}"
                        )
                    else:
                        patience_counter += 1
                        print(
                            f"No improvement for {patience_counter} epochs "
                            f"(best was epoch {best_epoch}, "
                            f"{args.monitor}={best_score * 100:.2f}%)"
                        )

                        if patience_counter >= args.patience:
                            stop_tensor.fill_(1)

                    last_checkpoint = {
                        "model_state": model.module.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "epoch": epoch,
                        "arch": args.arch,
                        "heads": args.heads,
                        "classes_by_head": classes_by_head,
                        "class_to_idx_by_head": class_to_idx_by_head,
                        "num_classes_by_head": num_classes_by_head,
                        "head_loss_weights": head_loss_weights,
                        "monitor": args.monitor,
                        "best_score": best_score,
                        "best_epoch": best_epoch,
                        "best_val_metrics": best_val_metrics,
                        "patience_counter": patience_counter,
                        "img_size": args.img_size,
                    }
                    save_checkpoint_atomic(last_checkpoint, last_model_path)
                    print(f"Saved last checkpoint after epoch {epoch}: {last_model_path}")

                dist.broadcast(stop_tensor, src=0)

                if stop_tensor.item() == 1:
                    if rank == 0:
                        print(f"\nEarly stopping triggered after {epoch} epochs.")
                        print(
                            f"Best validation {args.monitor}: "
                            f"{best_score * 100:.2f}% at epoch {best_epoch}"
                        )
                    break

            if rank == 0:
                print(
                    f"\nTraining completed after {epochs_trained} epochs. "
                    f"Best validation {args.monitor}: "
                    f"{best_score * 100:.2f}% at epoch {best_epoch}"
                )

                if best_val_metrics is not None:
                    for head in args.heads:
                        print(
                            f"  {head}: "
                            f"{best_val_metrics['heads'][head]['acc'] * 100:.2f}%"
                        )

                plot_logs(metrics_path, evaluation_dir, args.heads)

        finally:
            if metrics_file is not None:
                metrics_file.close()
            if perf_file is not None:
                perf_file.close()

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
