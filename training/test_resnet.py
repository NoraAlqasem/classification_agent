"""Evaluate a multi-head ResNet checkpoint against every row of a test CSV.

Adapted from the user's test_resnet.py. Inference, per-class accuracy and
confusion-matrix artifacts are retained; machine-readable counts make it safe
for the training agent to use the result for its next-attempt decision.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    # Also support the supplied script's original `python training/test_resnet.py` entry point.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

from training.data import read_dataset_csv
from training.artifact_paths import checkpoint_training_config


class MultiHeadResNet(nn.Module):
    def __init__(self, arch, num_classes_by_head):
        super().__init__()
        if arch not in {"resnet18", "resnet34", "resnet50"}:
            raise ValueError(f"Unsupported architecture: {arch}")
        backbone = getattr(models, arch)(weights=None)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.heads = nn.ModuleDict({
            head: nn.Linear(feature_dim, count)
            for head, count in num_classes_by_head.items()
        })

    def forward(self, images):
        features = self.backbone(images)
        return {head: layer(features) for head, layer in self.heads.items()}


def _validated_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Multi-head checkpoint must be a dictionary")
    required = {"model_state", "arch", "heads", "classes_by_head"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Multi-head checkpoint is missing: {sorted(missing)}")
    if checkpoint["arch"] not in {"resnet18", "resnet34", "resnet50"}:
        raise ValueError(f"Unsupported checkpoint architecture: {checkpoint['arch']}")
    heads, classes = checkpoint["heads"], checkpoint["classes_by_head"]
    if not isinstance(heads, (list, tuple)) or not heads:
        raise ValueError("Checkpoint heads must be a nonempty list")
    if any(not isinstance(head, str) or not head.isidentifier()
           or hasattr(nn.ModuleDict(), head) for head in heads):
        raise ValueError("Checkpoint has invalid or reserved head names")
    if len(set(heads)) != len(heads):
        raise ValueError("Checkpoint heads contain duplicates")
    if not isinstance(classes, dict) or set(classes) != set(heads):
        raise ValueError("Checkpoint class lists must match its heads exactly")
    for head in heads:
        labels = classes[head]
        if not isinstance(labels, (list, tuple)) or not labels:
            raise ValueError(f"Checkpoint head '{head}' has an empty or invalid class list")
        if any(not isinstance(label, str) or not label.strip() or label != label.strip()
               for label in labels) or len(set(labels)) != len(labels):
            raise ValueError(f"Checkpoint head '{head}' has invalid or duplicate class labels")
    counts = {head: len(classes[head]) for head in heads}
    if "num_classes_by_head" in checkpoint:
        supplied_counts = checkpoint["num_classes_by_head"]
        if not isinstance(supplied_counts, dict) or supplied_counts != counts or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in supplied_counts.values()
        ):
            raise ValueError("Checkpoint num_classes_by_head differs from its class lists")
    expected_mapping = {
        head: {label: index for index, label in enumerate(classes[head])}
        for head in heads
    }
    if "class_to_idx_by_head" in checkpoint:
        mapping = checkpoint["class_to_idx_by_head"]
        if not isinstance(mapping, dict) or mapping != expected_mapping or any(
            isinstance(index, bool) or not isinstance(index, int)
            for value in mapping.values() if isinstance(value, dict)
            for index in value.values()
        ):
            raise ValueError("Checkpoint class_to_idx_by_head differs from its class lists")
    state = checkpoint["model_state"]
    if not isinstance(state, Mapping) or not state or any(
        not isinstance(key, str) or not isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError("Checkpoint model_state must contain named tensors")
    stripped_state = {key.removeprefix("module."): value for key, value in state.items()}
    if len(stripped_state) != len(state):
        raise ValueError("Checkpoint contains conflicting DDP-prefixed state keys")
    checkpoint = dict(checkpoint)
    checkpoint.update({"heads": list(heads), "model_state": stripped_state,
                       "num_classes_by_head": counts,
                       "class_to_idx_by_head": expected_mapping})
    return checkpoint


def load_checkpoint(path):
    """Keep the original script's public loading contract."""
    checkpoint = _validated_checkpoint(path)
    return (checkpoint["arch"], checkpoint["heads"], checkpoint["classes_by_head"],
            checkpoint["model_state"])


def _image_size(checkpoint, config, checkpoint_path):
    # The trained checkpoint always wins over a later attempt's configuration.
    value = checkpoint.get("img_size")
    if value is None:
        value = (config or {}).get("img_size")
    if value is None:
        config_path = checkpoint_training_config(checkpoint_path)
        if config_path.is_file():
            value = json.loads(config_path.read_text(encoding="utf-8")).get("img_size")
    if not isinstance(value, (list, tuple)) or len(value) != 2 or any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in value
    ):
        raise ValueError("Checkpoint/evaluation configuration must specify positive img_size [height, width]")
    return list(value)


def read_ground_truth(path, heads, *, require_root=True):
    df = read_dataset_csv(path, heads)
    required = {"image_name", *heads}
    if require_root:
        required.add("root_folder")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Test CSV is missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Test CSV is empty")
    for column in ["image_name", *heads, *(["root_folder"] if "root_folder" in df else [])]:
        if df[column].isna().any() or df[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Test CSV has missing values in '{column}'")
        df[column] = df[column].astype(str).str.strip()
    return df


def resolve_image_paths(df, args):
    if args.input:
        input_path = Path(args.input)
        if input_path.is_file():
            if len(df) != 1:
                raise ValueError("--input is one file but the test CSV has multiple rows")
            paths = [input_path]
        else:
            if df["image_name"].duplicated().any():
                raise ValueError("--input is a flat directory but image_name repeats in the test CSV")
            paths = [input_path / name for name in df["image_name"]]
    else:
        if "root_folder" not in df:
            raise ValueError("Test CSV needs root_folder, or provide --input")
        paths = []
        for root, name in zip(df["root_folder"], df["image_name"]):
            folder = Path(root)
            if not folder.is_absolute() and args.data_root:
                folder = Path(args.data_root) / folder
            paths.append(folder / name)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} test images are missing. Examples: {missing[:5]}")
    return paths


def run_inference(df, paths, arch, classes_by_head, state, heads, args):
    requested_device = getattr(args, "device", None)
    device = torch.device(requested_device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Inference device: {device}; images: {len(paths)}")
    model = MultiHeadResNet(arch, {head: len(classes) for head, classes in classes_by_head.items()})
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    transform = transforms.Compose([
        transforms.Resize(tuple(args.img_size)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    pred_df = df[[column for column in ("root_folder", "image_name") if column in df]].copy()
    predictions, confidences = {head: [] for head in heads}, {head: [] for head in heads}
    with torch.inference_mode():
        for start in range(0, len(paths), args.batch_size):
            images = []
            for path in paths[start:start + args.batch_size]:
                with Image.open(path) as image:
                    images.append(transform(image.convert("RGB")))
            outputs = model(torch.stack(images).to(device))
            for head in heads:
                logits = outputs[head]
                if logits.shape != (len(images), len(classes_by_head[head])) or not torch.isfinite(logits).all():
                    raise ValueError(f"Invalid or nonfinite inference logits for head '{head}'")
                conf, index = torch.softmax(logits, dim=1).max(dim=1)
                predictions[head].extend(classes_by_head[head][i] for i in index.cpu().tolist())
                confidences[head].extend(conf.cpu().tolist())
    for head in heads:
        pred_df[f"{head}_pred"], pred_df[f"{head}_conf"] = predictions[head], confidences[head]
    return pred_df


def match_predictions(gt_df, pred_df, heads):
    required = {"image_name"} | {f"{head}_{suffix}" for head in heads for suffix in ("pred", "conf")}
    missing = required - set(pred_df)
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {sorted(missing)}")
    join_cols = ["image_name"]
    if "root_folder" in gt_df and "root_folder" in pred_df:
        join_cols.insert(0, "root_folder")
    gt_df, pred_df = gt_df.copy(), pred_df.copy()
    for col in join_cols:
        if pred_df[col].isna().any():
            raise ValueError(f"Prediction CSV has missing '{col}' values")
        pred_df[col] = pred_df[col].astype(str).str.strip()
    for name, frame in (("test", gt_df), ("prediction", pred_df)):
        if frame.duplicated(join_cols).any():
            raise ValueError(f"Duplicate {join_cols} in {name} CSV; cannot match images uniquely")
    prediction_cols = join_cols + [f"{head}_{suffix}" for head in heads for suffix in ("pred", "conf")]
    merged = gt_df.merge(pred_df[prediction_cols], on=join_cols, how="left",
                         validate="one_to_one", indicator=True)
    unmatched = int((merged["_merge"] != "both").sum())
    if unmatched:
        raise ValueError(f"Incomplete test coverage: {unmatched} test rows have no matching prediction")
    if len(pred_df) != len(gt_df):
        raise ValueError(f"Prediction CSV has {len(pred_df) - len(gt_df)} rows that do not match the test CSV")
    merged = merged.drop(columns="_merge")
    if merged.empty:
        raise ValueError("Test CSV is empty")
    for head in heads:
        pred_col, conf_col = f"{head}_pred", f"{head}_conf"
        if merged[pred_col].isna().any() or merged[pred_col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing predictions for head '{head}'")
        merged[pred_col] = merged[pred_col].astype(str).str.strip()
        merged[conf_col] = pd.to_numeric(merged[conf_col], errors="coerce")
        if merged[conf_col].isna().any() or not merged[conf_col].between(0, 1).all():
            raise ValueError(f"Invalid confidence values for head '{head}'")
        merged[f"{head}_true"], merged[f"{head}_correct"] = merged[head], merged[head] == merged[pred_col]
    columns = [col for col in ("root_folder", "image_name") if col in merged]
    columns += [f"{head}_{suffix}" for head in heads for suffix in ("true", "pred", "conf", "correct")]
    return merged[columns]


def fix_arabic_labels(labels):
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
    except ImportError:
        return [str(label) for label in labels]
    return [get_display(arabic_reshaper.reshape(str(label))) for label in labels]


def save_confusion_matrix_png(cm, labels, path, normalize=False):
    values = cm.astype(float)
    if normalize:
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums != 0)
    size = max(8, min(24, len(labels) * 0.65))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(values, cmap="Purples", interpolation="nearest")
    fig.colorbar(im, ax=ax)
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks, labels=fix_arabic_labels(labels), rotation=90)
    ax.set_yticks(ticks, labels=fix_arabic_labels(labels))
    ax.set(xlabel="Predicted label", ylabel="Ground truth label",
           title="Normalized confusion matrix" if normalize else "Confusion matrix")
    if len(labels) <= 40:
        threshold = values.max() / 2 if values.size else 0
        for row, column in np.ndindex(values.shape):
            value = values[row, column]
            ax.text(column, row, f"{value:.2f}" if normalize else str(int(value)),
                    ha="center", va="center", fontsize=8,
                    color="white" if value > threshold else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def evaluate_head(results, head, labels, threshold, output_dir):
    out = output_dir / head
    out.mkdir(parents=True, exist_ok=True)
    evaluated = results[results[f"{head}_conf"] >= threshold].copy()
    true_col, pred_col = f"{head}_true", f"{head}_pred"
    correct, count = int(evaluated[f"{head}_correct"].sum()), len(evaluated)
    per_class = (evaluated.groupby(true_col)
                 .agg(total=("image_name", "size"), correct=(f"{head}_correct", "sum"))
                 .reset_index().rename(columns={true_col: "true_label"}))
    per_class["accuracy"] = per_class["correct"] / per_class["total"]
    per_class = per_class.sort_values(["accuracy", "total"], ascending=[False, False])
    per_class.to_csv(
        out / "per_class_accuracy.csv", index=False, encoding="utf-8-sig")
    class_accuracy = [
        {"class": str(row.true_label), "correct": int(row.correct),
         "total": int(row.total), "accuracy": float(row.accuracy)}
        for row in per_class.itertuples(index=False)
    ]
    all_labels = list(labels) if labels is not None else []
    for value in pd.concat([evaluated[true_col], evaluated[pred_col]]).unique():
        if value not in all_labels:
            all_labels.append(value)
    if all_labels:
        indices = {label: i for i, label in enumerate(all_labels)}
        cm = np.zeros((len(all_labels), len(all_labels)), dtype=np.int64)
        for true, pred in zip(evaluated[true_col], evaluated[pred_col]):
            cm[indices[true], indices[pred]] += 1
        pd.DataFrame(cm, index=[f"GT: {v}" for v in all_labels],
                     columns=[f"Pred: {v}" for v in all_labels]).to_csv(
                         out / "confusion_matrix_counts.csv", encoding="utf-8-sig")
        save_confusion_matrix_png(cm, all_labels, out / "confusion_matrix_counts.png")
        save_confusion_matrix_png(cm, all_labels, out / "confusion_matrix_normalized.png", normalize=True)
    accuracy = correct / count if count else None
    print(f"{head}: {correct}/{count} correct; below threshold={len(results) - count}")
    return {"head": head, "matched": len(results), "evaluated": count,
            "total": len(results), "count": count, "coverage": count / len(results),
            "skipped_by_threshold": len(results) - count, "correct": correct,
            "accuracy": accuracy, "acc": accuracy, "per_class_accuracy": class_accuracy}


def _check_labels(df, heads, classes_by_head):
    for head in heads:
        unseen = sorted(set(df[head]) - set(classes_by_head[head]))
        if unseen:
            raise ValueError(f"Test CSV head '{head}' contains classes absent from checkpoint: {unseen[:10]}")


def _save_evaluation(results, heads, classes_by_head, output_dir, threshold=0.0):
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "inference_results.csv", index=False, encoding="utf-8-sig")
    summaries = [evaluate_head(results, head, classes_by_head[head] if classes_by_head else None,
                               threshold, output_dir) for head in heads]
    pd.DataFrame([{key: value for key, value in summary.items() if key != "per_class_accuracy"}
                  for summary in summaries]).to_csv(output_dir / "summary.csv", index=False)
    full_coverage = all(summary["count"] == len(results) for summary in summaries)
    scores = [summary["acc"] for summary in summaries]
    mean_acc = sum(scores) / len(scores) if all(score is not None for score in scores) else None
    metrics = {"status": "ok" if full_coverage else "partial_coverage", "mean_acc": mean_acc,
               "heads": {head: summary for head, summary in zip(heads, summaries)},
               "total_images": len(results), "matched_images": len(results),
               "evaluated_images": min(summary["count"] for summary in summaries),
               "full_coverage": full_coverage, "threshold": threshold,
               "output_dir": str(output_dir.resolve())}
    metrics.update({f"{head}_acc": summary["acc"] for head, summary in zip(heads, summaries)})
    return metrics


def evaluate_checkpoint(checkpoint_path: str, csv_path: str, output_dir: str, *,
                        data_root: str = "", heads: list[str] | None = None,
                        config: dict[str, Any] | None = None, batch_size: int = 32,
                        device: str | None = None) -> dict[str, Any]:
    """Infer on all test rows at threshold zero; fail instead of dropping rows.

    Every checkpoint head participates in mean_acc. The agent's expected heads
    must match the checkpoint exactly, preventing a subset from changing the
    stopping metric or silently accepting an unrelated checkpoint.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("Evaluation batch_size must be a positive integer")
    checkpoint_path, csv_path = str(Path(checkpoint_path).resolve()), str(Path(csv_path).resolve())
    checkpoint = _validated_checkpoint(checkpoint_path)
    checkpoint_heads = checkpoint["heads"]
    if heads is not None and (not isinstance(heads, (list, tuple)) or
                              len(heads) != len(checkpoint_heads) or set(heads) != set(checkpoint_heads)):
        raise ValueError("Evaluation heads must match every checkpoint head exactly")
    heads = checkpoint_heads
    img_size = _image_size(checkpoint, config, checkpoint_path)
    gt_df = read_ground_truth(csv_path, heads)
    _check_labels(gt_df, heads, checkpoint["classes_by_head"])
    args = argparse.Namespace(input=None, data_root=data_root, img_size=img_size,
                              batch_size=batch_size, device=device)
    paths = resolve_image_paths(gt_df, args)
    predictions = run_inference(gt_df, paths, checkpoint["arch"],
                                checkpoint["classes_by_head"], checkpoint["model_state"], heads, args)
    results = match_predictions(gt_df, predictions, heads)
    if len(results) != len(gt_df):
        raise ValueError("Incomplete test evaluation: not every test row was evaluated")
    metrics = _save_evaluation(results, heads, checkpoint["classes_by_head"], Path(output_dir))
    metrics.update({"checkpoint": checkpoint_path, "test_csv": csv_path,
                    "arch": checkpoint["arch"], "img_size": img_size,
                    "split": "test", "decision_split": "test", "test_evaluated": True})
    Path(output_dir, "summary.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a multi-head ResNet on every test CSV row.")
    parser.add_argument("--gt_csv", required=True)
    parser.add_argument("--ckpt")
    parser.add_argument("--pred_csv")
    parser.add_argument("--heads", nargs="+")
    parser.add_argument("--input", help="Optional flat image directory or single-image override")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--arch", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--img-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"),
                        help="Fallback only when checkpoint does not store its training image size")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--out_dir", default="evaluation_results")
    args = parser.parse_args()
    if not args.ckpt and not args.pred_csv:
        parser.error("Provide --ckpt for inference or --pred_csv for saved predictions")
    if not args.ckpt and not args.heads:
        parser.error("--heads is required with --pred_csv and no checkpoint")
    if args.batch_size <= 0 or (args.img_size is not None and any(size <= 0 for size in args.img_size)):
        parser.error("--batch-size and --img-size must be positive")
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    return args


def main():
    args = parse_args()
    checkpoint, classes = None, None
    if args.ckpt:
        checkpoint = _validated_checkpoint(args.ckpt)
        classes = checkpoint["classes_by_head"]
        if args.arch and args.arch != checkpoint["arch"]:
            raise ValueError("--arch differs from checkpoint architecture")
    heads = args.heads or checkpoint["heads"]
    if len(set(heads)) != len(heads):
        raise ValueError("--heads must not contain duplicates")
    if checkpoint and set(heads) - set(checkpoint["heads"]):
        raise ValueError("Requested heads are absent from checkpoint")
    gt_df = read_ground_truth(args.gt_csv, heads,
                             require_root=not (getattr(args, "input", None) or args.pred_csv))
    if checkpoint:
        _check_labels(gt_df, heads, classes)
    if args.pred_csv:
        pred_df = pd.read_csv(args.pred_csv, dtype=str, keep_default_na=False)
    else:
        args.img_size = _image_size(checkpoint, {"img_size": args.img_size}, args.ckpt)
        paths = resolve_image_paths(gt_df, args)
        pred_df = run_inference(gt_df, paths, checkpoint["arch"], classes,
                                checkpoint["model_state"], heads, args)
    results = match_predictions(gt_df, pred_df, heads)
    if classes:
        for head in heads:
            if set(results[f"{head}_pred"]) - set(classes[head]):
                raise ValueError(f"Prediction CSV has labels absent from checkpoint for '{head}'")
    output_dir = Path(args.out_dir)
    metrics = _save_evaluation(results, heads, classes, output_dir, args.threshold)
    metrics.update({"test_csv": str(Path(args.gt_csv).resolve()),
                    "checkpoint": str(Path(args.ckpt).resolve()) if args.ckpt else None,
                    "split": "test", "test_evaluated": True})
    (output_dir / "summary.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved multi-head results to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
