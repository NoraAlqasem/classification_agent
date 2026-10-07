"""Export standalone multi-head ResNet checkpoints for inference."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from training.artifact_paths import checkpoint_training_config


def export_resnet_onnx(weights_path: str, config: dict[str, Any]) -> str:
    """Export one logits tensor per classification head and its input metadata."""
    import torch
    import torch.nn as nn

    from training.train_multihead_ddp import MultiHeadResNet

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    heads = checkpoint["heads"]
    model = MultiHeadResNet(
        checkpoint["arch"], None, checkpoint["num_classes_by_head"],
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    class LogitsOutputs(nn.Module):
        def __init__(self, classifier):
            super().__init__()
            self.classifier = classifier

        def forward(self, images):
            outputs = self.classifier(images)
            return tuple(outputs[head] for head in heads)

    # Preserve the selected model's training resolution, even when a later
    # attempt used a different resolution.
    img_size = checkpoint.get("img_size") or config.get("img_size")
    if not img_size:
        config_path = checkpoint_training_config(weights_path)
        if config_path.is_file():
            img_size = json.loads(config_path.read_text()).get("img_size")
    if not img_size or len(img_size) != 2:
        raise ValueError("Checkpoint/export configuration must specify training img_size [height, width].")
    height, width = (int(value) for value in img_size)
    if height <= 0 or width <= 0:
        raise ValueError("ONNX image height and width must be positive.")

    path = Path(weights_path).with_suffix(".onnx")
    output_names = [f"{head}_logits" for head in heads]
    dynamic_axes = {"images": {0: "batch"}}
    dynamic_axes.update({name: {0: "batch"} for name in output_names})
    torch.onnx.export(
        LogitsOutputs(model).eval(), torch.zeros(1, 3, height, width), str(path),
        input_names=["images"], output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=17, dynamo=False,
    )
    import onnx

    onnx.checker.check_model(onnx.load(str(path)))
    metadata = {
        "arch": checkpoint["arch"], "heads": heads,
        "classes_by_head": checkpoint["classes_by_head"],
        "class_to_idx_by_head": checkpoint["class_to_idx_by_head"],
        "outputs": dict(zip(heads, output_names)),
        "output_type": "logits; apply softmax per head for probabilities",
        "preprocessing": {
            "color_mode": "RGB", "img_size": [height, width],
            "resize_interpolation": "bilinear", "resize_antialias": True,
            "layout": "NCHW", "dtype": "float32", "pixel_scale": 1.0 / 255.0,
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        },
    }
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return str(path)
