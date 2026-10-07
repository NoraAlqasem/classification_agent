from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.tools import tool


def validate_onnx_file(onnx_path: str) -> dict[str, Any]:
    """Validate an ONNX model file. Returns a result dict with 'valid' and metadata."""
    p = Path(onnx_path)
    if not p.exists():
        return {"valid": False, "error": f"File not found: {onnx_path}"}
    try:
        import onnx
        model = onnx.load(str(p))
        onnx.checker.check_model(model)
        opset = model.opset_import[0].version if model.opset_import else "unknown"
        size_mb = round(p.stat().st_size / 1_048_576, 2)
        return {
            "valid": True,
            "opset_version": opset,
            "size_mb": size_mb,
            "path": str(p),
        }
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


@tool
def list_export_artefacts(run_dir: str) -> str:
    """
    List ResNet ONNX models and their preprocessing/class metadata sidecars.
    """
    p = Path(run_dir)
    if not p.exists():
        return json.dumps({"error": f"Directory not found: {run_dir}"})

    artefacts = [
        {"path": str(f), "relative_path": str(f.relative_to(p)),
         "size_mb": round(f.stat().st_size / 1_048_576, 2)}
        for f in sorted(p.rglob("*"))
        if f.is_file() and (f.suffix == ".onnx" or f.name.endswith(".metadata.json"))
    ]
    return json.dumps({"artefacts": artefacts})


@tool
def get_model_info(weights_path: str) -> str:
    """
    Return architecture, head classes, and tensor count from a ResNet checkpoint.
    """
    try:
        import torch

        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        tensors = checkpoint["model_state"]
        return json.dumps({
            "path": weights_path,
            "arch": checkpoint["arch"],
            "heads": checkpoint["heads"],
            "classes_by_head": checkpoint["classes_by_head"],
            "num_classes_by_head": checkpoint["num_classes_by_head"],
            "state_tensor_count": len(tensors),
            "state_element_count": sum(tensor.numel() for tensor in tensors.values()),
            "best_score": checkpoint.get("best_score"),
            "best_epoch": checkpoint.get("best_epoch"),
            "monitor": checkpoint.get("monitor"),
        })
    except Exception as exc:
        return json.dumps({"error": str(exc), "path": weights_path})
