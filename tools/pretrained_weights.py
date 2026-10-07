"""Discover local ResNet backbone weights that the training script can load."""
from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import logging
from pathlib import Path

import torch
from torchvision import models


PRETRAINED_WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "weights" / "pretrained"
SUPPORTED_ARCHITECTURES = ("resnet18", "resnet34", "resnet50")
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _backbone_shapes() -> dict[str, dict[str, tuple[int, ...]]]:
    # Meta tensors retain shapes without allocating backbone tensors on CPU/GPU.
    with torch.device("meta"):
        return {
            arch: {key: tuple(value.shape) for key, value in
                   getattr(models, arch)(weights=None).state_dict().items()}
            for arch in SUPPORTED_ARCHITECTURES
        }


def _checkpoint_architecture(path: Path) -> str | None:
    state = torch.load(path, weights_only=True, map_location="meta")
    # The trainer loads a raw backbone state dict with strict=True. Wrapped
    # training checkpoints and DDP-prefixed keys do not meet that contract.
    if not isinstance(state, Mapping) or not state or any(
        not isinstance(key, str) or not isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        return None
    shapes = {key: tuple(value.shape) for key, value in state.items()}
    metadata = getattr(state, "_metadata", None)
    for arch, expected in _backbone_shapes().items():
        if set(shapes) - set(expected) or any(
            shapes[key] != expected[key] for key in shapes if key in expected
        ):
            continue
        missing = set(expected) - set(shapes)
        # Original torchvision weights predate BatchNorm's tracking buffer.
        # The trainer's strict load adds that buffer for pre-v2/absent metadata.
        compatible = True
        for key in missing:
            if not key.endswith(".num_batches_tracked"):
                compatible = False
                break
            local_metadata = metadata.get(key.rsplit(".", 1)[0], {}) if metadata else {}
            version = local_metadata.get("version")
            if version is not None and version >= 2:
                compatible = False
                break
        if compatible:
            return arch
    return None


def discover_pretrained_weights(directory: str | Path | None = None) -> list[dict[str, str]]:
    """Return compatible local files, inferring architecture from tensor shapes.

    Inspect only immediate ``.pth``/``.pt`` files in the chosen directory. A
    symlink cannot expose files outside that directory. Invalid files are
    reported locally and omitted from the choices shown to the LLM.
    """
    directory = Path(directory) if directory is not None else PRETRAINED_WEIGHTS_DIR
    try:
        allowed_directory = directory.resolve(strict=True)
        if not allowed_directory.is_dir():
            logger.warning("Pretrained weights directory is not a directory: %s", directory)
            return []
        paths = sorted(allowed_directory.iterdir(), key=lambda path: path.name)
    except FileNotFoundError:
        return []
    except OSError as exc:
        logger.warning("Cannot list pretrained weights in %s: %s", directory, exc)
        return []

    candidates, seen = [], set()
    for path in paths:
        if path.suffix.lower() not in {".pth", ".pt"}:
            continue
        try:
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(allowed_directory):
                logger.warning("Ignoring pretrained weights outside allowed directory: %s", path)
                continue
            if not resolved.is_file() or resolved in seen:
                continue
            seen.add(resolved)
            arch = _checkpoint_architecture(resolved)
            if arch is None:
                logger.warning("Ignoring incompatible ResNet backbone weights: %s", path)
                continue
            candidates.append({"arch": arch, "weights": str(resolved)})
        except Exception as exc:
            logger.warning("Cannot inspect pretrained weights %s: %s", path, exc)
    return sorted(candidates, key=lambda candidate: (candidate["arch"], candidate["weights"]))
