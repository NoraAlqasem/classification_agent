"""Paths shared by training and readers of current or historical artifacts."""
from __future__ import annotations

from pathlib import Path


def weights_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / "weights"


def train_evaluation_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / "train_evaluation"


def _find_file(directory: Path, run_dir: str | Path, filename: str) -> Path:
    current = directory / filename
    if current.is_file():
        return current
    legacy = Path(run_dir) / filename
    return legacy if legacy.is_file() else current


def find_weights_file(run_dir: str | Path, filename: str) -> Path:
    """Prefer weights/ and retain compatibility with a flat historical run."""
    return _find_file(weights_dir(run_dir), run_dir, filename)


def find_training_file(run_dir: str | Path, filename: str) -> Path:
    """Prefer train_evaluation/ and fall back to a historical run's root."""
    return _find_file(train_evaluation_dir(run_dir), run_dir, filename)


def checkpoint_run_dir(checkpoint_path: str | Path) -> Path:
    checkpoint = Path(checkpoint_path)
    return checkpoint.parent.parent if checkpoint.parent.name == "weights" else checkpoint.parent


def checkpoint_training_config(checkpoint_path: str | Path) -> Path:
    return find_training_file(checkpoint_run_dir(checkpoint_path), "training_config.json")


def prepare_new_run_directory(run_dir: str | Path, *, allow_existing: bool = False) -> None:
    """Create native artifact folders without replacing an earlier attempt.

    The agent creates empty weights/ and starts streaming training.log before
    torchrun starts. Those are the only existing entries accepted by the CLI.
    """
    run_dir = Path(run_dir)
    if allow_existing and run_dir.is_dir():
        previous = []
        for entry in run_dir.iterdir():
            if entry.name not in {"weights", "train_evaluation"} or not entry.is_dir() or entry.is_symlink():
                previous.append(entry.name)
                continue
            for child in entry.iterdir():
                if (entry.name != "train_evaluation" or child.name != "training.log"
                        or not child.is_file() or child.is_symlink()):
                    previous.append(f"{entry.name}/{child.name}")
        if previous:
            raise FileExistsError(f"Refusing to overwrite existing run artifacts in {run_dir}: {sorted(previous)}")
    run_dir.mkdir(parents=True, exist_ok=allow_existing)
    weights_dir(run_dir).mkdir(exist_ok=True)
    train_evaluation_dir(run_dir).mkdir(exist_ok=True)
