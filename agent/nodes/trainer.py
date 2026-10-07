from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from rich.console import Console

from agent.state import AgentState
from tools.config_tools import sanitize_resnet_config
from training.artifact_paths import train_evaluation_dir, weights_dir

console = Console()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_SCRIPT = PROJECT_ROOT / "training" / "train_multihead_ddp.py"


def build_training_command(config: dict[str, Any], run_dir: Path) -> list[str]:
    """Translate a validated agent configuration to the supplied DDP script CLI."""
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--nproc_per_node", str(config["nproc_per_node"]), str(TRAINING_SCRIPT),
        "--train-csv", str(config["train_csv"]),
        "--val-csv", str(config["val_csv"]),
        "--arch", config["arch"], "--weights", str(config["weights"]),
        "--heads", *config["heads"],
        "--img-size", *(str(value) for value in config["img_size"]),
        "--run-dir", str(run_dir),
    ]
    for key in (
        "epochs", "batch_size", "num_workers", "lr", "weight_decay",
        "gamma", "patience", "monitor",
    ):
        command.extend(["--" + key.replace("_", "-"), str(config[key])])
    for key in ("data_root", "alpha_json"):
        if config.get(key):
            command.extend(["--" + key.replace("_", "-"), str(config[key])])
    if config.get("head_loss_weights"):
        command.extend([
            "--head-loss-weights",
            *(f"{head}={weight}" for head, weight in config["head_loss_weights"].items()),
        ])
    return command


def _terminate_workers(process: subprocess.Popen) -> None:
    """Stop the torchrun process group, including GPU workers and loaders."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def trainer_node(state: AgentState) -> AgentState:
    """Launch one multi-head ResNet attempt under CUDA/NCCL torchrun."""
    attempt = state.get("attempt_number", 0) + 1
    config: dict[str, Any] = dict(state.get("current_config") or {})
    start = time.monotonic()
    run_dir: Path | None = None
    process: subprocess.Popen | None = None
    training_result: dict[str, Any] = {"status": "error", "attempt": attempt}

    console.rule(f"[bold]ResNet Training — Attempt {attempt}[/bold]")
    try:
        import torch

        config, removed = sanitize_resnet_config(config)
        if removed:
            console.print(f"[yellow]Stripped invalid config keys: {', '.join(removed)}[/yellow]")
        console.print(json.dumps(config, indent=2, default=str))

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by the supplied NCCL DDP training script.")
        requested = config["nproc_per_node"]
        available = torch.cuda.device_count()
        if requested > available:
            raise ValueError(
                f"Requested {requested} GPU processes, but only {available} CUDA devices are visible."
            )
        if not torch.distributed.is_nccl_available():
            raise RuntimeError("This PyTorch installation does not provide NCCL for CUDA DDP training.")

        requested_run_dir = (Path(config["out"]) / config["name"]).resolve()
        requested_run_dir.mkdir(parents=True, exist_ok=False)
        run_dir = requested_run_dir
        weights_dir(run_dir).mkdir()
        train_evaluation_dir(run_dir).mkdir()
        command = build_training_command(config, run_dir)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        tail: list[str] = []
        log_path = train_evaluation_dir(run_dir) / "training.log"
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=PROJECT_ROOT, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                start_new_session=True,
            )
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    console.print(line.rstrip(), markup=False, highlight=False)
                    tail.append(line.rstrip())
                    if len(tail) > 30:
                        tail.pop(0)
                return_code = process.wait()
            finally:
                process.stdout.close()
            if return_code:
                training_result.update({"exit_code": return_code, "log_tail": tail})
                raise RuntimeError(
                    f"ResNet torchrun exited with code {return_code}; see {log_path}."
                )

        best = weights_dir(run_dir) / "best_model.pth"
        last = weights_dir(run_dir) / "last_model.pth"
        if not best.is_file() or not last.is_file():
            raise RuntimeError("Training exited without best_model.pth and last_model.pth checkpoints.")
        training_result.update({
            "status": "success", "best_weights": str(best), "last_weights": str(last),
        })
        console.print(f"[green]Best checkpoint:[/green] {best}")
    except KeyboardInterrupt:
        if process is not None:
            _terminate_workers(process)
        raise
    except Exception as exc:
        if process is not None and process.poll() is None:
            _terminate_workers(process)
        training_result["error"] = str(exc)
        console.print(f"[red]Training failed: {exc}[/red]")

    training_result.update({
        "config": config,
        "run_dir": str(run_dir) if run_dir is not None else "",
        "elapsed_seconds": round(time.monotonic() - start, 1),
    })
    return {
        "attempt_number": attempt,
        "training_result": training_result,
        "run_dir": training_result["run_dir"],
    }
