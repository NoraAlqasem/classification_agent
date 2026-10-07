from __future__ import annotations

from pathlib import Path

import yaml
from rich.console import Console

from agent.state import AgentState
from agent.session_summary import write_session_summary
from memory.session_paths import allocate_session_directory
from tools.config_tools import sanitize_resnet_config

console = Console()
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def config_generator_node(state: AgentState) -> dict:
    """Build the first attempt from YAML; LLM tuning follows measured results."""
    console.rule("[bold]ResNet YAML Training Configuration[/bold]")
    with (PROJECT_ROOT / "config" / "defaults.yaml").open() as handle:
        defaults = yaml.safe_load(handle)["training"]
    report = state["validation_report"]
    target_metric = state.get("target_metric", "mean_acc")
    monitor = "mean" if target_metric == "mean_acc" else target_metric.removesuffix("_acc")
    session_dir = state.get("session_dir") or str(
        allocate_session_directory(PROJECT_ROOT / "runs" / "resnet")
    )
    config = {
        **defaults,
        "arch": state["model_type"],
        "weights": state["weights_path"],
        "train_csv": report.get("train_csv", state["train_csv"]),
        "val_csv": report.get("val_csv", state["val_csv"]),
        "data_root": state.get("data_root"),
        "heads": state["heads"],
        "monitor": monitor,
        "head_loss_weights": state.get("head_loss_weights", {}),
        "alpha_json": state.get("alpha_json"),
        "nproc_per_node": state.get("nproc_per_node", 1),
        "out": session_dir,
        "name": f"exp_{state.get('attempt_number', 0) + 1:02d}",
    }
    config, _ = sanitize_resnet_config(config)
    summary_path = ""
    try:
        summary_path = str(write_session_summary(
            session_dir, state.get("experiment_history", []), status="In Progress",
            heads=state["heads"], target_metric=target_metric,
            target_value=state.get("target_value"), decision_split=state.get("decision_split", "val"),
        ))
    except (OSError, ValueError, TypeError) as exc:
        console.print(f"Unable to create session summary: {exc}", style="yellow", markup=False)
    console.print("Using config/defaults.yaml for the first training attempt.")
    console.print(
        f"{config['arch']}: {config['epochs']} epochs, batch {config['batch_size']}/GPU, "
        f"lr={config['lr']}, monitor={config['monitor']}"
    )
    return {
        "current_config": config,
        "config_history": [{"attempt": state.get("attempt_number", 0) + 1, "config": config}],
        "session_dir": session_dir,
        "session_summary_path": summary_path,
        "messages": [],
    }
