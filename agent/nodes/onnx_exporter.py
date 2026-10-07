from __future__ import annotations

from pathlib import Path

from rich.console import Console

from agent.state import AgentState
from memory.experiment_store import experiment_score
from training.artifact_paths import checkpoint_run_dir, find_weights_file
from training.export import export_resnet_onnx

console = Console()


def _checkpoint_path(path: str) -> Path:
    checkpoint = Path(path)
    return find_weights_file(checkpoint_run_dir(checkpoint), checkpoint.name)


def _resolve_best_weights(best_exp: dict, training_result: dict) -> str | None:
    """Prefer the session's best checkpoint, then the latest attempt."""
    candidates: list[Path] = []
    if best_exp.get("best_weights"):
        candidates.append(_checkpoint_path(best_exp["best_weights"]))
    if best_exp.get("run_dir"):
        candidates.append(find_weights_file(best_exp["run_dir"], "best_model.pth"))
    if training_result.get("best_weights"):
        candidates.append(_checkpoint_path(training_result["best_weights"]))
    if training_result.get("run_dir"):
        candidates.append(find_weights_file(training_result["run_dir"], "best_model.pth"))
    if training_result.get("last_weights"):
        candidates.append(_checkpoint_path(training_result["last_weights"]))
    if training_result.get("run_dir"):
        candidates.append(find_weights_file(training_result["run_dir"], "last_model.pth"))
    return next((str(path) for path in candidates if path.is_file()), None)


def onnx_exporter_node(state: AgentState) -> AgentState:
    """Export the best multi-head ResNet checkpoint if requested."""
    export_format = state.get("export_format", "onnx")
    if export_format == "none":
        console.print("[cyan]Model export skipped by configuration.[/cyan]")
        return {"onnx_path": ""}
    console.rule("[bold]ResNet ONNX Export[/bold]")
    if export_format != "onnx":
        return {"onnx_path": "", "error": f"Unsupported ResNet export format: {export_format}"}
    best_exp = state.get("best_experiment") or {}
    if experiment_score(best_exp, state.get("target_metric", "mean_acc")) is None:
        return {"onnx_path": "", "error": "No successful ResNet checkpoint with valid metrics is available for export"}
    weights = _resolve_best_weights(best_exp, {})
    if weights is None:
        return {"onnx_path": "", "error": "No ResNet checkpoint found for export"}
    config = best_exp.get("config") or state.get("current_config") or {}
    try:
        path = export_resnet_onnx(weights, config)
        console.print(f"[green]Exported and validated:[/green] {path}")
        return {"onnx_path": path}
    except Exception as exc:
        console.print(f"[red]Export failed: {exc}[/red]")
        return {"onnx_path": "", "error": f"Export failed: {exc}"}
