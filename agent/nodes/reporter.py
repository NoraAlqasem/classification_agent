from __future__ import annotations

from rich.console import Console
from rich.table import Table

from agent.state import AgentState
from agent.session_summary import write_session_summary
from memory.experiment_store import DEFAULT_RUNS_DIR
from memory.session_paths import allocate_session_directory

console = Console()


def collect_session_errors(state: AgentState) -> list[str]:
    """Retain failure diagnostics in the session's single summary."""
    errors: list[str] = []

    def add(value: object) -> None:
        if value and str(value) not in errors:
            errors.append(str(value))

    for error in state.get("validation_report", {}).get("errors", []):
        add(error)
    for experiment in state.get("experiment_history", []):
        attempt = experiment.get("attempt")
        details = [experiment.get("error"), experiment.get("metrics", {}).get("error")]
        for detail in details:
            if detail:
                add(f"Attempt {attempt}: {detail}")
        if detail := experiment.get("evaluation", {}).get("llm_error"):
            add(f"Attempt {attempt} evaluation: {detail}")
    for key in ("error", "export_error", "onnx_error", "optimisation_error"):
        add(state.get(key))
    return errors


def _display(value: object) -> str:
    if value is None or value == "":
        return "—"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def reporter_node(state: AgentState) -> AgentState:
    """Finalize session_summary.md locally without another LLM request."""
    console.rule("[bold]Session Summary[/bold]")
    history = state.get("experiment_history", [])
    best = state.get("best_experiment", {})
    metrics = best.get("metrics", {})
    config = state.get("current_config", {})
    session_dir = state.get("session_dir") or config.get("out") or str(
        allocate_session_directory(DEFAULT_RUNS_DIR)
    )
    valid = state.get("is_valid", True)
    reason = state.get("stop_reason", "unknown") if valid else "validation_failed"
    fatal_error = any(state.get(key) for key in ("error", "optimisation_error", "export_error", "onnx_error"))
    session_status = (
        "Interrupted" if reason in {"interrupted", "cancelled"} else
        "Failed" if not valid or reason in {"error", "validation_failed", "unstable"} or fatal_error or not best else
        "Completed" if reason in {"target_reached", "max_attempts", "no_allowed_changes"} else "Stopped"
    )
    target_metric = state.get("target_metric", "mean_acc")
    decision_split = best.get("decision_split") or state.get("decision_split", "val")
    summary_path = write_session_summary(
        session_dir, history, status=session_status,
        heads=config.get("heads") or state.get("heads", []),
        target_metric=target_metric, target_value=state.get("target_value", 0.85),
        decision_split=decision_split, stop_reason=reason,
        errors=collect_session_errors(state), onnx_path=state.get("onnx_path", ""),
        last_model_path=best.get("last_weights", ""),
        llm_allowed_config_keys=state.get("llm_allowed_config_keys"),
    )
    table = Table(title="Training Session Summary")
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    for label, value in (
        ("Status", session_status), ("Stop reason", reason), ("Attempts", len(history)),
        ("Target metric", target_metric), ("Decision split", decision_split),
        ("Best target accuracy", metrics.get(target_metric)),
        ("Best model", best.get("best_weights", "")),
        ("ONNX export", state.get("onnx_path", "")), ("Session summary", str(summary_path)),
    ):
        table.add_row(label, _display(value))
    console.print(table)
    return {"session_dir": session_dir, "session_summary_path": str(summary_path)}
