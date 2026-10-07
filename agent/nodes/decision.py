from __future__ import annotations

from typing import Any

from rich.console import Console

from agent.state import AgentState
from agent.nodes.evaluator import _has_nonfinite_losses, target_is_reached
from memory.experiment_store import test_evaluation_error

console = Console()


def _is_truly_diverged(metrics: dict[str, Any], training_result: dict[str, Any]) -> bool:
    """Nonfinite recorded losses prove instability; low accuracy does not."""
    return (
        training_result.get("status") == "success"
        and metrics.get("status") == "ok"
        and _has_nonfinite_losses(metrics)
    )


def decision_node(state: AgentState) -> AgentState:
    """Stop at a confirmed target, the attempt limit, or nonfinite losses."""
    console.rule("[bold]Decision[/bold]")
    attempt = state.get("attempt_number", 0)
    max_attempts = state.get("max_attempts", 5)
    metrics = state.get("metrics", {})
    training_result = state.get("training_result", {})
    metric = state.get("target_metric", "mean_acc")
    target = state.get("target_value", 0.85)
    decision_split = state.get("decision_split", "val")
    test_csv = state.get("test_csv") or state.get("current_config", {}).get("test_csv")
    test_error = (test_evaluation_error(metrics, state.get("test_result"), test_csv)
                  if decision_split == "test" else None)
    reached = target_is_reached(metrics, training_result, metric, target,
                                decision_split, state.get("test_result"), test_csv)

    if test_error:
        reason = "error"
        console.print(f"[red]Cannot decide from the required test evaluation: {test_error}[/red]")
    elif reached:
        reason = "target_reached"
        console.print(f"[green]Target {metric} ≥ {target} reached after {attempt} attempt(s).[/green]")
    elif attempt >= max_attempts:
        reason = "max_attempts"
        console.print(f"[yellow]Maximum attempts ({max_attempts}) reached.[/yellow]")
    elif _is_truly_diverged(metrics, training_result):
        reason = "unstable"
        console.print("[red]Nonfinite training losses detected. Stopping this session.[/red]")
    else:
        reason = ""
        console.print(f"[cyan]Continuing optimisation: attempt {attempt}/{max_attempts}, {decision_split} {metric}={metrics.get(metric, 'unavailable')}.[/cyan]")
    result = {"stop_reason": reason, "target_reached": reached}
    if test_error:
        result["error"] = test_error
    return result
