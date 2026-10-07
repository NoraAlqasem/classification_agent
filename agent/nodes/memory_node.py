from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console

from agent.state import AgentState
from agent.session_summary import write_session_summary
from memory.experiment_store import ExperimentStore, experiment_score
from tools.config_tools import TUNABLE_CONFIG_KEYS

console = Console()


def memory_node(state: AgentState) -> AgentState:
    """Save accuracy evidence and completed before/after optimisation lessons."""
    console.rule("[bold]Memory — Saving Experiment[/bold]")
    attempt = state.get("attempt_number", 1)
    config = state.get("current_config", {})
    metrics = state.get("metrics", {})
    training = state.get("training_result", {})
    target_metric = state.get("target_metric", "mean_acc")
    decision_split = state.get("decision_split", "val")
    test_csv = state.get("test_csv") or config.get("test_csv", "")
    run_dir = state.get("run_dir") or metrics.get("run_dir") or training.get("run_dir", "")
    experiment: dict[str, Any] = {
        "attempt": attempt, "timestamp": datetime.now(timezone.utc).isoformat(),
        "train_csv": config.get("train_csv") or state.get("train_csv", ""),
        "val_csv": config.get("val_csv") or state.get("val_csv", ""),
        "test_csv": test_csv, "decision_split": decision_split,
        "test_result": state.get("test_result", {}),
        "heads": config.get("heads") or state.get("heads", []),
        "arch": config.get("arch"), "weights": config.get("weights"),
        "target_metric": target_metric, "target_value": state.get("target_value"),
        "config": config, "metrics": deepcopy(metrics),
        "evaluation": state.get("evaluation_result", {}),
        "issues_detected": state.get("issues_detected", []),
        "optimisation_rationale": state.get("optimisation_rationale", "Initial run."),
        "run_dir": run_dir, "training_status": training.get("status"),
        "error": state.get("error") or metrics.get("error") or training.get("error"),
        "best_weights": training.get("best_weights", ""),
        "last_weights": training.get("last_weights", ""),
        "target_reached": state.get("target_reached", False),
    }
    score = experiment_score(experiment, target_metric)
    experiment["score"] = score
    history = state.get("experiment_history", [])
    previous_scores = [value for entry in history
                       if ExperimentStore._comparison_key(entry) == ExperimentStore._comparison_key(experiment)
                       and (value := experiment_score(entry, target_metric, decision_split, test_csv or None)) is not None]
    experiment["improved"] = score is not None and (not previous_scores or score > max(previous_scores))

    # The configured output is a session directory; its parent is the ResNet store.
    store = ExperimentStore(Path(config["out"]).parent if config.get("out") else None)
    store.save_experiment(experiment)
    if history:
        previous = history[-1]
        before = experiment_score(previous, target_metric, decision_split, test_csv or None)
        differences = {
            key: {"before": previous.get("config", {}).get(key), "after": config.get(key)}
            for key in sorted(TUNABLE_CONFIG_KEYS | {"arch", "weights"})
            if previous.get("config", {}).get(key) != config.get(key)
        }
        if (differences and before is not None and score is not None
                and ExperimentStore._comparison_key(previous) == ExperimentStore._comparison_key(experiment)):
            previous_issues = previous.get("issues_detected") or previous.get("evaluation", {}).get("issues", [])
            lesson = {
                "attempt": attempt, "target_metric": target_metric,
                "issue": ", ".join(map(str, previous_issues)),
                "change": state.get("optimisation_rationale", "Configuration updated."),
                "config_diff": differences, "score_before": before, "score_after": score,
                "outcome": "compared",
                "improved": score > before,
                "train_csv": experiment["train_csv"], "val_csv": experiment["val_csv"],
                "test_csv": test_csv, "decision_split": decision_split,
                "heads": experiment["heads"], "arch": config.get("arch"), "weights": config.get("weights"),
                "model_before": {key: previous.get("config", {}).get(key) for key in ("arch", "weights")},
                "model_after": {key: config.get(key) for key in ("arch", "weights")},
            }
            store.add_lesson(lesson)

    valid = [entry for entry in [*history, experiment]
             if ExperimentStore._comparison_key(entry) == ExperimentStore._comparison_key(experiment)
             and experiment_score(entry, target_metric, decision_split, test_csv or None) is not None]
    best = max(valid, key=lambda entry: experiment_score(entry, target_metric), default={})
    summary_path = ""
    session_dir = state.get("session_dir") or config.get("out")
    if session_dir:
        try:
            summary_path = str(write_session_summary(
                session_dir, [*history, experiment], status="In Progress",
                heads=experiment["heads"], target_metric=target_metric,
                target_value=state.get("target_value"), decision_split=decision_split,
            ))
        except (OSError, ValueError, TypeError) as exc:
            console.print(f"Unable to update session summary: {exc}", style="yellow", markup=False)
    console.print(f"[green]Experiment {attempt} saved.[/green] {target_metric}={score}; best={best.get('score', 'unavailable')}")
    return {"experiment_history": [experiment], "best_experiment": best,
            "session_summary_path": summary_path}
