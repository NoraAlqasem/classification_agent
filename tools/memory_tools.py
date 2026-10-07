from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from memory.experiment_store import DEFAULT_RUNS_DIR, ExperimentStore

_DEFAULT_DIR = str(DEFAULT_RUNS_DIR)


def _matches_decision_context(entry: dict[str, Any], decision_split: str, test_csv: str | None) -> bool:
    if entry.get("decision_split", "val") != decision_split:
        return False
    actual_csv = entry.get("test_csv") or entry.get("config", {}).get("test_csv")
    return test_csv is None or bool(actual_csv and Path(actual_csv).resolve() == Path(test_csv).resolve())


def _knowledge_base(runs_dir: str) -> dict[str, Any]:
    path = Path(runs_dir) / "knowledge_base.json"
    if not path.is_file():
        return {"experiments": [], "lessons": [], "best_configs": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"experiments": [], "lessons": [], "best_configs": {}, "error": "Could not read knowledge base"}


@tool
def load_knowledge_base(runs_dir: str = _DEFAULT_DIR, decision_split: str = "val",
                         test_csv: str | None = None) -> str:
    """Load history for the current decision split and exact test CSV when using test."""
    knowledge = _knowledge_base(runs_dir)
    for key in ("experiments", "lessons"):
        knowledge[key] = [entry for entry in knowledge.get(key, [])
                          if _matches_decision_context(entry, decision_split, test_csv)]
    knowledge["best_configs"] = {
        key: entry for key, entry in knowledge.get("best_configs", {}).items()
        if _matches_decision_context(entry, decision_split, test_csv)
    }
    return json.dumps(knowledge, indent=2, default=str)


@tool
def get_best_known_config(arch: str = "resnet50", runs_dir: str = _DEFAULT_DIR,
                          target_metric: str = "mean_acc", train_csv: str | None = None,
                          val_csv: str | None = None, heads: list[str] | None = None,
                          test_csv: str | None = None, decision_split: str = "val") -> str:
    """Return the best successful ResNet config for the exact metric and dataset.

    Provide all current CSVs, decision_split and head names to avoid comparing
    validation results with test results or scores from different datasets.
    """
    if not (Path(runs_dir) / "knowledge_base.json").is_file():
        return json.dumps({"result": "No ResNet history available."})
    best = ExperimentStore(runs_dir).get_best_experiment(
        target_metric=target_metric, arch=arch, train_csv=train_csv, val_csv=val_csv, heads=heads,
        test_csv=test_csv, decision_split=decision_split,
    )
    if best is None:
        return json.dumps({"result": "No matching successful experiment."})
    return json.dumps({"score": best.get("score"), "target_metric": target_metric,
                       "decision_split": best.get("decision_split", "val"), "test_csv": best.get("test_csv"),
                       "config": best.get("config", {}), "best_weights": best.get("best_weights"),
                       "run_dir": best.get("run_dir")}, indent=2, default=str)


@tool
def search_lessons_by_issue(issue_keyword: str, runs_dir: str = _DEFAULT_DIR,
                            decision_split: str = "val", test_csv: str | None = None) -> str:
    """Find completed before/after lessons for a previous classification issue."""
    lessons = _knowledge_base(runs_dir).get("lessons", [])
    matches = [lesson for lesson in lessons
               if issue_keyword.lower() in str(lesson.get("issue", "")).lower()
               and _matches_decision_context(lesson, decision_split, test_csv)]
    return json.dumps({"matches": matches[-10:]}, indent=2, default=str)


@tool
def get_attempt_history_summary(runs_dir: str = _DEFAULT_DIR, decision_split: str = "val",
                                 test_csv: str | None = None) -> str:
    """Summarise ResNet attempts with metric identity, target score, and status."""
    experiments = _knowledge_base(runs_dir).get("experiments", [])
    return json.dumps({"summary": [
        {key: experiment.get(key) for key in
         ("attempt", "arch", "target_metric", "score", "training_status", "improved", "train_csv", "val_csv", "test_csv", "decision_split", "heads")}
        for experiment in experiments if _matches_decision_context(experiment, decision_split, test_csv)
    ]}, indent=2, default=str)
