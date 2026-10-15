from __future__ import annotations

import json
import math
import hashlib
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memory.experiment_format import (
    deserialize_experiment, deserialize_public_experiment, normalize_internal_experiment, serialize_experiment,
)
from memory.session_paths import session_number

DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[1] / "runs" / "resnet"


def _finite_accuracy(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and 0 <= number <= 1 else None


def _test_counts_error(result: dict[str, Any]) -> str | None:
    """Accept only accuracy measured over every labelled test image."""
    if result.get("status") != "ok" or result.get("full_coverage") is not True:
        return result.get("error") or "Test inference did not complete with full coverage"
    if result.get("nonfinite_accuracy") or result.get("nonfinite_loss"):
        return "Test inference contains nonfinite metrics"
    counts = [result.get(key) for key in ("total_images", "evaluated_images", "matched_images")]
    if any(type(count) is not int or count <= 0 for count in counts) or len(set(counts)) != 1:
        return "Test image counts are missing, inconsistent, or incomplete"
    total = counts[0]
    heads = result.get("heads")
    if not isinstance(heads, dict) or not heads:
        return "Test per-head accuracy evidence is missing"
    accuracies = []
    for head, evidence in heads.items():
        if not isinstance(evidence, dict):
            return f"Test head '{head}' has invalid evidence"
        correct, count = evidence.get("correct"), evidence.get("count")
        accuracy = _finite_accuracy(evidence.get("acc"))
        primary = _finite_accuracy(result.get(f"{head}_acc"))
        if (type(correct) is not int or not 0 <= correct <= total or type(count) is not int
                or count != total or evidence.get("coverage") != 1.0
                or evidence.get("skipped_by_threshold", 0) != 0):
            return f"Test head '{head}' was not scored on every test image"
        for key in ("total", "evaluated", "matched"):
            if key in evidence and (type(evidence[key]) is not int or evidence[key] != total):
                return f"Test head '{head}' has incomplete {key} counts"
        if (accuracy is None or primary is None or not math.isclose(accuracy, correct / total, abs_tol=1e-12)
                or not math.isclose(primary, accuracy, abs_tol=1e-12)):
            return f"Test head '{head}' accuracy does not match its full-denominator counts"
        accuracies.append(accuracy)
    mean = _finite_accuracy(result.get("mean_acc"))
    if mean is None or not math.isclose(mean, sum(accuracies) / len(accuracies), abs_tol=1e-12):
        return "Test mean accuracy does not match its per-head accuracies"
    return None


def _is_macro_metric(metric: str | None, heads: Any = None) -> bool:
    if not isinstance(metric, str):
        return False
    if metric == "macro_acc":
        return True
    if heads is not None and not isinstance(heads, (dict, list, tuple, set, frozenset)):
        heads = ()
    # A literal head named make_macro still owns make_macro_acc as its
    # ordinary accuracy. Input validation rejects ambiguous head pairs.
    if heads is not None and metric.endswith("_acc") and metric[:-4] in heads:
        return False
    return metric.endswith("_macro_acc") and (heads is None or metric[:-10] in heads)


def _macro_accuracy_error(metrics: dict[str, Any], *, require_class_counts: bool) -> str | None:
    """Require recorded macro scores; test scores also need a complete class partition."""
    heads = metrics.get("heads")
    if not isinstance(heads, dict) or not heads:
        return "Macro accuracy has no per-head evidence"
    macros = []
    for head, evidence in heads.items():
        if not isinstance(head, str) or not head or not isinstance(evidence, dict):
            return "Macro accuracy has invalid per-head evidence"
        macro = _finite_accuracy(evidence.get("macro_acc"))
        primary = _finite_accuracy(metrics.get(f"{head}_macro_acc"))
        if macro is None or primary is None or not math.isclose(macro, primary, abs_tol=1e-12):
            return f"Head '{head}' macro accuracy is missing, invalid, or inconsistent"
        if require_class_counts:
            rows = evidence.get("per_class_accuracy")
            if not isinstance(rows, list) or not rows or evidence.get("omitted_classes", 0) != 0:
                return f"Test head '{head}' macro accuracy requires complete per-class counts"
            labels, total, correct, recalls = set(), 0, 0, []
            for row in rows:
                if not isinstance(row, dict) or set(row) != {"class", "correct", "total", "accuracy"}:
                    return f"Test head '{head}' has invalid per-class macro evidence"
                label = row["class"]
                if not isinstance(label, str) or not label.strip() or label != label.strip() or label in labels:
                    return f"Test head '{head}' has invalid or duplicate per-class labels"
                labels.add(label)
                class_total, class_correct = row["total"], row["correct"]
                recall = _finite_accuracy(row["accuracy"])
                if (type(class_total) is not int or class_total <= 0 or type(class_correct) is not int
                        or not 0 <= class_correct <= class_total or recall is None
                        or not math.isclose(recall, class_correct / class_total, abs_tol=1e-12)):
                    return f"Test head '{head}' has invalid per-class accuracy counts"
                total += class_total
                correct += class_correct
                recalls.append(recall)
            if total != evidence.get("count") or correct != evidence.get("correct"):
                return f"Test head '{head}' per-class counts do not cover its full-denominator result"
            if not math.isclose(macro, sum(recalls) / len(recalls), abs_tol=1e-12):
                return f"Test head '{head}' macro accuracy does not match its per-class counts"
        macros.append(macro)
    mean = _finite_accuracy(metrics.get("macro_acc"))
    if mean is None or not math.isclose(mean, sum(macros) / len(macros), abs_tol=1e-12):
        return "Macro accuracy does not match its equally weighted per-head macro accuracies"
    return None


def test_evaluation_error(metrics: dict[str, Any], test_result: dict[str, Any] | None = None,
                          test_csv: str | None = None, *, target_metric: str | None = None) -> str | None:
    """Explain why a mandatory test result cannot drive a training decision."""
    if (metrics.get("decision_split") != "test" or metrics.get("split") != "test"
            or metrics.get("test_evaluated") is not True):
        return metrics.get("error") or "The required test checkpoint evaluation is unavailable"
    error = _test_counts_error(metrics)
    if error:
        return error
    if not isinstance(test_result, dict):
        return "The required test inference result is unavailable"
    if test_result.get("status") != "ok":
        return test_result.get("error") or "The required test inference result failed"
    result = test_result.get("metrics", test_result)
    if not isinstance(result, dict):
        return "Test inference metrics are unavailable"
    error = _test_counts_error(result)
    if error:
        return error
    if set(metrics["heads"]) != set(result["heads"]):
        return "Decision metrics do not match the tested heads"
    for key in ("total_images", "evaluated_images", "matched_images"):
        if metrics[key] != result[key]:
            return "Decision image counts do not match test inference"
    for key in ["mean_acc", *(f"{head}_acc" for head in result["heads"])]:
        if not math.isclose(float(metrics[key]), float(result[key]), abs_tol=1e-12):
            return f"Decision metric '{key}' does not match test inference"
    if _is_macro_metric(target_metric, metrics["heads"]):
        for source in (metrics, result):
            error = _macro_accuracy_error(source, require_class_counts=True)
            if error:
                return error
        for key in ["macro_acc", *(f"{head}_macro_acc" for head in result["heads"])]:
            if not math.isclose(float(metrics[key]), float(result[key]), abs_tol=1e-12):
                return f"Decision metric '{key}' does not match test inference"
        for head in result["heads"]:
            # Ordering is presentation only; class identities and full counts
            # must agree between the decision record and raw inference proof.
            decision_rows = {row["class"]: (row["correct"], row["total"])
                             for row in metrics["heads"][head]["per_class_accuracy"]}
            inference_rows = {row["class"]: (row["correct"], row["total"])
                              for row in result["heads"][head]["per_class_accuracy"]}
            if decision_rows != inference_rows:
                return f"Decision per-class counts for head '{head}' do not match test inference"
    source_csv = result.get("test_csv") or test_result.get("test_csv") or test_result.get("csv")
    expected_csv = test_csv or metrics.get("test_csv")
    if expected_csv and (not source_csv or Path(source_csv).resolve() != Path(expected_csv).resolve()):
        return "Test inference used a different CSV from the requested test split"
    return None


def decision_score(metrics: dict[str, Any], training_status: str | None, metric: str,
                   decision_split: str | None = None, test_result: dict[str, Any] | None = None,
                   test_csv: str | None = None) -> float | None:
    """Return the exact verified score for the requested decision split."""
    split = decision_split or metrics.get("decision_split", "val")
    if training_status != "success" or metrics.get("status") != "ok" or split not in {"val", "test"}:
        return None
    if metrics.get("nonfinite_loss") or metrics.get("nonfinite_accuracy"):
        return None
    heads = metrics.get("heads", {})
    if not isinstance(heads, dict) or any(not isinstance(value, dict) for value in heads.values()):
        return None
    if (metric.endswith("_macro_acc") and metric[:-4] not in heads
            and metric[:-10] not in heads):
        return None
    if split == "test":
        if test_evaluation_error(metrics, test_result, test_csv, target_metric=metric):
            return None
        if metric not in {"mean_acc", "macro_acc"} and not (
            _is_macro_metric(metric, metrics["heads"])
            or (metric.endswith("_acc") and metric[:-4] in metrics["heads"])
        ):
            return None
    elif metrics.get("decision_split", "val") != "val" or metrics.get("split", "val") != "val":
        return None
    elif _is_macro_metric(metric, metrics.get("heads", {})):
        if _macro_accuracy_error(metrics, require_class_counts=False):
            return None
        if metric != "macro_acc" and metric[:-10] not in metrics["heads"]:
            return None
    losses = [metrics.get(key) for key in ("train_loss", "val_loss", "best_val_loss")]
    losses.extend(value.get("loss") for value in heads.values())
    losses.extend(metrics.get("train_loss_history", []))
    losses.extend(metrics.get("val_loss_history", []))
    for loss in losses:
        if loss is not None:
            try:
                if not math.isfinite(float(loss)):
                    return None
            except (TypeError, ValueError):
                return None
    return _finite_accuracy(metrics.get(metric))


def _clean_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    return value


def experiment_score(experiment: dict[str, Any], target_metric: str | None = None,
                     decision_split: str | None = None, test_csv: str | None = None) -> float | None:
    """Only successful attempts with finite logs can participate in best selection."""
    experiment = deserialize_experiment(experiment)
    split = experiment.get("decision_split", "val")
    if decision_split is not None and split != decision_split:
        return None
    config = experiment.get("config", {})
    actual_test_csv = experiment.get("test_csv") or config.get("test_csv")
    if test_csv is not None and (not actual_test_csv or Path(actual_test_csv).resolve() != Path(test_csv).resolve()):
        return None
    key = target_metric or experiment.get("target_metric", "mean_acc")
    if target_metric is not None and experiment.get("target_metric") != target_metric:
        return None
    return decision_score(experiment.get("metrics", {}), experiment.get("training_status"), key,
                          split, experiment.get("test_result"), actual_test_csv)


class ExperimentStore:
    """ResNet experiment history, kept separate from historical YOLO runs."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir is not None else DEFAULT_RUNS_DIR
        self.exp_dir = self.base_dir / "experiments"
        self.kb_path = self.base_dir / "knowledge_base.json"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        if not self.kb_path.exists():
            self._write_kb({"schema_version": 3, "experiments": [], "lessons": [], "best_configs": {}})

    def _read_kb(self) -> dict[str, Any]:
        with self.kb_path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def _write_kb(self, data: dict[str, Any]) -> None:
        temporary = self.kb_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(_clean_json(data), indent=2, default=str, allow_nan=False), encoding="utf-8")
        temporary.replace(self.kb_path)

    @staticmethod
    def _context_key(experiment: dict[str, Any]) -> str:
        experiment = deserialize_experiment(experiment)
        config = experiment.get("config", {})
        return json.dumps([
            config.get("arch"), experiment.get("target_metric"),
            str(Path(path).resolve()) if (path := config.get("train_csv") or experiment.get("train_csv")) else None,
            str(Path(path).resolve()) if (path := config.get("val_csv") or experiment.get("val_csv")) else None,
            config.get("heads") or experiment.get("heads", []),
            experiment.get("decision_split", "val"),
            str(Path(path).resolve()) if (path := experiment.get("test_csv") or config.get("test_csv")) else None,
        ], separators=(",", ":"))

    @staticmethod
    def _comparison_key(experiment: dict[str, Any]) -> str:
        """Match a session's task and score evidence across ResNet choices.

        Architecture stays in the persistent key, while session history can
        compare models trained and tested on the same classification task.
        """
        return json.dumps(json.loads(ExperimentStore._context_key(experiment))[1:], separators=(",", ":"))

    def save_experiment(self, experiment: dict[str, Any]) -> Path:
        experiment = normalize_internal_experiment(experiment)
        attempt = experiment.get("attempt", 0)
        config = experiment.get("config", {})
        session_path = Path(config["out"]) if config.get("out") else None
        if experiment.get("run_dir"):
            destination = Path(experiment["run_dir"])
        elif session_path is not None and config.get("name"):
            destination = session_path / config["name"]
        else:
            # Old callers may have no training attempt directory. Their
            # central history remains readable and retains its old fallback.
            destination = self.exp_dir
        if session_path is not None and destination != self.exp_dir:
            resolved = destination.resolve()
            if resolved == session_path.resolve() or not resolved.is_relative_to(session_path.resolve()):
                raise ValueError("Experiment run directory must be inside its requested session directory")
        destination.mkdir(parents=True, exist_ok=True)
        number = session_number(config.get("out", "")) or experiment.get("session_id")
        if number is not None and (type(number) is not int or number < 1):
            raise ValueError("Experiment session_id must be a positive integer")
        if destination != self.exp_dir:
            exp_path = destination / f"exp_{attempt}.json"
        else:
            # Standalone historical records have no attempt folder, so retain
            # their timestamp fallback without replacing earlier records.
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            exp_path = destination / f"exp_{attempt:02d}_{timestamp}.json"
        document = _clean_json(serialize_experiment(experiment, session_id=number))
        payload = json.dumps(document, indent=2, default=str, allow_nan=False) + "\n"
        checksum = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination,
                                         prefix=".experiment-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        try:
            # Linking a completed file creates the destination atomically and
            # raises FileExistsError instead of replacing an earlier attempt.
            os.link(temporary, exp_path)
        finally:
            temporary.unlink(missing_ok=True)
        score = experiment_score(experiment)
        kb = self._read_kb()
        kb.setdefault("experiments", []).append({
            "attempt": attempt, "timestamp": experiment.get("timestamp"),
            "file": str(exp_path), "arch": experiment.get("config", {}).get("arch"),
            "weights": experiment.get("config", {}).get("weights"),
            "target_metric": experiment.get("target_metric"), "score": score,
            "improved": experiment.get("improved", False),
            "training_status": experiment.get("training_status"),
            "train_csv": experiment.get("train_csv") or experiment.get("config", {}).get("train_csv"),
            "val_csv": experiment.get("val_csv") or experiment.get("config", {}).get("val_csv"),
            "test_csv": experiment.get("test_csv") or experiment.get("config", {}).get("test_csv"),
            "decision_split": experiment.get("decision_split", "val"),
            "heads": experiment.get("heads") or experiment.get("config", {}).get("heads", []),
            "record": _clean_json(experiment), "sha256": checksum,
        })
        context_key = self._context_key(experiment)
        best_configs = kb.setdefault("best_configs", {})
        previous = best_configs.get(context_key, {}).get("score")
        if score is not None and (previous is None or score > previous):
            best_configs[context_key] = {
                "score": score, "target_metric": experiment.get("target_metric"),
                "decision_split": experiment.get("decision_split", "val"),
                "test_csv": experiment.get("test_csv") or experiment.get("config", {}).get("test_csv"),
                "config": experiment.get("config", {}), "file": str(exp_path),
            }
        self._write_kb(kb)
        return exp_path

    def add_lesson(self, lesson: dict[str, Any]) -> None:
        kb = self._read_kb()
        entry = dict(lesson, timestamp=datetime.now(timezone.utc).isoformat())
        kb.setdefault("lessons", []).append(entry)
        self._write_kb(kb)

    def get_best_experiment(self, target_metric: str = "mean_acc", arch: str | None = None,
                            train_csv: str | None = None, val_csv: str | None = None,
                            heads: list[str] | None = None, test_csv: str | None = None,
                            decision_split: str = "val") -> dict[str, Any] | None:
        """Choose the best target score, with optional model and dataset identity filters."""
        candidates: list[dict[str, Any]] = []
        for experiment in self.load_all_experiments():
            config = experiment.get("config", {})
            if arch is not None and config.get("arch") != arch:
                continue
            if heads is not None and config.get("heads") != heads:
                continue
            matches = True
            for key, expected in (("train_csv", train_csv), ("val_csv", val_csv), ("test_csv", test_csv)):
                actual = experiment.get(key) or config.get(key)
                if expected is not None and (not actual or Path(actual).resolve() != Path(expected).resolve()):
                    matches = False
            if matches and experiment_score(experiment, target_metric, decision_split, test_csv) is not None:
                candidates.append(experiment)
        return max(candidates, key=lambda experiment: experiment_score(experiment, target_metric), default=None)

    def load_all_experiments(self) -> list[dict[str, Any]]:
        experiments: list[dict[str, Any]] = []
        for entry in self._read_kb().get("experiments", []):
            try:
                raw = Path(entry["file"]).read_bytes()
                document = json.loads(raw)
                internal = entry.get("record")
                if isinstance(internal, dict) and entry.get("sha256") == hashlib.sha256(raw).hexdigest():
                    try:
                        normalized = normalize_internal_experiment(internal)
                        if _clean_json(serialize_experiment(normalized)) == document:
                            experiments.append(normalized)
                            continue
                    except (KeyError, ValueError, TypeError, AttributeError):
                        pass
                # Once a summary is bound to an internal record, editing its
                # shape cannot turn it into trusted historical proof.
                experiments.append(deserialize_public_experiment(document) if "record" in entry
                                   else deserialize_experiment(document))
            except (OSError, KeyError, ValueError, TypeError, AttributeError):
                continue
        return experiments

    def get_lessons(self, issue_filter: str | None = None, decision_split: str | None = None,
                    test_csv: str | None = None) -> list[dict[str, Any]]:
        lessons = self._read_kb().get("lessons", [])
        return [lesson for lesson in lessons
                if (not issue_filter or issue_filter.lower() in str(lesson.get("issue", "")).lower())
                and (decision_split is None or lesson.get("decision_split", "val") == decision_split)
                and (test_csv is None or (lesson.get("test_csv")
                     and Path(lesson["test_csv"]).resolve() == Path(test_csv).resolve()))]
