"""Readable experiment documents and adapters for the agent's flat records."""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from memory.session_paths import session_number


_TRAINING_FIELDS = {
    "total_epochs": ("epochs_completed",), "final_epoch": ("final_epoch",),
    "best_epoch": ("best_epoch",), "best_score": ("best_score",), "loss_epoch": ("loss_epoch",),
    "train_loss": ("loss", "train_final"), "val_loss": ("loss", "val_final"),
    "best_val_loss": ("loss", "best_val"),
    "train_loss_history": ("history", "train_loss"), "val_loss_history": ("history", "val_loss"),
    "mean_acc_history": ("history", "mean_accuracy"), "head_acc_history": ("history", "head_accuracy"),
    "macro_acc_history": ("history", "macro_accuracy"),
    "head_macro_acc_history": ("history", "head_macro_accuracy"),
    "nonfinite_loss": ("nonfinite_loss",), "nonfinite_accuracy": ("nonfinite_accuracy",),
}
_CONFIG_FIELDS = {
    "arch": ("architecture",), "lr": ("optimizer", "lr"),
    "weight_decay": ("optimizer", "weight_decay"), "gamma": ("loss", "gamma"),
    "head_loss_weights": ("loss", "head_weights"), "patience": ("early_stopping", "patience"),
}
_DATA_FIELDS = ("train_csv", "val_csv", "test_csv", "data_root")
_BOOKKEEPING = (
    "score", "improved", "optimisation_rationale", "issues_detected", "error", "target_reached",
)
_FLAT_FIELDS = {
    "attempt", "timestamp", "session_id", "session_dir", "training_status", "config", "metrics",
    "evaluation", "test_result", "heads", "target_metric", "target_value", "decision_split",
    "arch", "weights", "run_dir", "best_weights", "last_weights", *_DATA_FIELDS, *_BOOKKEEPING,
}


def _put(target: dict, path: tuple[str, ...], value: Any) -> None:
    for key in path[:-1]:
        target = target.setdefault(key, {})
    target[path[-1]] = deepcopy(value)


def _mapped(source: dict, mapping: dict[str, tuple[str, ...]]) -> dict:
    target = {}
    for key, path in mapping.items():
        if key in source:
            _put(target, path, source[key])
    return target


def _unmapped(source: dict, mapping: dict[str, tuple[str, ...]]) -> dict:
    target = {}
    for key, path in mapping.items():
        current = source
        for part in path:
            if not isinstance(current, dict) or part not in current:
                break
            current = current[part]
        else:
            target[key] = deepcopy(current)
    return target


def _changes(base: dict, result: dict) -> dict:
    """Keep independent inference evidence without duplicating full metrics."""
    changes = {}
    removed = [key for key in base if key not in result]
    if removed:
        changes["remove"] = removed
    for key, value in result.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            patch = _changes(base[key], value)
            if patch:
                changes.setdefault("patch", {})[key] = patch
        elif key not in base or base[key] != value:
            changes.setdefault("set", {})[key] = deepcopy(value)
    return changes


def _apply_changes(base: dict, changes: dict) -> dict:
    result = deepcopy(base)
    for key in changes.get("remove", []):
        result.pop(key, None)
    result.update(deepcopy(changes.get("set", {})))
    for key, patch in changes.get("patch", {}).items():
        result[key] = _apply_changes(result.get(key, {}), patch)
    return result


def _encode_split(metrics: dict, *, test: bool = False) -> dict:
    result = {}
    head_names = metrics.get("heads", {})
    head_names = head_names if isinstance(head_names, dict) else {}
    for key, value in metrics.items():
        if key in _TRAINING_FIELDS or key.startswith("final_") or key == "validation_metrics":
            continue
        if key == "mean_acc":
            result["mean_accuracy"] = deepcopy(value)
        elif key == "macro_acc":
            result["macro_accuracy"] = deepcopy(value)
        elif key == "heads" and isinstance(value, dict):
            result["heads"] = {head: _encode_head(evidence, test=test) for head, evidence in value.items()}
        elif key.endswith("_acc") and key[:-4] in head_names:
            result.setdefault("primary_accuracy", {})[key[:-4]] = deepcopy(value)
        elif key.endswith("_macro_acc"):
            result.setdefault("primary_macro_accuracy", {})[key[:-10]] = deepcopy(value)
        elif key.endswith("_acc"):
            result.setdefault("primary_accuracy", {})[key[:-4]] = deepcopy(value)
        else:
            result[key] = deepcopy(value)
    # An explicit empty map preserves a missing requested metric: head evidence
    # alone must never fabricate the primary accuracy used by score guards.
    result.setdefault("primary_accuracy", {})
    result.setdefault("primary_macro_accuracy", {})
    return result


def _encode_head(evidence: Any, *, test: bool) -> Any:
    if not isinstance(evidence, dict):
        return deepcopy(evidence)
    result = {}
    if "acc" in evidence:
        result["accuracy"] = deepcopy(evidence["acc"])
    result.update({("macro_accuracy" if field == "macro_acc" else
                    "validation_loss" if field == "loss" and test else field): deepcopy(item)
                   for field, item in evidence.items() if field not in {"acc", "accuracy"}})
    if "accuracy" in evidence:
        result["accuracy_alias"] = True
        if "acc" not in evidence or evidence["accuracy"] != evidence["acc"]:
            result["reported_accuracy"] = deepcopy(evidence["accuracy"])
    return result


def _decode_head(evidence: Any) -> Any:
    if not isinstance(evidence, dict):
        return deepcopy(evidence)
    result = {("acc" if field == "accuracy" else "macro_acc" if field == "macro_accuracy" else
               "loss" if field == "validation_loss" else field): deepcopy(item)
              for field, item in evidence.items() if field not in {"accuracy_alias", "reported_accuracy"}}
    if evidence.get("accuracy_alias"):
        result["accuracy"] = deepcopy(evidence["reported_accuracy"] if "reported_accuracy" in evidence else evidence.get("accuracy"))
    return result


def _decode_split(section: dict) -> dict:
    result = {}
    for key, value in section.items():
        if key in {"primary_accuracy", "primary_macro_accuracy", "is_snapshot", "training_overrides", "inference_overrides", "inference_available"}:
            continue
        if key == "mean_accuracy":
            result["mean_acc"] = deepcopy(value)
        elif key == "macro_accuracy":
            result["macro_acc"] = deepcopy(value)
        elif key == "heads" and isinstance(value, dict):
            result["heads"] = {head: _decode_head(evidence) for head, evidence in value.items()}
        else:
            result[key] = deepcopy(value)
    scores = section.get("primary_accuracy")
    if scores is None:
        scores = {head: value["accuracy"] for head, value in section.get("heads", {}).items()
                  if isinstance(value, dict) and "accuracy" in value}
    for head, value in scores.items():
        result[f"{head}_acc"] = deepcopy(value)
    macro_scores = section.get("primary_macro_accuracy")
    if macro_scores is None:
        macro_scores = {head: value["macro_accuracy"] for head, value in section.get("heads", {}).items()
                        if isinstance(value, dict) and "macro_accuracy" in value}
    for head, value in macro_scores.items():
        result[f"{head}_macro_acc"] = deepcopy(value)
    return result


def _training_metrics(metrics: dict) -> dict:
    result = {key: deepcopy(value) for key, value in metrics.items() if key in _TRAINING_FIELDS}
    result.update({key: deepcopy(value) for key, value in metrics.items() if key.startswith("final_")})
    return result


def _inference_base(metrics: dict) -> dict:
    result = {key: deepcopy(value) for key, value in metrics.items()
              if key not in {*_TRAINING_FIELDS, "validation_metrics", "monitor", "run_dir"}
              and not key.startswith("final_")}
    if isinstance(result.get("heads"), dict):
        for evidence in result["heads"].values():
            if isinstance(evidence, dict):
                evidence.pop("loss", None)
    wrapper = {**result, "metrics": deepcopy(result)}
    if "test_csv" in result:
        wrapper["csv"] = result["test_csv"]
    return wrapper


def _nonfinite_losses(metrics: dict) -> bool:
    losses = [metrics.get(key) for key in ("train_loss", "val_loss", "best_val_loss")]
    losses.extend(metrics.get("train_loss_history", []))
    losses.extend(metrics.get("val_loss_history", []))
    heads = metrics.get("heads", {})
    if isinstance(heads, dict):
        losses.extend(value.get("loss") for value in heads.values() if isinstance(value, dict))
    return any(isinstance(value, float) and not math.isfinite(value) for value in losses)


def serialize_internal_experiment(experiment: dict[str, Any], *, session_id: int | None = None) -> dict[str, Any]:
    """Group an internal experiment without losing score evidence or inputs."""
    if isinstance(experiment.get("experiment"), dict):
        return deepcopy(experiment)
    config = deepcopy(experiment.get("config", {}))
    metrics = deepcopy(experiment.get("metrics", {}))
    split = experiment.get("decision_split", "val")
    snapshot = metrics.get("validation_metrics")
    snapshot = snapshot if isinstance(snapshot, dict) else None
    common = _training_metrics(metrics)
    # Older records dropped outer histories but retained the validation snapshot.
    for key, value in _training_metrics(snapshot or {}).items():
        common.setdefault(key, value)
    if _nonfinite_losses(metrics) or _nonfinite_losses(snapshot or {}):
        common["nonfinite_loss"] = True
    training = _mapped(common, _TRAINING_FIELDS)
    head_names = set(config.get("heads") or experiment.get("heads") or metrics.get("heads", {}))
    macro_final_keys = {key for key in common if key == "final_macro_acc"
                        or (key.startswith("final_") and key.endswith("_macro_acc")
                            and key[6:-4] not in head_names)}
    training["final_accuracy"] = {key[6:-4] if key.endswith("_acc") else key[6:]: value
                                  for key, value in common.items() if key.startswith("final_")
                                  and key != "final_epoch" and key not in macro_final_keys}
    if not training["final_accuracy"]:
        training.pop("final_accuracy")
    final_macro = {("mean" if key == "final_macro_acc" else key[6:-10]): value
                   for key, value in common.items() if key in macro_final_keys}
    if final_macro:
        training["final_macro_accuracy"] = final_macro
    validation_source = snapshot if snapshot is not None else (metrics if split == "val" else {})
    validation = _encode_split(validation_source)
    validation["is_snapshot"] = snapshot is not None
    if snapshot is not None:
        differences = _changes(common, _training_metrics(snapshot))
        if differences:
            validation["training_overrides"] = differences
    test = _encode_split(metrics, test=True) if split == "test" else {}
    for section, preferred in ((validation, ("mean_accuracy", "macro_accuracy", "heads")),
                               (test, ("mean_accuracy", "macro_accuracy", "total_images", "heads"))):
        reordered = {**{key: section[key] for key in preferred if key in section},
                     **{key: value for key, value in section.items() if key not in preferred}}
        section.clear()
        section.update(reordered)

    data = {key: deepcopy(experiment.get(key, config.get(key))) for key in _DATA_FIELDS
            if key in experiment or key in config}
    config_paths = {key: value for key, value in config.items()
                    if key in _DATA_FIELDS and (key == "test_csv" or data.get(key) != value)}
    if config_paths:
        data["config_paths"] = config_paths
    objective = {key: deepcopy(experiment[key]) for key in ("heads", "target_metric", "target_value") if key in experiment}
    objective.setdefault("heads", deepcopy(config.get("heads", [])))
    objective["decision_split"] = split
    if "monitor" in config:
        objective["monitor"] = config["monitor"]
    if "heads" in config and config["heads"] != objective["heads"]:
        objective["config_heads"] = deepcopy(config["heads"])
    grouped_config = _mapped(config, _CONFIG_FIELDS)
    grouped_config.update({key: value for key, value in config.items()
                           if key not in {*_CONFIG_FIELDS, *_DATA_FIELDS, "heads", "monitor", "out"}})
    grouped_config.setdefault("architecture", experiment.get("arch"))
    if "weights" not in grouped_config and "weights" in experiment:
        grouped_config["weights"] = deepcopy(experiment["weights"])
    preferred = ("architecture", "epochs", "batch_size", "num_workers", "img_size", "optimizer", "loss", "early_stopping")
    grouped_config = {**{key: grouped_config[key] for key in preferred if key in grouped_config},
                      **{key: value for key, value in grouped_config.items() if key not in preferred}}
    metadata = {"id": f"exp_{int(experiment.get('attempt', 0)):02d}",
                "attempt": experiment.get("attempt", 0), "status": experiment.get("training_status")}
    if "timestamp" in experiment:
        metadata.update(timestamp=experiment["timestamp"], date=str(experiment["timestamp"])[:10])
    number = session_id or experiment.get("session_id") or session_number(config.get("out", ""))
    if config.get("out") or experiment.get("session_dir") or number:
        metadata["session"] = {"id": number, "path": config.get("out") or experiment.get("session_dir")}
    extras = {key: deepcopy(value) for key, value in experiment.items() if key not in _FLAT_FIELDS}
    if extras:
        metadata["details"] = extras
    evaluation = deepcopy(experiment.get("evaluation", {}))
    collisions = {key: evaluation[key] for key in _BOOKKEEPING if key in evaluation and key in experiment
                  and evaluation[key] != experiment[key]}
    if collisions:
        evaluation["analysis_overrides"] = collisions
    evaluation.update({key: deepcopy(experiment[key]) for key in _BOOKKEEPING if key in experiment})
    evaluation["analysis_fields"] = list(experiment.get("evaluation", {}))
    artifacts = {key: deepcopy(experiment[key]) for key in ("run_dir", "best_weights", "last_weights") if key in experiment}
    if config.get("out"):
        artifacts["session_dir"] = config["out"]
    if split == "test" or "test_result" in experiment:
        result = experiment.get("test_result", {})
        test["inference_available"] = bool(result)
        if result:
            # The raw inference and merged decision metrics can disagree. Keep
            # their differences rather than silently replacing the raw proof.
            restored_metrics = {**common, **_decode_split(test)}
            if snapshot is not None:
                restored_metrics["validation_metrics"] = snapshot
            base_result = _inference_base(restored_metrics)
            test["inference_overrides"] = _changes(base_result, result)
            artifacts["test"] = {key: deepcopy(result[key]) for key in ("checkpoint", "csv", "output_dir") if key in result}
    return {"schema_version": 1, "experiment": metadata, "objective": objective, "config": grouped_config,
            "data": data, "training": training, "validation": validation, "test": test,
            "evaluation": evaluation, "artifacts": artifacts}


def serialize_experiment(experiment: dict[str, Any], *, session_id: int | None = None) -> dict[str, Any]:
    """Expose exactly the requested public fields, using recorded values only."""
    grouped = serialize_internal_experiment(experiment, session_id=session_id)
    metadata, objective, config, data, training, validation, test, evaluation, artifacts = (
        grouped.get(key, {}) for key in (
            "experiment", "objective", "config", "data", "training", "validation", "test", "evaluation", "artifacts"))

    def fields(source: dict, names: tuple[str, ...]) -> dict:
        return {key: deepcopy(source.get(key)) for key in names}

    def heads(source: dict, *, include_loss: bool) -> dict:
        names = ("accuracy", "loss") if include_loss else ("accuracy",)
        return {head: {**fields(evidence, names),
                       **({"macro_accuracy": deepcopy(evidence["macro_accuracy"])} if "macro_accuracy" in evidence else {})}
                for head, evidence in source.get("heads", {}).items()
                if isinstance(evidence, dict)}

    return {
        "experiment": fields(metadata, ("id", "date", "status")),
        "objective": fields(objective, ("heads", "target_metric", "target_value", "monitor")),
        "config": {
            **fields(config, ("architecture", "epochs", "batch_size", "num_workers", "img_size")),
            "optimizer": fields(config.get("optimizer", {}), ("lr", "weight_decay")),
            "loss": fields(config.get("loss", {}), ("gamma", "head_weights")),
            "early_stopping": fields(config.get("early_stopping", {}), ("patience",)),
        },
        "data": fields(data, ("train_csv", "val_csv", "test_csv")),
        "training": {
            **fields(training, ("epochs_completed", "best_epoch")),
            "loss": fields(training.get("loss", {}), ("train_final", "val_final", "best_val")),
            "history": {**fields(training.get("history", {}), ("train_loss", "val_loss", "mean_accuracy", "head_accuracy")),
                        **{key: deepcopy(value) for key, value in training.get("history", {}).items()
                           if key in {"macro_accuracy", "head_macro_accuracy"}}},
        },
        "validation": {"mean_accuracy": deepcopy(validation.get("mean_accuracy")),
                       **({"macro_accuracy": deepcopy(validation["macro_accuracy"])} if "macro_accuracy" in validation else {}),
                       "heads": heads(validation, include_loss=True)},
        "test": {"mean_accuracy": deepcopy(test.get("mean_accuracy")),
                 **({"macro_accuracy": deepcopy(test["macro_accuracy"])} if "macro_accuracy" in test else {}),
                 "total_images": deepcopy(test.get("total_images")), "heads": heads(test, include_loss=False)},
        "evaluation": fields(evaluation, ("verdict", "target_reached", "issues", "recommendations", "reasoning")),
        "artifacts": fields(artifacts, ("run_dir", "best_weights", "last_weights")),
    }


def normalize_internal_experiment(experiment: dict[str, Any]) -> dict[str, Any]:
    """Recover complete histories and keep nonfinite evidence before JSON cleanup."""
    return deserialize_experiment(serialize_internal_experiment(experiment))


def deserialize_experiment(document: dict[str, Any]) -> dict[str, Any]:
    """Normalize grouped JSON or historical flat JSON to the internal record."""
    if not isinstance(document, dict):
        raise ValueError("Experiment must be a JSON object")
    if not isinstance(document.get("experiment"), dict):
        return deepcopy(document)
    if "schema_version" not in document:
        return deserialize_public_experiment(document)
    for key in ("objective", "config", "data", "training", "validation", "test", "evaluation", "artifacts"):
        if not isinstance(document.get(key, {}), dict):
            raise ValueError(f"Experiment section '{key}' must be an object")
    metadata = document["experiment"]
    objective, data, artifacts = (document.get(key, {}) for key in ("objective", "data", "artifacts"))
    grouped_config = document.get("config", {})
    config = _unmapped(grouped_config, _CONFIG_FIELDS)
    config.update({key: deepcopy(value) for key, value in grouped_config.items()
                   if key not in {"architecture", "optimizer", "loss", "early_stopping"}})
    # Test CSV belongs to the agent's decision dataset, while the training
    # configuration normally contains only train/validation paths.
    config.update({key: deepcopy(value) for key, value in data.items() if key in _DATA_FIELDS and key != "test_csv"})
    config.update(deepcopy(data.get("config_paths", {})))
    config["heads"] = deepcopy(objective.get("config_heads", objective.get("heads", [])))
    if "monitor" in objective:
        config["monitor"] = objective["monitor"]
    session = metadata.get("session", {})
    if isinstance(session, dict) and session.get("path"):
        config["out"] = session["path"]
    elif artifacts.get("session_dir"):
        config["out"] = artifacts["session_dir"]
    split = objective.get("decision_split", "val")
    training = document.get("training", {})
    common = _unmapped(training, _TRAINING_FIELDS)
    for head, value in training.get("final_accuracy", {}).items():
        common[f"final_{head}_acc"] = deepcopy(value)
    for head, value in training.get("final_macro_accuracy", {}).items():
        common["final_macro_acc" if head == "mean" else f"final_{head}_macro_acc"] = deepcopy(value)
    validation, test = document.get("validation", {}), document.get("test", {})
    metrics = {**common, **_decode_split(test if split == "test" else validation)}
    if validation.get("is_snapshot"):
        metrics["validation_metrics"] = {**_apply_changes(common, validation.get("training_overrides", {})),
                                         **_decode_split(validation)}
    result = deepcopy(metadata.get("details", {}))
    attempt = metadata["attempt"] if "attempt" in metadata else int(str(metadata.get("id", "exp_0")).removeprefix("exp_"))
    result.update(attempt=attempt,
                  training_status=metadata.get("status"), config=config, metrics=metrics,
                  decision_split=split, heads=deepcopy(objective.get("heads", [])))
    if "timestamp" in metadata:
        result["timestamp"] = metadata["timestamp"]
    elif "date" in metadata:
        result["timestamp"] = metadata["date"]
    if isinstance(session, dict) and session.get("id"):
        result["session_id"] = session["id"]
    for key in ("target_metric", "target_value"):
        if key in objective:
            result[key] = deepcopy(objective[key])
    result.update({key: deepcopy(value) for key, value in data.items() if key in _DATA_FIELDS})
    result.update({key: deepcopy(artifacts[key]) for key in ("run_dir", "best_weights", "last_weights") if key in artifacts})
    for key in ("arch", "weights"):
        if key in config:
            result[key] = deepcopy(config[key])
    evaluation = document.get("evaluation", {})
    analysis_fields = evaluation.get("analysis_fields", [key for key in evaluation
        if key not in (set(_BOOKKEEPING) - {"target_reached"}) | {"analysis_fields", "analysis_overrides"}])
    result["evaluation"] = {key: deepcopy(evaluation[key]) for key in analysis_fields if key in evaluation}
    result["evaluation"].update(deepcopy(evaluation.get("analysis_overrides", {})))
    result.update({key: deepcopy(evaluation[key]) for key in _BOOKKEEPING if key in evaluation})
    if test.get("inference_available"):
        base = _inference_base(metrics)
        result["test_result"] = _apply_changes(base, test.get("inference_overrides", {}))
    elif "inference_available" in test:
        result["test_result"] = {}
    return result


def deserialize_public_experiment(document: dict[str, Any]) -> dict[str, Any]:
    """Public summaries are readable evidence, never proof of a test result."""
    # Reuse field mappings for diagnostics, while explicitly withholding every
    # status/count/proof flag needed for checkpoint selection or target checks.
    grouped = deepcopy(document)
    grouped["schema_version"] = 1
    objective = grouped.setdefault("objective", {})
    objective["decision_split"] = "test"
    grouped.setdefault("validation", {})["is_snapshot"] = True
    grouped.setdefault("test", {})["inference_available"] = False
    result = deserialize_experiment(grouped)
    result["metrics"]["status"] = "unverified_public"
    result["metrics"]["error"] = "Public experiment summary has no verified internal training/test record"
    result["test_result"] = {}
    result["public_summary"] = True
    return result
