"""Keep independent score evidence intact when experiments become grouped JSON."""
from __future__ import annotations

from copy import deepcopy
import json
import unittest

from memory.experiment_format import deserialize_experiment, serialize_internal_experiment
from memory.experiment_store import _clean_json, experiment_score, test_evaluation_error


def saved_test_experiment():
    csv = "/dataset/test.csv"
    heads = {
        "make": {"head": "make", "matched": 10, "evaluated": 10, "total": 10,
                 "count": 10, "coverage": 1.0, "skipped_by_threshold": 0, "correct": 6,
                 "accuracy": 0.6, "acc": 0.6, "per_class_accuracy": [
                     {"class": "A", "correct": 4, "total": 6, "accuracy": 4 / 6},
                     {"class": "B", "correct": 2, "total": 4, "accuracy": 0.5}]},
        "vehicle_type": {"head": "vehicle_type", "matched": 10, "evaluated": 10, "total": 10,
                         "count": 10, "coverage": 1.0, "skipped_by_threshold": 0, "correct": 8,
                         "accuracy": 0.8, "acc": 0.8, "per_class_accuracy": [
                             {"class": "X", "correct": 5, "total": 5, "accuracy": 1.0},
                             {"class": "Y", "correct": 3, "total": 5, "accuracy": 0.6}]},
    }
    inference = {
        "status": "ok", "mean_acc": 0.7, "make_acc": 0.6, "vehicle_type_acc": 0.8,
        "heads": heads, "total_images": 10, "matched_images": 10, "evaluated_images": 10,
        "full_coverage": True, "threshold": 0.0, "split": "test", "decision_split": "test",
        "test_evaluated": True, "test_csv": csv, "checkpoint": "/runs/session_1/exp_1/best_model.pth",
        "output_dir": "/runs/session_1/exp_1/test_evaluation", "arch": "resnet18", "img_size": [64, 64],
    }
    validation = {
        "status": "ok", "mean_acc": 0.75, "make_acc": 0.65, "vehicle_type_acc": 0.85,
        "heads": {"make": {"acc": 0.65, "loss": 0.9}, "vehicle_type": {"acc": 0.85, "loss": 0.2}},
        "total_epochs": 2, "final_epoch": 2, "best_epoch": 2, "best_score": 0.75,
        "train_loss": 0.8, "val_loss": 1.1, "best_val_loss": 1.1, "loss_epoch": 2,
        "nonfinite_loss": False, "nonfinite_accuracy": False,
        "train_loss_history": [1.4, 0.8], "val_loss_history": [1.5, 1.1],
        "mean_acc_history": [0.55, 0.75],
        "head_acc_history": {"make": [0.4, 0.65], "vehicle_type": [0.7, 0.85]},
    }
    metrics = {**deepcopy(validation), **deepcopy(inference), "validation_metrics": deepcopy(validation)}
    for head in metrics["heads"]:
        metrics["heads"][head]["loss"] = validation["heads"][head]["loss"]
    # This is the existing saved-record shape: outer histories were omitted,
    # while its validation snapshot still holds the full measured histories.
    for key in ("train_loss_history", "val_loss_history", "mean_acc_history", "head_acc_history"):
        metrics.pop(key)
    return {
        "attempt": 1, "timestamp": "2026-10-06T15:51:50+00:00", "training_status": "success",
        "train_csv": "/dataset/train.csv", "val_csv": "/dataset/val.csv", "test_csv": csv,
        "decision_split": "test", "heads": ["make", "vehicle_type"], "target_metric": "mean_acc",
        "target_value": 0.99, "score": 0.7, "improved": True,
        "config": {"arch": "resnet18", "weights": "/weights/resnet18.pth", "epochs": 2,
                   "batch_size": 8, "img_size": [64, 64], "lr": 0.0001, "weight_decay": 0.0001,
                   "gamma": 1.0, "patience": 10, "monitor": "mean", "heads": ["make", "vehicle_type"],
                   "train_csv": "/dataset/train.csv", "val_csv": "/dataset/val.csv",
                   "out": "/runs/session_1", "name": "exp_1"},
        "metrics": metrics, "test_result": {**deepcopy(inference), "metrics": deepcopy(inference), "csv": csv},
        "evaluation": {"verdict": "underfitting", "issues": ["Too few epochs"]},
        "run_dir": "/runs/session_1/exp_1", "best_weights": "/runs/session_1/exp_1/best_model.pth",
        "last_weights": "/runs/session_1/exp_1/last_model.pth",
    }


def disk_roundtrip(record):
    document = _clean_json(serialize_internal_experiment(record, session_id=1))
    return deserialize_experiment(json.loads(json.dumps(document, allow_nan=False)))


class ExperimentFormatEvidenceRegressions(unittest.TestCase):
    def test_legacy_saved_shape_recovers_histories_and_preserves_all_head_and_test_evidence(self):
        old = saved_test_experiment()
        original = deepcopy(old)
        restored = disk_roundtrip(old)
        self.assertEqual(experiment_score(old), 0.7)
        self.assertEqual(experiment_score(restored), 0.7)
        self.assertIsNone(test_evaluation_error(restored["metrics"], restored["test_result"], restored["test_csv"]))
        self.assertEqual(restored["test_result"], old["test_result"])
        self.assertEqual(restored["metrics"]["heads"], old["metrics"]["heads"])
        self.assertEqual(restored["metrics"]["validation_metrics"], old["metrics"]["validation_metrics"])
        for key in ("train_loss_history", "val_loss_history", "mean_acc_history", "head_acc_history"):
            self.assertEqual(restored["metrics"][key], old["metrics"]["validation_metrics"][key])
        self.assertEqual(old, original)

    def test_conflicting_accuracy_alias_never_repairs_invalid_decision_evidence(self):
        old = saved_test_experiment()
        old["metrics"]["heads"]["make"]["acc"] = 0.1
        self.assertIsNone(experiment_score(old))
        restored = disk_roundtrip(old)
        self.assertIsNone(experiment_score(restored))
        self.assertEqual(restored["metrics"]["heads"]["make"], old["metrics"]["heads"]["make"])

    def test_secondary_accuracy_alias_does_not_replace_valid_primary_acc(self):
        old = saved_test_experiment()
        old["metrics"]["heads"]["make"]["accuracy"] = 0.1
        self.assertEqual(experiment_score(old), 0.7)
        restored = disk_roundtrip(old)
        self.assertEqual(experiment_score(restored), 0.7)
        self.assertEqual(restored["metrics"]["heads"]["make"], old["metrics"]["heads"]["make"])

    def test_missing_primary_acc_is_not_fabricated_from_secondary_accuracy(self):
        old = saved_test_experiment()
        old["metrics"]["heads"]["make"].pop("acc")
        self.assertIsNone(experiment_score(old))
        restored = disk_roundtrip(old)
        self.assertIsNone(experiment_score(restored))
        self.assertNotIn("acc", restored["metrics"]["heads"]["make"])

    def test_independent_inference_mismatches_and_missing_evidence_remain_unscorable(self):
        for fault in ("count", "correct", "primary", "missing_acc", "missing_heads", "missing_result", "wrong_csv"):
            with self.subTest(fault=fault):
                old = saved_test_experiment()
                raw = old["test_result"]["metrics"]
                if fault == "count":
                    raw["heads"]["make"]["count"] = 9
                elif fault == "correct":
                    raw["heads"]["make"]["correct"] = 1
                elif fault == "primary":
                    raw["make_acc"] = 0.1
                elif fault == "missing_acc":
                    raw["heads"]["make"].pop("acc")
                elif fault == "missing_heads":
                    raw.pop("heads")
                elif fault == "missing_result":
                    old.pop("test_result")
                elif fault == "wrong_csv":
                    raw["test_csv"] = "/dataset/different_test.csv"
                self.assertIsNone(experiment_score(old))
                restored = disk_roundtrip(old)
                self.assertIsNone(experiment_score(restored))
                if fault != "missing_result":
                    self.assertEqual(restored["test_result"], old["test_result"])

    def test_nonfinite_loss_rejection_survives_strict_json_null_conversion(self):
        for fault in ("train_loss", "val_loss", "best_val_loss", "head_loss", "history"):
            with self.subTest(fault=fault):
                old = saved_test_experiment()
                if fault == "head_loss":
                    old["metrics"]["heads"]["make"]["loss"] = float("inf")
                elif fault == "history":
                    old["metrics"]["train_loss_history"] = [1.0, float("nan")]
                else:
                    old["metrics"][fault] = float("inf")
                self.assertIsNone(experiment_score(old))
                restored = disk_roundtrip(old)
                self.assertIsNone(experiment_score(restored))
                self.assertTrue(restored["metrics"]["nonfinite_loss"])

    def test_explicit_attempt_is_used_even_when_grouped_id_contains_session_name(self):
        document = serialize_internal_experiment(saved_test_experiment(), session_id=1)
        document["experiment"]["id"] = "exp_1_session_1"
        self.assertEqual(deserialize_experiment(document)["attempt"], 1)


if __name__ == "__main__":
    unittest.main()
