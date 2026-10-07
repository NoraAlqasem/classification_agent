from __future__ import annotations

from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rich.console import Console

from agent.nodes.memory_node import memory_node
from memory.experiment_format import deserialize_experiment, serialize_experiment, serialize_internal_experiment
from memory.experiment_store import ExperimentStore, experiment_score


def make_experiment(root: Path) -> dict:
    session = root / "session_1"
    run = session / "exp_01"
    config = {
        "arch": "resnet18", "weights": str(root / "resnet18.pth"),
        "epochs": 20, "batch_size": 16, "num_workers": 2, "img_size": [64, 96],
        "lr": .001, "weight_decay": .0001, "gamma": 2., "head_loss_weights": {"make": 1., "color": .5},
        "patience": 5, "alpha_json": str(root / "alpha.json"), "nproc_per_node": 2,
        "train_csv": str(root / "train.csv"), "val_csv": str(root / "val.csv"),
        "data_root": str(root / "images"), "heads": ["make", "color"], "monitor": "make",
        "out": str(session), "name": "exp_01",
    }
    validation = {
        "status": "ok", "run_dir": str(run), "total_epochs": 2, "best_epoch": 2,
        "final_epoch": 2, "loss_epoch": 2, "monitor": "make", "best_score": .6,
        "mean_acc": .75, "make_acc": .6, "color_acc": .9,
        "heads": {"make": {"acc": .6, "loss": .4}, "color": {"acc": .9, "loss": .2}},
        "train_loss": .3, "val_loss": .6, "best_val_loss": .6,
        "train_loss_history": [.8, .3], "val_loss_history": [.9, .6],
        "mean_acc_history": [.65, .75], "head_acc_history": {"make": [.5, .6], "color": [.8, .9]},
        "final_mean_acc": .75, "final_make_acc": .6, "final_color_acc": .9,
        "nonfinite_loss": False, "nonfinite_accuracy": False,
    }
    inference = {
        "status": "ok", "decision_split": "test", "split": "test", "test_evaluated": True,
        "full_coverage": True, "total_images": 100, "evaluated_images": 100, "matched_images": 100,
        "checkpoint": str(run / "best_model.pth"), "test_csv": str(root / "test.csv"),
        "output_dir": str(run / "test_evaluation"), "img_size": [64, 96],
        "mean_acc": .8, "make_acc": .7, "color_acc": .9,
        "heads": {
            head: {"acc": accuracy, "accuracy": accuracy, "correct": correct, "count": 100,
                   "total": 100, "matched": 100, "evaluated": 100, "coverage": 1.,
                   "skipped_by_threshold": 0, "per_class_accuracy": [
                       {"class": name, "correct": correct // 2, "total": 50, "accuracy": accuracy}
                       for name in names]}
            for head, accuracy, correct, names in (("make", .7, 70, ["Ford", "Toyota"]),
                                                    ("color", .9, 90, ["blue", "red"]))
        },
    }
    metrics = {**deepcopy(validation), **deepcopy(inference), "validation_metrics": deepcopy(validation)}
    for head in metrics["heads"]:
        metrics["heads"][head]["loss"] = validation["heads"][head]["loss"]
    return {
        "attempt": 1, "timestamp": "2026-10-06T15:00:00.123456+00:00", "session_id": 1,
        "training_status": "success", "train_csv": config["train_csv"], "val_csv": config["val_csv"],
        "test_csv": inference["test_csv"], "data_root": config["data_root"], "heads": config["heads"],
        "arch": config["arch"], "weights": config["weights"], "config": config, "metrics": metrics,
        "decision_split": "test", "target_metric": "make_acc", "target_value": .85,
        "evaluation": {"verdict": "underfitting", "target_reached": False, "issues": ["Make target unmet"],
                       "recommendations": ["Increase epochs"], "reasoning": "Measured test make accuracy is .7"},
        "issues_detected": ["Make target unmet"], "optimisation_rationale": "Initial run.",
        "score": .7, "improved": True, "target_reached": False, "error": "",
        "run_dir": str(run), "best_weights": inference["checkpoint"], "last_weights": str(run / "last_model.pth"),
        "test_result": {**deepcopy(inference), "csv": inference["test_csv"], "metrics": deepcopy(inference)},
        "note": "Preserve an additional experiment annotation.",
    }


class ExperimentFormatTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="resnet-format-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.record = make_experiment(self.root)

    def test_document_has_requested_sections_and_readable_parameter_order(self):
        document = serialize_experiment(self.record)
        self.assertEqual(list(document), ["experiment", "objective", "config", "data",
                                         "training", "validation", "test", "evaluation", "artifacts"])
        expected = {
            "experiment": {"id", "date", "status"},
            "objective": {"heads", "target_metric", "target_value", "monitor"},
            "config": {"architecture", "epochs", "batch_size", "num_workers", "img_size", "optimizer", "loss", "early_stopping"},
            "data": {"train_csv", "val_csv", "test_csv"},
            "training": {"epochs_completed", "best_epoch", "loss", "history"},
            "validation": {"mean_accuracy", "heads"}, "test": {"mean_accuracy", "total_images", "heads"},
            "evaluation": {"verdict", "target_reached", "issues", "recommendations", "reasoning"},
            "artifacts": {"run_dir", "best_weights", "last_weights"},
        }
        for section, fields in expected.items():
            self.assertEqual(set(document[section]), fields, section)
        self.assertEqual(document["experiment"]["id"], "exp_01")
        self.assertEqual(document["experiment"]["date"], "2026-10-06")
        self.assertEqual(list(document["config"])[:8], ["architecture", "epochs", "batch_size", "num_workers",
                                                       "img_size", "optimizer", "loss", "early_stopping"])
        self.assertEqual(list(document["data"]), ["train_csv", "val_csv", "test_csv"])
        self.assertEqual(document["config"]["optimizer"], {"lr": .001, "weight_decay": .0001})
        self.assertEqual(document["config"]["loss"], {"gamma": 2., "head_weights": {"make": 1., "color": .5}})
        self.assertEqual(document["config"]["early_stopping"], {"patience": 5})
        self.assertNotIn("metrics", document)
        self.assertNotIn("test_result", document)
        self.assertEqual(document["training"]["loss"].keys(), {"train_final", "val_final", "best_val"})
        self.assertEqual(document["training"]["history"].keys(), {"train_loss", "val_loss", "mean_accuracy", "head_accuracy"})
        for evidence in document["test"]["heads"].values():
            self.assertEqual(set(evidence), {"accuracy"})
        for evidence in document["validation"]["heads"].values():
            self.assertEqual(set(evidence), {"accuracy", "loss"})
        self.assertEqual(document["test"]["heads"]["make"]["accuracy"], .7)
        self.assertEqual(document["validation"]["heads"]["make"]["loss"], .4)

    def test_config_evidence_metadata_and_evaluation_roundtrip_without_mutation(self):
        original = deepcopy(self.record)
        document = serialize_internal_experiment(self.record)
        restored = deserialize_experiment(json.loads(json.dumps(document, allow_nan=False)))
        self.assertEqual(restored, original)
        self.assertEqual(self.record, original)
        self.assertEqual(restored["test_result"], original["test_result"])
        self.assertEqual(document["test"]["inference_overrides"], {})
        self.assertEqual(experiment_score(document), .7)
        self.assertEqual(experiment_score(restored), .7)
        public = serialize_experiment(self.record)
        self.assertIsNone(experiment_score(public))
        diagnostics = deserialize_experiment(public)
        self.assertEqual(diagnostics["metrics"]["status"], "unverified_public")
        self.assertEqual(diagnostics["metrics"]["validation_metrics"]["mean_acc"], .75)
        self.assertEqual(diagnostics["metrics"]["validation_metrics"]["make_acc"], .6)
        self.assertEqual(diagnostics["test_result"], {})
        self.assertNotIn("evaluated_images", diagnostics["metrics"])
        self.assertNotIn("count", diagnostics["metrics"]["heads"]["make"])

    def test_training_section_preserves_each_epoch_and_validation_checkpoint(self):
        document = serialize_experiment(self.record)
        training = document["training"]
        self.assertEqual(training["epochs_completed"], 2)
        self.assertEqual(training["best_epoch"], 2)
        self.assertEqual(training["loss"], {"train_final": .3, "val_final": .6, "best_val": .6})
        self.assertEqual(training["history"], {
            "train_loss": [.8, .3], "val_loss": [.9, .6], "mean_accuracy": [.65, .75],
            "head_accuracy": {"make": [.5, .6], "color": [.8, .9]},
        })
        self.assertEqual(document["validation"]["mean_accuracy"], .75)
        self.assertEqual(document["test"]["mean_accuracy"], .8)
        self.assertEqual(document["test"]["total_images"], 100)
        self.assertEqual(document["test"]["heads"]["make"], {"accuracy": .7})

    def test_legacy_flat_records_are_accepted_and_copied(self):
        legacy = deepcopy(self.record)
        legacy.pop("session_id")
        legacy.pop("arch")
        legacy.pop("weights")
        normalized = deserialize_experiment(legacy)
        self.assertEqual(normalized, legacy)
        self.assertIsNot(normalized, legacy)
        document = serialize_internal_experiment(legacy)
        self.assertEqual(document["config"]["architecture"], "resnet18")
        self.assertEqual(document["experiment"]["session"]["id"], 1)
        self.assertEqual(experiment_score(document), experiment_score(legacy))

    def test_memory_keeps_histories_and_saves_nested_document_for_store_loading(self):
        state = {
            "attempt_number": 1, "current_config": self.record["config"], "metrics": self.record["metrics"],
            "training_result": {"status": "success", "run_dir": self.record["run_dir"],
                                "best_weights": self.record["best_weights"], "last_weights": self.record["last_weights"]},
            "test_csv": self.record["test_csv"], "test_result": self.record["test_result"],
            "decision_split": "test", "target_metric": "make_acc", "target_value": .85,
            "evaluation_result": self.record["evaluation"], "issues_detected": self.record["issues_detected"],
        }
        with patch("agent.nodes.memory_node.console", Console(file=io.StringIO())):
            result = memory_node(state)
        saved = result["experiment_history"][0]
        self.assertEqual(saved["metrics"]["train_loss_history"], [.8, .3])
        self.assertEqual(saved["metrics"]["head_acc_history"], self.record["metrics"]["head_acc_history"])
        document = json.loads((self.root / "session_1" / "exp_01" / "exp_1.json").read_text(encoding="utf-8"))
        self.assertEqual(document["training"]["history"]["train_loss"], [.8, .3])
        loaded = ExperimentStore(self.root).load_all_experiments()[0]
        self.assertEqual(loaded["metrics"], saved["metrics"])
        self.assertEqual(loaded["config"], saved["config"])
        self.assertEqual(experiment_score(loaded), .7)
        knowledge = json.loads((self.root / "knowledge_base.json").read_text(encoding="utf-8"))
        self.assertEqual(knowledge["experiments"][0]["record"]["test_result"], saved["test_result"])
        self.assertEqual(knowledge["experiments"][0]["record"]["metrics"], saved["metrics"])
        self.assertEqual(len(knowledge["experiments"][0]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
