"""Check bounded context preserves the evidence used to tune ResNet."""
from __future__ import annotations

import copy
import json
import unittest

from agent.llm_context import compact_experiment_history, compact_metrics, compact_test_result


def metrics_fixture(epochs: int = 30) -> dict:
    history = [0.01 * index for index in range(epochs)]
    return {
        "status": "ok", "decision_split": "val", "mean_acc": 0.85,
        "make_acc": 0.8, "vehicle_type_acc": 0.9, "train_loss": 0.2, "val_loss": 0.4,
        "heads": {"make": {"acc": 0.8, "loss": 0.3}, "vehicle_type": {"acc": 0.9, "loss": 0.1}},
        "train_loss_history": list(history), "val_loss_history": list(history),
        "mean_acc_history": list(history),
        "head_acc_history": {"make": list(history), "vehicle_type": list(history)},
    }


def experiment_fixture(attempt: int, score: float, epochs: int = 30) -> dict:
    metrics = metrics_fixture(epochs)
    metrics["mean_acc"] = score
    return {
        "attempt": attempt, "decision_split": "val", "test_csv": "",
        "target_metric": "mean_acc", "score": score, "training_status": "success",
        "metrics": metrics,
        "config": {"arch": "resnet50", "lr": 0.0001, "epochs": epochs,
                   "head_loss_weights": {"make": 1.0, "vehicle_type": 1.0},
                   "train_csv": "/tmp/train.csv", "weights": "/tmp/backbone.pth"},
        "evaluation": {"verdict": "overfitting", "issues": ["Validation loss increased"],
                       "recommendations": ["Increase weight decay"]},
    }


class CompactLLMContextTests(unittest.TestCase):
    def test_last_five_trends_and_exact_head_scores_are_preserved(self):
        metrics = metrics_fixture()
        original = copy.deepcopy(metrics)
        result = compact_metrics(metrics)
        for key in ("train_loss_history", "val_loss_history", "mean_acc_history"):
            self.assertEqual(result[key], metrics[key][-5:])
            self.assertEqual(result[f"{key}_length"], 30)
        self.assertEqual(result["head_acc_history"]["make"], metrics["head_acc_history"]["make"][-5:])
        self.assertEqual(result["head_acc_history_lengths"], {"make": 30, "vehicle_type": 30})
        self.assertEqual(result["heads"], metrics["heads"])
        self.assertEqual(result["make_acc"], 0.8)
        self.assertEqual(result["decision_split"], "val")
        result["train_loss_history"][0] = -1
        result["heads"]["make"]["acc"] = -1
        self.assertEqual(metrics, original)

    def test_validation_is_compacted_once_and_failures_remain_visible(self):
        metrics = metrics_fixture()
        validation = metrics_fixture(60)
        validation["validation_metrics"] = metrics_fixture(100)
        metrics.update(validation_metrics=validation, error="Inference image unreadable", nonfinite_loss=True)
        result = compact_metrics(metrics)
        self.assertEqual(result["validation_metrics"]["val_loss_history_length"], 60)
        self.assertEqual(len(result["validation_metrics"]["val_loss_history"]), 5)
        self.assertNotIn("validation_metrics", result["validation_metrics"])
        self.assertTrue(result["nonfinite_loss"])
        self.assertEqual(result["error"], "Inference image unreadable")

    def test_test_evidence_is_preserved_without_nested_duplicate(self):
        evidence = {
            "status": "ok", "split": "test", "decision_split": "test", "full_coverage": True,
            "test_evaluated": True, "test_csv": "/tmp/test.csv", "checkpoint": "/tmp/best_model.pth",
            "total_images": 100, "evaluated_images": 100, "matched_images": 100,
            "mean_acc": 0.8, "make_acc": 0.8,
            "heads": {"make": {"acc": 0.8, "correct": 80, "count": 100, "coverage": 1.0}},
        }
        raw = {**copy.deepcopy(evidence), "csv": "/tmp/test.csv", "metrics": copy.deepcopy(evidence)}
        original = copy.deepcopy(raw)
        result = compact_test_result(raw)
        self.assertNotIn("metrics", result)
        for key, value in evidence.items():
            self.assertEqual(result[key], value)
        result["heads"]["make"]["count"] = 0
        self.assertEqual(raw, original)
        nested_only = compact_test_result({"status": "ok", "metrics": evidence})
        self.assertEqual(nested_only["matched_images"], 100)
        self.assertEqual(nested_only["make_acc"], 0.8)

    def test_small_per_class_results_keep_support_zero_correct_and_all_classes(self):
        metrics = metrics_fixture()
        classes = [
            {"class": "Toyota", "correct": 1, "total": 3, "accuracy": 1 / 3},
            {"class": "Kia", "correct": 1, "total": 1, "accuracy": 1.0},
            {"class": "Ford", "correct": 0, "total": 1, "accuracy": 0.0},
            {"class": "Audi", "correct": 1, "total": 3, "accuracy": 1 / 3},
        ]
        metrics["heads"]["make"]["per_class_accuracy"] = classes
        original = copy.deepcopy(metrics)
        result = compact_metrics(metrics)
        summary = result["heads"]["make"]["per_class_accuracy"]
        self.assertEqual(summary["columns"], ["class", "correct", "total", "accuracy"])
        self.assertEqual(summary["rows"], [
            ["Ford", 0, 1, 0.0], ["Audi", 1, 3, 0.333333],
            ["Toyota", 1, 3, 0.333333], ["Kia", 1, 1, 1.0],
        ])
        self.assertEqual(summary["total_classes"], 4)
        self.assertEqual(summary["shown_classes"], 4)
        self.assertEqual(summary["omitted_classes"], 0)
        self.assertEqual(summary["selection"], "all")
        self.assertEqual(summary["macro_accuracy"], round((1 / 3 + 1 + 0 + 1 / 3) / 4, 6))
        self.assertEqual(result["mean_acc"], metrics["mean_acc"])
        self.assertEqual(result["heads"]["make"]["acc"], metrics["heads"]["make"]["acc"])
        self.assertEqual(metrics, original)
        summary["rows"][0][1] = -1
        self.assertEqual(metrics, original)

    def test_more_than_fifty_classes_keep_worst_results_and_full_macro_accuracy(self):
        classes = [{"class": f"Make{index:02d}", "correct": index, "total": 100,
                    "accuracy": index / 100} for index in range(55)]
        metrics = metrics_fixture()
        metrics["heads"]["make"]["per_class_accuracy"] = list(reversed(classes))
        metrics["heads"]["vehicle_type"]["per_class_accuracy"] = [
            {"class": "sedan", "correct": 8, "total": 10, "accuracy": 0.8},
        ]
        original = copy.deepcopy(metrics)
        summary = compact_metrics(metrics)["heads"]["make"]["per_class_accuracy"]
        self.assertEqual(summary["total_classes"], 55)
        self.assertEqual(summary["shown_classes"], 50)
        self.assertEqual(summary["omitted_classes"], 5)
        self.assertEqual(summary["selection"], "lowest_accuracy")
        self.assertEqual([row[0] for row in summary["rows"]], [f"Make{index:02d}" for index in range(50)])
        self.assertEqual(summary["macro_accuracy"], 0.27)
        self.assertNotEqual(summary["macro_accuracy"], sum(row[3] for row in summary["rows"]) / 50)
        other = compact_metrics(metrics)["heads"]["vehicle_type"]["per_class_accuracy"]
        self.assertEqual(other["shown_classes"], 1)
        self.assertEqual(other["selection"], "all")
        self.assertEqual(metrics, original)
        boundary = compact_metrics({"heads": {"make": {"per_class_accuracy": classes[:50]}}})
        boundary_summary = boundary["heads"]["make"]["per_class_accuracy"]
        self.assertEqual(boundary_summary["selection"], "all")
        self.assertEqual(boundary_summary["shown_classes"], 50)
        self.assertEqual(boundary_summary["omitted_classes"], 0)

    def test_per_class_sort_prioritizes_lower_accuracy_then_support_then_label(self):
        metrics = {"heads": {"make": {"per_class_accuracy": [
            {"class": "Volvo", "correct": 1, "total": 2, "accuracy": 0.5},
            {"class": "Toyota", "correct": 3, "total": 6, "accuracy": 0.5},
            {"class": "Ford", "correct": 3, "total": 6, "accuracy": 0.5},
            {"class": "Kia", "correct": 1, "total": 4, "accuracy": 0.25},
        ]}}}
        rows = compact_metrics(metrics)["heads"]["make"]["per_class_accuracy"]["rows"]
        self.assertEqual([row[0] for row in rows], ["Kia", "Ford", "Toyota", "Volvo"])
        self.assertEqual([row[2] for row in rows], [4, 6, 6, 2])

    def test_invalid_per_class_records_are_omitted_without_changing_scores(self):
        valid = {"class": "Toyota", "correct": 1, "total": 2, "accuracy": 0.5}
        invalid = [None, "invalid", {},
            {**valid, "class": ""}, {**valid, "class": "   "}, {**valid, "class": 7},
            {**valid, "total": 0}, {**valid, "total": -1}, {**valid, "total": True},
            {**valid, "total": 2.0}, {**valid, "correct": -1}, {**valid, "correct": 3},
            {**valid, "correct": True}, {**valid, "correct": 1.0},
            {**valid, "accuracy": float("nan")}, {**valid, "accuracy": float("inf")},
            {**valid, "accuracy": 0.7}, {**valid, "accuracy": -0.1}, {**valid, "accuracy": 1.1},
        ]
        metrics = metrics_fixture()
        metrics["heads"]["make"]["per_class_accuracy"] = [*invalid, valid]
        original_json = json.dumps(metrics)
        result = compact_metrics(metrics)
        summary = result["heads"]["make"]["per_class_accuracy"]
        self.assertEqual(summary["rows"], [["Toyota", 1, 2, 0.5]])
        self.assertEqual(summary["total_classes"], 1)
        self.assertEqual(summary["macro_accuracy"], 0.5)
        self.assertEqual(result["mean_acc"], 0.85)
        self.assertEqual(result["make_acc"], 0.8)
        self.assertEqual(result["heads"]["make"]["acc"], 0.8)
        self.assertEqual(json.dumps(metrics), original_json)
        for malformed in (None, {}, "not a list", [], invalid):
            with self.subTest(malformed=malformed):
                head = compact_metrics({"heads": {"make": {"acc": 0.8, "per_class_accuracy": malformed}}})["heads"]["make"]
                self.assertNotIn("per_class_accuracy", head)
                self.assertEqual(head["acc"], 0.8)

    def test_nested_test_result_compacts_per_class_evidence_and_keeps_raw_records(self):
        records = [{"class": "SUV", "correct": 1, "total": 3, "accuracy": 1 / 3}]
        raw = {"status": "ok", "csv": "/tmp/test.csv", "metrics": {
            "status": "ok", "decision_split": "test", "full_coverage": True,
            "mean_acc": 1 / 3, "total_images": 3,
            "heads": {"vehicle_type": {"acc": 1 / 3, "correct": 1, "count": 3,
                                          "per_class_accuracy": records}},
        }}
        original = copy.deepcopy(raw)
        result = compact_test_result(raw)
        self.assertNotIn("metrics", result)
        self.assertEqual(result["mean_acc"], 1 / 3)
        head = result["heads"]["vehicle_type"]
        self.assertEqual(head["acc"], 1 / 3)
        self.assertEqual(head["per_class_accuracy"]["rows"], [["SUV", 1, 3, 0.333333]])
        self.assertEqual(head["per_class_accuracy"]["macro_accuracy"], 0.333333)
        head["per_class_accuracy"]["rows"][0][0] = "changed"
        self.assertEqual(raw, original)

    def test_older_best_and_last_three_are_kept_with_tunable_settings_only(self):
        records = [experiment_fixture(index, 0.5) for index in range(1, 21)]
        records[2]["metrics"]["mean_acc"] = 0.95
        original = copy.deepcopy(records)
        result = compact_experiment_history(records)
        self.assertEqual([record["attempt"] for record in result], [3, 18, 19, 20])
        self.assertEqual(result[0]["score"], 0.95)
        self.assertEqual(result[0]["metric_value"], 0.95)
        self.assertEqual(result[0]["arch"], "resnet50")
        self.assertNotIn("train_csv", result[0]["config"])
        self.assertEqual(result[0]["config"]["weights"], records[2]["config"]["weights"])
        self.assertNotIn("metrics", result[0])
        result[0]["config"]["head_loss_weights"]["make"] = -1
        self.assertEqual(records, original)

    def test_best_record_requires_valid_score_and_exact_target_metric(self):
        records = [experiment_fixture(index, 0.5) for index in range(1, 7)]
        for record in records:
            record["target_metric"] = "make_acc"
        records[0]["metrics"].update(mean_acc=0.99, make_acc=0.7)
        records[1]["metrics"].update(mean_acc=0.8, make_acc=0.95)
        records[2]["metrics"]["make_acc"] = 0.99
        records[2]["training_status"] = "error"
        result = compact_experiment_history(records, "make_acc")
        self.assertEqual([record["attempt"] for record in result], [2, 4, 5, 6])
        self.assertEqual(result[0]["score"], 0.95)
        self.assertEqual(result[0]["target_metric"], "make_acc")
        self.assertEqual(result[0]["metric_value"], 0.95)
        self.assertEqual(compact_experiment_history([]), [])

    def test_repeated_thirty_epoch_history_shrinks_and_stays_bounded(self):
        metrics = metrics_fixture()
        metrics["validation_metrics"] = metrics_fixture()
        evidence = {"status": "ok", "full_coverage": True, "total_images": 100,
                    "checkpoint": "/tmp/best_model.pth"}
        raw = {"metrics": metrics, "validation_metrics": metrics["validation_metrics"],
               "test_result": {**evidence, "metrics": evidence},
               "experiment_history": [experiment_fixture(index, 0.5) for index in range(1, 21)]}
        original = copy.deepcopy(raw)
        compact = {"metrics": compact_metrics(raw["metrics"]),
                   "validation_metrics": compact_metrics(raw["validation_metrics"]),
                   "test_result": compact_test_result(raw["test_result"]),
                   "experiment_history": compact_experiment_history(raw["experiment_history"])}
        self.assertLess(len(json.dumps(compact)), len(json.dumps(raw)) / 5)
        self.assertLess(len(json.dumps(compact)), 7000)
        longer = [experiment_fixture(index, 0.5, 300) for index in range(1, 101)]
        self.assertEqual(len(compact_experiment_history(longer)), 4)
        self.assertLess(len(json.dumps(compact_experiment_history(longer))), 2500)
        self.assertEqual(raw, original)
        self.assertEqual(len(raw["metrics"]["train_loss_history"]), 30)


if __name__ == "__main__":
    unittest.main()
