"""Read grouped attempt artifacts without losing historical flat compatibility."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent.nodes import monitor
from tools import export_tools, metrics_tools, training_tools


class ArtifactReadersTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="resnet-artifact-readers-")
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name) / "exp_01"
        self.run.mkdir()

    def metric_log(self, directory, *, best=0.9, final=0.85):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "training_metrics.csv"
        path.write_text(
            "Epoch,Split,TotalLoss,MeanAcc,make_Loss,make_Acc\n"
            "1,train,1.4,0.6,1.4,0.6\n"
            f"1,val,0.7,{best},0.7,{best}\n"
            "2,train,0.6,0.9,0.6,0.9\n"
            f"2,val,0.9,{final},0.9,{final}\n", encoding="utf-8")
        return path

    def test_monitor_and_metric_tools_prefer_grouped_log_to_legacy_flat_log(self):
        self.metric_log(self.run, best=0.3, final=0.2)
        self.metric_log(self.run / "train_evaluation")
        with patch.object(monitor.console, "rule"), patch.object(monitor.console, "print"):
            result = monitor.monitor_node({"training_result": {"status": "success", "run_dir": str(self.run)},
                                           "current_config": {"monitor": "mean"}, "attempt_number": 1})
        self.assertEqual(result["metrics"]["status"], "ok")
        self.assertEqual(result["metrics"]["best_epoch"], 1)
        self.assertEqual(result["metrics"]["mean_acc"], 0.9)
        arguments = {"run_dir": str(self.run)}
        selected = json.loads(metrics_tools.read_results_csv.invoke(arguments))
        final = json.loads(metrics_tools.get_final_metrics.invoke(arguments))
        history = json.loads(metrics_tools.read_loss_history.invoke(arguments))
        trend = json.loads(metrics_tools.compute_overfitting_score.invoke(arguments))
        self.assertEqual(selected["mean_acc"], 0.9)
        self.assertEqual(selected["best_epoch"], 1)
        self.assertEqual(final["mean_acc"], 0.85)
        self.assertEqual(final["epoch"], 2)
        self.assertEqual(history["val_loss"], [0.7, 0.9])
        self.assertEqual(history["head_acc"]["make"], [0.9, 0.85])
        self.assertEqual(trend["score"], 0)

    def test_class_mapping_uses_weights_folder_and_retains_flat_fallback(self):
        self.metric_log(self.run)
        classes = {"make": ["A", "B"]}
        flat = self.run / "classes_by_head.json"
        flat.write_text(json.dumps(classes), encoding="utf-8")
        arguments = {"run_dir": str(self.run)}
        self.assertEqual(json.loads(metrics_tools.read_per_class_metrics.invoke(arguments))["classes_by_head"], classes)
        grouped = self.run / "weights"
        grouped.mkdir()
        new_classes = {"make": ["Toyota", "Ford"]}
        (grouped / "classes_by_head.json").write_text(json.dumps(new_classes), encoding="utf-8")
        result = json.loads(metrics_tools.read_per_class_metrics.invoke(arguments))
        self.assertEqual(result["classes_by_head"], new_classes)
        self.assertEqual(result["heads"]["make"]["acc"], 0.9)

    def test_legacy_training_csv_still_drives_monitor_and_tools(self):
        self.metric_log(self.run, best=0.7, final=0.6)
        with patch.object(monitor.console, "rule"), patch.object(monitor.console, "print"):
            result = monitor.monitor_node({"training_result": {"status": "success", "run_dir": str(self.run)}})
        self.assertEqual(result["metrics"]["mean_acc"], 0.7)
        self.assertEqual(json.loads(metrics_tools.read_results_csv.invoke({"run_dir": str(self.run)}))["mean_acc"], 0.7)

    def test_training_log_prefers_grouped_location_then_legacy_flat_log(self):
        flat = self.run / "training.log"
        flat.write_text("Old training log.\n", encoding="utf-8")
        folder = self.run / "train_evaluation"
        folder.mkdir()
        grouped = folder / "training.log"
        grouped.write_text("Epoch one.\nEpoch two.\nTraining finished.\n", encoding="utf-8")
        arguments = {"run_dir": str(self.run), "last_n_lines": 2}
        result = json.loads(training_tools.read_training_log.invoke(arguments))
        self.assertEqual(result["log_file"], str(grouped))
        self.assertEqual(result["lines"], ["Epoch two.", "Training finished."])
        grouped.unlink()
        result = json.loads(training_tools.read_training_log.invoke(arguments))
        self.assertEqual(result["log_file"], str(flat))
        self.assertEqual(result["lines"], ["Old training log."])

    def test_recursive_artifact_lists_include_relative_grouped_paths(self):
        files = ["weights/best_model.pth", "weights/best_model.onnx", "weights/best_model.metadata.json",
                 "train_evaluation/training_metrics.csv", "train_evaluation/training.log",
                 "test_evaluation/make/per_class_accuracy.csv"]
        for name in files:
            path = self.run / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture", encoding="utf-8")
        artifacts = json.loads(metrics_tools.list_run_artefacts.invoke({"run_dir": str(self.run)}))
        self.assertEqual(artifacts["files"], sorted(files))
        exports = json.loads(export_tools.list_export_artefacts.invoke({"run_dir": str(self.run)}))["artefacts"]
        self.assertEqual({row["relative_path"] for row in exports},
                         {"weights/best_model.onnx", "weights/best_model.metadata.json"})
        self.assertTrue(all(Path(row["path"]).is_file() for row in exports))

    def test_test_per_class_metrics_continue_using_unchanged_test_evaluation_folder(self):
        test = self.run / "test_evaluation"
        (test / "make").mkdir(parents=True)
        evidence = {"heads": {"make": {"acc": 0.5}}, "full_coverage": True}
        (test / "summary.json").write_text(json.dumps(evidence), encoding="utf-8")
        (test / "make" / "per_class_accuracy.csv").write_text(
            "true_label,total,correct,accuracy\nToyota,2,1,0.5\n", encoding="utf-8-sig")
        result = json.loads(metrics_tools.read_per_class_metrics.invoke({"run_dir": str(self.run)}))
        self.assertEqual(result["split"], "test")
        self.assertTrue(result["full_coverage"])
        self.assertEqual(result["per_class_accuracy"]["make"],
                         [{"true_label": "Toyota", "total": "2", "correct": "1", "accuracy": "0.5"}])


if __name__ == "__main__":
    unittest.main()
