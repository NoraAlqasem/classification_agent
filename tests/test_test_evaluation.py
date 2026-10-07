from __future__ import annotations

import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import pandas as pd
    import torch
    from PIL import Image
    from training import test_resnet as evaluation
    from agent.nodes import tester
except ImportError as exc:
    IMPORT_ERROR = str(exc)
else:
    IMPORT_ERROR = ""


@unittest.skipIf(bool(IMPORT_ERROR), f"Test evaluation dependencies unavailable: {IMPORT_ERROR}")
class CheckpointEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory(prefix="resnet-real-test-evaluation-")
        cls.directory = Path(cls.temporary.name)
        cls.images = cls.directory / "images"
        cls.images.mkdir()
        for index in range(3):
            Image.new("RGB", (45, 37), (30 + index, 100, 170)).save(cls.images / f"car{index}.png")
        cls.csv = cls.directory / "test.csv"
        pd.DataFrame({"root_folder": ["images"] * 3,
                      "image_name": [f"car{index}.png" for index in range(3)],
                      "make": ["A", "B", "B"], "vehicle_type": ["X", "Y", "X"]}).to_csv(cls.csv, index=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4)
            model = evaluation.MultiHeadResNet("resnet18", {"make": 2, "vehicle_type": 2})
            with torch.no_grad():
                for layer in model.heads.values():
                    layer.weight.zero_()
                    layer.bias.copy_(torch.tensor([2.0, 0.0]))
        cls.checkpoint_data = {
            "arch": "resnet18", "heads": ["make", "vehicle_type"],
            "classes_by_head": {"make": ["A", "B"], "vehicle_type": ["X", "Y"]},
            "class_to_idx_by_head": {"make": {"A": 0, "B": 1}, "vehicle_type": {"X": 0, "Y": 1}},
            "num_classes_by_head": {"make": 2, "vehicle_type": 2},
            "model_state": model.state_dict(), "img_size": [32, 40],
        }
        cls.checkpoint = cls.directory / "best_model.pth"
        torch.save(cls.checkpoint_data, cls.checkpoint)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.output = tempfile.TemporaryDirectory(prefix="resnet-test-artifacts-")
        self.addCleanup(self.output.cleanup)

    def evaluate(self, **kwargs):
        parameters = {"data_root": str(self.directory), "device": "cpu", "batch_size": 2}
        parameters.update(kwargs)
        return evaluation.evaluate_checkpoint(
            str(self.checkpoint), str(self.csv), self.output.name, **parameters)

    def altered_checkpoint(self, key, value):
        checkpoint = dict(self.checkpoint_data)
        checkpoint[key] = value
        path = Path(self.output.name) / "altered_model.pth"
        torch.save(checkpoint, path)
        return path

    def test_real_cpu_inference_counts_every_row_and_saves_all_artifacts(self):
        metrics = self.evaluate()
        self.assertEqual(metrics["status"], "ok")
        self.assertTrue(metrics["full_coverage"])
        self.assertEqual([metrics[key] for key in ("total_images", "matched_images", "evaluated_images")], [3, 3, 3])
        self.assertEqual(metrics["make_acc"], 1 / 3)
        self.assertEqual(metrics["vehicle_type_acc"], 2 / 3)
        self.assertEqual(metrics["mean_acc"], 0.5)
        self.assertEqual(metrics["heads"]["make"]["correct"], 1)
        self.assertEqual(metrics["heads"]["vehicle_type"]["correct"], 2)
        expected_classes = {
            "make": [{"class": "A", "correct": 1, "total": 1, "accuracy": 1.0},
                     {"class": "B", "correct": 0, "total": 2, "accuracy": 0.0}],
            "vehicle_type": [{"class": "X", "correct": 2, "total": 2, "accuracy": 1.0},
                             {"class": "Y", "correct": 0, "total": 1, "accuracy": 0.0}],
        }
        for head in ("make", "vehicle_type"):
            self.assertEqual(metrics["heads"][head]["count"], 3)
            self.assertEqual(metrics["heads"][head]["coverage"], 1.0)
            self.assertEqual(metrics["heads"][head]["skipped_by_threshold"], 0)
            for filename in ("per_class_accuracy.csv", "confusion_matrix_counts.csv",
                             "confusion_matrix_counts.png", "confusion_matrix_normalized.png"):
                self.assertTrue((Path(self.output.name) / head / filename).is_file(), filename)
            cm = pd.read_csv(Path(self.output.name) / head / "confusion_matrix_counts.csv", index_col=0)
            self.assertEqual(int(cm.to_numpy().sum()), 3)
            classes = metrics["heads"][head]["per_class_accuracy"]
            self.assertEqual(classes, expected_classes[head])
            saved_classes = pd.read_csv(Path(self.output.name) / head / "per_class_accuracy.csv")
            saved_records = saved_classes.rename(columns={"true_label": "class"}).to_dict("records")
            self.assertEqual(classes, saved_records)
            self.assertEqual(sum(row["total"] for row in classes), metrics["heads"][head]["count"])
            self.assertEqual(sum(row["correct"] for row in classes), metrics["heads"][head]["correct"])
        predictions = pd.read_csv(Path(self.output.name) / "inference_results.csv")
        self.assertEqual(len(predictions), 3)
        summary = json.loads((Path(self.output.name) / "summary.json").read_text())
        self.assertEqual(summary, metrics)
        self.assertEqual(summary["checkpoint"], str(self.checkpoint.resolve()))
        self.assertEqual(summary["test_csv"], str(self.csv.resolve()))
        saved_summary = pd.read_csv(Path(self.output.name) / "summary.csv")
        self.assertEqual(list(saved_summary.columns),
                         ["head", "matched", "evaluated", "total", "count", "coverage",
                          "skipped_by_threshold", "correct", "accuracy", "acc"])

    def test_classes_without_ground_truth_are_not_reported_as_zero_accuracy(self):
        subset = Path(self.output.name) / "subset.csv"
        pd.read_csv(self.csv).iloc[:1].to_csv(subset, index=False)
        with patch.object(evaluation, "save_confusion_matrix_png"):
            metrics = evaluation.evaluate_checkpoint(
                str(self.checkpoint), str(subset), self.output.name,
                data_root=str(self.directory), device="cpu")
        self.assertEqual(metrics["heads"]["make"]["per_class_accuracy"],
                         [{"class": "A", "correct": 1, "total": 1, "accuracy": 1.0}])
        self.assertEqual(metrics["heads"]["vehicle_type"]["per_class_accuracy"],
                         [{"class": "X", "correct": 1, "total": 1, "accuracy": 1.0}])
        self.assertTrue(metrics["full_coverage"])
        self.assertEqual(metrics["mean_acc"], 1.0)

    def test_standalone_threshold_class_counts_include_only_evaluated_rows(self):
        prediction_csv = Path(self.output.name) / "threshold_predictions.csv"
        pd.DataFrame({"image_name": ["car0.png", "car1.png", "car2.png"],
                      "make_pred": ["A", "A", "B"], "make_conf": [0.9, 0.1, 0.9],
                      "vehicle_type_pred": ["X", "Y", "Y"],
                      "vehicle_type_conf": [0.9, 0.9, 0.1]}).to_csv(prediction_csv, index=False)
        for threshold in (0.5, 1.0):
            with self.subTest(threshold=threshold):
                output = Path(self.output.name) / str(threshold)
                args = argparse.Namespace(ckpt=None, pred_csv=str(prediction_csv),
                                          heads=["make", "vehicle_type"], gt_csv=str(self.csv),
                                          threshold=threshold, out_dir=str(output))
                with patch.object(evaluation, "parse_args", return_value=args), \
                     patch.object(evaluation, "save_confusion_matrix_png"):
                    evaluation.main()
                metrics = json.loads((output / "summary.json").read_text())
                self.assertEqual(metrics["status"], "partial_coverage")
                self.assertFalse(metrics["full_coverage"])
                self.assertEqual(metrics["total_images"], 3)
                for head, labels in (("make", ["A", "B"]), ("vehicle_type", ["X", "Y"])):
                    classes = metrics["heads"][head]["per_class_accuracy"]
                    if threshold == 0.5:
                        self.assertEqual(classes, [
                            {"class": label, "correct": 1, "total": 1, "accuracy": 1.0}
                            for label in labels])
                    else:
                        self.assertEqual(classes, [])
                    self.assertEqual(sum(row["total"] for row in classes), metrics["heads"][head]["count"])
                    self.assertEqual(sum(row["correct"] for row in classes), metrics["heads"][head]["correct"])
                    saved = pd.read_csv(output / head / "per_class_accuracy.csv")
                    self.assertEqual(classes, saved.rename(columns={"true_label": "class"}).to_dict("records"))
                self.assertNotIn("per_class_accuracy", pd.read_csv(output / "summary.csv").columns)

    def test_checkpoint_training_size_wins_over_later_config_size(self):
        received_shapes = []
        original = evaluation.MultiHeadResNet.forward

        def forward(model, images):
            received_shapes.append(list(images.shape[2:]))
            return original(model, images)

        with patch.object(evaluation.MultiHeadResNet, "forward", forward), \
             patch.object(evaluation, "save_confusion_matrix_png"):
            metrics = self.evaluate(config={"img_size": [48, 56]})
        self.assertEqual(metrics["img_size"], [32, 40])
        self.assertEqual(received_shapes, [[32, 40], [32, 40]])

    def test_ddp_prefixed_checkpoint_is_loaded_with_weights_only(self):
        weights = self.altered_checkpoint("model_state", {
            f"module.{key}": value for key, value in self.checkpoint_data["model_state"].items()})
        original_load = evaluation.torch.load
        with patch.object(evaluation.torch, "load", wraps=original_load) as load, \
             patch.object(evaluation, "save_confusion_matrix_png"):
            metrics = evaluation.evaluate_checkpoint(str(weights), str(self.csv), self.output.name,
                                                     data_root=str(self.directory), device="cpu")
        self.assertEqual(metrics["mean_acc"], 0.5)
        self.assertTrue(load.call_args.kwargs["weights_only"])

    def test_image_path_and_type_csv_aliases_use_original_dataset_without_rewriting(self):
        aliased = Path(self.output.name) / "original.csv"
        pd.DataFrame({"image_path": [f"images/car{index}.png" for index in range(3)],
                      "make": ["A ", "B", " B "], "type": ["X", " Y ", "X"]}).to_csv(aliased, index=False)
        original = aliased.read_bytes()
        with patch.object(evaluation, "save_confusion_matrix_png"):
            result = evaluation.evaluate_checkpoint(str(self.checkpoint), str(aliased), self.output.name,
                                                     data_root=str(self.directory), device="cpu")
        self.assertEqual(result["mean_acc"], 0.5)
        self.assertEqual(aliased.read_bytes(), original)

    def test_unseen_ground_truth_labels_fail_before_inference(self):
        unseen_csv = Path(self.output.name) / "unseen.csv"
        df = pd.read_csv(self.csv)
        df.loc[1, "make"] = "Unknown"
        df.to_csv(unseen_csv, index=False)
        with patch.object(evaluation, "run_inference") as inference:
            with self.assertRaisesRegex(ValueError, "classes absent from checkpoint"):
                evaluation.evaluate_checkpoint(str(self.checkpoint), str(unseen_csv), self.output.name,
                                                 data_root=str(self.directory), device="cpu")
        inference.assert_not_called()

    def test_missing_test_image_fails_instead_of_dropping_row(self):
        missing_csv = Path(self.output.name) / "missing.csv"
        df = pd.read_csv(self.csv)
        df.loc[1, "image_name"] = "missing.png"
        df.to_csv(missing_csv, index=False)
        with patch.object(evaluation, "run_inference") as inference:
            with self.assertRaisesRegex(FileNotFoundError, "1 test images are missing"):
                evaluation.evaluate_checkpoint(str(self.checkpoint), str(missing_csv), self.output.name,
                                                 data_root=str(self.directory), device="cpu")
        inference.assert_not_called()

    def test_unreadable_image_fails_complete_evaluation(self):
        broken = Path(self.output.name) / "broken.png"
        broken.write_text("not an image")
        broken_csv = Path(self.output.name) / "broken.csv"
        pd.DataFrame({"root_folder": [self.output.name], "image_name": [broken.name],
                      "make": ["A"], "vehicle_type": ["X"]}).to_csv(broken_csv, index=False)
        with self.assertRaises(OSError):
            evaluation.evaluate_checkpoint(str(self.checkpoint), str(broken_csv), self.output.name, device="cpu")

    def test_checkpoint_head_mapping_and_classes_are_validated(self):
        invalids = [
            ("heads", ["make", "make"]),
            ("heads", ["make", "type"]),
            ("classes_by_head", {"make": ["A", "A"], "vehicle_type": ["X", "Y"]}),
            ("classes_by_head", {"make": [], "vehicle_type": ["X", "Y"]}),
            ("num_classes_by_head", {"make": 3, "vehicle_type": 2}),
            ("class_to_idx_by_head", {"make": {"A": 1, "B": 0}, "vehicle_type": {"X": 0, "Y": 1}}),
            ("img_size", [0, 40]),
        ]
        for key, value in invalids:
            with self.subTest(key=key, value=value):
                path = self.altered_checkpoint(key, value)
                with self.assertRaises(ValueError):
                    evaluation.evaluate_checkpoint(str(path), str(self.csv), self.output.name,
                                                     data_root=str(self.directory), device="cpu")

    def test_heads_subset_is_not_accepted_for_agent_mean_accuracy(self):
        with self.assertRaisesRegex(ValueError, "every checkpoint head"):
            self.evaluate(heads=["make"])

    def test_empty_csv_and_missing_labels_are_not_valid_test_sets(self):
        for frame in (pd.DataFrame(columns=["root_folder", "image_name", "make", "vehicle_type"]),
                      pd.DataFrame({"root_folder": ["images"], "image_name": ["car0.png"],
                                    "make": [" "], "vehicle_type": ["X"]}),
                      pd.DataFrame({"root_folder": ["images"], "image_name": ["car0.png"], "make": ["A"]})):
            with self.subTest(columns=list(frame), rows=len(frame)):
                csv = Path(self.output.name) / "invalid.csv"
                frame.to_csv(csv, index=False)
                with patch.object(evaluation, "run_inference") as infer:
                    with self.assertRaisesRegex(ValueError, "Test CSV"):
                        evaluation.evaluate_checkpoint(str(self.checkpoint), str(csv), self.output.name,
                                                         data_root=str(self.directory), device="cpu")
                infer.assert_not_called()

    def test_missing_image_size_uses_explicit_training_configuration(self):
        checkpoint = dict(self.checkpoint_data)
        checkpoint.pop("img_size")
        path = Path(self.output.name) / "legacy_checkpoint.pth"
        torch.save(checkpoint, path)
        with patch.object(evaluation, "save_confusion_matrix_png"):
            result = evaluation.evaluate_checkpoint(str(path), str(self.csv), self.output.name,
                                                     data_root=str(self.directory), device="cpu", config={"img_size": [32, 40]})
        self.assertEqual(result["img_size"], [32, 40])
        with self.assertRaisesRegex(ValueError, "img_size"):
            evaluation.evaluate_checkpoint(str(path), str(self.csv), self.output.name,
                                             data_root=str(self.directory), device="cpu")

    def test_grouped_checkpoint_reads_its_training_configuration_and_preserves_test_counts(self):
        run = Path(self.output.name) / "exp_01"
        (run / "weights").mkdir(parents=True)
        (run / "train_evaluation").mkdir()
        checkpoint = dict(self.checkpoint_data)
        checkpoint.pop("img_size")
        path = run / "weights" / "best_model.pth"
        torch.save(checkpoint, path)
        (run / "train_evaluation" / "training_config.json").write_text(
            json.dumps({"img_size": [32, 40]}), encoding="utf-8")
        # An old flat config must not replace this attempt's grouped config.
        (run / "training_config.json").write_text(
            json.dumps({"img_size": [48, 56]}), encoding="utf-8")
        with patch.object(evaluation, "save_confusion_matrix_png"):
            result = evaluation.evaluate_checkpoint(
                str(path), str(self.csv), str(run / "test_evaluation"),
                data_root=str(self.directory), device="cpu")
        self.assertEqual(result["img_size"], [32, 40])
        self.assertEqual(result["checkpoint"], str(path.resolve()))
        self.assertEqual(result["evaluated_images"], 3)
        self.assertTrue(result["full_coverage"])
        self.assertEqual(result["mean_acc"], 0.5)
        self.assertEqual(sum(row["total"] for row in result["heads"]["make"]["per_class_accuracy"]), 3)
        self.assertEqual(json.loads((run / "test_evaluation" / "summary.json").read_text()), result)

    def test_nonfinite_model_outputs_cannot_produce_success(self):
        original = evaluation.MultiHeadResNet.forward

        def nan_outputs(model, images):
            outputs = original(model, images)
            outputs["make"][0, 0] = float("nan")
            return outputs

        with patch.object(evaluation.MultiHeadResNet, "forward", nan_outputs):
            with self.assertRaisesRegex(ValueError, "nonfinite inference logits"):
                self.evaluate()

    def test_prediction_only_matching_rejects_unmatched_ground_truth(self):
        gt = evaluation.read_ground_truth(self.csv, ["make", "vehicle_type"])
        prediction = pd.DataFrame({"image_name": ["car0.png", "car1.png"],
                                    "make_pred": ["A", "B"], "make_conf": [0.9, 0.9],
                                    "vehicle_type_pred": ["X", "Y"], "vehicle_type_conf": [0.9, 0.9]})
        with self.assertRaisesRegex(ValueError, "1 test rows have no matching prediction"):
            evaluation.match_predictions(gt, prediction, ["make", "vehicle_type"])

    def test_prediction_only_standalone_rejects_missing_rows_with_failure_exit(self):
        prediction_csv = Path(self.output.name) / "predictions.csv"
        pd.DataFrame({"image_name": ["car0.png"], "make_pred": ["A"],
                      "make_conf": [0.9], "vehicle_type_pred": ["X"],
                      "vehicle_type_conf": [0.9]}).to_csv(prediction_csv, index=False)
        args = argparse.Namespace(ckpt=None, pred_csv=str(prediction_csv), heads=["make", "vehicle_type"],
                                  gt_csv=str(self.csv), threshold=0.0, out_dir=self.output.name)
        with patch.object(evaluation, "parse_args", return_value=args):
            with self.assertRaisesRegex(ValueError, "Incomplete test coverage"):
                evaluation.main()
        self.assertFalse((Path(self.output.name) / "summary.json").is_file())

    def test_prediction_csv_extra_unmatched_rows_are_rejected(self):
        gt = evaluation.read_ground_truth(self.csv, ["make", "vehicle_type"])
        prediction = pd.DataFrame({"image_name": ["car0.png", "car1.png", "car2.png", "other.png"],
                                    "make_pred": ["A"] * 4, "make_conf": [0.9] * 4,
                                    "vehicle_type_pred": ["X"] * 4, "vehicle_type_conf": [0.9] * 4})
        with self.assertRaisesRegex(ValueError, "1 rows that do not match the test CSV"):
            evaluation.match_predictions(gt, prediction, ["make", "vehicle_type"])

    def test_standalone_input_directory_supports_csv_without_root_folder(self):
        csv = Path(self.output.name) / "flat_test.csv"
        pd.read_csv(self.csv).drop(columns="root_folder").to_csv(csv, index=False)
        args = argparse.Namespace(ckpt=str(self.checkpoint), pred_csv=None, heads=None,
                                  gt_csv=str(csv), threshold=0.0, out_dir=self.output.name,
                                  input=str(self.images), data_root="", batch_size=2,
                                  arch=None, img_size=None, device="cpu")
        with patch.object(evaluation, "parse_args", return_value=args), \
             patch.object(evaluation, "save_confusion_matrix_png"):
            evaluation.main()
        summary = json.loads((Path(self.output.name) / "summary.json").read_text())
        self.assertEqual(summary["mean_acc"], 0.5)
        self.assertEqual(summary["total_images"], 3)
        self.assertTrue(summary["full_coverage"])


@unittest.skipIf(bool(IMPORT_ERROR), f"Test evaluation dependencies unavailable: {IMPORT_ERROR}")
class TesterNodeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="resnet-tester-node-")
        self.addCleanup(self.temporary.cleanup)
        self.checkpoint = Path(self.temporary.name) / "best_model.pth"
        self.checkpoint.touch()
        self.test_csv = Path(self.temporary.name) / "test.csv"
        self.validation = {"status": "ok", "mean_acc": 0.99, "make_acc": 0.98,
                           "vehicle_type_acc": 1.0, "best_epoch": 4, "val_loss": 0.2,
                           "train_loss_history": [0.8, 0.2], "val_loss_history": [0.9, 0.3],
                           "mean_acc_history": [0.9, 0.99], "head_acc_history": {"make": [0.8, 0.98]},
                           "heads": {"make": {"acc": 0.98, "loss": 0.2},
                                     "vehicle_type": {"acc": 1.0, "loss": 0.1}}}
        self.state = {"decision_split": "test", "metrics": self.validation,
                      "heads": ["make", "vehicle_type"], "test_csv": str(self.test_csv),
                      "data_root": "/data/images", "attempt_number": 2, "error": "old failure",
                      "training_result": {"status": "success", "best_weights": str(self.checkpoint),
                                          "run_dir": self.temporary.name, "config": {"img_size": [32, 40], "batch_size": 2}}}
        self.test_metrics = {"status": "ok", "mean_acc": 0.5, "make_acc": 1 / 3,
                             "vehicle_type_acc": 2 / 3, "full_coverage": True,
                             "total_images": 3, "matched_images": 3, "evaluated_images": 3,
                             "heads": {"make": {"acc": 1 / 3, "correct": 1, "count": 3, "coverage": 1.0,
                                                "per_class_accuracy": [
                                                    {"class": "A", "correct": 1, "total": 1, "accuracy": 1.0},
                                                    {"class": "B", "correct": 0, "total": 2, "accuracy": 0.0}]},
                                       "vehicle_type": {"acc": 2 / 3, "correct": 2, "count": 3, "coverage": 1.0,
                                                        "per_class_accuracy": [
                                                            {"class": "X", "correct": 2, "total": 2, "accuracy": 1.0},
                                                            {"class": "Y", "correct": 0, "total": 1, "accuracy": 0.0}]}},
                             "checkpoint": str(self.checkpoint), "test_csv": str(self.test_csv),
                             "output_dir": str(Path(self.temporary.name) / "test_evaluation")}

    def test_legacy_validation_mode_is_unchanged(self):
        for state in ({}, {**self.state, "decision_split": "val"}):
            with patch.object(evaluation, "evaluate_checkpoint") as infer:
                self.assertEqual(tester.test_checkpoint_node(state), {})
            infer.assert_not_called()

    def test_low_test_accuracy_replaces_high_validation_for_retry_decision(self):
        original = copy.deepcopy(self.validation)
        with patch.object(evaluation, "evaluate_checkpoint", return_value=self.test_metrics) as infer:
            result = tester.test_checkpoint_node(self.state)
        self.assertEqual(result["metrics"]["mean_acc"], 0.5)
        self.assertEqual(result["metrics"]["make_acc"], 1 / 3)
        self.assertEqual(result["metrics"]["heads"]["make"]["acc"], 1 / 3)
        self.assertEqual(result["metrics"]["heads"]["make"]["loss"], 0.2)
        for head in self.state["heads"]:
            self.assertEqual(result["metrics"]["heads"][head]["per_class_accuracy"],
                             self.test_metrics["heads"][head]["per_class_accuracy"])
            self.assertEqual(result["test_result"]["metrics"]["heads"][head]["per_class_accuracy"],
                             self.test_metrics["heads"][head]["per_class_accuracy"])
        self.assertEqual(result["metrics"]["validation_metrics"], original)
        self.assertEqual(result["validation_metrics"], original)
        self.assertEqual(self.validation, original)
        self.assertEqual(result["metrics"]["val_loss_history"], original["val_loss_history"])
        self.assertEqual(result["metrics"]["decision_split"], "test")
        self.assertTrue(result["metrics"]["test_evaluated"])
        self.assertEqual(result["error"], "")
        self.assertEqual(result["test_result"]["checkpoint"], str(self.checkpoint))
        infer.assert_called_once_with(str(self.checkpoint), str(self.test_csv),
                                       str(Path(self.temporary.name) / "test_evaluation"),
                                       data_root="/data/images", heads=["make", "vehicle_type"],
                                       config={"img_size": [32, 40], "batch_size": 2}, batch_size=2)

    def test_test_failure_never_inherits_validation_primary_scores(self):
        with patch.object(evaluation, "evaluate_checkpoint", side_effect=RuntimeError("CUDA out of memory")):
            result = tester.test_checkpoint_node(self.state)
        metrics = result["metrics"]
        self.assertEqual(metrics["status"], "test_failed")
        self.assertFalse(metrics["test_evaluated"])
        self.assertFalse(metrics["full_coverage"])
        for field in ("mean_acc", "make_acc", "vehicle_type_acc", "heads"):
            self.assertNotIn(field, metrics)
        self.assertEqual(metrics["validation_metrics"]["mean_acc"], 0.99)
        self.assertIn("CUDA out of memory", result["error"])
        self.assertEqual(result["test_result"]["status"], "error")

    def test_missing_checkpoint_or_csv_prevents_inference(self):
        states = [{**self.state, "test_csv": ""},
                  {**self.state, "training_result": {**self.state["training_result"], "best_weights": "/missing/model.pth"}}]
        for state in states:
            with self.subTest(state=state), patch.object(evaluation, "evaluate_checkpoint") as infer:
                result = tester.test_checkpoint_node(state)
            infer.assert_not_called()
            self.assertEqual(result["metrics"]["status"], "test_failed")

    def test_failed_training_or_monitoring_prevents_inference(self):
        states = [{**self.state, "training_result": {"status": "error", "error": "training crash"}},
                  {**self.state, "metrics": {"status": "no_results_csv", "error": "missing log"}}]
        for state in states:
            with self.subTest(state=state), patch.object(evaluation, "evaluate_checkpoint") as infer:
                result = tester.test_checkpoint_node(state)
            infer.assert_not_called()
            self.assertEqual(result["metrics"]["status"], "test_failed")

    def test_partial_coverage_result_is_failure(self):
        incomplete = {**self.test_metrics, "full_coverage": False, "status": "partial_coverage"}
        with patch.object(evaluation, "evaluate_checkpoint", return_value=incomplete):
            result = tester.test_checkpoint_node(self.state)
        self.assertEqual(result["metrics"]["status"], "test_failed")
        self.assertNotIn("mean_acc", result["metrics"])

    def test_parent_cuda_cache_is_released_after_success_and_inference_failure(self):
        for outcome in (self.test_metrics, RuntimeError("inference failed")):
            with self.subTest(outcome=type(outcome).__name__):
                with patch.object(evaluation, "evaluate_checkpoint", side_effect=[outcome]), \
                     patch.object(torch.cuda, "is_available", return_value=True), \
                     patch.object(torch.cuda, "empty_cache") as empty_cache:
                    result = tester.test_checkpoint_node(self.state)
                empty_cache.assert_called_once_with()
                self.assertEqual(result["metrics"]["status"], "test_failed" if isinstance(outcome, Exception) else "ok")


if __name__ == "__main__":
    unittest.main()
