from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from agent.nodes import onnx_exporter
except ImportError as exc:
    NODE_IMPORT_ERROR = str(exc)
else:
    NODE_IMPORT_ERROR = ""

try:
    import numpy as np
    import onnx
    import torch
    from training.export import export_resnet_onnx
    from training.train_multihead_ddp import MultiHeadResNet
except ImportError as exc:
    EXPORT_IMPORT_ERROR = str(exc)
else:
    EXPORT_IMPORT_ERROR = ""


@unittest.skipIf(bool(NODE_IMPORT_ERROR), f"Export node dependencies unavailable: {NODE_IMPORT_ERROR}")
class ExportNodeTests(unittest.TestCase):
    def valid_best_experiment(self, weights: str, config: dict | None = None) -> dict:
        return {
            "best_weights": weights,
            "config": config or {"img_size": [32, 40]},
            "training_status": "success", "target_metric": "mean_acc",
            "metrics": {
                "status": "ok", "mean_acc": 0.75,
                "nonfinite_loss": False, "nonfinite_accuracy": False,
                "train_loss": 0.5, "val_loss": 0.6, "best_val_loss": 0.6,
                "heads": {"make": {"acc": 0.75, "loss": 0.6}},
            },
        }

    def test_export_none_skips_checkpoint_loading(self):
        with patch.object(onnx_exporter, "export_resnet_onnx") as export:
            result = onnx_exporter.onnx_exporter_node({"export_format": "none"})
        export.assert_not_called()
        self.assertEqual(result, {"onnx_path": ""})

    def test_best_experiment_checkpoint_takes_priority_over_latest(self):
        with tempfile.TemporaryDirectory(prefix="resnet-export-priority-") as temporary:
            best = Path(temporary) / "best"
            latest = Path(temporary) / "latest"
            best.mkdir()
            latest.mkdir()
            for directory in (best, latest):
                (directory / "best_model.pth").touch()
            chosen = onnx_exporter._resolve_best_weights({"run_dir": str(best)}, {"run_dir": str(latest)})
            self.assertEqual(chosen, str(best / "best_model.pth"))

    def test_missing_best_checkpoint_falls_back_to_latest_last_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="resnet-export-fallback-") as temporary:
            last = Path(temporary) / "last_model.pth"
            last.touch()
            chosen = onnx_exporter._resolve_best_weights({"run_dir": "/missing/best/run"}, {"last_weights": str(last)})
            self.assertEqual(chosen, str(last))

    def test_grouped_best_checkpoint_is_preferred_over_flat_and_latest_checkpoints(self):
        with tempfile.TemporaryDirectory(prefix="resnet-export-grouped-priority-") as temporary:
            best = Path(temporary) / "best"
            latest = Path(temporary) / "latest"
            for directory in (best, latest):
                (directory / "weights").mkdir(parents=True)
                (directory / "weights" / "best_model.pth").touch()
                (directory / "best_model.pth").touch()
            chosen = onnx_exporter._resolve_best_weights(
                {"best_weights": str(best / "best_model.pth"), "run_dir": str(best)},
                {"run_dir": str(latest)})
            self.assertEqual(chosen, str(best / "weights" / "best_model.pth"))

    def test_grouped_latest_last_checkpoint_remains_available_to_path_resolver(self):
        with tempfile.TemporaryDirectory(prefix="resnet-export-grouped-last-") as temporary:
            run = Path(temporary)
            (run / "weights").mkdir()
            checkpoint = run / "weights" / "last_model.pth"
            checkpoint.touch()
            self.assertEqual(onnx_exporter._resolve_best_weights({}, {"run_dir": str(run)}), str(checkpoint))

    def test_export_node_never_uses_grouped_last_checkpoint_without_validated_best(self):
        with tempfile.TemporaryDirectory(prefix="resnet-export-grouped-best-missing-") as temporary:
            run = Path(temporary)
            (run / "weights").mkdir()
            (run / "weights" / "last_model.pth").touch()
            best = self.valid_best_experiment(str(run / "weights" / "best_model.pth"))
            best["run_dir"] = str(run)
            with patch.object(onnx_exporter, "export_resnet_onnx") as export:
                result = onnx_exporter.onnx_exporter_node({"export_format": "onnx", "best_experiment": best})
            export.assert_not_called()
            self.assertIn("No ResNet checkpoint found", result["error"])

    def test_no_checkpoints_returns_error_without_export(self):
        with patch.object(onnx_exporter, "export_resnet_onnx") as export:
            result = onnx_exporter.onnx_exporter_node({"export_format": "onnx"})
        export.assert_not_called()
        self.assertIn("No successful ResNet checkpoint", result["error"])

    def test_node_passes_best_model_training_configuration_to_export(self):
        with tempfile.TemporaryDirectory(prefix="resnet-export-config-") as temporary:
            weights = Path(temporary) / "best_model.pth"
            weights.touch()
            best_config = {"img_size": [32, 40]}
            state = {
                "export_format": "onnx",
                "best_experiment": self.valid_best_experiment(str(weights), best_config),
                "training_result": {"config": {"img_size": [224, 224]}},
            }
            with patch.object(onnx_exporter, "export_resnet_onnx", return_value="model.onnx") as export:
                result = onnx_exporter.onnx_exporter_node(state)
            export.assert_called_once_with(str(weights), best_config)
            self.assertEqual(result, {"onnx_path": "model.onnx"})

    def test_invalid_only_attempt_with_checkpoint_is_not_exported(self):
        with tempfile.TemporaryDirectory(prefix="resnet-export-invalid-only-") as temporary:
            checkpoint = Path(temporary) / "best_model.pth"
            checkpoint.touch()
            state = {
                "export_format": "onnx", "target_metric": "mean_acc",
                "best_experiment": {},
                "training_result": {"status": "success", "best_weights": str(checkpoint)},
                "metrics": {"status": "ok", "mean_acc": 0.75, "nonfinite_loss": True},
            }
            with patch.object(onnx_exporter, "export_resnet_onnx") as export:
                result = onnx_exporter.onnx_exporter_node(state)
            export.assert_not_called()
            self.assertEqual(result["onnx_path"], "")
            self.assertIn("valid metrics", result["error"])

    def test_missing_validated_best_weights_does_not_export_invalid_latest(self):
        with tempfile.TemporaryDirectory(prefix="resnet-export-best-missing-") as temporary:
            latest = Path(temporary) / "latest"
            latest.mkdir()
            checkpoint = latest / "best_model.pth"
            checkpoint.touch()
            state = {
                "export_format": "onnx", "target_metric": "mean_acc",
                "best_experiment": self.valid_best_experiment(str(Path(temporary) / "missing" / "best_model.pth")),
                "training_result": {"status": "success", "best_weights": str(checkpoint), "run_dir": str(latest)},
                "metrics": {"status": "ok", "nonfinite_loss": True, "mean_acc": 0.75},
            }
            with patch.object(onnx_exporter, "export_resnet_onnx") as export:
                result = onnx_exporter.onnx_exporter_node(state)
            export.assert_not_called()
            self.assertEqual(result["onnx_path"], "")
            self.assertIn("No ResNet checkpoint found", result["error"])

    def test_nonfinite_best_experiment_is_rejected_even_when_checkpoint_exists(self):
        with tempfile.TemporaryDirectory(prefix="resnet-export-invalid-best-") as temporary:
            weights = Path(temporary) / "best_model.pth"
            weights.touch()
            for field in ("nonfinite_loss", "nonfinite_accuracy"):
                with self.subTest(field=field):
                    best = self.valid_best_experiment(str(weights))
                    best["metrics"][field] = True
                    with patch.object(onnx_exporter, "export_resnet_onnx") as export:
                        result = onnx_exporter.onnx_exporter_node({"export_format": "onnx", "best_experiment": best})
                    export.assert_not_called()
                    self.assertEqual(result["onnx_path"], "")
                    self.assertIn("valid metrics", result["error"])


@unittest.skipIf(bool(EXPORT_IMPORT_ERROR), f"ONNX export dependencies unavailable: {EXPORT_IMPORT_ERROR}")
class ResNetOnnxIntegrationTests(unittest.TestCase):
    def test_grouped_checkpoint_export_infers_image_size_from_training_evaluation_folder(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        with tempfile.TemporaryDirectory(prefix="resnet-onnx-grouped-") as temporary, torch.random.fork_rng(devices=[]):
            run = Path(temporary)
            (run / "weights").mkdir()
            (run / "train_evaluation").mkdir()
            model = MultiHeadResNet("resnet18", None, {"make": 2, "vehicle_type": 2}).eval()
            checkpoint = {
                "model_state": model.state_dict(), "arch": "resnet18", "heads": ["make", "vehicle_type"],
                "num_classes_by_head": {"make": 2, "vehicle_type": 2},
                "classes_by_head": {"make": ["A", "B"], "vehicle_type": ["X", "Y"]},
                "class_to_idx_by_head": {"make": {"A": 0, "B": 1}, "vehicle_type": {"X": 0, "Y": 1}},
            }
            weights = run / "weights" / "best_model.pth"
            torch.save(checkpoint, weights)
            (run / "train_evaluation" / "training_config.json").write_text(
                json.dumps({"img_size": [32, 40]}), encoding="utf-8")
            (run / "training_config.json").write_text(json.dumps({"img_size": [48, 56]}), encoding="utf-8")
            exported = Path(export_resnet_onnx(str(weights), {}))
            graph = onnx.load(str(exported))
            onnx.checker.check_model(graph)
            self.assertEqual(exported.parent, run / "weights")
            dimensions = graph.graph.input[0].type.tensor_type.shape.dim
            self.assertEqual([dimension.dim_value for dimension in dimensions[1:]], [3, 32, 40])
            metadata = json.loads(exported.with_suffix(".metadata.json").read_text())
            self.assertEqual(metadata["preprocessing"]["img_size"], [32, 40])
            self.assertEqual(metadata["classes_by_head"], checkpoint["classes_by_head"])

    def test_checkpoint_export_matches_pytorch_logits_and_preserves_metadata(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        with tempfile.TemporaryDirectory(prefix="resnet-onnx-integration-") as temporary, torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            model = MultiHeadResNet("resnet18", None, {"make": 3, "color": 2}).eval()
            checkpoint = {
                "model_state": model.state_dict(), "arch": "resnet18",
                "heads": ["make", "color"], "num_classes_by_head": {"make": 3, "color": 2},
                "classes_by_head": {"make": ["Ford", "Honda", "Toyota"], "color": ["blue", "red"]},
                "class_to_idx_by_head": {"make": {"Ford": 0, "Honda": 1, "Toyota": 2}, "color": {"blue": 0, "red": 1}},
                "img_size": [32, 40],
            }
            weights = Path(temporary) / "best_model.pth"
            torch.save(checkpoint, weights)
            exported_path = Path(export_resnet_onnx(str(weights), {"img_size": [224, 224]}))
            exported = onnx.load(str(exported_path))
            onnx.checker.check_model(exported)
            self.assertEqual([output.name for output in exported.graph.output], ["make_logits", "color_logits"])
            dims = exported.graph.input[0].type.tensor_type.shape.dim
            self.assertEqual(dims[0].dim_param, "batch")
            self.assertEqual([dimension.dim_value for dimension in dims[1:]], [3, 32, 40])
            metadata = json.loads(exported_path.with_suffix(".metadata.json").read_text())
            self.assertEqual(metadata["classes_by_head"], checkpoint["classes_by_head"])
            self.assertEqual(metadata["class_to_idx_by_head"], checkpoint["class_to_idx_by_head"])
            self.assertEqual(metadata["preprocessing"]["img_size"], [32, 40])
            self.assertEqual(metadata["preprocessing"]["mean"], [0.485, 0.456, 0.406])
            self.assertEqual(metadata["preprocessing"]["std"], [0.229, 0.224, 0.225])
            self.assertEqual(metadata["outputs"], {"make": "make_logits", "color": "color_logits"})

            # A batch of two exercises the exported dynamic batch dimension.
            images = torch.randn(2, 3, 32, 40)
            with torch.no_grad():
                expected = model(images)
            try:
                import onnxruntime as ort
            except ImportError:
                from onnx.reference import ReferenceEvaluator

                actual = ReferenceEvaluator(exported).run(None, {"images": images.numpy()})
            else:
                session = ort.InferenceSession(str(exported_path), providers=["CPUExecutionProvider"])
                actual = session.run(None, {"images": images.numpy()})
            for head, logits in zip(checkpoint["heads"], actual):
                np.testing.assert_allclose(logits, expected[head].numpy(), rtol=1e-4, atol=1e-5, err_msg=f"{head} logits differ after export")


if __name__ == "__main__":
    unittest.main()
