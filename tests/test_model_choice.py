"""Allow only catalogued local backbones while preserving training controls."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.nodes import config_optimizer


class OptimizerModelChoiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.available = []
        for arch in ("resnet18", "resnet34", "resnet50"):
            weights = self.root / f"{arch}.pth"
            weights.write_bytes(b"Catalog fixture: actual checkpoint discovery is mocked")
            self.available.append({"arch": arch, "weights": str(weights)})
        self.current = {
            "arch": "resnet18", "weights": self.available[0]["weights"],
            "train_csv": str(self.root / "train.csv"), "val_csv": str(self.root / "val.csv"),
            "data_root": None, "heads": ["make", "vehicle_type"], "epochs": 30,
            "batch_size": 32, "num_workers": 4, "img_size": [224, 224], "lr": 1e-4,
            "weight_decay": 1e-4, "gamma": 2.0, "patience": 7, "monitor": "mean",
            "head_loss_weights": {"make": 1.0, "vehicle_type": 1.0}, "alpha_json": None,
            "nproc_per_node": 1, "out": str(self.root / "session"), "name": "exp_01",
        }

    def optimize(self, proposed, available=None):
        response = AIMessage(content=json.dumps({
            "updated_config": proposed, "rationale": "Compare a supported local backbone.",
        }))
        invoke = Mock(return_value=response)
        with patch.object(config_optimizer, "discover_pretrained_weights", return_value=(
                self.available if available is None else available)) as discovery, \
             patch.object(config_optimizer, "get_llm", return_value=SimpleNamespace(invoke=invoke)), \
             patch.object(config_optimizer, "console"):
            result = config_optimizer.config_optimizer_node({
                "current_config": self.current, "attempt_number": 1,
                "model_type": self.current["arch"], "weights_path": self.current["weights"],
                "session_dir": str(self.root / "session"), "metrics": {"status": "ok", "mean_acc": 0.7},
            })
        discovery.assert_called_once()
        invoke.assert_called_once()
        return result, invoke

    def assert_selected(self, result, arch, weights):
        self.assertEqual(result["optimisation_status"], "success")
        self.assertEqual(result["current_config"]["arch"], arch)
        self.assertEqual(result["current_config"]["weights"], weights)
        self.assertEqual(result["model_type"], arch)
        self.assertEqual(result["weights_path"], weights)
        self.assertEqual(result["current_config"]["name"], "exp_02")
        self.assertEqual(result["config_history"][0]["config"], result["current_config"])

    def assert_rejected(self, result):
        self.assertEqual(result["optimisation_status"], "error")
        self.assertEqual(result["stop_reason"], "error")
        for key in ("current_config", "config_history", "model_type", "weights_path"):
            self.assertNotIn(key, result)

    def test_matching_architecture_and_weights_update_both_configuration_and_state(self):
        original = copy.deepcopy(self.current)
        selection = self.available[1]
        result, _ = self.optimize({**selection, "lr": 2e-4})
        self.assert_selected(result, selection["arch"], selection["weights"])
        self.assertEqual(result["current_config"]["lr"], 2e-4)
        for key in ("train_csv", "val_csv", "data_root", "heads", "monitor", "nproc_per_node", "out"):
            self.assertEqual(result["current_config"][key], original[key])
        self.assertEqual(self.current, original)

    def test_architecture_only_selects_its_unique_matching_file(self):
        selection = self.available[2]
        result, _ = self.optimize({"arch": selection["arch"]})
        self.assert_selected(result, selection["arch"], selection["weights"])

    def test_weights_path_or_basename_infers_the_matching_architecture(self):
        selection = self.available[1]
        for weights in (selection["weights"], Path(selection["weights"]).name):
            with self.subTest(weights=weights):
                result, _ = self.optimize({"weights": weights})
                self.assert_selected(result, selection["arch"], selection["weights"])

    def test_mismatched_unknown_missing_and_outside_model_choices_are_rejected(self):
        outside = self.root / "outside" / "resnet34.pth"
        outside.parent.mkdir()
        outside.write_bytes(b"Same basename as an offered model; path is outside the catalog")
        proposals = [
            {"arch": "resnet50", "weights": self.available[1]["weights"]},
            {"arch": "resnet101"},
            {"weights": "unknown.pth"},
            {"arch": "resnet34", "weights": str(self.root / "missing.pth")},
            {"arch": "resnet34", "weights": str(outside)},
        ]
        original = copy.deepcopy(self.current)
        for proposal in proposals:
            with self.subTest(proposal=proposal):
                result, _ = self.optimize(proposal)
                self.assert_rejected(result)
                self.assertEqual(self.current, original)

    def test_architecture_only_is_rejected_when_matching_files_are_ambiguous(self):
        extra = {"arch": "resnet34", "weights": str(self.root / "resnet34-other.pth")}
        Path(extra["weights"]).write_bytes(b"A second catalogued ResNet34 fixture")
        available = [*self.available, extra]
        result, _ = self.optimize({"arch": "resnet34"}, available)
        self.assert_rejected(result)
        result, _ = self.optimize(extra, available)
        self.assert_selected(result, extra["arch"], extra["weights"])

    def test_ambiguous_basename_requires_the_exact_weights_path(self):
        extra_path = self.root / "duplicate" / "resnet34.pth"
        extra_path.parent.mkdir()
        extra_path.write_bytes(b"Catalog fixture with a duplicate basename")
        extra = {"arch": "resnet34", "weights": str(extra_path)}
        available = [*self.available, extra]
        result, _ = self.optimize({"weights": "resnet34.pth"}, available)
        self.assert_rejected(result)
        result, _ = self.optimize({"weights": str(extra_path)}, available)
        self.assert_selected(result, extra["arch"], extra["weights"])

    def test_unchanged_custom_initial_weights_outside_the_catalog_are_retained(self):
        custom = self.root / "custom-initial-backbone.pth"
        custom.write_bytes(b"User supplied initial weights")
        self.current["weights"] = str(custom)
        for proposed in ({"lr": 2e-4}, {"arch": "resnet18", "weights": str(custom), "lr": 2e-4}):
            with self.subTest(proposed=proposed):
                result, _ = self.optimize(proposed, [])
                self.assert_selected(result, "resnet18", str(custom))
                self.assertEqual(result["current_config"]["lr"], 2e-4)

    def test_five_hyperparameter_changes_plus_model_pair_are_allowed_but_six_are_rejected(self):
        changes = {"epochs": 40, "batch_size": 16, "img_size": [192, 192],
                   "lr": 2e-4, "weight_decay": 2e-4}
        selection = self.available[1]
        result, _ = self.optimize({**selection, **changes})
        self.assert_selected(result, selection["arch"], selection["weights"])
        for key, value in changes.items():
            self.assertEqual(result["current_config"][key], value)
        rejected, _ = self.optimize({**selection, **changes, "gamma": 3.0})
        self.assert_rejected(rejected)
        self.assertIn("five", rejected["optimisation_error"].lower())

    def test_single_request_contains_the_exact_local_model_catalog(self):
        result, invoke = self.optimize({"lr": 2e-4})
        self.assertEqual(result["optimisation_status"], "success")
        messages = invoke.call_args.args[0]
        self.assertEqual(len(messages), 2)
        self.assertIsInstance(messages[0], SystemMessage)
        self.assertIsInstance(messages[1], HumanMessage)
        context = json.loads(messages[1].content)
        self.assertEqual(context["available_pretrained_models"], self.available)
        self.assertEqual(context["current_config"]["arch"], "resnet18")
        self.assertEqual(context["current_config"]["weights"], self.available[0]["weights"])
        for selection in self.available:
            self.assertIn(selection["arch"], messages[1].content)
            self.assertIn(selection["weights"], messages[1].content)
        self.assertIn("available_pretrained_models", messages[0].content)


if __name__ == "__main__":
    unittest.main()
