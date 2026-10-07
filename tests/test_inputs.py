"""Input validation tests; these never load a model or use CUDA/an API."""
from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from rich.console import Console

from agent.nodes.dataset_validator import canonical_target_metric, dataset_validator_node
from agent.nodes.user_input import user_input_node
from main import parse_args
from training.data import read_dataset_csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class InputValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.images = self.root / "images"
        self.images.mkdir()
        Image.new("RGB", (8, 8), color="white").save(self.images / "car.png")
        # Validation checks local weights existence, without deserializing them.
        self.weights = self.root / "resnet18.pth"
        self.weights.write_bytes(b"validator fixture; not a model")
        self.rows = [{"root_folder": "images", "image_name": "car.png",
                      "make": "Toyota", "color": "white"}]
        self._write_split("train", self.rows)
        self._write_split("val", self.rows)
        output = Console(file=io.StringIO())
        self.addCleanup(patch.stopall)
        patch("agent.nodes.user_input.console", output).start()
        patch("agent.nodes.dataset_validator.console", output).start()
        self.state = {
            "train_csv": str(self.root / "train.csv"),
            "val_csv": str(self.root / "val.csv"),
            "decision_split": "val",
            "data_root": str(self.root),
            "heads": ["make", "color"],
            "model_type": "resnet18",
            "weights_path": str(self.weights),
            "target_metric": "mean_acc",
            "target_value": 0.85,
            "max_attempts": 5,
            "nproc_per_node": 1,
            "export_format": "none",
            "interactive": False,
        }

    def _write_split(self, split: str, rows: list[dict[str, str]],
                     fields: list[str] | None = None) -> None:
        with (self.root / f"{split}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields or ["root_folder", "image_name", "make", "color"])
            writer.writeheader()
            writer.writerows(rows)

    def _validate(self, **updates):
        return dataset_validator_node({**self.state, **updates})

    def _assert_invalid(self, result, expected: str) -> None:
        self.assertFalse(result["is_valid"])
        self.assertIn(expected, "\n".join(result["validation_report"]["errors"]))

    def _cli_state(self, argv: list[str]) -> dict:
        args = parse_args(["--weights", str(self.weights), "--export", "none",
                           "--decision-split", "val", *argv])
        return {
            "dataset_path": args.dataset,
            "train_csv": args.train_csv,
            "val_csv": args.val_csv,
            "test_csv": args.test_csv,
            "decision_split": args.decision_split,
            "data_root": args.data_root,
            "heads": args.heads,
            "model_type": args.arch,
            "weights_path": args.weights,
            "target_metric": args.target_metric,
            "target_value": args.target_value,
            "max_attempts": args.max_attempts,
            "nproc_per_node": args.nproc_per_node,
            "export_format": args.export,
            "interactive": False,
        }

    def test_environment_paths_and_heads_validate_without_cli_data_inputs(self) -> None:
        environment = {
            "TRAIN_CSV": str(self.root / "train.csv"),
            "VAL_CSV": str(self.root / "val.csv"),
            "DATA_ROOT": str(self.root),
            "HEADS": "make color",
        }
        with patch.dict(os.environ, environment, clear=True):
            normalized = user_input_node(self._cli_state([]))
        self.assertEqual(normalized["train_csv"], environment["TRAIN_CSV"])
        self.assertEqual(normalized["val_csv"], environment["VAL_CSV"])
        self.assertEqual(normalized["data_root"], environment["DATA_ROOT"])
        self.assertEqual(normalized["heads"], ["make", "color"])
        result = dataset_validator_node(normalized)
        self.assertTrue(result["is_valid"], result["validation_report"]["errors"])

    def test_cli_data_inputs_override_environment_paths_and_heads(self) -> None:
        environment = {
            "TRAIN_CSV": str(self.root / "missing_train.csv"),
            "VAL_CSV": str(self.root / "missing_val.csv"),
            "DATA_ROOT": str(self.root / "missing_images"),
            "HEADS": "unknown",
            "DATASET_PATH": str(self.root / "other_dataset"),
        }
        argv = ["--train-csv", str(self.root / "train.csv"),
                "--val-csv", str(self.root / "val.csv"),
                "--data-root", str(self.root), "--heads", "color", "make"]
        with patch.dict(os.environ, environment, clear=True):
            normalized = user_input_node(self._cli_state(argv))
        self.assertEqual(normalized["train_csv"], str(self.root / "train.csv"))
        self.assertEqual(normalized["val_csv"], str(self.root / "val.csv"))
        self.assertEqual(normalized["data_root"], str(self.root))
        self.assertEqual(normalized["heads"], ["color", "make"])
        result = dataset_validator_node(normalized)
        self.assertTrue(result["is_valid"], result["validation_report"]["errors"])

    def test_cli_dataset_shorthand_overrides_environment_split_paths(self) -> None:
        environment = {
            "TRAIN_CSV": str(self.root / "missing_train.csv"),
            "VAL_CSV": str(self.root / "missing_val.csv"),
            "DATASET_PATH": str(self.root / "other_dataset"),
            "DATA_ROOT": str(self.root), "HEADS": "make color",
        }
        with patch.dict(os.environ, environment, clear=True):
            normalized = user_input_node(self._cli_state(["--dataset", str(self.root)]))
        self.assertEqual(normalized["dataset_path"], str(self.root))
        self.assertEqual(normalized["train_csv"], str(self.root / "train.csv"))
        self.assertEqual(normalized["val_csv"], str(self.root / "val.csv"))
        result = dataset_validator_node(normalized)
        self.assertTrue(result["is_valid"], result["validation_report"]["errors"])

    def test_environment_dataset_shorthand_allows_individual_split_overrides(self) -> None:
        self._write_split("alternate_train", self.rows)
        self._write_split("alternate_val", self.rows)
        for overrides in ({}, {"TRAIN_CSV": str(self.root / "alternate_train.csv")},
                          {"VAL_CSV": str(self.root / "alternate_val.csv")}):
            with self.subTest(overrides=overrides):
                environment = {
                    "DATASET_PATH": str(self.root), "DATA_ROOT": str(self.root),
                    "HEADS": "make color", **overrides,
                }
                with patch.dict(os.environ, environment, clear=True):
                    normalized = user_input_node(self._cli_state([]))
                self.assertEqual(normalized["train_csv"],
                                 overrides.get("TRAIN_CSV", str(self.root / "train.csv")))
                self.assertEqual(normalized["val_csv"],
                                 overrides.get("VAL_CSV", str(self.root / "val.csv")))
                result = dataset_validator_node(normalized)
                self.assertTrue(result["is_valid"], result["validation_report"]["errors"])

    def test_environment_paths_and_heads_strip_surrounding_whitespace(self) -> None:
        environment = {
            "TRAIN_CSV": f"  {self.root / 'train.csv'}  ",
            "VAL_CSV": f"\t{self.root / 'val.csv'}\t",
            "DATA_ROOT": f" {self.root} ",
            "HEADS": "  make   color  ",
        }
        with patch.dict(os.environ, environment, clear=True):
            normalized = user_input_node(self._cli_state([]))
        self.assertEqual(normalized["train_csv"], str(self.root / "train.csv"))
        self.assertEqual(normalized["val_csv"], str(self.root / "val.csv"))
        self.assertEqual(normalized["data_root"], str(self.root))
        self.assertEqual(normalized["heads"], ["make", "color"])
        result = dataset_validator_node(normalized)
        self.assertTrue(result["is_valid"], result["validation_report"]["errors"])

    def test_environment_data_inputs_skip_interactive_prompts(self) -> None:
        environment = {
            "TRAIN_CSV": str(self.root / "train.csv"),
            "VAL_CSV": str(self.root / "val.csv"),
            "DATA_ROOT": str(self.root), "HEADS": "make color",
        }
        with patch.dict(os.environ, environment, clear=True), \
                patch("agent.nodes.user_input.Prompt.ask") as prompt:
            normalized = user_input_node({**self._cli_state([]), "interactive": True})
        prompt.assert_not_called()
        self.assertEqual(normalized["heads"], ["make", "color"])

    def test_parser_uses_yaml_scalar_defaults_and_preserves_explicit_cli_values(self) -> None:
        defaults = {"arch": "resnet50", "target_metric": "color_acc", "target_value": 0.99,
                    "max_attempts": 20, "nproc_per_node": 2, "export_format": "none"}
        with patch("agent.nodes.user_input._load_defaults", return_value=defaults):
            args = parse_args([])
            explicit = parse_args(["--arch", "resnet18", "--target-metric", "mean_acc",
                                   "--target-value", "0", "--max-attempts", "5",
                                   "--nproc-per-node", "1", "--export", "onnx"])
        self.assertEqual((args.arch, args.target_metric, args.target_value, args.max_attempts,
                          args.nproc_per_node, args.export),
                         ("resnet50", "color_acc", 0.99, 20, 2, "none"))
        self.assertEqual((explicit.arch, explicit.target_metric, explicit.target_value,
                          explicit.max_attempts, explicit.nproc_per_node, explicit.export),
                         ("resnet18", "mean_acc", 0, 5, 1, "onnx"))

    def test_valid_dataset_returns_resolved_paths_and_training_class_mapping(self) -> None:
        second_row = {**self.rows[0], "make": "Honda", "color": "black"}
        self._write_split("train", [self.rows[0], second_row])
        result = self._validate()
        self.assertTrue(result["is_valid"], result["validation_report"]["errors"])
        report = result["validation_report"]
        self.assertEqual(report["classes_by_head"], {"make": ["Honda", "Toyota"], "color": ["black", "white"]})
        self.assertEqual(report["class_to_idx_by_head"]["make"], {"Honda": 0, "Toyota": 1})
        self.assertEqual(report["num_classes_by_head"], {"make": 2, "color": 2})
        self.assertEqual(report["split_stats"], {"train": {"images": 2}, "val": {"images": 1}})
        for key in ("train_csv", "val_csv", "data_root", "weights_path"):
            self.assertTrue(Path(result[key]).is_absolute())

    def test_absolute_csv_image_root_overrides_data_root(self) -> None:
        row = {**self.rows[0], "root_folder": str(self.images)}
        self._write_split("train", [row])
        self._write_split("val", [row])
        result = self._validate(data_root=str(PROJECT_ROOT))
        self.assertTrue(result["is_valid"], result["validation_report"]["errors"])

    def test_dataset_shorthand_and_duplicate_heads_preserve_order(self) -> None:
        raw = {**self.state, "dataset_path": str(self.root), "train_csv": "", "val_csv": "",
               "heads": ["color", "make", "color"]}
        normalized = user_input_node(raw)
        self.assertEqual(normalized["heads"], ["color", "make"])
        self.assertEqual(normalized["train_csv"], str(self.root / "train.csv"))
        self.assertEqual(normalized["val_csv"], str(self.root / "val.csv"))
        result = dataset_validator_node(normalized)
        self.assertTrue(result["is_valid"])

    def test_unseen_validation_label_is_rejected(self) -> None:
        self._write_split("val", [{**self.rows[0], "make": "Honda"}])
        self._assert_invalid(self._validate(), "classes absent from training")

    def test_missing_image_is_rejected(self) -> None:
        self._write_split("val", [{**self.rows[0], "image_name": "missing.png"}])
        self._assert_invalid(self._validate(), "image files do not exist")

    def test_unreadable_image_is_rejected(self) -> None:
        (self.images / "car.png").write_bytes(b"corrupt image")
        self._assert_invalid(self._validate(), "unreadable image")

    def test_missing_and_whitespace_head_labels_are_rejected(self) -> None:
        for value in ("", "   "):
            with self.subTest(value=value):
                self._write_split("train", [{**self.rows[0], "color": value}])
                self._assert_invalid(self._validate(), "missing labels")

    def test_empty_split_and_missing_csv_column_are_rejected(self) -> None:
        self._write_split("val", [])
        self._assert_invalid(self._validate(), "CSV is empty")
        self._write_split("val", [{"root_folder": "images", "image_name": "car.png", "make": "Toyota"}],
                          ["root_folder", "image_name", "make"])
        self._assert_invalid(self._validate(), "missing columns")

    def test_missing_weights_file_is_rejected(self) -> None:
        self._assert_invalid(self._validate(weights_path=str(self.root / "absent.pth")), "Pretrained weights not found")

    def test_moduledict_reserved_and_dotted_head_names_are_rejected(self) -> None:
        for name in ("vehicle.type", "train", "keys", "__getitem__", "training"):
            with self.subTest(name=name):
                self._assert_invalid(self._validate(heads=[name]), "ModuleDict key")
        self._assert_invalid(self._validate(heads=["bad,name"]), "cannot contain commas")
        for name in ("mean", "final_make"):
            with self.subTest(name=name):
                self._assert_invalid(self._validate(heads=[name]), "accuracy metric names")

    def test_target_aliases_retain_head_case_and_unknown_targets_fail(self) -> None:
        self.assertEqual(canonical_target_metric("accuracy", ["Make"]), "mean_acc")
        self.assertEqual(canonical_target_metric("mean", ["Make"]), "mean_acc")
        self.assertEqual(canonical_target_metric("Make", ["Make"]), "Make_acc")
        self.assertEqual(canonical_target_metric("Make_acc", ["Make"]), "Make_acc")
        self.assertEqual(self._validate(target_metric="make")["target_metric"], "make_acc")
        self._assert_invalid(self._validate(target_metric="mAP50"), "Unknown target_metric")

    def test_zero_target_is_preserved_and_nonfinite_targets_fail(self) -> None:
        normalized = user_input_node({**self.state, "target_value": 0})
        self.assertEqual(normalized["target_value"], 0)
        self.assertTrue(dataset_validator_node(normalized)["is_valid"])
        for value in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(value=value):
                self._assert_invalid(self._validate(target_value=value), "target_value")

    def test_invalid_numeric_user_inputs_are_reported_without_crashing(self) -> None:
        for field, value in (("target_value", "invalid"), ("target_value", "nan"),
                             ("max_attempts", "1.5"), ("nproc_per_node", "invalid")):
            with self.subTest(field=field, value=value):
                normalized = user_input_node({**self.state, field: value})
                self.assertTrue(normalized["input_errors"])
                self._assert_invalid(dataset_validator_node(normalized), field)
        self._assert_invalid(self._validate(max_attempts=0), "max_attempts")
        self._assert_invalid(self._validate(nproc_per_node=0), "nproc_per_node")

    def test_head_loss_weights_validate_values_and_parse_cli_pairs(self) -> None:
        normalized = user_input_node({**self.state, "head_loss_weights": ["make=0.5", "color=0"]})
        self.assertEqual(normalized["head_loss_weights"], {"make": 0.5, "color": 0.0})
        self.assertTrue(dataset_validator_node(normalized)["is_valid"])
        self._assert_invalid(self._validate(head_loss_weights={"make": 0, "color": 0}), "At least one")
        for value in (-1, float("inf"), float("nan")):
            with self.subTest(value=value):
                self._assert_invalid(self._validate(head_loss_weights={"make": value}), "finite and nonnegative")
        normalized = user_input_node({**self.state, "head_loss_weights": ["unknown=1", "malformed"]})
        self.assertEqual(len(normalized["input_errors"]), 2)
        self.assertFalse(dataset_validator_node(normalized)["is_valid"])

    def test_alpha_config_requires_nested_nonnegative_finite_weights(self) -> None:
        path = self.root / "alpha.json"
        path.write_text(json.dumps({"make": {"Toyota": 0.5}}))
        self.assertTrue(self._validate(alpha_json=str(path))["is_valid"])
        path.write_text(json.dumps({"make": {"Toyota": 0}}))
        self.assertTrue(self._validate(alpha_json=str(path))["is_valid"])
        for value in ([], {"make": []},
                      {"make": {"Toyota": -1}}, {"make": {"Toyota": float("inf")}},
                      {"make": {"Toyota": float("nan")}}):
            with self.subTest(value=value):
                path.write_text(json.dumps(value))
                self._assert_invalid(self._validate(alpha_json=str(path)), "Invalid alpha JSON")

    def test_cli_model_alias_and_gpu_options(self) -> None:
        args = parse_args(["--model", "resnet50", "--heads", "make", "color",
                           "--nproc-per-node", "2", "--head-loss-weights", "make=0.5", "color=1"])
        self.assertEqual(args.arch, "resnet50")
        self.assertEqual(args.nproc_per_node, 2)
        self.assertEqual(args.heads, ["make", "color"])
        self.assertEqual(args.head_loss_weights, ["make=0.5", "color=1"])

    def test_test_decision_requires_test_csv_and_defaults_to_test(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(parse_args([]).decision_split, "test")
            normalized = user_input_node({**self.state, "decision_split": ""})
        self.assertEqual(normalized["decision_split"], "test")
        self._assert_invalid(dataset_validator_node(normalized), "Missing test_csv")
        self._assert_invalid(self._validate(decision_split="unknown"), "decision_split")

    def test_test_csv_cli_and_dataset_override_environment(self) -> None:
        self._write_split("test", self.rows)
        environment = {
            "TRAIN_CSV": str(self.root / "missing_train.csv"),
            "VAL_CSV": str(self.root / "missing_val.csv"),
            "TEST_CSV": str(self.root / "missing_test.csv"),
            "DATA_ROOT": str(self.root), "HEADS": "make color",
            "DECISION_SPLIT": "val",
        }
        with patch.dict(os.environ, environment, clear=True):
            normalized = user_input_node(self._cli_state([
                "--dataset", str(self.root), "--decision-split", "test"]))
        self.assertEqual(normalized["test_csv"], str(self.root / "test.csv"))
        self.assertEqual(normalized["decision_split"], "test")
        self.assertTrue(dataset_validator_node(normalized)["is_valid"])
        with patch.dict(os.environ, environment, clear=True):
            normalized = user_input_node(self._cli_state([
                "--train-csv", str(self.root / "train.csv"),
                "--val-csv", str(self.root / "val.csv"),
                "--test-csv", str(self.root / "test.csv"),
                "--decision-split", "test"]))
        self.assertEqual(normalized["test_csv"], str(self.root / "test.csv"))
        self.assertTrue(dataset_validator_node(normalized)["is_valid"])

    def test_test_csv_environment_and_dataset_fallback(self) -> None:
        self._write_split("test", self.rows)
        self._write_split("alternate_test", self.rows)
        for overrides in ({}, {"TEST_CSV": str(self.root / "alternate_test.csv")}):
            with self.subTest(overrides=overrides):
                environment = {
                    "DATASET_PATH": str(self.root), "DATA_ROOT": str(self.root),
                    "HEADS": "make color", "DECISION_SPLIT": "test", **overrides,
                }
                with patch.dict(os.environ, environment, clear=True):
                    raw = self._cli_state([])
                    raw["decision_split"] = ""
                    normalized = user_input_node(raw)
                self.assertEqual(normalized["test_csv"],
                                 overrides.get("TEST_CSV", str(self.root / "test.csv")))
                result = dataset_validator_node(normalized)
                self.assertTrue(result["is_valid"], result["validation_report"]["errors"])
                self.assertEqual(result["validation_report"]["split_stats"]["test"], {"images": 1})

    def test_test_labels_and_image_paths_are_validated(self) -> None:
        self._write_split("test", [{**self.rows[0], "make": "Honda"}])
        test_inputs = {"decision_split": "test", "test_csv": str(self.root / "test.csv")}
        self._assert_invalid(self._validate(**test_inputs), "test: head 'make' contains classes absent from training")
        self._write_split("test", [{**self.rows[0], "image_name": "missing.png"}])
        self._assert_invalid(self._validate(**test_inputs), "test: 1 image files do not exist")
        self._write_split("test", [{**self.rows[0], "make": "   "}])
        self._assert_invalid(self._validate(**test_inputs), "test: head 'make' has missing labels")
        # Supplied test data is checked even when validation drives decisions.
        self._assert_invalid(self._validate(**{**test_inputs, "decision_split": "val"}), "missing labels")

    def test_real_image_path_type_schema_validates_without_rewriting_csvs(self) -> None:
        fields = ["image_path", "make", "filename", "model_id", "model_name", "type"]
        row = {"image_path": "images/car.png", "make": "Toyota ", "filename": "car.png",
               "model_id": "1", "model_name": "Corolla", "type": " sedan "}
        for split in ("train", "val", "test"):
            self._write_split(split, [row], fields)
        original = (self.root / "train.csv").read_text()
        normalized = user_input_node({**self.state, "heads": ["make", "type"],
                                      "target_metric": "type_acc", "decision_split": "test",
                                      "test_csv": str(self.root / "test.csv")})
        self.assertEqual(normalized["heads"], ["make", "vehicle_type"])
        self.assertEqual(normalized["target_metric"], "vehicle_type_acc")
        result = dataset_validator_node(normalized)
        self.assertTrue(result["is_valid"], result["validation_report"]["errors"])
        self.assertEqual(result["validation_report"]["classes_by_head"],
                         {"make": ["Toyota"], "vehicle_type": ["sedan"]})
        self.assertEqual((self.root / "train.csv").read_text(), original)
        frame = read_dataset_csv(self.root / "train.csv", normalized["heads"])
        self.assertEqual(frame.loc[0, "root_folder"], "images")
        self.assertEqual(frame.loc[0, "image_name"], "car.png")

    def test_csv_adapter_rejects_conflicting_path_or_type_columns(self) -> None:
        row = {**self.rows[0], "image_path": "images/other.png"}
        self._write_split("train", [row], list(row))
        self._assert_invalid(self._validate(), "image_name conflicts with image_path")
        row = {**self.rows[0], "vehicle_type": "SUV", "type": "sedan"}
        self._write_split("train", [row], list(row))
        self._assert_invalid(self._validate(heads=["make", "vehicle_type"]), "vehicle_type conflicts with type")

    def test_decision_split_cli_overrides_environment_and_environment_overrides_yaml(self) -> None:
        with patch.dict(os.environ, {"DECISION_SPLIT": "val"}, clear=True):
            self.assertEqual(parse_args([]).decision_split, "val")
            self.assertEqual(parse_args(["--decision-split", "test"]).decision_split, "test")

    def test_validate_only_cli_runs_without_torch_or_agent_dependencies(self) -> None:
        # Fail explicitly if validate-only accidentally imports model/LLM modules.
        bootstrap = (
            "import builtins, runpy; "
            "original_import = builtins.__import__; "
            "blocked = {'torch', 'torchvision', 'langchain', 'langchain_openai', 'langgraph'}; "
            "exec('def guarded_import(name, *args, **kwargs):\\n"
            "    if name.split(\".\")[0] in blocked: raise AssertionError(\"Unexpected import: \" + name)\\n"
            "    return original_import(name, *args, **kwargs)'); "
            "builtins.__import__ = guarded_import; runpy.run_path('main.py', run_name='__main__')"
        )
        environment = os.environ.copy()
        environment.pop("OPENAI_API_KEY", None)
        command = [sys.executable, "-c", bootstrap, "--validate-only", "--dataset", str(self.root),
                   "--data-root", str(self.root), "--heads", "make", "color", "--weights", str(self.weights),
                   "--target-value", "0", "--export", "none", "--decision-split", "val"]
        result = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Dataset validation passed", result.stdout)
        result = subprocess.run([*command, "--target-value", "nan"], cwd=PROJECT_ROOT, env=environment,
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("target_value", result.stdout)


if __name__ == "__main__":
    unittest.main()
