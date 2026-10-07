"""Preserve numbered experiment files and load historical records together."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from memory.experiment_store import ExperimentStore


class NumberedExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = ExperimentStore(self.root / "runs" / "resnet")
        self.assertFalse(self.store.exp_dir.exists())

    def experiment(self, session, attempt=1, score=0.8, arch="resnet18"):
        session_dir = self.store.base_dir / session
        run_dir = session_dir / f"exp_{attempt:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        best = run_dir / "best_model.pth"
        last = run_dir / "last_model.pth"
        best.write_bytes(b"Synthetic checkpoint; inference is not run")
        last.write_bytes(b"Synthetic latest checkpoint")
        config = {
            "arch": arch, "weights": str(self.root / f"{arch}.pth"),
            "train_csv": str(self.root / "train.csv"), "val_csv": str(self.root / "val.csv"),
            "data_root": str(self.root / "images"), "heads": ["make", "vehicle_type"],
            "epochs": 2, "batch_size": 8, "num_workers": 0, "img_size": [64, 64],
            "lr": 0.0001, "weight_decay": 0.0001, "gamma": 1.0, "patience": 10,
            "monitor": "mean", "head_loss_weights": {"make": 1.0, "vehicle_type": 1.0},
            "alpha_json": None, "nproc_per_node": 1, "out": str(session_dir), "name": run_dir.name,
        }
        return {
            "attempt": attempt, "timestamp": "2026-10-06T12:00:00+00:00", "training_status": "success",
            "decision_split": "val", "target_metric": "mean_acc", "target_value": 0.99,
            "config": config, "train_csv": config["train_csv"], "val_csv": config["val_csv"],
            "test_csv": str(self.root / "test.csv"), "heads": config["heads"],
            "metrics": {"status": "ok", "decision_split": "val", "mean_acc": score,
                        "make_acc": score, "vehicle_type_acc": score,
                        "total_epochs": 2, "best_epoch": 1, "final_epoch": 2,
                        "train_loss": 0.4, "val_loss": 0.5,
                        "heads": {head: {"acc": score, "loss": 0.5} for head in config["heads"]}},
            "evaluation": {"verdict": "underfitting", "target_reached": False, "reasoning": "Fixture result"},
            "score": score, "improved": True, "run_dir": str(run_dir),
            "best_weights": str(best), "last_weights": str(last), "target_reached": False,
        }

    def test_numbered_filenames_distinguish_attempts_and_sessions_and_group_details(self):
        records = [self.experiment("session_1", 1, 0.75),
                   self.experiment("session_1", 2, 0.85, "resnet34"),
                   self.experiment("session_2", 1, 0.8, "resnet50")]
        expected_names = ["exp_1.json", "exp_2.json", "exp_1.json"]
        for record, expected_name, number in zip(records, expected_names, (1, 1, 2)):
            path = self.store.save_experiment(record)
            self.assertEqual(path.name, expected_name)
            self.assertEqual(path.parent, Path(record["run_dir"]))
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(set(saved), {"experiment", "objective", "config", "data", "training", "validation", "test", "evaluation", "artifacts"})
            for section in ("experiment", "objective", "config", "data", "training", "validation", "test", "evaluation", "artifacts"):
                self.assertIsInstance(saved[section], dict)
            self.assertNotIn("metrics", saved)
            self.assertNotIn("attempt", saved)
            self.assertEqual(saved["experiment"], {"id": f"exp_{record['attempt']:02d}", "date": "2026-10-06", "status": "success"})
            self.assertEqual(saved["config"]["architecture"], record["config"]["arch"])
            self.assertEqual(saved["config"]["optimizer"]["lr"], 0.0001)
            self.assertEqual(saved["validation"]["mean_accuracy"], record["metrics"]["mean_acc"])
            self.assertEqual(saved["artifacts"]["best_weights"], record["best_weights"])
        self.assertEqual(len(self.store.load_all_experiments()), 3)
        self.assertFalse(self.store.exp_dir.exists())
        kb = json.loads(self.store.kb_path.read_text(encoding="utf-8"))
        self.assertEqual([Path(entry["file"]).name for entry in kb["experiments"]], expected_names)
        self.assertEqual([Path(entry["file"]).parent for entry in kb["experiments"]],
                         [Path(record["run_dir"]) for record in records])
        self.assertEqual([entry["record"]["config"] for entry in kb["experiments"]], [record["config"] for record in records])
        self.assertTrue(all(len(entry["sha256"]) == 64 for entry in kb["experiments"]))

    def test_duplicate_numbered_attempt_preserves_original_file_and_knowledge_base(self):
        original = self.experiment("session_1")
        path = self.store.save_experiment(original)
        saved_bytes = path.read_bytes()
        kb_bytes = self.store.kb_path.read_bytes()
        duplicate = copy.deepcopy(original)
        duplicate["metrics"]["mean_acc"] = 0.99
        duplicate["evaluation"]["reasoning"] = "Must not replace the original result"
        with self.assertRaises(FileExistsError):
            self.store.save_experiment(duplicate)
        self.assertEqual(path.read_bytes(), saved_bytes)
        self.assertEqual(self.store.kb_path.read_bytes(), kb_bytes)
        self.assertEqual(len(self.store.load_all_experiments()), 1)
        self.assertEqual(list(path.parent.glob("*.json")), [path])
        self.assertEqual(list(path.parent.glob(".experiment-*.tmp")), [])
        self.assertFalse(self.store.exp_dir.exists())

    def test_mixed_flat_and_grouped_records_load_and_select_best_without_losing_identity(self):
        grouped = self.experiment("session_1", score=0.8, arch="resnet34")
        self.store.save_experiment(grouped)
        legacy = self.experiment("session_20261006_120000_123456_abcdef01", score=0.9, arch="resnet18")
        self.store.exp_dir.mkdir()
        legacy_path = self.store.exp_dir / "exp_01_20261006_120000_123456.json"
        legacy_path.write_text(json.dumps(legacy, indent=2), encoding="utf-8")
        kb = json.loads(self.store.kb_path.read_text(encoding="utf-8"))
        kb["experiments"].append({"file": str(legacy_path), "attempt": 1})
        self.store.kb_path.write_text(json.dumps(kb, indent=2), encoding="utf-8")
        loaded = self.store.load_all_experiments()
        self.assertEqual(len(loaded), 2)
        self.assertEqual([record["config"]["arch"] for record in loaded], ["resnet34", "resnet18"])
        self.assertEqual([record["metrics"]["mean_acc"] for record in loaded], [0.8, 0.9])
        best = self.store.get_best_experiment("mean_acc", train_csv=legacy["train_csv"],
            val_csv=legacy["val_csv"], test_csv=legacy["test_csv"], heads=legacy["heads"], decision_split="val")
        self.assertEqual(best["config"]["arch"], "resnet18")
        self.assertEqual(best["best_weights"], legacy["best_weights"])
        self.assertTrue(Path(best["best_weights"]).is_file())
        for key in ("target_metric", "train_csv", "val_csv", "test_csv"):
            self.assertEqual(best[key], legacy[key])
        selected_grouped = self.store.get_best_experiment("mean_acc", arch="resnet34", decision_split="val")
        self.assertEqual(selected_grouped["best_weights"], grouped["best_weights"])
        self.assertEqual(selected_grouped["config"]["weights"], grouped["config"]["weights"])
        self.assertEqual(selected_grouped["config"]["out"], grouped["config"]["out"])
        self.assertEqual(selected_grouped["session_id"], 1)

    def test_explicit_training_run_directory_takes_priority_over_config_name(self):
        record = self.experiment("session_1")
        record["config"]["name"] = "planned_attempt"
        path = self.store.save_experiment(record)
        self.assertEqual(path, Path(record["run_dir"]) / "exp_1.json")
        self.assertFalse((Path(record["config"]["out"]) / "planned_attempt").exists())
        self.assertFalse(self.store.exp_dir.exists())

    def test_config_output_and_name_supply_missing_explicit_run_directory(self):
        record = self.experiment("session_2", attempt=3)
        record.pop("run_dir")
        path = self.store.save_experiment(record)
        self.assertEqual(path, Path(record["config"]["out"]) / "exp_03" / "exp_3.json")
        self.assertFalse(self.store.exp_dir.exists())
        kb = json.loads(self.store.kb_path.read_text(encoding="utf-8"))
        self.assertEqual(kb["experiments"][0]["file"], str(path))
        self.assertEqual(next(iter(kb["best_configs"].values()))["file"], str(path))

    def test_legacy_records_without_attempt_destination_keep_central_fallback(self):
        record = self.experiment("custom-session")
        record.pop("run_dir")
        record["config"].pop("out")
        record["config"].pop("name")
        path = self.store.save_experiment(record)
        self.assertEqual(path.parent, self.store.exp_dir)
        self.assertTrue(path.name.startswith("exp_01_"))
        self.assertEqual(len(self.store.load_all_experiments()), 1)

    def test_explicit_destination_cannot_escape_requested_session(self):
        record = self.experiment("session_1")
        record["run_dir"] = str(self.store.base_dir / "session_2" / "exp_01")
        before = self.store.kb_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "requested session"):
            self.store.save_experiment(record)
        self.assertEqual(self.store.kb_path.read_bytes(), before)
        self.assertFalse(Path(record["run_dir"]).exists())
        self.assertFalse(self.store.exp_dir.exists())


if __name__ == "__main__":
    unittest.main()
