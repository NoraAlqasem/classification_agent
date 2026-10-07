"""Reserve distinct numbered training sessions without touching existing data."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from agent.nodes import config_generator
from memory.session_paths import allocate_session_directory, session_number


class SessionDirectoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "runs" / "resnet"

    def test_first_and_next_calls_reserve_existing_directories(self):
        first = allocate_session_directory(self.base)
        second = allocate_session_directory(self.base)
        self.assertIsInstance(first, Path)
        self.assertEqual(first, self.base / "session_1")
        self.assertEqual(second, self.base / "session_2")
        self.assertTrue(first.is_dir())
        self.assertTrue(second.is_dir())
        self.assertEqual(session_number(first), 1)
        self.assertEqual(session_number(second), 2)

    def test_gaps_choose_maximum_plus_one_and_ignore_legacy_and_noncanonical_names(self):
        self.base.mkdir(parents=True)
        for name in ("session_1", "session_4", "session_20261006_152950_487918_abcdef01",
                     "session_00099", "session_0", "notes", "session_2_backup"):
            (self.base / name).mkdir()
        for name in ("README.txt", "session_100.json", "session_latest"):
            (self.base / name).write_text("Existing file must survive", encoding="utf-8")
        reserved = allocate_session_directory(self.base)
        self.assertEqual(reserved, self.base / "session_5")
        self.assertFalse((self.base / "session_2").exists())
        self.assertTrue(reserved.is_dir())
        for name in ("README.txt", "session_100.json", "session_latest"):
            self.assertEqual((self.base / name).read_text(encoding="utf-8"), "Existing file must survive")

    def test_numbered_file_is_reserved_and_existing_session_contents_are_preserved(self):
        self.base.mkdir(parents=True)
        previous = self.base / "session_4"
        previous.mkdir()
        sentinel = previous / "best_model.pth"
        sentinel.write_bytes(b"Existing checkpoint")
        occupied = self.base / "session_5"
        occupied.write_bytes(b"A file occupying a canonical session name")
        reserved = allocate_session_directory(self.base)
        self.assertEqual(reserved, self.base / "session_6")
        self.assertTrue(reserved.is_dir())
        self.assertTrue(occupied.is_file())
        self.assertEqual(occupied.read_bytes(), b"A file occupying a canonical session name")
        self.assertEqual(sentinel.read_bytes(), b"Existing checkpoint")

    def test_concurrent_calls_reserve_unique_sequential_sessions(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            directories = list(pool.map(lambda _: allocate_session_directory(self.base), range(12)))
        self.assertEqual(len(set(directories)), 12)
        self.assertEqual(sorted(session_number(directory) for directory in directories), list(range(1, 13)))
        self.assertTrue(all(directory.is_dir() and directory.parent == self.base for directory in directories))
        self.assertEqual({directory.name for directory in self.base.iterdir()},
                         {f"session_{number}" for number in range(1, 13)})

    def test_session_number_accepts_only_canonical_positive_session_names(self):
        for name, expected in (("session_1", 1), ("session_27", 27), ("session_1000", 1000)):
            for path in (name, self.root / name):
                with self.subTest(path=path):
                    self.assertEqual(session_number(path), expected)
        for name in ("session_0", "session_01", "session_0001", "session_-1", "session_1_backup",
                     "session_20261006_152950_487918_abcdef01", "session_", "exp_1", "mysession_1"):
            with self.subTest(name=name):
                self.assertIsNone(session_number(self.root / name))

    def test_file_at_base_path_is_preserved_when_allocation_fails(self):
        base_file = self.root / "occupied-run-root"
        base_file.write_bytes(b"Do not replace this file")
        with self.assertRaises(OSError):
            allocate_session_directory(base_file)
        self.assertEqual(base_file.read_bytes(), b"Do not replace this file")


class ConfigurationSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        config_dir = self.root / "config"
        config_dir.mkdir()
        self.defaults = {
            "arch": "resnet50", "epochs": 2, "batch_size": 8, "num_workers": 0,
            "img_size": [64, 64], "lr": 0.0001, "weight_decay": 0.0001,
            "gamma": 1.0, "patience": 10, "monitor": "mean", "out": "runs/resnet", "name": "experiment",
        }
        (config_dir / "defaults.yaml").write_text(json.dumps({"training": self.defaults}), encoding="utf-8")
        self.state = {
            "validation_report": {}, "model_type": "resnet18", "weights_path": str(self.root / "weights.pth"),
            "train_csv": str(self.root / "train.csv"), "val_csv": str(self.root / "val.csv"),
            "heads": ["make", "vehicle_type"], "target_metric": "mean_acc", "nproc_per_node": 1,
        }

    def generate(self, state):
        with patch.object(config_generator, "PROJECT_ROOT", self.root), \
             patch.object(config_generator, "console"):
            return config_generator.config_generator_node(state)

    def test_generator_allocates_numbered_session_before_first_yaml_attempt(self):
        original_state = copy.deepcopy(self.state)
        first = self.generate(self.state)
        second = self.generate(self.state)
        for result, number in ((first, 1), (second, 2)):
            expected = self.root / "runs" / "resnet" / f"session_{number}"
            self.assertEqual(result["session_dir"], str(expected))
            self.assertEqual(result["current_config"]["out"], str(expected))
            self.assertTrue(expected.is_dir())
            self.assertEqual(result["current_config"]["name"], "exp_01")
            self.assertEqual(result["config_history"][0]["config"], result["current_config"])
            self.assertEqual(result["messages"], [])
            for key in ("epochs", "batch_size", "num_workers", "img_size", "lr", "weight_decay", "gamma", "patience"):
                self.assertEqual(result["current_config"][key], self.defaults[key])
        self.assertEqual(self.state, original_state)

    def test_generator_preserves_an_explicit_existing_custom_session(self):
        custom = self.root / "custom training session"
        custom.mkdir()
        sentinel = custom / "notes.txt"
        sentinel.write_text("Existing session notes", encoding="utf-8")
        state = {**self.state, "session_dir": str(custom)}
        result = self.generate(state)
        self.assertEqual(result["session_dir"], str(custom))
        self.assertEqual(result["current_config"]["out"], str(custom))
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "Existing session notes")
        self.assertEqual(set(custom.iterdir()), {sentinel, custom / "session_summary.md"})
        self.assertFalse((self.root / "runs").exists())


if __name__ == "__main__":
    unittest.main()
