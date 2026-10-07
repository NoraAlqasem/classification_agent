from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from training.artifact_paths import (
    checkpoint_run_dir, checkpoint_training_config, find_training_file, find_weights_file,
    prepare_new_run_directory, train_evaluation_dir, weights_dir,
)

try:
    import torch
    from training import train_multihead_ddp as supplied
except ImportError as exc:
    IMPORT_ERROR = str(exc)
else:
    IMPORT_ERROR = ""


class ArtifactPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="resnet-layout-")
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / "session_1" / "exp_01"

    def test_new_paths_are_expected_when_files_are_missing(self):
        self.assertEqual(find_weights_file(self.run, "best_model.pth"), self.run / "weights" / "best_model.pth")
        self.assertEqual(find_training_file(self.run, "training_metrics.csv"), self.run / "train_evaluation" / "training_metrics.csv")
        self.assertFalse(self.run.exists())

    def test_readers_prefer_native_files_and_fall_back_to_legacy(self):
        prepare_new_run_directory(self.run)
        for name, finder, directory in (("best_model.pth", find_weights_file, weights_dir(self.run)),
                                        ("training_config.json", find_training_file, train_evaluation_dir(self.run))):
            legacy = self.run / name
            legacy.write_text("legacy")
            self.assertEqual(finder(self.run, name), legacy)
            current = directory / name
            current.write_text("current")
            self.assertEqual(finder(self.run, name), current)

    def test_checkpoint_paths_locate_the_attempt_root_and_matching_config(self):
        prepare_new_run_directory(self.run)
        current = self.run / "weights" / "last_model.pth"
        config = self.run / "train_evaluation" / "training_config.json"
        config.write_text("{}")
        self.assertEqual(checkpoint_run_dir(current), self.run)
        self.assertEqual(checkpoint_training_config(current), config)
        config.unlink()
        legacy = self.run / "training_config.json"
        legacy.write_text("{}")
        self.assertEqual(checkpoint_training_config(self.run / "last_model.pth"), legacy)
        self.assertEqual(checkpoint_training_config(current), legacy)

    def test_guard_accepts_only_empty_weights_and_the_streaming_training_log(self):
        prepare_new_run_directory(self.run)
        log = train_evaluation_dir(self.run) / "training.log"
        log.write_text("Streaming already started.\n")
        prepare_new_run_directory(self.run, allow_existing=True)
        self.assertEqual(log.read_text(), "Streaming already started.\n")
        with self.assertRaises(FileExistsError):
            prepare_new_run_directory(self.run)

    def test_guard_rejects_previous_artifacts_without_modifying_them(self):
        for relative in ("best_model.pth", "training.log", "weights/best_model.pth",
                         "weights/classes_by_head.json", "train_evaluation/training_metrics.csv",
                         "test_evaluation/summary.json", "exp_1_session_1.json"):
            with self.subTest(artifact=relative):
                run = self.run / relative.replace("/", "_")
                artifact = run / relative
                artifact.parent.mkdir(parents=True)
                artifact.write_text("Existing artifact.")
                with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                    prepare_new_run_directory(run, allow_existing=True)
                self.assertEqual(artifact.read_text(), "Existing artifact.")


@unittest.skipIf(bool(IMPORT_ERROR), f"Training dependencies unavailable: {IMPORT_ERROR}")
class TrainingCliLayoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="resnet-cli-layout-")
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / "session_1" / "exp_01"

    def run_cli(self, *, resume=None, interrupt=False):
        # Exercise the actual CLI and native disk writes while replacing data,
        # model training, NCCL and every GPU operation with CPU fixtures.
        class CpuTorch:
            def __getattr__(self, name):
                return getattr(torch, name)

            def device(self, name):
                return torch.device("cpu")

        metric = {"total_loss": .25, "mean_acc": .8, "sample_count": 2,
                  "heads": {"make": {"loss": .25, "acc": .8}}}

        def train_epoch(**kwargs):
            if interrupt and kwargs["epoch"] > 1:
                raise KeyboardInterrupt("Fixture interrupted after one complete epoch")
            performance = {"epoch": kwargs["epoch"], "iterations_per_gpu": 1,
                           "epoch_time_sec": .01, "avg_iter_time_sec": .01,
                           "max_peak_mem_allocated_gb": 0., "max_peak_mem_reserved_gb": 0.}
            return metric, performance

        arguments = ["train_multihead_ddp.py", "--train-csv", "/fixture/train.csv",
                     "--val-csv", "/fixture/val.csv", "--heads", "make", "--epochs", "2",
                     "--num-workers", "0", "--img-size", "32", "40"]
        if resume is None:
            arguments += ["--run-dir", str(self.run), "--weights", "/fixture/resnet18.pth"]
        else:
            arguments += ["--resume", str(resume)]
        optimizer, scheduler = MagicMock(), MagicMock()
        optimizer.state_dict.return_value = {}
        scheduler.state_dict.return_value = {}
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(patch("sys.argv", arguments))
            stack.enter_context(patch.dict(os.environ, {"LOCAL_RANK": "0"}))
            stack.enter_context(patch.object(supplied, "torch", CpuTorch()))
            stack.enter_context(patch("torch.cuda.is_available", return_value=True))
            stack.enter_context(patch("torch.cuda.set_device"))
            for name, value in (("get_rank", 0), ("get_world_size", 1), ("is_initialized", True)):
                stack.enter_context(patch.object(supplied.dist, name, return_value=value))
            for name in ("init_process_group", "destroy_process_group", "barrier", "broadcast"):
                stack.enter_context(patch.object(supplied.dist, name))
            stack.enter_context(patch.object(supplied, "build_class_mappings", return_value=(
                {"make": ["Ford", "Toyota"]}, {"make": {"Ford": 0, "Toyota": 1}}, {"make": 2})))
            stack.enter_context(patch.object(supplied, "get_dataloaders", return_value=([], [])))
            stack.enter_context(patch.object(supplied, "MultiHeadResNet", return_value=torch.nn.Linear(1, 1)))
            stack.enter_context(patch.object(supplied, "DDP", side_effect=lambda model, **kwargs: SimpleNamespace(
                module=model, parameters=model.parameters)))
            stack.enter_context(patch.object(supplied.optim, "AdamW", return_value=optimizer))
            stack.enter_context(patch.object(supplied.optim.lr_scheduler, "CosineAnnealingLR", return_value=scheduler))
            stack.enter_context(patch.object(supplied, "train_one_epoch", side_effect=train_epoch))
            stack.enter_context(patch.object(supplied, "evaluate", return_value=metric))
            supplied.main()

    def test_cli_writes_artifacts_natively_inside_the_requested_folders(self):
        prepare_new_run_directory(self.run)
        log = train_evaluation_dir(self.run) / "training.log"
        log.write_text("Agent-opened streaming log.\n")
        self.run_cli()
        self.assertEqual(sorted(path.name for path in self.run.iterdir()), ["train_evaluation", "weights"])
        self.assertEqual(sorted(path.name for path in weights_dir(self.run).iterdir()),
                         ["best_model.pth", "class_to_idx_by_head.json", "classes_by_head.json", "last_model.pth"])
        self.assertEqual(sorted(path.name for path in train_evaluation_dir(self.run).iterdir()),
                         ["performance_metrics.csv", "training.log", "training_config.json", "training_metrics.csv", "training_plots.png"])
        config = json.loads(checkpoint_training_config(weights_dir(self.run) / "last_model.pth").read_text())
        self.assertEqual(config["epochs"], 2)
        self.assertEqual(log.read_text(), "Agent-opened streaming log.\n")
        self.assertEqual(len((train_evaluation_dir(self.run) / "training_metrics.csv").read_text().splitlines()), 5)

    def test_interrupted_native_run_resumes_in_the_same_attempt_root(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_cli(interrupt=True)
        last = weights_dir(self.run) / "last_model.pth"
        self.assertEqual(torch.load(last, map_location="cpu", weights_only=False)["epoch"], 1)
        self.run_cli(resume=last)
        self.assertEqual(torch.load(last, map_location="cpu", weights_only=False)["epoch"], 2)
        self.assertEqual(len((train_evaluation_dir(self.run) / "training_metrics.csv").read_text().splitlines()), 5)
        self.assertFalse((weights_dir(self.run) / "weights").exists())
        self.assertFalse((self.run / "last_model.pth").exists())

    def test_legacy_root_checkpoint_resume_preserves_the_historical_layout(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_cli(interrupt=True)
        for directory in (weights_dir(self.run), train_evaluation_dir(self.run)):
            for artifact in directory.iterdir():
                artifact.rename(self.run / artifact.name)
            directory.rmdir()
        last = self.run / "last_model.pth"
        self.run_cli(resume=last)
        self.assertEqual(torch.load(last, map_location="cpu", weights_only=False)["epoch"], 2)
        self.assertTrue((self.run / "training_plots.png").is_file())
        self.assertEqual(len((self.run / "training_metrics.csv").read_text().splitlines()), 5)
        self.assertFalse(weights_dir(self.run).exists())
        self.assertFalse(train_evaluation_dir(self.run).exists())


if __name__ == "__main__":
    unittest.main()
