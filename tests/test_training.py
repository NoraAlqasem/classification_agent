from __future__ import annotations

import io
import signal
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    import torch
    from agent.nodes import trainer
    from training import train_multihead_ddp as supplied
except ImportError as exc:
    IMPORT_ERROR = str(exc)
else:
    IMPORT_ERROR = ""


def training_config(directory: str) -> dict:
    return {
        "arch": "resnet18", "weights": "/data/weights/resnet18.pth",
        "train_csv": "/data/train.csv", "val_csv": "/data/val.csv",
        "data_root": "/data/images", "heads": ["make", "color"],
        "epochs": 2, "batch_size": 4, "num_workers": 0,
        "img_size": [32, 40], "lr": 0.001, "weight_decay": 0.0001,
        "gamma": 2.0, "patience": 2, "monitor": "mean",
        "head_loss_weights": {"make": 1.0, "color": 0.5},
        "alpha_json": None, "nproc_per_node": 1,
        "out": directory, "name": "exp_01",
    }


@unittest.skipIf(bool(IMPORT_ERROR), f"Training dependencies unavailable: {IMPORT_ERROR}")
class TrainerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="resnet-trainer-test-")
        self.addCleanup(self.temporary.cleanup)
        self.config = training_config(self.temporary.name)
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch.object(trainer, "console"))
        self.patches.enter_context(patch("torch.cuda.is_available", return_value=True))
        self.patches.enter_context(patch("torch.cuda.device_count", return_value=2))
        self.patches.enter_context(patch("torch.distributed.is_nccl_available", return_value=True))

    def process(self, exit_code=0, output="training output\n"):
        process = MagicMock()
        process.stdout = io.StringIO(output)
        process.wait.return_value = exit_code
        process.poll.return_value = exit_code
        return process

    def test_command_keeps_paths_as_arguments_and_maps_ddp_configuration(self):
        self.config.update({
            "weights": "/data/model weights $(echo unsafe).pth",
            "train_csv": "/data/train set.csv", "alpha_json": "/data/alpha values.json",
            "nproc_per_node": 2,
        })
        run_dir = Path(self.temporary.name) / "run with spaces"
        command = trainer.build_training_command(self.config, run_dir)
        self.assertEqual(command[1:4], ["-m", "torch.distributed.run", "--standalone"])
        for flag, expected in (
            ("--nproc_per_node", "2"), ("--weights", self.config["weights"]),
            ("--train-csv", self.config["train_csv"]), ("--run-dir", str(run_dir)),
            ("--batch-size", "4"), ("--alpha-json", self.config["alpha_json"]),
        ):
            self.assertEqual(command[command.index(flag) + 1], expected)
        index = command.index("--img-size")
        self.assertEqual(command[index + 1:index + 3], ["32", "40"])
        index = command.index("--heads")
        self.assertEqual(command[index + 1:index + 3], ["make", "color"])
        self.assertEqual(command[-3:], ["--head-loss-weights", "make=1.0", "color=0.5"])
        self.assertNotIn("--resume", command)

    def test_success_records_checkpoints_and_streams_log(self):
        process = self.process(output="[Train] epoch 1\n[Val] epoch 1\n")

        def launched(command, **kwargs):
            run_dir = Path(command[command.index("--run-dir") + 1])
            (run_dir / "weights" / "best_model.pth").write_bytes(b"best checkpoint")
            (run_dir / "weights" / "last_model.pth").write_bytes(b"last checkpoint")
            self.assertEqual(kwargs["cwd"], trainer.PROJECT_ROOT)
            self.assertTrue(kwargs["start_new_session"])
            self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
            self.assertNotIn("shell", kwargs)
            return process

        with patch.object(trainer.subprocess, "Popen", side_effect=launched) as launch:
            result = trainer.trainer_node({"current_config": self.config, "attempt_number": 2})
        self.assertEqual(result["attempt_number"], 3)
        record = result["training_result"]
        self.assertEqual(record["status"], "success")
        run_dir = Path(result["run_dir"])
        self.assertEqual(record["best_weights"], str(run_dir / "weights" / "best_model.pth"))
        self.assertEqual(record["last_weights"], str(run_dir / "weights" / "last_model.pth"))
        self.assertEqual((run_dir / "train_evaluation" / "training.log").read_text(), "[Train] epoch 1\n[Val] epoch 1\n")
        self.assertFalse((run_dir / "training.log").exists())
        self.assertEqual(record["config"]["arch"], "resnet18")
        self.assertGreaterEqual(record["elapsed_seconds"], 0)
        launch.assert_called_once()

    def test_failed_process_records_exit_code_and_log_tail(self):
        with patch.object(trainer.subprocess, "Popen", return_value=self.process(7, "CUDA out of memory\n")):
            result = trainer.trainer_node({"current_config": self.config})
        record = result["training_result"]
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["exit_code"], 7)
        self.assertIn("code 7", record["error"])
        self.assertEqual(record["log_tail"], ["CUDA out of memory"])
        self.assertNotIn("best_weights", record)
        self.assertTrue(Path(result["run_dir"], "train_evaluation", "training.log").is_file())

    def test_successful_exit_without_checkpoints_is_an_error(self):
        with patch.object(trainer.subprocess, "Popen", return_value=self.process()):
            result = trainer.trainer_node({"current_config": self.config})
        self.assertEqual(result["training_result"]["status"], "error")
        self.assertIn("without best_model.pth", result["training_result"]["error"])

    def test_existing_attempt_directory_is_preserved_without_launch(self):
        run_dir = Path(self.temporary.name) / "exp_01"
        run_dir.mkdir()
        checkpoint = run_dir / "best_model.pth"
        checkpoint.write_bytes(b"existing weights")
        with patch.object(trainer.subprocess, "Popen") as launch:
            result = trainer.trainer_node({"current_config": self.config, "run_dir": "/old/run"})
        launch.assert_not_called()
        self.assertEqual(result["training_result"]["status"], "error")
        self.assertEqual(checkpoint.read_bytes(), b"existing weights")
        self.assertEqual(result["run_dir"], "")
        self.assertFalse((run_dir / "training.log").exists())

    def test_no_cuda_returns_clear_error_without_launch_or_directory(self):
        with patch("torch.cuda.is_available", return_value=False), patch.object(trainer.subprocess, "Popen") as launch:
            result = trainer.trainer_node({"current_config": self.config})
        launch.assert_not_called()
        self.assertEqual(result["training_result"]["status"], "error")
        self.assertIn("CUDA is required", result["training_result"]["error"])
        self.assertEqual(result["run_dir"], "")
        self.assertFalse((Path(self.temporary.name) / "exp_01").exists())

    def test_requested_gpu_count_exceeding_visible_devices_does_not_launch(self):
        self.config["nproc_per_node"] = 3
        with patch.object(trainer.subprocess, "Popen") as launch:
            result = trainer.trainer_node({"current_config": self.config})
        launch.assert_not_called()
        self.assertIn("only 2 CUDA devices", result["training_result"]["error"])

    def test_missing_nccl_returns_error_without_launch(self):
        with patch("torch.distributed.is_nccl_available", return_value=False), patch.object(trainer.subprocess, "Popen") as launch:
            result = trainer.trainer_node({"current_config": self.config})
        launch.assert_not_called()
        self.assertIn("does not provide NCCL", result["training_result"]["error"])

    def test_keyboard_interrupt_terminates_workers_and_propagates(self):
        process = self.process()

        def interrupted_lines():
            yield "started\n"
            raise KeyboardInterrupt

        process.stdout = MagicMock()
        process.stdout.__iter__.side_effect = interrupted_lines
        with patch.object(trainer.subprocess, "Popen", return_value=process), patch.object(trainer, "_terminate_workers") as terminate:
            with self.assertRaises(KeyboardInterrupt):
                trainer.trainer_node({"current_config": self.config})
        terminate.assert_called_once_with(process)

    def test_worker_termination_escalates_when_process_ignores_sigterm(self):
        process = self.process()
        process.pid = 1234
        process.wait.side_effect = [subprocess.TimeoutExpired("torchrun", 10), 0]
        with patch.object(trainer.os, "killpg") as kill:
            trainer._terminate_workers(process)
        self.assertEqual([call.args for call in kill.call_args_list], [(1234, signal.SIGTERM), (1234, signal.SIGKILL)])


@unittest.skipIf(bool(IMPORT_ERROR), f"Training dependencies unavailable: {IMPORT_ERROR}")
class DistributedValidationTests(unittest.TestCase):
    def test_validation_synchronizes_buffers_once_and_avoids_ddp_forward(self):
        class Classifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("running_stat", torch.zeros(1))

            def forward(self, images):
                return {"make": images}

        class DDPWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.module = model

            def forward(self, images):
                raise AssertionError("Uneven validation must bypass DDP's forward collectives")

        wrapper = DDPWrapper(Classifier())
        loader = [
            (torch.tensor([[3.0, 0.0], [0.0, 3.0]]), {"make": torch.tensor([0, 1])}),
            (torch.tensor([[3.0, 0.0]]), {"make": torch.tensor([0])}),
        ]
        with patch.object(supplied, "DDP", DDPWrapper), patch.object(supplied.dist, "broadcast") as broadcast, patch.object(supplied.dist, "all_reduce"):
            metrics = supplied.evaluate(wrapper, loader, {"make": supplied.FocalLoss()}, {"make": 1.0}, torch.device("cpu"), 1, ["make"], 1)
        broadcast.assert_called_once()
        self.assertEqual(metrics["sample_count"], 3)
        self.assertEqual(metrics["mean_acc"], 1.0)

    def test_no_padding_sampler_supports_ranks_with_zero_validation_images(self):
        dataset = ["image one", "image two"]
        shards = [list(supplied.DistributedEvalSampler(dataset, 4, rank)) for rank in range(4)]
        self.assertEqual(shards, [[0], [1], [], []])
        self.assertEqual(sorted(index for shard in shards for index in shard), [0, 1])

    def test_empty_local_validation_shard_still_participates_in_global_metrics(self):
        model = MagicMock()

        def other_ranks_metrics(tensor, **kwargs):
            tensor.copy_(torch.tensor([1.5, 3.0, 1.5, 2.0], dtype=tensor.dtype))

        with patch.object(supplied.dist, "all_reduce", side_effect=other_ranks_metrics) as reduce:
            metrics = supplied.evaluate(model, [], {}, {"make": 1.0}, torch.device("cpu"), 1, ["make"], 1)
        model.assert_not_called()
        reduce.assert_called_once()
        self.assertEqual(metrics["sample_count"], 3)
        self.assertAlmostEqual(metrics["mean_acc"], 2 / 3)
        self.assertEqual(metrics["total_loss"], 0.5)


if __name__ == "__main__":
    unittest.main()
