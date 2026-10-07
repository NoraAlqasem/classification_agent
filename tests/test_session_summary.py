"""Summaries use verified metrics and keep incomplete attempts visibly incomplete."""
from __future__ import annotations

import copy
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage

from agent.nodes import reporter
from agent.session_summary import render_session_summary, write_session_summary
from memory.experiment_format import serialize_internal_experiment


def experiment(attempt=1, make=0.7, vehicle_type=0.9, *, target_metric="mean_acc", heads=None):
    scores = {"make": make, "vehicle_type": vehicle_type} if heads is None else heads
    total = 100
    evidence = {
        "status": "ok", "decision_split": "test", "split": "test", "test_evaluated": True,
        "full_coverage": True, "total_images": total, "evaluated_images": total, "matched_images": total,
        "test_csv": "/tmp/test.csv", "checkpoint": f"/tmp/session_1/exp_{attempt:02d}/best_model.pth",
        "mean_acc": sum(scores.values()) / len(scores), "best_epoch": 1,
        "heads": {head: {"acc": score, "correct": round(score * total), "count": total, "coverage": 1.0}
                  for head, score in scores.items()},
        **{f"{head}_acc": score for head, score in scores.items()},
    }
    return {
        "attempt": attempt, "training_status": "success", "target_metric": target_metric,
        "target_value": 0.99, "decision_split": "test", "test_csv": "/tmp/test.csv", "heads": list(scores),
        "config": {"arch": "resnet18", "weights": "/tmp/resnet18.pth", "out": "/tmp/session_1",
                   "train_csv": "/tmp/train.csv", "val_csv": "/tmp/val.csv", "heads": list(scores),
                   "epochs": 2, "batch_size": 32, "lr": 0.0001, "weight_decay": 0.0001,
                   "gamma": 1.0, "head_loss_weights": {head: 1.0 for head in scores}, "img_size": [224, 224]},
        "metrics": evidence, "test_result": {"status": "ok", "csv": "/tmp/test.csv", "metrics": copy.deepcopy(evidence)},
        "evaluation": {"verdict": "underfitting"}, "best_weights": evidence["checkpoint"],
    }


class SessionSummaryTests(unittest.TestCase):
    def test_summary_selects_verified_overall_best_and_separate_head_bests(self):
        first = experiment(1, make=0.7, vehicle_type=0.9)
        second = experiment(2, make=0.85, vehicle_type=0.85)
        original = copy.deepcopy([first, second])
        summary = render_session_summary("/tmp/session_1", [first, second], status="Completed", target_value=0.99)
        self.assertIn("| Session | session_1 |", summary)
        self.assertIn("| Target Value | 99.00% |", summary)
        self.assertIn("| Total Experiments | 2 |", summary)
        self.assertIn("| Status | Completed |", summary)
        self.assertIn("| Target Reached | No |", summary)
        self.assertIn("**Best Experiment:** `exp_02`", summary)
        self.assertIn("**Best Mean Accuracy:** 85.00%", summary)
        self.assertIn("| Make | `exp_02` | 85.00% |", summary)
        self.assertIn("| Vehicle Type | `exp_01` | 90.00% |", summary)
        self.assertIn("| Mean | `exp_02` | 85.00% |", summary)
        self.assertIn("Make Test Accuracy | Type Test Accuracy", summary)
        self.assertIn("| Image Size | 224 × 224 |", summary)
        self.assertIn("| Metric | Value |", summary)
        self.assertIn("| Make Accuracy | 85.00% |", summary)
        self.assertIn("| Vehicle Type Accuracy | 85.00% |", summary)
        self.assertIn("| Mean Accuracy | 85.00% |", summary)
        self.assertIn("| Best Epoch | 1 |", summary)
        self.assertIn("| Make Loss Weight | 1.0 |", summary)
        self.assertIn("| Vehicle Type Loss Weight | 1.0 |", summary)
        self.assertIn("| Head | Best Experiment | Accuracy |", summary)
        self.assertEqual([first, second], original)

    def test_named_head_target_picks_that_head_instead_of_the_highest_mean(self):
        first = experiment(1, make=0.9, vehicle_type=0.5, target_metric="make_acc")
        second = experiment(2, make=0.8, vehicle_type=1.0, target_metric="make_acc")
        summary = render_session_summary("/tmp/session_1", [first, second], target_metric="make_acc", target_value=0.85)
        self.assertIn("| Target Metric | Make Accuracy |", summary)
        self.assertIn("**Best Experiment:** `exp_01`", summary)
        self.assertIn("**Target Make Accuracy:** 90.00%", summary)
        self.assertIn("**Best Mean Accuracy:** 90.00% (`exp_02`)", summary)
        self.assertIn("| Mean Accuracy | 70.00% |", summary)
        self.assertIn("| Target Reached | Yes |", summary)
        self.assertIn("| Mean | `exp_02` | 90.00% |", summary)
        self.assertIn("| Vehicle Type | `exp_02` | 100.00% |", summary)

    def test_incomplete_second_attempt_is_counted_without_inventing_test_results(self):
        completed = experiment()
        incomplete = experiment(2, make=1.0, vehicle_type=1.0)
        incomplete.update(training_status="interrupted", test_result={},
                          metrics={"status": "training_incomplete", "total_epochs": 2, "train_loss": 0.4},
                          evaluation={})
        incomplete["config"]["epochs"] = 30
        summary = render_session_summary("/tmp/session_1", [completed, incomplete], status="Interrupted", target_value=0.99)
        self.assertIn("| Total Experiments | 2 |", summary)
        self.assertIn("| Status | Interrupted |", summary)
        self.assertIn("**Best Experiment:** `exp_01`", summary)
        self.assertIn("**Best Mean Accuracy:** 80.00%", summary)
        row = next(line for line in summary.splitlines() if line.startswith("| `exp_02` |"))
        self.assertIn("| 30 |", row)
        self.assertIn("| — | — | — | — | interrupted |", row)
        self.assertNotIn("100.00%", summary)

    def test_partial_wrong_csv_failed_and_nonfinite_records_cannot_display_or_win(self):
        valid = experiment()
        invalid = []
        partial = experiment(2, 1.0, 1.0)
        partial["metrics"]["evaluated_images"] = 99
        invalid.append(partial)
        wrong_csv = experiment(3, 1.0, 1.0)
        wrong_csv["test_result"]["csv"] = "/tmp/other.csv"
        wrong_csv["test_result"]["metrics"]["test_csv"] = "/tmp/other.csv"
        invalid.append(wrong_csv)
        failed = experiment(4, 1.0, 1.0)
        failed.update(training_status="error", error="Training failed")
        invalid.append(failed)
        nonfinite = experiment(5, 1.0, 1.0)
        nonfinite["metrics"]["train_loss"] = float("inf")
        invalid.append(nonfinite)
        summary = render_session_summary("/tmp/session_1", [valid, *invalid], target_value=0.99)
        self.assertIn("**Best Experiment:** `exp_01`", summary)
        self.assertIn("**Best Mean Accuracy:** 80.00%", summary)
        self.assertNotIn("100.00%", summary)
        for attempt in range(2, 6):
            row = next(line for line in summary.splitlines() if line.startswith(f"| `exp_{attempt:02d}` |"))
            self.assertIn("| — | — | — |", row)
        self.assertIn("| Mean | `exp_01` | 80.00% |", summary)

    def test_dynamic_heads_and_nested_documents_preserve_measured_results(self):
        raw = experiment(heads={"color": 0.75, "body_style": 0.95})
        summary = render_session_summary("/tmp/session_1", [serialize_internal_experiment(raw)], target_value=0.8)
        self.assertIn("| Heads | `color`, `body_style` |", summary)
        self.assertIn("Color Weight | Body Style Weight", summary)
        self.assertIn("| Color | `exp_01` | 75.00% |", summary)
        self.assertIn("| Body Style | `exp_01` | 95.00% |", summary)
        self.assertIn("**Best Experiment:** `exp_01`", summary)
        self.assertIn("**Best Mean Accuracy:** 85.00%", summary)
        self.assertNotIn("Vehicle Type", summary)

    def test_missing_target_metric_and_empty_sessions_have_no_fabricated_best(self):
        missing = experiment()
        missing["target_metric"] = "unknown_acc"
        summary = render_session_summary("/tmp/session_1", [missing], target_metric="unknown_acc", target_value=0.85)
        self.assertIn("No verified result is available", summary)
        self.assertNotIn("80.00%", summary)
        empty = render_session_summary("/tmp/session_1", [], heads=["make", "vehicle_type"], target_value=0.99)
        self.assertIn("| Total Experiments | 0 |", empty)
        self.assertIn("| Status | In Progress |", empty)
        self.assertIn("| Make | — | — |", empty)
        self.assertIn("No verified result is available", empty)

    def test_writer_updates_only_the_fixed_summary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "session_1"
            first = write_session_summary(session, [], heads=["make"])
            self.assertEqual(first, session / "session_summary.md")
            self.assertIn("| Total Experiments | 0 |", first.read_text(encoding="utf-8"))
            second = write_session_summary(session, [experiment()], status="Completed", target_value=0.99)
            self.assertEqual(second, first)
            self.assertIn("| Total Experiments | 1 |", second.read_text(encoding="utf-8"))
            self.assertEqual(list(session.iterdir()), [first])

    def test_unspecified_head_weights_show_the_effective_trainer_default(self):
        measured = experiment()
        measured["config"]["head_loss_weights"] = {}
        summary = render_session_summary("/tmp/session_1", [measured])
        self.assertIn("| Make Loss Weight | 1.0 |", summary)
        self.assertIn("| Vehicle Type Loss Weight | 1.0 |", summary)
        comparison = next(line for line in summary.splitlines() if line.startswith("| `exp_01` |"))
        self.assertIn("| 1.0 | 1.0 | 70.00% | 90.00% |", comparison)
        self.assertEqual(measured["config"]["head_loss_weights"], {})

    def test_reporter_writes_summary_without_an_additional_llm_request(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "session_1"
            measured = experiment()
            state = {"session_dir": str(session), "experiment_history": [measured],
                     "best_experiment": measured, "current_config": measured["config"],
                     "heads": measured["heads"], "decision_split": "test", "attempt_number": 1,
                     "target_metric": "mean_acc", "target_value": 0.99,
                     "stop_reason": "max_attempts", "is_valid": True}
            invoke = Mock(return_value=AIMessage(content="Recorded session interpretation"))
            with patch.object(reporter, "_get_llm", return_value=SimpleNamespace(invoke=invoke)), \
                 patch.object(reporter, "console"):
                result = reporter.reporter_node(state)
            invoke.assert_called_once()
            path = Path(result["session_summary_path"])
            self.assertEqual(path, session / "session_summary.md")
            self.assertEqual(result["report_summary"]["summary_md"], str(path))
            self.assertIn("| Status | Completed |", path.read_text(encoding="utf-8"))
            self.assertIn("| Target Reached | No |", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
