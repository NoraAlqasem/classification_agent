"""Interrupted graph runs retain completed proof and identify partial attempts."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from main import _save_session_outcome


class SessionOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.session = Path(self.temp.name) / "session_1"
        self.state = {
            "session_dir": str(self.session), "attempt_number": 1,
            "current_config": {"out": str(self.session), "name": "exp_02", "epochs": 30, "monitor": "mean"},
            "experiment_history": [{"attempt": 1, "training_status": "success", "score": .8}],
            "heads": ["make"], "target_metric": "mean_acc", "target_value": .99, "decision_split": "test",
        }

    def test_interrupted_attempt_retains_training_progress_without_test_proof(self):
        directory = self.session / "exp_02/train_evaluation"
        directory.mkdir(parents=True)
        (directory / "training_metrics.csv").write_text(
            "Epoch,Split,TotalLoss,MeanAcc,make_Loss,make_Acc\n"
            "1,train,1.0,0.6,1.0,0.6\n1,val,0.9,0.7,0.9,0.7\n"
        )
        original = copy.deepcopy(self.state)
        with patch("agent.session_summary.write_session_summary") as save:
            _save_session_outcome(self.state, "Interrupted")
        save.assert_called_once()
        history = save.call_args.args[1]
        self.assertEqual(history[0], original["experiment_history"][0])
        self.assertEqual(history[1]["attempt"], 2)
        self.assertEqual(history[1]["training_status"], "interrupted")
        self.assertEqual(history[1]["metrics"]["total_epochs"], 1)
        self.assertNotIn("test_result", history[1])
        self.assertEqual(save.call_args.kwargs["status"], "Interrupted")
        self.assertEqual(self.state, original)

    def test_existing_attempt_is_not_duplicated_and_missing_state_writes_nothing(self):
        (self.session / "exp_02").mkdir(parents=True)
        self.state["experiment_history"].append({"attempt": 2, "training_status": "error"})
        with patch("agent.session_summary.write_session_summary") as save:
            _save_session_outcome(self.state, "Failed")
            _save_session_outcome(None, "Interrupted")
            _save_session_outcome({"heads": ["make"]}, "Interrupted")
        save.assert_called_once()
        self.assertEqual(len(save.call_args.args[1]), 2)
        self.assertEqual(save.call_args.kwargs["status"], "Failed")

    def test_summary_failure_does_not_mask_the_original_stop(self):
        with patch("agent.session_summary.write_session_summary", side_effect=OSError("disk unavailable")), \
             patch("main.console.print") as warning:
            _save_session_outcome(self.state, "Interrupted")
        warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
