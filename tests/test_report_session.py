"""Keep reports and their LLM requests in the same reserved session."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage

from agent.nodes import reporter


class ReportSessionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="resnet-report-session-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base = self.root / "runs" / "resnet"

    def run_report(self, state, *, factory):
        with patch.object(reporter, "DEFAULT_RUNS_DIR", self.base), \
             patch.object(reporter, "_get_llm", side_effect=factory), \
             patch.object(reporter.console, "print"), \
             patch.object(reporter.console, "rule"):
            return reporter.reporter_node(state)

    def test_invalid_dataset_reserves_once_and_saves_report_without_llm_json(self):
        state = {"is_valid": False, "attempt_number": 0,
                 "validation_report": {"errors": ["Training CSV does not exist"]},
                 "train_csv": "/missing/train.csv", "experiment_history": []}
        original = deepcopy(state)
        session = self.base / "session_1"
        response = AIMessage(content="Dataset validation failed; no model was trained.")
        invoke = Mock(return_value=response)

        def factory():
            self.assertTrue(session.is_dir())
            self.assertFalse((session / "llm_responses.json").exists())
            return SimpleNamespace(invoke=invoke)

        with patch.object(reporter, "allocate_session_directory", wraps=reporter.allocate_session_directory) as reserve:
            result = self.run_report(state, factory=factory)
        reserve.assert_called_once_with(self.base)
        invoke.assert_called_once()
        self.assertEqual(json.loads(invoke.call_args.args[0][-1].content)["stop_reason"], "validation_failed")
        self.assertEqual(result["session_dir"], str(session))
        report = Path(result["report_summary"]["report_md"])
        self.assertEqual(report.parent, session)
        self.assertTrue(report.is_file())
        self.assertTrue((session / "report_summary.json").is_file())
        self.assertEqual(result["report_summary"]["stop_reason"], "validation_failed")
        self.assertEqual(result["report_summary"]["total_attempts"], 0)
        self.assertEqual(result["report_summary"]["best_model_path"], "")
        self.assertFalse((session / "llm_responses.json").exists())
        self.assertEqual([path.name for path in self.base.iterdir()], ["session_1"])
        self.assertEqual(state, original)

    def test_invalid_dataset_provider_error_still_saves_factual_report_without_llm_json(self):
        state = {"is_valid": False, "attempt_number": 0,
                 "validation_report": {"errors": ["Test CSV is missing"]}}
        original = deepcopy(state)
        factory = Mock(side_effect=RuntimeError("Offline provider"))
        result = self.run_report(state, factory=factory)
        factory.assert_called_once_with()
        session = Path(result["session_dir"])
        self.assertEqual(session, self.base / "session_1")
        self.assertFalse((session / "llm_responses.json").exists())
        self.assertEqual(Path(result["report_summary"]["report_md"]).parent, session)
        self.assertIn("Test CSV is missing", result["final_report"])
        self.assertIn("No successful checkpoint available", result["final_report"])
        self.assertTrue(any("Offline provider" in error for error in result["report_summary"]["errors"]))
        self.assertTrue((session / "report_summary.json").is_file())
        self.assertEqual(state, original)

    def test_existing_session_takes_priority_and_never_allocates_or_mutates_state(self):
        session = self.base / "session_7"
        state = {"session_dir": str(session), "attempt_number": 2,
                 "current_config": {"out": str(self.base / "session_99"), "heads": ["make"]},
                 "is_valid": True, "stop_reason": "max_attempts", "experiment_history": []}
        original = deepcopy(state)
        invoke = Mock(return_value=AIMessage(content="No successful checkpoint is available."))
        with patch.object(reporter, "allocate_session_directory") as reserve:
            result = self.run_report(state, factory=lambda: SimpleNamespace(invoke=invoke))
        reserve.assert_not_called()
        self.assertEqual(result["session_dir"], str(session))
        self.assertEqual(Path(result["report_summary"]["report_md"]).parent, session)
        self.assertTrue((session / "report_summary.json").is_file())
        self.assertFalse((session / "llm_responses.json").exists())
        self.assertFalse((self.base / "session_99").exists())
        self.assertEqual(state, original)

    def test_configured_output_is_reused_when_state_has_no_session_directory(self):
        session = self.root / "custom-output" / "session_3"
        state = {"current_config": {"out": str(session)}, "attempt_number": 1,
                 "experiment_history": [], "stop_reason": "error"}
        original = deepcopy(state)
        invoke = Mock(return_value=AIMessage(content="Training has no successful checkpoint."))
        with patch.object(reporter, "allocate_session_directory") as reserve:
            result = self.run_report(state, factory=lambda: SimpleNamespace(invoke=invoke))
        reserve.assert_not_called()
        self.assertEqual(result["session_dir"], str(session))
        self.assertEqual(Path(result["report_summary"]["report_md"]).parent, session)
        self.assertTrue((session / "report_summary.json").is_file())
        self.assertFalse((session / "llm_responses.json").exists())
        self.assertEqual(state, original)


if __name__ == "__main__":
    unittest.main()
