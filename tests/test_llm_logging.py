"""Keep readable JSON LLM history intact without API or GPU calls."""
from __future__ import annotations

import copy
import io
import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from openai import RateLimitError

from agent import llm_logging, llm_retry
from agent.llm import parse_json_response
from agent.session_log_format import session_log_records
from agent.nodes import config_optimizer, evaluator, reporter


class LLMLoggingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runs = self.root / "runs" / "resnet"
        self.path = self.runs / "session_1" / "llm_responses.json"
        path_patch = patch.object(llm_logging, "DEFAULT_RUNS_DIR", self.runs)
        path_patch.start()
        self.addCleanup(path_patch.stop)
        env_patch = patch.dict(os.environ, {
            "OPENAI_MODEL": "fixture-model", "OPENAI_API_KEY": "secret-test-key-not-for-log",
        })
        env_patch.start()
        self.addCleanup(env_patch.stop)
        self.messages = [
            SystemMessage(content="Evaluate the supplied metrics.\nKeep the requested format."),
            HumanMessage(content="Accuracy is 0.6.\nThe make head needs improvement."),
        ]
        self.state = {"session_dir": str(self.path.parent), "attempt_number": 1}

    def read_log(self, path=None):
        target = path or self.path
        document = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(document["session"]["session_id"], target.parent.name)
        return document

    def read_records(self, path=None):
        return session_log_records(self.read_log(path))

    def test_runtime_ignores_legacy_and_shared_logs_and_preserves_their_bytes(self):
        self.runs.mkdir(parents=True)
        shared_log = self.runs / "llm_responses.log"
        shared_json = self.runs / "llm_responses.json"
        shared_log.write_bytes(b"Previous readable log must be left intact")
        shared_json.write_bytes(b'{"requests": [{"response": "Historical shared log"}]}')
        self.path.parent.mkdir()
        sibling_log = self.path.with_suffix(".log")
        sibling_log.write_bytes(b"A session sibling legacy log must also be ignored")
        preserved = {path: path.read_bytes() for path in (shared_log, shared_json, sibling_log)}
        self.invoke(AIMessage(content="First response"))
        self.invoke(AIMessage(content="Second response"))
        records = self.read_records()
        self.assertEqual([record["response"] for record in records], ["Second response"])
        self.assertTrue(all(record["status"] == "success" for record in records))
        self.assertTrue(all(record.get("source") != "legacy_log" for record in records))
        for path, original in preserved.items():
            self.assertEqual(path.read_bytes(), original)

    def invoke(self, response, *, state=None, stage="optimization", factory=None):
        if factory is None:
            factory = lambda: SimpleNamespace(invoke=Mock(return_value=response))
        return llm_logging.invoke_logged_llm(
            self.messages, state=state or self.state, stage=stage, llm_factory=factory,
        )

    def assert_minimal_document(self, document):
        self.assertEqual(set(document), {"session"})
        session = document["session"]
        self.assertEqual(set(session), {"session_id", "model", "experiments"})
        for experiment in session["experiments"].values():
            self.assertEqual(set(experiment), {"attempt", "stages"})
            self.assertLessEqual(set(experiment["stages"]), {"evaluation", "optimization"})
            for stage in experiment["stages"].values():
                self.assertEqual(set(stage), {"status", "response"})

    def assert_failed_record(self, error, status="error"):
        records = self.read_records()
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["status"], status)
        self.assertIsNone(record["response"])
        self.assert_minimal_document(self.read_log())
        self.assertNotIn(str(error), self.path.read_text(encoding="utf-8"))

    def test_sessions_have_separate_minimal_files_preserving_multiline_responses(self):
        first = AIMessage(content="Increase epochs to 40.\nRaise the make loss weight.")
        second = AIMessage(content="The second session converged.\nKeep its checkpoint.")
        self.assertIs(self.invoke(first), first)
        first_bytes = self.path.read_bytes()
        other_session = self.runs / "session_2"
        other_path = other_session / "llm_responses.json"
        other_state = {"session_dir": str(other_session), "attempt_number": 3}
        self.assertIs(self.invoke(second, state=other_state, stage="evaluation"), second)
        self.assertEqual(self.path.read_bytes(), first_bytes)
        first_records, second_records = self.read_records(), self.read_records(other_path)
        for records, response, attempt, stage in ((first_records, first, 1, "optimization"),
                                                  (second_records, second, 3, "evaluation")):
            self.assertEqual(records, [{"attempt": attempt, "stage": stage, "model": "fixture-model",
                                        "status": "success", "response": response.content}])
        self.assertIn("exp_01", self.read_log()["session"]["experiments"])
        self.assertIn("exp_03", self.read_log(other_path)["session"]["experiments"])
        self.assertFalse((self.runs / "llm_responses.json").exists())
        for path in (self.path, other_path):
            self.assert_minimal_document(self.read_log(path))
            saved = path.read_text(encoding="utf-8")
            self.assertNotIn("secret-test-key-not-for-log", saved)
            self.assertNotIn(self.messages[0].content.splitlines()[0], saved)

    def test_pending_response_is_minimal_valid_json_before_factory_and_invocation(self):
        response = AIMessage(content="Measured results justify more training.")
        checks = []

        def assert_pending():
            document = self.read_log()
            self.assert_minimal_document(document)
            payload = document["session"]["experiments"]["exp_01"]["stages"]["optimization"]
            self.assertEqual(payload, {"status": "pending", "response": None})

        def factory():
            assert_pending()
            checks.append("factory")

            def invoke(messages):
                self.assertIs(messages, self.messages)
                assert_pending()
                checks.append("invoke")
                return response

            return SimpleNamespace(invoke=invoke)

        self.invoke(response, factory=factory)
        self.assertEqual(checks, ["factory", "invoke"])
        self.assertEqual(self.read_records()[0]["response"], response.content)
        self.assertEqual(self.read_records()[0]["status"], "success")

    def test_json_response_is_native_and_extraneous_response_metadata_is_removed(self):
        self.messages[-1] = HumanMessage(content='{"private_context":"Do not retain this prompt"}')
        expected = {"updated_config": {"epochs": 40, "arch": "resnet34", "head_loss_weights": {"make": 1.5}},
                    "rationale": "Increase epochs.\nPrioritize make."}
        response = AIMessage(content=json.dumps({**expected, "request_id": "metadata", "usage": {"tokens": 50}}, indent=2))
        self.assertIs(self.invoke(response), response)
        self.assertEqual(self.read_records()[0]["response"], expected)
        self.assert_minimal_document(self.read_log())
        saved = self.path.read_text(encoding="utf-8")
        self.assertNotIn("private_context", saved)
        self.assertNotIn("request_id", saved)
        self.assertNotIn("usage", saved)
        self.assertIn('"updated_config": {', saved)

    def test_client_construction_error_is_logged_and_reraised(self):
        error = ValueError("LLM timeout setting is invalid")
        factory = Mock(side_effect=error)
        with self.assertRaises(ValueError) as raised:
            self.invoke(None, factory=factory)
        self.assertIs(raised.exception, error)
        factory.assert_called_once_with()
        self.assert_failed_record(error)

    def test_provider_error_is_logged_and_reraised(self):
        error = RuntimeError("413: requested input tokens exceed provider limit")
        invoke = Mock(side_effect=error)
        with self.assertRaises(RuntimeError) as raised:
            self.invoke(None, factory=lambda: SimpleNamespace(invoke=invoke))
        self.assertIs(raised.exception, error)
        invoke.assert_called_once_with(self.messages)
        self.assert_failed_record(error)

    def test_keyboard_interrupt_is_logged_and_reraised(self):
        interrupt = KeyboardInterrupt("User stopped the LLM request")
        with self.assertRaises(KeyboardInterrupt) as raised:
            self.invoke(None, factory=lambda: SimpleNamespace(invoke=Mock(side_effect=interrupt)))
        self.assertIs(raised.exception, interrupt)
        self.assert_failed_record(interrupt, "interrupted")

    @staticmethod
    def temporary_rate_limit():
        request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
        message = "Rate limit reached on input tokens per minute; Please try again in 4.26s."
        return RateLimitError(message, response=httpx.Response(429, request=request),
                              body={"error": {"message": message, "code": "rate_limit_exceeded", "type": "tokens"}})

    def assert_pending_stage(self):
        document = self.read_log()
        self.assert_minimal_document(document)
        self.assertEqual(document["session"]["experiments"]["exp_01"]["stages"]["optimization"],
                         {"status": "pending", "response": None})

    def test_retry_keeps_one_pending_stage_until_single_success(self):
        error = self.temporary_rate_limit()
        response = AIMessage(content=json.dumps({"updated_config": {"epochs": 30}, "rationale": "Converge."}))
        calls = []
        def provider(messages):
            self.assert_pending_stage()
            calls.append(messages)
            if len(calls) == 1:
                raise error
            return response
        def wait(seconds):
            self.assert_pending_stage()
            self.assertGreaterEqual(seconds, 4.26)
        factory = Mock(return_value=SimpleNamespace(invoke=provider))
        with patch.dict(os.environ, {"LLM_MAX_RETRIES": "2", "LLM_RATE_LIMIT_MAX_WAIT_SECONDS": "60"}), \
             patch.object(llm_retry.time, "sleep", side_effect=wait) as sleep, \
             patch.object(llm_retry.console, "print"):
            result = self.invoke(None, factory=factory)
        self.assertIs(result, response)
        factory.assert_called_once_with()
        self.assertEqual(len(calls), 2)
        sleep.assert_called_once()
        self.assertEqual(self.read_records(), [{"attempt": 1, "stage": "optimization", "model": "fixture-model",
                                               "status": "success", "response": json.loads(response.content)}])
        self.assert_minimal_document(self.read_log())
        self.assertEqual(self.path.with_name(".llm_responses.json.lock").read_text(encoding="utf-8"), "")
        self.assertNotIn(str(error), self.path.read_text(encoding="utf-8"))

    def test_retry_exhaustion_records_only_one_final_error(self):
        error = self.temporary_rate_limit()
        invoke = Mock(side_effect=error)
        with patch.dict(os.environ, {"LLM_MAX_RETRIES": "2", "LLM_RATE_LIMIT_MAX_WAIT_SECONDS": "60"}), \
             patch.object(llm_retry.time, "sleep", side_effect=lambda seconds: self.assert_pending_stage()) as sleep, \
             patch.object(llm_retry.console, "print"), self.assertRaises(RateLimitError) as raised:
            self.invoke(None, factory=lambda: SimpleNamespace(invoke=invoke))
        self.assertIs(raised.exception, error)
        self.assertEqual(invoke.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assert_failed_record(error)
        self.assertEqual(self.path.with_name(".llm_responses.json.lock").read_text(encoding="utf-8"), "")

    def test_interrupting_retry_wait_records_interrupted_and_preserves_interrupt(self):
        error = self.temporary_rate_limit()
        interrupt = KeyboardInterrupt("User interrupted the retry wait")
        invoke = Mock(side_effect=error)
        def interrupt_wait(seconds):
            self.assert_pending_stage()
            raise interrupt
        with patch.dict(os.environ, {"LLM_MAX_RETRIES": "2", "LLM_RATE_LIMIT_MAX_WAIT_SECONDS": "60"}), \
             patch.object(llm_retry.time, "sleep", side_effect=interrupt_wait), \
             patch.object(llm_retry.console, "print"), self.assertRaises(KeyboardInterrupt) as raised:
            self.invoke(None, factory=lambda: SimpleNamespace(invoke=invoke))
        self.assertIs(raised.exception, interrupt)
        invoke.assert_called_once_with(self.messages)
        self.assert_failed_record(interrupt, "interrupted")
        self.assertEqual(self.path.with_name(".llm_responses.json.lock").read_text(encoding="utf-8"), "")

    def test_report_calls_use_retry_without_creating_log(self):
        response = AIMessage(content="Session report")
        invoke = Mock(side_effect=[self.temporary_rate_limit(), response])
        with patch.dict(os.environ, {"LLM_MAX_RETRIES": "2", "LLM_RATE_LIMIT_MAX_WAIT_SECONDS": "60"}), \
             patch.object(llm_retry.time, "sleep") as sleep, patch.object(llm_retry.console, "print"):
            result = self.invoke(None, stage="report", factory=lambda: SimpleNamespace(invoke=invoke))
        self.assertIs(result, response)
        self.assertEqual(invoke.call_count, 2)
        sleep.assert_called_once()
        self.assertFalse(self.runs.exists())

    def test_raw_malformed_json_is_saved_before_caller_parsing_fails(self):
        raw = "{\"updated_config\": {\"epochs\": 40},\n\"rationale\": \"Increase epochs"
        response = self.invoke(AIMessage(content=raw))
        with self.assertRaises(ValueError):
            parse_json_response(response.content)
        record = self.read_records()[0]
        self.assertEqual(record["response"], raw)
        self.assertEqual(record["status"], "success")

    def test_structured_response_content_is_saved_in_full(self):
        content = [
            {"type": "text", "text": "Increase epochs.\nAccuracies remain low."},
            {"type": "text", "text": "تعلم تصنيف السيارات"},
        ]
        response = AIMessage(content=content)
        self.assertIs(self.invoke(response), response)
        record = self.read_records()[0]
        self.assertEqual(record["response"], content)
        self.assertIn("تعلم تصنيف السيارات", self.path.read_text(encoding="utf-8"))

    def test_unwritable_log_warns_without_changing_valid_response(self):
        blocker = self.root / "not-a-directory"
        blocker.write_text("fixture")
        response = AIMessage(content="A valid response survives an unavailable log file.")
        with patch.object(llm_logging, "session_log_path", return_value=blocker / "llm_responses.json"), \
             patch.object(llm_logging.console, "print") as warning:
            self.assertIs(self.invoke(response), response)
        warning.assert_called()

    def test_unwritable_log_preserves_the_original_provider_error(self):
        blocker = self.root / "not-a-directory"
        blocker.write_text("fixture")
        error = RuntimeError("Provider failed while logging was unavailable")
        with patch.object(llm_logging, "session_log_path", return_value=blocker / "llm_responses.json"), \
             patch.object(llm_logging.console, "print") as warning:
            with self.assertRaises(RuntimeError) as raised:
                self.invoke(None, factory=lambda: SimpleNamespace(invoke=Mock(side_effect=error)))
        self.assertIs(raised.exception, error)
        warning.assert_called()

    def test_invalid_existing_json_or_schema_is_preserved_without_blocking_the_llm(self):
        self.path.parent.mkdir(parents=True)
        invalid_documents = ("{broken json\n", "[]\n", '{"schema_version":1,"requests":[]}\n',
                             '{"session":{"session_id":"session_1","experiments":[]}}\n',
                             '{"session":{"session_id":"session_1","experiments":{"exp_01":null}}}\n')
        response = AIMessage(content="Do not overwrite the previous log.")
        for original in invalid_documents:
            with self.subTest(original=original):
                self.path.write_text(original, encoding="utf-8")
                with patch.object(llm_logging.console, "print") as warning:
                    self.assertIs(self.invoke(response), response)
                warning.assert_called()
                self.assertEqual(self.path.read_text(encoding="utf-8"), original)

    def test_atomic_replace_failure_preserves_a_valid_existing_document(self):
        self.invoke(AIMessage(content="Previously saved response."))
        previous = self.path.read_bytes()
        response = AIMessage(content="The new response remains available even if saving fails.")
        with patch("os.replace", side_effect=PermissionError("Cannot replace the log file")), \
             patch.object(llm_logging.console, "print") as warning:
            self.assertIs(self.invoke(response), response)
        warning.assert_called()
        self.assertEqual(self.path.read_bytes(), previous)
        self.assertEqual(len(self.read_records()), 1)

    def test_concurrent_requests_keep_one_latest_stage_and_clear_temporary_nonces(self):
        barrier = threading.Barrier(3)

        def call(index):
            response = AIMessage(content=json.dumps({"rationale": f"call{index}"}))
            messages = [HumanMessage(content=json.dumps({"call": index}))]

            def factory():
                # All pending records must be written before any provider completes.
                barrier.wait(timeout=5)
                self.assertEqual(len(self.read_records()), 1)
                return SimpleNamespace(invoke=Mock(return_value=response))

            return llm_logging.invoke_logged_llm(messages, state=self.state,
                                               stage="optimization", llm_factory=factory)

        with ThreadPoolExecutor(max_workers=3) as pool:
            responses = list(pool.map(call, range(3)))
        records = self.read_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "success")
        self.assertIn(records[0]["response"]["rationale"], {"call0", "call1", "call2"})
        self.assert_minimal_document(self.read_log())
        self.assertEqual([json.loads(response.content)["rationale"] for response in responses], ["call0", "call1", "call2"])
        self.assertEqual(self.path.with_name(".llm_responses.json.lock").read_text(encoding="utf-8"), "")

    def test_multiple_attempts_and_stages_keep_only_the_latest_response(self):
        first = AIMessage(content="First evaluation")
        updated = AIMessage(content="Repeated evaluation")
        self.invoke(first, stage="evaluation")
        self.invoke(updated, stage="evaluation")
        self.invoke(AIMessage(content="Optimizer response"), stage="optimization")
        self.invoke(AIMessage(content="Second attempt evaluation"),
                    state={**self.state, "attempt_number": 2}, stage="evaluation")
        session = self.read_log()["session"]
        self.assertEqual(session["model"], "fixture-model")
        self.assertEqual(set(session["experiments"]), {"exp_01", "exp_02"})
        stages = session["experiments"]["exp_01"]["stages"]
        self.assertEqual(set(stages), {"evaluation", "optimization"})
        self.assertEqual(stages["evaluation"]["response"], updated.content)
        self.assertNotIn("previous_requests", stages["evaluation"])
        self.assertEqual(len(self.read_records()), 3)
        self.assert_minimal_document(self.read_log())
        self.assertEqual(list(self.runs.glob("*/llm_responses.json")), [self.path])

    def test_old_request_finishing_last_does_not_replace_the_newer_stage_request(self):
        old_started = threading.Event()
        release_old = threading.Event()
        old_response = AIMessage(content="Older request completed later")
        newer_response = AIMessage(content="Newer request remains the visible result")

        def slow_invoke(messages):
            old_started.set()
            if not release_old.wait(timeout=5):
                raise TimeoutError("Test did not release the old request")
            return old_response

        with ThreadPoolExecutor(max_workers=1) as pool:
            old = pool.submit(self.invoke, old_response,
                              factory=lambda: SimpleNamespace(invoke=slow_invoke))
            try:
                self.assertTrue(old_started.wait(timeout=5))
                self.invoke(newer_response)
            finally:
                release_old.set()
            self.assertIs(old.result(timeout=5), old_response)
        stage = self.read_log()["session"]["experiments"]["exp_01"]["stages"]["optimization"]
        self.assertEqual(stage["response"], newer_response.content)
        self.assertEqual(stage["status"], "success")
        self.assertEqual(stage, {"status": "success", "response": newer_response.content})
        self.assertEqual(len(self.read_records()), 1)
        self.assertNotIn(old_response.content, self.path.read_text(encoding="utf-8"))
        self.assertEqual(self.path.with_name(".llm_responses.json.lock").read_text(encoding="utf-8"), "")

    def test_missing_session_reserves_and_reuses_one_numbered_directory(self):
        state = {"attempt_number": 1}
        self.invoke(AIMessage(content="Evaluation"), state=state, stage="evaluation")
        self.assertEqual(state["session_dir"], str(self.runs / "session_1"))
        self.invoke(AIMessage(content="Optimization"), state=state, stage="optimization")
        session = self.read_log()["session"]
        self.assertEqual(set(session["experiments"]["exp_01"]["stages"]), {"evaluation", "optimization"})
        self.assertFalse((self.runs / "session_2").exists())

    def test_report_and_requests_without_training_attempt_are_not_logged(self):
        factory = Mock(return_value=SimpleNamespace(invoke=Mock(return_value=AIMessage(content="Report"))))
        for stage, state in (("report", self.state), ("evaluation", {"attempt_number": 0})):
            result = self.invoke(None, state=state, stage=stage, factory=factory)
            self.assertEqual(result.content, "Report")
        self.assertEqual(factory.call_count, 2)
        self.assertFalse(self.runs.exists())

    def test_session_path_prefers_explicit_session_then_configuration_output(self):
        out = self.runs / "session_7"
        explicit = self.runs / "session_9"
        self.assertEqual(llm_logging.session_log_path({"current_config": {"out": str(out)}}),
                         out / "llm_responses.json")
        self.assertEqual(llm_logging.session_log_path({"session_dir": str(explicit),
                                                      "current_config": {"out": str(out)}}),
                         explicit / "llm_responses.json")
        self.assertFalse(self.runs.exists())

    def test_document_for_different_session_is_preserved_without_blocking_response(self):
        self.path.parent.mkdir(parents=True)
        original = json.dumps({"session": {"session_id": "session_2", "model": "old-model", "experiments": {}}})
        self.path.write_text(original, encoding="utf-8")
        response = AIMessage(content="Current response is still available")
        with patch.object(llm_logging.console, "print") as warning:
            self.assertIs(self.invoke(response), response)
        warning.assert_called()
        self.assertEqual(self.path.read_text(encoding="utf-8"), original)

    def test_only_evaluation_and_optimization_are_logged_while_all_nodes_receive_metrics(self):
        session = self.runs / "session_4"
        self.path = session / "llm_responses.json"
        config = {
            "arch": "resnet18", "weights": str(self.root / "weights.pth"),
            "train_csv": str(self.root / "train.csv"), "val_csv": str(self.root / "val.csv"),
            "data_root": None, "heads": ["make", "vehicle_type"], "epochs": 2,
            "batch_size": 8, "num_workers": 0, "img_size": [64, 64], "lr": 0.0001,
            "weight_decay": 0.0001, "gamma": 1.0, "patience": 10, "monitor": "mean",
            "head_loss_weights": {"make": 1.0, "vehicle_type": 1.0}, "alpha_json": None,
            "nproc_per_node": 1, "out": str(session), "name": "exp_01",
        }
        test_metrics = {
            "status": "ok", "split": "test", "decision_split": "test", "test_evaluated": True,
            "full_coverage": True, "total_images": 5, "evaluated_images": 5, "matched_images": 5,
            "test_csv": str(self.root / "test.csv"), "checkpoint": str(session / "best_model.pth"),
            "mean_acc": 0.7, "make_acc": 0.6, "vehicle_type_acc": 0.8,
            "train_loss": 1.0, "val_loss": 1.2,
            "heads": {
                "make": {"acc": 0.6, "correct": 3, "count": 5, "coverage": 1.0,
                         "per_class_accuracy": [
                             {"class": "Toyota", "correct": 2, "total": 3, "accuracy": 2 / 3},
                             {"class": "Ford", "correct": 1, "total": 2, "accuracy": 0.5},
                         ]},
                "vehicle_type": {"acc": 0.8, "correct": 4, "count": 5, "coverage": 1.0,
                                 "per_class_accuracy": [
                                     {"class": "sedan", "correct": 3, "total": 3, "accuracy": 1.0},
                                     {"class": "SUV", "correct": 1, "total": 2, "accuracy": 0.5},
                                 ]},
            },
        }
        state = {
            "session_dir": str(session), "attempt_number": 1, "current_config": config,
            "training_result": {"status": "success"}, "decision_split": "test",
            "test_csv": test_metrics["test_csv"], "metrics": test_metrics,
            "test_result": {"status": "ok", "metrics": test_metrics, "csv": test_metrics["test_csv"]},
            "best_experiment": {"config": config, "metrics": test_metrics, "decision_split": "test",
                                "best_weights": test_metrics["checkpoint"]},
            "target_metric": "mean_acc", "target_value": 0.99, "stop_reason": "max_attempts",
        }
        original_metrics = copy.deepcopy(test_metrics)
        evaluation_response = AIMessage(content=json.dumps({
            "verdict": "underfitting", "target_reached": False, "issues": [],
            "recommendations": ["Increase epochs"], "reasoning": "Only two epochs completed.",
        }))
        optimization_response = AIMessage(content=json.dumps({
            "updated_config": {"epochs": 40},
            "rationale": "Increase epochs.\nPrioritize learning the make task.",
        }))
        report_response = AIMessage(content="The target remains unmet.\nFurther training was recommended.")
        evaluation_invoke = Mock(return_value=evaluation_response)
        optimization_invoke = Mock(return_value=optimization_response)
        report_invoke = Mock(return_value=report_response)
        with redirect_stdout(io.StringIO()), \
             patch.object(evaluator, "_get_llm", return_value=SimpleNamespace(invoke=evaluation_invoke)), \
             patch.object(config_optimizer, "get_llm", return_value=SimpleNamespace(invoke=optimization_invoke)), \
             patch.object(config_optimizer, "discover_pretrained_weights", return_value=[]), \
             patch.object(reporter, "_get_llm", return_value=SimpleNamespace(invoke=report_invoke)):
            evaluation = evaluator.evaluator_node(state)
            optimization = config_optimizer.config_optimizer_node({**state, **evaluation})
            report = reporter.reporter_node({**state, **evaluation, **optimization})
        records = self.read_records()
        self.assertEqual(len(records), 2)
        self.assertEqual([record["stage"] for record in records], ["evaluation", "optimization"])
        self.assertEqual([record["status"] for record in records], ["success"] * 2)
        self.assertEqual(records[0]["response"], json.loads(evaluation_response.content))
        self.assertEqual(records[1]["response"], json.loads(optimization_response.content))
        self.assertEqual(optimization["current_config"]["epochs"], 40)
        self.assertIn(report_response.content, report["final_report"])
        for invoke, metric_key in ((evaluation_invoke, "metrics"), (optimization_invoke, "metrics"),
                                   (report_invoke, "best_metrics")):
            invoke.assert_called_once()
            human_context = json.loads(invoke.call_args.args[0][-1].content)
            make = human_context[metric_key]["heads"]["make"]["per_class_accuracy"]
            vehicle_type = human_context[metric_key]["heads"]["vehicle_type"]["per_class_accuracy"]
            self.assertEqual(make["rows"], [["Ford", 1, 2, 0.5], ["Toyota", 2, 3, 0.666667]])
            self.assertEqual(vehicle_type["rows"], [["SUV", 1, 2, 0.5], ["sedan", 3, 3, 1.0]])
            self.assertEqual(make["total_classes"], 2)
            self.assertEqual(make["shown_classes"], 2)
            self.assertEqual(make["macro_accuracy"], 0.583333)
            self.assertEqual(vehicle_type["macro_accuracy"], 0.75)
            self.assertEqual(human_context[metric_key]["mean_acc"], 0.7)
        self.assert_minimal_document(self.read_log())
        self.assertNotIn('"per_class_accuracy"', self.path.read_text(encoding="utf-8"))
        self.assertNotIn(report_response.content, self.path.read_text(encoding="utf-8"))
        self.assertEqual(test_metrics, original_metrics)
        self.assertEqual(report["report_summary"]["best_metrics"], original_metrics)


if __name__ == "__main__":
    unittest.main()
