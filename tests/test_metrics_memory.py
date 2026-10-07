from __future__ import annotations

import csv
import copy
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rich.console import Console

from agent.nodes.memory_node import memory_node
from agent.nodes.monitor import _parse_results_csv, monitor_node
from memory.experiment_store import ExperimentStore, experiment_score

_FRAMEWORK_AVAILABLE = (
    importlib.util.find_spec("langchain_core") is not None
    and importlib.util.find_spec("langgraph") is not None
)


def test_evidence(make: float = .7, color: float = .9, csv_path: str = "/tmp/test.csv") -> dict:
    total = 100
    return {
        "status": "ok", "decision_split": "test", "split": "test", "test_evaluated": True,
        "full_coverage": True, "total_images": total, "evaluated_images": total, "matched_images": total,
        "test_csv": csv_path, "mean_acc": (make + color) / 2, "make_acc": make, "color_acc": color,
        "heads": {head: {"acc": score, "correct": round(score * total), "count": total, "coverage": 1.0}
                  for head, score in (("make", make), ("color", color))},
    }


class MetricsParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="resnet-metrics-test-")
        self.addCleanup(self.directory.cleanup)
        self.run_dir = Path(self.directory.name)
        self.csv_path = self.run_dir / "training_metrics.csv"
        self.columns = [
            "Epoch", "Split", "TotalLoss", "MeanAcc",
            "make_Loss", "make_Acc", "color_Loss", "color_Acc",
        ]
        self.rows = [
            [1, "train", .9, .5, .8, .5, 1., .5],
            [1, "val", 1., .7, .8, .8, 1.2, .6],
            [2, "train", .5, .8, .4, .8, .6, .8],
            [2, "val", .9, .75, 1., .6, .8, .9],
            [3, "train", .3, .9, .2, .9, .4, .9],
            [3, "val", 1.1, .75, 1., .7, 1.2, .8],
        ]
        self.write_log()

    def write_log(self) -> None:
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(self.columns)
            writer.writerows(self.rows)

    def test_mean_monitor_keeps_heads_at_same_checkpoint_epoch(self) -> None:
        metrics = _parse_results_csv(self.csv_path)
        self.assertEqual(metrics["best_epoch"], 2)
        self.assertEqual(metrics["best_score"], .75)
        self.assertEqual(metrics["make_acc"], .6)
        self.assertEqual(metrics["color_acc"], .9)
        self.assertEqual(metrics["heads"]["make"], {"acc": .6, "loss": 1.})
        self.assertEqual(metrics["final_epoch"], 3)
        self.assertEqual(metrics["final_make_acc"], .7)
        self.assertEqual(metrics["val_loss"], 1.1)
        self.assertEqual(metrics["loss_epoch"], 3)

    def test_named_head_selects_its_checkpoint_instead_of_mean_maximum(self) -> None:
        metrics = _parse_results_csv(self.csv_path, monitor="make")
        self.assertEqual(metrics["best_epoch"], 1)
        self.assertEqual(metrics["best_score"], .8)
        self.assertEqual(metrics["mean_acc"], .7)
        self.assertEqual(metrics["color_acc"], .6)

    def test_tied_best_score_selects_first_epoch_matching_trainer(self) -> None:
        self.rows[5][5] = .8
        self.write_log()
        metrics = _parse_results_csv(self.csv_path, monitor="make")
        self.assertEqual(metrics["best_epoch"], 1)
        self.assertEqual(metrics["final_make_acc"], .8)

    def test_nonfinite_losses_are_flagged_without_nonstandard_json_numbers(self) -> None:
        for token in ("nan", "inf", "-inf"):
            with self.subTest(loss=token):
                self.rows[3][2] = token
                self.write_log()
                metrics = _parse_results_csv(self.csv_path)
                self.assertTrue(metrics["nonfinite_loss"])
                self.assertIsNone(metrics["val_loss_history"][1])
                json.dumps(metrics, allow_nan=False)

    def test_unpaired_epochs_and_unknown_monitor_reject_incompatible_logs(self) -> None:
        with self.assertRaisesRegex(ValueError, "monitor"):
            _parse_results_csv(self.csv_path, monitor="missing_head")
        self.rows.pop()
        self.write_log()
        with self.assertRaisesRegex(ValueError, "train/val pair"):
            _parse_results_csv(self.csv_path)

    def test_monitor_returns_reportable_error_for_missing_or_failed_log(self) -> None:
        with patch("agent.nodes.monitor.console", Console(file=io.StringIO())):
            missing = monitor_node({
                "training_result": {"status": "success", "run_dir": str(self.run_dir / "missing")},
            })
            failed = monitor_node({
                "training_result": {"status": "error", "error": "CUDA unavailable", "run_dir": str(self.run_dir)},
            })
        self.assertEqual(missing["metrics"]["status"], "no_results_csv")
        self.assertEqual(failed["metrics"]["status"], "training_failed")
        self.assertEqual(failed["metrics"]["error"], "CUDA unavailable")


@unittest.skipUnless(_FRAMEWORK_AVAILABLE, "Install the LangChain/LangGraph dependencies to run evaluator tests")
class DeterministicEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        from agent.nodes import evaluator
        self.evaluator = evaluator
        self.metrics = {
            "status": "ok", "mean_acc": .95, "make_acc": .7,
            "train_loss": .3, "val_loss": .6,
            "train_loss_history": [1., .8, .6, .4, .3],
            "val_loss_history": [1., .9, .8, .7, .6],
            "nonfinite_loss": False, "nonfinite_accuracy": False,
            "heads": {"make": {"acc": .7, "loss": .6}},
        }
        self.state = {
            "attempt_number": 1, "training_result": {"status": "success"},
            "metrics": self.metrics, "target_metric": "make_acc", "target_value": .85,
            "current_config": {"monitor": "make"}, "run_dir": "/tmp/synthetic-resnet-run",
        }

    def evaluate_with_response(self, reached: bool, verdict: str = "good"):
        llm = MagicMock()
        llm.invoke.return_value = SimpleNamespace(content=json.dumps({
            "verdict": verdict, "target_reached": reached, "issues": [],
            "recommendations": [], "reasoning": "Synthetic evaluation",
        }))
        with patch.object(self.evaluator, "_get_llm", return_value=llm), \
             patch.object(self.evaluator, "console", Console(file=io.StringIO())):
            result = self.evaluator.evaluator_node(self.state)
        llm.invoke.assert_called_once()
        return result

    def test_llm_true_cannot_accept_wrong_head_or_below_threshold(self) -> None:
        result = self.evaluate_with_response(reached=True)
        self.assertFalse(result["target_reached"])
        self.assertFalse(result["evaluation_result"]["target_reached"])

    def test_llm_false_cannot_reject_measured_target_success(self) -> None:
        self.metrics["make_acc"] = .9
        result = self.evaluate_with_response(reached=False)
        self.assertTrue(result["target_reached"])
        self.assertTrue(result["evaluation_result"]["target_reached"])

    def test_missing_target_never_falls_back_to_mean_accuracy(self) -> None:
        self.state["target_metric"] = "missing_acc"
        result = self.evaluate_with_response(reached=True)
        self.assertFalse(result["target_reached"])
        self.assertTrue(any("missing_acc" in issue for issue in result["issues_detected"]))

    def test_nonfinite_and_failed_attempts_cannot_reach_target(self) -> None:
        self.metrics["make_acc"] = .99
        cases = [
            ({"nonfinite_loss": True}, {"status": "success"}),
            ({"val_loss": float("nan")}, {"status": "success"}),
            ({"make_acc": float("inf")}, {"status": "success"}),
            ({"status": "no_results_csv"}, {"status": "success"}),
            ({}, {"status": "error"}),
        ]
        for changes, training in cases:
            with self.subTest(changes=changes, training=training):
                metrics = {**self.metrics, **changes}
                self.assertFalse(self.evaluator.target_is_reached(metrics, training, "make_acc", .85))

    def test_low_accuracy_is_not_divergence_and_zero_is_a_real_metric(self) -> None:
        from agent.nodes.decision import _is_truly_diverged
        self.metrics["make_acc"] = 0.
        self.metrics["mean_acc"] = 0.
        result = self.evaluate_with_response(reached=True, verdict="diverged")
        self.assertNotEqual(result["evaluation_result"]["verdict"], "diverged")
        self.assertFalse(_is_truly_diverged(self.metrics, {"status": "success"}))
        self.assertTrue(self.evaluator.target_is_reached(self.metrics, {"status": "success"}, "make_acc", 0.))

    def test_llm_failure_preserves_deterministic_target_decision(self) -> None:
        self.metrics["make_acc"] = .9
        with patch.object(self.evaluator, "_get_llm", side_effect=RuntimeError("offline")), \
             patch.object(self.evaluator, "console", Console(file=io.StringIO())):
            result = self.evaluator.evaluator_node(self.state)
        self.assertTrue(result["target_reached"])
        self.assertIn("offline", result["evaluation_result"]["llm_error"])

    def test_413_and_timeout_fallback_use_exact_test_head_without_repeating_request(self) -> None:
        import httpx
        from openai import APIStatusError, APITimeoutError

        request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
        errors = (
            APIStatusError("Request too large", response=httpx.Response(413, request=request), body={}),
            APITimeoutError(request=request),
        )
        for error in errors:
            for make, color, reached in ((.7, .99, False), (.9, .5, True)):
                with self.subTest(error=type(error).__name__, make=make):
                    self.use_test_split(make, color)
                    llm = MagicMock()
                    llm.invoke.side_effect = error
                    with patch.object(self.evaluator, "_get_llm", return_value=llm), \
                         patch.object(self.evaluator, "console", Console(file=io.StringIO())):
                        result = self.evaluator.evaluator_node(self.state)
                    llm.invoke.assert_called_once()
                    self.assertEqual(result["target_reached"], reached)
                    self.assertEqual(result["evaluation_result"]["target_reached"], reached)
                    self.assertEqual(result["evaluation_result"]["decision_split"], "test")
                    self.assertIn(type(error).__name__, result["evaluation_result"]["llm_error"])

    def use_test_split(self, make: float = .7, color: float = .9) -> None:
        validation = copy.deepcopy(self.metrics)
        self.metrics.update(test_evidence(make, color))
        self.metrics["validation_metrics"] = validation
        self.state.update(decision_split="test", test_csv="/tmp/test.csv",
                          test_result={"status": "ok", "csv": "/tmp/test.csv",
                                       "metrics": test_evidence(make, color)})

    def test_high_validation_cannot_replace_lower_test_head_score(self) -> None:
        self.metrics["make_acc"] = .99
        self.use_test_split(.7, .99)
        result = self.evaluate_with_response(reached=True)
        self.assertFalse(result["target_reached"])
        self.assertEqual(result["evaluation_result"]["decision_split"], "test")

    def test_measured_test_target_accepts_even_when_validation_target_is_low(self) -> None:
        self.metrics["make_acc"] = .4
        self.use_test_split(.9, .5)
        result = self.evaluate_with_response(reached=False)
        self.assertTrue(result["target_reached"])

    def test_missing_test_and_wrong_exact_head_cannot_reach_target(self) -> None:
        self.use_test_split(.99, .99)
        for metric, test_result in (("missing_acc", self.state["test_result"]), ("make_acc", None)):
            with self.subTest(metric=metric, test_result=test_result):
                self.assertFalse(self.evaluator.target_is_reached(
                    self.metrics, self.state["training_result"], metric, .85,
                    "test", test_result, "/tmp/test.csv"))

    def test_partial_test_inference_wrong_denominator_and_replaced_metrics_are_rejected(self) -> None:
        self.use_test_split(.99, .99)
        cases = [
            {"evaluated_images": 99}, {"matched_images": 99}, {"full_coverage": False},
            {"test_evaluated": False}, {"split": "val"}, {"status": "test_failed"},
            {"total_images": 0, "evaluated_images": 0, "matched_images": 0},
            {"make_acc": 1.0}, {"mean_acc": float("nan")},
            {"heads": {"make": {"acc": .99, "correct": 99, "count": 99, "coverage": 1.0}}},
        ]
        for changes in cases:
            with self.subTest(changes=changes):
                self.assertFalse(self.evaluator.target_is_reached(
                    {**self.metrics, **changes}, self.state["training_result"], "make_acc", .85,
                    "test", self.state["test_result"], "/tmp/test.csv"))
        for result in ({"status": "error", "error": "image failed to decode"},
                       {"status": "ok", "csv": "/tmp/other.csv", "metrics": test_evidence(.99, .99, "/tmp/other.csv")},
                       {"status": "ok", "metrics": test_evidence(.7, .99)}):
            with self.subTest(result=result):
                self.assertFalse(self.evaluator.target_is_reached(
                    self.metrics, self.state["training_result"], "make_acc", .85,
                    "test", result, "/tmp/test.csv"))

    def test_test_failure_is_diagnosed_and_stops_before_limit_without_llm(self) -> None:
        from agent.nodes.decision import decision_node
        self.state.update(decision_split="test", max_attempts=20, test_csv="/tmp/test.csv",
                          test_result={"status": "error", "error": "unreadable test image"})
        self.metrics.update(status="test_failed", error="unreadable test image")
        with patch.object(self.evaluator, "_get_llm") as llm, \
             patch.object(self.evaluator, "console", Console(file=io.StringIO())), \
             patch("agent.nodes.decision.console", Console(file=io.StringIO())):
            evaluated = self.evaluator.evaluator_node(self.state)
            decision = decision_node({**self.state, **evaluated})
        llm.assert_not_called()
        self.assertEqual(evaluated["evaluation_result"]["verdict"], "test_failed")
        self.assertEqual(decision["stop_reason"], "error")
        self.assertFalse(decision["target_reached"])
        self.assertIn("unreadable", decision["error"])

    def test_low_test_score_continues_and_zero_uses_full_denominator(self) -> None:
        from agent.nodes.decision import decision_node
        self.use_test_split(0., .9)
        with patch("agent.nodes.decision.console", Console(file=io.StringIO())):
            result = decision_node({**self.state, "max_attempts": 20, "target_reached": True})
        self.assertEqual(result["stop_reason"], "")
        self.assertFalse(result["target_reached"])
        self.assertTrue(self.evaluator.target_is_reached(
            self.metrics, self.state["training_result"], "make_acc", 0.,
            "test", self.state["test_result"], "/tmp/test.csv"))


class ExperimentMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="resnet-memory-test-")
        self.addCleanup(self.directory.cleanup)
        self.store_dir = Path(self.directory.name)
        self.session_dir = self.store_dir / "session_test"
        self.config = {
            "arch": "resnet50", "heads": ["make", "color"],
            "train_csv": str(self.store_dir / "train.csv"),
            "val_csv": str(self.store_dir / "val.csv"),
            "out": str(self.session_dir), "name": "exp_01", "lr": .001,
        }

    def state(self, attempt: int, score: float, *, previous=None, status="success", issues=None):
        run_dir = self.session_dir / f"exp_{attempt:02d}"
        return {
            "attempt_number": attempt, "target_metric": "make_acc", "target_value": .9,
            "current_config": dict(self.config, name=f"exp_{attempt:02d}"),
            "training_result": {
                "status": status, "run_dir": str(run_dir),
                "best_weights": str(run_dir / "best_model.pth") if status == "success" else "",
                "last_weights": str(run_dir / "last_model.pth") if status == "success" else "",
                "error": "worker failure" if status != "success" else "",
            },
            "metrics": {
                "status": "ok" if status == "success" else "training_failed",
                "make_acc": score, "mean_acc": .95,
                "heads": {"make": {"acc": score, "loss": .5}},
                "nonfinite_loss": False, "nonfinite_accuracy": False,
            },
            "evaluation_result": {"verdict": "good", "issues": issues or []},
            "issues_detected": issues or [], "experiment_history": previous or [],
            "run_dir": str(run_dir), "optimisation_rationale": "Reduced learning rate.",
        }

    def save(self, state):
        with patch("agent.nodes.memory_node.console", Console(file=io.StringIO())):
            return memory_node(state)

    def test_lesson_uses_actual_before_after_and_previous_issues(self) -> None:
        first = self.save(self.state(1, .6, issues=["validation loss rising"]))
        second_state = self.state(2, .8, previous=first["experiment_history"], issues=["weak color head"])
        second_state["current_config"]["lr"] = .0003
        second = self.save(second_state)
        lessons = ExperimentStore(self.store_dir).get_lessons()
        self.assertEqual(len(lessons), 1)
        lesson = lessons[0]
        self.assertEqual(lesson["issue"], "validation loss rising")
        self.assertEqual(lesson["target_metric"], "make_acc")
        self.assertEqual(lesson["score_before"], .6)
        self.assertEqual(lesson["score_after"], .8)
        self.assertEqual(lesson["config_diff"]["lr"], {"before": .001, "after": .0003})
        self.assertTrue(lesson["improved"])
        self.assertEqual(second["best_experiment"]["attempt"], 2)

    def test_failed_high_score_never_displaces_successful_checkpoint(self) -> None:
        first = self.save(self.state(1, .6))
        failed = self.save(self.state(2, .99, previous=first["experiment_history"], status="error"))
        self.assertEqual(failed["best_experiment"]["attempt"], 1)
        self.assertIsNone(failed["experiment_history"][0]["score"])
        self.assertFalse(failed["experiment_history"][0]["improved"])
        self.assertEqual(ExperimentStore(self.store_dir).get_lessons(), [])
        best = ExperimentStore(self.store_dir).get_best_experiment(target_metric="make_acc")
        self.assertEqual(best["attempt"], 1)
        self.assertTrue(best["best_weights"].endswith("exp_01/best_model.pth"))

    def test_first_failed_attempt_does_not_create_best_experiment(self) -> None:
        result = self.save(self.state(1, .99, status="error"))
        self.assertEqual(result["best_experiment"], {})
        self.assertIsNone(ExperimentStore(self.store_dir).get_best_experiment(target_metric="make_acc"))

    def test_new_attempt_name_alone_does_not_create_an_optimisation_lesson(self) -> None:
        first = self.save(self.state(1, .6))
        self.save(self.state(2, .7, previous=first["experiment_history"]))
        self.assertEqual(ExperimentStore(self.store_dir).get_lessons(), [])

    def test_store_filters_dataset_architecture_heads_and_target_identity(self) -> None:
        self.save(self.state(1, .6))
        other = self.state(2, .98)
        other["current_config"]["train_csv"] = str(self.store_dir / "other_train.csv")
        self.save(other)
        store = ExperimentStore(self.store_dir)
        matching = store.get_best_experiment(
            target_metric="make_acc", arch="resnet50",
            train_csv=self.config["train_csv"], val_csv=self.config["val_csv"],
            heads=["make", "color"],
        )
        self.assertEqual(matching["attempt"], 1)
        self.assertIsNone(store.get_best_experiment(target_metric="mean_acc"))
        self.assertIsNone(store.get_best_experiment(target_metric="make_acc", arch="resnet18"))
        self.assertIsNone(store.get_best_experiment(target_metric="make_acc", heads=["make"]))
        self.assertIsNone(store.get_best_experiment(target_metric="make_acc", val_csv=str(self.store_dir / "other_val.csv")))

    def test_nonfinite_loss_cannot_become_best_known_configuration(self) -> None:
        state = self.state(1, .99)
        state["metrics"]["heads"]["make"]["loss"] = float("inf")
        result = self.save(state)
        self.assertEqual(result["best_experiment"], {})
        kb = json.loads((self.store_dir / "knowledge_base.json").read_text(encoding="utf-8"))
        self.assertEqual(kb["best_configs"], {})

    def make_test_state(self, attempt: int, score: float, *, previous=None, test_csv=None, validation_score=.99):
        state = self.state(attempt, validation_score, previous=previous)
        csv_path = test_csv or str(self.store_dir / "test.csv")
        state["current_config"]["test_csv"] = csv_path
        state.update(decision_split="test", test_csv=csv_path)
        validation = copy.deepcopy(state["metrics"])
        state["metrics"].update(test_evidence(score, .9, csv_path))
        state["metrics"]["validation_metrics"] = validation
        state["test_result"] = {"status": "ok", "csv": csv_path, "metrics": test_evidence(score, .9, csv_path)}
        return state

    def test_best_checkpoint_uses_test_target_instead_of_validation_maximum(self) -> None:
        first = self.save(self.make_test_state(1, .7, validation_score=.99))
        second = self.save(self.make_test_state(2, .8, previous=first["experiment_history"], validation_score=.6))
        self.assertEqual(second["best_experiment"]["attempt"], 2)
        self.assertEqual(second["best_experiment"]["score"], .8)
        self.assertEqual(second["best_experiment"]["metrics"]["validation_metrics"]["make_acc"], .6)
        self.assertEqual(second["best_experiment"]["decision_split"], "test")
        self.assertEqual(second["best_experiment"]["test_result"]["metrics"]["make_acc"], .8)

    def test_failed_or_partial_test_does_not_displace_verified_best(self) -> None:
        first = self.save(self.make_test_state(1, .7))
        for attempt, partial in enumerate((False, True), start=2):
            with self.subTest(partial=partial):
                failed = self.make_test_state(attempt, .99, previous=first["experiment_history"])
                if partial:
                    failed["metrics"]["heads"]["make"]["count"] = 99
                else:
                    failed["test_result"] = {"status": "error", "error": "decode failed"}
                saved = self.save(failed)
                self.assertEqual(saved["best_experiment"]["attempt"], 1)
                self.assertIsNone(saved["experiment_history"][0]["score"])

    def test_memory_and_knowledge_queries_separate_test_csv_and_decision_split(self) -> None:
        validation = self.save(self.state(1, .99))
        tested = self.save(self.make_test_state(2, .8, previous=validation["experiment_history"]))
        other_csv = str(self.store_dir / "other_test.csv")
        other = self.save(self.make_test_state(3, .95, test_csv=other_csv))
        self.assertEqual(tested["best_experiment"]["attempt"], 2)
        test_csv = str(self.store_dir / "test.csv")
        store = ExperimentStore(self.store_dir)
        self.assertEqual(store.get_best_experiment(target_metric="make_acc")["attempt"], 1)
        self.assertEqual(store.get_best_experiment(target_metric="make_acc", decision_split="test", test_csv=test_csv)["attempt"], 2)
        self.assertEqual(store.get_best_experiment(target_metric="make_acc", decision_split="test", test_csv=other_csv)["attempt"], 3)
        self.assertIsNone(experiment_score(validation["experiment_history"][0], "make_acc", "test", test_csv))
        self.assertIsNone(experiment_score(other["experiment_history"][0], "make_acc", "test", test_csv))
        knowledge = json.loads((self.store_dir / "knowledge_base.json").read_text(encoding="utf-8"))
        self.assertEqual(len(knowledge["best_configs"]), 3)
        if _FRAMEWORK_AVAILABLE:
            from tools.memory_tools import get_best_known_config, get_attempt_history_summary, load_knowledge_base
            arguments = {"runs_dir": str(self.store_dir), "decision_split": "test", "test_csv": test_csv}
            best = json.loads(get_best_known_config.invoke({**arguments, "target_metric": "make_acc"}))
            history = json.loads(get_attempt_history_summary.invoke(arguments))
            filtered = json.loads(load_knowledge_base.invoke(arguments))
            self.assertEqual(best["score"], .8)
            self.assertEqual([entry["attempt"] for entry in history["summary"]], [2])
            self.assertEqual([entry["attempt"] for entry in filtered["experiments"]], [2])
            self.assertEqual(len(filtered["best_configs"]), 1)

    def test_optimisation_lesson_records_the_test_context_and_scores(self) -> None:
        first = self.save(self.make_test_state(1, .6, previous=[], validation_score=.99))
        second_state = self.make_test_state(2, .8, previous=first["experiment_history"], validation_score=.5)
        second_state["current_config"]["lr"] = .0003
        self.save(second_state)
        store = ExperimentStore(self.store_dir)
        self.assertEqual(store.get_lessons(decision_split="val"), [])
        lesson = store.get_lessons(decision_split="test", test_csv=str(self.store_dir / "test.csv"))[0]
        self.assertEqual((lesson["score_before"], lesson["score_after"]), (.6, .8))
        self.assertEqual(lesson["decision_split"], "test")

    def make_model_state(self, attempt: int, score: float, arch: str, *, previous=None, test_csv=None):
        state = self.make_test_state(attempt, score, previous=previous, test_csv=test_csv)
        state["current_config"].update(arch=arch, weights=str(self.store_dir / f"{arch}.pth"))
        return state

    def test_stronger_previous_architecture_remains_best_and_exports_its_own_config(self) -> None:
        first = self.save(self.make_model_state(1, .9, "resnet18"))
        original = copy.deepcopy(first["experiment_history"][0])
        second_state = self.make_model_state(2, .8, "resnet50", previous=first["experiment_history"])
        second = self.save(second_state)
        self.assertEqual(second["best_experiment"], original)
        current = second["experiment_history"][0]
        self.assertFalse(current["improved"])
        self.assertEqual(current["arch"], "resnet50")
        self.assertEqual(current["weights"], second_state["current_config"]["weights"])
        self.assertEqual(current["config"], second_state["current_config"])
        self.assertEqual(first["experiment_history"][0], original)
        checkpoint = Path(original["best_weights"])
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.touch()
        from agent.nodes import onnx_exporter
        with patch.object(onnx_exporter, "export_resnet_onnx", return_value="selected.onnx") as export, \
             patch.object(onnx_exporter, "console", Console(file=io.StringIO())):
            result = onnx_exporter.onnx_exporter_node({**second_state, **second, "export_format": "onnx"})
        export.assert_called_once_with(str(checkpoint), original["config"])
        self.assertEqual(result["onnx_path"], "selected.onnx")

    def test_stronger_new_architecture_becomes_best_without_changing_previous_identity(self) -> None:
        first = self.save(self.make_model_state(1, .6, "resnet18"))
        original = copy.deepcopy(first["experiment_history"][0])
        second = self.save(self.make_model_state(2, .9, "resnet50", previous=first["experiment_history"]))
        self.assertEqual(second["best_experiment"]["attempt"], 2)
        self.assertEqual(second["best_experiment"]["config"]["arch"], "resnet50")
        self.assertTrue(second["experiment_history"][0]["improved"])
        self.assertEqual(first["experiment_history"][0], original)

    def test_architecture_neutral_comparison_still_rejects_other_task_or_split(self) -> None:
        first = self.save(self.make_model_state(1, .9, "resnet18"))
        for attempt, identity in enumerate(
            ("train_csv", "val_csv", "heads", "target_metric", "test_csv", "decision_split"), start=2,
        ):
            with self.subTest(identity=identity):
                state = self.make_model_state(attempt, .8, "resnet50", previous=first["experiment_history"])
                if identity == "test_csv":
                    state = self.make_model_state(attempt, .8, "resnet50", previous=first["experiment_history"],
                                                  test_csv=str(self.store_dir / "other_test.csv"))
                elif identity in {"train_csv", "val_csv"}:
                    state["current_config"][identity] = str(self.store_dir / f"other_{identity}.csv")
                elif identity == "heads":
                    state["current_config"]["heads"] = ["make"]
                    for evidence in (state["metrics"], state["test_result"]["metrics"]):
                        evidence["heads"].pop("color")
                        evidence.pop("color_acc")
                        evidence["mean_acc"] = evidence["make_acc"]
                elif identity == "target_metric":
                    state["target_metric"] = "color_acc"
                elif identity == "decision_split":
                    state["decision_split"] = "val"
                    state["metrics"] = state["metrics"]["validation_metrics"]
                saved = self.save(state)
                self.assertEqual(saved["best_experiment"]["attempt"], attempt)
                self.assertTrue(saved["experiment_history"][0]["improved"])
        self.assertEqual(ExperimentStore(self.store_dir).get_lessons(), [])

    def test_persistent_best_configs_and_architecture_filters_remain_separate(self) -> None:
        first = self.save(self.make_model_state(1, .9, "resnet18"))
        second = self.save(self.make_model_state(2, .8, "resnet50", previous=first["experiment_history"]))
        store = ExperimentStore(self.store_dir)
        old_record, new_record = first["experiment_history"][0], second["experiment_history"][0]
        self.assertNotEqual(store._context_key(old_record), store._context_key(new_record))
        self.assertEqual(store._comparison_key(old_record), store._comparison_key(new_record))
        legacy = copy.deepcopy(old_record)
        legacy.pop("arch")
        legacy.pop("weights")
        self.assertEqual(store._context_key(legacy), store._context_key(old_record))
        self.assertEqual(store._comparison_key(legacy), store._comparison_key(old_record))
        arguments = {"target_metric": "make_acc", "decision_split": "test",
                     "test_csv": str(self.store_dir / "test.csv")}
        self.assertEqual(store.get_best_experiment(**arguments, arch="resnet18")["attempt"], 1)
        self.assertEqual(store.get_best_experiment(**arguments, arch="resnet50")["attempt"], 2)
        knowledge = json.loads((self.store_dir / "knowledge_base.json").read_text(encoding="utf-8"))
        self.assertEqual(len(knowledge["best_configs"]), 2)
        self.assertEqual([(entry["arch"], entry["weights"]) for entry in knowledge["experiments"]],
                         [("resnet18", old_record["config"]["weights"]), ("resnet50", new_record["config"]["weights"])])

    def test_architecture_switch_lesson_records_both_models_and_measured_outcome(self) -> None:
        first_state = self.make_model_state(1, .9, "resnet18")
        first_state["issues_detected"] = ["Weak performance on uncommon makes"]
        first = self.save(first_state)
        next_state = self.make_model_state(2, .8, "resnet50", previous=first["experiment_history"])
        next_state["optimisation_rationale"] = "Compare the larger ResNet50 backbone."
        self.save(next_state)
        lesson = ExperimentStore(self.store_dir).get_lessons()[0]
        self.assertEqual(lesson["issue"], "Weak performance on uncommon makes")
        self.assertEqual(lesson["config_diff"]["arch"], {"before": "resnet18", "after": "resnet50"})
        self.assertEqual(lesson["config_diff"]["weights"], {
            "before": first_state["current_config"]["weights"], "after": next_state["current_config"]["weights"]})
        self.assertEqual(lesson["model_before"], {"arch": "resnet18", "weights": first_state["current_config"]["weights"]})
        self.assertEqual(lesson["model_after"], {"arch": "resnet50", "weights": next_state["current_config"]["weights"]})
        self.assertEqual((lesson["score_before"], lesson["score_after"]), (.9, .8))
        self.assertFalse(lesson["improved"])


if __name__ == "__main__":
    unittest.main()
