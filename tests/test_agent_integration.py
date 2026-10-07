"""Exercise the actual graph without API calls or GPU training."""
from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from openai import APIStatusError, APITimeoutError
from PIL import Image
from langchain_core.messages import AIMessage

from agent import graph
from agent.llm import parse_json_response
from agent.nodes import config_generator, config_optimizer, evaluator, reporter
from memory.experiment_store import ExperimentStore
from tools.config_tools import sanitize_resnet_config


class AgentIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        Image.new("RGB", (16, 16), "white").save(self.root / "image.png")
        for split in ("train", "val"):
            with (self.root / f"{split}.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["root_folder", "image_name", "make", "color"])
                writer.writerow([str(self.root), "image.png", "Toyota", "white"])
        with (self.root / "test.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["root_folder", "image_name", "make", "color"])
            for index in range(10):
                name = f"test_{index}.png"
                Image.new("RGB", (16, 16), "white").save(self.root / name)
                writer.writerow([str(self.root), name, "Toyota", "white"])
        (self.root / "weights.pth").write_bytes(b"validation fixture; trainer is mocked")
        self.state = {
            "train_csv": str(self.root / "train.csv"),
            "val_csv": str(self.root / "val.csv"),
            "heads": ["make", "color"],
            "weights_path": str(self.root / "weights.pth"),
            "model_type": "resnet18", "nproc_per_node": 1,
            "target_metric": "mean_acc", "target_value": 0.99,
            "max_attempts": 5, "export_format": "none", "interactive": False,
            "decision_split": "val",
            "session_dir": str(self.root / "runs" / "session"),
        }

    @staticmethod
    def fake_train(state):
        config = state["current_config"]
        attempt = state.get("attempt_number", 0) + 1
        run_dir = Path(config["out"]) / config["name"]
        run_dir.mkdir(parents=True, exist_ok=False)
        make, color = 0.9, 0.2 + 0.1 * attempt
        with (run_dir / "training_metrics.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Epoch", "Split", "TotalLoss", "MeanAcc", "make_Loss", "make_Acc", "color_Loss", "color_Acc"])
            writer.writerow([1, "train", 0.4, 0.9, 0.2, 0.9, 0.2, 0.9])
            writer.writerow([1, "val", 0.6, (make + color) / 2, 0.3, make, 0.3, color])
        best, last = run_dir / "best_model.pth", run_dir / "last_model.pth"
        best.touch()
        last.touch()
        return {
            "attempt_number": attempt, "run_dir": str(run_dir),
            "training_result": {"status": "success", "run_dir": str(run_dir),
                                "best_weights": str(best), "last_weights": str(last), "config": config},
        }

    def mocked_services(self, stack):
        stack.enter_context(patch.object(config_optimizer, "discover_pretrained_weights", return_value=[]))

        def optimize(messages, **kwargs):
            context = json.loads(messages[-1].content)
            return AIMessage(content=json.dumps({
                "updated_config": {"lr": context["current_config"]["lr"] * 0.9},
                "rationale": "Reduce learning rate to refine validation accuracy.",
            }))

        stack.enter_context(patch.object(config_optimizer, "get_llm", return_value=SimpleNamespace(invoke=optimize)))
        # The LLM wrongly claims every attempt reached the target.
        stack.enter_context(patch.object(evaluator, "_get_llm", return_value=SimpleNamespace(
            invoke=lambda messages: AIMessage(content=json.dumps({
                "verdict": "underfitting", "target_reached": True,
                "issues": ["Target accuracy remains low"], "recommendations": [], "reasoning": "Synthetic evaluation",
            }))
        )))
        stack.enter_context(patch.object(reporter, "_get_llm", side_effect=RuntimeError("No API calls in tests")))
        stack.enter_context(patch.object(graph, "trainer_node", side_effect=self.fake_train))

    def fake_test(self, state, scores):
        """Synthetic inference evidence, with counts for every fixture test row."""
        attempt = state["attempt_number"]
        score = scores[attempt - 1]
        validation = state["metrics"]
        output_dir = Path(state["run_dir"]) / "test_evaluation"
        output_dir.mkdir()
        evidence = {"status": "ok", "decision_split": "test", "split": "test",
                    "test_evaluated": True, "full_coverage": True,
                    "total_images": 10, "evaluated_images": 10, "matched_images": 10,
                    "mean_acc": score, "checkpoint": state["training_result"]["best_weights"],
                    "test_csv": state["test_csv"], "output_dir": str(output_dir),
                    "heads": {head: {"acc": score, "correct": round(score * 10),
                                     "count": 10, "coverage": 1.0}
                              for head in state["heads"]}}
        evidence.update({f"{head}_acc": score for head in state["heads"]})
        (output_dir / "summary.json").write_text(json.dumps(evidence))
        return {"metrics": {**validation, **evidence, "validation_metrics": validation},
                "validation_metrics": validation,
                "test_result": {**evidence, "metrics": evidence, "csv": state["test_csv"]},
                "error": ""}

    def test_first_attempt_uses_yaml_before_llm_tunes_the_second_attempt(self):
        training_defaults = {
            "arch": "resnet50", "epochs": 30, "batch_size": 8,
            "num_workers": 0, "img_size": [64, 96], "lr": 0.0003,
            "weight_decay": 0.0002, "gamma": 1.0, "patience": 10,
            "monitor": "mean", "out": "runs/resnet", "name": "experiment",
        }
        config_dir = self.root / "config"
        config_dir.mkdir()
        # JSON is valid YAML, so this fixture exercises the production YAML loader.
        (config_dir / "defaults.yaml").write_text(json.dumps({"training": training_defaults}))
        state = {**self.state, "decision_split": "test",
                 "test_csv": str(self.root / "test.csv"),
                 "target_value": 0.85, "max_attempts": 2}
        events = []
        observed_configs = []

        def train(state):
            attempt = state.get("attempt_number", 0) + 1
            if attempt == 1:
                self.assertEqual(events, [])
                self.assertEqual(state.get("messages"), [])
            events.append(f"train{attempt}")
            observed_configs.append(json.loads(json.dumps(state["current_config"])))
            return self.fake_train(state)

        def test_checkpoint(state):
            attempt = state["attempt_number"]
            self.assertEqual(events[-1], f"train{attempt}")
            result = self.fake_test(state, [0.3, 0.9])
            events.append(f"test{attempt}")
            return result

        def evaluate(messages):
            context = json.loads(messages[-1].content)
            attempt = context["attempt"]
            self.assertEqual(events[-1], f"test{attempt}")
            events.append(f"evaluate{attempt}")
            return AIMessage(content=json.dumps({
                "verdict": "underfitting", "target_reached": False,
                "issues": [], "recommendations": [], "reasoning": "Measured fixture results.",
            }))

        def optimize(messages):
            context = json.loads(messages[-1].content)
            self.assertEqual(events[-1], "evaluate1")
            self.assertEqual(context["current_config"]["epochs"], 30)
            self.assertEqual(context["metrics"]["mean_acc"], 0.3)
            events.append("optimize1")
            return AIMessage(content=json.dumps({
                "updated_config": {"epochs": 40},
                "rationale": "Use more epochs after the measured first attempt.",
            }))

        with ExitStack() as stack, redirect_stdout(io.StringIO()):
            self.mocked_services(stack)
            stack.enter_context(patch.object(config_generator, "PROJECT_ROOT", self.root))
            unexpected_llm = stack.enter_context(patch("agent.llm.get_llm",
                side_effect=AssertionError("LLM requests must follow training and test evaluation")))
            previous_best = stack.enter_context(patch.object(ExperimentStore, "get_best_experiment",
                side_effect=AssertionError("Previous experiments must not replace the YAML baseline")))
            stack.enter_context(patch.object(graph, "trainer_node", side_effect=train))
            stack.enter_context(patch.object(graph, "test_checkpoint_node", side_effect=test_checkpoint))
            stack.enter_context(patch.object(evaluator, "_get_llm",
                return_value=SimpleNamespace(invoke=evaluate)))
            optimizer = stack.enter_context(patch.object(config_optimizer, "get_llm",
                return_value=SimpleNamespace(invoke=optimize)))
            final = graph.compile_graph().invoke(state, config={"recursion_limit": 70})

        unexpected_llm.assert_not_called()
        previous_best.assert_not_called()
        optimizer.assert_called_once()
        self.assertEqual(events, ["train1", "test1", "evaluate1", "optimize1",
                                  "train2", "test2", "evaluate2"])
        first, second = observed_configs
        for key in ("epochs", "batch_size", "num_workers", "img_size", "lr",
                    "weight_decay", "gamma", "patience"):
            self.assertEqual(first[key], training_defaults[key], key)
            self.assertEqual(second[key], 40 if key == "epochs" else training_defaults[key], key)
        self.assertEqual(first["arch"], state["model_type"])
        self.assertEqual(first["weights"], state["weights_path"])
        self.assertEqual(first["out"], state["session_dir"])
        self.assertEqual(first["name"], "exp_01")
        self.assertEqual(second["name"], "exp_02")
        self.assertEqual(final["config_history"][0]["config"], first)
        self.assertEqual(final["config_history"][1]["config"], second)
        self.assertEqual(final["attempt_number"], 2)
        self.assertEqual(final["stop_reason"], "target_reached")

    def test_test_accuracy_controls_retraining_despite_high_validation_and_llm_success(self):
        state = {**self.state, "decision_split": "test", "test_csv": str(self.root / "test.csv"),
                 "target_value": 0.85}

        def high_validation_train(state):
            result = self.fake_train(state)
            path = Path(result["run_dir"]) / "training_metrics.csv"
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                row.update({"MeanAcc": 0.99, "make_Acc": 0.99, "color_Acc": 0.99})
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            return result

        with ExitStack() as stack, redirect_stdout(io.StringIO()):
            self.mocked_services(stack)
            stack.enter_context(patch.object(graph, "trainer_node", side_effect=high_validation_train))
            test_node = stack.enter_context(patch.object(graph, "test_checkpoint_node",
                side_effect=lambda state: self.fake_test(state, [0.3, 0.9])))
            final = graph.compile_graph().invoke(state, config={"recursion_limit": 70})
        self.assertEqual(test_node.call_count, 2)
        self.assertEqual(final["attempt_number"], 2)
        self.assertEqual(final["stop_reason"], "target_reached")
        self.assertEqual(final["experiment_history"][0]["metrics"]["mean_acc"], 0.3)
        self.assertFalse(final["experiment_history"][0]["target_reached"])
        self.assertEqual(final["best_experiment"]["score"], 0.9)
        self.assertEqual(final["report_summary"]["decision_split"], "test")
        self.assertEqual(final["report_summary"]["best_validation_metrics"]["mean_acc"], 0.99)
        self.assertIn("Mean test accuracy", final["final_report"])

    def test_best_attempt_uses_test_score_when_validation_improves_but_test_declines(self):
        state = {**self.state, "decision_split": "test", "test_csv": str(self.root / "test.csv"),
                 "max_attempts": 3}
        with ExitStack() as stack, redirect_stdout(io.StringIO()):
            self.mocked_services(stack)
            stack.enter_context(patch.object(graph, "test_checkpoint_node",
                side_effect=lambda state: self.fake_test(state, [0.6, 0.5, 0.4])))
            final = graph.compile_graph().invoke(state, config={"recursion_limit": 70})
        self.assertEqual(final["stop_reason"], "max_attempts")
        self.assertEqual(final["best_experiment"]["attempt"], 1)
        self.assertEqual(final["best_experiment"]["score"], 0.6)
        self.assertEqual(final["report_summary"]["best_model_path"],
                         final["experiment_history"][0]["best_weights"])
        self.assertGreater(final["experiment_history"][-1]["metrics"]["validation_metrics"]["mean_acc"],
                           final["experiment_history"][0]["metrics"]["validation_metrics"]["mean_acc"])

    def test_second_attempt_uses_selected_backbone_and_exports_the_earlier_best(self):
        weights34 = self.root / "resnet34.pth"
        weights34.write_bytes(b"catalog is mocked; no real weights are loaded")
        available = [
            {"arch": "resnet18", "weights": self.state["weights_path"]},
            {"arch": "resnet34", "weights": str(weights34)},
        ]
        state = {**self.state, "decision_split": "test", "test_csv": str(self.root / "test.csv"),
                 "max_attempts": 2, "target_value": 0.99, "export_format": "onnx"}
        observed = []

        def train(current_state):
            observed.append({key: current_state[key] for key in ("model_type", "weights_path", "current_config")})
            if len(observed) == 1:
                self.assertEqual(current_state["current_config"]["arch"], "resnet18")
                self.assertEqual(current_state.get("messages"), [])
            else:
                self.assertEqual(current_state["model_type"], "resnet34")
                self.assertEqual(current_state["weights_path"], str(weights34))
                self.assertEqual(current_state["current_config"]["arch"], "resnet34")
                self.assertEqual(current_state["current_config"]["weights"], str(weights34))
            return self.fake_train(current_state)

        invoke = Mock(return_value=AIMessage(content=json.dumps({
            "updated_config": {"arch": "resnet34", "weights": str(weights34)},
            "rationale": "Compare the locally available ResNet34 backbone.",
        })))
        with ExitStack() as stack, redirect_stdout(io.StringIO()):
            self.mocked_services(stack)
            stack.enter_context(patch.object(config_optimizer, "discover_pretrained_weights", return_value=available))
            stack.enter_context(patch.object(config_optimizer, "get_llm", return_value=SimpleNamespace(invoke=invoke)))
            stack.enter_context(patch.object(graph, "trainer_node", side_effect=train))
            stack.enter_context(patch.object(graph, "test_checkpoint_node",
                side_effect=lambda current: self.fake_test(current, [0.9, 0.8])))
            export = stack.enter_context(patch.object(graph, "onnx_exporter_node", return_value={
                "onnx_path": str(self.root / "selected-best.onnx")}))
            final = graph.compile_graph().invoke(state, config={"recursion_limit": 70})
        invoke.assert_called_once()
        self.assertEqual(len(observed), 2)
        self.assertEqual(final["attempt_number"], 2)
        self.assertEqual(final["stop_reason"], "max_attempts")
        self.assertEqual([record["config"]["arch"] for record in final["config_history"]], ["resnet18", "resnet34"])
        self.assertEqual([record["config"]["weights"] for record in final["experiment_history"]],
                         [state["weights_path"], str(weights34)])
        self.assertEqual(final["model_type"], "resnet34")
        self.assertEqual(final["weights_path"], str(weights34))
        self.assertEqual(final["best_experiment"]["attempt"], 1)
        self.assertEqual(final["best_experiment"]["score"], 0.9)
        self.assertEqual(final["best_experiment"]["config"]["arch"], "resnet18")
        export.assert_called_once()
        selected = export.call_args.args[0]["best_experiment"]
        self.assertEqual(selected["config"]["arch"], "resnet18")
        self.assertEqual(selected["best_weights"], final["report_summary"]["best_model_path"])
        self.assertEqual(final["report_summary"]["best_config"]["arch"], "resnet18")
        self.assertEqual(final["report_summary"]["best_config"]["weights"], state["weights_path"])

    def test_invalid_selected_architecture_weights_pair_stops_before_another_training(self):
        available = [{"arch": "resnet18", "weights": self.state["weights_path"]}]
        state = {**self.state, "decision_split": "test", "test_csv": str(self.root / "test.csv"),
                 "max_attempts": 2, "export_format": "onnx"}
        invoke = Mock(return_value=AIMessage(content=json.dumps({
            "updated_config": {"arch": "resnet34", "weights": self.state["weights_path"]},
            "rationale": "This proposed pair must be rejected.",
        })))
        with ExitStack() as stack, redirect_stdout(io.StringIO()):
            self.mocked_services(stack)
            stack.enter_context(patch.object(config_optimizer, "discover_pretrained_weights", return_value=available))
            stack.enter_context(patch.object(config_optimizer, "get_llm", return_value=SimpleNamespace(invoke=invoke)))
            train = stack.enter_context(patch.object(graph, "trainer_node", side_effect=self.fake_train))
            stack.enter_context(patch.object(graph, "test_checkpoint_node",
                side_effect=lambda current: self.fake_test(current, [0.6])))
            export = stack.enter_context(patch.object(graph, "onnx_exporter_node", return_value={
                "onnx_path": str(self.root / "best-before-rejected-model.onnx")}))
            final = graph.compile_graph().invoke(state, config={"recursion_limit": 70})
        train.assert_called_once()
        invoke.assert_called_once()
        export.assert_called_once()
        self.assertEqual(final["optimisation_status"], "error")
        self.assertEqual(final["stop_reason"], "error")
        self.assertEqual(final["attempt_number"], 1)
        self.assertEqual(len(final["config_history"]), 1)
        self.assertEqual(final["current_config"]["arch"], "resnet18")
        self.assertEqual(final["current_config"]["weights"], state["weights_path"])
        self.assertEqual(final["best_experiment"]["score"], 0.6)
        self.assertEqual(final["report_summary"]["best_config"]["arch"], "resnet18")
        self.assertTrue(Path(final["report_summary"]["best_model_path"]).is_file())

    def test_failed_required_test_stops_without_validation_fallback(self):
        state = {**self.state, "decision_split": "test", "test_csv": str(self.root / "test.csv"),
                 "target_value": 0}

        def failed_test(state):
            return {"metrics": {"status": "test_failed", "error": "Test image unreadable",
                                "decision_split": "test", "split": "test", "test_evaluated": False},
                    "test_result": {"status": "error", "error": "Test image unreadable"},
                    "error": "Test image unreadable"}

        with ExitStack() as stack, redirect_stdout(io.StringIO()):
            self.mocked_services(stack)
            stack.enter_context(patch.object(graph, "test_checkpoint_node", side_effect=failed_test))
            final = graph.compile_graph().invoke(state, config={"recursion_limit": 70})
        self.assertEqual(final["attempt_number"], 1)
        self.assertEqual(final["stop_reason"], "error")
        self.assertFalse(final["target_reached"])
        self.assertFalse(final["best_experiment"])
        self.assertIn("Test image unreadable", final["final_report"])

    def test_optimizer_quota_error_preserves_best_and_does_not_train_again(self):
        state = {**self.state, "decision_split": "test", "test_csv": str(self.root / "test.csv"),
                 "export_format": "onnx"}
        error = RuntimeError("429: output token limit 1000, requested 1266")
        with ExitStack() as stack, redirect_stdout(io.StringIO()):
            self.mocked_services(stack)
            train = stack.enter_context(patch.object(graph, "trainer_node", side_effect=self.fake_train))
            stack.enter_context(patch.object(graph, "test_checkpoint_node",
                side_effect=lambda state: self.fake_test(state, [0.6])))
            invoke = Mock(side_effect=error)
            stack.enter_context(patch.object(config_optimizer, "get_llm", return_value=SimpleNamespace(invoke=invoke)))
            export = stack.enter_context(patch.object(graph, "onnx_exporter_node", return_value={
                "onnx_path": str(self.root / 'best.onnx')}))
            final = graph.compile_graph().invoke(state, config={"recursion_limit": 70})
        self.assertEqual(train.call_count, 1)
        invoke.assert_called_once()
        self.assertEqual(final["attempt_number"], 1)
        self.assertEqual(final["stop_reason"], "error")
        self.assertEqual(final["optimisation_status"], "error")
        self.assertFalse(final["target_reached"])
        self.assertEqual(final["best_experiment"]["score"], 0.6)
        self.assertEqual(final["current_config"]["name"], "exp_01")
        self.assertEqual(len(final["config_history"]), 1)
        export.assert_called_once()
        self.assertEqual(export.call_args.args[0]["best_experiment"]["best_weights"],
                         final["report_summary"]["best_model_path"])
        self.assertTrue(Path(final["report_summary"]["best_model_path"]).is_file())
        self.assertIn("output token limit", final["final_report"])

    def test_evaluator_quota_error_uses_measured_score_and_reports_the_error(self):
        state = {**self.state, "decision_split": "test", "test_csv": str(self.root / "test.csv"),
                 "target_value": 0.5}
        with ExitStack() as stack, redirect_stdout(io.StringIO()):
            self.mocked_services(stack)
            train = stack.enter_context(patch.object(graph, "trainer_node", side_effect=self.fake_train))
            stack.enter_context(patch.object(graph, "test_checkpoint_node",
                side_effect=lambda state: self.fake_test(state, [0.6])))
            invoke = Mock(side_effect=RuntimeError("429: output token allowance exceeded"))
            stack.enter_context(patch.object(evaluator, "_get_llm", return_value=SimpleNamespace(invoke=invoke)))
            output = stack.enter_context(patch.object(evaluator, "console"))
            final = graph.compile_graph().invoke(state, config={"recursion_limit": 70})
        self.assertEqual(train.call_count, 1)
        invoke.assert_called_once()
        self.assertEqual(final["stop_reason"], "target_reached")
        self.assertTrue(final["target_reached"])
        self.assertEqual(final["best_experiment"]["score"], 0.6)
        self.assertIn("429", final["evaluation_result"]["llm_error"])
        notices = [str(call.args[0]) for call in output.print.call_args_list if "429" in str(call.args[0])]
        self.assertEqual(len(notices), 1)
        self.assertIn("429", final["final_report"])

    @staticmethod
    def bounded_request_errors():
        request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
        return (
            APIStatusError(
                "Request too large: input token limit 7000, requested 8891",
                response=httpx.Response(413, request=request),
                body={"error": {"code": "rate_limit_exceeded"}},
            ),
            APITimeoutError(request=request),
        )

    def test_optimizer_413_and_timeout_end_after_one_request_and_preserve_checkpoint(self):
        for index, error in enumerate(self.bounded_request_errors()):
            with self.subTest(error=type(error).__name__):
                state = {
                    **self.state, "decision_split": "test",
                    "test_csv": str(self.root / "test.csv"),
                    "session_dir": str(self.root / "runs" / f"optimizer_error_{index}"),
                    "max_attempts": 20,
                }
                invoke = Mock(side_effect=error)
                with ExitStack() as stack, redirect_stdout(io.StringIO()):
                    self.mocked_services(stack)
                    train = stack.enter_context(patch.object(graph, "trainer_node", side_effect=self.fake_train))
                    stack.enter_context(patch.object(graph, "test_checkpoint_node",
                        side_effect=lambda current: self.fake_test(current, [0.6])))
                    stack.enter_context(patch.object(config_optimizer, "get_llm",
                        return_value=SimpleNamespace(invoke=invoke)))
                    final = graph.compile_graph().invoke(state, config={"recursion_limit": 70})
                invoke.assert_called_once()
                train.assert_called_once()
                self.assertEqual(final["attempt_number"], 1)
                self.assertEqual(final["stop_reason"], "error")
                self.assertEqual(final["optimisation_status"], "error")
                self.assertFalse(final["target_reached"])
                self.assertEqual(final["best_experiment"]["score"], 0.6)
                self.assertEqual(len(final["config_history"]), 1)
                self.assertEqual(final["current_config"]["name"], "exp_01")
                self.assertTrue(Path(final["report_summary"]["best_model_path"]).is_file())
                self.assertIn(type(error).__name__, final["final_report"])

    def test_evaluator_413_and_timeout_use_measured_test_score_after_one_request(self):
        for index, error in enumerate(self.bounded_request_errors()):
            with self.subTest(error=type(error).__name__):
                state = {
                    **self.state, "decision_split": "test",
                    "test_csv": str(self.root / "test.csv"), "target_value": 0.5,
                    "session_dir": str(self.root / "runs" / f"evaluator_error_{index}"),
                }
                invoke = Mock(side_effect=error)
                with ExitStack() as stack, redirect_stdout(io.StringIO()):
                    self.mocked_services(stack)
                    train = stack.enter_context(patch.object(graph, "trainer_node", side_effect=self.fake_train))
                    stack.enter_context(patch.object(graph, "test_checkpoint_node",
                        side_effect=lambda current: self.fake_test(current, [0.6])))
                    stack.enter_context(patch.object(evaluator, "_get_llm",
                        return_value=SimpleNamespace(invoke=invoke)))
                    final = graph.compile_graph().invoke(state, config={"recursion_limit": 70})
                invoke.assert_called_once()
                train.assert_called_once()
                self.assertEqual(final["stop_reason"], "target_reached")
                self.assertTrue(final["target_reached"])
                self.assertEqual(final["best_experiment"]["score"], 0.6)
                self.assertIn(type(error).__name__, final["evaluation_result"]["llm_error"])

    def test_five_attempt_loop_preserves_history_and_reports_without_api(self):
        with ExitStack() as stack, redirect_stdout(io.StringIO()):
            self.mocked_services(stack)
            final = graph.compile_graph().invoke(self.state, config={"recursion_limit": 60})
        self.assertEqual(final["stop_reason"], "max_attempts")
        self.assertEqual(final["attempt_number"], 5)
        self.assertEqual(len(final["experiment_history"]), 5)
        self.assertEqual(len(final["config_history"]), 5)
        self.assertEqual(final["best_experiment"]["attempt"], 5)
        self.assertEqual(final["session_dir"], self.state["session_dir"])
        self.assertTrue(Path(final["report_summary"]["report_md"]).is_file())
        self.assertIn("color", final["final_report"])
        for experiment in final["experiment_history"]:
            self.assertEqual(experiment["config"]["weights"], self.state["weights_path"])
        self.assertTrue((self.root / "runs" / "knowledge_base.json").is_file())

    def test_head_target_ignores_high_mean_and_stops_on_exact_head(self):
        state = {**self.state, "target_metric": "color_acc", "target_value": 0.49}
        with ExitStack() as stack, redirect_stdout(io.StringIO()):
            self.mocked_services(stack)
            final = graph.compile_graph().invoke(state, config={"recursion_limit": 60})
        self.assertEqual(final["stop_reason"], "target_reached")
        self.assertEqual(final["attempt_number"], 3)
        self.assertEqual(final["current_config"]["monitor"], "color")
        self.assertAlmostEqual(final["best_experiment"]["score"], 0.5)

    def test_invalid_csv_routes_to_report_without_training(self):
        state = {**self.state, "val_csv": str(self.root / "missing.csv")}
        with ExitStack() as stack, redirect_stdout(io.StringIO()):
            stack.enter_context(patch.object(reporter, "_get_llm", side_effect=RuntimeError("offline")))
            trainer = stack.enter_context(patch.object(graph, "trainer_node"))
            final = graph.compile_graph().invoke(state)
        trainer.assert_not_called()
        self.assertFalse(final["is_valid"])
        self.assertEqual(final["report_summary"]["stop_reason"], "validation_failed")
        self.assertTrue(Path(final["report_summary"]["report_md"]).is_file())


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        discovery_patch = patch.object(config_optimizer, "discover_pretrained_weights", return_value=[])
        discovery_patch.start()
        self.addCleanup(discovery_patch.stop)

    def base_config(self):
        return {
            "arch": "resnet18", "weights": "/tmp/weights.pth",
            "train_csv": "/tmp/train.csv", "val_csv": "/tmp/val.csv", "data_root": None,
            "heads": ["make", "color"], "epochs": 30, "batch_size": 32,
            "num_workers": 4, "img_size": [224, 224], "lr": 1e-4,
            "weight_decay": 1e-4, "gamma": 2.0, "patience": 7,
            "monitor": "color", "head_loss_weights": {}, "alpha_json": None,
            "nproc_per_node": 1, "out": "/tmp/session", "name": "exp_01",
        }

    def test_invalid_gpu_parameters_rejected_and_unsupported_keys_filtered(self):
        config, removed = sanitize_resnet_config({**self.base_config(), "mosaic": 1})
        self.assertEqual(removed, ["mosaic"])
        self.assertEqual(config["head_loss_weights"], {"make": 1.0, "color": 1.0})
        for key, value in [("lr", float("nan")), ("epochs", True), ("batch_size", -1),
                           ("img_size", [0, 224]), ("head_loss_weights", {"make": 0, "color": 0})]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                sanitize_resnet_config({**self.base_config(), key: value})

    def test_optimizer_cannot_replace_dataset_heads_monitor_or_gpu_count(self):
        original = self.base_config()
        response = AIMessage(content=json.dumps({
            "updated_config": {"train_csv": "/tmp/other.csv", "val_csv": "/tmp/other-val.csv",
                               "data_root": "/tmp/other-images", "heads": ["other"],
                               "monitor": "mean", "nproc_per_node": 8, "lr": 2e-4},
            "rationale": "Tune learning rate.",
        }))
        with patch.object(config_optimizer, "get_llm", return_value=SimpleNamespace(invoke=lambda *args, **kwargs: response)), \
             redirect_stdout(io.StringIO()):
            result = config_optimizer.config_optimizer_node({"current_config": original, "attempt_number": 1})
        updated = result["current_config"]
        for key in ("arch", "weights", "train_csv", "val_csv", "data_root", "heads", "monitor", "nproc_per_node"):
            self.assertEqual(updated[key], original[key])
        self.assertEqual(updated["lr"], 2e-4)
        self.assertEqual(updated["name"], "exp_02")
        self.assertEqual(original["name"], "exp_01")

    def test_optimizer_failure_reports_once_and_keeps_attempt_configuration(self):
        original = self.base_config()
        invoke = Mock(side_effect=RuntimeError("429: output token limit"))
        with patch.object(config_optimizer, "get_llm", return_value=SimpleNamespace(
                 invoke=invoke)), \
             patch.object(config_optimizer, "console") as output:
            result = config_optimizer.config_optimizer_node({"current_config": original, "attempt_number": 1})
        self.assertEqual(result["optimisation_status"], "error")
        invoke.assert_called_once()
        self.assertEqual(result["stop_reason"], "error")
        self.assertNotIn("current_config", result)
        self.assertNotIn("config_history", result)
        self.assertEqual(original["name"], "exp_01")
        failure_notices = [str(call.args[0]) for call in output.print.call_args_list
                           if "429" in str(call.args[0])]
        self.assertEqual(len(failure_notices), 1)
        self.assertIn("429", failure_notices[0])

    def test_optimizer_truncated_json_stops_without_scheduling_next_attempt(self):
        with patch.object(config_optimizer, "get_llm", return_value=SimpleNamespace(
                 invoke=lambda *args, **kwargs: AIMessage(content='{"updated_config": {"lr":'))), \
             patch.object(config_optimizer, "console"):
            result = config_optimizer.config_optimizer_node({"current_config": self.base_config(), "attempt_number": 1})
        self.assertEqual(result["optimisation_status"], "error")
        self.assertEqual(graph.route_after_optimisation(result), "onnx_exporter")
        self.assertNotIn("current_config", result)

    def test_optimizer_prompt_summarizes_long_history_without_mutating_records(self):
        original = self.base_config()
        metrics = {"status": "ok", "mean_acc": 0.8,
                   "train_loss_history": [1 / (epoch + 1) for epoch in range(30)],
                   "val_loss_history": [0.3 + epoch / 100 for epoch in range(30)]}
        history = [{"attempt": index + 1, "target_metric": "mean_acc", "training_status": "success",
                    "decision_split": "val", "config": original, "metrics": metrics, "score": 0.8}
                   for index in range(20)]
        invoke = Mock(return_value=AIMessage(content=json.dumps({
            "updated_config": {"lr": 2e-4}, "rationale": "Refine accuracy."})))
        with patch.object(config_optimizer, "get_llm", return_value=SimpleNamespace(invoke=invoke)), \
             patch.object(config_optimizer, "console"):
            result = config_optimizer.config_optimizer_node({
                "current_config": original, "attempt_number": 20, "metrics": metrics,
                "experiment_history": history,
            })
        self.assertEqual(result["optimisation_status"], "success")
        invoke.assert_called_once()
        prompt = invoke.call_args.args[0][-1].content
        context = json.loads(prompt)
        self.assertLess(len(prompt), 12000)
        self.assertEqual(context["total_attempts"], 20)
        self.assertLessEqual(len(context["experiment_history"]), 4)
        self.assertEqual(context["metrics"]["train_loss_history"], metrics["train_loss_history"][-5:])
        self.assertEqual(len(history), 20)
        self.assertEqual(len(metrics["train_loss_history"]), 30)

    def test_json_parser_rejects_nonobjects_and_handles_text_blocks(self):
        self.assertEqual(parse_json_response('```json\n{"lr":0.001}\n```'), {"lr": 0.001})
        self.assertEqual(parse_json_response([{"type": "text", "text": '{"gamma":2}'}]), {"gamma": 2})
        with self.assertRaises(ValueError):
            parse_json_response("[]")


if __name__ == "__main__":
    unittest.main()
