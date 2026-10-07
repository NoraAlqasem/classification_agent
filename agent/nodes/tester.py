"""Test the trained checkpoint before the agent evaluates whether to retry."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from rich.console import Console

from agent.state import AgentState

console = Console()


def test_checkpoint_node(state: AgentState) -> dict:
    if state.get("decision_split", "val") != "test":
        return {}
    validation = deepcopy(state.get("metrics", {}))
    training = state.get("training_result", {})
    checkpoint = training.get("best_weights", "")
    test_csv = state.get("test_csv", "")
    run_dir = training.get("run_dir") or state.get("run_dir", "")
    output_dir = str(Path(run_dir) / "test_evaluation") if run_dir else ""

    def failed(message):
        # Preserve validation diagnostics but remove every primary score so a
        # failed test cannot inherit a passing validation target.
        metrics = deepcopy(validation)
        metrics.pop("mean_acc", None)
        metrics.pop("heads", None)
        for head in state.get("heads", validation.get("heads", {})):
            metrics.pop(f"{head}_acc", None)
        metrics.update({"status": "test_failed", "error": message,
                        "decision_split": "test", "split": "test",
                        "test_evaluated": False, "full_coverage": False,
                        "validation_metrics": validation})
        return {"metrics": metrics, "validation_metrics": validation,
                "test_result": {"status": "error", "error": message,
                                "checkpoint": checkpoint, "csv": test_csv,
                                "output_dir": output_dir, "metrics": metrics},
                "error": message}

    if training.get("status") != "success":
        return failed(f"Test evaluation unavailable: training failed ({training.get('error', 'no successful training result')})")
    if validation.get("status") != "ok":
        return failed(f"Test evaluation unavailable: invalid validation metrics ({validation.get('error', validation.get('status', 'missing metrics'))})")
    if not checkpoint or not Path(checkpoint).is_file():
        return failed(f"Test evaluation failed: missing best checkpoint: {checkpoint or '(not specified)'}")
    if not test_csv:
        return failed("Test evaluation failed: TEST_CSV/--test-csv is required in test decision mode")
    if not output_dir:
        return failed("Test evaluation failed: training attempt has no output directory")
    failure_message = ""
    try:
        from training.test_resnet import evaluate_checkpoint

        config = training.get("config") or state.get("current_config", {})
        result = evaluate_checkpoint(
            checkpoint, test_csv, output_dir, data_root=state.get("data_root", ""),
            heads=state.get("heads"), config=config,
            batch_size=int(config.get("batch_size", 32)),
        )
        if result.get("status") != "ok" or not result.get("full_coverage"):
            raise ValueError("Evaluator did not return complete successful test coverage")
    except Exception as exc:
        failure_message = f"Test evaluation failed: {exc}"
    finally:
        # Testing runs in this parent process; the next DDP training attempt
        # starts in child processes. Release the parent's unused CUDA allocator
        # blocks after inference/exception references have left their scope.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            console.print(f"[yellow]CUDA cache cleanup failed: {exc}[/yellow]")
    if failure_message:
        console.print(f"[red]{failure_message}[/red]")
        return failed(failure_message)

    metrics = deepcopy(validation)
    # All primary head accuracies come from this checkpoint's test predictions;
    # losses/history remain validation diagnostics for optimiser reasoning.
    test_heads = {
        head: {**validation.get("heads", {}).get(head, {}), **values}
        for head, values in result["heads"].items()
    }
    metrics.update(result)
    metrics.update({"heads": test_heads, "validation_metrics": validation,
                    "status": "ok", "decision_split": "test", "split": "test",
                    "test_evaluated": True})
    test_result = {**result, "status": "ok", "checkpoint": result["checkpoint"],
                   "csv": result["test_csv"], "metrics": result,
                   "output_dir": result["output_dir"]}
    console.print(f"[cyan]Test accuracy — attempt {state.get('attempt_number', '?')}: "
                  f"mean={result['mean_acc']:.4f}, images={result['total_images']}[/cyan]")
    return {"metrics": metrics, "validation_metrics": validation,
            "test_result": test_result, "error": ""}
