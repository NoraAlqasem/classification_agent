from __future__ import annotations

"""Autonomous multi-head ResNet training agent.

Example:
    python main.py --train-csv data/train.csv --val-csv data/val.csv \
        --data-root /path/to/images --heads make vehicle_type color \
        --arch resnet18 --weights weights/resnet18.pth
"""

import argparse
import os
import sys

from rich.console import Console

console = Console()


def _save_session_outcome(state: dict | None, status: str, error: str | None = None) -> None:
    """Keep the session summary factual when the graph stops before finalization."""
    if not state:
        return
    from pathlib import Path
    from agent.session_summary import write_session_summary
    from agent.nodes.monitor import _parse_results_csv
    from agent.nodes.reporter import collect_session_errors
    from training.artifact_paths import find_training_file

    config = state.get("current_config", {})
    session_dir = state.get("session_dir") or config.get("out")
    if not session_dir:
        return
    history = list(state.get("experiment_history", []))
    name = config.get("name", "")
    if name.startswith("exp_") and name[4:].isdigit():
        attempt = int(name[4:])
        run_dir = Path(session_dir) / name
        if run_dir.is_dir() and not any(entry.get("attempt") == attempt for entry in history):
            metrics = {}
            csv_path = find_training_file(run_dir, "training_metrics.csv")
            if csv_path.is_file():
                try:
                    metrics = _parse_results_csv(csv_path, config.get("monitor", "mean"))
                except (OSError, ValueError, TypeError, KeyError):
                    pass
            history.append({"attempt": attempt, "config": dict(config), "metrics": metrics,
                            "training_status": "interrupted" if status == "Interrupted" else "error",
                            "decision_split": state.get("decision_split", "val"),
                            "target_metric": state.get("target_metric", "mean_acc"),
                            "target_value": state.get("target_value"), "run_dir": str(run_dir)})
    errors = collect_session_errors({**state, "experiment_history": history})
    if error and error not in errors:
        errors.append(error)
    try:
        write_session_summary(
            session_dir, history, status=status, heads=state.get("heads"),
            target_metric=state.get("target_metric", "mean_acc"),
            target_value=state.get("target_value"), decision_split=state.get("decision_split", "val"),
            stop_reason="interrupted" if status == "Interrupted" else "error", errors=errors,
            onnx_path=state.get("onnx_path", ""),
            last_model_path=state.get("best_experiment", {}).get("last_weights", ""),
            llm_allowed_config_keys=state.get("llm_allowed_config_keys"),
        )
    except (OSError, ValueError, TypeError) as exc:
        console.print(f"Unable to update session summary: {exc}", style="yellow", markup=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from agent.nodes.user_input import _load_defaults

    defaults = _load_defaults()
    parser = argparse.ArgumentParser(
        description="Autonomous multi-head ResNet training with LangGraph and PyTorch DDP"
    )
    parser.add_argument("--dataset", default="", help="Folder containing train.csv, val.csv and test.csv (or DATASET_PATH)")
    parser.add_argument("--train-csv", default="", help="Training CSV (or TRAIN_CSV)")
    parser.add_argument("--val-csv", default="", help="Validation CSV (or VAL_CSV)")
    parser.add_argument("--test-csv", default="", help="Test CSV evaluated after each attempt (or TEST_CSV)")
    parser.add_argument("--decision-split", choices=["test", "val"],
                        default=os.getenv("DECISION_SPLIT", "").strip() or defaults.get("decision_split", "test"),
                        help="Split used for stopping and choosing the best attempt (default: test)")
    parser.add_argument("--data-root", default="", help="Prefix for relative root_folder values (or DATA_ROOT)")
    parser.add_argument("--heads", nargs="+", default=None, help="CSV label columns, e.g. make color (or HEADS)")
    parser.add_argument("--arch", "--model", dest="arch", default=defaults.get("arch", "resnet18"),
                        choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--weights", default="", help="Local ImageNet backbone state_dict matching --arch")
    parser.add_argument("--nproc-per-node", type=int, default=defaults.get("nproc_per_node", 1),
                        help="Number of GPUs/DDP processes")
    parser.add_argument("--alpha-json", default="", help="Optional per-head focal-alpha JSON")
    parser.add_argument("--head-loss-weights", nargs="*", default=None, metavar="HEAD=WEIGHT")
    parser.add_argument("--target-metric", default=defaults.get("target_metric", "mean_acc"),
                        help="mean_acc, macro_acc, HEAD_acc, or HEAD_macro_acc")
    parser.add_argument("--target-value", type=float, default=defaults.get("target_value", 0.85),
                        help="Target accuracy between 0 and 1")
    parser.add_argument("--max-attempts", type=int, default=defaults.get("max_attempts", 5),
                        help="Maximum training attempts")
    parser.add_argument("--llm-allowed-config-keys", nargs="+", default=None, metavar="KEY",
                        help="Configuration keys the LLM may change (comma/space-separated); "
                             "overrides LLM_ALLOWED_CONFIG_KEYS. Use all to allow supported keys "
                             "(default), or none to lock all")
    parser.add_argument("--export", default=defaults.get("export_format", "onnx"), choices=["onnx", "none"])
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate inputs and data without training, CUDA, or an OpenAI API key")
    return parser.parse_args(argv)


def main() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv()
    args = parse_args()

    initial_state = {
        "dataset_path": args.dataset,
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "test_csv": args.test_csv,
        "decision_split": args.decision_split,
        "data_root": args.data_root,
        "heads": args.heads,
        "target_metric": args.target_metric,
        "target_value": args.target_value,
        "max_attempts": args.max_attempts,
        "llm_allowed_config_keys": args.llm_allowed_config_keys,
        "model_type": args.arch,
        "weights_path": args.weights,
        "nproc_per_node": args.nproc_per_node,
        "alpha_json": args.alpha_json,
        "head_loss_weights": args.head_loss_weights,
        "export_format": args.export,
    }

    if args.validate_only:
        from agent.nodes.user_input import user_input_node
        from agent.nodes.dataset_validator import dataset_validator_node

        # Missing inputs are reported instead of prompting without a terminal.
        initial_state["interactive"] = sys.stdin.isatty()
        initial_state.update(user_input_node(initial_state))
        result = dataset_validator_node(initial_state)
        raise SystemExit(0 if result["is_valid"] else 1)

    if not os.getenv("OPENAI_API_KEY"):
        console.print(
            "[bold red]Error:[/bold red] OPENAI_API_KEY is not set. "
            "Add it to .env, or use --validate-only to check your inputs and dataset."
        )
        raise SystemExit(1)

    from agent.graph import compile_graph

    graph = compile_graph()
    console.print("[bold cyan]\nStarting Autonomous ResNet Training Agent...[/bold cyan]\n")
    final_state = None
    try:
        # The default graph limit is too small for five training attempts.
        recursion_limit = 20 + max(1, args.max_attempts) * 10
        for event in graph.stream(initial_state, config={"recursion_limit": recursion_limit},
                                  stream_mode="values"):
            final_state = event
    except KeyboardInterrupt:
        _save_session_outcome(final_state, "Interrupted")
        console.print("\n[yellow]Agent interrupted by user.[/yellow]")
        raise SystemExit(130)
    except Exception as exc:
        _save_session_outcome(final_state, "Failed", error=f"{type(exc).__name__}: {exc}")
        console.print(f"\n[bold red]Unhandled error:[/bold red] {exc}")
        raise

    if final_state:
        if final_state.get("session_summary_path"):
            console.print(f"\n[bold green]Session summary saved:[/bold green] {final_state['session_summary_path']}")
        if final_state.get("onnx_path"):
            console.print(f"[bold green]ONNX model:[/bold green] {final_state['onnx_path']}")
        if not final_state.get("is_valid", False) or final_state.get("stop_reason") in {"error", "unstable"}:
            raise SystemExit(1)
        if not final_state.get("best_experiment") or any(
            final_state.get(key) for key in ("error", "optimisation_error", "export_error", "onnx_error")
        ):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
