from __future__ import annotations

import operator
from typing import Annotated, Any
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    # ── User-supplied inputs ─────────────────────────────────────────────────
    dataset_path: str
    train_csv: str
    val_csv: str
    test_csv: str
    decision_split: str      # test (default) or val for stopping/best attempt
    data_root: str
    heads: list[str]
    weights_path: str
    alpha_json: str | None
    head_loss_weights: dict[str, float]
    nproc_per_node: int
    interactive: bool
    input_errors: list[str]
    target_metric: str       # mean_acc, macro_acc, HEAD_acc, or HEAD_macro_acc
    target_value: float      # e.g. 0.85
    max_attempts: int
    model_type: str          # resnet18, resnet34, or resnet50
    export_format: str       # onnx or none
    llm_allowed_config_keys: list[str]   # resolved user allowlist; empty locks all

    # ── Dataset validation ───────────────────────────────────────────────────
    validation_report: dict[str, Any]
    is_valid: bool

    # ── Training configuration ───────────────────────────────────────────────
    current_config: dict[str, Any]
    session_dir: str                     # numbered session output directory
    # config_history accumulates across updates (each update appends one dict)
    config_history: Annotated[list[dict[str, Any]], operator.add]

    # ── Training run ─────────────────────────────────────────────────────────
    attempt_number: int
    training_result: dict[str, Any]      # output from trainer node
    test_result: dict[str, Any]          # checkpoint evaluation on test.csv
    run_dir: str                         # per-attempt ResNet run folder

    # ── Monitoring ───────────────────────────────────────────────────────────
    metrics: dict[str, Any]             # extracted from training_metrics.csv
    validation_metrics: dict[str, Any]  # validation metrics preserved separately

    # ── Evaluation (deep-agent reasoning) ────────────────────────────────────
    evaluation_result: dict[str, Any]
    issues_detected: list[str]
    target_reached: bool

    # ── Self-learning memory ─────────────────────────────────────────────────
    # Each entry is a full experiment record; accumulates across training loops
    experiment_history: Annotated[list[dict[str, Any]], operator.add]
    best_experiment: dict[str, Any]

    # ── Configuration optimisation ────────────────────────────────────────────
    optimisation_status: str             # success, disabled, or error
    optimisation_error: str              # retained even if export also fails
    optimisation_rationale: str          # LLM explanation of the changes made

    # ── Export ───────────────────────────────────────────────────────────────
    onnx_path: str

    # ── Session summary ──────────────────────────────────────────────────────
    session_summary_path: str

    # ── Control flow ─────────────────────────────────────────────────────────
    stop_reason: str   # target_reached, max_attempts, no_allowed_changes, unstable, error
    error: str

    # ── LLM conversation log (accumulates) ───────────────────────────────────
    messages: Annotated[list[Any], operator.add]
