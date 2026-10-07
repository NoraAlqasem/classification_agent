from __future__ import annotations

from tools.config_tools import (
    VALID_RESNET_CONFIG_KEYS_DOC,
    read_config,
    sanitize_resnet_config,
)
from tools.export_tools import validate_onnx_file
from tools.memory_tools import (
    get_attempt_history_summary,
    get_best_known_config,
    load_knowledge_base,
    search_lessons_by_issue,
)
from tools.metrics_tools import (
    compute_overfitting_score,
    get_final_metrics,
    list_run_artefacts,
    read_loss_history,
    read_per_class_metrics,
    read_results_csv,
    read_test_metrics,
)
from tools.training_tools import check_gpu_availability, estimate_training_time, read_training_log

EVALUATOR_TOOLS = [
    read_loss_history,
    read_per_class_metrics,
    list_run_artefacts,
    read_results_csv,
    read_test_metrics,
    compute_overfitting_score,
    get_final_metrics,
    read_training_log,
]

OPTIMIZER_TOOLS = [
    read_test_metrics,
    read_per_class_metrics,
    load_knowledge_base,
    search_lessons_by_issue,
    read_config,
    get_best_known_config,
    get_attempt_history_summary,
]

__all__ = [
    "EVALUATOR_TOOLS",
    "OPTIMIZER_TOOLS",
    "VALID_RESNET_CONFIG_KEYS_DOC",
    "sanitize_resnet_config",
    "check_gpu_availability",
    "compute_overfitting_score",
    "estimate_training_time",
    "get_attempt_history_summary",
    "get_best_known_config",
    "load_knowledge_base",
    "read_config",
    "read_loss_history",
    "read_per_class_metrics",
    "read_results_csv",
    "read_test_metrics",
    "read_training_log",
    "search_lessons_by_issue",
    "validate_onnx_file",
]
