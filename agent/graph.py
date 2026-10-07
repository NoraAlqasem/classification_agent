from __future__ import annotations

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes.user_input import user_input_node
from agent.nodes.dataset_validator import dataset_validator_node
from agent.nodes.config_generator import config_generator_node
from agent.nodes.trainer import trainer_node
from agent.nodes.monitor import monitor_node
from agent.nodes.tester import test_checkpoint_node
from agent.nodes.evaluator import evaluator_node
from agent.nodes.config_optimizer import config_optimizer_node
from agent.nodes.memory_node import memory_node
from agent.nodes.decision import decision_node
from agent.nodes.onnx_exporter import onnx_exporter_node
from agent.nodes.reporter import reporter_node


# ── Conditional edge functions ────────────────────────────────────────────────

def route_after_validation(state: AgentState) -> str:
    if not state.get("is_valid", False):
        return "reporter"          # report the validation failure and exit
    return "config_generator"


def route_after_decision(state: AgentState) -> str:
    reason = state.get("stop_reason", "")
    if reason in ("target_reached", "max_attempts", "unstable", "error"):
        return "onnx_exporter"
    return "config_optimizer"      # continue → optimise → retrain


def route_after_optimisation(state: AgentState) -> str:
    if state.get("optimisation_status") == "error":
        return "onnx_exporter"
    return "trainer"


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    # Nodes
    g.add_node("user_input",       user_input_node)
    g.add_node("dataset_validator", dataset_validator_node)
    g.add_node("config_generator", config_generator_node)
    g.add_node("trainer",          trainer_node)
    g.add_node("monitor",          monitor_node)
    g.add_node("test_checkpoint",  test_checkpoint_node)
    g.add_node("evaluator",        evaluator_node)
    g.add_node("memory",           memory_node)
    g.add_node("decision",         decision_node)
    g.add_node("config_optimizer", config_optimizer_node)
    g.add_node("onnx_exporter",    onnx_exporter_node)
    g.add_node("reporter",         reporter_node)

    # Entry point
    g.set_entry_point("user_input")

    # Linear edges
    g.add_edge("user_input",       "dataset_validator")
    g.add_edge("config_generator", "trainer")
    g.add_edge("trainer",          "monitor")
    g.add_edge("monitor",          "test_checkpoint")
    g.add_edge("test_checkpoint",  "evaluator")
    g.add_edge("evaluator",        "memory")
    g.add_edge("memory",           "decision")
    g.add_edge("reporter",         END)

    # Conditional edges
    g.add_conditional_edges(
        "dataset_validator",
        route_after_validation,
        {"config_generator": "config_generator", "reporter": "reporter"},
    )
    g.add_conditional_edges(
        "decision",
        route_after_decision,
        {"config_optimizer": "config_optimizer", "onnx_exporter": "onnx_exporter"},
    )
    g.add_conditional_edges(
        "config_optimizer",
        route_after_optimisation,
        {"trainer": "trainer", "onnx_exporter": "onnx_exporter"},
    )
    g.add_edge("onnx_exporter", "reporter")

    return g


def compile_graph():
    return build_graph().compile()
