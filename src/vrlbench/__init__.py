"""VRL-Bench: a unified simulator for agent memory systems."""

from .specs import (
    WorkloadSpec, SimulationSpec, BenchmarkSpec, EvaluationSpec,
    ActorSpec, FactorySpec, MemorySpec, ModelSpec, ReplayGateSpec,
)
from .results import SimulationResult, EvaluationResult, aggregate_cases, compare_results
from .statistics import analyze_results, posthoc_power_mcnemar
from .replay import ReplayContext
from .simulation import plan, simulate
from .development import CandidateFeedback, evaluate_candidate
from .methods import (
    ActorContext, CaseConfig, MemoryMethod, TrialRecord,
    list_methods, load_method, render_trajectory,
)
from ._version import __version__

evaluate = simulate

__all__ = ["WorkloadSpec", "SimulationSpec", "SimulationResult", "simulate", "plan",
           "ActorSpec", "FactorySpec", "MemorySpec", "ModelSpec", "ReplayGateSpec", "ReplayContext", "aggregate_cases", "compare_results",
           "analyze_results", "posthoc_power_mcnemar",
           "CandidateFeedback", "evaluate_candidate",
           "BenchmarkSpec", "EvaluationSpec", "EvaluationResult", "evaluate",
           "ActorContext", "CaseConfig", "MemoryMethod", "TrialRecord",
           "list_methods", "load_method", "render_trajectory"]
