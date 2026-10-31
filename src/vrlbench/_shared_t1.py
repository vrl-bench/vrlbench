"""Read shared first trials without importing plugins or a model runtime."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ._comparison import comparable_workload
from .results import SimulationResult


def load_shared_first_trials(spec: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Match a completed SDK run and extract only its memory-free first trials.

    New records separate the actor's usage from the memory update after failure.
    Older built-in Retry records need no split because Retry has no such update.
    A custom source without that split must be rerun, not guessed to be Retry.
    """
    source = Path(spec["shared_t1_dir"]).resolve() / "evaluation.json"
    if not source.is_file():
        raise ValueError("shared_t1_dir must contain a completed SDK evaluation.json")
    payload = SimulationResult.from_path(source).require_complete().to_dict()
    original = payload.get("config", {})
    for field in ("model", "benchmark", "trials", "max_environment_steps",
                  "max_total_tokens", "runtime_env", "require_provider_usage"):
        before, after = original.get(field), spec.get(field)
        if field == "benchmark":
            before, after = comparable_workload(before), comparable_workload(after)
        if before != after:
            raise ValueError(f"shared first-trial source has different {field}")
    memory = original.get("memory") or {}
    legacy_retry = (memory.get("name") == "retry" and memory.get("factory") is None
                    and memory.get("method_factory") is None)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    first_trials = {}
    for case in payload["cases"]:
        first = deepcopy(case["trials"][0])
        if any(first.get(key) not in (None, "")
               for key in ("context", "system_context", "user_context")):
            raise ValueError("shared first trial must have no cross-trial memory context")
        metadata = first.setdefault("trial_metadata", {})
        if (not isinstance(metadata, dict)
                or metadata.get("max_environment_steps") != spec["max_environment_steps"]
                or any(not isinstance(metadata.get(key), str) or not metadata[key].strip()
                       for key in ("case_identity", "task"))):
            raise ValueError("shared first trial lacks matching task and action-budget metadata")
        usage = metadata.get("actor_llm_usage")
        if usage is None and legacy_retry:
            usage = first.get("llm_usage")
        if (not isinstance(usage, Mapping)
                or any(type(usage.get(key)) is not int or usage[key] < 0
                       for key in ("api_calls", "total_tokens"))
                or type(usage.get("provider_usage_complete")) is not bool):
            raise ValueError("shared first trial lacks actor-only model usage; rerun its source")
        first["llm_usage"] = deepcopy(dict(usage))
        metadata["actor_llm_usage"] = deepcopy(dict(usage))
        metadata["shared_t1_source"] = {"evaluation_path": str(source),
                                        "result_sha256": digest, "case_id": case["case_id"]}
        # A candidate recomputes its own post-failure state from this experience.
        for key in ("spatial_state_pre", "spatial_state", "controller_state_pre",
                    "controller_state", "temporal_decision", "controller_decision"):
            first[key] = None
        first_trials[case["case_id"]] = {"trial": first, "task": case["task"],
                                         "source_identity": deepcopy(case.get("source_identity", {}))}
    selected = {case["case_id"] for case in spec["benchmark"]["cases"]}
    if set(first_trials) != selected:
        raise ValueError("shared first-trial source has different case selection")
    return first_trials
