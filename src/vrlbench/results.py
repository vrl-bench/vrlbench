"""Recorded memory-system simulations and shared metrics; no model dependencies."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import json
from math import fsum
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from ._comparison import comparable_workload


RESOURCE_FIELDS = (
    "api_calls", "prompt_tokens", "completion_tokens", "total_tokens",
    "environment_steps",
)


def _integer(value: Any, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _count(value: Any) -> int | None:
    return value if _integer(value) else None


def _validated_cases(cases: Iterable[Mapping[str, Any]], trials: int) -> list[dict]:
    if not _integer(trials, minimum=1):
        raise ValueError("trials must be a positive integer")
    records = []
    seen = set()
    for case in cases:
        if not isinstance(case, Mapping):
            raise TypeError("each case must be a mapping")
        case_id, task = case.get("case_id"), case.get("task")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("case_id must be a nonempty string")
        if case_id in seen:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen.add(case_id)
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"{case_id}: task must be a nonempty string")
        rows = case.get("trials")
        if not isinstance(rows, list):
            raise TypeError(f"{case_id}: trials must be a list")
        if len(rows) > trials:
            raise ValueError(f"{case_id}: trial budget exceeded")
        solved = False
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise TypeError(f"{case_id}: each trial must be a mapping")
            if not _integer(row.get("trial")) or row["trial"] != index:
                raise ValueError(f"{case_id}: trial indices must be contiguous from zero")
            if type(row.get("success")) is not bool:
                raise TypeError(f"{case_id}: success must be a boolean")
            if solved:
                raise ValueError(f"{case_id}: a trial follows success")
            solved = row["success"]
            if not isinstance(row.get("trajectory", ""), str):
                raise TypeError(f"{case_id}: trajectory must be a string")
            for field in ("trial_metadata", "llm_usage"):
                if not isinstance(row.get(field, {}), Mapping):
                    raise TypeError(f"{case_id}: {field} must be a mapping")
        if not isinstance(case.get("resource_usage", {}), Mapping):
            raise TypeError(f"{case_id}: resource_usage must be a mapping")
        records.append(deepcopy(dict(case)))
    return records


def _resource_record(case: Mapping[str, Any]) -> tuple[dict, dict, bool | None]:
    """Return complete amounts, reported subtotals, and usage completeness.

    A ledger marked incomplete may contain only a subtotal. Such token counts
    remain visible as reported subtotals, never as complete aggregate usage.
    """
    usage = case.get("resource_usage", {})
    model_fields = (*RESOURCE_FIELDS[:4], "provider_usage_complete")
    llm = usage.get("llm", usage)
    if not isinstance(llm, Mapping):
        # An explicitly malformed ledger is not absence of a ledger.
        llm = {"provider_usage_complete": False}
    elif not any(field in llm for field in model_fields):
        # Accept flat model fields beside an empty nested ledger, but never
        # let environment-only metadata suppress usable per-trial snapshots.
        llm = usage
    trial_rows = case["trials"]
    if not any(field in llm for field in model_fields) and trial_rows:
        snapshots = [row.get("llm_usage", {}) for row in trial_rows]
        llm = {}
        for field in RESOURCE_FIELDS[:4]:
            values = [_count(snapshot.get(field)) for snapshot in snapshots]
            if all(value is not None for value in values):
                llm[field] = sum(values)
        flags = [snapshot.get("provider_usage_complete") for snapshot in snapshots]
        llm["provider_usage_complete"] = (
            False if any(flag is False for flag in flags)
            else True if all(flag is True for flag in flags) else None
        )
    flag = llm.get("provider_usage_complete")
    complete = flag if type(flag) is bool else None
    amounts = {field: _count(llm.get(field)) for field in RESOURCE_FIELDS[:4]}
    subtotals = dict(amounts)
    prompt, completion, total = (
        amounts["prompt_tokens"], amounts["completion_tokens"], amounts["total_tokens"]
    )
    invalid = any(
        field in llm and llm[field] is not None and _count(llm[field]) is None
        for field in RESOURCE_FIELDS[:4]
    )
    if total is not None:
        invalid = invalid or any(
            value is not None and value > total for value in (prompt, completion)
        )
        invalid = invalid or (
            prompt is not None and completion is not None and prompt + completion != total
        )
    elif prompt is not None and completion is not None:
        amounts["total_tokens"] = prompt + completion
        subtotals["total_tokens"] = prompt + completion
    if invalid:
        complete = False
    if complete is True and amounts["total_tokens"] is None:
        complete = False
    if complete is False:
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            amounts[field] = None

    environment = usage.get("environment", {})
    if not isinstance(environment, Mapping):
        environment = {}
    steps = _count(usage.get("environment_steps", usage.get("environment_transitions")))
    if steps is None:
        steps = _count(environment.get("environment_steps", environment.get("environment_transitions")))
    if steps is None and trial_rows:
        values = [_count(row.get("trial_metadata", {}).get("environment_steps"))
                  for row in trial_rows]
        if all(value is not None for value in values):
            steps = sum(values)
    amounts["environment_steps"] = steps
    subtotals["environment_steps"] = steps
    return amounts, subtotals, complete


def _resources(cases: Sequence[Mapping[str, Any]]) -> dict:
    records = [_resource_record(case) for case in cases]
    amounts = {}
    known_subtotals = {}
    known_cases = {}
    for field in RESOURCE_FIELDS:
        values = [row[0][field] for row in records]
        known_cases[field] = sum(value is not None for value in values)
        amounts[field] = (sum(values) if values and all(value is not None for value in values)
                          else None)
        reported = [row[1][field] for row in records if row[1][field] is not None]
        known_subtotals[field] = sum(reported) if reported else None
    flags = [row[2] for row in records]
    amounts["provider_usage_complete"] = (
        False if any(flag is False for flag in flags)
        else True if flags and all(flag is True for flag in flags) else None
    )
    amounts.update({"basis": "received_cases_including_incomplete",
                    "cases_received": len(cases), "known_subtotals": known_subtotals,
                    "known_cases": known_cases})
    return amounts


def aggregate_cases(
    cases: Iterable[Mapping[str, Any]], trials: int, planned_cases: int | None = None,
) -> dict[str, Any]:
    """Summarize completed cases without scoring incomplete attempts as failures.

    A case completes on success or after executing its full trial budget.
    Metrics describe only completed cases; ``counts.planned`` and the explicit
    basis preserve the distinction between partial progress and a full run.
    Resource totals cover all received cases, including partial execution.
    """
    records = _validated_cases(cases, trials)
    planned = len(records) if planned_cases is None else planned_cases
    if not _integer(planned) or planned < len(records):
        raise ValueError("planned_cases must be an integer at least the received case count")
    completed = [case for case in records if case["trials"] and
                 (case["trials"][-1]["success"] or len(case["trials"]) == trials)]
    solved = sum(case["trials"][-1]["success"] for case in completed)
    first_failed = [case for case in completed if not case["trials"][0]["success"]]
    recovered = sum(case["trials"][-1]["success"] for case in first_failed)
    tasks = defaultdict(list)
    attempts = []
    for case in completed:
        success = case["trials"][-1]["success"]
        tasks[case["task"]].append(int(success))
        attempts.append(len(case["trials"]) if success else trials + 1)
    n = len(completed)
    completed_ids = {case["case_id"] for case in completed}
    return {
        "trials": trials,
        "counts": {"planned": planned, "completed": n, "solved": solved,
                   "first_trial_failed": len(first_failed), "recovered": recovered},
        "metrics": {
            "sr_at_t": solved / n if n else None,
            "rr_at_t": recovered / len(first_failed) if first_failed else None,
            "avg_t_at_t": fsum(attempts) / n if n else None,
            "macro_task_sr_at_t": (
                fsum(sum(values) / len(values) for values in tasks.values()) / len(tasks)
                if tasks else None
            ),
        },
        "metrics_basis": "completed_cases",
        "incomplete_case_ids": [case["case_id"] for case in records
                                if case["case_id"] not in completed_ids],
        "resources": _resources(records),
    }


@dataclass(frozen=True)
class SimulationResult:
    """One recorded trial process with outcomes, usage and completion status.

    The payload retains the version-1 result format. Workload, model and memory
    views expose the declared system without duplicating case/trial records.
    """

    payload: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping):
            raise TypeError("evaluation payload must be a mapping")
        object.__setattr__(self, "payload", deepcopy(dict(self.payload)))

    @property
    def status(self) -> str:
        return str(self.payload.get("status", "unknown"))

    @property
    def workload(self) -> dict | None:
        config = self.payload.get("config", {})
        if not isinstance(config, Mapping):
            return None
        return deepcopy(config.get("workload", config.get("benchmark")))

    @property
    def model(self) -> dict | None:
        config = self.payload.get("config", {})
        return deepcopy(config.get("model")) if isinstance(config, Mapping) else None

    @property
    def memory(self) -> dict | None:
        config = self.payload.get("config", {})
        return deepcopy(config.get("memory")) if isinstance(config, Mapping) else None

    @property
    def metrics(self) -> dict:
        return deepcopy(self.payload.get("metrics", {}))

    @property
    def counts(self) -> dict:
        return deepcopy(self.payload.get("counts", {}))

    @property
    def cases(self) -> list[dict]:
        return deepcopy(self.payload.get("cases", []))

    @property
    def resources(self) -> dict:
        return deepcopy(self.payload.get("resources", {}))

    @property
    def output_dir(self) -> Path | None:
        value = self.payload.get("output_dir")
        return Path(value) if value is not None else None

    def require_complete(self) -> SimulationResult:
        counts = self.counts
        planned, completed = counts.get("planned"), counts.get("completed")
        if (self.status != "complete" or not _integer(planned, minimum=1)
                or not _integer(completed) or planned != completed):
            raise RuntimeError("evaluation is not complete; partial or failed runs cannot be scored")
        return self

    def to_dict(self) -> dict:
        return deepcopy(self.payload)

    @classmethod
    def from_path(cls, path: str | Path) -> SimulationResult:
        with Path(path).open(encoding="utf-8") as handle:
            return cls(json.load(handle))

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        content = json.dumps(self.payload, ensure_ascii=False, indent=2, sort_keys=True,
                             allow_nan=False) + "\n"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent,
                                             prefix=f".{destination.name}.", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(destination)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return destination


def _comparison_context(payload: Mapping[str, Any]) -> dict:
    config = payload.get("config", {})
    if not isinstance(config, Mapping):
        raise ValueError("comparison requires a config mapping")
    model = config.get("model", payload.get("model"))
    benchmark = comparable_workload(config.get("benchmark", payload.get("benchmark")))
    trials = payload.get("trials", config.get("trials"))
    if model is None or benchmark is None or not _integer(trials, minimum=1):
        raise ValueError("comparison requires declared model, benchmark, and trial budget")
    if "trials" in config and config["trials"] != trials:
        raise ValueError("result and config disagree on the trial budget")
    context = {"model": model, "benchmark": benchmark, "trials": trials}
    for field in ("max_environment_steps", "max_actor_steps", "max_steps", "trial_seed_mode",
                  "max_total_tokens", "runtime_env"):
        context[field] = config.get(field, payload.get(field))
    return context


def _component_identity_check(payloads: Sequence[Mapping[str, Any]], context: dict) -> dict:
    """Check recorded shared-plugin sources, not the intentionally varied memory.

    Old results may lack source hashes. They remain readable, but the report
    explicitly distinguishes missing identity from verified agreement. Built-in
    components without plugin factories are outside this source-hash check.
    """
    roles = ("benchmark", "model_transport")
    sources = []
    for payload in payloads:
        identity = payload.get("identity")
        if identity is None:
            identity = {}
        if not isinstance(identity, Mapping):
            raise ValueError("comparison identity must be a mapping when recorded")
        plugins = identity.get("plugins")
        if plugins is None:
            plugins = []
        if not isinstance(plugins, list):
            raise ValueError("recorded plugin identities must be a list")
        recorded = {}
        for plugin in plugins:
            if not isinstance(plugin, Mapping):
                raise ValueError("each recorded plugin identity must be a mapping")
            role = plugin.get("role")
            if role not in roles:
                continue
            if role in recorded:
                raise ValueError(f"duplicate recorded {role} plugin identity")
            digest = plugin.get("source_sha256")
            if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                                       or any(char not in "0123456789abcdefABCDEF" for char in digest)):
                raise ValueError(f"recorded {role} source_sha256 must be a SHA-256 digest")
            recorded[role] = digest.lower() if digest is not None else None
        sources.append(recorded)

    factories = {
        "benchmark": context["benchmark"].get("factory") if isinstance(context["benchmark"], Mapping) else None,
        "model_transport": context["model"].get("transport") if isinstance(context["model"], Mapping) else None,
    }
    components = {}
    for role in roles:
        applicable = factories[role] is not None or any(role in source for source in sources)
        digests = {source[role] for source in sources if source.get(role) is not None}
        if len(digests) > 1:
            raise ValueError(f"results have different {role} plugin source identities")
        missing = [index for index, source in enumerate(sources) if source.get(role) is None]
        status = "not_applicable" if not applicable else "unverified" if missing else "verified"
        components[role] = {"status": status, "source_sha256": next(iter(digests), None),
                            "missing_run_indices": missing if applicable else []}
    statuses = {component["status"] for component in components.values()}
    status = ("unverified" if "unverified" in statuses else "verified" if "verified" in statuses
              else "not_applicable")
    return {"scope": "recorded_non_memory_plugin_sources", "status": status,
            "components": components}


def _environment_identity_check(payloads: Sequence[Mapping[str, Any]], context: dict) -> dict:
    """Match recorded native WebShop source/data/index identities, not build hosts."""
    digests, missing = set(), []
    benchmark = context["benchmark"]
    applicable = (isinstance(benchmark, Mapping) and benchmark.get("name") == "webshop"
                  and benchmark.get("factory") is None)
    for index, payload in enumerate(payloads):
        environment = (payload.get("identity") or {}).get("environment")
        if environment is None:
            missing.append(index)
            continue
        applicable = True
        if not isinstance(environment, Mapping) or environment.get("benchmark") != "webshop":
            raise ValueError("recorded environment identity must describe WebShop")
        digest = environment.get("source_data_index_root_sha256")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdefABCDEF" for char in digest)):
            raise ValueError("recorded WebShop source/data/index identity must be a SHA-256 digest")
        digests.add(digest.lower())
    if len(digests) > 1:
        raise ValueError("results have different WebShop source/data/index identities")
    return {"scope": "recorded_native_webshop_source_data_index",
            "status": "not_applicable" if not applicable else "unverified" if missing else "verified",
            "source_data_index_root_sha256": next(iter(digests), None),
            "missing_run_indices": missing if applicable else []}


def compare_results(results: Sequence[SimulationResult | Mapping[str, Any]]) -> dict:
    """Describe paired effects relative to the first complete, compatible run.

    Matching task cases and interaction/trial settings does not establish
    equal model-call or token cost. Recorded benchmark/model-transport source
    hashes must agree when available; missing hashes are reported as unverified.
    No statistical interval is inferred here.
    """
    if not results:
        raise ValueError("at least one result is required")
    normalized = [item if isinstance(item, SimulationResult) else SimulationResult(dict(item))
                  for item in results]
    contexts, case_maps, summaries, payloads = [], [], [], []
    for result in normalized:
        result.require_complete()
        payload = result.to_dict()
        context = _comparison_context(payload)
        rows = _validated_cases(result.cases, context["trials"])
        summary = aggregate_cases(rows, context["trials"], result.counts["planned"])
        if summary["counts"] != result.counts or summary["incomplete_case_ids"]:
            raise ValueError("stored counts do not describe complete case records")
        contexts.append(context)
        case_maps.append({row["case_id"]: row for row in rows})
        summaries.append(summary)
        payloads.append(payload)
    reference_context, reference_cases = contexts[0], case_maps[0]
    for context, cases in zip(contexts[1:], case_maps[1:]):
        if context != reference_context:
            raise ValueError("results have incompatible models, environments, or budgets")
        if ({key: row["task"] for key, row in cases.items()} !=
                {key: row["task"] for key, row in reference_cases.items()}):
            raise ValueError("results must contain the same case IDs and task assignments")
    identity_check = _component_identity_check(payloads, reference_context)
    environment_check = _environment_identity_check(payloads, reference_context)
    rows, effects = [], []
    for index, (result, cases, summary) in enumerate(zip(normalized, case_maps, summaries)):
        payload = result.to_dict()
        config = payload.get("config", {})
        memory = config.get("memory", {})
        memory_name = memory.get("name") if isinstance(memory, Mapping) else memory
        method = payload.get("method", config.get("method", memory_name or f"result_{index}"))
        rows.append({"method": method, "counts": summary["counts"],
                     "metrics": summary["metrics"],
                     "resources": result.resources or summary["resources"]})
        if config.get("replay_gate") is not None:
            rows[-1]["replay_gate"] = deepcopy(config["replay_gate"])
        if index:
            differences, attempt_differences = [], []
            for case_id, case in cases.items():
                reference = reference_cases[case_id]
                success, reference_success = case["trials"][-1]["success"], reference["trials"][-1]["success"]
                differences.append(int(success) - int(reference_success))
                attempt_differences.append(
                    (len(case["trials"]) if success else reference_context["trials"] + 1) -
                    (len(reference["trials"]) if reference_success else reference_context["trials"] + 1)
                )
            effects.append({
                "target": method, "reference": rows[0]["method"],
                "sr_difference_pp": 100 * fsum(differences) / len(differences),
                "avg_t_difference": fsum(attempt_differences) / len(attempt_differences),
                "positive_pairs": differences.count(1), "negative_pairs": differences.count(-1),
                "tied_pairs": differences.count(0),
            })
    return {"comparison": "descriptive_paired", "context": reference_context,
            "component_identity_check": identity_check,
            "environment_identity_check": environment_check, "equal_compute_assumed": False,
            "rows": rows, "paired_effects": effects}


# The earlier result name is the same class, not a second serialization format.
EvaluationResult = SimulationResult

__all__ = ["SimulationResult", "EvaluationResult", "aggregate_cases", "compare_results"]
