"""Evaluate complete candidate implementations through the real simulation path.

Candidates are trusted local Python plugins, executed in the simulation worker,
not a security sandbox. This module supplies task feedback, not a fitness rule
or an automatic claim that a changed implementation is better.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any

from .results import SimulationResult, _comparison_context, compare_results
from .simulation import plan, simulate
from .specs import FactorySpec, MemorySpec, SimulationSpec


@dataclass(frozen=True)
class CandidateFeedback:
    """Recorded task feedback; incomplete or unverifiable runs are not scores.

    ``regressed_case_ids`` identifies success losses or increased censored
    trials-to-success relative to the reference. ``None`` means no valid paired
    comparison was available, not that no regressions occurred.
    ``reference`` identifies the result used for comparison and its recorded
    memory sources; it is not necessarily the memory-free condition.
    """

    status: str
    scorable: bool
    metrics: dict[str, Any]
    counts: dict[str, Any]
    resources: dict[str, Any]
    cases: list[dict[str, Any]]
    comparison: dict[str, Any] | None
    regressed_case_ids: list[str] | None
    candidate_path: str
    candidate_sha256: str
    candidate_source_path: str
    candidate_identity_verified: bool | None
    output_dir: str
    evaluation_path: str
    error: dict[str, Any] | None
    reference: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return an independent JSON-compatible view of this feedback."""
        return asdict(self)


def evaluate_candidate(
    candidate_path: str | Path,
    spec: SimulationSpec,
    *,
    factory: str = "create_method",
    interface: str = "method",
    factory_kwargs: dict[str, Any] | None = None,
    reference: SimulationResult | str | Path | None = None,
) -> CandidateFeedback:
    """Replace only memory, execute the candidate, and save ``feedback.json``.

    ``interface='method'`` accepts a complete ``MemoryMethod`` factory;
    ``interface='memory'`` accepts a ``MemorySystem`` factory. Neither restricts
    changes to parameters. The caller selects a fresh ``spec.output_dir`` for
    each evaluation; existing nonempty runs are never overwritten. Model,
    workload, budgets, replay control and shared-first-trial source are retained.

    Obvious reference incompatibilities are rejected before candidate execution.
    Final paired comparison also checks the recorded component identities.
    Failed and timed-out runs retain their records and costs but expose no task
    scores. This call can incur API charges when the supplied workload does.
    """
    if not isinstance(spec, SimulationSpec):
        raise TypeError("spec must be a SimulationSpec")
    if interface not in {"method", "memory"}:
        raise ValueError("interface must be method or memory")
    source = Path(candidate_path).resolve(strict=True)
    if source.suffix != ".py" or not source.is_file():
        raise ValueError("candidate_path must identify a Python source file")
    source_bytes = source.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    binding = FactorySpec(f"{source}:{factory}", deepcopy({} if factory_kwargs is None else factory_kwargs))
    memory = MemorySpec(
        name=f"candidate_{digest[:16]}",
        **{"method_factory" if interface == "method" else "factory": binding},
    )
    candidate_spec = replace(spec, memory=memory).resolved()
    # Planning validates paths and the existing shared source, but never imports
    # candidate code. Keep this before any new run directory is created.
    plan(candidate_spec)
    baseline = reference
    reference_identity = None
    if baseline is not None:
        reference_path = None
        if not isinstance(baseline, SimulationResult):
            path = Path(baseline)
            reference_path = (path / "evaluation.json" if path.is_dir() else path).resolve(strict=True)
            baseline = SimulationResult.from_path(reference_path)
        reference_context = compare_results([baseline])["context"]
        if reference_context != _comparison_context({"config": candidate_spec.to_dict()}):
            raise ValueError("reference has incompatible models, environments, or budgets")
        baseline_payload = baseline.to_dict()
        canonical_result = json.dumps(baseline_payload, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":"), allow_nan=False).encode("utf-8")
        reference_identity = {
            "evaluation_path": str(reference_path) if reference_path is not None else None,
            "result_sha256": hashlib.sha256(canonical_result).hexdigest(),
            "memory": baseline.memory,
            "memory_sources": [deepcopy(entry) for entry in
                               ((baseline_payload.get("identity") or {}).get("plugins") or [])
                               if entry.get("role") in {"memory", "memory_method"}],
        }

    result = simulate(candidate_spec)
    payload = result.to_dict()
    error = deepcopy(payload.get("error"))
    scorable = False
    try:
        result.require_complete()
        scorable = True
    except RuntimeError:
        pass
    role = "memory_method" if interface == "method" else "memory"
    identities = [entry for entry in payload.get("identity", {}).get("plugins", [])
                  if entry.get("role") == role]
    verified = None if not identities and not scorable else (
        len(identities) == 1
        and identities[0].get("source_path") == str(source)
        and identities[0].get("source_sha256") == digest
    )
    if scorable and verified is not True:
        scorable = False
        error = {"type": "CandidateIdentityMismatch",
                 "message": "The worker did not execute the recorded candidate source hash"}
    comparison, regressions = None, None
    if scorable and baseline is not None:
        try:
            comparison = compare_results([baseline, result])
        except (ValueError, RuntimeError) as exc:
            scorable = False
            error = {"type": type(exc).__name__, "message": str(exc)}
        else:
            regressions = sorted(row["case_id"] for row in comparison["paired_effects"][0]["case_deltas"]
                                 if row["censored_trials_difference"] > 0)
    metrics = result.metrics if scorable else {key: None for key in result.metrics}
    source_snapshot = result.output_dir / "candidate_source.py"
    with source_snapshot.open("xb") as handle:
        handle.write(source_bytes)
    feedback = CandidateFeedback(
        status=result.status, scorable=scorable, metrics=metrics,
        counts=result.counts, resources=result.resources, cases=result.cases,
        comparison=comparison, regressed_case_ids=regressions,
        candidate_path=str(source), candidate_sha256=digest,
        candidate_source_path=str(source_snapshot),
        candidate_identity_verified=verified, output_dir=str(result.output_dir),
        evaluation_path=str(result.output_dir / "evaluation.json"), error=error,
        reference=reference_identity,
    )
    with (result.output_dir / "feedback.json").open("x", encoding="utf-8") as handle:
        json.dump(feedback.to_dict(), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return feedback


__all__ = ["CandidateFeedback", "evaluate_candidate"]
