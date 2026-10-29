"""Read recorded trial inputs and behavior without executing a strategy."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .results import SimulationResult


TRIAL_FIELDS = (
    "trial", "success", "context", "system_context", "user_context",
    "trajectory", "actor_trajectory", "policy_trajectory", "trial_metadata",
    "llm_usage", "controller_decision",
)


def inspect_case(result: SimulationResult, case_id: str | None = None) -> dict[str, Any]:
    """Return an index or one case's recorded fields, including partial runs.

    Absent fields remain ``None``; a recorded empty context remains ``""``.
    This view neither reconstructs missing memory nor infers why an action
    was chosen. Trial numbers for display are one-based; ``trial`` is the
    unchanged zero-based value in the evaluation record.
    """
    payload = result.to_dict()
    cases = payload.get("cases", [])
    if not isinstance(cases, list) or any(not isinstance(case, dict) for case in cases):
        raise ValueError("inspection requires a list of case records")
    identities = [case.get("case_id") for case in cases]
    if any(not isinstance(value, str) or not value for value in identities):
        raise ValueError("inspection requires nonempty string case IDs")
    if len(set(identities)) != len(identities):
        raise ValueError("inspection requires unique case IDs")
    config = payload.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("inspection requires a config mapping")
    view = {
        "status": result.status,
        "counts": result.counts,
        "model": config.get("model"),
        "benchmark": config.get("benchmark"),
        "memory": config.get("memory"),
        "trial_budget": payload.get("trials", config.get("trials")),
        "interpretation": "recorded_inputs_and_behavior_not_a_causal_explanation",
    }
    if case_id is None:
        view["cases"] = [{"case_id": case["case_id"], "task": case.get("task")}
                         for case in cases]
        return view
    matching = [case for case in cases if case["case_id"] == case_id]
    if not matching:
        raise ValueError(f"case {case_id!r} is not recorded; omit --case to list case IDs")
    case = matching[0]
    rows = case.get("trials", [])
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("inspection requires a list of trial records")
    trials = []
    for row in rows:
        trial = row.get("trial")
        if type(trial) is not int or trial < 0:
            raise ValueError("inspection requires nonnegative integer trial indices")
        if row.get("success") is not None and type(row["success"]) is not bool:
            raise ValueError("a recorded trial outcome must be boolean or unknown")
        trials.append({"trial_number": trial + 1,
                       **{field: deepcopy(row.get(field)) for field in TRIAL_FIELDS}})
    partial = case.get("in_progress_trial")
    if partial is not None:
        budget = view["trial_budget"]
        if (not isinstance(partial, dict) or type(partial.get("trial")) is not int
                or partial["trial"] != len(rows) or type(budget) is not int
                or not 0 <= partial["trial"] < budget
                or (rows and rows[-1].get("success") is True)):
            raise ValueError("an in-progress trial must follow completed trials within the trial budget")
        if ("success" not in partial or partial["success"] is not None
                or partial.get("status") not in {"in_progress", "interrupted"}
                or partial.get("phase") not in {"reset", "invalid_action",
                    "awaiting_environment_feedback", "environment_feedback", "environment_complete"}):
            raise ValueError("an in-progress trial cannot claim a finalized outcome")
        if (not isinstance(partial.get("trial_metadata"), dict)
                or not isinstance(partial.get("actor_trajectory"), str)
                or any(partial.get(key) is not None and not isinstance(partial[key], str)
                       for key in ("context", "system_context", "user_context"))):
            raise ValueError("in-progress memory and action records are malformed")
        partial = {"trial_number": partial["trial"] + 1, **deepcopy(partial)}
    view["case"] = {"case_id": case_id, "task": case.get("task"),
                    "resource_usage": deepcopy(case.get("resource_usage")),
                    "trials": trials}
    if partial is not None:
        view["case"]["in_progress_trial"] = partial
    return view


def render_inspection(view: dict[str, Any]) -> str:
    """Render full stored contexts and traces, not a generated summary."""
    import json

    lines = [f"Simulation status: {view['status']}",
             f"Case counts: {json.dumps(view['counts'], sort_keys=True)}",
             "Recorded inputs and behavior; not a causal explanation.",
             "Missing fields are unknown; recorded empty strings are empty."]
    if "case" not in view:
        lines.append("\nRecorded cases (use --case ID):")
        lines.extend(f"  {case['case_id']}  [{case.get('task')}]" for case in view["cases"])
        return "\n".join(lines)
    case = view["case"]
    lines.append(f"\nCase: {case['case_id']}  Task: {case.get('task')}")
    partial = case.get("in_progress_trial")
    for row in [*case["trials"], *([partial] if partial is not None else [])]:
        outcome = {True: "SUCCESS", False: "FAIL"}.get(row["success"], "UNKNOWN")
        if row is partial:
            interrupted = (row["status"] == "interrupted"
                           or view["status"] in {"failed", "timeout", "timed_out", "cancelled"})
            outcome = ("INCOMPLETE (" + ("interrupted; " if interrupted else "")
                       + "last saved checkpoint: " + row["status"] + ")")
        lines.append(f"\n=== Trial {row['trial_number']} / {view['trial_budget']}: {outcome} ===")
        fields = TRIAL_FIELDS[2:] + (("phase", "pending_action", "environment_result", "error")
                                    if row is partial else ())
        for field in fields:
            value = row.get(field)
            shown = ("[unknown / not recorded]" if value is None else "[empty]" if value == ""
                     else value if isinstance(value, str)
                     else json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
            lines.append(f"\n{field}:\n{shown}")
    return "\n".join(lines)
