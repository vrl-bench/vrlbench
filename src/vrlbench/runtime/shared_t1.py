from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from vrlbench.runtime.budget import ACTION_ATTEMPT_BUDGET_UNIT, AO_BUDGET_UNIT


def validate_formal_trial_environment_accounting(
    metadata: Mapping[str, Any] | None,
    *,
    benchmark: str,
    source: str,
    expected_max_environment_steps: int,
    success: bool,
    expected_trial_index: int,
) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{source}: shared T1 lacks environment accounting")
    horizon = int(expected_max_environment_steps)
    if horizon <= 0:
        raise ValueError("shared T1 environment horizon must be positive")
    benchmark = benchmark.strip().lower()
    if benchmark not in {"miniwob", "webshop", "webarena", "osworld"}:
        raise ValueError(f"unsupported shared T1 benchmark: {benchmark}")

    def integer(field: str) -> int:
        value = metadata.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{source}: {field} must be an integer")
        return value

    if integer("trial_index") != expected_trial_index:
        raise ValueError(f"{source}: shared trial index differs")
    if integer("environment_resets") != 1:
        raise ValueError(f"{source}: shared T1 must contain one reset")
    if integer("max_environment_resets") != 1:
        raise ValueError(f"{source}: shared T1 reset horizon differs")
    steps = integer("environment_steps")
    if not 0 <= steps <= horizon:
        raise ValueError(f"{source}: shared T1 exceeds the action horizon")
    if integer("max_environment_steps") != horizon:
        raise ValueError(f"{source}: shared T1 action horizon differs")
    budget_unit = metadata.get("budget_unit")
    if budget_unit not in {AO_BUDGET_UNIT, ACTION_ATTEMPT_BUDGET_UNIT}:
        raise ValueError(f"{source}: shared T1 uses a different budget unit")
    if budget_unit == ACTION_ATTEMPT_BUDGET_UNIT:
        invalid = integer("invalid_action_attempts")
        attempts = integer("action_attempts")
        if invalid < 0 or attempts != steps + invalid or attempts > horizon:
            raise ValueError(f"{source}: action attempts disagree with environment steps and invalid responses")
        if integer("remaining_environment_steps") != horizon - attempts:
            raise ValueError(f"{source}: remaining action budget is inconsistent")
        events = metadata.get("invalid_action_events")
        if not isinstance(events, list) or len(events) != invalid:
            raise ValueError(f"{source}: invalid response events disagree with action accounting")
        if integer("invalid_actor_responses") != invalid:
            raise ValueError(f"{source}: invalid response count disagrees with action accounting")
        last_environment_step = 0
        for index, event in enumerate(events):
            if not isinstance(event, Mapping):
                raise ValueError(f"{source}: invalid response event must be an object")
            position = event.get("after_environment_step")
            before = event.get("budget_used_before")
            after = event.get("budget_used_after")
            if (type(position) is not int or not last_environment_step <= position <= steps
                    or type(before) is not int or type(after) is not int
                    or before != position + index or after != before + 1
                    or after > attempts
                    or not isinstance(event.get("raw_response"), str)
                    or not isinstance(event.get("reason"), str) or not event["reason"].strip()):
                raise ValueError(f"{source}: invalid response event has inconsistent position or budget")
            last_environment_step = position
        if metadata.get("trial_close_reason") == "ao_horizon" and attempts != horizon:
            raise ValueError(f"{source}: action horizon reported before the budget was exhausted")

    terminated = metadata.get("environment_terminated")
    truncated = metadata.get("environment_truncated")
    close_reason = metadata.get("trial_close_reason")
    protocol_failure = metadata.get("protocol_failure")
    evidence_eligible = metadata.get("policy_evidence_eligible")
    if not isinstance(terminated, bool) or not isinstance(truncated, bool):
        raise ValueError(f"{source}: termination state is missing")
    if terminated and truncated:
        raise ValueError(f"{source}: a trial cannot terminate and truncate")
    if not isinstance(close_reason, str) or not close_reason:
        raise ValueError(f"{source}: trial close reason is missing")
    if not isinstance(protocol_failure, bool) or not isinstance(
        evidence_eligible, bool
    ):
        raise ValueError(f"{source}: trial close state is incomplete")

    states = {
        "environment_success": (True, True, False, False, True),
        "environment_terminal_failure": (False, True, False, False, True),
        "ao_horizon": (False, False, True, False, True),
    }
    if benchmark in {"webarena", "osworld"}:
        states["environment_truncated"] = (False, False, True, False, True)
        if budget_unit == ACTION_ATTEMPT_BUDGET_UNIT:
            # Empty replies can exhaust the budget without any native action.
            # Keep that valid trial, but do not invent a failure trace for memory.
            states["ao_horizon"] = (False, False, True, False, steps > 0)
    if budget_unit == AO_BUDGET_UNIT:
        states["actor_protocol_failure"] = (False, False, False, True, steps > 0)
        if benchmark == "miniwob":
            states["actor_decision_cap"] = (False, False, False, False, True)
        else:
            states["provider_call_safety_cap"] = (
                False,
                False,
                False,
                True,
                steps > 0,
            )
    expected = states.get(close_reason)
    actual = (
        bool(success),
        terminated,
        truncated,
        protocol_failure,
        evidence_eligible,
    )
    if expected is None or actual != expected:
        raise ValueError(f"{source}: invalid trial close state")
    return dict(metadata)


def validate_shared_t1_environment_accounting(
    metadata: Mapping[str, Any] | None,
    *,
    benchmark: str,
    source: str,
    expected_max_environment_steps: int,
    success: bool,
) -> dict[str, Any]:
    return validate_formal_trial_environment_accounting(
        metadata,
        benchmark=benchmark,
        source=source,
        expected_max_environment_steps=expected_max_environment_steps,
        success=success,
        expected_trial_index=0,
    )


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def validate_shared_t1_source(
    root: str | Path,
    *,
    benchmark: str,
    model: str,
    max_environment_steps: int,
    require_complete_accounting: bool,
    expected_provider: str | None = None,
    expected_base_url: str | None = None,
    expected_trial_seed_mode: str | None = None,
    **_unused: Any,
) -> None:
    del expected_base_url, expected_trial_seed_mode
    source_root = Path(root).expanduser().resolve()
    episodes_path = source_root / "episodes.json"
    results_path = source_root / "results.json"
    if not episodes_path.is_file() or not results_path.is_file():
        raise FileNotFoundError(
            f"shared T1 requires episodes.json and results.json: {source_root}"
        )
    result = _load_object(results_path)
    run = result.get("run_meta")
    run = run if isinstance(run, Mapping) else {}
    source_benchmark = str(
        result.get("benchmark") or run.get("benchmark") or benchmark
    ).lower()
    source_model = str(result.get("model") or run.get("model") or "")
    if source_benchmark != benchmark.lower() or source_model != model:
        raise ValueError("shared T1 benchmark or model differs")
    if str(result.get("er") or run.get("er") or "none") != "none":
        raise ValueError("shared T1 must come from the memory-free condition")
    if str(result.get("controller") or run.get("controller") or "none") != "none":
        raise ValueError("shared T1 must come from the controller-free condition")
    if result.get("checkpoint") is not False:
        raise ValueError("shared T1 source is incomplete")
    recorded_horizon = result.get("max_environment_steps")
    if recorded_horizon is not None and int(recorded_horizon) != int(
        max_environment_steps
    ):
        raise ValueError("shared T1 action horizon differs")
    if expected_provider is not None:
        provider = str(result.get("model_provider") or run.get("model_provider") or "")
        if provider and provider != expected_provider:
            raise ValueError("shared T1 provider differs")
    if require_complete_accounting:
        resource = result.get("resource_usage")
        usage = resource.get("llm") if isinstance(resource, Mapping) else None
        validate_actor_only_t1_usage(
            usage,
            source=str(results_path),
            require_complete_accounting=True,
        )
        if not isinstance(usage, Mapping) or usage.get("provider_usage_complete") is not True:
            raise ValueError("shared T1 provider token counts are incomplete")
def validate_actor_only_t1_usage(
    usage: Mapping[str, Any] | None,
    *,
    source: str,
    require_complete_accounting: bool,
) -> None:
    if not isinstance(usage, Mapping):
        if require_complete_accounting:
            raise ValueError(f"{source}: shared T1 lacks token counts")
        return
    total_tokens = usage.get("total_tokens")
    if (
        isinstance(total_tokens, bool)
        or not isinstance(total_tokens, int)
        or total_tokens < 0
    ):
        raise ValueError(f"{source}: shared T1 total token count is invalid")
    if require_complete_accounting and usage.get("provider_usage_complete") is not True:
        raise ValueError(f"{source}: shared T1 token count is incomplete")
