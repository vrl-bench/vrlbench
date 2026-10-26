
from __future__ import annotations

from typing import Any, Iterable


MINIWOB_PAIRED_TRIAL_STRIDE = 1_000_003


def miniwob_reset_seed(base_seed: int, trial_index: int, *, mode: str) -> int:
    """Return the reset seed for one MiniWoB trial under the declared mode."""

    base_seed = int(base_seed)
    trial_index = int(trial_index)
    if trial_index < 0:
        raise ValueError("trial_index must be non-negative")
    if mode == "fixed_instance":
        return base_seed
    if mode == "paired_trial":
        return base_seed + MINIWOB_PAIRED_TRIAL_STRIDE * trial_index
    raise ValueError(f"unsupported MiniWoB trial_seed_mode: {mode!r}")


def miniwob_episode_seed_issues(
    episodes: Iterable[Any],
    *,
    expected_mode: str,
) -> list[str]:
    """Validate per-trial reset metadata, including success-stop prefixes."""

    issues: list[str] = []
    for episode_index, episode in enumerate(episodes):
        trials = getattr(episode, "trials", None)
        if not isinstance(trials, list) or not trials:
            issues.append(f"episodes[{episode_index}] has no executed trials")
            continue
        reset_seeds: list[int] = []
        episode_base_seed: int | None = None
        for position, trial in enumerate(trials):
            metadata = getattr(trial, "trial_metadata", None)
            if not isinstance(metadata, dict):
                issues.append(
                    f"episodes[{episode_index}].trials[{position}] lacks trial_metadata"
                )
                continue
            if metadata.get("trial_index") != position:
                issues.append(
                    f"episodes[{episode_index}].trials[{position}].trial_index "
                    f"must equal {position}"
                )
            if metadata.get("trial_seed_mode") != expected_mode:
                issues.append(
                    f"episodes[{episode_index}].trials[{position}].trial_seed_mode "
                    f"must equal {expected_mode}"
                )
            base_seed = metadata.get("base_seed")
            reset_seed = metadata.get("reset_seed")
            if not isinstance(base_seed, int) or isinstance(base_seed, bool):
                issues.append(
                    f"episodes[{episode_index}].trials[{position}].base_seed is invalid"
                )
                continue
            if not isinstance(reset_seed, int) or isinstance(reset_seed, bool):
                issues.append(
                    f"episodes[{episode_index}].trials[{position}].reset_seed is invalid"
                )
                continue
            if episode_base_seed is None:
                episode_base_seed = base_seed
            elif base_seed != episode_base_seed:
                issues.append(
                    f"episodes[{episode_index}] changes base_seed within an episode"
                )
            expected_seed = miniwob_reset_seed(
                base_seed,
                position,
                mode=expected_mode,
            )
            if reset_seed != expected_seed:
                issues.append(
                    f"episodes[{episode_index}].trials[{position}].reset_seed "
                    f"must equal {expected_seed}"
                )
            reset_seeds.append(reset_seed)
        if expected_mode == "paired_trial" and len(reset_seeds) != len(set(reset_seeds)):
            issues.append(
                f"episodes[{episode_index}] reuses a reset seed across paired trials"
            )
    return issues


def miniwob_serialized_seed_schedule(
    episode_rows: Any,
) -> tuple[dict[str, tuple[int, ...]], list[str]]:
    """Extract per-instance reset schedules from serialized episodes.json."""

    if not isinstance(episode_rows, list):
        return {}, ["episodes.json is not a list"]
    schedule: dict[str, tuple[int, ...]] = {}
    issues: list[str] = []
    for index, row in enumerate(episode_rows):
        if not isinstance(row, dict):
            issues.append(f"episodes[{index}] is not an object")
            continue
        name = str(row.get("name") or f"episode_{index:04d}")
        trials = row.get("trials")
        if not isinstance(trials, list):
            issues.append(f"{name} lacks trials")
            continue
        seeds: list[int] = []
        for trial_index, trial in enumerate(trials):
            metadata = trial.get("trial_metadata") if isinstance(trial, dict) else None
            reset_seed = metadata.get("reset_seed") if isinstance(metadata, dict) else None
            if not isinstance(reset_seed, int) or isinstance(reset_seed, bool):
                issues.append(f"{name} trial {trial_index} lacks a valid reset_seed")
                continue
            seeds.append(reset_seed)
        schedule[name] = tuple(seeds)
    return schedule, issues


def paired_miniwob_schedule_issues(
    baseline_rows: Any,
    compared_rows: Any,
) -> list[str]:
    """Compare actual reset seeds over each success-stop common prefix."""

    baseline, issues = miniwob_serialized_seed_schedule(baseline_rows)
    compared, compared_issues = miniwob_serialized_seed_schedule(compared_rows)
    issues.extend(compared_issues)
    if set(baseline) != set(compared):
        issues.append("MiniWoB episode identities differ across methods")
        return issues
    for name in sorted(baseline):
        common = min(len(baseline[name]), len(compared[name]))
        if baseline[name][:common] != compared[name][:common]:
            issues.append(
                f"{name} reset seeds differ across methods within the executed prefix"
            )
    return issues
