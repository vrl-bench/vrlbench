from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from vrlbench.runtime.loop import EpisodeResult

def atomic_write_json(path: str | Path, payload: Any, *, indent: int=2) -> None:
    """Write JSON by atomic replace so interrupted runs keep valid outputs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    try:
        with open(tmp, 'w') as f:
            json.dump(payload, f, indent=indent)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass

def vrl_task_sr(episodes: List[EpisodeResult]) -> float:
    """VRL Task SR: fraction of episodes solved within T trials.

    Args:
        episodes: List of EpisodeResult from run_episode().

    Returns:
        Success rate in [0, 1].
    """
    if not episodes:
        return 0.0
    return sum((e.solved for e in episodes)) / len(episodes)

def mean_censored_attempt(episodes: List[EpisodeResult], total_trials: int) -> Optional[float]:
    """Mean solve attempt with unsolved episodes counted as T+1."""
    if not episodes:
        return None
    attempts = [int(episode.solve_trial) + 1 if episode.solve_trial is not None else total_trials + 1 for episode in episodes]
    return sum(attempts) / len(attempts)

def completion_metrics(episodes: List[EpisodeResult], total_trials: int) -> Dict[str, Any]:
    """Return SR@T and AvgT@T with their aggregate counts."""
    return {'sr_at_t': vrl_task_sr(episodes), 'avg_t_at_t': mean_censored_attempt(episodes, total_trials=total_trials), 'num_solved': sum((episode.solved for episode in episodes)), 'num_total': len(episodes)}

def resource_usage_summary(episodes: List[EpisodeResult]) -> Dict[str, Any]:
    total_tokens = 0
    api_calls = 0
    environment_transitions = 0
    provider_usage_complete = True
    for episode in episodes:
        usage = episode.resource_usage if isinstance(episode.resource_usage, dict) else {}
        value = usage.get('total_tokens', 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            value = 0
            provider_usage_complete = False
        total_tokens += value
        calls = usage.get('api_calls', 0)
        if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
            calls = 0
            provider_usage_complete = False
        api_calls += calls
        if usage.get('provider_usage_complete') is not True:
            provider_usage_complete = False
        for trial in episode.trials:
            metadata = trial.trial_metadata if isinstance(trial.trial_metadata, dict) else {}
            transitions = metadata.get('environment_steps', 0)
            if isinstance(transitions, bool) or not isinstance(transitions, int) or transitions < 0:
                transitions = 0
            environment_transitions += transitions
    return {'num_instances': len(episodes), 'llm': {'total_tokens': total_tokens, 'api_calls': api_calls, 'provider_usage_complete': provider_usage_complete}, 'environment': {'environment_transitions': environment_transitions}}

def per_trial_sr(episodes: List[EpisodeResult], total_trials: int) -> Dict[int, float]:
    """Compute success rate at each trial index (learning curve).

    An episode is counted as "solved at trial t" if it was solved on
    trial t or earlier (cumulative). This reproduces the learning curves
    in Figures 1a and 3.

    Args:
        episodes:     List of EpisodeResult.
        total_trials: Total budget T (defines the x-axis length).

    Returns:
        Dict mapping trial index (1-indexed, paper convention) → cumulative SR.
    """
    result: Dict[int, float] = {}
    for t in range(total_trials):
        solved = sum((any((r.trial <= t and r.success for r in e.trials)) for e in episodes))
        result[t + 1] = solved / len(episodes) if episodes else 0.0
    return result

def write_env_results(episodes: List[EpisodeResult], output_dir: str, total_trials: int, names: Optional[List[str]]=None) -> None:
    """Write ``env_results_trial_{t}.json`` for each trial.

    Format matches the trial-level result convention consumed by
    ``export_rigorous_appendix_tables.py`` and other analysis scripts:
    a JSON list of ``{"name": ..., "is_success": bool}`` per instance.

    Args:
        episodes:     Ordered list of EpisodeResult.
        output_dir:   Directory to write files into.
        total_trials: Budget T (writes trial 0 through T-1).
        names:        Optional instance labels; defaults to ``sample_XXXX``.
    """
    os.makedirs(output_dir, exist_ok=True)
    if names is None:
        names = [f'sample_{i:04d}' for i in range(len(episodes))]
    for t in range(total_trials):
        rows = []
        for ep, name in zip(episodes, names):
            solved_by_t = any((r.trial <= t and r.success for r in ep.trials))
            rows.append({'name': name, 'is_success': solved_by_t})
        path = Path(output_dir) / f'env_results_trial_{t}.json'
        atomic_write_json(path, rows)
