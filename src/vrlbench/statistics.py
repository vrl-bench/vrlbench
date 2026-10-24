"""Optional paired inference for completed runs, without model calls.

The conditional McNemar power calculation retains the original experiment
analysis's two-sided rejection rule. The adapter uses matched run records,
without historical benchmark loaders or results.
"""

from __future__ import annotations

from collections import defaultdict
from math import fsum, isfinite
from numbers import Real
from typing import Any, Mapping, Sequence

from .results import SimulationResult, compare_results


def _probability(value: float, name: str, *, open_interval: bool = False) -> None:
    if (isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value)
            or not (0 < value < 1 if open_interval else 0 <= value <= 1)):
        bounds = "(0, 1)" if open_interval else "[0, 1]"
        raise ValueError(f"{name} must be a finite probability in {bounds}")


def _scipy_stats():
    try:
        from scipy import stats
    except ImportError as exc:
        raise RuntimeError("Statistical analysis requires 'vrlbench[analysis]'") from exc
    return stats


def posthoc_power_mcnemar(n_disc: int, p_alt: float, alpha: float = 0.05) -> float:
    """Exact conditional power of the two-sided discordant-pair test.

    Condition on n_disc discordant pairs. Under the alternative, the target
    wins each pair with probability p_alt. Sum its binomial probabilities
    over the null rejection set, using p-value < alpha as in the original
    analysis. Zero discordant pairs return zero.

    An observed p_alt yields post-hoc power, not an independent measure of
    effectiveness or a sample-size recommendation.
    """
    if type(n_disc) is not int or n_disc < 0:
        raise ValueError("n_disc must be a nonnegative integer")
    _probability(p_alt, "p_alt")
    _probability(alpha, "alpha", open_interval=True)
    if n_disc == 0:
        return 0.0
    stats = _scipy_stats()
    reject = [k for k in range(n_disc + 1)
              if stats.binomtest(k, n_disc, 0.5).pvalue < alpha]
    return min(1.0, max(0.0, float(fsum(
        stats.binom.pmf(k, n_disc, p_alt) for k in reject
    ))))


def _success_units(cases: Mapping[str, dict], unit: str) -> list[float]:
    if unit == "case":
        return [float(cases[key]["trials"][-1]["success"]) for key in sorted(cases)]
    tasks: dict[str, list[float]] = defaultdict(list)
    for key in sorted(cases):
        case = cases[key]
        tasks[case["task"]].append(float(case["trials"][-1]["success"]))
    return [fsum(tasks[key]) / len(tasks[key]) for key in sorted(tasks)]


def _paired_interval(differences: list[float], *, resamples: int,
                     seed: int, alpha: float) -> tuple[float, float]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Statistical analysis requires 'vrlbench[analysis]'") from exc
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    # Resample complete paired units, never individual trials. Batch temporary
    # allocations so a larger case set need not allocate all draws at once.
    batch_size = max(1, min(256, 1_000_000 // len(values)))
    estimates = np.empty(resamples, dtype=float)
    for offset in range(0, resamples, batch_size):
        count = min(batch_size, resamples - offset)
        draws = rng.integers(0, len(values), size=(count, len(values)))
        estimates[offset:offset + count] = values[draws].mean(axis=1)
    lower, upper = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    return float(100 * lower), float(100 * upper)


def analyze_results(
    results: Sequence[SimulationResult | Mapping[str, Any]], *,
    analysis_unit: str, resamples: int = 10_000, seed: int = 0, alpha: float = 0.05,
) -> dict:
    """Add paired SR intervals, exact tests and conditional power to a comparison.

    The first run is the reference. Select independent cases or task-family
    means explicitly. Task means receive equal weight regardless of case
    count. Default descriptive comparisons remain unchanged.
    """
    if analysis_unit not in {"case", "task"}:
        raise ValueError("analysis_unit must be case or task")
    if type(resamples) is not int or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    _probability(alpha, "alpha", open_interval=True)
    normalized = [result if isinstance(result, SimulationResult)
                  else SimulationResult(dict(result)) for result in results]
    if len(normalized) < 2:
        raise ValueError("statistical analysis requires a reference and a target run")
    if not normalized[0].cases:
        raise ValueError("statistical analysis requires at least two analysis units")
    compared = compare_results(normalized)
    case_maps = [{row["case_id"]: row for row in result.cases} for result in normalized]
    units = [_success_units(cases, analysis_unit) for cases in case_maps]
    if len(units[0]) < 2:
        raise ValueError("statistical analysis requires at least two analysis units")
    stats = _scipy_stats()
    exact_test = "exact_mcnemar" if analysis_unit == "case" else "exact_sign"
    effects = []
    for index, target in enumerate(units[1:], start=1):
        differences = [left - right for left, right in zip(target, units[0])]
        lower, upper = _paired_interval(differences, resamples=resamples, seed=seed, alpha=alpha)
        positive = sum(value > 0 for value in differences)
        negative = sum(value < 0 for value in differences)
        n_disc = positive + negative
        p_alt = positive / n_disc if n_disc else None
        effects.append({
            "target": compared["rows"][index]["method"],
            "reference": compared["rows"][0]["method"],
            "estimate_pp": 100 * fsum(differences) / len(differences),
            "ci_lower_pp": lower, "ci_upper_pp": upper,
            "exact_test": exact_test,
            "exact_test_null": "positive_probability_given_non_tie=0.5",
            "exact_p_value": float(stats.binomtest(positive, n_disc, 0.5).pvalue) if n_disc else 1.0,
            "positive_units": positive, "negative_units": negative,
            "tied_units": len(differences) - n_disc,
            "power": {
                "estimate": posthoc_power_mcnemar(n_disc, p_alt if p_alt is not None else 0.5, alpha),
                "alternative_probability": p_alt, "conditioned_units": n_disc,
                "basis": "observed_alternative_conditional", "test": exact_test,
                "estimand": "positive_probability_given_non_tie",
            },
        })
    compared["statistics"] = {
        "analysis_unit": analysis_unit,
        "metric": "sr_at_t" if analysis_unit == "case" else "macro_task_sr_at_t",
        "effect_estimand": "mean_paired_success_difference",
        "n_units": len(units[0]), "confidence_level": 1 - alpha,
        "bootstrap": {"method": "paired_percentile", "resamples": resamples, "seed": seed},
        "multiplicity_adjusted": False, "effects": effects,
    }
    return compared


__all__ = ["analyze_results", "posthoc_power_mcnemar"]
