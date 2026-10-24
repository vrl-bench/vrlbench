"""统计功能的离线回归；只使用合成的已完成结果，不执行模型。"""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from test_sdk_results import make_case, make_result
from vrlbench.cli import main
from vrlbench.results import compare_results
from vrlbench.statistics import analyze_results, posthoc_power_mcnemar

try:
    from scipy import stats
except ImportError:
    stats = None


def paired_runs(reference, target, *, tasks=None):
    tasks = tasks if tasks is not None else ["task"] * len(reference)
    runs = []
    for outcomes, method in ((reference, "reflexion"), (target, "ace")):
        cases = [make_case(str(index), [success], task)
                 for index, (success, task) in enumerate(zip(outcomes, tasks))]
        runs.append(make_result(cases, trials=1, method=method))
    return runs


class ConditionalPowerTests(unittest.TestCase):
    @unittest.skipUnless(stats is not None, "requires the optional analysis dependencies")
    def test_six_discordant_pairs_have_the_analytic_power(self):
        for probability in (0, .2, .5, .8, 1):
            with self.subTest(probability=probability):
                self.assertAlmostEqual(posthoc_power_mcnemar(6, probability),
                                       probability ** 6 + (1 - probability) ** 6)
        # 原脚本使用严格的小于号；恰好等于阈值时不拒绝原假设。
        self.assertEqual(posthoc_power_mcnemar(6, .8, alpha=.03125), 0)

    @unittest.skipUnless(stats is not None, "requires the optional analysis dependencies")
    def test_small_samples_have_no_rejection_region(self):
        for count in range(6):
            for probability in (0, .5, 1):
                with self.subTest(count=count, probability=probability):
                    self.assertEqual(posthoc_power_mcnemar(count, probability), 0)

    @unittest.skipUnless(stats is not None, "requires the optional analysis dependencies")
    def test_power_matches_original_scipy_enumeration_and_symmetry(self):
        for count, probability, alpha in ((8, .7, .05), (20, .8, .1),
                                          (12, 0, .05), (12, .5, .05), (12, 1, .05)):
            expected = float(sum(stats.binom.pmf(k, count, probability)
                                 for k in range(count + 1)
                                 if stats.binomtest(k, count, .5).pvalue < alpha))
            with self.subTest(count=count, probability=probability, alpha=alpha):
                actual = posthoc_power_mcnemar(count, probability, alpha)
                self.assertAlmostEqual(actual, expected)
                self.assertAlmostEqual(actual, posthoc_power_mcnemar(count, 1 - probability, alpha))
                self.assertGreaterEqual(actual, 0)
                self.assertLessEqual(actual, 1)

    def test_invalid_power_parameters_are_rejected_before_dependency_use(self):
        self.assertEqual(posthoc_power_mcnemar(0, .5), 0)
        for count in (-1, True, 1.5, "6", None):
            with self.subTest(count=count), self.assertRaises((TypeError, ValueError)):
                posthoc_power_mcnemar(count, .5)
        for probability in (-.1, 1.1, True, float("nan"), float("inf"), ".5", None):
            with self.subTest(probability=probability), self.assertRaises((TypeError, ValueError)):
                posthoc_power_mcnemar(0, probability)
        for alpha in (0, 1, -.1, True, float("nan"), float("inf"), ".05", None):
            with self.subTest(alpha=alpha), self.assertRaises((TypeError, ValueError)):
                posthoc_power_mcnemar(0, .5, alpha)


@unittest.skipUnless(stats is not None, "requires the optional analysis dependencies")
class AnalyzeResultsTests(unittest.TestCase):
    def test_case_pairing_is_order_invariant_and_reference_need_not_be_retry(self):
        reference, ace = paired_runs([False, True, False, True], [True, False, True, True],
                                     tasks=["a", "a", "b", "b"])
        vex2 = make_result([make_case(str(index), [True], task)
                            for index, task in enumerate(["a", "a", "b", "b"])],
                           trials=1, method="vex2")
        runs = [reference.to_dict(), ace.to_dict(), vex2.to_dict()]
        before = deepcopy(runs)
        descriptive = compare_results(runs)
        self.assertNotIn("statistics", descriptive)
        analyzed = analyze_results(runs, analysis_unit="case", resamples=200, seed=7)
        self.assertEqual({key: value for key, value in analyzed.items() if key != "statistics"},
                         descriptive)
        self.assertEqual(runs, before)
        statistics = analyzed["statistics"]
        self.assertEqual((statistics["analysis_unit"], statistics["metric"], statistics["n_units"]),
                         ("case", "sr_at_t", 4))
        self.assertEqual([(effect["reference"], effect["target"])
                          for effect in statistics["effects"]],
                         [("reflexion", "ace"), ("reflexion", "vex2")])
        effect = statistics["effects"][0]
        self.assertEqual(effect["estimate_pp"], 25)
        self.assertEqual((effect["positive_units"], effect["negative_units"], effect["tied_units"]),
                         (2, 1, 1))
        self.assertEqual(effect["exact_test"], "exact_mcnemar")
        self.assertEqual(effect["exact_p_value"], stats.binomtest(2, 3, .5).pvalue)
        self.assertEqual(effect["power"]["alternative_probability"], 2 / 3)
        self.assertEqual(effect["power"]["conditioned_units"], 3)
        self.assertEqual(effect["power"]["basis"], "observed_alternative_conditional")
        self.assertEqual(effect["power"]["test"], "exact_mcnemar")
        for index, run in enumerate(runs):
            run["cases"] = run["cases"][index:] + run["cases"][:index]
        runs[0]["cases"].reverse()
        self.assertEqual(analyze_results(runs, analysis_unit="case", resamples=200, seed=7), analyzed)

    def test_task_analysis_weights_task_means_not_case_counts(self):
        runs = paired_runs([False, False, False, True], [True, True, True, False],
                           tasks=["a", "a", "a", "b"])
        case = analyze_results(runs, analysis_unit="case", resamples=200)["statistics"]
        task = analyze_results(runs, analysis_unit="task", resamples=200)["statistics"]
        self.assertEqual(case["effects"][0]["estimate_pp"], 50)
        self.assertEqual((task["metric"], task["n_units"]), ("macro_task_sr_at_t", 2))
        effect = task["effects"][0]
        self.assertEqual(effect["estimate_pp"], 0)
        self.assertEqual((effect["positive_units"], effect["negative_units"], effect["tied_units"]),
                         (1, 1, 0))
        self.assertEqual(effect["exact_test"], "exact_sign")
        self.assertEqual(effect["exact_p_value"], 1)
        self.assertEqual(effect["power"]["test"], "exact_sign")

    def test_task_mean_and_sign_test_report_distinct_estimands(self):
        reference, target, tasks = [False], [True], ["positive"]
        for index in range(9):
            reference.extend([True] + [False] * 99)
            target.extend([False] * 100)
            tasks.extend([f"negative_{index}"] * 100)
        runs = paired_runs(reference, target, tasks=tasks)
        statistics = analyze_results(runs, analysis_unit="task", resamples=200)["statistics"]
        effect = statistics["effects"][0]
        # 一项任务的大幅改善可使平均差为正，但并不表示多数任务改善。
        self.assertEqual(statistics["n_units"], 10)
        self.assertAlmostEqual(effect["estimate_pp"], 9.1)
        self.assertEqual((effect["positive_units"], effect["negative_units"], effect["tied_units"]),
                         (1, 9, 0))
        self.assertEqual(effect["exact_test"], "exact_sign")
        self.assertEqual(effect["exact_p_value"], .021484375)
        self.assertEqual(statistics["effect_estimand"], "mean_paired_success_difference")
        self.assertEqual(effect["exact_test_null"], "positive_probability_given_non_tie=0.5")
        self.assertEqual(effect["power"]["estimand"], "positive_probability_given_non_tie")

    def test_tied_units_have_no_observed_alternative_probability(self):
        runs = paired_runs([True, False], [True, False], tasks=["a", "b"])
        for unit in ("case", "task"):
            with self.subTest(unit=unit):
                effect = analyze_results(runs, analysis_unit=unit, resamples=50)["statistics"]["effects"][0]
                self.assertEqual((effect["estimate_pp"], effect["ci_lower_pp"], effect["ci_upper_pp"]),
                                 (0, 0, 0))
                self.assertEqual((effect["positive_units"], effect["negative_units"], effect["tied_units"]),
                                 (0, 0, 2))
                self.assertEqual(effect["exact_p_value"], 1)
                self.assertEqual(effect["power"]["estimate"], 0)
                self.assertEqual(effect["power"]["conditioned_units"], 0)
                self.assertIsNone(effect["power"]["alternative_probability"])

    def test_analysis_unit_and_numeric_parameters_must_be_explicit_and_valid(self):
        runs = paired_runs([False, True], [True, False])
        with self.assertRaises(TypeError):
            analyze_results(runs)
        invalid = [{"analysis_unit": value} for value in (None, "trial", "CASE")]
        invalid += [{"resamples": value} for value in (0, -1, True, 1.5)]
        invalid += [{"seed": value} for value in (-1, True, 1.5)]
        invalid += [{"alpha": value} for value in (0, 1, True, float("nan"))]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises((TypeError, ValueError)):
                analyze_results(runs, **{"analysis_unit": "case", **options})

    def test_analysis_requires_two_runs_and_two_independent_units(self):
        one_case = paired_runs([False], [True])
        one_task = paired_runs([False, True], [True, True])
        for runs, unit in ((one_case, "case"), (one_task, "task"),
                           (one_task[:1], "case"), ([], "case")):
            with self.subTest(unit=unit, runs=len(runs)), self.assertRaises(ValueError):
                analyze_results(runs, analysis_unit=unit, resamples=50)

    def test_incomplete_mismatched_and_inconsistent_records_remain_rejected(self):
        reference, target = paired_runs([False, True], [True, False], tasks=["a", "b"])
        variants = []
        incomplete = target.to_dict(); incomplete["status"] = "partial"
        variants.append(incomplete)
        missing_trials = target.to_dict(); missing_trials["cases"][0]["trials"] = []
        variants.append(missing_trials)
        budget = target.to_dict(); budget["config"]["max_environment_steps"] = 11
        variants.append(budget)
        case_id = target.to_dict(); case_id["cases"][0]["case_id"] = "different"
        variants.append(case_id)
        task = target.to_dict(); task["cases"][0]["task"] = "different"
        variants.append(task)
        counts = target.to_dict(); counts["counts"]["solved"] = 0
        variants.append(counts)
        for payload in variants:
            for unit in ("case", "task"):
                with self.subTest(unit=unit, payload=payload), self.assertRaises((ValueError, RuntimeError)):
                    analyze_results([reference, payload], analysis_unit=unit, resamples=50)

    def test_analysis_inherits_shared_component_identity_checks_and_legacy_status(self):
        reference, target = paired_runs([False, True], [True, False])
        recorded = target.to_dict()
        recorded["identity"] = {"plugins": [
            {"role": "benchmark", "source_sha256": "a" * 64},
        ]}
        analyzed = analyze_results([reference, recorded], analysis_unit="case", resamples=50)
        checked = analyzed["component_identity_check"]
        self.assertEqual(checked["status"], "unverified")
        self.assertEqual(checked["components"]["benchmark"]["missing_run_indices"], [0])
        other = reference.to_dict()
        other["identity"] = {"plugins": [
            {"role": "benchmark", "source_sha256": "b" * 64},
        ]}
        with self.assertRaisesRegex(ValueError, "different benchmark plugin source"):
            analyze_results([other, recorded], analysis_unit="case", resamples=50)

    def test_bootstrap_is_seeded_and_reports_its_settings(self):
        runs = paired_runs([False, False, True, True], [True, False, False, True])
        first = analyze_results(runs, analysis_unit="case", resamples=127, seed=23, alpha=.2)
        second = analyze_results(runs, analysis_unit="case", resamples=127, seed=23, alpha=.2)
        self.assertEqual(first, second)
        statistics = first["statistics"]
        self.assertEqual(statistics["confidence_level"], .8)
        self.assertEqual(statistics["bootstrap"],
                         {"method": "paired_percentile", "resamples": 127, "seed": 23})
        effect = statistics["effects"][0]
        self.assertLessEqual(effect["ci_lower_pp"], effect["estimate_pp"])
        self.assertGreaterEqual(effect["ci_upper_pp"], effect["estimate_pp"])

    def test_cli_statistics_are_opt_in_and_require_an_analysis_unit(self):
        def command(*arguments):
            output, errors = io.StringIO(), io.StringIO()
            with redirect_stdout(output), redirect_stderr(errors):
                status = main(list(arguments))
            return status, output.getvalue(), errors.getvalue()

        runs = paired_runs([False, True], [True, False], tasks=["a", "b"])
        with tempfile.TemporaryDirectory() as temporary:
            paths = [str(run.write(Path(temporary) / f"run-{index}.json"))
                     for index, run in enumerate(runs)]
            status, output, errors = command("compare", *paths)
            self.assertEqual((status, errors), (0, ""))
            self.assertNotIn("statistics", json.loads(output))
            for unit in ("case", "task"):
                with self.subTest(unit=unit):
                    status, output, errors = command("compare", *paths, "--statistics",
                                                     "--analysis-unit", unit, "--resamples", "50")
                    self.assertEqual((status, errors), (0, ""))
                    self.assertEqual(json.loads(output)["statistics"]["analysis_unit"], unit)
            status, output, errors = command("compare", *paths, "--statistics")
            self.assertEqual((status, output), (2, ""))
            self.assertIn("--analysis-unit", errors)


if __name__ == "__main__":
    unittest.main()
