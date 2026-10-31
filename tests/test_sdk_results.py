"""Metric and record checks using only the standard library and synthetic cases."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vrlbench.results import EvaluationResult, aggregate_cases, compare_results


def make_case(case_id, outcomes, task="task", usage=None, steps=None):
    return {
        "case_id": case_id, "task": task,
        "trials": [{"trial": index, "success": success, "trajectory": "trace",
                    "trial_metadata": {} if steps is None else {"environment_steps": steps},
                    "llm_usage": {}}
                   for index, success in enumerate(outcomes)],
        "resource_usage": {} if usage is None else usage,
    }


def make_result(cases, trials=3, method="retry", **kwargs):
    payload = {"status": "complete", "config": {
        "model": {"name": "synthetic", "request_options": {}},
        "benchmark": {"name": "synthetic", "params": {}},
        "memory": {"name": method}, "trials": trials, "max_environment_steps": 10,
    }, **aggregate_cases(cases, trials), "cases": cases, **kwargs}
    return EvaluationResult(payload)


class AggregateCasesTests(unittest.TestCase):
    def test_micro_macro_rr_and_censored_trials(self):
        cases = [make_case("a1", [True], "a"), make_case("a2", [False] * 3, "a"),
                 make_case("a3", [False, True], "a"), make_case("b1", [False] * 3, "b")]
        result = aggregate_cases(cases, trials=3)
        self.assertEqual(result["counts"], {"planned": 4, "completed": 4, "solved": 2,
                                            "first_trial_failed": 3, "recovered": 1})
        self.assertEqual(result["metrics"]["sr_at_t"], .5)
        self.assertAlmostEqual(result["metrics"]["rr_at_t"], 1 / 3)
        self.assertEqual(result["metrics"]["avg_t_at_t"], 2.75)
        self.assertAlmostEqual(result["metrics"]["macro_task_sr_at_t"], 1 / 3)

    def test_incomplete_cases_are_not_synthetic_failures(self):
        cases = [make_case("done", [True]), make_case("partial", [False]),
                 make_case("unstarted", [])]
        result = aggregate_cases(cases, 3, planned_cases=4)
        self.assertEqual(result["counts"]["planned"], 4)
        self.assertEqual(result["counts"]["completed"], 1)
        self.assertEqual(result["metrics"]["sr_at_t"], 1)
        self.assertIsNone(result["metrics"]["rr_at_t"])
        self.assertEqual(result["incomplete_case_ids"], ["partial", "unstarted"])
        self.assertEqual(result["metrics_basis"], "completed_cases")

    def test_empty_denominators_are_unknown(self):
        result = aggregate_cases([], 6, planned_cases=10)
        self.assertTrue(all(value is None for value in result["metrics"].values()))
        self.assertEqual(result["counts"]["completed"], 0)
        self.assertIsNone(result["resources"]["total_tokens"])
        self.assertIsNone(result["resources"]["provider_usage_complete"])
        self.assertIsNone(aggregate_cases([make_case("a", [True])], 6)["metrics"]["rr_at_t"])

    def test_final_budget_failure_completes_and_last_trial_success_recovers(self):
        result = aggregate_cases([make_case("fail", [False] * 3),
                                  make_case("recover", [False, False, True])], 3)
        self.assertEqual(result["counts"]["completed"], 2)
        self.assertEqual(result["metrics"]["rr_at_t"], .5)
        self.assertEqual(result["metrics"]["avg_t_at_t"], 3.5)

    def test_invalid_sequence_and_types_are_rejected(self):
        invalid = []
        missing_zero = make_case("a", [False]); missing_zero["trials"][0]["trial"] = 1
        invalid.append(missing_zero)
        gap = make_case("a", [False, True]); gap["trials"][1]["trial"] = 2
        invalid.append(gap)
        duplicate = make_case("a", [False, True]); duplicate["trials"][1]["trial"] = 0
        invalid.append(duplicate)
        boolean_trial = make_case("a", [True]); boolean_trial["trials"][0]["trial"] = False
        invalid.append(boolean_trial)
        integer_success = make_case("a", [1]); invalid.append(integer_success)
        invalid.extend([make_case("a", [True, False]), make_case("a", [False] * 4)])
        for case in invalid:
            with self.subTest(case=case):
                with self.assertRaises((TypeError, ValueError)):
                    aggregate_cases([case], 3)

    def test_duplicate_cases_and_invalid_budgets_are_rejected(self):
        case = make_case("same", [True])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            aggregate_cases([case, deepcopy(case)], 3)
        for budget in (0, -1, True, 1.5):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                aggregate_cases([], budget)
        for planned in (0, -1, True, 1.5):
            with self.subTest(planned=planned), self.assertRaises(ValueError):
                aggregate_cases([case], 3, planned_cases=planned)

    def test_unknown_usage_is_not_reported_as_zero(self):
        known = {"api_calls": 2, "prompt_tokens": 3, "completion_tokens": 4,
                 "total_tokens": 7, "provider_usage_complete": True}
        result = aggregate_cases([make_case("known", [True], usage=known, steps=2),
                                  make_case("unknown", [False], steps=3)], 1)
        resources = result["resources"]
        self.assertIsNone(resources["api_calls"])
        self.assertIsNone(resources["total_tokens"])
        self.assertIsNone(resources["provider_usage_complete"])
        self.assertEqual(resources["known_subtotals"]["total_tokens"], 7)
        self.assertEqual(resources["known_cases"]["total_tokens"], 1)
        self.assertEqual(resources["environment_steps"], 5)

    def test_incomplete_provider_ledger_is_only_a_subtotal(self):
        usage = {"api_calls": 2, "total_tokens": 7, "provider_usage_complete": False}
        resources = aggregate_cases([make_case("a", [True], usage=usage)], 3)["resources"]
        self.assertEqual(resources["api_calls"], 2)
        self.assertIsNone(resources["total_tokens"])
        self.assertEqual(resources["known_subtotals"]["total_tokens"], 7)
        self.assertFalse(resources["provider_usage_complete"])

    def test_components_can_supply_total_but_total_cannot_invent_components(self):
        components = {"api_calls": 1, "prompt_tokens": 2, "completion_tokens": 3,
                      "provider_usage_complete": True}
        total = {"api_calls": 1, "total_tokens": 4, "provider_usage_complete": True}
        resources = aggregate_cases([make_case("a", [True], usage=components),
                                    make_case("b", [True], usage=total)], 1)["resources"]
        self.assertEqual(resources["total_tokens"], 9)
        self.assertIsNone(resources["prompt_tokens"])
        self.assertIsNone(resources["completion_tokens"])
        self.assertTrue(resources["provider_usage_complete"])

    def test_invalid_or_contradictory_resource_counts_stay_unknown(self):
        for usage in ({"api_calls": True, "total_tokens": -1},
                      {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 7,
                       "provider_usage_complete": True}):
            with self.subTest(usage=usage):
                resources = aggregate_cases([make_case("a", [True], usage=usage)], 1)["resources"]
                self.assertIsNone(resources["total_tokens"])
                self.assertFalse(resources["provider_usage_complete"])

    def test_partial_execution_costs_remain_visible(self):
        usage = {"api_calls": 1, "total_tokens": 0, "provider_usage_complete": True}
        result = aggregate_cases([make_case("partial", [False], usage=usage, steps=2)], 3)
        self.assertEqual(result["counts"]["completed"], 0)
        self.assertIsNone(result["metrics"]["sr_at_t"])
        self.assertEqual(result["resources"]["total_tokens"], 0)
        self.assertEqual(result["resources"]["environment_steps"], 2)

    def test_trial_usage_fallback_and_nested_resource_shapes(self):
        case = make_case("a", [False, True], steps=1)
        for row in case["trials"]:
            row["llm_usage"] = {"api_calls": 1, "total_tokens": 4,
                                "provider_usage_complete": True}
        resources = aggregate_cases([case], 3)["resources"]
        self.assertEqual(resources["total_tokens"], 8)
        nested = make_case("b", [True], usage={"llm": {"api_calls": 1, "total_tokens": 3,
                                                       "provider_usage_complete": True},
                                               "environment": {"environment_transitions": 7}})
        self.assertEqual(aggregate_cases([nested], 3)["resources"]["environment_steps"], 7)

    def test_environment_only_usage_falls_back_to_trial_model_usage(self):
        for usage in ({"environment_steps": 1},
                      {"environment_steps": 1, "llm": {}},
                      {"environment": {"environment_transitions": 1}, "llm": {}}):
            with self.subTest(usage=usage):
                case = make_case("a", [True], usage=usage)
                case["trials"][0]["llm_usage"] = {
                    "api_calls": 1, "prompt_tokens": 2, "completion_tokens": 3,
                    "total_tokens": 5, "provider_usage_complete": True,
                }
                resources = aggregate_cases([case], 3)["resources"]
                for field, expected in (("api_calls", 1), ("prompt_tokens", 2),
                                        ("completion_tokens", 3), ("total_tokens", 5),
                                        ("environment_steps", 1)):
                    self.assertEqual(resources[field], expected)
                self.assertTrue(resources["provider_usage_complete"])

    def test_explicit_model_ledger_prevents_trial_fallback(self):
        for usage in ({"environment_steps": 1, "provider_usage_complete": False},
                      {"environment_steps": 1, "total_tokens": None},
                      {"environment_steps": 1, "llm": {"total_tokens": -1}},
                      {"environment_steps": 1, "llm": None},
                      {"environment_steps": 1, "llm": {}, "total_tokens": None,
                       "provider_usage_complete": False}):
            with self.subTest(usage=usage):
                case = make_case("a", [True], usage=usage)
                case["trials"][0]["llm_usage"] = {
                    "api_calls": 1, "prompt_tokens": 2, "completion_tokens": 3,
                    "total_tokens": 5, "provider_usage_complete": True,
                }
                resources = aggregate_cases([case], 3)["resources"]
                self.assertIsNone(resources["total_tokens"])
                self.assertIsNot(resources["provider_usage_complete"], True)

    def test_no_input_mutation(self):
        cases = [make_case("a", [True])]
        before = deepcopy(cases)
        aggregate_cases(cases, 3)
        self.assertEqual(cases, before)

class EvaluationResultTests(unittest.TestCase):
    def test_roundtrip_and_copied_properties(self):
        result = make_result([make_case("a", [True])], output_dir="runs/demo",
                             identity={"candidate_sha256": "abc"}, artifacts={"stderr": "stderr.log"})
        with tempfile.TemporaryDirectory() as temporary:
            path = result.write(Path(temporary) / "nested/evaluation.json")
            loaded = EvaluationResult.from_path(path)
            self.assertEqual(loaded.to_dict(), result.to_dict())
            self.assertEqual(loaded.output_dir, Path("runs/demo"))
            self.assertEqual(loaded.status, "complete")
            self.assertEqual(loaded.to_dict()["identity"], {"candidate_sha256": "abc"})
        copied = result.metrics
        copied["sr_at_t"] = -1
        self.assertEqual(result.metrics["sr_at_t"], 1)

    def test_require_complete_refuses_failed_partial_empty_or_missing_counts(self):
        complete = make_result([make_case("a", [True])])
        self.assertIs(complete.require_complete(), complete)
        for payload in (make_result([make_case("a", [False])]).to_dict(),
                        {**complete.to_dict(), "status": "timeout"},
                        {"status": "complete", "counts": {}},
                        make_result([]).to_dict()):
            with self.subTest(payload=payload), self.assertRaises(RuntimeError):
                EvaluationResult(payload).require_complete()

    def test_nonfinite_values_cannot_overwrite_a_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evaluation.json"
            path.write_text("preserve")
            with self.assertRaises(ValueError):
                EvaluationResult({"metrics": {"sr_at_t": float("nan")}}).write(path)
            self.assertEqual(path.read_text(), "preserve")


class CompareResultsTests(unittest.TestCase):
    def webshop_result(self, digest=None, method="retry"):
        identity = {} if digest is None else {"environment": {
            "benchmark": "webshop", "source_data_index_root_sha256": digest,
        }}
        payload = make_result([make_case("a", [True])], method=method, identity=identity).to_dict()
        payload["config"]["benchmark"] = {"name": "webshop", "params": {}}
        return payload

    def test_native_webshop_identity_is_shared_across_memory_methods(self):
        compared = compare_results([self.webshop_result("a" * 64, "reflexion"),
                                    self.webshop_result("a" * 64, "ace")])
        self.assertEqual(compared["environment_identity_check"], {
            "scope": "recorded_native_webshop_source_data_index", "status": "verified",
            "source_data_index_root_sha256": "a" * 64, "missing_run_indices": [],
        })

    def test_native_webshop_index_mismatch_rejects_even_after_unknown_reference(self):
        runs = [self.webshop_result(char * 64) for char in ("a", "b")]
        for values in (runs, [self.webshop_result(), *runs]):
            with self.subTest(legacy=len(values) == 3):
                with self.assertRaisesRegex(ValueError, "different WebShop source/data/index"):
                    compare_results(values)

    def test_native_webshop_legacy_identity_is_explicitly_unverified(self):
        compared = compare_results([self.webshop_result(), self.webshop_result("a" * 64)])
        checked = compared["environment_identity_check"]
        self.assertEqual(checked["status"], "unverified")
        self.assertEqual(checked["missing_run_indices"], [0])
        for invalid in ("not-a-hash", "g" * 64):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "SHA-256"):
                compare_results([self.webshop_result(invalid)])

    def test_shared_plugin_sources_are_checked_but_memory_sources_can_differ(self):
        def run(method, memory_hash):
            return make_result([make_case("a", [True])], method=method, identity={
                "sdk_version": method,
                "plugins": [
                    {"role": "benchmark", "source_sha256": "a" * 64, "source_path": method},
                    {"role": "model_transport", "source_sha256": "b" * 64},
                    {"role": "memory", "source_sha256": memory_hash},
                    {"role": "memory_method", "source_sha256": memory_hash},
                ],
            })
        reference, target = run("reflexion", "c" * 64), run("candidate", "d" * 64)
        before = [reference.to_dict(), target.to_dict()]
        checked = compare_results([reference, target])["component_identity_check"]
        self.assertEqual(checked["scope"], "recorded_non_memory_plugin_sources")
        self.assertEqual(checked["status"], "verified")
        self.assertEqual(set(checked["components"]), {"benchmark", "model_transport"})
        for role, digest in (("benchmark", "a" * 64), ("model_transport", "b" * 64)):
            self.assertEqual(checked["components"][role], {
                "status": "verified", "source_sha256": digest, "missing_run_indices": [],
            })
        self.assertEqual([reference.to_dict(), target.to_dict()], before)

    def test_different_shared_plugin_sources_are_rejected_even_after_legacy_reference(self):
        legacy = make_result([make_case("a", [True])])
        for role in ("benchmark", "model_transport"):
            runs = [make_result([make_case("a", [True])], identity={"plugins": [
                {"role": role, "source_sha256": digest * 64}]}) for digest in ("a", "b")]
            for compared in (runs, [legacy, *runs]):
                with self.subTest(role=role, legacy=len(compared) == 3):
                    with self.assertRaisesRegex(ValueError, f"different {role} plugin source"):
                        compare_results(compared)

    def test_missing_shared_plugin_identity_remains_compatible_and_explicit(self):
        legacy = make_result([make_case("a", [True])]).to_dict()
        current = make_result([make_case("a", [True])], method="candidate", identity={
            "plugins": [{"role": "benchmark", "source_sha256": "a" * 64}],
        }).to_dict()
        factory = {"entrypoint": "same_task:create", "kwargs": {}}
        for payload in (legacy, current):
            payload["config"]["benchmark"]["factory"] = factory
            payload["config"]["model"]["transport"] = factory
        for identity in ({}, None, {"plugins": None}):
            legacy["identity"] = identity
            with self.subTest(identity=identity):
                compared = compare_results([legacy, current])
                checked = compared["component_identity_check"]
                self.assertEqual(checked["status"], "unverified")
                self.assertEqual(checked["components"]["benchmark"], {
                    "status": "unverified", "source_sha256": "a" * 64, "missing_run_indices": [0],
                })
                self.assertEqual(checked["components"]["model_transport"], {
                    "status": "unverified", "source_sha256": None, "missing_run_indices": [0, 1],
                })
                self.assertEqual(compared["paired_effects"][0]["sr_difference_pp"], 0)

    def test_no_shared_plugins_is_not_a_claim_of_verified_builtin_sources(self):
        reference = make_result([make_case("a", [True])])
        target = make_result([make_case("a", [True])], method="candidate", identity={
            "plugins": [{"role": "memory", "source_sha256": "a" * 64}],
        })
        checked = compare_results([reference, target])["component_identity_check"]
        self.assertEqual(checked["status"], "not_applicable")
        self.assertTrue(all(value == {"status": "not_applicable", "source_sha256": None,
                                      "missing_run_indices": []}
                            for value in checked["components"].values()))

    def test_invalid_or_ambiguous_shared_source_identity_is_rejected(self):
        reference = make_result([make_case("a", [True])])
        variants = [
            {"plugins": [{"role": "benchmark", "source_sha256": "not-a-hash"}]},
            {"plugins": [{"role": "benchmark", "source_sha256": "a" * 64},
                         {"role": "benchmark", "source_sha256": "a" * 64}]},
        ]
        for identity in variants:
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                compare_results([reference, make_result([make_case("a", [True])], identity=identity)])

    def test_descriptive_pairs_allow_different_cost_and_case_order(self):
        baseline = make_result([make_case("a", [False] * 3), make_case("b", [True])])
        target = make_result([make_case("b", [False, True]), make_case("a", [False, True])],
                             method="candidate")
        compared = compare_results([baseline, target.to_dict()])
        self.assertFalse(compared["equal_compute_assumed"])
        effect = compared["paired_effects"][0]
        self.assertEqual(effect["target"], "candidate")
        self.assertEqual(effect["sr_difference_pp"], 50)
        self.assertEqual(effect["avg_t_difference"], -.5)
        self.assertEqual(effect["positive_pairs"], 1)
        self.assertEqual(effect["negative_pairs"], 0)
        self.assertEqual(effect["tied_pairs"], 1)
        deltas = {row["case_id"]: row for row in effect["case_deltas"]}
        self.assertEqual(deltas["a"], {
            "case_id": "a", "task": "task",
            "reference_success": False, "target_success": True,
            "reference_trials_executed": 3, "target_trials_executed": 2,
            "success_difference": 1, "censored_trials_difference": -2,
        })
        self.assertEqual(deltas["b"]["success_difference"], 0)
        self.assertEqual(deltas["b"]["censored_trials_difference"], 1)

    def test_incompatible_inputs_are_rejected(self):
        baseline = make_result([make_case("a", [True])])
        variants = []
        for field, value in (("model", {"name": "other"}), ("trials", 4),
                             ("benchmark", {"name": "other"}), ("max_environment_steps", 11)):
            payload = baseline.to_dict()
            payload["config"][field] = value
            if field == "trials":
                payload["trials"] = value
            variants.append(payload)
        payload = baseline.to_dict(); payload["cases"][0]["case_id"] = "other"
        variants.append(payload)
        payload = baseline.to_dict(); payload["cases"][0]["task"] = "other"
        variants.append(payload)
        payload = baseline.to_dict(); payload["status"] = "failed"
        variants.append(payload)
        payload = baseline.to_dict(); payload["counts"]["solved"] = 0
        variants.append(payload)
        payload = baseline.to_dict(); payload["config"]["trials"] = 4
        variants.append(payload)
        for payload in variants:
            with self.subTest(payload=payload), self.assertRaises((ValueError, RuntimeError)):
                compare_results([baseline, payload])
        with self.assertRaises(ValueError):
            compare_results([])


if __name__ == "__main__":
    unittest.main()
