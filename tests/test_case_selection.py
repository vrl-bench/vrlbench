"""Order-independent built-in case selections, with other settings kept strict."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vrlbench import FactorySpec, MemorySpec, ModelSpec, SimulationSpec, WorkloadSpec
from vrlbench import aggregate_cases, compare_results, plan, simulate
from vrlbench.results import SimulationResult


def recorded_fixture(spec):
    """A complete local record fixture for preflight, not benchmark evidence."""
    cases = [
        {"case_id": str(index), "task": "fixture", "resource_usage": {},
         "trials": [{"trial": 0, "success": True, "trajectory": "fixture"}]}
        for index, _ in enumerate(spec.workload.cases)
    ]
    return SimulationResult({"status": "complete", "config": spec.to_dict(),
                             **aggregate_cases(cases, spec.trials), "cases": cases})


class CaseSelectionCompatibility(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("openai"), "install [runtime] for the offline trial loop")
    def test_actual_builtin_simulations_compare_after_case_reordering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = SimulationSpec(
                WorkloadSpec("demo", ["a", "b"]), ModelSpec("scripted-local"),
                MemorySpec("local_trace", factory=FactorySpec("vrlbench.backends:create_demo_memory")),
                trials=3, output_dir=str(root / "first"), require_provider_usage=False,
            )
            first = simulate(spec).require_complete()
            second = simulate(replace(spec, workload=replace(spec.workload, cases=["b", "a"]),
                                      output_dir=str(root / "second"))).require_complete()
            before = second.to_dict()
            compared = compare_results([first, second])
            self.assertEqual(compared["paired_effects"][0]["sr_difference_pp"], 0)
            self.assertEqual(compared["paired_effects"][0]["tied_pairs"], 2)
            self.assertEqual(second.to_dict(), before)

    def test_native_shared_first_trial_plan_accepts_reordered_selection(self):
        selections = {
            "webshop": [0, 1],
            "miniwob": [{"task": "click-test", "episode": 0},
                        {"task": "click-test", "episode": 1}],
        }
        for name, cases in selections.items():
            with self.subTest(workload=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = SimulationSpec(WorkloadSpec(name, cases), ModelSpec("fixture"),
                                        require_provider_usage=False)
                recorded_fixture(source).write(root / "evaluation.json")
                (root / "request.json").write_text(json.dumps(source.to_dict()))
                candidate = replace(source, workload=replace(source.workload, cases=list(reversed(cases))),
                                    memory=MemorySpec("reflexion"), shared_t1_dir=str(root))
                before = candidate.to_dict()
                self.assertEqual(plan(candidate)["model_calls_executed"], 0)
                from vrlbench.backends import _validate_sdk_shared_source
                _validate_sdk_shared_source(candidate.to_dict())
                self.assertEqual(candidate.to_dict(), before)
                self.assertEqual(compare_results([recorded_fixture(source), recorded_fixture(candidate)])
                                 ["paired_effects"][0]["sr_difference_pp"], 0)

    def test_custom_adapter_order_is_not_assumed_independent(self):
        for name in ("custom", "webshop"):
            with self.subTest(workload=name), tempfile.TemporaryDirectory() as directory:
                source = SimulationSpec(
                    WorkloadSpec(name, [0, 1], factory=FactorySpec("custom_backend:create")),
                    ModelSpec("fixture"), require_provider_usage=False,
                )
                target = replace(source, workload=replace(source.workload, cases=[1, 0]))
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    compare_results([recorded_fixture(source), recorded_fixture(target)])
                root = Path(directory)
                recorded_fixture(source).write(root / "evaluation.json")
                with self.assertRaisesRegex(ValueError, "different benchmark"):
                    plan(replace(target, shared_t1_dir=str(root)))

    def test_other_model_environment_and_credential_settings_remain_strict(self):
        source = SimulationSpec(WorkloadSpec("demo", ["a", "b"]), ModelSpec("fixture"),
                                require_provider_usage=False)
        variants = [
            replace(source, model=replace(source.model, name="other-model")),
            replace(source, model=replace(source.model, api_key_env="OTHER_API_KEY")),
            replace(source, runtime_env={"TASK_MODE": "other"}),
            replace(source, workload=replace(source.workload, params={"success_after": 2})),
            replace(source, max_environment_steps=16),
        ]
        for target in variants:
            with self.subTest(config=target.to_dict()), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    compare_results([recorded_fixture(source), recorded_fixture(target)])
                root = Path(directory)
                recorded_fixture(source).write(root / "evaluation.json")
                with self.assertRaisesRegex(ValueError, "different"):
                    plan(replace(target, shared_t1_dir=str(root)))


if __name__ == "__main__":
    unittest.main()
