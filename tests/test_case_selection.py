"""Order-independent built-in case selections, with other settings kept strict."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vrlbench import FactorySpec, MemorySpec, ModelSpec, SimulationSpec, WorkloadSpec
from vrlbench import aggregate_cases, compare_results, plan
from vrlbench.results import SimulationResult


def recorded_fixture(spec):
    """A complete local record fixture for preflight, not benchmark evidence."""
    from vrlbench.worker import expected_case_ids
    cases = [{"case_id": identity, "task": "fixture", "resource_usage": {},
              "trials": [{"trial": 0, "success": True, "trajectory": "fixture",
                          "trial_metadata": {"task": "fixture", "case_identity": identity,
                              "max_environment_steps": spec.max_environment_steps,
                              "actor_llm_usage": {"api_calls": 0, "total_tokens": 0,
                                                  "provider_usage_complete": True}}}]}
             for identity in sorted(expected_case_ids(spec.to_dict()["benchmark"]))]
    return SimulationResult({"status": "complete", "config": spec.to_dict(),
                             **aggregate_cases(cases, spec.trials), "cases": cases})


class CaseSelectionCompatibility(unittest.TestCase):
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
                                        trials=6, max_environment_steps=15, require_provider_usage=False)
                recorded_fixture(source).write(root / "evaluation.json")
                (root / "request.json").write_text(json.dumps(source.to_dict()))
                candidate = replace(source, workload=replace(source.workload, cases=list(reversed(cases))),
                                    memory=MemorySpec("reflexion"), shared_t1_dir=str(root))
                before = candidate.to_dict()
                self.assertEqual(plan(candidate)["model_calls_executed"], 0)
                from vrlbench._shared_t1 import load_shared_first_trials
                load_shared_first_trials(candidate.to_dict())
                self.assertEqual(candidate.to_dict(), before)
                self.assertEqual(compare_results([recorded_fixture(source), recorded_fixture(candidate)])
                                 ["paired_effects"][0]["sr_difference_pp"], 0)

    def test_custom_adapter_order_is_not_assumed_independent(self):
        for name in ("custom", "webshop"):
            with self.subTest(workload=name), tempfile.TemporaryDirectory() as directory:
                source = SimulationSpec(
                    WorkloadSpec(name, [0, 1], factory=FactorySpec("custom_backend:create")),
                    ModelSpec("fixture"), trials=6, max_environment_steps=15, require_provider_usage=False,
                )
                target = replace(source, workload=replace(source.workload, cases=[1, 0]))
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    compare_results([recorded_fixture(source), recorded_fixture(target)])
                root = Path(directory)
                recorded_fixture(source).write(root / "evaluation.json")
                with self.assertRaisesRegex(ValueError, "different benchmark"):
                    plan(replace(target, shared_t1_dir=str(root)))

    def test_other_model_environment_and_credential_settings_remain_strict(self):
        source = SimulationSpec(WorkloadSpec("webshop", [0, 1]), ModelSpec("fixture"),
                                trials=6, max_environment_steps=15, require_provider_usage=False)
        variants = [
            replace(source, model=replace(source.model, name="other-model")),
            replace(source, model=replace(source.model, api_key_env="OTHER_API_KEY")),
            replace(source, runtime_env={"TASK_MODE": "other"}),
            replace(source, workload=replace(source.workload, params={"capture_observations": True})),
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
