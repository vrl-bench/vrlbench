"""Offline task feedback for complete candidate memory implementations."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vrlbench import CandidateFeedback, SimulationResult, SimulationSpec, evaluate_candidate


IGNORE_FEEDBACK = '''
from vrlbench import ActorContext, MemoryMethod
class Candidate(MemoryMethod):
    def reset_case(self, config): pass
    def context_for_trial(self, trial_index): return ActorContext()
    def observe_trial(self, record): pass
def create_method(): return Candidate()
'''

ACCUMULATE_FEEDBACK = '''
import json
from vrlbench import ActorContext, MemoryMethod
class Candidate(MemoryMethod):
    def reset_case(self, config): self.rejected = []
    def context_for_trial(self, trial_index):
        return ActorContext(system=("Rejected controls JSON: " + json.dumps(self.rejected))
                            if self.rejected else "")
    def observe_trial(self, record):
        if record.success or not record.trajectory: return
        feedback = json.loads(record.trajectory)["observation"]["feedback"]
        if feedback.startswith("Rejected control: "):
            self.rejected.append(feedback[len("Rejected control: "):-1])
def create_method(): return Candidate()
'''

MEMORY_CODE = '''
import json
from vrlbench.memory import MemorySystem
class Candidate(MemorySystem):
    def __init__(self, label): self.label = label; self.reset()
    def reset(self): self.rejected = []
    def on_failure(self, trajectory, model):
        feedback = json.loads(trajectory)["observation"]["feedback"]
        self.rejected.append(feedback[len("Rejected control: "):-1])
    def build_section(self):
        return "Rejected controls JSON: " + json.dumps(self.rejected) if self.rejected else ""
def make_memory(label): return Candidate(label)
'''


@unittest.skipUnless(importlib.util.find_spec("openai"), "Install the runtime extra")
class CandidateEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.original = SimulationSpec.from_path(ROOT / "examples/configs/local_retry.json")
        self.counter = 0

    def spec(self, **changes):
        self.counter += 1
        return replace(self.original, output_dir=str(self.directory / f"run-{self.counter}"), **changes)

    def source(self, code, name="candidate.py"):
        path = self.directory / name
        path.write_text(code, encoding="utf-8")
        return path

    def test_complete_code_change_produces_real_actions_and_paired_feedback(self):
        path = self.source(IGNORE_FEEDBACK)
        original_spec = self.spec()
        initial = evaluate_candidate(path, original_spec)
        self.assertIsInstance(initial, CandidateFeedback)
        self.assertTrue(initial.scorable)
        self.assertAlmostEqual(initial.metrics["sr_at_t"], 1 / 3)
        self.assertIsNone(initial.comparison)
        self.assertIsNone(initial.regressed_case_ids)
        self.assertIsNone(initial.reference)
        self.assertEqual(original_spec.memory.name, "retry")
        path.write_text(ACCUMULATE_FEEDBACK, encoding="utf-8")
        improved = evaluate_candidate(path, self.spec(), reference=initial.evaluation_path)
        self.assertTrue(improved.scorable)
        self.assertEqual(improved.metrics["sr_at_t"], 1)
        self.assertEqual(improved.metrics["rr_at_t"], 1)
        self.assertEqual(improved.metrics["avg_t_at_t"], 2)
        self.assertEqual(improved.regressed_case_ids, [])
        self.assertAlmostEqual(improved.comparison["paired_effects"][0]["sr_difference_pp"], 200 / 3)
        self.assertEqual(improved.reference["evaluation_path"], initial.evaluation_path)
        self.assertEqual(improved.reference["memory_sources"][0]["source_sha256"],
                         initial.candidate_sha256)
        parent_payload = json.loads(Path(initial.evaluation_path).read_text())
        canonical_parent = json.dumps(parent_payload, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.assertEqual(improved.reference["result_sha256"], hashlib.sha256(canonical_parent).hexdigest())
        deltas = {row["case_id"]: row for row in improved.comparison["paired_effects"][0]["case_deltas"]}
        self.assertEqual(deltas["case_b"]["success_difference"], 1)
        self.assertEqual(deltas["case_b"]["censored_trials_difference"], -1)
        case = next(case for case in improved.cases if case["case_id"] == "case_b")
        self.assertEqual([json.loads(row["trajectory"])["action"] for row in case["trials"]],
                         ["amber", "blue", "green"])
        self.assertTrue(improved.candidate_identity_verified)
        self.assertNotEqual(initial.candidate_sha256, improved.candidate_sha256)
        self.assertEqual(improved.candidate_sha256, hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(Path(improved.candidate_source_path).read_bytes(), path.read_bytes())
        self.assertEqual(improved.resources["api_calls"], 0)
        saved = json.loads((Path(improved.output_dir) / "feedback.json").read_text())
        self.assertEqual(saved, improved.to_dict())
        evaluation = json.loads(Path(improved.evaluation_path).read_text())
        for key, value in self.original.to_dict().items():
            if key not in {"memory", "output_dir"}:
                self.assertEqual(evaluation["config"][key], value)
        path.write_text(IGNORE_FEEDBACK, encoding="utf-8")
        regressed = evaluate_candidate(path, self.spec(), reference=improved.output_dir)
        self.assertEqual(regressed.regressed_case_ids, ["case_a", "case_b"])
        self.assertEqual(regressed.reference["evaluation_path"], improved.evaluation_path)
        self.assertEqual(regressed.reference["memory_sources"][0]["source_sha256"],
                         improved.candidate_sha256)

    def test_memory_interface_and_factory_arguments(self):
        path = self.source(MEMORY_CODE)
        result = evaluate_candidate(path, self.spec(), interface="memory", factory="make_memory",
                                    factory_kwargs={"label": "complete-memory-code"})
        self.assertTrue(result.scorable)
        self.assertTrue(result.candidate_identity_verified)
        self.assertEqual(result.metrics["sr_at_t"], 1)
        legacy = SimulationResult.from_path(result.evaluation_path).to_dict()
        legacy["identity"] = None
        compared = evaluate_candidate(path, self.spec(), interface="memory", factory="make_memory",
                                      factory_kwargs={"label": "complete-memory-code"},
                                      reference=SimulationResult(legacy))
        self.assertTrue(compared.scorable)
        self.assertEqual(compared.reference["memory_sources"], [])

    def test_bad_candidate_and_timeout_are_not_zero_scores(self):
        broken = self.source("raise RuntimeError('candidate import failed')\n")
        failed = evaluate_candidate(broken, self.spec())
        self.assertEqual(failed.status, "failed")
        self.assertFalse(failed.scorable)
        self.assertTrue(all(value is None for value in failed.metrics.values()))
        self.assertEqual(failed.cases, [])
        self.assertIn("candidate import failed", failed.error["message"])
        blocked = self.source("import time\ntime.sleep(30)\n", name="blocked.py")
        timed_out = evaluate_candidate(blocked, self.spec(timeout_seconds=0.2))
        self.assertEqual(timed_out.status, "timeout")
        self.assertFalse(timed_out.scorable)
        self.assertTrue(all(value is None for value in timed_out.metrics.values()))

    def test_incompatible_reference_is_rejected_before_candidate_execution(self):
        baseline = evaluate_candidate(self.source(IGNORE_FEEDBACK), self.spec())
        rejected_spec = self.spec(trials=4)
        with patch("vrlbench.development.simulate") as execute:
            with self.assertRaisesRegex(ValueError, "incompatible"):
                evaluate_candidate(self.source("raise AssertionError('must not import')\n"),
                                   rejected_spec, reference=baseline.evaluation_path)
            execute.assert_not_called()
        self.assertFalse(Path(rejected_spec.output_dir).exists())

    def test_candidate_import_runs_in_worker_not_parent(self):
        code = ("import os\n"
                f"if os.getpid() == {os.getpid()}: raise AssertionError('parent imported candidate')\n"
                + IGNORE_FEEDBACK)
        result = evaluate_candidate(self.source(code), self.spec())
        self.assertTrue(result.scorable)

    def test_source_hash_mismatch_cannot_be_scored(self):
        path = self.source(IGNORE_FEEDBACK)
        from vrlbench.simulation import simulate

        def changed_before_worker(spec):
            path.write_text(ACCUMULATE_FEEDBACK, encoding="utf-8")
            return simulate(spec)

        with patch("vrlbench.development.simulate", side_effect=changed_before_worker):
            result = evaluate_candidate(path, self.spec())
        self.assertEqual(result.status, "complete")
        self.assertFalse(result.scorable)
        self.assertFalse(result.candidate_identity_verified)
        self.assertEqual(result.error["type"], "CandidateIdentityMismatch")
        self.assertEqual(Path(result.candidate_source_path).read_text(), IGNORE_FEEDBACK)
        self.assertEqual(hashlib.sha256(Path(result.candidate_source_path).read_bytes()).hexdigest(),
                         result.candidate_sha256)
        self.assertTrue(all(value is None for value in result.metrics.values()))

    def test_final_paired_identity_check_can_reject_an_executed_candidate(self):
        source = self.source(IGNORE_FEEDBACK)
        initial = evaluate_candidate(source, self.spec())
        payload = SimulationResult.from_path(initial.evaluation_path).to_dict()
        next(plugin for plugin in payload["identity"]["plugins"]
             if plugin["role"] == "benchmark")["source_sha256"] = "0" * 64
        result = evaluate_candidate(source, self.spec(), reference=SimulationResult(payload))
        self.assertEqual(result.status, "complete")
        self.assertFalse(result.scorable)
        self.assertIsNone(result.comparison)
        self.assertIsNone(result.regressed_case_ids)
        self.assertIn("plugin source identities", result.error["message"])
        self.assertIsNone(result.reference["evaluation_path"])
        self.assertTrue(all(value is None for value in result.metrics.values()))

    def test_nonempty_output_is_never_overwritten(self):
        spec = self.spec()
        source = self.source(IGNORE_FEEDBACK)
        evaluate_candidate(source, spec)
        with self.assertRaises(FileExistsError):
            evaluate_candidate(source, spec)


if __name__ == "__main__":
    unittest.main()
