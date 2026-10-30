"""Offline exposure checks; no network clients or benchmark downloads."""

from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vrlbench import FactorySpec, MemorySpec, ModelSpec, ReplayGateSpec, SimulationSpec, WorkloadSpec
from vrlbench.replay import ReplayContext, epsilon_draw, wrap_run_episode

HAS_RUNTIME = importlib.util.find_spec("openai") is not None


class ReplayConfiguration(unittest.TestCase):
    def test_default_wire_format_stays_unchanged(self):
        spec = SimulationSpec(WorkloadSpec("demo", ["a"]), ModelSpec("scripted"),
                              trials=3, max_environment_steps=15, require_provider_usage=False)
        self.assertNotIn("replay_gate", spec.to_dict())
        spec = replace(spec, replay_gate=ReplayGateSpec(epsilon=.5, seed=42))
        self.assertEqual(SimulationSpec.from_dict(spec.to_dict()), spec)
        spec.validate()

    def test_policy_validation_and_factory_resolution(self):
        for gate in [ReplayGateSpec(), ReplayGateSpec(epsilon=True), ReplayGateSpec(epsilon=-.1),
                     ReplayGateSpec(epsilon=float("nan")), ReplayGateSpec(epsilon=1.1),
                     ReplayGateSpec(epsilon=.5, seed=True), ReplayGateSpec(policy="unknown"),
                     ReplayGateSpec(policy="custom")]:
            with self.subTest(gate=gate), self.assertRaises(ValueError):
                gate.validate()
        spec = SimulationSpec.from_dict({
            "workload": {"name": "demo", "cases": ["a"]}, "model": {"name": "local"},
            "trials": 3, "max_environment_steps": 15,
            "replay_gate": {"policy": "custom", "factory": {"entrypoint": "policy.py:create"}},
        }).resolved(ROOT)
        self.assertEqual(spec.replay_gate.factory.entrypoint, str(ROOT / "policy.py") + ":create")

    def test_shared_draws_do_not_depend_on_order_or_epsilon(self):
        values = {(case, trial): epsilon_draw(42, case, trial)
                  for case in ["a", "b"] for trial in [1, 2, 3]}
        reordered = {(case, trial): epsilon_draw(42, case, trial)
                     for case in ["b", "a"] for trial in [3, 1, 2]}
        self.assertEqual(values, reordered)
        self.assertTrue(all(0 <= draw < 1 for draw in values.values()))
        self.assertNotEqual(values[("a", 1)], epsilon_draw(43, "a", 1))


@unittest.skipUnless(HAS_RUNTIME, "requires the runtime extra")
class ReplayLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vrlbench.plugins import enable_reference_imports
        enable_reference_imports()

    def run_loop(self, epsilon, *, placement="system", controller=False, initial=None, fail=False):
        from vrlbench.runtime import llm, loop
        from vrlbench.runtime.er.base import ERSlot
        from vrlbench.runtime.controllers.base import ControllerContext

        class Memory(ERSlot):
            def reset(self): self.history = []
            def on_failure(self, trajectory, model): self.history.append(trajectory)
            def build_section(self): return "\n".join(self.history)
            def read_placement(self): return placement

        class Controller:
            method_id = "test"
            def reset(self): self.history = []
            def set_total_budget(self, total_budget): pass
            def build_trial_context(self, **kwargs):
                return ControllerContext(verbatim_system_context="  raw policy\nα\n" if kwargs["trial_index"] else "")
            def on_failure(self, trajectory, model, **kwargs): self.history.append(trajectory)
            def on_success(self, trajectory, **kwargs): pass
            def snapshot(self): return {"history": self.history[:]}
            def continuation_failure(self): return None

        seen = []
        class Actor:
            def set_trial_context(self, *, trial_index, total_trials): self.index = trial_index
            def __call__(self):
                seen.append((llm.get_context(), llm.get_user_context()))
                if fail: raise ConnectionError("synthetic interruption")
                return f"failed trial {self.index}"

        memory = Memory()
        selected = Controller() if controller else None
        kwargs = dict(actor=Actor(), manager=lambda _: False, er_slot=memory, model="local", total_trials=3,
                      metadata={"case_id": "case-a"}, controller=selected)
        if controller:
            class Empty(Memory):
                def build_section(self): return ""
            memory = kwargs["er_slot"] = Empty()
        if initial is not None:
            kwargs["initial_result"] = initial
        gated = wrap_run_episode(loop.run_episode, gate=asdict(ReplayGateSpec(epsilon=epsilon, seed=42)),
                                 benchmark="custom")
        result = gated(**kwargs)
        return result, seen, memory, selected

    def test_expose_or_withhold_both_placements_without_stopping_writes(self):
        for placement in ["system", "user_prefix"]:
            with self.subTest(placement=placement):
                hidden, seen, memory, _ = self.run_loop(1, placement=placement)
                self.assertEqual(seen, [("", "")] * 3)
                self.assertEqual(len(memory.history), 2)
                for trial in hidden.trials[1:]:
                    self.assertEqual((trial.context, trial.system_context, trial.user_context), ("", "", ""))
                    gate = trial.trial_metadata["replay_gate"]
                    self.assertFalse(gate["exposed"])
                    self.assertIn("failed trial", gate["prepared_system_context"] + gate["prepared_user_context"])
                exposed, seen, memory, _ = self.run_loop(0, placement=placement)
                self.assertEqual(len(memory.history), 2)
                self.assertEqual(seen[0], ("", ""))
                for trial, delivered in zip(exposed.trials, seen):
                    self.assertEqual((trial.system_context, trial.user_context), delivered)

    def test_whole_controller_response_preserves_raw_text_when_exposed(self):
        exposed, seen, _, controller = self.run_loop(0, controller=True)
        self.assertEqual(seen[1][0], "  raw policy\nα\n")
        self.assertEqual(len(controller.history), 2)
        hidden, seen, _, controller = self.run_loop(1, controller=True)
        self.assertEqual(seen[1], ("", ""))
        self.assertEqual(hidden.trials[1].trial_metadata["replay_gate"]["prepared_system_context"],
                         "  raw policy\nα\n")
        self.assertEqual(len(controller.history), 2)

    def test_shared_first_trial_is_not_rewritten(self):
        from vrlbench.runtime.loop import EpisodeResult, TrialResult
        shared = TrialResult(0, False, "original T1", trial_metadata={"source": "unchanged"})
        prefix = EpisodeResult([shared])
        before = asdict(prefix)
        result, _, _, _ = self.run_loop(1, initial=prefix)
        self.assertEqual(asdict(prefix), before)
        self.assertIs(result.trials[0], shared)
        self.assertNotIn("replay_gate", shared.trial_metadata)

    def test_custom_policy_bool_contract_and_context_restoration(self):
        from vrlbench.runtime import llm
        calls = []
        def factory():
            def policy(context):
                calls.append(context)
                return False
            return policy
        class Actor:
            def __call__(self):
                self.seen = llm.get_context(), llm.get_user_context()
                raise ConnectionError("synthetic")
        actor = Actor()
        def original(actor, **kwargs):
            actor.set_trial_context(trial_index=1, total_trials=2)
            llm.set_context("prepared-system")
            llm.set_user_context("prepared-user")
            return actor()
        gate = {"policy": "custom", "seed": 7}
        wrapped = wrap_run_episode(original, gate=gate, benchmark="custom", policy_factory=factory)
        try:
            with self.assertRaises(ConnectionError):
                wrapped(actor=actor, metadata={"case_id": "a"}, total_trials=2)
            self.assertEqual(actor.seen, ("", ""))
            self.assertEqual((llm.get_context(), llm.get_user_context()), ("prepared-system", "prepared-user"))
            self.assertEqual(calls[0].seed, 7)
            invalid = wrap_run_episode(original, gate=gate, benchmark="custom",
                                       policy_factory=lambda: lambda context: "false")
            with self.assertRaisesRegex(TypeError, "must return bool"):
                invalid(actor=actor, metadata={"case_id": "a"}, total_trials=2)
            with self.assertRaisesRegex(ValueError, "stable case identity"):
                wrapped(actor=actor, metadata={}, total_trials=2)
        finally:
            llm.set_context(""); llm.set_user_context("")

    def test_custom_policy_is_loaded_by_the_public_worker(self):
        from vrlbench import plan, simulate
        source = SimulationSpec.from_path(ROOT / "examples/configs/local_feedback.json")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plugin = root / "policy.py"
            plugin.write_text("def create():\n    return lambda context: False\n")
            spec = replace(source, output_dir=str(root / "run"), replay_gate=ReplayGateSpec(
                policy="custom", factory=FactorySpec(f"{plugin}:create")))
            self.assertEqual(plan(spec)["model_calls_executed"], 0)
            result = simulate(spec).require_complete()
            self.assertEqual(result.metrics["rr_at_t"], 0)
            identities = result.to_dict()["identity"]["plugins"]
            self.assertEqual(len([row for row in identities if row["role"] == "replay_policy"]), 1)
            self.assertTrue(all(trial["context"] == "" for case in result.cases for trial in case["trials"]))

    def test_native_webshop_persists_delivered_and_prepared_whole_method_context(self):
        from test_method_extension import PLUGIN, RAW_CONTEXT
        from vrlbench import simulate
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plugin = root / "candidate.py"
            plugin.write_text(PLUGIN)
            source = SimulationSpec(
                WorkloadSpec("webshop", [0], factory=FactorySpec(f"{plugin}:create_webshop_backend")),
                ModelSpec("scripted", transport=FactorySpec(f"{plugin}:create_client")),
                trials=3, max_environment_steps=15, output_dir=str(root / "retry"), require_provider_usage=False,
            )
            baseline = simulate(source).require_complete()
            source_bytes = (root / "retry/evaluation.json").read_bytes()
            candidate = replace(source, output_dir=str(root / "candidate"),
                                shared_t1_dir=str(root / "retry"),
                                memory=MemorySpec("joint_method", method_factory=FactorySpec(
                                    f"{plugin}:create_method", {"log": str(root / "events.jsonl")})),
                                replay_gate=ReplayGateSpec(epsilon=1))
            result = simulate(candidate).require_complete()
            first, second = result.cases[0]["trials"][:2]
            self.assertNotIn("replay_gate", first["trial_metadata"])
            self.assertEqual(first["trajectory"], baseline.cases[0]["trials"][0]["trajectory"])
            self.assertEqual((second["context"], second["system_context"], second["user_context"]), ("", "", ""))
            self.assertEqual(second["trial_metadata"]["replay_gate"]["prepared_system_context"], RAW_CONTEXT)
            persisted = json.loads((root / "candidate/reference/episodes.json").read_text())[0]["trials"][1]
            self.assertEqual(persisted["trial_metadata"]["replay_gate"], second["trial_metadata"]["replay_gate"])
            self.assertEqual(persisted["system_context"], "")
            self.assertEqual((root / "retry/evaluation.json").read_bytes(), source_bytes)

    def test_sdk_binding_restores_loop_hooks_after_exception(self):
        from vrlbench.backends import memory_bindings
        from vrlbench.runtime import framework, loop
        source = SimulationSpec.from_path(ROOT / "examples/configs/local_feedback.json")
        spec = replace(source, replay_gate=ReplayGateSpec(epsilon=.5)).to_dict()
        before = (loop.run_episode, framework.run_episode)
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with memory_bindings(spec):
                self.assertIsNot(loop.run_episode, before[0])
                raise RuntimeError("synthetic")
        self.assertEqual((loop.run_episode, framework.run_episode), before)

    def test_custom_policy_factory_cannot_reuse_case_state(self):
        from vrlbench.runtime.loop import EpisodeResult
        def original(actor, **kwargs): return EpisodeResult()
        singleton = lambda context: True
        wrapped = wrap_run_episode(original, gate={"policy": "custom", "seed": 0},
                                   benchmark="custom", policy_factory=lambda: singleton)
        with self.assertRaisesRegex(TypeError, "total_trials"):
            wrapped(actor=lambda: "", metadata={"case_id": "a"})
        wrapped(actor=lambda: "", total_trials=2, metadata={"case_id": "a"})
        with self.assertRaisesRegex(ValueError, "fresh instance"):
            wrapped(actor=lambda: "", total_trials=2, metadata={"case_id": "b"})
        fresh = wrap_run_episode(original, gate={"policy": "custom", "seed": 0},
                                 benchmark="custom", policy_factory=lambda: lambda context: True)
        fresh(actor=lambda: "", total_trials=2, metadata={"case_id": "a"})
        fresh(actor=lambda: "", total_trials=2, metadata={"case_id": "b"})

    def test_public_simulator_gate_defaults_and_both_endpoints(self):
        from vrlbench import simulate
        source = SimulationSpec.from_path(ROOT / "examples/configs/local_feedback.json")
        with tempfile.TemporaryDirectory() as directory:
            results = []
            for name, epsilon in [("default", None), ("exposed", 0), ("hidden", 1)]:
                spec = replace(source, output_dir=str(Path(directory) / name),
                               replay_gate=None if epsilon is None else ReplayGateSpec(epsilon=epsilon))
                results.append(simulate(spec).require_complete())
            default, exposed, hidden = results
            self.assertEqual(default.metrics, exposed.metrics)
            self.assertEqual(default.metrics["sr_at_t"], 1)
            self.assertEqual(hidden.metrics["rr_at_t"], 0)
            for case in default.cases:
                self.assertTrue(all("replay_gate" not in trial["trial_metadata"] for trial in case["trials"]))
            for case in hidden.cases:
                for trial in case["trials"]:
                    self.assertEqual(trial["context"], "")
                    self.assertIn("replay_gate", trial["trial_metadata"])
            from vrlbench.results import compare_results
            report = compare_results([default, hidden])
            self.assertEqual(report["rows"][1]["replay_gate"]["epsilon"], 1)


if __name__ == "__main__":
    unittest.main()
