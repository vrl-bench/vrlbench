"""Offline checks that every closed trial reaches the whole-method API once."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@unittest.skipUnless(importlib.util.find_spec("openai") is not None, "requires runtime extra")
class TerminalTrialLifecycle(unittest.TestCase):
    def setUp(self):
        from vrlbench.backends import _MethodController
        from vrlbench.methods import ActorContext, MemoryMethod
        from vrlbench.runtime import llm, loop
        from vrlbench.runtime.er.base import ERSlot
        from vrlbench.runtime.trajectory import render_policy_trajectory

        class Method(MemoryMethod):
            def reset_case(self, config):
                self.config, self.records = config, []

            def context_for_trial(self, trial_index):
                return ActorContext()

            def observe_trial(self, record):
                self.records.append(record)

            def snapshot(self):
                return {"observed": len(self.records)}

        class Slot(ERSlot):
            def __init__(self):
                self.reset()

            def reset(self):
                self.writes = []

            def build_section(self):
                return ""

            def on_trial_end(self, trajectory, **kwargs):
                self.writes.append((trajectory, kwargs))

        class Actor:
            def __init__(self, eligibility, success_at=None):
                self.eligibility, self.success_at = eligibility, success_at

            def set_trial_context(self, *, trial_index, total_trials):
                self.index = trial_index

            def __call__(self):
                return self.get_actor_trajectory()

            def get_actor_trajectory(self):
                outcome = "success" if self.index == self.success_at else "failure"
                return f"raw Actor response {self.index}: {outcome}"

            def get_trial_metadata(self):
                eligible = self.eligibility[self.index]
                return {"policy_evidence_eligible": eligible,
                        "protocol_failure": not eligible,
                        "trial_close_reason": "task_failure" if eligible else "actor_protocol_failure"}

            def get_policy_evidence(self):
                return {"schema_version": 1, "task": "synthetic task",
                        "initial_observation": f"reset {self.index}",
                        "steps": [{"action": "actual action", "observation": f"visible outcome {self.index}"}]
                        if self.eligibility[self.index] else [],
                        "trial_close_reason": self.get_trial_metadata()["trial_close_reason"]}

        self.loop, self.Method, self.Slot, self.Actor = loop, Method, Slot, Actor
        self.Controller, self.render = _MethodController, render_policy_trajectory
        self.no_api = patch.object(llm, "complete", side_effect=AssertionError("unexpected model call"))
        self.no_api.start()
        self.addCleanup(self.no_api.stop)

    def controller(self):
        method = self.Method()
        return method, self.Controller(method, name="probe", model="offline", native_rules=False)

    def run_fresh(self, eligibility, *, success_at=None, controller=None):
        actor, slot = self.Actor(eligibility, success_at), self.Slot()
        result = self.loop.run_episode(
            actor, lambda trace: trace.endswith("success"), slot, "offline",
            total_trials=len(eligibility), controller=controller,
            metadata={"benchmark": "custom"},
        )
        return result, slot

    def prepare_shared(self, *, eligible=True, success=False, controller=None):
        actor = self.Actor([eligible], 0 if success else None)
        actor.index = 0
        slot = self.Slot()
        result, _ledger = self.loop.prepare_shared_t1(
            trajectory=actor(), policy_trajectory=self.render(actor.get_policy_evidence()),
            actor_trajectory=actor.get_actor_trajectory(), success=success,
            er_slot=slot, model="offline", total_trials=1, controller=controller,
            metadata={"benchmark": "custom"}, trial_metadata=actor.get_trial_metadata(),
        )
        return result, slot

    def test_fresh_valid_failures_notify_once_and_keep_the_final_trace(self):
        for total in (1, 2):
            with self.subTest(total=total):
                method, controller = self.controller()
                result, slot = self.run_fresh([True] * total, controller=controller)
                self.assertEqual([record.trial_index for record in method.records], list(range(total)))
                for record, trial in zip(method.records, result.trials):
                    self.assertFalse(record.success)
                    self.assertTrue(record.metadata["policy_evidence_eligible"])
                    self.assertEqual(record.trajectory, trial.policy_trajectory)
                    self.assertIn("policy_ao_trace", record.trajectory)
                final = method.records[-1]
                self.assertTrue(final.metadata["trace_available"])
                self.assertFalse(final.metadata["has_future_trial"])
                self.assertEqual(final.actor_trajectory, result.trials[-1].actor_trajectory)
                self.assertEqual(len(slot.writes), total - 1)

    def test_fresh_ineligible_failures_do_not_become_policy_traces(self):
        for eligibility in ([False], [False, True]):
            with self.subTest(eligibility=eligibility):
                method, controller = self.controller()
                result, slot = self.run_fresh(eligibility, controller=controller)
                self.assertEqual(len(method.records), len(result.trials))
                invalid = method.records[0]
                self.assertEqual(invalid.trajectory, "")
                self.assertFalse(invalid.metadata["policy_evidence_eligible"])
                self.assertFalse(invalid.metadata["trace_available"])
                self.assertEqual(invalid.metadata["has_future_trial"], len(eligibility) > 1)
                self.assertEqual(slot.writes, [])

    def test_shared_terminal_failure_preserves_eligibility_and_real_trace(self):
        for eligible in (True, False):
            with self.subTest(eligible=eligible):
                method, controller = self.controller()
                result, slot = self.prepare_shared(eligible=eligible, controller=controller)
                self.assertEqual(len(method.records), 1)
                record, trial = method.records[0], result.trials[0]
                self.assertEqual(record.trial_index, 0)
                self.assertFalse(record.success)
                self.assertEqual(record.metadata["policy_evidence_eligible"], eligible)
                self.assertEqual(record.metadata["trace_available"], eligible)
                self.assertFalse(record.metadata["has_future_trial"])
                self.assertEqual(record.trajectory, trial.policy_trajectory if eligible else "")
                self.assertEqual(record.actor_trajectory, trial.actor_trajectory if eligible else None)
                self.assertEqual(slot.writes, [])

    def test_success_notifies_once_and_never_writes_a_following_update(self):
        for shared in (False, True):
            with self.subTest(shared=shared):
                method, controller = self.controller()
                result, slot = (self.prepare_shared(success=True, controller=controller)
                                if shared else self.run_fresh([True, True], success_at=0, controller=controller))
                self.assertTrue(result.solved)
                self.assertEqual(len(method.records), 1)
                self.assertTrue(method.records[0].success)
                self.assertEqual(method.records[0].trajectory, result.trials[0].policy_trajectory)
                self.assertEqual(slot.writes, [])

    def test_optional_close_hooks_preserve_legacy_controller_compatibility(self):
        from vrlbench.runtime.controllers.base import ControllerContext

        class Controller:
            def reset(self):
                self.closed = []

            def set_total_budget(self, total):
                pass

            def build_trial_context(self, **kwargs):
                return ControllerContext()

            def snapshot(self):
                return {}

        class LegacyController(Controller):
            def on_trial_closed_without_policy_evidence(self, **kwargs):
                self.closed.append(kwargs)

        for kind in (Controller, LegacyController):
            for shared in (False, True):
                with self.subTest(controller=kind.__name__, shared=shared):
                    controller = kind()
                    controller.reset()
                    result, slot = (self.prepare_shared(controller=controller) if shared
                                    else self.run_fresh([True], controller=controller))
                    self.assertEqual(len(result.trials), 1)
                    self.assertEqual(slot.writes, [])
                    self.assertEqual(len(controller.closed), 1 if kind is LegacyController else 0)
                    if controller.closed:
                        self.assertEqual(controller.closed[0]["trial_index"], 0)
                        self.assertFalse(controller.closed[0]["has_future_trial"])


if __name__ == "__main__":
    unittest.main()
