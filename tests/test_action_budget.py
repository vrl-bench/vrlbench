"""离线回归：总 Trial 数与每 Trial 动作机会分别计数，且不虚构环境动作。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vrlbench.runtime.budget import (
    ACTION_ATTEMPT_BUDGET_UNIT,
    AO_BUDGET_UNIT,
    EnvironmentBudgetExceeded,
    EnvironmentInteractionLedger,
    TokenUsageLedger,
)
from vrlbench.runtime.shared_t1 import validate_shared_t1_environment_accounting


class ActionBudgetLedgerTests(unittest.TestCase):
    def test_real_steps_and_invalid_replies_share_one_horizon(self):
        ledger = EnvironmentInteractionLedger(3, count_invalid_actions=True)
        ledger.begin_episode(0)
        ledger.charge_reset()
        ledger.charge_actor_turn()
        ledger.charge_step()
        ledger.charge_actor_turn()
        ledger.charge_invalid_action()
        ledger.charge_restore()
        self.assertEqual((ledger.steps, ledger.invalid_action_attempts,
                          ledger.action_attempts, ledger.remaining_steps), (1, 1, 2, 1))
        ledger.charge_invalid_action()
        snapshot = ledger.snapshot()
        self.assertEqual(snapshot["budget_unit"], ACTION_ATTEMPT_BUDGET_UNIT)
        self.assertEqual(snapshot["environment_steps"], 1)
        self.assertEqual(snapshot["invalid_action_attempts"], 2)
        self.assertEqual(snapshot["action_attempts"], 3)
        self.assertEqual(snapshot["remaining_environment_steps"], 0)
        self.assertEqual(snapshot["environment_resets"], 1)
        self.assertEqual(snapshot["restore_count"], 1)
        self.assertEqual(snapshot["actor_turns"], 2)
        for operation in (ledger.require_step, ledger.charge_step,
                          ledger.charge_invalid_action, ledger.charge_reset):
            with self.subTest(operation=operation.__name__):
                with self.assertRaises(EnvironmentBudgetExceeded):
                    operation()
                self.assertEqual(ledger.snapshot(), snapshot)

    def test_new_trial_resets_both_action_counters_and_reset_allowance(self):
        ledger = EnvironmentInteractionLedger(2, count_invalid_actions=True)
        ledger.begin_episode(0)
        ledger.charge_reset()
        ledger.charge_step()
        ledger.charge_invalid_action()
        ledger.charge_actor_turn()
        ledger.charge_restore()
        ledger.begin_episode(1)
        self.assertEqual((ledger.trial_index, ledger.steps, ledger.invalid_action_attempts,
                          ledger.action_attempts, ledger.remaining_steps), (1, 0, 0, 0, 2))
        self.assertEqual((ledger.resets, ledger.actor_turns, ledger.restore_count), (0, 0, 0))
        ledger.charge_reset()
        ledger.charge_step(2)
        self.assertEqual((ledger.steps, ledger.action_attempts, ledger.remaining_steps), (2, 2, 0))

    def test_multi_step_admission_is_atomic_and_requires_active_trial(self):
        ledger = EnvironmentInteractionLedger(2, count_invalid_actions=True)
        with self.assertRaises(RuntimeError):
            ledger.charge_invalid_action()
        ledger.begin_episode(0)
        ledger.charge_invalid_action()
        before = ledger.snapshot()
        for operation in (ledger.require_step, ledger.charge_step):
            with self.subTest(operation=operation.__name__):
                with self.assertRaises(EnvironmentBudgetExceeded):
                    operation(2)
                self.assertEqual(ledger.snapshot(), before)
                for count in (0, -1):
                    with self.assertRaises(ValueError):
                        operation(count)
                self.assertEqual(ledger.snapshot(), before)

    def test_legacy_ledger_still_counts_only_real_environment_transitions(self):
        ledger = EnvironmentInteractionLedger(2)
        ledger.begin_episode(4)
        ledger.charge_reset()
        ledger.charge_actor_turn()
        ledger.charge_step()
        before = ledger.snapshot()
        self.assertEqual(before["budget_unit"], AO_BUDGET_UNIT)
        self.assertNotIn("action_attempts", before)
        self.assertNotIn("invalid_action_attempts", before)
        self.assertEqual(before["environment_steps"], 1)
        self.assertEqual(before["remaining_environment_steps"], 1)
        with self.assertRaises(RuntimeError):
            ledger.charge_invalid_action()
        self.assertEqual(ledger.snapshot(), before)
        ledger.charge_step()
        with self.assertRaises(EnvironmentBudgetExceeded):
            ledger.charge_step()
        self.assertEqual(ledger.steps, 2)


class SharedTrialActionAccountingTests(unittest.TestCase):
    @staticmethod
    def invalid_trial():
        return {"trial_index": 0, "environment_resets": 1, "max_environment_resets": 1,
                "environment_steps": 0, "max_environment_steps": 2,
                "budget_unit": ACTION_ATTEMPT_BUDGET_UNIT, "action_attempts": 2,
                "invalid_action_attempts": 2, "invalid_actor_responses": 2,
                "remaining_environment_steps": 0, "environment_terminated": False,
                "environment_truncated": True, "trial_close_reason": "ao_horizon",
                "protocol_failure": False, "policy_evidence_eligible": True,
                "invalid_action_events": [
                    {"after_environment_step": 0, "raw_response": "", "reason": "empty response",
                     "budget_used_before": 0, "budget_used_after": 1},
                    {"after_environment_step": 0, "raw_response": "invalid", "reason": "no action",
                     "budget_used_before": 1, "budget_used_after": 2}]}

    def validate(self, metadata, *, benchmark="miniwob", horizon=2, success=False):
        return validate_shared_t1_environment_accounting(
            metadata, benchmark=benchmark, source="offline regression",
            expected_max_environment_steps=horizon, success=success)

    def test_all_invalid_trial_is_valid_without_fictitious_environment_steps(self):
        metadata = self.invalid_trial()
        self.assertEqual(self.validate(metadata), metadata)

    def test_horizon_close_before_the_action_budget_is_exhausted_is_rejected(self):
        metadata = self.invalid_trial()
        metadata.update(action_attempts=1, invalid_action_attempts=1,
                        invalid_actor_responses=1, remaining_environment_steps=1)
        metadata["invalid_action_events"] = metadata["invalid_action_events"][:1]
        with self.assertRaisesRegex(ValueError, "before the budget was exhausted"):
            self.validate(metadata)

    def test_forged_invalid_event_position_or_counter_is_rejected(self):
        for change in ({"after_environment_step": 1}, {"budget_used_before": 0},
                       {"budget_used_after": 3}, {"raw_response": None}, {"reason": ""}):
            with self.subTest(change=change):
                metadata = deepcopy(self.invalid_trial())
                metadata["invalid_action_events"][1].update(change)
                with self.assertRaisesRegex(ValueError, "inconsistent position or budget"):
                    self.validate(metadata)
        metadata = self.invalid_trial()
        metadata["invalid_actor_responses"] = 1
        with self.assertRaisesRegex(ValueError, "count disagrees"):
            self.validate(metadata)

    def test_action_counts_and_remaining_budget_must_agree(self):
        for change in ({"action_attempts": 1}, {"action_attempts": 3},
                       {"environment_steps": 1}, {"invalid_action_attempts": -1},
                       {"remaining_environment_steps": 1}, {"action_attempts": True},
                       {"invalid_action_attempts": False}, {"invalid_action_events": []}):
            with self.subTest(change=change):
                metadata = self.invalid_trial()
                metadata.update(change)
                with self.assertRaises(ValueError):
                    self.validate(metadata)

    def test_interleaved_real_actions_and_invalid_replies_are_valid(self):
        metadata = self.invalid_trial()
        metadata.update(environment_steps=2, max_environment_steps=4, action_attempts=4)
        for index, (event, position) in enumerate(zip(metadata["invalid_action_events"], (1, 2))):
            event.update(after_environment_step=position,
                         budget_used_before=position + index,
                         budget_used_after=position + index + 1)
        self.assertEqual(self.validate(metadata, horizon=4), metadata)
        metadata["invalid_action_events"].reverse()
        with self.assertRaises(ValueError):
            self.validate(metadata, horizon=4)

    def test_environment_termination_can_leave_unused_action_budget(self):
        for success in (False, True):
            with self.subTest(success=success):
                metadata = self.invalid_trial()
                metadata.update(environment_steps=1, action_attempts=1,
                                invalid_action_attempts=0, invalid_actor_responses=0,
                                remaining_environment_steps=1, invalid_action_events=[],
                                environment_terminated=True, environment_truncated=False,
                                trial_close_reason="environment_success" if success
                                else "environment_terminal_failure")
                self.assertEqual(self.validate(metadata, success=success), metadata)

    def test_new_budget_does_not_accept_legacy_early_actor_close_reasons(self):
        for benchmark, reason, protocol_failure in (
                ("miniwob", "actor_protocol_failure", True),
                ("miniwob", "actor_decision_cap", False),
                ("webshop", "actor_protocol_failure", True),
                ("webshop", "provider_call_safety_cap", True)):
            with self.subTest(benchmark=benchmark, reason=reason):
                metadata = self.invalid_trial()
                metadata.update(environment_steps=1, action_attempts=1,
                                invalid_action_attempts=0, invalid_actor_responses=0,
                                remaining_environment_steps=1, invalid_action_events=[],
                                environment_truncated=False, trial_close_reason=reason,
                                protocol_failure=protocol_failure)
                with self.assertRaisesRegex(ValueError, "invalid trial close state"):
                    self.validate(metadata, benchmark=benchmark)

    def test_legacy_ao_records_keep_their_original_close_states(self):
        for benchmark, reason, protocol_failure in (
                ("miniwob", "actor_protocol_failure", True),
                ("miniwob", "actor_decision_cap", False),
                ("webshop", "actor_protocol_failure", True),
                ("webshop", "provider_call_safety_cap", True)):
            with self.subTest(benchmark=benchmark, reason=reason):
                metadata = self.invalid_trial()
                for key in ("action_attempts", "invalid_action_attempts",
                            "invalid_actor_responses", "invalid_action_events",
                            "remaining_environment_steps"):
                    metadata.pop(key)
                metadata.update(budget_unit=AO_BUDGET_UNIT, environment_steps=1,
                                environment_truncated=False, trial_close_reason=reason,
                                protocol_failure=protocol_failure)
                self.assertEqual(self.validate(metadata, benchmark=benchmark), metadata)



if __name__ == "__main__":
    unittest.main()
