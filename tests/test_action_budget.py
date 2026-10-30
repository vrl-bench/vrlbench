"""离线回归：总 Trial 数与每 Trial 动作机会分别计数，且不虚构环境动作。"""

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vrlbench.environments import miniwob, webshop
from vrlbench.runtime import llm, loop
from vrlbench.runtime.budget import (
    ACTION_ATTEMPT_BUDGET_UNIT,
    AO_BUDGET_UNIT,
    EnvironmentBudgetExceeded,
    EnvironmentInteractionLedger,
    TokenUsageLedger,
)
from vrlbench.runtime.er.no_er import EmptyReplay
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


class MiniWoBEnvironment:
    subdomain = "click-test"

    def __init__(self, *, succeeds=True):
        self.resets = []
        self.actions = []
        self.succeeds = succeeds
        self.instance = SimpleNamespace(driver=SimpleNamespace(execute_script=Mock()),
                                        start_time=None)

    def reset(self, *, seed):
        self.resets.append(seed)
        return '<div id="query">Click the button.</div><button>Submit</button>'

    def get_task(self):
        return "Click the button."

    def step(self, action):
        self.actions.append((len(self.resets) - 1, action))
        return "Observed page", 1.0 if self.succeeds else 0.0, self.succeeds, {}


class WebShopEnvironment:
    def __init__(self, *, succeeds=True):
        self.resets = []
        self.actions = []
        self.succeeds = succeeds

    def step(self, session, action):
        if action == "reset":
            self.resets.append(session)
            return "Instruction: Buy a sofa\nInitial page", 0.0, False
        self.actions.append((len(self.resets) - 1, action))
        return "Observed page", 1.0 if self.succeeds else 0.0, self.succeeds


def miniwob_stack():
    def action(kind):
        return lambda value: (kind, value)

    keys = SimpleNamespace(LEFT="left", RIGHT="right", BACKSPACE="backspace",
                           UP="up", DOWN="down")
    return (None, action("type"), action("click"), action("option"), action("move"), keys)


class TwoLevelEpisodeBudgetTests(unittest.TestCase):
    def setUp(self):
        self.no_client = patch.object(llm, "get_client", side_effect=AssertionError("禁止真实模型请求"))
        self.no_client.start()
        self.addCleanup(self.no_client.stop)

    def test_missing_actor_budget_cannot_use_environment_default(self):
        for module in (miniwob, webshop):
            with self.subTest(environment=module.__name__):
                env = Mock()
                with patch.dict(os.environ, {"MINIWOB_MAX_ENVIRONMENT_STEPS": "99"}):
                    with self.assertRaisesRegex(ValueError, "explicit positive integer"):
                        module.make_actor(env, 0, llm_fn=Mock())
                env.reset.assert_not_called()
                env.step.assert_not_called()

    def actor_case(self, stack, benchmark, responses, *, succeeds=True, **budget):
        model = Mock(side_effect=responses) if isinstance(responses, list) else Mock(return_value=responses)
        if benchmark == "miniwob":
            stack.enter_context(patch.dict(os.environ, {"MINIWOB_ACTOR_PROFILE": "react_batch"}))
            stack.enter_context(patch.object(miniwob, "_load_mw_stack", return_value=miniwob_stack()))
            env = MiniWoBEnvironment(succeeds=succeeds)
            actor = miniwob.make_actor(env, seed=7, model="offline", llm_fn=model, **budget)
            valid = 'Thought: Click the visible button.\nAction:\n```\nagent.click_xpath("//button")\n```'
            manager = miniwob.make_manager()
        else:
            env = WebShopEnvironment(succeeds=succeeds)
            actor = webshop.make_actor(env, 0, model="offline", llm_fn=model, **budget)
            valid = "search[sofa]"
            manager = webshop.make_manager()
        return env, model, actor, manager, valid

    def run_episode(self, actor, manager, benchmark):
        return loop.run_episode(actor, manager, EmptyReplay(), "offline", total_trials=3,
                                metadata={"benchmark": benchmark})

    def test_all_invalid_replies_use_exactly_two_calls_in_each_of_three_trials(self):
        for benchmark in ("miniwob", "webshop"):
            with self.subTest(benchmark=benchmark), ExitStack() as stack:
                env, model, actor, manager, _ = self.actor_case(
                    stack, benchmark, "not a dispatchable action", max_environment_steps=2)
                result = self.run_episode(actor, manager, benchmark)
                self.assertFalse(result.solved)
                self.assertEqual([trial.trial for trial in result.trials], [0, 1, 2])
                self.assertEqual(model.call_count, 6)
                self.assertEqual(len(env.resets), 3)
                self.assertEqual(env.actions, [])
                for trial in result.trials:
                    metadata = trial.trial_metadata
                    self.assertEqual((metadata["actor_turns"], metadata["environment_resets"]), (2, 1))
                    self.assertEqual((metadata["environment_steps"], metadata["invalid_action_attempts"],
                                      metadata["action_attempts"], metadata["remaining_environment_steps"]),
                                     (0, 2, 2, 0))
                    self.assertEqual(metadata["budget_unit"], ACTION_ATTEMPT_BUDGET_UNIT)
                    self.assertEqual(metadata["trial_close_reason"], "ao_horizon")
                    self.assertTrue(metadata["environment_truncated"])
                    self.assertFalse(metadata["protocol_failure"])
                    self.assertTrue(metadata["policy_evidence_eligible"])
                    self.assertEqual(metadata["policy_ao_steps"], 0)
                    events = metadata["invalid_action_events"]
                    self.assertEqual([event["after_environment_step"] for event in events], [0, 0])
                    self.assertEqual([(event["budget_used_before"], event["budget_used_after"])
                                      for event in events], [(0, 1), (1, 2)])
                self.assertEqual(actor.get_policy_evidence()["steps"], [])

    def test_success_in_second_trial_prevents_a_third_reset_or_model_call(self):
        for benchmark in ("miniwob", "webshop"):
            with self.subTest(benchmark=benchmark), ExitStack() as stack:
                env, model, actor, manager, valid = self.actor_case(
                    stack, benchmark, [], max_environment_steps=2)
                model.side_effect = ["invalid", "", "invalid", valid,
                                     AssertionError("成功后不得继续请求模型")]
                result = self.run_episode(actor, manager, benchmark)
                self.assertTrue(result.solved)
                self.assertEqual([trial.success for trial in result.trials], [False, True])
                self.assertEqual(model.call_count, 4)
                self.assertEqual(len(env.resets), 2)
                self.assertEqual(len(env.actions), 1)
                self.assertEqual(env.actions[0][0], 1)
                first, second = [trial.trial_metadata for trial in result.trials]
                self.assertEqual((first["environment_steps"], first["invalid_action_attempts"]), (0, 2))
                self.assertEqual((second["environment_steps"], second["invalid_action_attempts"],
                                  second["action_attempts"]), (1, 1, 2))
                self.assertEqual(second["trial_close_reason"], "environment_success")
                self.assertEqual(second["policy_ao_steps"], 1)

    def test_miniwob_has_no_hidden_task_decision_cap_and_legacy_max_steps_is_only_fallback(self):
        for budget, expected in (({"max_environment_steps": 7}, 7),
                                 ({"max_steps": 2}, 2),
                                 ({"max_steps": 0, "max_environment_steps": 7}, 7),
                                 ({"max_steps": 1, "max_environment_steps": 7}, 7)):
            with self.subTest(budget=budget), ExitStack() as stack:
                env, model, actor, manager, valid = self.actor_case(
                    stack, "miniwob", "", succeeds=False, **budget)
                model.return_value = valid
                self.assertFalse(manager(actor()))
                metadata = actor.get_trial_metadata()
                self.assertEqual(model.call_count, expected)
                self.assertEqual(len(env.actions), expected)
                self.assertEqual(metadata["action_attempts"], expected)
                self.assertEqual(metadata["max_environment_steps"], expected)
                self.assertEqual(metadata["trial_close_reason"], "ao_horizon")

    def test_explicit_webshop_action_budget_ignores_unused_legacy_max_steps(self):
        with ExitStack() as stack:
            env, model, actor, manager, _ = self.actor_case(
                stack, "webshop", "invalid", max_steps=0, max_environment_steps=2)
            self.assertFalse(manager(actor()))
            metadata = actor.get_trial_metadata()
            self.assertEqual(model.call_count, 2)
            self.assertEqual(env.actions, [])
            self.assertEqual(metadata["max_environment_steps"], 2)
            self.assertEqual(metadata["action_attempts"], 2)
            self.assertEqual(metadata["trial_close_reason"], "ao_horizon")

    def test_miniwob_batch_executes_only_primitives_that_fit_after_invalid_reply(self):
        with ExitStack() as stack:
            env, model, actor, manager, _ = self.actor_case(
                stack, "miniwob", [], succeeds=False, max_environment_steps=2)
            model.side_effect = ["invalid", 'Thought: The actions are known.\nAction:\n```\n'
                                 'agent.click_xpath("//button")\nagent.type("hello")\n```',
                                 AssertionError("动作预算耗尽后不得继续请求模型")]
            self.assertFalse(manager(actor()))
            metadata = actor.get_trial_metadata()
            self.assertEqual(model.call_count, 2)
            self.assertEqual(env.actions, [(0, ("click", "//button"))])
            self.assertEqual((metadata["environment_steps"], metadata["invalid_action_attempts"],
                              metadata["action_attempts"], metadata["remaining_environment_steps"]),
                             (1, 1, 2, 0))
            self.assertEqual(len(actor.get_policy_evidence()["steps"]), 1)
            self.assertEqual(metadata["trial_close_reason"], "ao_horizon")


class ProviderActionBoundaryTests(unittest.TestCase):
    def provider_scope(self, stack, create, ledger):
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        stack.enter_context(patch.object(llm, "get_client", return_value=client))
        stack.enter_context(patch.object(llm, "_respect_provider_call_spacing"))
        stack.enter_context(patch.object(llm, "_estimate_message_tokens", return_value=1))
        stack.enter_context(patch.dict(os.environ, {"VEX2_ALLOWED_RESPONSE_MODELS": "offline",
                                                  "VEX2_REQUIRE_SYSTEM_FINGERPRINT": "0"}))
        stack.enter_context(patch.object(llm.time, "sleep"))
        stack.enter_context(llm.token_budget_scope(ledger))

    @staticmethod
    def response(content):
        return SimpleNamespace(model="offline", usage=SimpleNamespace(
            prompt_tokens=3, completion_tokens=2, total_tokens=5),
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    def test_default_actor_transports_charge_real_empty_content_but_not_provider_failures(self):
        for benchmark in ("miniwob", "webshop"):
            for fails in (False, True):
                with self.subTest(benchmark=benchmark, fails=fails), ExitStack() as stack:
                    error = OSError("离线模拟的连接故障")
                    create = Mock(side_effect=error) if fails else Mock(return_value=self.response(""))
                    ledger = TokenUsageLedger()
                    self.provider_scope(stack, create, ledger)
                    if benchmark == "miniwob":
                        stack.enter_context(patch.dict(os.environ, {"MINIWOB_ACTOR_PROFILE": "react_batch"}))
                        stack.enter_context(patch.object(miniwob, "_load_mw_stack", return_value=miniwob_stack()))
                        env = MiniWoBEnvironment()
                        actor = miniwob.make_actor(env, seed=7, model="offline", max_environment_steps=2)
                    else:
                        env = WebShopEnvironment()
                        actor = webshop.make_actor(env, 0, model="offline", max_environment_steps=2)
                    if fails:
                        with self.assertRaises(RuntimeError):
                            actor()
                    else:
                        actor()
                    metadata = actor.get_trial_metadata()
                    expected = 0 if fails else 2
                    self.assertEqual(create.call_count, 1 if fails else 2)
                    self.assertEqual(metadata["invalid_action_attempts"], expected)
                    self.assertEqual(metadata["action_attempts"], expected)
                    self.assertEqual(metadata["environment_steps"], 0)
                    self.assertEqual(len(metadata["invalid_action_events"]), expected)
                    self.assertEqual(env.actions, [])
                    self.assertEqual(len(env.resets), 1)


if __name__ == "__main__":
    unittest.main()
