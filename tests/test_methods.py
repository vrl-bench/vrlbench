"""Offline lifecycle checks with synthetic traces and fake provider responses."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vrlbench.methods import (ActorContext, CaseConfig, MemoryMethod, TrialRecord,
                             list_methods, load_method, render_trajectory)

HAS_RUNTIME = importlib.util.find_spec("openai") is not None


class FakeClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=value))],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18),
            model=kwargs["model"], system_fingerprint=None, id="synthetic-response")


class LazyMethodTests(unittest.TestCase):
    def test_import_and_factory_do_not_load_runtime(self):
        source = (
            "import sys; import vrlbench; "
            "assert vrlbench.list_methods() == ('retry','reflexion','dc_cu','ace','vex2'); "
            "[vrlbench.load_method(n) for n in vrlbench.list_methods()]; "
            "assert 'vrlbench.runtime' not in sys.modules; assert 'openai' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", source], check=True, cwd=ROOT / "src",
                       env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))

    def test_repository_root_imports_real_package_method_api_lazily(self):
        source = (
            "import sys; from pathlib import Path; import vrlbench; "
            "assert Path(vrlbench.__file__).resolve() == Path.cwd() / 'src' / 'vrlbench' / '__init__.py'; "
            "names = ('ActorContext','CaseConfig','MemoryMethod','TrialRecord',"
            "'list_methods','load_method','render_trajectory'); "
            "assert all(hasattr(vrlbench, name) for name in names); "
            "assert vrlbench.ActorContext().system == ''; "
            "assert vrlbench.CaseConfig(total_trials=3).total_trials == 3; "
            "[vrlbench.load_method(n) for n in vrlbench.list_methods()]; "
            "assert 'vrlbench.runtime' not in sys.modules; assert 'openai' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", source], check=True, cwd=ROOT,
                       env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))

    def test_explicit_trial_budget_and_aliases(self):
        with self.assertRaises(TypeError):
            CaseConfig()
        first, second = CaseConfig(total_trials=3), CaseConfig(total_trials=3)
        first.metadata["key"] = 1
        self.assertEqual(second.metadata, {})
        self.assertEqual(ActorContext(), ActorContext("", ""))
        for alias in ("Reflexion", "DC-Cu*", "ACE*", "VEX²", "dsst",
                      "dynamic_cheatsheet_adapted", "ace_adapted"):
            self.assertIsInstance(load_method(alias), MemoryMethod)
        with self.assertRaises(ValueError):
            load_method("unimplemented")


@unittest.skipUnless(HAS_RUNTIME, "Install the runtime extra for reference methods")
class MethodTests(unittest.TestCase):
    def test_runtime_registry_keeps_current_methods_without_retired_imports(self):
        from vrlbench.runtime import VEX2
        from vrlbench.runtime.controllers import make_controller
        from vrlbench.runtime.er import _SLOTS, make_er_slot

        self.assertEqual(set(_SLOTS), {"none", "reflexion", "dynamic_cheatsheet_adapted", "ace_adapted"})
        for name in ("per", "her"):
            with self.subTest(method=name), self.assertRaises(ValueError):
                make_er_slot(name)
        with self.assertRaises(ValueError):
            make_controller("unsupported_controller")
        with self.assertRaises(ValueError):
            VEX2(trials=3, controller="unsupported_controller")
        self.assertIsNone(make_controller("none"))
        self.assertEqual(make_controller("dsst").method_id, "dsst")
        with self.assertRaisesRegex(ValueError, "gold or oracle"):
            VEX2(trials=3, allow_gold_feedback=True).run(actor=lambda: "", manager=lambda _: False)
        with self.assertRaisesRegex(ValueError, "requires er='none'"):
            VEX2(trials=3, er="reflexion").run(actor=lambda: "", manager=lambda _: False)
        self.assertFalse(any(name.startswith("vrlbench.runtime.scheduling") or
                             name in {"vrlbench.runtime.er.per", "vrlbench.runtime.er.her",
                                      "vrlbench.runtime.controllers.integrated"}
                             for name in sys.modules))

    def trace(self, *, reason="interaction_limit", task="Buy the grey sofa"):
        return render_trajectory(task=task, initial_observation="Search page",
                                 steps=[{"action": "click[sofa]", "observation": "Wrong color"}],
                                 close_reason=reason)

    def method(self, name, *responses, trials=3, metadata=None):
        client = FakeClient(*responses)
        method = load_method(name, client=client)
        method.reset_case(CaseConfig(total_trials=trials, environment_rules="Click visible controls only.",
                                     metadata=metadata or {}))
        self.assertEqual(method.context_for_trial(0), ActorContext())
        return method, client

    def failure(self, method, index=0, **kwargs):
        method.observe_trial(TrialRecord(index, self.trace(), False, **kwargs))

    def test_retry_lifecycle_and_record_validation(self):
        method, client = self.method("retry", trials=2)
        with self.assertRaises(ValueError):
            method.context_for_trial(1)
        with self.assertRaises(ValueError):
            method.observe_trial(TrialRecord(0, "invented summary", False))
        self.failure(method)
        self.assertEqual(method.context_for_trial(1), ActorContext())
        self.failure(method, 1)
        self.assertEqual(client.calls, [])
        with self.assertRaises(RuntimeError):
            method.context_for_trial(2)
        self.assertEqual(method.snapshot()["completed_trials"], 2)
        self.assertEqual(method.snapshot()["history"], [self.trace(), self.trace()])

    def test_reflexion_source_prompt_and_user_context(self):
        method, client = self.method("reflexion", "Inspect before buying.",
                                     metadata={"benchmark": "webshop"})
        self.failure(method)
        context = method.context_for_trial(1)
        self.assertEqual(context.system, "")
        self.assertIn("Your memory for the task below:", context.user)
        self.assertIn("Trial 0:\nInspect before buying.", context.user)
        self.assertEqual(method.context_for_trial(1), context)
        self.assertEqual(len(client.calls), 1)
        messages = client.calls[0]["messages"]
        self.assertIn("[Public benchmark environment]", messages[0]["content"])
        self.assertIn("> click[sofa]\nObservation:\nWrong color", messages[-1]["content"])

    def test_reflexion_explicit_benchmark_and_native_miniwob_trace(self):
        method = load_method("reflexion", client=FakeClient())
        with self.assertRaisesRegex(ValueError, "explicit benchmark"):
            method.reset_case(CaseConfig(total_trials=3))
        method, client = self.method("reflexion", "Native reflection",
                                     metadata={"benchmark": "miniwob"})
        with self.assertRaisesRegex(ValueError, "source-native Actor"):
            self.failure(method)
        self.failure(method, actor_trajectory="DECISION BLOCK: click target")
        self.assertIn("DECISION BLOCK: click target", client.calls[0]["messages"][-1]["content"])

    def test_dc_cu_reuses_cheatsheet_writer_and_actor_trajectory(self):
        method, client = self.method("dc_cu", "<cheatsheet>Check the color.</cheatsheet>",
                                     metadata={"task": "Original task"})
        self.failure(method, actor_trajectory="RAW ACTOR DECISION AND OBSERVATIONS")
        self.assertEqual(method.context_for_trial(1), ActorContext(
            user="CHEATSHEET:\n'''\nCheck the color.\n'''"))
        prompt = client.calls[0]["messages"][-1]["content"]
        self.assertIn("RAW ACTOR DECISION AND OBSERVATIONS", prompt)
        self.assertIn("Original task", prompt)
        self.assertEqual(method.snapshot()["history"], [self.trace()])

    def test_ace_preserves_two_call_reflector_curator_flow(self):
        curator = json.dumps({"reasoning": "Synthetic curator response", "operations": [
            {"type": "ADD", "section": "verification_checklist", "content": "Verify color"}]})
        method, client = self.method("ace", "Synthetic reflection", curator)
        self.failure(method, actor_trajectory="RAW ACE ACTOR TRACE")
        context = method.context_for_trial(1)
        self.assertIn("ACE PLAYBOOK FROM PRIOR ATTEMPTS", context.user)
        self.assertIn("Verify color", context.user)
        self.assertEqual(context.system, "")
        self.assertEqual(len(client.calls), 2)
        self.assertIn("RAW ACE ACTOR TRACE", client.calls[0]["messages"][-1]["content"])
        self.assertIn("Synthetic reflection", client.calls[1]["messages"][-1]["content"])

    def test_vex_joint_raw_response_and_budget_are_unchanged(self):
        raw = "  Grounded semantic tree\nNEXT: policy A; allocation = 2\n\n"
        method, client = self.method("vex2", raw, "second response", trials=4)
        self.failure(method)
        self.assertEqual(client.calls, [])  # scheduling occurs at the next read
        context = method.context_for_trial(1)
        self.assertEqual(context, ActorContext(system=raw))
        self.assertEqual(method.context_for_trial(1), context)
        self.assertEqual(len(client.calls), 1)
        payload = json.loads(client.calls[0]["messages"][-1]["content"])
        self.assertEqual(payload["remaining_complete_attempts"], 3)
        self.assertEqual(payload["task_or_task_family_scope"], "Buy the grey sofa")
        self.assertEqual(payload["basic_environment_rules"], "Click visible controls only.")
        forest = payload["complete_ordered_failure_only_ao_forest"]["attempts"]
        self.assertEqual(forest, [{"attempt": 1, "canonical_policy_ao_trace": self.trace()}])
        method.observe_trial(TrialRecord(1, "", False, metadata={"policy_evidence_eligible": False}))
        method.context_for_trial(2)
        payload = json.loads(client.calls[-1]["messages"][-1]["content"])
        self.assertEqual(payload["remaining_complete_attempts"], 2)
        self.assertEqual(len(payload["complete_ordered_failure_only_ao_forest"]["attempts"]), 1)

    def test_stop_reset_and_no_final_failure_writer_for_all_methods(self):
        for name in list_methods():
            with self.subTest(name=name):
                method, client = self.method(name, trials=1, metadata={"benchmark": "webshop"})
                self.failure(method)
                self.assertEqual(client.calls, [])
                with self.assertRaises(RuntimeError):
                    method.context_for_trial(1)
                method.reset_case(CaseConfig(total_trials=3, metadata={"benchmark": "webshop"},
                                             environment_rules="Click visible controls only."))
                self.assertEqual(method.context_for_trial(0), ActorContext())
                self.assertEqual(method.snapshot()["history"], [])
                method.observe_trial(TrialRecord(0, self.trace(reason="success"), True))
                self.assertEqual(client.calls, [])
                with self.assertRaises(RuntimeError):
                    method.context_for_trial(1)

    def test_ineligible_closure_does_not_write_or_enter_history(self):
        method, client = self.method("ace")
        method.observe_trial(TrialRecord(0, "", False,
                                        metadata={"policy_evidence_eligible": False}))
        self.assertEqual(method.context_for_trial(1), ActorContext())
        self.assertEqual(method.snapshot()["history"], [])
        self.assertEqual(client.calls, [])

    def test_global_client_context_and_active_accounting_are_restored(self):
        from vrlbench.runtime import llm
        from vrlbench.runtime.budget import TokenUsageLedger

        method, client = self.method("dc_cu", "<cheatsheet>Color</cheatsheet>")
        previous_client, system, user = llm._client, llm.get_context(), llm.get_user_context()
        sentinel = object()
        try:
            llm._client = sentinel
            llm.set_context("caller system")
            llm.set_user_context("caller user")
            ledger = TokenUsageLedger(require_provider_usage=True)
            with llm.token_budget_scope(ledger):
                self.failure(method)
                self.assertIs(llm._get_token_ledger(), ledger)
            self.assertEqual(ledger.api_calls, 1)
            self.assertEqual(ledger.total_tokens, 18)
            self.assertIs(llm._client, sentinel)
            self.assertEqual(llm.get_context(), "caller system")
            self.assertEqual(llm.get_user_context(), "caller user")
            prompt = client.calls[0]["messages"][-1]["content"]
            self.assertNotIn("caller user", prompt)
            self.assertNotIn("caller system", str(client.calls))
        finally:
            llm._client = previous_client
            llm.set_context(system)
            llm.set_user_context(user)

    def test_provider_exception_restores_global_state_without_disabling_ledger(self):
        from vrlbench.runtime import llm
        from vrlbench.runtime.budget import TokenUsageLedger

        method, client = self.method("dc_cu", RuntimeError("synthetic transport failure"))
        previous_client, system, user = llm._client, llm.get_context(), llm.get_user_context()
        sentinel = object()
        try:
            llm._client = sentinel
            llm.set_context("outside system")
            llm.set_user_context("outside user")
            ledger = TokenUsageLedger(require_provider_usage=True)
            with llm.token_budget_scope(ledger):
                with self.assertRaises(llm.ProviderCallUncertain):
                    self.failure(method)
                self.assertIs(llm._get_token_ledger(), ledger)
            self.assertEqual(len(client.calls), 1)
            self.assertEqual(ledger.api_failures, 1)
            self.assertIs(llm._client, sentinel)
            self.assertEqual(llm.get_context(), "outside system")
            self.assertEqual(llm.get_user_context(), "outside user")
        finally:
            llm._client = previous_client
            llm.set_context(system)
            llm.set_user_context(user)

    def test_default_client_is_lazy_cached_but_worker_client_is_borrowed(self):
        from vrlbench.runtime import llm
        from vrlbench.runtime.budget import TokenUsageLedger

        own = FakeClient("<cheatsheet>first</cheatsheet>", "<cheatsheet>second</cheatsheet>")
        worker = FakeClient("<cheatsheet>worker</cheatsheet>")
        method = load_method("dc_cu")
        previous = llm._client
        try:
            llm._client = worker
            with patch.object(llm, "OpenAI", return_value=own) as create:
                method.reset_case(CaseConfig(total_trials=3))
                method.context_for_trial(0)
                create.assert_not_called()
                self.failure(method)
                self.assertIs(llm._client, worker)
                self.failure(method, 1)
                self.assertEqual(create.call_count, 1)
                method.reset_case(CaseConfig(total_trials=3))
                ledger = TokenUsageLedger(require_provider_usage=True)
                with llm.token_budget_scope(ledger):
                    self.failure(method)
                self.assertEqual(create.call_count, 1)
                self.assertEqual(ledger.api_calls, 1)
                self.assertEqual(len(worker.calls), 1)
                self.assertIs(llm._client, worker)
        finally:
            llm._client = previous


if __name__ == "__main__":
    unittest.main()
