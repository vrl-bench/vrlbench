"""WebArena contracts and optional native scorer checks, without model calls."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vrlbench.environments.webarena import WebArenaSession


class ActionParsingError(ValueError):
    pass


class FakeBrowser:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.snapshot = b"\x89PNG\r\n\x1a\nlive browser viewport"
        self.page = SimpleNamespace(url="http://benchmark.local/",
                                    screenshot=lambda **kwargs: self.snapshot)
        self.reset_calls = []
        self.actions = []
        self.closed = False

    def reset(self, *, options):
        self.reset_calls.append(options)
        return {"text": "[1] button Continue"}, {"page": self.page, "fail_error": ""}

    def step(self, action):
        self.actions.append(action)
        # A successful browser action is deliberately not task success.
        return {"text": "[2] text unchanged task"}, 1.0, False, False, {"page": self.page}

    def get_page_client(self, page):
        assert page is self.page
        return "client"

    def close(self):
        self.closed = True


class WebArenaAdapterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "case.json"
        self.path.write_text(json.dumps({
            "task_id": 23, "intent": "Update the project description",
            "eval": {"eval_types": ["string_match"],
                     "reference_answers": {"exact_match": "SCORER-ONLY-TARGET"}},
        }), encoding="utf-8")
        self.case = {"case_id": "webarena-23", "config_file": str(self.path)}
        self.reset_calls = []
        self.score = 0.0
        self.evaluations = []

        def parse(action):
            if action == "broken":
                raise ActionParsingError("invalid")
            return {"action_type": "stop" if action.startswith("stop") else "click"}

        def stop(answer):
            return {"action_type": "stop", "answer": answer}

        def router(path):
            self.assertEqual(Path(path), self.path.resolve())

            def evaluator(**kwargs):
                self.evaluations.append(kwargs)
                return self.score

            return evaluator

        self.upstream = (FakeBrowser, parse, lambda: {"action_type": "none"}, stop,
                         SimpleNamespace(STOP="stop"), ActionParsingError, router)
        self.patchers = [
            patch("vrlbench.environments.webarena._upstream", return_value=self.upstream),
            patch("vrlbench.environments.webarena.create_plugin", return_value=(
                lambda case: self.reset_calls.append(case), {"entrypoint": "test:reset"},
            )),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def session(self, **params):
        return WebArenaSession(self.case, {
            "reset_factory": {"entrypoint": "test:reset_factory"}, **params,
        })

    def test_requires_explicit_website_reset(self):
        with self.assertRaisesRegex(ValueError, "reset_factory"):
            WebArenaSession(self.case, {})

    def test_configuration_and_comparison_preserve_task_identity(self):
        from copy import deepcopy
        from dataclasses import replace
        from vrlbench import ModelSpec, SimulationSpec, WorkloadSpec, plan
        from vrlbench.results import aggregate_cases, compare_results

        params = {"reset_factory": {"entrypoint": "test:reset_factory"}}
        spec = SimulationSpec(WorkloadSpec("webarena", [self.case], params), ModelSpec("fixture"),
                              trials=6, max_environment_steps=15)
        self.assertEqual(plan(spec)["model_calls_executed"], 0)
        for invalid in (None, "", " ", 3):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "judge_model"):
                replace(spec, workload=replace(spec.workload, params={**params, "judge_model": invalid})).validate()
        with self.assertRaisesRegex(ValueError, "reset_factory"):
            replace(spec, workload=replace(spec.workload, params={})).validate()
        with self.assertRaisesRegex(ValueError, "completed SDK"):
            plan(replace(spec, shared_t1_dir=self.directory.name))
        relative = replace(spec, workload=replace(spec.workload, cases=[{**self.case, "config_file": "case.json"}]))
        self.assertEqual(relative.resolved(Path(self.directory.name)).workload.cases[0]["config_file"], str(self.path))
        case = {"case_id": self.case["case_id"], "task": "webarena",
                "trials": [{"trial": 0, "success": True, "trajectory": "done"}],
                "source_identity": self.session().source_identity}
        payload = {"status": "complete", "config": spec.to_dict(),
                   **aggregate_cases([case], 6), "cases": [case]}
        report = compare_results([payload, payload])
        self.assertEqual(report["rows"][0]["counts"]["completed"], 1)
        self.assertEqual(report["environment_identity_check"]["status"], "unverified")
        case["source_identity"]["reset_source_sha256"] = "a" * 64
        self.assertEqual(compare_results([payload, payload])["environment_identity_check"]["status"], "verified")
        changed = deepcopy(payload)
        changed["cases"][0]["source_identity"]["config_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "task/environment identities"):
            compare_results([payload, changed])

    def test_resets_sites_including_first_trial_and_keeps_task_identity(self):
        session = self.session()
        first = session.reset(trial_index=1)
        session.step("click [1]")
        second = session.reset(trial_index=2)
        self.assertEqual(first.identity, second.identity)
        self.assertEqual(len(self.reset_calls), 2)
        self.assertEqual(len(session.env.reset_calls), 2)
        self.assertEqual(len(session.trajectory), 1)
        self.assertNotIn("SCORER-ONLY-TARGET", first.task + first.observation.text)
        self.assertEqual(session.env.kwargs["observation_type"], "accessibility_tree")

    def test_config_contents_change_identity(self):
        first = self.session().reset(trial_index=1)
        task = json.loads(self.path.read_text())
        task["intent"] = "A different task"
        self.path.write_text(json.dumps(task), encoding="utf-8")
        second = self.session().reset(trial_index=1)
        self.assertNotEqual(first.identity, second.identity)

    def test_browser_action_reward_is_not_task_success(self):
        session = self.session()
        session.reset(trial_index=1)
        result = session.step("click [1]")
        self.assertFalse(result.terminated)
        self.assertFalse(session.evaluate())
        self.assertEqual(session.evaluation_score, 0.0)
        self.assertEqual(self.evaluations[0]["trajectory"][-1]["action_type"], "stop")
        self.assertEqual(len(session.trajectory), 4)
        self.assertEqual(self.evaluations[0]["client"], "client")

    def test_explicit_stop_does_not_execute_browser_action(self):
        session = self.session()
        session.reset(trial_index=1)
        result = session.step("stop [answer]")
        self.score = 1.0
        self.assertTrue(result.terminated)
        self.assertTrue(session.evaluate())
        self.assertFalse(session.env.actions)
        self.assertEqual(len(session.trajectory), 2)
        with self.assertRaisesRegex(RuntimeError, "after stop"):
            session.step("click [1]")

    def test_invalid_action_is_a_counted_noop_without_model_repair(self):
        session = self.session()
        session.reset(trial_index=1)
        result = session.step("broken")
        self.assertIn("Invalid action", result.observation.text)
        self.assertEqual(len(session.env.actions), 1)
        self.assertEqual(session.env.actions[0]["action_type"], "none")
        self.assertEqual(session.env.actions[0]["raw_prediction"], "broken")

    def test_recording_captures_live_page_without_making_actor_visual(self):
        session = self.session()
        with self.assertRaisesRegex(RuntimeError, "before capturing"):
            session.capture_observation()
        state = session.reset(trial_index=1)
        self.assertEqual(session.capture_observation(), session.env.snapshot)
        self.assertIsNone(state.observation.image)
        self.assertEqual(len(session.trajectory), 1)

    def test_website_reset_failure_is_not_task_failure(self):
        session = self.session()
        session._reset_sites = lambda case: False
        with self.assertRaisesRegex(RuntimeError, "website reset failed"):
            session.reset(trial_index=1)
        self.assertFalse(session.env.reset_calls)

    def test_evaluator_failure_propagates(self):
        session = self.session()
        session.reset(trial_index=1)
        self.score = float("nan")
        with self.assertRaisesRegex(RuntimeError, "invalid score"):
            session.evaluate()

    def test_fuzzy_match_keeps_targets_out_of_actor_observations(self):
        task = json.loads(self.path.read_text())
        task["eval"]["reference_answers"] = {"fuzzy_match": ["target"]}
        self.path.write_text(json.dumps(task), encoding="utf-8")
        state = self.session().reset(trial_index=0)
        self.assertNotIn("target", state.task + state.observation.text)

    def test_changed_config_cannot_change_scoring_under_old_identity(self):
        session = self.session()
        session.reset(trial_index=1)
        self.path.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "config changed"):
            session.evaluate()

    def test_close_releases_environment(self):
        session = self.session()
        session.close()
        self.assertTrue(session.env.closed)


@unittest.skipUnless(importlib.util.find_spec("webarena") and importlib.util.find_spec("openai"),
                     "requires the native libwebarena/runtime dependencies")
class NativeJudgeTests(unittest.TestCase):
    def test_native_action_rejection_is_charged_without_browser_dispatch(self):
        from vrlbench.environments.interactive import make_actor

        sites = {key: f"http://{key.lower()}.invalid" for key in
                 ("REDDIT", "SHOPPING", "SHOPPING_ADMIN", "GITLAB", "WIKIPEDIA", "MAP", "HOMEPAGE")}
        with patch.dict(os.environ, sites):
            from webarena.browser_env.actions import (
                ActionParsingError as NativeParsingError, ActionTypes,
                create_id_based_action, create_none_action, create_stop_action,
            )
        upstream = (FakeBrowser, create_id_based_action, create_none_action, create_stop_action,
                    ActionTypes, NativeParsingError, lambda path: lambda **kwargs: 0.0)
        replies = iter(["type [12] example text [1]", "type [12] [example text] [1]"])
        prompts = []

        def complete(prompt, **kwargs):
            prompts.append(prompt)
            return next(replies)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.json"
            path.write_text(json.dumps({"intent": "Search the site",
                                        "eval": {"eval_types": ["url_match"]}}))
            with (patch("vrlbench.environments.webarena._upstream", return_value=upstream),
                  patch("vrlbench.environments.webarena.create_plugin", return_value=(lambda case: None, {}))):
                session = WebArenaSession({"case_id": "syntax-fixture", "config_file": str(path)},
                                          {"reset_factory": {"entrypoint": "test:reset"}})
            self.addCleanup(session.close)
            actor = make_actor(session, model="offline-fixture", max_environment_steps=2,
                               artifact_dir=directory, llm_fn=complete)
            actor()
            metadata = actor.get_trial_metadata()
            self.assertEqual(metadata["action_attempts"], 2)
            self.assertEqual(metadata["invalid_action_attempts"], 1)
            self.assertEqual(metadata["environment_steps"], 1)
            self.assertEqual(len(session.env.actions), 1)
            self.assertEqual(session.env.actions[0]["action_type"], ActionTypes.TYPE)
            self.assertEqual(metadata["invalid_action_events"][0]["raw_response"], "type [12] example text [1]")
            self.assertEqual(actor.get_policy_evidence()["steps"][0]["action"], "type [12] [example text] [1]")
            self.assertIn("No environment action was executed", prompts[1])
            self.assertIn("Keep the square brackets", prompts[1])
            self.assertIn("actions remaining: 1", prompts[1])

    def test_native_fuzzy_and_unachievable_scoring_use_accounted_unmodified_requests(self):
        from vrlbench.runtime import llm
        from vrlbench.runtime.budget import TokenUsageLedger
        from vrlbench.provider_usage import ProviderUsageJournal, read_provider_usage
        from vrlbench.worker import _ClientProxy

        # Imports need the native site configuration, but the scorer-only fixture
        # never opens these URLs, starts a browser, or invokes a real model.
        sites = {key: f"http://{key.lower()}.invalid" for key in
                 ("REDDIT", "SHOPPING", "SHOPPING_ADMIN", "GITLAB", "WIKIPEDIA", "MAP", "HOMEPAGE")}
        with patch.dict(os.environ, sites):
            from webarena.browser_env.actions import (
                ActionParsingError as NativeParsingError, ActionTypes,
                create_id_based_action, create_none_action, create_stop_action,
            )
            from webarena.evaluation_harness import helper_functions
            from webarena.evaluation_harness.evaluators import evaluator_router

        class ScorerBrowser(FakeBrowser):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.page = helper_functions.PseudoPage(self.page, self.page.url)

            def get_page_client(self, page):
                assert page is self.page
                return None

        calls = []
        replies = iter(["correct", "same", "correct"])

        def create(**kwargs):
            calls.append(kwargs)
            if len(calls) == 4:
                raise ConnectionError("fixture judge unavailable")
            observed_model = kwargs["model"] + "-2026-01-01" if len(calls) == 3 else kwargs["model"]
            return SimpleNamespace(model=observed_model,
                                   choices=[SimpleNamespace(message=SimpleNamespace(content=next(replies)))],
                                   usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15))

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        ledger = TokenUsageLedger(require_provider_usage=True)
        original = helper_functions.generate_from_openai_chat_completion
        upstream = (ScorerBrowser, create_id_based_action, create_none_action, create_stop_action,
                    ActionTypes, NativeParsingError, evaluator_router)
        with tempfile.TemporaryDirectory() as directory:
            journal = ProviderUsageJournal(Path(directory))
            proxy = _ClientProxy(lambda: client,
                                 {"temperature": 0.9, "top_p": 0.2, "max_tokens": 7, "seed": 777}, journal)
            with (patch("vrlbench.environments.webarena._upstream", return_value=upstream),
                  patch("vrlbench.environments.webarena.create_plugin", return_value=(lambda case: None, {})),
                  patch.object(llm, "get_client", return_value=proxy),
                  patch.dict(os.environ, {"VEX2_ALLOWED_RESPONSE_MODELS": "actor-only-model"}),
                  llm.token_budget_scope(ledger)):
                llm.set_context("ACTOR-SYSTEM-MEMORY")
                llm.set_user_context("ACTOR-USER-MEMORY")
                try:
                    settings = [(["SCORER-ONLY-TARGET"], None), ("N/A", None),
                                (["SCORER-ONLY-TARGET"], "fixture-judge"),
                                (["SCORER-ONLY-TARGET"], None)]
                    for index, (reference, override) in enumerate(settings):
                        path = Path(directory) / f"case-{index}.json"
                        path.write_text(json.dumps({
                            "intent": "Answer the visible task",
                            "eval": {"eval_types": ["string_match"],
                                     "reference_answers": {"fuzzy_match": reference},
                                     "string_note": "SCORER-ONLY-REASON"},
                        }), encoding="utf-8")
                        params = {"reset_factory": {"entrypoint": "test:reset"}}
                        if override is not None:
                            params["judge_model"] = override
                        session = WebArenaSession({"case_id": str(index), "config_file": str(path)}, params)
                        try:
                            self.assertEqual(session.source_identity["judge_model"], override or "gpt-4-1106-preview")
                            state = session.reset(trial_index=0)
                            self.assertNotIn("SCORER-ONLY", state.task + state.observation.text)
                            session.step("stop [student answer]")
                            if index < 3:
                                self.assertTrue(session.evaluate())
                                self.assertEqual(session.evaluation_score, 1.0)
                            else:
                                with self.assertRaisesRegex(ConnectionError, "fixture judge unavailable"):
                                    session.evaluate()
                                self.assertIsNone(session.evaluation_score)
                        finally:
                            session.close()
                        self.assertIs(helper_functions.generate_from_openai_chat_completion, original)
                finally:
                    llm.set_context("")
                    llm.set_user_context("")
            usage = read_provider_usage(Path(directory))
            self.assertEqual(usage["responses_received"], 3)
            self.assertEqual(usage["failed_requests"], 1)
            self.assertEqual(usage["known_subtotals"]["total_tokens"], 45)
        self.assertEqual(len(calls), 4)  # The failed judge call was not retried.
        self.assertEqual(ledger.snapshot()["api_calls"], 3)
        self.assertEqual(ledger.snapshot()["total_tokens"], 45)
        self.assertFalse(ledger.snapshot()["provider_usage_complete"])
        for index, call in enumerate(calls):
            self.assertEqual(set(call), {"model", "messages", "temperature", "max_tokens", "top_p", "stop"})
            self.assertEqual((call["model"], call["temperature"], call["max_tokens"], call["top_p"], call["stop"]),
                             ("fixture-judge" if index == 2 else "gpt-4-1106-preview", 0, 768, 1.0, None))
            self.assertNotIn("ACTOR-", str(call["messages"]))
        self.assertIn("reference answer: SCORER-ONLY-TARGET", calls[0]["messages"][1]["content"])
        self.assertIn("actual unachievable reason: SCORER-ONLY-REASON", calls[1]["messages"][1]["content"])
        with self.assertRaises(llm.ProviderIdentityMismatch):
            llm._record_resource_usage(SimpleNamespace(model="gpt-4.1-mini-2025-04-14"),
                                       expected_model="gpt-4.1-nano")


if __name__ == "__main__":
    unittest.main()
