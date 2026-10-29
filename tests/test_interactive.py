"""Zero-API checks for the shared interactive adapter protocol."""

from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vrlbench.environments.interactive import (
    Observation, TaskState, Transition, make_actor, run_cases,
)


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRskAAAAASUVORK5CYII="
)


class FakeSession:
    system_prompt = "Choose an environment action."
    basic_rules = "Use the visible interface; a failed trial resets the same task."
    task_scope = "Complete a synthetic task."
    source_identity = {"fixture": "synthetic"}

    def __init__(self, *, outcome=False, done=False, truncated=False, image=None):
        self.outcome, self.done, self.truncated, self.image = outcome, done, truncated, image
        self.resets, self.actions, self.evaluations, self.closed = [], [], 0, False

    def reset(self, *, trial_index):
        self.resets.append(trial_index)
        return TaskState("Complete the task.", Observation("Initial state", self.image), "fixed-case")

    def step(self, action):
        self.actions.append(action)
        return Transition(Observation("Visible feedback", self.image), self.done, self.truncated)

    def evaluate(self):
        self.evaluations += 1
        return self.outcome

    def close(self):
        self.closed = True


def make_checkpoint_backend():
    """Explicit synthetic fixture for a third-trial environment failure."""
    class Backend:
        def run(self, spec, runtime):
            session = FakeSession(image=PNG)
            step = session.step

            def fail_on_third_trial(action):
                if len(session.resets) == 3:
                    raise ConnectionError("fixture third-trial environment failure")
                return step(action)

            session.step = fail_on_third_trial
            return run_cases(spec, runtime, lambda case, params: session)

    return Backend()


def make_checkpoint_transport():
    """Local response fixture; never contacts a provider."""
    from types import SimpleNamespace

    def create(**kwargs):
        return SimpleNamespace(model=kwargs["model"],
                               choices=[SimpleNamespace(message=SimpleNamespace(content="look"))],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15))

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


class InteractiveActorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.requests = []

    def actor(self, session=None, *, horizon=2, response="look", capture=False):
        self.session = session or FakeSession()

        def complete(prompt, **kwargs):
            self.requests.append((prompt, kwargs))
            return response

        return make_actor(self.session, model="offline", max_environment_steps=horizon,
                          artifact_dir=self.temp.name, capture_observations=capture,
                          llm_fn=complete)

    def test_horizon_has_one_reset_and_exact_step_count(self):
        actor = self.actor()
        trace = actor()
        metadata = actor.get_trial_metadata()
        self.assertIn("STATUS: FAIL", trace)
        self.assertEqual(self.session.resets, [0])
        self.assertEqual(self.session.actions, ["look", "look"])
        self.assertEqual(self.session.evaluations, 1)
        self.assertEqual(metadata["environment_steps"], 2)
        self.assertEqual(metadata["environment_resets"], 1)
        self.assertEqual(metadata["trial_close_reason"], "ao_horizon")
        self.assertTrue(metadata["environment_truncated"])
        self.assertEqual(len(actor.get_policy_evidence()["steps"]), 2)
        self.assertEqual(len(self.requests), 2)
        self.assertIn("Action: look", self.requests[1][0])

    def test_done_is_not_success(self):
        actor = self.actor(FakeSession(done=True, outcome=False), horizon=1)
        self.session.evaluation_score = 0.75
        actor()
        self.assertFalse(actor.success)
        metadata = actor.get_trial_metadata()
        self.assertEqual(metadata["trial_close_reason"], "environment_terminal_failure")
        self.assertEqual(metadata["environment_score"], 0.75)
        self.assertTrue(metadata["environment_terminated"])
        self.assertFalse(metadata["environment_truncated"])

    def test_verified_horizon_success_has_consistent_close_state(self):
        actor = self.actor(FakeSession(outcome=True), horizon=1)
        actor()
        self.assertTrue(actor.success)
        metadata = actor.get_trial_metadata()
        self.assertEqual(metadata["trial_close_reason"], "environment_success")
        self.assertTrue(metadata["environment_terminated"])
        self.assertFalse(metadata["environment_truncated"])

    def test_native_truncation_preserved(self):
        actor = self.actor(FakeSession(truncated=True))
        actor()
        self.assertEqual(actor.get_trial_metadata()["trial_close_reason"], "environment_truncated")
        self.assertEqual(self.session.actions, ["look"])

    def test_preparation_is_idempotent_and_trial_resets_same_case(self):
        actor = self.actor(FakeSession(done=True))
        actor.prepare_trial()
        actor.prepare_trial()
        actor()
        self.assertEqual(self.session.resets, [0])
        actor.set_trial_context(trial_index=1, total_trials=6)
        actor()
        self.assertEqual(self.session.resets, [0, 1])
        self.assertEqual(actor.get_trial_metadata()["trial_index"], 1)
        self.assertEqual(actor.get_trial_metadata()["environment_steps"], 1)

    def test_changed_case_identity_is_infrastructure_failure(self):
        actor = self.actor(FakeSession(done=True))
        actor()
        actor.set_trial_context(trial_index=1, total_trials=2)
        self.session.reset = lambda **kwargs: TaskState("Complete the task.", Observation("x"), "different-case")
        with self.assertRaisesRegex(ValueError, "changed the task identity"):
            actor()
        self.assertEqual(len(self.session.actions), 1)

    def test_empty_output_spends_budget_without_inventing_environment_steps(self):
        actor = self.actor(response=" \n")
        actor()
        metadata = actor.get_trial_metadata()
        self.assertFalse(metadata["protocol_failure"])
        self.assertFalse(metadata["policy_evidence_eligible"])
        self.assertEqual(metadata["environment_steps"], 0)
        self.assertEqual(metadata["actor_turns"], 2)
        self.assertEqual(metadata["action_attempts"], 2)
        self.assertEqual(metadata["invalid_action_attempts"], 2)
        self.assertEqual(metadata["remaining_environment_steps"], 0)
        self.assertEqual(metadata["trial_close_reason"], "ao_horizon")
        self.assertEqual(self.session.actions, [])
        self.assertEqual(self.session.evaluations, 1)
        self.assertEqual(actor.get_policy_evidence()["steps"], [])
        self.assertEqual(metadata["invalid_action_events"][1]["budget_used_after"], 2)

    def test_empty_reply_then_valid_action_uses_remaining_budget_without_reset(self):
        actor = self.actor()
        replies = iter(["", "look"])

        def complete(prompt, **kwargs):
            self.requests.append((prompt, kwargs))
            return next(replies)

        actor.llm_fn = complete
        actor()
        metadata = actor.get_trial_metadata()
        self.assertEqual(self.session.resets, [0])
        self.assertEqual(self.session.actions, ["look"])
        self.assertEqual(metadata["action_attempts"], 2)
        self.assertEqual(metadata["environment_steps"], 1)
        self.assertIn("observation is unchanged", self.requests[1][0])
        self.assertIn("actions remaining: 1", self.requests[1][0])
        self.assertIs(self.requests[0][1]["retry_on_error"], False)

    def test_environment_error_is_not_scored_as_task_failure(self):
        actor = self.actor()
        self.session.step = lambda action: (_ for _ in ()).throw(ConnectionError("native environment unavailable"))
        with self.assertRaises(ConnectionError):
            actor()
        self.assertEqual(self.session.evaluations, 0)

    def test_nonboolean_evaluator_rejected(self):
        actor = self.actor(FakeSession(done=True, outcome=0.5))
        with self.assertRaisesRegex(TypeError, "boolean verified"):
            actor()

    def test_screenshot_passed_to_model_and_saved_without_base64_in_trace(self):
        actor = self.actor(FakeSession(done=True, image=PNG))
        trace = actor()
        self.assertEqual(self.requests[0][1]["images"], [PNG])
        screenshots = actor.get_trial_metadata()["screenshots"]
        self.assertEqual(len(screenshots), 2)
        self.assertEqual(Path(screenshots[0]["path"]).read_bytes(), PNG)
        self.assertIn("Screenshot:", trace)
        self.assertNotIn("iVBOR", trace)
        self.assertNotIn("iVBOR", actor.get_actor_trajectory())

    def test_optional_text_actor_capture_does_not_change_model_input(self):
        session = FakeSession(done=True)
        session.capture_observation = lambda: PNG
        actor = self.actor(session, capture=True)
        trace = actor()
        captures = actor.get_trial_metadata()["replay_observations"]
        self.assertEqual(len(captures), 2)
        self.assertEqual(Path(captures[0]["path"]).read_bytes(), PNG)
        self.assertNotIn("images", self.requests[0][1])
        self.assertNotIn("Screenshot:", trace)
        self.assertNotIn("Screenshot:", self.requests[0][0])

    def test_policy_evidence_is_defensive_copy(self):
        actor = self.actor(FakeSession(done=True))
        actor()
        evidence = actor.get_policy_evidence()
        evidence["steps"][0]["observation"] = "modified"
        self.assertEqual(actor.get_policy_evidence()["steps"][0]["observation"], "Visible feedback")


@unittest.skipUnless(importlib.util.find_spec("openai") is not None, "requires runtime extra")
class InteractiveRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.spec = {"benchmark": {"name": "interactive-fixture", "params": {},
                                   "cases": [{"case_id": "a"}, {"case_id": "b"}]},
                     "max_environment_steps": 1, "output_dir": self.temp.name,
                     "model": {"name": "offline"}, "trials": 2,
                     "require_provider_usage": False}
        from vrlbench.backends import BackendRuntime
        self.runtime = BackendRuntime(self.spec, "none", "none")

    def test_actual_trial_loop_stop_reset_and_checkpoint(self):
        from vrlbench.runtime import llm
        sessions, params_seen = [], []

        def factory(case, params):
            params_seen.append(params)
            session = FakeSession(done=True, outcome=case["case_id"] == "b")
            sessions.append(session)
            return session

        with patch.object(llm, "complete", return_value="look"):
            rows = run_cases(self.spec, self.runtime, factory)
        self.assertEqual([len(row["trials"]) for row in rows], [2, 1])
        self.assertEqual(sessions[0].resets, [0, 1])
        self.assertEqual(sessions[1].resets, [0])
        self.assertTrue(all(session.closed for session in sessions))
        self.assertTrue(all(value["max_environment_steps"] == 1 for value in params_seen))
        self.assertEqual(rows[0]["task"], "interactive-fixture")
        self.assertEqual(json.loads((Path(self.temp.name) / "reference/episodes.json").read_text()), rows)
        self.assertIn("policy_ao_trace", rows[0]["trials"][0]["policy_trajectory"])

    def test_empty_replies_exhaust_each_trial_without_failing_the_runtime(self):
        from vrlbench.runtime import llm

        self.spec["benchmark"] = {"name": "osworld", "params": {},
                                  "cases": [{"case_id": "a"}]}
        session = FakeSession()
        with patch.object(llm, "complete", return_value=""):
            rows = run_cases(self.spec, self.runtime, lambda case, params: session)
        self.assertEqual(session.resets, [0, 1])
        self.assertEqual(session.actions, [])
        self.assertEqual([trial["success"] for trial in rows[0]["trials"]], [False, False])
        for trial in rows[0]["trials"]:
            metadata = trial["trial_metadata"]
            self.assertEqual(metadata["trial_close_reason"], "ao_horizon")
            self.assertEqual(metadata["invalid_action_attempts"], 1)
            self.assertFalse(metadata["policy_evidence_eligible"])

    def test_close_on_native_error_without_false_completed_case(self):
        from vrlbench.runtime import llm
        session = FakeSession()
        session.step = lambda action: (_ for _ in ()).throw(ConnectionError("native failure"))
        with patch.object(llm, "complete", return_value="look"), self.assertRaises(ConnectionError):
            run_cases(self.spec, self.runtime, lambda case, params: session)
        self.assertTrue(session.closed)
        self.assertFalse((Path(self.temp.name) / "reference/episodes.json").exists())

    def test_shared_source_rejected_before_environment_creation(self):
        self.spec["shared_t1_dir"] = "/unused"
        with self.assertRaisesRegex(ValueError, "independent first trials"):
            run_cases(self.spec, self.runtime,
                      lambda case, params: self.fail("must not initialize environment"))

    def test_duplicate_case_ids_rejected_before_environment_creation(self):
        self.spec["benchmark"]["cases"] = [{"case_id": "a"}, {"case_id": "a"}]
        with self.assertRaisesRegex(ValueError, "duplicate case_id"):
            run_cases(self.spec, self.runtime,
                      lambda case, params: self.fail("must not initialize environment"))

    def test_completed_checkpoint_survives_later_case_infrastructure_failure(self):
        from vrlbench.runtime import llm
        sessions = []

        def factory(case, params):
            session = FakeSession(done=True, outcome=True)
            if case["case_id"] == "b":
                session.step = lambda action: (_ for _ in ()).throw(ConnectionError("native failure"))
            sessions.append(session)
            return session

        with patch.object(llm, "complete", return_value="look"), self.assertRaises(ConnectionError):
            run_cases(self.spec, self.runtime, factory)
        rows = json.loads((Path(self.temp.name) / "reference/episodes.json").read_text())
        self.assertEqual([row["case_id"] for row in rows], ["a"])
        self.assertTrue(all(session.closed for session in sessions))

    def test_interrupted_case_retains_completed_trials_and_delivered_gate_context(self):
        from vrlbench import FactorySpec, MemorySpec, ModelSpec, ReplayGateSpec, SimulationSpec, WorkloadSpec, simulate
        from vrlbench.inspection import inspect_case
        from vrlbench.replay_viewer import replay_data

        source = Path(__file__).resolve()
        output = Path(self.temp.name) / "partial-run"
        spec = SimulationSpec(
            workload=WorkloadSpec("webarena", [{"case_id": "partial"}],
                                  factory=FactorySpec(f"{source}:make_checkpoint_backend")),
            model=ModelSpec("fixture", transport=FactorySpec(f"{source}:make_checkpoint_transport")),
            memory=MemorySpec("trace", factory=FactorySpec("vrlbench.backends:create_demo_memory")),
            trials=4, max_environment_steps=1, output_dir=str(output),
            replay_gate=ReplayGateSpec(policy="epsilon", epsilon=1, seed=0),
        )
        result = simulate(spec)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.counts["completed"], 0)
        self.assertEqual(result.to_dict()["incomplete_case_ids"], ["partial"])
        self.assertIsNone(result.metrics["sr_at_t"])
        with self.assertRaises(RuntimeError):
            result.require_complete()
        checkpoint = json.loads((output / "reference/episodes.json").read_text())
        self.assertEqual([trial["trial"] for trial in checkpoint[0]["trials"]], [0, 1])
        self.assertEqual(checkpoint[0]["resource_usage"]["api_calls"], 2)
        self.assertEqual(result.resources["api_calls"], 3)  # Includes the interrupted trial's model call.
        view = inspect_case(result, "partial")["case"]
        self.assertEqual([trial["trial"] for trial in view["trials"]], [0, 1])
        second = view["trials"][1]
        self.assertEqual(second["context"], "")
        self.assertEqual(second["system_context"], "")
        self.assertIn("look", second["trial_metadata"]["replay_gate"]["prepared_system_context"])
        replay = replay_data([output])["runs"][0]
        self.assertEqual(replay["status"], "failed")
        trials = replay["cases"][0]["trials"]
        self.assertEqual(len(trials), 2)
        self.assertEqual(trials[1]["memory"]["system_context"], "")
        self.assertIn("look", trials[1]["metadata"]["replay_gate"]["prepared_system_context"])
        self.assertEqual(trials[1]["frames"][0]["action"], "look")
        self.assertIsNotNone(trials[1]["initial_image"])

    def test_memory_reaches_the_actual_next_trial_model_call(self):
        from vrlbench.runtime import llm, loop
        from vrlbench.runtime.er.base import ERSlot

        class TraceMemory(ERSlot):
            def reset(self):
                self.trace = ""

            def on_failure(self, trajectory, model):
                self.trace = trajectory

            def build_section(self):
                return "Retained failed attempt: " + self.trace if self.trace else ""

        class Runtime:
            model = "offline"

            def run_episode(inner_self, actor, manager, *, metadata, on_trial=None):
                return loop.run_episode(actor, manager, TraceMemory(), "offline",
                                        total_trials=2, metadata=metadata, on_trial=on_trial)

        session = FakeSession()
        session.evaluate = lambda: session.actions[-1] == "finish"
        contexts = []

        def complete(prompt, **kwargs):
            contexts.append(llm.get_context())
            return "finish" if "Retained failed attempt" in contexts[-1] else "look"

        self.spec["benchmark"]["cases"] = [{"case_id": "a"}]
        with patch.object(llm, "complete", side_effect=complete):
            rows = run_cases(self.spec, Runtime(), lambda case, params: session)
        self.assertEqual(session.actions, ["look", "finish"])
        self.assertEqual(session.resets, [0, 1])
        self.assertEqual(contexts[0], "")
        self.assertIn('"action":"look"', contexts[1])
        self.assertEqual([trial["success"] for trial in rows[0]["trials"]], [False, True])
        self.assertEqual(rows[0]["trials"][1]["system_context"], contexts[1])


    def test_screenshot_transport_retains_memory_and_records_provider_usage(self):
        from vrlbench.runtime import llm
        from vrlbench.runtime.budget import TokenUsageLedger

        png = b"\x89PNG\r\n\x1a\nfixture"
        client = make_checkpoint_transport()
        create = client.chat.completions.create
        calls = []
        def capture(**kwargs):
            calls.append(kwargs)
            return create(**kwargs)
        client.chat.completions.create = capture
        ledger = TokenUsageLedger(require_provider_usage=True)
        with patch.object(llm, "get_client", return_value=client):
            llm.set_context("retained reflection")
            llm.set_user_context("user memory")
            try:
                with llm.token_budget_scope(ledger):
                    llm.complete("current state", model="fixture", images=[png])
            finally:
                llm.set_context("")
                llm.set_user_context("")
        messages = calls[0]["messages"]
        self.assertEqual(messages[0]["content"], "retained reflection")
        self.assertEqual(messages[1]["content"][0]["text"], "user memory\n\ncurrent state")
        self.assertTrue(messages[1]["content"][1]["image_url"]["url"].endswith(base64.b64encode(png).decode()))
        self.assertEqual(ledger.snapshot()["api_calls"], 1)
        self.assertEqual(ledger.snapshot()["total_tokens"], 15)
        with patch.object(llm, "get_client", return_value=client):
            with llm.token_budget_scope(TokenUsageLedger(max_total_tokens=100)):
                with self.assertRaisesRegex(ValueError, "image requests"):
                    llm.complete("current state", model="fixture", images=[png])
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
