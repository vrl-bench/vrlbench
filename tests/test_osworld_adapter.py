"""Offline checks of the native DesktopEnv protocol; no VM or provider calls."""

import base64
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from vrlbench.environments.osworld import OSWorldSession, _action
from vrlbench.environments.interactive import make_actor


PNG_3_BY_2 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAMAAAACCAIAAAASFvFNAAAAC0lEQVR4nGNgwAQAABQAAX3+Hu4AAAAASUVORK5CYII="
)
PNG_2_BY_3 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAADCAIAAAA2iEnWAAAAC0lEQVR4nGNgwAIAABUAAapll8QAAAAASUVORK5CYII="
)


ACCESSIBILITY_XML = """<desktop-frame
    xmlns:st="https://accessibility.ubuntu.example.org/ns/state"
    xmlns:cp="https://accessibility.ubuntu.example.org/ns/component"
    xmlns:attr="https://accessibility.ubuntu.example.org/ns/attributes">
  <application name="Fixture">
    <push-button name="Save" st:showing="true" st:visible="true" st:enabled="true"
        cp:screencoord="(12, 34)" cp:size="(56, 20)"
        attr:toolkit="redundant-toolkit" attr:description="Save document">Save &amp; close</push-button>
    <combo-box name="Choice" st:showing="true" st:visible="true" st:enabled="true"
        st:selected="true" cp:screencoord="(70, 34)" cp:size="(60, 20)">Option A</combo-box>
    <menu-item name="Offscreen item" st:visible="true" st:enabled="true"
        cp:screencoord="(2, 3)" cp:size="(4, 5)">Offscreen text</menu-item>
    <label name="Disabled label" st:showing="true" st:visible="true"
        cp:screencoord="(2, 3)" cp:size="(4, 5)">Disabled text</label>
    <push-button name="Negative position" st:showing="true" st:visible="true" st:enabled="true"
        cp:screencoord="(-1, 3)" cp:size="(4, 5)" />
    <push-button name="Empty size" st:showing="true" st:visible="true" st:enabled="true"
        cp:screencoord="(2, 3)" cp:size="(0, 5)" />
    <panel name="Structural container" st:showing="true" st:visible="true" st:enabled="true"
        cp:screencoord="(0, 0)" cp:size="(100, 100)" />
  </application>
</desktop-frame>"""


class FakeDesktopEnv:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.resets = []
        self.actions = []
        self.score = 0.0
        self.evaluations = 0
        self.closed = False
        self.screenshot = PNG_3_BY_2
        self.accessibility_tree = ACCESSIBILITY_XML

    def observation(self):
        return {"screenshot": self.screenshot,
                "accessibility_tree": self.accessibility_tree, "terminal": None,
                "instruction": "Task instruction", "hidden": "not actor-visible"}

    def reset(self, *, task_config):
        self.resets.append(deepcopy(task_config))
        task_config["instruction"] = "upstream mutable input"
        return self.observation()

    def step(self, action, *, pause):
        self.actions.append((action, pause))
        return self.observation(), 1.0, action in {"DONE", "FAIL"}, {"evaluator": "hidden"}

    def evaluate(self):
        self.evaluations += 1
        return self.score

    def close(self):
        self.closed = True


class OSWorldAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "task.json"
        self.config = {"id": "task-id", "instruction": "Save the document.",
                       "config": [], "evaluator": {"func": "synthetic", "expected": "SECRET"}}
        self.path.write_text(json.dumps(self.config))
        self.case = {"case_id": "task-id", "config_file": str(self.path)}
        self.params = {"env_kwargs": {"path_to_vm": "/prepared/task.vmx",
                                      "snapshot_name": "clean"}, "pause": 0}
        self.env = None

    def factory(self, **kwargs):
        self.env = FakeDesktopEnv(**kwargs)
        return self.env

    def session(self):
        session = OSWorldSession(self.case, self.params, env_factory=self.factory)
        self.addCleanup(session.close)
        return session

    def test_fixed_config_and_identity_across_trials(self):
        self.params["env_kwargs"]["screen_size"] = (1920, 1080)
        session = self.session()
        first = session.reset(trial_index=0)
        second = session.reset(trial_index=1)
        self.assertEqual(self.env.resets, [self.config, self.config])
        self.assertEqual(first.identity, second.identity)
        self.assertEqual(first.task, "Save the document.")
        self.assertEqual(self.env.kwargs["action_space"], "pyautogui")
        self.assertNotIn("SECRET", first.observation.text)
        self.assertNotIn("hidden", first.observation.text)
        self.assertIn("3 x 2 pixels", first.observation.text)
        self.assertIn("0 <= x < 3; 0 <= y < 2", first.observation.text)
        self.assertIs(first.observation.image, PNG_3_BY_2)

    def test_accessibility_xml_uses_compact_upstream_visible_nodes(self):
        session = self.session()
        observation = session.reset(trial_index=0).observation
        self.assertIn("role\tname\ttext\tstates\tbounds (x, y, w, h)", observation.text)
        self.assertIn('push-button\t"Save"\t"Save & close"', observation.text)
        self.assertIn("(12, 34, 56, 20)", observation.text)
        self.assertIn('"Save document"', observation.text)
        self.assertIn("selected=true", observation.text)
        for excluded in ("Offscreen item", "Disabled label", "Negative position",
                         "Empty size", "Structural container", "redundant-toolkit",
                         "xmlns:", "https://accessibility", "<push-button"):
            self.assertNotIn(excluded, observation.text)
        self.assertIs(observation.image, PNG_3_BY_2)
        with patch.object(self.env, "observation", return_value={
                "screenshot": PNG_3_BY_2, "accessibility_tree": "<desktop-frame>"}):
            with self.assertRaisesRegex(RuntimeError, "malformed accessibility XML"):
                session.reset(trial_index=0)

    def test_configuration_and_comparison_preserve_task_and_snapshot(self):
        from dataclasses import replace
        from vrlbench import ModelSpec, SimulationSpec, WorkloadSpec, plan
        from vrlbench.results import aggregate_cases, compare_results

        spec = SimulationSpec(WorkloadSpec("osworld", [self.case], self.params), ModelSpec("fixture"))
        self.assertEqual(plan(spec)["model_calls_executed"], 0)
        relative = replace(spec, workload=replace(spec.workload, cases=[{**self.case, "config_file": "task.json"}],
                                                  params={"env_kwargs": {"path_to_vm": "vm/image.vmx"}}))
        resolved = relative.resolved(Path(self.temporary.name))
        self.assertEqual(resolved.workload.cases[0]["config_file"], str(self.path))
        self.assertEqual(resolved.workload.params["env_kwargs"]["path_to_vm"], str(Path(self.temporary.name) / "vm/image.vmx"))
        with self.assertRaisesRegex(ValueError, "provisioned VM"):
            replace(spec, workload=replace(spec.workload, params={})).validate()
        with self.assertRaisesRegex(ValueError, "screenshot"):
            replace(spec, max_total_tokens=100).validate()
        with self.assertRaisesRegex(ValueError, "shared first"):
            replace(spec, shared_t1_dir=self.temporary.name).validate()
        with self.assertRaisesRegex(ValueError, "snapshot"):
            replace(spec, workload=replace(spec.workload, params={"env_kwargs": {**self.params["env_kwargs"], "snapshot_name": ""}})).validate()
        case = {"case_id": self.case["case_id"], "task": "osworld",
                "trials": [{"trial": 0, "success": True, "trajectory": "done"}],
                "source_identity": self.session().source_identity}
        payload = {"status": "complete", "config": spec.to_dict(), **aggregate_cases([case], 6), "cases": [case]}
        self.assertEqual(compare_results([payload, payload])["environment_identity_check"]["status"], "verified")
        missing = deepcopy(payload)
        missing["cases"][0]["source_identity"]["snapshot_name"] = None
        self.assertEqual(compare_results([missing, missing])["environment_identity_check"]["status"], "unverified")
        changed = deepcopy(payload)
        changed["cases"][0]["source_identity"]["config_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "task/environment identities"):
            compare_results([payload, changed])

    def test_step_is_native_vm_dispatch_without_reward_success(self):
        session = self.session()
        initial = session.reset(trial_index=0)
        self.env.screenshot = PNG_2_BY_3
        transition = session.step("```python\nimport pyautogui\npyautogui.click(20, 30)\n```")
        self.assertIn("3 x 2 pixels", initial.observation.text)
        self.assertIn("2 x 3 pixels", transition.observation.text)
        self.assertIn("0 <= x < 2; 0 <= y < 3", transition.observation.text)
        self.assertIs(transition.observation.image, PNG_2_BY_3)
        self.assertEqual(self.env.actions, [("import pyautogui\npyautogui.click(20, 30)", 0.0)])
        self.assertFalse(transition.terminated)
        self.assertFalse(session.evaluate())

    def test_done_is_not_success_and_evaluation_cached(self):
        session = self.session()
        session.reset(trial_index=0)
        self.assertTrue(session.step("DONE").terminated)
        self.assertFalse(session.evaluate())
        self.assertFalse(session.evaluate())
        self.assertEqual(self.env.evaluations, 1)
        session.reset(trial_index=1)
        self.assertIsNone(session.evaluation_score)
        self.env.score = 0.75
        self.assertFalse(session.evaluate())
        self.assertEqual(session.evaluation_score, 0.75)
        self.assertEqual(self.env.evaluations, 2)

    def test_infeasible_fail_can_pass_upstream_evaluator(self):
        session = self.session()
        session.reset(trial_index=0)
        self.assertTrue(session.step("FAIL").terminated)
        self.env.score = 1
        self.assertTrue(session.evaluate())

    def test_upstream_errors_propagate_and_cleanup_is_idempotent(self):
        session = self.session()
        session.reset(trial_index=0)
        with patch.object(self.env, "evaluate", side_effect=RuntimeError("VM disconnected")):
            with self.assertRaisesRegex(RuntimeError, "VM disconnected"):
                session.evaluate()
        session.close()
        session.close()
        self.assertTrue(self.env.closed)
        with self.assertRaises(RuntimeError):
            session.reset(trial_index=1)

    def test_missing_or_invalid_screenshot_fails_instead_of_text_only_execution(self):
        session = self.session()
        with patch.object(self.env, "observation", return_value={"screenshot": None}):
            with self.assertRaisesRegex(RuntimeError, "no screenshot"):
                session.reset(trial_index=0)
        for image in (b"\x89PNG\r\n\x1a\nsynthetic-fixture", b"\xff\xd8\xffnot-native-PNG"):
            with self.subTest(image=image), patch.object(
                    self.env, "observation", return_value={"screenshot": image}):
                with self.assertRaisesRegex(RuntimeError, "native PNG"):
                    session.reset(trial_index=0)
        for width, height in ((0, 2), (3, 0), (2**31, 2), (3, 2**31)):
            image = PNG_3_BY_2[:16] + width.to_bytes(4, "big") + height.to_bytes(4, "big") + PNG_3_BY_2[24:]
            with self.subTest(width=width, height=height), patch.object(
                    self.env, "observation", return_value={"screenshot": image}):
                with self.assertRaisesRegex(RuntimeError, "dimensions"):
                    session.reset(trial_index=0)

    def test_file_read_failures_cannot_be_swallowed_into_a_zero_score(self):
        session = self.session()
        for error in (ConnectionError("VM unavailable"), RuntimeError("HTTP 503")):
            session.reset(trial_index=0)
            def get_file(path, *, strict=False):
                self.assertTrue(strict)
                raise error
            self.env.controller = SimpleNamespace(get_file=get_file)
            def native_evaluate():
                try:
                    self.env.controller.get_file("/task/output")
                except Exception:
                    return 0.0  # Native getters hide this exception.
            self.env.evaluate = native_evaluate
            with self.subTest(error=error), self.assertRaisesRegex(RuntimeError, "file retrieval"):
                session.evaluate()
            self.assertIs(self.env.controller.get_file, get_file)

    def test_missing_or_incorrect_task_file_remains_a_normal_task_failure(self):
        session = self.session()
        for content, expected in ((None, False), (b"wrong output", False), (b"correct output", True)):
            session.reset(trial_index=0)
            def get_file(path, *, strict=False):
                self.assertTrue(strict)
                return content  # Native strict mode returns None only for 404.
            self.env.controller = SimpleNamespace(get_file=get_file)
            self.env.evaluate = lambda: float(self.env.controller.get_file("/task/output") == b"correct output")
            with self.subTest(content=content):
                self.assertEqual(session.evaluate(), expected)
            self.assertIs(self.env.controller.get_file, get_file)

    def test_rejects_implicit_vm_creation_and_wrong_action_space(self):
        for kwargs in ({}, {"path_to_vm": ""},
                       {"path_to_vm": "/prepared/vm", "action_space": "computer_13"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                OSWorldSession(self.case, {"env_kwargs": kwargs}, env_factory=self.factory)
        self.assertIsNone(self.env)

    def test_lazy_native_import_and_factory(self):
        package = ModuleType("desktop_env")
        module = ModuleType("desktop_env.desktop_env")
        controllers = ModuleType("desktop_env.controllers")
        python_controller = ModuleType("desktop_env.controllers.python")
        class LegacyController:
            def get_file(self, file_path):
                return None
        class CurrentController:
            def get_file(self, file_path, *, strict=False):
                return None
        python_controller.PythonController = LegacyController
        module.DesktopEnv = self.factory
        with patch.dict(sys.modules, {"desktop_env": package, "desktop_env.desktop_env": module,
                                      "desktop_env.controllers": controllers,
                                      "desktop_env.controllers.python": python_controller}):
            with self.assertRaisesRegex(RuntimeError, "Unsupported OSWorld file API"):
                OSWorldSession(self.case, self.params)
            self.assertIsNone(self.env)  # Reject before starting the VM or a model.
            python_controller.PythonController = CurrentController
            session = OSWorldSession(self.case, self.params)
        self.addCleanup(session.close)
        self.assertEqual(self.env.kwargs["path_to_vm"], "/prepared/task.vmx")

    def test_invalid_score_is_not_a_task_failure(self):
        session = self.session()
        session.reset(trial_index=0)
        for score in (True, float("nan"), -1, 2, None):
            self.env.score = score
            with self.subTest(score=score), self.assertRaises(ValueError):
                session.evaluate()

    def test_action_wrapper_is_syntax_only(self):
        self.assertEqual(_action(" DONE "), "DONE")
        code = "```python\npyautogui.click(1, 2)\n```\n```python\nprint('extra')\n```"
        self.assertEqual(_action(code), code)
        session = self.session()
        program = "import pyautogui\npyautogui.click(1, 2)"
        for raw, expected in ((program, program),
                              (f"```python\n{program}\n```", program),
                              ("print(' DONE ')", "print(' DONE ')"),
                              ("message = '```DONE```'", "message = '```DONE```'"),
                              ("DONE", "DONE"), ("FAIL", "FAIL"), ("WAIT", "WAIT")):
            with self.subTest(raw=raw):
                self.assertEqual(session.parse_response(raw), expected)
        with self.assertRaisesRegex(ValueError, "not mixed with Python"):
            session.parse_response(program + "\nDONE")
        for invalid in ("return", "break", "continue"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "Python syntax error"):
                session.parse_response(program + "\n" + invalid)
        for raw in (code, f"```python\n{program}",
                    f"I will click.\n```python\n{program}\n```"):
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, "fully closed Python code block"):
                session.parse_response(raw)
        self.assertEqual(self.env.actions, [])

    def test_shared_actor_receives_image_and_evaluates_at_horizon(self):
        session = self.session()
        calls = []
        program = "import pyautogui\npyautogui.click(1, 2)"
        responses = iter([program + "\nDONE", f"```python\n{program}\n```"])

        def model(prompt, **kwargs):
            calls.append((prompt, kwargs))
            return next(responses)

        actor = make_actor(session, model="offline-fixture", max_environment_steps=2,
                           artifact_dir=Path(self.temporary.name) / "frames", llm_fn=model)
        actor()
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1]["images"], [self.env.observation()["screenshot"]])
        self.assertIs(calls[0][1]["images"][0], PNG_3_BY_2)
        self.assertIn("3 x 2 pixels", calls[0][0])
        self.assertIn('push-button\t"Save"\t"Save & close"', calls[0][0])
        self.assertNotIn("redundant-toolkit", calls[0][0])
        self.assertIn('push-button\t"Save"\t"Save & close"', actor.get_policy_evidence()["initial_observation"])
        self.assertFalse(actor.success)
        self.assertEqual(self.env.evaluations, 1)
        metadata = actor.get_trial_metadata()
        self.assertEqual(metadata["action_attempts"], 2)
        self.assertEqual(metadata["invalid_action_attempts"], 1)
        self.assertEqual(metadata["environment_steps"], 1)
        self.assertEqual(self.env.actions[0][0], program)
        self.assertEqual(len(self.env.actions), 1)
        self.assertIn("not mixed with Python", calls[1][0])
        self.assertIn("actions remaining: 1", calls[1][0])
        self.assertEqual(metadata["invalid_action_events"][0]["raw_response"], program + "\nDONE")
        self.assertEqual(actor.get_policy_evidence()["steps"][0]["action"], program)
        self.assertEqual(metadata["trial_close_reason"], "ao_horizon")
        self.assertEqual(len(metadata["screenshots"]), 2)
        self.assertNotIn("SECRET", calls[0][0])


if __name__ == "__main__":
    unittest.main()
