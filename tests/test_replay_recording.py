"""录制只供回放，不改变动作、模型输入或任务结果。"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vrlbench import MemorySpec, ModelSpec, SimulationSpec, WorkloadSpec
from vrlbench import backends
from vrlbench.environments import miniwob, webshop


class MiniWoBPage:
    subdomain = "click-test"

    def __init__(self, capture_error=False):
        self.calls = []
        self.phase = 0
        self.capture_phases = []
        self.capture_error = capture_error
        self.instance = SimpleNamespace(driver=SimpleNamespace(
            execute_script=Mock(), get_screenshot_as_png=self.screenshot), start_time=None)

    def reset(self, *, seed):
        self.calls.append(("reset", seed))
        self.phase = 0
        return '<div id="query">Complete the task.</div>'

    def get_task(self):
        return "Complete the task."

    def step(self, action):
        self.calls.append(action)
        self.phase += 1
        done = self.phase == 2
        return f"Observed phase {self.phase}", float(done), done, {}

    def screenshot(self):
        self.capture_phases.append(self.phase)
        if self.capture_error:
            raise OSError("camera unavailable")
        return b"\x89PNG\r\n\x1a\n" + bytes([self.phase])


class WebShopPage:
    def __init__(self):
        self.browser = SimpleNamespace(page_source="")
        self.calls = []

    def reset(self, *, session):
        self.calls.append(("reset", session))
        self.browser.page_source = "<html>Instruction: Buy a sofa. Search</html>"
        return "Instruction: Buy a sofa.\nSearch", {}

    def step(self, action):
        self.calls.append(action)
        if action.startswith("search["):
            self.browser.page_source = "<html>sofa1 Buy Now</html>"
            return "Instruction: Buy a sofa.\nsofa1\nBuy Now", 0.0, False, {}
        self.browser.page_source = "<html>Done<div hidden>Target SECRET</div></html>"
        return "Purchased [SEP] Target [SEP] SECRET", 1.0, True, {}

    def get_available_actions(self):
        return {"has_search_bar": True, "clickables": ["sofa1", "buy now"]}


class ReplayRecordingTests(unittest.TestCase):
    def test_miniwob_records_each_primitive_without_changing_execution(self):
        action = lambda value: value
        keys = SimpleNamespace(LEFT="left", RIGHT="right", BACKSPACE="backspace", UP="up", DOWN="down")
        stack = (None, action, action, action, action, keys)
        response = 'Thought: Controls visible.\nAction:\n```\nagent.click_xpath("//button")\nagent.type("hello")\nagent.press("enter")\n```'
        outcomes = []
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {"MINIWOB_ACTOR_PROFILE": "react_batch"}), \
                patch.object(miniwob, "_load_mw_stack", return_value=stack):
            for recording in (False, True):
                env = MiniWoBPage()
                model = Mock(side_effect=["invalid response", response])
                actor = miniwob.make_actor(env, 7, llm_fn=model,
                    max_environment_steps=4, artifact_dir=tmp if recording else None)
                actor.set_trial_context(trial_index=0, total_trials=6)
                trajectory = actor()
                metadata = actor.get_trial_metadata()
                frames = metadata.pop("replay_observations", [])
                outcomes.append((env.calls, model.call_args_list, trajectory,
                                 actor.get_policy_evidence(), actor.get_actor_trajectory(), metadata))
                self.assertEqual(env.capture_phases, [0, 1, 2] if recording else [])
                if recording:
                    self.assertEqual([frame["observation_index"] for frame in frames], [0, 1, 2])
                    for phase, frame in enumerate(frames):
                        raw = Path(frame["path"]).read_bytes()
                        self.assertEqual(raw, b"\x89PNG\r\n\x1a\n" + bytes([phase]))
                        self.assertEqual(frame["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(outcomes[0], outcomes[1])

    def test_miniwob_missing_capture_does_not_fail_or_slow_legacy_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            for profile in ("react_batch", "synapse_lightweight"):
                with self.subTest(profile=profile), \
                        patch.dict(os.environ, {"MINIWOB_ACTOR_PROFILE": profile}), \
                        patch.object(miniwob, "_load_mw_stack", return_value=(None, str, str, str, str, None)):
                    env = MiniWoBPage(capture_error=True)
                    code = 'agent.click_xpath("//button")'
                    response = f"Thought: Click.\nAction:\n```\n{code}\n```" if profile == "react_batch" else code
                    actor = miniwob.make_actor(env, 7, llm_fn=Mock(return_value=response),
                        max_environment_steps=2, artifact_dir=Path(tmp) / profile)
                    actor.set_trial_context(trial_index=0, total_trials=6)
                    self.assertTrue(miniwob.make_manager()(actor()))
                    frames = actor.get_trial_metadata()["replay_observations"]
                    self.assertTrue(all(frame["kind"] == "unavailable" for frame in frames))
                    self.assertFalse(any("path" in frame for frame in frames))
                    self.assertEqual(env.capture_phases, [0, 1, 2] if profile == "react_batch" else [])
            self.assertFalse(list(Path(tmp).rglob("*.png")))

    def test_webshop_native_html_is_read_only_and_terminal_details_are_excluded(self):
        outcomes = []
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ("off", "html", "error", "no_hook"):
                page = WebShopPage()
                env = webshop._OfficialWebShopTextAdapter(page)
                if mode == "error":
                    env.capture_observation = Mock(side_effect=OSError("no capture"))
                elif mode == "no_hook":
                    env.capture_observation = None
                model = Mock(side_effect=["search[sofa]", "click[buy now]"])
                actor = webshop.make_actor(env, 0, llm_fn=model, max_environment_steps=2,
                    artifact_dir=None if mode == "off" else Path(tmp) / mode)
                actor.set_trial_context(trial_index=0, total_trials=6)
                trajectory = actor()
                metadata = actor.get_trial_metadata()
                frames = metadata.pop("replay_observations", [])
                outcomes.append((page.calls, model.call_args_list, trajectory,
                                 actor.get_policy_evidence(), actor.get_actor_trajectory(), metadata))
                if mode == "html":
                    self.assertEqual([frame["kind"] for frame in frames], ["html", "html", "unavailable"])
                    self.assertEqual(frames[-1]["reason"], "terminal evaluator details excluded")
                    self.assertEqual(Path(frames[1]["path"]).read_text(), "<html>sofa1 Buy Now</html>")
                elif mode == "error":
                    self.assertTrue(all(frame["kind"] == "unavailable" for frame in frames))
                else:
                    self.assertEqual(frames, [])
            self.assertTrue(all(value == outcomes[0] for value in outcomes))
            self.assertNotIn("SECRET", "".join(path.read_text() for path in Path(tmp).rglob("*.html")))

    def test_capture_option_is_strict_and_reaches_both_native_runners(self):
        for name, cases in (("webshop", [0]), ("miniwob", [{"task": "click-test", "episode": 0}])):
            for value in (False, True, "true", 1, None):
                with self.subTest(name=name, value=value):
                    spec = SimulationSpec(workload=WorkloadSpec(name, cases, params={"capture_observations": value}),
                        model=ModelSpec("offline"), memory=MemorySpec("retry"),
                        trials=6, max_environment_steps=15, require_provider_usage=False)
                    if type(value) is not bool:
                        with self.assertRaisesRegex(ValueError, "capture_observations must be a boolean"):
                            spec.validate()
                        continue
                    spec.validate()
                    module = webshop if name == "webshop" else miniwob
                    with tempfile.TemporaryDirectory() as tmp, patch.object(module, "run") as run, \
                            patch.object(backends, "_require_finished"), patch.object(backends, "_case_rows", return_value=[]):
                        wire = spec.to_dict()
                        wire["output_dir"] = tmp
                        backends.run_reference(wire, "none", "none")
                        self.assertIs(run.call_args.kwargs["capture_observations"], value)


if __name__ == "__main__":
    unittest.main()
