"""Offline native-runner checks after retiring the experimental controller."""

from __future__ import annotations

from contextlib import ExitStack
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@unittest.skipUnless(importlib.util.find_spec("openai"), "requires the runtime extra")
class NativeControllerSurface(unittest.TestCase):
    def setUp(self):
        from vrlbench.environments import miniwob, webshop
        from vrlbench.runtime import llm

        self.miniwob, self.webshop = miniwob, webshop
        self.no_api = patch.object(llm, "complete", side_effect=AssertionError("unexpected model call"))
        self.no_api.start()
        self.addCleanup(self.no_api.stop)

    def test_retired_controller_and_spatial_parameter_are_not_accepted(self):
        for normalize in (self.miniwob._normalize_miniwob_controller,
                          self.webshop._normalize_webshop_controller):
            for name in ("none", "dsst"):
                self.assertEqual(normalize(name.upper()), name)
            with self.assertRaisesRegex(ValueError, "none or dsst"):
                normalize("lightweight_vex")
        self.assertNotIn("spatial_mode", inspect.signature(self.miniwob.run).parameters)

    def test_lightweight_actor_profile_remains_available(self):
        with patch.dict(os.environ, {"MINIWOB_ACTOR_PROFILE": "synapse_lightweight"}):
            self.assertEqual(self.miniwob._actor_profile(), "synapse_lightweight")
        self.assertEqual(self.miniwob._system_instruction_for_profile("synapse_lightweight"),
                         self.miniwob.SYNAPSE_LIGHTWEIGHT_SYSTEM_INSTRUCTION)
        self.assertEqual(self.miniwob._format_repair_prompt_for_profile("synapse_lightweight"),
                         self.miniwob.SYNAPSE_LIGHTWEIGHT_FORMAT_REPAIR_PROMPT_TEMPLATE)

    def run_native(self, benchmark, output, *, er="none", controller="none",
                   shared=None, success_after=1):
        from vrlbench.runtime.budget import AO_BUDGET_UNIT

        class Environment:
            def __init__(self, **kwargs):
                pass

            def close(self):
                pass

        class Actor:
            def set_trial_context(self, *, trial_index, total_trials):
                self.index = trial_index

            def __call__(self):
                return self.get_actor_trajectory()

            def get_actor_trajectory(self):
                outcome = "SUCCESS" if self.index + 1 >= success_after else "FAIL"
                return f"Instruction: synthetic native task\nAction: synthetic action\n{outcome}"

            def get_policy_evidence(self):
                return {"schema_version": 1, "task": "synthetic native task",
                        "initial_observation": "synthetic reset",
                        "steps": [{"action": "synthetic action", "observation": "visible result"}],
                        "trial_close_reason": self.get_trial_metadata()["trial_close_reason"]}

            def get_trial_metadata(self):
                return {"trial_index": self.index, "environment_resets": 1,
                        "max_environment_resets": 1, "environment_steps": 1,
                        "max_environment_steps": 4, "budget_unit": AO_BUDGET_UNIT,
                        "environment_terminated": True, "environment_truncated": False,
                        "protocol_failure": False, "policy_evidence_eligible": True,
                        "trial_close_reason": "environment_success" if self.index + 1 >= success_after
                        else "environment_terminal_failure",
                        "base_seed": 0, "reset_seed": 0, "trial_seed_mode": "fixed_instance",
                        "actor_profile": "synapse_lightweight", "sample_index": 0,
                        "task": "synthetic native task"}

        module = getattr(self, benchmark)
        with ExitStack() as stack:
            stack.enter_context(patch.object(module, "make_actor", side_effect=lambda *args, **kwargs: Actor()))
            if benchmark == "miniwob":
                stack.enter_context(patch.object(module, "_load_mw_stack", return_value=(Environment,)))
                stack.enter_context(patch.dict(os.environ, {"MINIWOB_ACTOR_PROFILE": "synapse_lightweight"}))
                selection = {"tasks": ["click-test"], "episode_indices": [0]}
            else:
                selection = {"sample_indices": [0], "env_factory": Environment}
            module.run(**selection, er=er, controller=controller, model="offline",
                       trials=2, max_environment_steps=4, output_dir=str(output),
                       shared_t1_dir=str(shared) if shared else None)
        return (json.loads((output / "results.json").read_text()),
                json.loads((output / "episodes.json").read_text())[0])

    def test_current_methods_keep_fresh_and_shared_success_stop(self):
        methods = (("none", "none"), ("reflexion", "none"),
                   ("dynamic_cheatsheet_adapted", "none"), ("ace_adapted", "none"),
                   ("none", "dsst"))
        for benchmark in ("miniwob", "webshop"):
            with self.subTest(benchmark=benchmark), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, source = self.run_native(benchmark, root / "source")
                for index, (er, controller) in enumerate(methods):
                    for shared in (None, root / "source"):
                        with self.subTest(er=er, controller=controller, shared=shared is not None):
                            summary, row = self.run_native(
                                benchmark, root / f"method-{index}-{shared is not None}",
                                er=er, controller=controller, shared=shared,
                            )
                            self.assertEqual(summary["sr_at_t"], 1)
                            self.assertFalse(summary["checkpoint"])
                            self.assertEqual(len(row["trials"]), 1)
                            for field in ("trajectory", "policy_trajectory", "actor_trajectory"):
                                self.assertEqual(row["trials"][0][field], source["trials"][0][field])
                            if benchmark == "miniwob":
                                self.assertEqual(summary["actor_profile"], "synapse_lightweight")
                                self.assertNotIn("spatial_mode", summary)

    def test_retry_shared_failure_continues_with_the_original_trace(self):
        for benchmark in ("miniwob", "webshop"):
            with self.subTest(benchmark=benchmark), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, fresh = self.run_native(benchmark, root / "fresh", success_after=2)
                _, shared = self.run_native(benchmark, root / "shared", shared=root / "fresh", success_after=2)
                self.assertEqual([trial["success"] for trial in shared["trials"]], [False, True])
                self.assertEqual([trial["trajectory"] for trial in shared["trials"]],
                                 [trial["trajectory"] for trial in fresh["trials"]])
                if benchmark == "miniwob":
                    self.assertIn("controller_state", shared["trials"][0])
                    self.assertIn("spatial_state", shared["trials"][0])


if __name__ == "__main__":
    unittest.main()
