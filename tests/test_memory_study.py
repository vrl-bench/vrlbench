"""Properties of the teaching intervention, not claims about LLM performance."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
EXAMPLE = ROOT / "examples/memory_study"


def load_example(name):
    spec = importlib.util.spec_from_file_location(f"study_{name}", EXAMPLE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixture = load_example("fixture")
study = load_example("run")


class MemoryStudy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temporary.name) / "default"
        cls.report = study.run_study(cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_outcome_follows_environment_action_not_trial_or_method(self):
        for target in fixture.CONTROLS:
            environment = fixture.RouteEnvironment(target)
            for action in fixture.CONTROLS:
                observation = environment.reset()
                self.assertNotIn("target", observation)
                self.assertEqual(fixture.select_action(observation, ""), "amber")
                feedback = environment.step(action)
                self.assertEqual(feedback["success"], action == target)
                self.assertEqual(environment.success, action == target)
                with self.assertRaises(RuntimeError):
                    environment.step(action)

    def test_memory_changes_action_via_the_actual_injected_context(self):
        records = self.report["conditions"]
        retry = records["retry"]["cases"]
        feedback = records["feedback"]["cases"]
        for left, right in zip(retry, feedback):
            self.assertEqual(left["trials"][0]["trajectory"], right["trials"][0]["trajectory"])
            self.assertEqual(right["trials"][0]["context"], "")
        case = feedback[1]
        actions = [json.loads(trial["trajectory"])["action"] for trial in case["trials"]]
        self.assertEqual(actions, ["amber", "blue", "green"])
        self.assertIn('Rejected controls JSON: ["amber"]', case["trials"][1]["context"])
        for trial in case["trials"]:
            trace = json.loads(trial["trajectory"])
            self.assertEqual(fixture.select_action(trace["initial_observation"], trial["context"]),
                             trace["action"])
            self.assertEqual(trial["success"], trace["observation"]["success"])
        self.assertEqual([json.loads(t["trajectory"])["action"] for t in retry[1]["trials"]],
                         ["amber", "amber", "amber"])

    def test_budget_success_stop_and_per_case_memory_isolation(self):
        for condition in self.report["conditions"].values():
            self.assertEqual(condition["resources"]["api_calls"], 0)
            for case in condition["cases"]:
                self.assertEqual(case["trials"][0]["context"], "")
                self.assertLessEqual(len(case["trials"]), 3)
                self.assertFalse(any(t["success"] for t in case["trials"][:-1]))
                self.assertEqual(sum(t["trial_metadata"]["environment_steps"]
                                     for t in case["trials"]), len(case["trials"]))
            self.assertEqual(len(condition["cases"][2]["trials"]), 1)
        short = study.run_study(Path(self.temporary.name) / "short", trials=1)
        self.assertEqual(short["conditions"]["retry"]["counts"],
                         short["conditions"]["feedback"]["counts"])
        self.assertTrue(all(len(c["trials"]) == 1 for c in short["conditions"]["feedback"]["cases"]))

    def test_forgetting_changes_observed_behavior_without_a_new_actor(self):
        report = study.run_study(Path(self.temporary.name) / "window1", keep_last=1)
        case = report["conditions"]["feedback"]["cases"][1]
        self.assertEqual([json.loads(t["trajectory"])["action"] for t in case["trials"]],
                         ["amber", "blue", "amber"])
        self.assertFalse(case["trials"][-1]["success"])

    def test_reports_are_exact_records_and_nonempty_output_is_protected(self):
        saved = json.loads((self.output / "report.json").read_text())
        self.assertEqual(saved, self.report)
        markdown = (self.output / "report.md").read_text()
        for condition in saved["conditions"].values():
            actual = json.loads((self.output / condition["result_path"]).read_text())
            self.assertEqual(condition["cases"], actual["cases"])
            self.assertEqual(condition["counts"], actual["counts"])
            self.assertEqual(condition["configuration"]["workload"], actual["config"]["benchmark"])
            if condition["memory_events"] is not None:
                event_path = (self.output / condition["result_path"]).parent / "memory_events.jsonl"
                self.assertEqual(condition["memory_events"],
                                 [json.loads(line) for line in event_path.read_text().splitlines()])
            for case in condition["cases"]:
                for trial in case["trials"]:
                    self.assertIn(trial["trajectory"], markdown)
                    if trial["context"]:
                        self.assertIn(trial["context"], markdown)
        self.assertIsNone(saved["conditions"]["retry"]["memory_events"])
        events = saved["conditions"]["feedback"]["memory_events"]
        case = saved["conditions"]["feedback"]["cases"][1]
        case_events = [event for event in events if event["case_id"] == "case_b"]
        self.assertEqual([(event["operation"], event["trial"]) for event in case_events],
                         [("write", 0), ("read", 1), ("write", 1), ("read", 2)])
        for event in case_events:
            trial = case["trials"][event["trial"]]
            if event["operation"] == "write":
                self.assertEqual(event["input_trajectory"], trial["trajectory"])
                self.assertFalse(trial["success"])
            else:
                self.assertIn(event["returned_context"], trial["context"])
        self.assertEqual(case_events[0]["retained_before"], [])
        self.assertEqual(case_events[0]["retained_after"], ["amber"])
        self.assertEqual(case_events[2]["retained_before"], ["amber"])
        self.assertEqual(case_events[2]["retained_after"], ["amber", "blue"])
        before = (self.output / "report.json").read_bytes()
        with self.assertRaises(FileExistsError):
            study.run_study(self.output)
        self.assertEqual((self.output / "report.json").read_bytes(), before)

    def test_cli_configures_the_same_simulation_and_script_runs_outside_checkout(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), OPENAI_API_KEY="not-a-real-key")
        for condition in ("retry", "feedback"):
            output = Path(self.temporary.name) / f"cli_{condition}"
            subprocess.run([sys.executable, "-m", "vrlbench", "simulate", "--config",
                            str(ROOT / f"examples/configs/local_{condition}.json"),
                            "--output-dir", str(output)], cwd=self.temporary.name,
                           check=True, capture_output=True, text=True, env=env)
            actual = json.loads((output / "evaluation.json").read_text())
            self.assertEqual(actual["cases"], self.report["conditions"][condition]["cases"])
            self.assertEqual(actual["resources"]["api_calls"], 0)
            self.assertEqual((output / "memory_events.jsonl").is_file(), condition == "feedback")
        output = Path(self.temporary.name) / "external"
        subprocess.run([sys.executable, str(EXAMPLE / "run.py"), "--output-dir", str(output)],
                       cwd=self.temporary.name, check=True, capture_output=True, text=True,
                       env=env)
        self.assertTrue((output / "report.md").is_file())
        # Invoke the transport in an isolated process with the original runtime
        # available, just as the worker does. No network implementation exists.
        command = ("from vrlbench.plugins import enable_reference_imports; enable_reference_imports(); "
                   f"import runpy; m=runpy.run_path({str(EXAMPLE / 'fixture.py')!r}); "
                   "m['create_no_model_client']().chat.completions.create(model='forbidden')")
        check = subprocess.run([sys.executable, "-c", command], cwd=self.temporary.name,
                               capture_output=True, text=True,
                               env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
        self.assertNotEqual(check.returncode, 0)
        self.assertIn("permits zero model API calls", check.stderr)


if __name__ == "__main__":
    unittest.main()
