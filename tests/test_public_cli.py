"""Offline CLI checks for source users and lazy public imports."""

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
LOCAL_CONFIG = ROOT / "examples/configs/local_feedback.json"


class PublicCLI(unittest.TestCase):
    def command(self, *args: str, no_site: bool = False) -> str:
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
        env.pop("OPENAI_BASE_URL", None)
        env.pop("OPENAI_API_KEY", None)
        return subprocess.check_output(
            [sys.executable, *(["-S"] if no_site else []), "-m", "vrlbench", *args],
            cwd=ROOT, env=env, text=True,
        )

    def test_help_without_optional_dependencies(self):
        output = self.command("--help", no_site=True)
        self.assertIn("simulate", output)
        self.assertIn("inspect", output)
        self.assertIn("compare", output)
        self.assertIn("replay", output)

    def test_run_alias_applies_model_overrides_without_execution(self):
        result = json.loads(self.command("run", "--config", str(LOCAL_CONFIG),
                                       "--model", "vision-test", "--base-url", "http://localhost:8000/v1",
                                       "--api-key-env", "TEST_MODEL_KEY", "--plan", no_site=True))
        self.assertEqual(result["config"]["model"]["name"], "vision-test")
        self.assertEqual(result["config"]["model"]["base_url"], "http://localhost:8000/v1")
        self.assertEqual(result["config"]["model"]["api_key_env"], "TEST_MODEL_KEY")
        self.assertEqual(result["model_calls_executed"], 0)

    def test_gui_preset_plans_without_browser_or_model_dependencies(self):
        result = json.loads(self.command("run", "--preset", "gui-smoke", "--model", "test-vision", "--plan", no_site=True))
        self.assertEqual(result["planned_cases"], 2)
        self.assertEqual(result["config"]["model"]["name"], "test-vision")
        self.assertEqual(result["config"]["benchmark"]["factory"]["entrypoint"],
                         "vrlbench.environments.browser_smoke:create_backend")
        self.assertEqual(result["model_calls_executed"], 0)

    def test_native_preset_matches_editable_config_and_batch_plan_is_lazy(self):
        result = json.loads(self.command("batch", "--preset", "miniwob-visual-v1",
            "--output-dir", "/tmp/vrlbench-plan-only", "--jobs", "2", "--plan", no_site=True))
        explicit = json.loads(self.command("run", "--config",
            str(ROOT / "examples/configs/miniwob_visual.json"), "--output-dir", "/tmp/vrlbench-plan-only",
            "--plan", no_site=True))
        self.assertEqual(result["config"], explicit["config"])
        self.assertEqual(result["planned_cases"], 16)
        self.assertEqual(result["model_calls_executed"], 0)

    def test_doctor_does_not_start_environment_or_create_output(self):
        result = subprocess.run([sys.executable, "-S", "-m", "vrlbench", "doctor",
                                 "--config", str(LOCAL_CONFIG)], cwd=ROOT,
                                env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
                                capture_output=True, text=True)
        output = json.loads(result.stdout)
        self.assertEqual(output["model_calls_executed"], 0)
        self.assertEqual(output["status"], "blocked")
        self.assertEqual(result.returncode, 1)
        self.assertTrue(any(c["name"] == "openai" and not c["ready"] for c in output["checks"]))

    def test_simulation_plan_does_not_import_or_execute_plugins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "request.json"
            config.write_text(json.dumps({
                "workload": {"name": "local_task", "cases": ["a"],
                             "factory": {"entrypoint": "not_installed_backend:create"}},
                "model": {"name": "unused-model"},
                "memory": {"name": "my_memory", "factory": {"entrypoint": "not_installed_memory:create"}},
                "trials": 4, "max_environment_steps": 15, "output_dir": str(root / "never-created"),
                "require_provider_usage": False,
            }))
            plan = json.loads(self.command("simulate", "--config", str(config), "--plan", no_site=True))
            self.assertEqual(plan["model_calls_executed"], 0)
            self.assertEqual(plan["config"]["trials"], 4)
            self.assertFalse((root / "never-created").exists())

    def test_missing_budgets_are_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "never-created"
            config = json.loads(self.command(
                "simulate", "--config", str(LOCAL_CONFIG), "--plan", no_site=True,
            ))["config"]
            config["output_dir"] = str(output)
            env = dict(os.environ, PYTHONPATH=str(ROOT / "src"),
                       PYTHONDONTWRITEBYTECODE="1")
            for field in ("trials", "max_environment_steps"):
                missing = dict(config)
                del missing[field]
                path = root / "config.json"
                path.write_text(json.dumps(missing))
                result = subprocess.run(
                    [sys.executable, "-S", "-m", "vrlbench", "simulate",
                     "--config", str(path)], cwd=ROOT, env=env,
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(field, result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertFalse(output.exists())

    def test_unsupported_commands_are_rejected(self):
        for name in ("demo", "results", "plan", "reproduce"):
            with self.subTest(command=name):
                result = subprocess.run(
                    [sys.executable, "-S", "-m", "vrlbench", name],
                    cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT / "src"),
                                      PYTHONDONTWRITEBYTECODE="1"),
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("invalid choice", result.stderr)

    def test_checkout_statistics_api_imports_without_analysis_dependencies(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
        code = """
import importlib.util
from pathlib import Path
import sys

assert importlib.util.find_spec("numpy") is None
assert importlib.util.find_spec("scipy") is None
from vrlbench import SimulationResult, analyze_results, posthoc_power_mcnemar
import vrlbench

assert Path(vrlbench.__file__).resolve() == Path.cwd() / "src" / "vrlbench" / "__init__.py"
assert callable(SimulationResult.from_path)
assert callable(analyze_results) and callable(posthoc_power_mcnemar)
assert not any(name.split(".")[0] in {"numpy", "scipy"} for name in sys.modules)
print("checkout statistics API imports without analysis dependencies")
"""
        output = subprocess.check_output([sys.executable, "-S", "-c", code],
                                         cwd=ROOT, env=env, text=True)
        self.assertEqual(output.strip(), "checkout statistics API imports without analysis dependencies")

    @unittest.skipUnless(importlib.util.find_spec("openai"), "install [runtime] to execute the offline loop")
    def test_simulation_and_inspection_use_recorded_run(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "local-feedback"
            result = json.loads(self.command("simulate", "--config", str(LOCAL_CONFIG),
                                             "--output-dir", str(output)))
            self.assertEqual(result["status"], "complete")
            inspection = json.loads(self.command("inspect", str(output), "--case", "case_b", "--format", "json"))
            self.assertEqual(inspection["case"]["case_id"], "case_b")
            self.assertEqual(len(inspection["case"]["trials"]), 3)


if __name__ == "__main__":
    unittest.main()
