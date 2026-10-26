"""Read-only behavior views must not invent absent memory or hide partial runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vrlbench.inspection import inspect_case, render_inspection
from vrlbench.results import EvaluationResult


def fixture() -> dict:
    return {"status": "timeout", "counts": {"planned": 2, "completed": 0},
            "trials": 3,
            "config": {"memory": {"name": "custom"}, "trials": 3},
            "cases": [{"case_id": "case-0", "task": "task", "trials": [
                {"trial": 0, "success": False, "context": "", "trajectory": "Action: left\nObservation: locked"},
                {"trial": 1, "success": False, "context": "keep this\nverbatim", "system_context": "sys",
                 "user_context": "user", "trajectory": "Action: right\nObservation: locked",
                 "llm_usage": {"api_calls": 0}}]}]}


class Inspection(unittest.TestCase):
    def test_exact_contexts_missing_fields_and_partial_status(self):
        payload = fixture()
        original = json.loads(json.dumps(payload))
        result = EvaluationResult(payload)
        view = inspect_case(result, "case-0")
        self.assertEqual(view["status"], "timeout")
        first, second = view["case"]["trials"]
        self.assertEqual(first["context"], "")
        self.assertIsNone(first["system_context"])
        self.assertIsNone(first["llm_usage"])
        self.assertEqual(second["context"], "keep this\nverbatim")
        self.assertEqual(second["trial"], 1)
        self.assertEqual(second["trial_number"], 2)
        text = render_inspection(view)
        self.assertIn("Simulation status: timeout", text)
        self.assertIn("keep this\nverbatim", text)
        self.assertIn("[unknown / not recorded]", text)
        self.assertIn("[empty]", text)
        self.assertEqual(result.to_dict(), original)
        self.assertEqual(payload, original)

    def test_index_selection_and_duplicate_rejection(self):
        result = EvaluationResult(fixture())
        self.assertEqual(inspect_case(result)["cases"], [{"case_id": "case-0", "task": "task"}])
        with self.assertRaisesRegex(ValueError, "not recorded"):
            inspect_case(result, "absent")
        payload = fixture()
        payload["cases"].append(payload["cases"][0])
        with self.assertRaisesRegex(ValueError, "unique"):
            inspect_case(EvaluationResult(payload))

    def test_cli_reads_without_loading_plugins_or_modifying_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = fixture()
            payload["config"]["memory"]["factory"] = {"entrypoint": "unavailable_plugin:create"}
            path = root / "evaluation.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            before = path.read_bytes()
            env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
            env.pop("OPENAI_API_KEY", None)
            output = subprocess.check_output(
                [sys.executable, "-m", "vrlbench", "inspect", str(root),
                 "--case", "case-0", "--format", "json"], cwd=directory, env=env, text=True)
            self.assertEqual(json.loads(output)["case"]["trials"][1]["system_context"], "sys")
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(sorted(item.name for item in root.iterdir()), ["evaluation.json"])


if __name__ == "__main__":
    unittest.main()
