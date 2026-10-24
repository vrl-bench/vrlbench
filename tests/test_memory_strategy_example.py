"""Checks for the public seed example, not measurements of task performance."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("memory_example_check", ROOT / "examples/memory_strategy/check.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class MemoryExample(unittest.TestCase):
    def test_empty_and_reset(self):
        memory = check.load_seed()
        self.assertEqual(memory.build_section(), "")
        memory.on_failure("failed trial", "synthetic")
        self.assertIn("failed trial", memory.build_section())
        memory.reset()
        self.assertEqual(memory.build_section(), "")

    def test_bounded_history(self):
        memory = check.load_seed()
        for name in ("first-failure", "second-failure", "third-failure"):
            memory.on_failure(name, "synthetic")
        section = memory.build_section()
        self.assertNotIn("first-failure", section)
        self.assertIn("second-failure", section)
        self.assertIn("third-failure", section)

    def test_example_reports_interface_only(self):
        output = subprocess.check_output(
            [sys.executable, str(ROOT / "examples/memory_strategy/check.py")],
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), text=True)
        result = json.loads(output)
        self.assertEqual(result["interface_check"], "passed")
        self.assertIs(result["task_performance_evaluated"], False)
        self.assertNotIn("combined_score", result)


if __name__ == "__main__":
    unittest.main()
