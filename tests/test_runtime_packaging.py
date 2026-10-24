"""Import the built wheel from an unrelated directory, never from the checkout."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile


@unittest.skipUnless(os.environ.get("VRLBENCH_TEST_WHEEL"), "build a wheel to check installed behavior")
class WheelRuntimeResources(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.installed = (self.root / "site-packages").resolve()
        with ZipFile(Path(os.environ["VRLBENCH_TEST_WHEEL"]).resolve()) as archive:
            archive.extractall(self.installed)
        self.env = dict(os.environ, PYTHONPATH=str(self.installed), PYTHONDONTWRITEBYTECODE="1")
        self.env.pop("OPENAI_API_KEY", None)
        self.env.pop("OPENAI_BASE_URL", None)

    def command(self, *args: str, no_site: bool = False) -> str:
        completed = subprocess.run(
            [sys.executable, *(["-S"] if no_site else []), *args], cwd=self.root,
            env=self.env, check=True, text=True, capture_output=True, timeout=30,
        )
        return completed.stdout

    def test_lazy_help_plan_and_statistics_api_without_optional_dependencies(self):
        source = """
import sys
from pathlib import Path
import vrlbench
from vrlbench import analyze_results, posthoc_power_mcnemar
assert Path(vrlbench.__file__).resolve().is_relative_to(Path.cwd() / 'site-packages')
assert callable(analyze_results) and callable(posthoc_power_mcnemar)
assert not any(name.split('.')[0] in {'openai', 'numpy', 'scipy', 'vex2', 'benchmarks'} for name in sys.modules)
print('installed API imports lazily')
"""
        self.assertEqual(self.command("-c", source, no_site=True).strip(), "installed API imports lazily")
        help_text = self.command("-m", "vrlbench", "--help", no_site=True)
        self.assertIn("simulate", help_text)
        plan = json.loads(self.command("-m", "vrlbench", "demo", "--plan", no_site=True))
        self.assertEqual(plan["model_calls_executed"], 0)

    def test_environment_import_resolves_packaged_locks_and_overlay(self):
        source = """
import json
import sys
import vrlbench
from vrlbench.environments import webshop_dependency as dependency
from vrlbench.environments.synapse_dependency import synapse_dependency_contract
contract = synapse_dependency_contract()
assert 'vex2' not in sys.modules and 'benchmarks' not in sys.modules
print(json.dumps({'sdk': vrlbench.__file__, 'module': dependency.__file__,
 'lock': str(dependency.WEBSHOP_REQUIREMENTS_LOCK),
 'issues': dependency.WEBSHOP_REQUIREMENTS_LOCK_ISSUES,
 'packages': sorted(dependency.WEBSHOP_RUNTIME_PACKAGE_VERSIONS),
 'overlay': contract['runtime_overlay']}))
"""
        result = json.loads(self.command("-c", source))
        for field in ("sdk", "module", "lock"):
            self.assertTrue(Path(result[field]).resolve().is_relative_to(self.installed), result)
        self.assertTrue(Path(result["lock"]).is_file())
        self.assertEqual(result["issues"], [])
        self.assertIn("pyserini", result["packages"])
        self.assertIn("spacy", result["packages"])

    def test_installed_demo_executes_the_real_trial_loop_offline(self):
        result = json.loads(self.command("-m", "vrlbench", "demo", "--output-dir", str(self.root / "demo")))
        self.assertEqual(result["status"], "complete")
        inspection = json.loads(self.command("-m", "vrlbench", "inspect", str(self.root / "demo"),
                                             "--case", "demo-0", "--format", "json"))
        self.assertEqual(inspection["case"]["case_id"], "demo-0")
        self.assertEqual(len(inspection["case"]["trials"]), 2)


if __name__ == "__main__":
    unittest.main()
