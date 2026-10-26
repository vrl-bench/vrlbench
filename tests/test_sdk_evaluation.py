"""End-to-end public API checks with local fixtures, never model API calls."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vrlbench import BenchmarkSpec, EvaluationSpec, FactorySpec, MemorySpec, ModelSpec, evaluate, plan
from vrlbench import __version__


def demo(output: Path, **kwargs) -> EvaluationSpec:
    return EvaluationSpec(benchmark=BenchmarkSpec("demo", ["a", "b"]),
                          model=ModelSpec("scripted-local"),
                          memory=MemorySpec("trace", FactorySpec("vrlbench.backends:create_demo_memory")),
                          trials=3, output_dir=str(output), require_provider_usage=False, **kwargs)


class PublicEvaluation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_plan_does_not_import_plugin_or_write_output(self):
        plugin = self.root / "forbidden.py"
        plugin.write_text("raise RuntimeError('a plan must not import me')\n")
        value = demo(self.root / "plan").to_dict()
        value["memory"]["factory"]["entrypoint"] = f"{plugin}:create_memory"
        resolved = plan(EvaluationSpec.from_dict(value))
        self.assertEqual(resolved["planned_cases"], 2)
        self.assertEqual(resolved["model_calls_executed"], 0)
        self.assertFalse((self.root / "plan").exists())

    def test_full_demo_keeps_parent_state_and_read_write_context(self):
        before = dict(os.environ)
        modules = {key for key in sys.modules if key == "vrlbench.runtime" or key.startswith("vrlbench.runtime.")}
        result = evaluate(demo(self.root / "run")).require_complete()
        self.assertEqual(result.metrics["sr_at_t"], 1)
        self.assertEqual(result.metrics["avg_t_at_t"], 2)
        self.assertEqual(result.counts["recovered"], 2)
        self.assertEqual(result.to_dict()["identity"]["sdk_version"], __version__)
        for case in result.cases:
            self.assertEqual(len(case["trials"]), 2)
            self.assertEqual(case["trials"][0]["context"], "")
            self.assertIn("FAIL", case["trials"][1]["context"])
        self.assertEqual(before, dict(os.environ))
        self.assertEqual(modules, {key for key in sys.modules if key == "vrlbench.runtime" or key.startswith("vrlbench.runtime.")})
        self.assertTrue((self.root / "run/evaluation.json").is_file())

    def test_concurrent_evaluations_use_separate_workers(self):
        with ThreadPoolExecutor(2) as executor:
            values = list(executor.map(evaluate, [demo(self.root / "one"), demo(self.root / "two")]))
        self.assertTrue(all(result.status == "complete" for result in values))
        self.assertNotEqual(values[0].output_dir, values[1].output_dir)

    def test_existing_output_is_never_overwritten(self):
        target = self.root / "existing"
        target.mkdir()
        (target / "keep.txt").write_text("keep")
        with self.assertRaises(FileExistsError):
            evaluate(demo(target))
        self.assertEqual((target / "keep.txt").read_text(), "keep")

    def test_timeout_is_not_a_task_failure_and_cleans_process_group(self):
        plugin = self.root / "slow.py"
        plugin.write_text("import subprocess,sys,time\nclass Backend:\n def run(self,spec,runtime):\n  subprocess.Popen([sys.executable,'-c',\"import time,pathlib;time.sleep(3);pathlib.Path('escaped').write_text('bad')\"])\n  time.sleep(30)\ndef create(): return Backend()\n")
        value = demo(self.root / "timeout").to_dict()
        value["benchmark"] = {"name": "slow", "cases": ["a"], "factory": {"entrypoint": f"{plugin}:create"}}
        value["timeout_seconds"] = 1
        result = evaluate(EvaluationSpec.from_dict(value))
        self.assertEqual(result.status, "timeout")
        self.assertEqual(result.counts["completed"], 0)
        self.assertIsNone(result.metrics["sr_at_t"])
        with self.assertRaises(RuntimeError):
            result.require_complete()
        time.sleep(3)
        self.assertFalse((self.root / "timeout/escaped").exists())

    def test_factory_exception_is_structured(self):
        plugin = self.root / "bad.py"
        plugin.write_text("def create(): raise ValueError('fixture failure')\n")
        value = demo(self.root / "bad").to_dict()
        value["memory"]["factory"]["entrypoint"] = f"{plugin}:create"
        result = evaluate(EvaluationSpec.from_dict(value))
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.to_dict()["error"]["type"], "ValueError")
        self.assertEqual(result.counts["completed"], 0)

    def test_timeout_retains_native_completed_cases_and_recorded_cost(self):
        plugin = self.root / "checkpoint.py"
        plugin.write_text(
            "import json,pathlib,time\n"
            "class Backend:\n"
            " def run(self,spec,runtime):\n"
            "  root=pathlib.Path(spec['output_dir'])/'reference';root.mkdir()\n"
            "  row={'name':'a','task':'fixture','trials':[{'trial':0,'success':True,'trajectory':'done'}],"
            "'resource_usage':{'api_calls':1,'prompt_tokens':10,'completion_tokens':15,'total_tokens':25,'provider_usage_complete':True}}\n"
            "  (root/'episodes.json').write_text(json.dumps([row]))\n"
            "  time.sleep(30)\n"
            "def create(): return Backend()\n"
        )
        value = demo(self.root / "checkpoint").to_dict()
        value["benchmark"] = {"name": "checkpoint", "cases": ["a", "b"],
                              "factory": {"entrypoint": f"{plugin}:create"}}
        value["timeout_seconds"] = 3
        result = evaluate(EvaluationSpec.from_dict(value))
        self.assertEqual(result.status, "timeout")
        self.assertEqual(result.counts["planned"], 2)
        self.assertEqual(result.counts["completed"], 1)
        self.assertEqual(result.cases[0]["case_id"], "a")
        self.assertEqual(result.resources["known_subtotals"]["total_tokens"], 25)
        with self.assertRaises(RuntimeError):
            result.require_complete()

    def test_spec_roundtrip_relative_paths_and_validation(self):
        config = self.root / "config.json"
        value = demo(Path("new-run")).to_dict()
        config.write_text(json.dumps(value))
        resolved = EvaluationSpec.from_path(config)
        self.assertEqual(resolved.output_dir, str((self.root / "new-run").resolve()))
        self.assertEqual(EvaluationSpec.from_dict(resolved.to_dict()).to_dict(), resolved.to_dict())
        for field, bad in (("trials", True), ("timeout_seconds", float("inf")), ("unknown", 1)):
            invalid = dict(value, **{field: bad})
            with self.assertRaises((ValueError, TypeError)):
                plan(EvaluationSpec.from_dict(invalid))

    def test_invalid_requests_fail_before_any_worker(self):
        value = demo(self.root / "never").to_dict()
        for section, key, bad in (("memory", "name", "retry"),
                                  ("model", "request_options", {"model": "wrong"}),
                                  ("model", "base_url", "https://user:secret@example.com/v1"),
                                  ("benchmark", "cases", ["same", "same"])):
            invalid = json.loads(json.dumps(value))
            invalid[section][key] = bad
            with self.assertRaises(ValueError):
                evaluate(EvaluationSpec.from_dict(invalid))
        self.assertFalse((self.root / "never").exists())


if __name__ == "__main__":
    unittest.main()
