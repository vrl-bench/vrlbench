"""Canonical simulation APIs must execute the same memory process as old callers."""

from __future__ import annotations

from dataclasses import fields, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vrlbench import (
    BenchmarkSpec, EvaluationResult, EvaluationSpec, FactorySpec, MemorySpec,
    ModelSpec, SimulationResult, SimulationSpec, WorkloadSpec,
    evaluate, list_methods, load_method, plan, simulate,
)
from vrlbench.evaluation import evaluate as compatible_evaluate
from vrlbench.inspection import inspect_case


def demo(output: Path) -> SimulationSpec:
    return SimulationSpec(
        workload=WorkloadSpec("demo", ["a", "b"]),
        model=ModelSpec("scripted-local"),
        memory=MemorySpec("trace", FactorySpec("vrlbench.backends:create_demo_memory")),
        trials=3, output_dir=str(output), require_provider_usage=False,
    )


class SimulationInputs(unittest.TestCase):
    def test_canonical_types_are_real_and_compatibility_names_are_same_types(self):
        self.assertIs(WorkloadSpec, BenchmarkSpec)
        self.assertIs(SimulationSpec, EvaluationSpec)
        self.assertIs(SimulationResult, EvaluationResult)
        self.assertEqual(WorkloadSpec.__name__, "WorkloadSpec")
        self.assertEqual(SimulationSpec.__name__, "SimulationSpec")
        self.assertEqual(SimulationResult.__name__, "SimulationResult")
        self.assertIn("workload", {field.name for field in fields(SimulationSpec)})
        self.assertNotIn("benchmark", {field.name for field in fields(SimulationSpec)})

    def test_canonical_and_old_construction_produce_one_stable_wire_format(self):
        workload, model = WorkloadSpec("demo", ["a"]), ModelSpec("scripted")
        current = SimulationSpec(workload=workload, model=model)
        previous = EvaluationSpec(benchmark=workload, model=model)
        positional = EvaluationSpec(workload, model)
        self.assertEqual(current, previous)
        self.assertEqual(current, positional)
        self.assertIs(current.workload, current.benchmark)
        self.assertEqual(current.to_dict(), previous.to_dict())
        self.assertIn("benchmark", current.to_dict())
        self.assertNotIn("workload", current.to_dict())
        self.assertEqual(replace(current, trials=2).trials, 2)
        replacement = WorkloadSpec("demo", ["new-case"])
        self.assertEqual(replace(previous, workload=replacement).benchmark, replacement)
        with self.assertRaisesRegex(TypeError, "not both"):
            SimulationSpec(workload=workload, benchmark=workload, model=model)

    def test_new_and_old_json_inputs_roundtrip_to_same_spec(self):
        previous = demo(Path("runs/example")).to_dict()
        current = dict(previous)
        current["workload"] = current.pop("benchmark")
        self.assertEqual(SimulationSpec.from_dict(current), EvaluationSpec.from_dict(previous))
        self.assertEqual(SimulationSpec.from_dict(current).to_dict(), previous)
        for config in (dict(current, benchmark=previous["benchmark"]),
                       dict(current, unexpected=True)):
            with self.assertRaises(ValueError):
                SimulationSpec.from_dict(config)

    def test_environment_budget_defaults_resolve_once_and_preserve_explicit_overrides(self):
        for name, cases, default in (
            ("miniwob", [{"task": "click-test", "episode": 0}], 128),
            ("webshop", [0], 15),
            ("demo", ["a"], 15),
        ):
            for key in ("workload", "benchmark"):
                with self.subTest(workload=name, key=key):
                    config = {key: {"name": name, "cases": cases}, "model": {"name": "scripted"},
                              "require_provider_usage": False}
                    spec = SimulationSpec.from_dict(config)
                    self.assertEqual((spec.trials, spec.max_environment_steps), (6, default))
                    self.assertEqual(plan(spec)["config"]["max_environment_steps"], default)
                    self.assertEqual(spec.resolved().max_environment_steps, default)
                    self.assertEqual(SimulationSpec.from_dict(spec.to_dict()), spec)
                    config["max_environment_steps"] = 3
                    explicit = SimulationSpec.from_dict(config)
                    self.assertEqual(explicit.resolved().max_environment_steps, 3)
                    self.assertEqual(explicit.to_dict()["max_environment_steps"], 3)

    def test_public_and_legacy_method_names_validate_without_rewriting_wire_identity(self):
        legacy = {"dc_cu": "dynamic_cheatsheet_adapted", "ace": "ace_adapted", "vex2": "dsst"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "never"
            for public in list_methods():
                for name in dict.fromkeys((public, legacy.get(public, public))):
                    with self.subTest(name=name):
                        self.assertEqual(load_method(name).name, public)
                        spec = SimulationSpec(WorkloadSpec("demo", ["a"]), ModelSpec("scripted"),
                                              MemorySpec(name), output_dir=str(output),
                                              require_provider_usage=False)
                        spec.validate()
                        wire = spec.to_dict()
                        for workload_key in ("benchmark", "workload"):
                            config = dict(wire)
                            config[workload_key] = config.pop("benchmark")
                            with patch("vrlbench.simulation.subprocess.Popen") as process:
                                planned = plan(SimulationSpec.from_dict(config))
                            process.assert_not_called()
                            self.assertEqual(planned["config"]["memory"], {"name": name, "factory": None})
                            self.assertIn("benchmark", planned["config"])
                            self.assertNotIn("workload", planned["config"])
                            self.assertEqual(planned["model_calls_executed"], 0)
                        self.assertEqual(SimulationSpec.from_dict(wire).to_dict(), wire)
            # Method-API presentation aliases remain distinct from simulation
            # identifiers; accepting Retry aliases here would affect shared T1.
            for name in ("none", "dc-cu", "DC-Cu*", "ACE*", "VEX²", "Reflexion"):
                with self.subTest(presentation_alias=name):
                    load_method(name)
                    spec = SimulationSpec(WorkloadSpec("demo", ["a"]), ModelSpec("scripted"),
                                          MemorySpec(name), output_dir=str(output),
                                          require_provider_usage=False)
                    with self.assertRaises(ValueError):
                        spec.validate()
            self.assertFalse(output.exists())

    def test_workload_configuration_resolves_paths_without_importing_factory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            plugin = root / "candidate.py"
            plugin.write_text("raise RuntimeError('planning imported the candidate')\n")
            value = demo(Path("runs/planned")).to_dict()
            value["workload"] = value.pop("benchmark")
            value["memory"]["factory"]["entrypoint"] = "candidate.py:create_memory"
            path = root / "simulation.json"
            path.write_text(json.dumps(value))
            spec = SimulationSpec.from_path(path)
            self.assertEqual(spec.output_dir, str(root / "runs/planned"))
            self.assertEqual(spec.memory.factory.entrypoint, f"{plugin}:create_memory")
            with patch("vrlbench.simulation.subprocess.Popen") as process:
                planned = plan(spec)
                process.assert_not_called()
            self.assertEqual(planned["planned_cases"], 2)
            self.assertEqual(planned["model_calls_executed"], 0)
            self.assertFalse((root / "runs").exists())

    def test_memory_system_uses_the_existing_lifecycle_base(self):
        # Import the optional runtime in a child, preserving the test parent's
        # module state just as normal simulation workers do.
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
        output = subprocess.check_output([
            sys.executable, "-c",
            "from vrlbench.memory import MemorySystem,ERSlot; "
            "from vrlbench.runtime.er.base import ERSlot as Original; "
            "assert MemorySystem is ERSlot is Original; print('same lifecycle')",
        ], cwd=tempfile.gettempdir(), env=env, text=True)
        self.assertEqual(output.strip(), "same lifecycle")


class SimulationProcess(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_single_execution_core_starts_one_worker_and_retains_memory_process(self):
        self.assertIs(simulate, evaluate)
        self.assertIs(simulate, compatible_evaluate)
        self.assertEqual(simulate.__module__, "vrlbench.simulation")
        environment = dict(os.environ)
        with patch("vrlbench.simulation.subprocess.Popen", wraps=subprocess.Popen) as process:
            result = simulate(demo(self.root / "simulation")).require_complete()
        self.assertEqual(process.call_count, 1)
        self.assertIs(type(result), SimulationResult)
        self.assertEqual(result.counts["completed"], 2)
        self.assertEqual(result.metrics["avg_t_at_t"], 2)
        self.assertEqual(result.resources["api_calls"], 0)
        self.assertEqual(os.environ, environment)
        for case in result.cases:
            first, second = case["trials"]
            self.assertEqual(first["context"], "")
            self.assertFalse(first["success"])
            self.assertIn(first["trajectory"], second["context"])
            self.assertTrue(second["success"])
        recorded = inspect_case(result, "a")["case"]["trials"]
        self.assertEqual(recorded[1]["context"], result.cases[0]["trials"][1]["context"])
        self.assertTrue((self.root / "simulation/evaluation.json").is_file())

    def test_result_system_views_share_records_without_mutation(self):
        result = simulate(demo(self.root / "views")).require_complete()
        snapshot = result.to_dict()
        self.assertEqual(result.workload, snapshot["config"]["benchmark"])
        self.assertEqual(result.memory, snapshot["config"]["memory"])
        self.assertEqual(result.model, snapshot["config"]["model"])
        result.workload["cases"].append("not-recorded")
        result.memory["name"] = "changed"
        result.model["name"] = "changed"
        self.assertEqual(result.to_dict(), snapshot)
        loaded = SimulationResult.from_path(self.root / "views/evaluation.json")
        self.assertEqual(loaded.to_dict(), snapshot)
        self.assertIs(type(loaded), EvaluationResult)

    def command(self, *arguments: str) -> subprocess.CompletedProcess:
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
        env.pop("OPENAI_API_KEY", None)
        env.pop("OPENAI_BASE_URL", None)
        return subprocess.run([sys.executable, "-m", "vrlbench", *arguments],
                              cwd=self.root, env=env, text=True, capture_output=True, check=True)

    def test_cli_simulate_and_evaluate_alias_execute_same_process_then_inspect(self):
        config = demo(self.root / "configured").to_dict()
        config["workload"] = config.pop("benchmark")
        path = self.root / "simulation.json"
        path.write_text(json.dumps(config))
        current = json.loads(self.command("simulate", "--config", str(path),
                                          "--output-dir", str(self.root / "current")).stdout)
        previous = json.loads(self.command("evaluate", "--config", str(path),
                                           "--output-dir", str(self.root / "previous")).stdout)
        self.assertEqual(current["status"], "complete")
        self.assertEqual(current["metrics"], previous["metrics"])
        view = json.loads(self.command("inspect", str(self.root / "current"),
                                       "--case", "a", "--format", "json").stdout)
        self.assertIn("FAIL", view["case"]["trials"][1]["context"])
        self.assertTrue(view["case"]["trials"][1]["success"])

    def test_cli_plan_remains_nonexecuting_and_demo_uses_simulation_core(self):
        config = demo(self.root / "never").to_dict()
        config["workload"] = config.pop("benchmark")
        plugin = self.root / "forbidden.py"
        plugin.write_text("raise RuntimeError('must not import for planning')\n")
        config["memory"]["factory"]["entrypoint"] = f"{plugin}:create"
        path = self.root / "plan.json"
        path.write_text(json.dumps(config))
        planned = json.loads(self.command("simulate", "--config", str(path), "--plan").stdout)
        self.assertEqual(planned["model_calls_executed"], 0)
        self.assertFalse((self.root / "never").exists())
        output = json.loads(self.command("demo", "--output-dir", str(self.root / "demo")).stdout)
        self.assertEqual(output["status"], "complete")
        self.assertEqual(output["counts"]["recovered"], 2)

    def test_cli_json_plans_accept_public_and_legacy_method_identifiers(self):
        for name in (*list_methods(), "dynamic_cheatsheet_adapted", "ace_adapted", "dsst"):
            config = SimulationSpec(WorkloadSpec("demo", ["a"]), ModelSpec("scripted"),
                                    MemorySpec(name), output_dir=str(self.root / "never"),
                                    require_provider_usage=False).to_dict()
            config["workload"] = config.pop("benchmark")
            path = self.root / "method.json"
            path.write_text(json.dumps(config))
            for command in ("simulate", "evaluate"):
                with self.subTest(name=name, command=command):
                    planned = json.loads(self.command(command, "--config", str(path), "--plan").stdout)
                    self.assertEqual(planned["config"]["memory"]["name"], name)
                    self.assertEqual(planned["model_calls_executed"], 0)
        self.assertFalse((self.root / "never").exists())


if __name__ == "__main__":
    unittest.main()
