"""Actor selection stays lazy, preserves old configurations, and bounds comparisons."""

from copy import deepcopy
from dataclasses import replace
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

from vrlbench import (ActorSpec, FactorySpec, MemorySpec, ModelSpec, SimulationSpec,
                      WorkloadSpec, aggregate_cases, compare_results, evaluate_candidate, plan)
from vrlbench.results import SimulationResult


def spec(**kwargs):
    return SimulationSpec(WorkloadSpec("webshop", [0]), ModelSpec("fixture"),
                          trials=2, max_environment_steps=1,
                          require_provider_usage=False, **kwargs)


def recorded(configuration):
    cases = [{"case_id": "sample_0000", "task": "fixture", "resource_usage": {},
              "trials": [{"trial": 0, "success": True, "trajectory": "saved"}]}]
    return SimulationResult({"status": "complete", "config": configuration.to_dict(),
                             "cases": cases, **aggregate_cases(cases, configuration.trials)})


class ActorConfiguration(unittest.TestCase):
    def test_default_react_wire_and_derived_memory_configuration_remain_compatible(self):
        original = spec().to_dict()
        self.assertNotIn("actor", original)
        self.assertEqual(SimulationSpec.from_dict(original).actor, ActorSpec())
        explicit = deepcopy(original)
        explicit["actor"] = {"name": "react", "params": {}}
        self.assertEqual(SimulationSpec.from_dict(explicit).to_dict(), original)
        selected = spec(actor=ActorSpec("openhands", {"max_output_tokens": 512}))
        changed_memory = replace(selected, memory=MemorySpec("reflexion"))
        self.assertEqual(SimulationSpec.from_dict(changed_memory.to_dict()).actor, selected.actor)
        self.assertEqual(changed_memory.resolved().actor, selected.actor)

    def test_actor_parameters_resolve_without_loading_the_condenser_or_sdk(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            interpreter = root / "environment/bin/python"
            interpreter.parent.mkdir(parents=True)
            interpreter.symlink_to(sys.executable)
            (root / "condenser.py").write_text("raise AssertionError('plan imported the condenser')\n")
            configured = spec(actor=ActorSpec("openhands", {
                "max_output_tokens": 512, "condenser": {"entrypoint": "condenser.py:create"},
            }))
            configured = replace(configured, workload=replace(configured.workload,
                params={"environment_python": "environment/bin/python"}))
            path = root / "config.json"
            path.write_text(json.dumps(configured.to_dict()))
            resolved = SimulationSpec.from_path(path)
            value = plan(resolved)["config"]
            self.assertEqual(value["actor"]["params"]["condenser"]["entrypoint"], f"{root / 'condenser.py'}:create")
            self.assertEqual(value["benchmark"]["params"]["environment_python"], str(root / "environment/bin/python"))
            with self.assertRaisesRegex(ValueError, "only supported"):
                replace(resolved, actor=ActorSpec()).validate()
            with self.assertRaisesRegex(ValueError, "max_total_tokens"):
                replace(resolved, max_total_tokens=100).validate()
            with self.assertRaisesRegex(ValueError, "model.transport"):
                replace(resolved, model=replace(resolved.model,
                    transport=FactorySpec("unused:create"))).validate()
        with self.assertRaisesRegex(ValueError, "unknown"):
            ActorSpec("openhands", {"api_key": "not-a-real-key"}).validate()
        with self.assertRaisesRegex(ValueError, "positive integer"):
            ActorSpec("openhands", {"max_output_tokens": True}).validate()

    def test_environment_subprocess_does_not_inherit_actor_python_paths(self):
        from vrlbench.environments.native_sessions import SubprocessSession

        metadata = {"system_prompt": "fixture", "basic_rules": "fixture",
                    "task_group": "fixture", "task_scope": "fixture",
                    "source_identity": {}, "allows_task_variation": False}
        explicit_path = "/environment/dependencies"
        with patch.dict(os.environ, {"PYTHONPATH": "/actor/python3.12",
                                     "PYTHONHOME": "/actor/python3.12"}), \
                patch("vrlbench.environments.native_sessions.subprocess.Popen") as start, \
                patch.object(SubprocessSession, "_request", return_value=metadata):
            start.return_value.poll.return_value = 0
            session = SubprocessSession("webshop", 0, {},
                python_executable="/environment/bin/python",
                environment_pythonpath=explicit_path)
            try:
                environment = start.call_args.kwargs["env"]
                self.assertEqual(environment["PYTHONPATH"],
                                 os.pathsep.join((str(ROOT / "src"), explicit_path)))
                self.assertNotIn("PYTHONHOME", environment)
            finally:
                session.close()

    def test_comparison_and_parent_preflight_require_the_same_actor(self):
        previous = recorded(spec())  # Older records contain no actor key.
        explicit = previous.to_dict()
        explicit["config"]["actor"] = {"name": "react", "params": {}}
        compare_results([previous, explicit])
        different = recorded(spec(actor=ActorSpec("openhands")))
        with self.assertRaisesRegex(ValueError, "incompatible"):
            compare_results([previous, different])
        default_options = recorded(spec(actor=ActorSpec("openhands", {"max_output_tokens": 1024})))
        compare_results([different, default_options])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "memory.py"
            candidate.write_text("raise AssertionError('incompatible candidate must not be imported')\n")
            with self.assertRaisesRegex(ValueError, "incompatible"):
                evaluate_candidate(candidate, spec(actor=ActorSpec("openhands"),
                    output_dir=str(root / "never-created")), reference=previous)
            self.assertFalse((root / "never-created").exists())

    def test_shared_native_case_ids_and_actor_mismatch_are_checked_without_execution(self):
        from vrlbench._shared_t1 import load_shared_first_trials

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, cases, case_id in (("webshop", [7], "sample_0007"),
                    ("miniwob", [{"task": "click-test", "episode": 3}], "click-test_ep3")):
                with self.subTest(workload=name):
                    configuration = replace(spec(actor=ActorSpec("openhands")),
                        workload=WorkloadSpec(name, cases), shared_t1_dir=str(root))
                    payload = recorded(configuration).to_dict()
                    payload["cases"][0]["case_id"] = case_id
                    payload["cases"][0]["trials"][0]["trial_metadata"] = {
                        "max_environment_steps": 1, "task": "fixture", "case_identity": case_id,
                        "actor_llm_usage": {"api_calls": 0, "total_tokens": 0,
                                            "provider_usage_complete": True},
                    }
                    (root / "evaluation.json").write_text(json.dumps(payload))
                    self.assertEqual(set(load_shared_first_trials(configuration.to_dict())), {case_id})
                    payload["config"].pop("actor")
                    (root / "evaluation.json").write_text(json.dumps(payload))
                    with self.assertRaisesRegex(ValueError, "actor"):
                        load_shared_first_trials(configuration.to_dict())

    def test_openhands_webshop_comparison_checks_recorded_environment_identity(self):
        configuration = spec(actor=ActorSpec("openhands"))
        original = recorded(configuration).to_dict()
        original["cases"][0]["source_identity"] = {
            "benchmark": "webshop", "sample_index": 0,
            "source_data_index_root_sha256": "a" * 64,
        }
        compare_results([original, deepcopy(original)])

        changed = deepcopy(original)
        changed["cases"][0]["source_identity"]["source_data_index_root_sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            compare_results([original, changed])

        contradictory = deepcopy(original)
        contradictory["identity"] = {"environment": {
            "benchmark": "webshop", "source_data_index_root_sha256": "b" * 64,
        }}
        with self.assertRaises(ValueError):
            compare_results([original, contradictory])

    def test_condenser_identity_matches_recorded_comparisons_and_shared_prefix(self):
        from vrlbench._shared_t1 import load_shared_first_trials
        from vrlbench.plugins import load_factory

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            condenser = root / "condenser.py"
            condenser.write_text(
                "def create():\n    raise AssertionError('the condenser factory must not be called')\n"
            )
            factory = {"entrypoint": f"{condenser}:create", "kwargs": {}}
            _create, identity = load_factory(factory)
            configuration = spec(actor=ActorSpec("openhands", {"condenser": factory}),
                                 output_dir=str(root / "candidate"))
            original = recorded(configuration).to_dict()
            original["cases"][0]["trials"][0]["trial_metadata"] = {
                "max_environment_steps": 1, "task": "fixture", "case_identity": "sample_0000",
                "actor_name": "openhands", "condenser_identity": identity,
                "actor_llm_usage": {"api_calls": 0, "total_tokens": 0,
                                    "provider_usage_complete": True},
            }
            compare_results([original, deepcopy(original)])
            changed = deepcopy(original)
            changed["cases"][0]["trials"][0]["trial_metadata"]["condenser_identity"]["source_sha256"] = "b" * 64
            with self.assertRaises(ValueError):
                compare_results([original, changed])

            (root / "evaluation.json").write_text(json.dumps(original))
            continuation = replace(configuration, shared_t1_dir=str(root))
            self.assertEqual(
                set(load_shared_first_trials(continuation.to_dict(), condenser_identity=identity)),
                {"sample_0000"},
            )
            condenser.write_text(
                "def create():\n    raise AssertionError('the changed factory must not be called')\n"
            )
            _changed_create, current = load_factory(factory)
            self.assertNotEqual(identity["source_sha256"], current["source_sha256"])
            with self.assertRaises(ValueError):
                load_shared_first_trials(continuation.to_dict(), condenser_identity=current)

    def test_cli_actor_override_can_plan_without_optional_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "config.json"
            path.write_text(json.dumps(spec(output_dir=str(root / "unused")).to_dict()))
            command = [sys.executable, "-S", "-m", "vrlbench", "simulate", "--config", str(path),
                       "--actor", "openhands", "--output-dir", str(root / "selected"), "--plan"]
            environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
            value = json.loads(subprocess.check_output(command, cwd=root, env=environment, text=True))
            self.assertEqual(value["config"]["actor"], {"name": "openhands", "params": {}})
            self.assertEqual(value["model_calls_executed"], 0)
            self.assertFalse((root / "selected").exists())


if __name__ == "__main__":
    unittest.main()
