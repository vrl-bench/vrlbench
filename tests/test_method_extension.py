"""Offline whole-method extension tests against the unchanged reference loop."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vrlbench.specs import FactorySpec, MemorySpec, ModelSpec, SimulationSpec, WorkloadSpec

HAS_RUNTIME = importlib.util.find_spec("openai") is not None
RAW_CONTEXT = "  Joint raw response\n<policy> α & β </policy>\n\n"

PLUGIN = '''
import json
import uuid
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from vrlbench.methods import ActorContext, MemoryMethod

RAW = "  Joint raw response\\n<policy> α & β </policy>\\n\\n"

class Method(MemoryMethod):
    def __init__(self, log, mode="normal"):
        self.log = Path(log)
        self.identity = str(uuid.uuid4())
        self.mode = mode
    def event(self, kind, **value):
        with self.log.open("a") as output:
            output.write(json.dumps({"kind": kind, "identity": self.identity, **value}) + "\\n")
    def reset_case(self, config):
        self.records = []
        self.event("reset", config=asdict(config))
    def context_for_trial(self, index):
        self.event("context", index=index)
        if self.mode == "wrong_context": return "not ActorContext"
        if self.mode == "first_memory": return ActorContext(system="unshared first memory")
        if self.mode == "user" and index: return ActorContext(user="must not be dropped")
        if self.mode == "model" and index:
            from vrlbench.runtime.llm import complete
            complete("must stay offline", model="scripted")
        return ActorContext(system=RAW if index else "")
    def observe_trial(self, record):
        self.records.append(asdict(record))
        self.event("observe", record=asdict(record))
    def snapshot(self):
        return {"identity": self.identity, "records": self.records}

def create_method(log, mode="normal"):
    return Method(log, mode)
_singleton = None
def create_singleton(log):
    global _singleton
    if _singleton is None: _singleton = Method(log)
    return _singleton
def wrong_factory(): return {}

class Backend:
    def run(self, spec, runtime):
        from vrlbench.runtime import llm
        rows = []
        for case in spec["benchmark"]["cases"]:
            options = spec["benchmark"].get("params", {})
            class Actor:
                def set_trial_context(self, *, trial_index, total_trials):
                    self.index = trial_index
                    return {}
                def __call__(self):
                    self.context = llm.get_context()
                    outcome = "success" if self.index + 1 >= options.get("success_after", 2) else "failure"
                    return "private Actor transcript " + outcome
                def get_actor_trajectory(self):
                    return "private raw Actor transcript"
                def get_policy_evidence(self):
                    return {"schema_version": 1, "task": "canonical custom task " + str(case),
                            "initial_observation": "visible initial state",
                            "steps": [{"action": "click actual target", "observation": "actual visible outcome"}],
                            "trial_close_reason": "done"}
                def get_trial_metadata(self):
                    return {"policy_evidence_eligible": self.index != options.get("ineligible_index", -1)}
            episode = runtime.run_episode(Actor(), lambda trace: trace.endswith("success"),
                metadata={"task_scope": "custom task family", "basic_environment_rules": "one actual click"})
            rows.append({"case_id": str(case), "task": "custom-task",
                         "trials": [asdict(trial) for trial in episode.trials],
                         "resource_usage": episode.resource_usage})
        return rows
def create_backend(): return Backend()

class Completion:
    def create(self, **kwargs):
        return SimpleNamespace(model="scripted",
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=5, total_tokens=12),
            choices=[SimpleNamespace(message=SimpleNamespace(content="click[buy now]"))])
def create_client():
    return SimpleNamespace(chat=SimpleNamespace(completions=Completion()))

class Environment:
    def __init__(self): self.resets = 0
    def step(self, session, action):
        if action == "reset":
            self.resets += 1
            return "Instruction: Buy synthetic sofa\\nAvailable actions: buy now", 0.0, False
        return "Synthetic purchase outcome", 1.0 if self.resets >= 2 else 0.0, True
class WebShopBackend:
    def run(self, spec, runtime):
        from vrlbench.environments import webshop
        previous = webshop._load_default_env_factory
        try:
            webshop._load_default_env_factory = lambda: Environment
            return runtime.run_reference()
        finally:
            webshop._load_default_env_factory = previous
def create_webshop_backend(): return WebShopBackend()
'''


class MethodConfigurationTests(unittest.TestCase):
    def test_old_wire_is_unchanged(self):
        spec = SimulationSpec(WorkloadSpec("demo", ["a"]), ModelSpec("scripted"))
        self.assertEqual(spec.to_dict()["memory"], {"name": "retry", "factory": None})

    def test_method_factory_json_resolution_and_mutual_exclusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugin = root / "method.py"
            plugin.write_text("def create(): return None\n")
            spec = SimulationSpec(
                WorkloadSpec("demo", ["a"]), ModelSpec("scripted"),
                MemorySpec("example", method_factory=FactorySpec("method.py:create")),
                require_provider_usage=False,
            )
            source = root / "request.json"
            source.write_text(json.dumps(spec.to_dict()))
            restored = SimulationSpec.from_path(source)
            restored.validate()
            self.assertEqual(restored.memory.method_factory.entrypoint, f"{plugin.resolve()}:create")
            both = SimulationSpec(WorkloadSpec("demo", ["a"]), ModelSpec("scripted"),
                MemorySpec("example", FactorySpec(f"{plugin}:create"), FactorySpec(f"{plugin}:create")))
            with self.assertRaisesRegex(ValueError, "not both"):
                both.validate()
            for name in ("retry", "reflexion", "dynamic_cheatsheet_adapted", "ace_adapted",
                         "dsst", "vex2", "ace", "dc_cu", "DC-CU", "ACE-ADAPTED",
                         "DYNAMIC-CHEATSHEET-ADAPTED", "VEX2", "none"):
                for field in ("factory", "method_factory"):
                    with self.subTest(name=name, field=field):
                        builtin = SimulationSpec(WorkloadSpec("demo", ["a"]), ModelSpec("scripted"),
                            MemorySpec(name, **{field: FactorySpec(f"{plugin}:create")}))
                        with self.assertRaisesRegex(ValueError, "override"):
                            builtin.validate()

    @unittest.skipUnless(HAS_RUNTIME, "Install the runtime extra for worker registration checks")
    def test_worker_cannot_override_reference_names_before_loading_a_factory(self):
        from vrlbench.plugins import enable_reference_imports
        enable_reference_imports()
        from vrlbench.backends import register_memory

        for name in ("retry", "reflexion", "dynamic_cheatsheet_adapted", "ace_adapted",
                     "dsst", "dc_cu", "ace", "vex2", "DC-CU", "ACE-ADAPTED",
                     "DYNAMIC-CHEATSHEET-ADAPTED", "VEX2", "none"):
            for field in ("factory", "method_factory"):
                with self.subTest(name=name, field=field), patch("vrlbench.backends.load_factory") as factory:
                    with self.assertRaisesRegex(ValueError, "replace reference"):
                        register_memory({"name": name, field: {"entrypoint": "builtins:object"}})
                    factory.assert_not_called()


@unittest.skipUnless(HAS_RUNTIME, "Install the runtime extra for reference-loop extension tests")
class WholeMethodWorkerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.plugin = self.root / "candidate.py"
        self.plugin.write_text(textwrap.dedent(PLUGIN))
        self.log = self.root / "events.jsonl"
        self.counter = 0

    def request(self, *, mode="normal", total=3, cases=None, params=None):
        self.counter += 1
        return SimulationSpec(
            WorkloadSpec("custom", ["first", "second"] if cases is None else cases,
                params=params or {}, factory=FactorySpec(f"{self.plugin}:create_backend")),
            ModelSpec("scripted"),
            MemorySpec("joint_method", method_factory=FactorySpec(
                f"{self.plugin}:create_method", {"log": str(self.log), "mode": mode})),
            trials=total, output_dir=str(self.root / f"run-{self.counter}"),
            require_provider_usage=False,
        ).to_dict()

    def run_worker(self, request):
        source = self.root / f"request-{self.counter}.json"
        source.write_text(json.dumps(request))
        command = subprocess.run([sys.executable, "-m", "vrlbench.worker", str(source)],
            cwd=self.root, env=dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1"),
            capture_output=True, text=True, timeout=30)
        result = json.loads((Path(request["output_dir"]) / "worker_result.json").read_text())
        return command, result

    def events(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_real_loop_preserves_raw_context_and_canonical_failure_per_case(self):
        command, result = self.run_worker(self.request())
        self.assertEqual(command.returncode, 0, result)
        resets = [row for row in self.events() if row["kind"] == "reset"]
        self.assertEqual(len(resets), 2)
        self.assertNotEqual(resets[0]["identity"], resets[1]["identity"])
        self.assertEqual(resets[0]["config"]["total_trials"], 3)
        self.assertEqual(resets[0]["config"]["task_scope"], "custom task family")
        self.assertEqual(resets[0]["config"]["environment_rules"], "one actual click")
        for case in result["cases"]:
            first, second = case["trials"]
            self.assertEqual(first["system_context"], "")
            self.assertEqual(second["system_context"], RAW_CONTEXT)
            self.assertEqual(second["user_context"], "")
            self.assertTrue(second["success"])
            self.assertEqual(len(first["controller_state"]["records"]), 1)
            records = second["controller_state"]["records"]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["trajectory"], first["policy_trajectory"])
            self.assertTrue(records[0]["trajectory"].startswith("policy_ao_trace\n"))
            self.assertNotIn("private Actor", records[0]["trajectory"])
        contexts = [row for row in self.events() if row["kind"] == "context"]
        self.assertEqual([row["index"] for row in contexts], [0, 1, 0, 1])
        self.assertEqual(result["identity"]["plugins"][0]["role"], "memory_method")

    def test_success_stops_and_final_failure_keeps_its_executed_trace(self):
        for after in (1, 20):
            command, result = self.run_worker(self.request(total=1, cases=[str(after)], params={"success_after": after}))
            self.assertEqual(command.returncode, 0, result)
            trial = result["cases"][0]["trials"][0]
            self.assertEqual(len(result["cases"][0]["trials"]), 1)
            record = trial["controller_state"]["records"][0]
            if after == 20:
                self.assertEqual(record["trajectory"], trial["policy_trajectory"])
                self.assertEqual(record["actor_trajectory"], trial["actor_trajectory"])
                self.assertTrue(record["metadata"]["policy_evidence_eligible"])
                self.assertFalse(record["metadata"]["has_future_trial"])
            else:
                self.assertTrue(record["success"])

    def test_ineligible_trial_does_not_become_failure_experience(self):
        command, result = self.run_worker(self.request(cases=["a"], params={"ineligible_index": 0}))
        self.assertEqual(command.returncode, 0, result)
        first = result["cases"][0]["trials"][0]
        record = first["controller_state"]["records"][0]
        self.assertEqual(record["trajectory"], "")
        self.assertFalse(record["metadata"]["policy_evidence_eligible"])
        self.assertTrue(record["metadata"]["has_future_trial"])

    def test_invalid_context_and_nonempty_user_are_explicit_errors(self):
        for mode, expected in (("wrong_context", "must return ActorContext"),
                               ("user", "ActorContext.user must be empty"),
                               ("first_memory", "first trial must not contain")):
            command, result = self.run_worker(self.request(mode=mode, cases=[mode]))
            self.assertEqual(command.returncode, 1)
            self.assertIn(expected, result["error"]["message"])

    def test_binding_restores_all_global_hooks_after_factory_error(self):
        from vrlbench.backends import BackendRuntime, _METHOD_FACTORIES, memory_bindings
        from vrlbench.plugins import enable_reference_imports

        enable_reference_imports()
        from vrlbench.environments import webshop
        from vrlbench.runtime import controllers, framework, loop

        before = (controllers.make_controller, framework.make_controller, framework._controller_name,
                  webshop.make_controller, webshop._normalize_webshop_controller,
                  loop.run_episode, loop.prepare_shared_t1, framework.run_episode,
                  webshop.run_episode, webshop.prepare_shared_t1)
        request = self.request()
        request["benchmark"]["name"] = "webshop"
        request["memory"]["method_factory"] = {"entrypoint": f"{self.plugin}:wrong_factory", "kwargs": {}}
        with self.assertRaisesRegex(TypeError, "must return.*MemoryMethod"):
            with memory_bindings(request) as (er, controller, _):
                BackendRuntime(request, er, controller).make_controller()
        after = (controllers.make_controller, framework.make_controller, framework._controller_name,
                 webshop.make_controller, webshop._normalize_webshop_controller,
                 loop.run_episode, loop.prepare_shared_t1, framework.run_episode,
                 webshop.run_episode, webshop.prepare_shared_t1)
        self.assertEqual(before, after)
        self.assertEqual(_METHOD_FACTORIES, {})

    def test_singleton_whole_method_factory_is_rejected(self):
        request = self.request()
        request["memory"]["method_factory"] = {
            "entrypoint": f"{self.plugin}:create_singleton", "kwargs": {"log": str(self.log)}}
        command, result = self.run_worker(request)
        self.assertEqual(command.returncode, 1)
        self.assertIn("fresh instance for every evaluation case", result["error"]["message"])

    def test_bridge_implements_controller_protocol_and_caches_context(self):
        from vrlbench.backends import BackendRuntime, memory_bindings
        from vrlbench.plugins import enable_reference_imports

        enable_reference_imports()
        from vrlbench.runtime.controllers import InterEpisodeController

        request = self.request(cases=["one"])
        with memory_bindings(request) as (er, controller, _):
            bridge = BackendRuntime(request, er, controller).make_controller()
            self.assertIsInstance(bridge, InterEpisodeController)
            bridge.set_total_budget(3)
            first = bridge.build_trial_context(trial_index=0)
            second = bridge.build_trial_context(trial_index=0)
            self.assertEqual(first, second)
        self.assertEqual(len([event for event in self.events() if event["kind"] == "context"]), 1)

    def test_native_webshop_shared_first_trial_and_real_method_identity(self):
        self.plugin.write_text(self.plugin.read_text().replace(
            "class Environment:\n",
            "class Environment:\n    _vrlbench_environment_identity = "
            "{'benchmark': 'webshop', 'source_data_index_root_sha256': '" + "a" * 64 + "'}\n",
        ))
        baseline = self.request(cases=[0])
        baseline["memory"] = {"name": "retry", "factory": None}
        baseline["benchmark"] = {"name": "webshop", "cases": [0], "params": {},
            "factory": {"entrypoint": f"{self.plugin}:create_webshop_backend", "kwargs": {}}}
        baseline["model"]["transport"] = {"entrypoint": f"{self.plugin}:create_client", "kwargs": {}}
        command, source = self.run_worker(baseline)
        self.assertEqual(command.returncode, 0, source)
        self.assertEqual(source["identity"]["environment"], {
            "benchmark": "webshop", "source_data_index_root_sha256": "a" * 64,
        })
        # Native artifacts are sufficient for the original shared-T1 checks.
        request = self.request(cases=[0])
        request["benchmark"] = baseline["benchmark"]
        request["model"] = baseline["model"]
        request["shared_t1_dir"] = str(Path(baseline["output_dir"]) / "reference")
        command, result = self.run_worker(request)
        self.assertEqual(command.returncode, 0, result)
        self.assertEqual(result["identity"]["environment"], source["identity"]["environment"])
        case = result["cases"][0]
        self.assertEqual(case["trials"][0]["policy_trajectory"], source["cases"][0]["trials"][0]["policy_trajectory"])
        self.assertEqual(case["trials"][1]["system_context"], RAW_CONTEXT)
        self.assertTrue(case["trials"][-1]["success"])
        native = json.loads((Path(request["output_dir"]) / "reference" / "results.json").read_text())
        self.assertEqual(native["controller"], "community_joint_method")
        self.assertNotEqual(native["controller"], "dsst")
        reset = next(row for row in self.events() if row["kind"] == "reset")
        self.assertEqual(reset["config"]["metadata"]["benchmark"], "webshop")
        self.assertEqual(reset["config"]["metadata"]["sample_index"], 0)
        self.assertEqual(reset["config"]["metadata"]["max_environment_steps"], request["max_environment_steps"])
        self.assertEqual(reset["config"]["metadata"]["shared_t1_trial_metadata"],
                         source["cases"][0]["trials"][0]["trial_metadata"])
        self.assertEqual(reset["config"]["metadata"]["shared_t1_llm_usage"],
                         source["cases"][0]["trials"][0]["llm_usage"])

    def test_native_shared_terminal_failure_notifies_method_once(self):
        baseline = self.request(total=1, cases=[0])
        baseline["memory"] = {"name": "retry", "factory": None}
        baseline["benchmark"] = {"name": "webshop", "cases": [0], "params": {},
            "factory": {"entrypoint": f"{self.plugin}:create_webshop_backend", "kwargs": {}}}
        baseline["model"]["transport"] = {"entrypoint": f"{self.plugin}:create_client", "kwargs": {}}
        command, source = self.run_worker(baseline)
        self.assertEqual(command.returncode, 0, source)

        request = self.request(total=1, cases=[0])
        request["benchmark"] = baseline["benchmark"]
        request["model"] = baseline["model"]
        request["shared_t1_dir"] = str(Path(baseline["output_dir"]) / "reference")
        command, result = self.run_worker(request)
        self.assertEqual(command.returncode, 0, result)
        trial = result["cases"][0]["trials"][0]
        records = [row["record"] for row in self.events() if row["kind"] == "observe"]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["trial_index"], 0)
        self.assertEqual(records[0]["trajectory"], source["cases"][0]["trials"][0]["policy_trajectory"])
        self.assertEqual(records[0]["actor_trajectory"], trial["actor_trajectory"])
        self.assertTrue(records[0]["metadata"]["policy_evidence_eligible"])
        self.assertFalse(records[0]["metadata"]["has_future_trial"])

    def test_native_webshop_fresh_case_receives_metadata_and_verbatim_context(self):
        request = self.request(cases=[0])
        request["benchmark"] = {"name": "webshop", "cases": [0], "params": {},
            "factory": {"entrypoint": f"{self.plugin}:create_webshop_backend", "kwargs": {}}}
        request["model"]["transport"] = {"entrypoint": f"{self.plugin}:create_client", "kwargs": {}}
        command, result = self.run_worker(request)
        self.assertEqual(command.returncode, 0, result)
        first, second = result["cases"][0]["trials"]
        self.assertEqual(first["system_context"], "")
        self.assertEqual(second["system_context"], RAW_CONTEXT)
        reset = next(row for row in self.events() if row["kind"] == "reset")
        self.assertEqual(reset["config"]["metadata"]["benchmark"], "webshop")
        self.assertEqual(reset["config"]["metadata"]["sample_index"], 0)
        self.assertEqual(reset["config"]["environment_rules"], reset["config"]["metadata"]["basic_environment_rules"])

    def test_demo_allows_a_no_model_whole_method_but_disables_model_calls(self):
        for mode, status in (("normal", 0), ("model", 1)):
            request = self.request(mode=mode, cases=[mode])
            request["benchmark"] = {"name": "demo", "cases": [mode], "params": {}, "factory": None}
            command, result = self.run_worker(request)
            self.assertEqual(command.returncode, status, result)
            if status:
                self.assertEqual(result["error"]["type"], "DemoModelCallsDisabled")
            else:
                self.assertEqual(result["cases"][0]["trials"][1]["system_context"], RAW_CONTEXT)


if __name__ == "__main__":
    unittest.main()
