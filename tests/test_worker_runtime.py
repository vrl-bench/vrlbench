"""Offline worker tests: real reference loop, synthetic tasks and transports."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vrlbench.plugins import load_factory  # noqa: E402

HAS_RUNTIME = importlib.util.find_spec("openai") is not None


class PluginTests(unittest.TestCase):
    def test_fast_same_size_rewrite_ignores_stale_bytecode(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidate.py"
            old_source = b"def create():\n    return 'old'\n"
            new_source = b"def create():\n    return 'new'\n"
            self.assertEqual(len(old_source), len(new_source))
            path.write_bytes(old_source)
            original_stat = path.stat()
            py_compile.compile(str(path), doraise=True,
                               invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
            previous, old_identity = load_factory({"entrypoint": f"{path}:create", "kwargs": {}})
            self.assertEqual(previous(), "old")
            path.write_bytes(new_source)
            os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            self.assertEqual(path.stat().st_size, original_stat.st_size)
            self.assertEqual(path.stat().st_mtime_ns, original_stat.st_mtime_ns)
            current, identity = load_factory({"entrypoint": f"{path}:create", "kwargs": {}})
            self.assertEqual(current(), "new")
            self.assertEqual(identity["source_sha256"], hashlib.sha256(new_source).hexdigest())
            self.assertNotEqual(identity["source_sha256"], old_identity["source_sha256"])

    def test_same_filename_is_content_isolated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for index in (1, 2):
                folder = root / str(index)
                folder.mkdir()
                path = folder / "strategy.py"
                path.write_text(f"def create(value=0):\n    return {index} + value\n")
                paths.append(path)
            first, identity = load_factory({"entrypoint": f"{paths[0]}:create", "kwargs": {}})
            second, _ = load_factory({"entrypoint": f"{paths[1]}:create", "kwargs": {}})
            self.assertEqual(first(value=5), 6)
            self.assertEqual(second(value=5), 7)
            self.assertEqual(identity["source_sha256"], hashlib.sha256(paths[0].read_bytes()).hexdigest())

    def test_bad_plugin_reference(self):
        with self.assertRaises(ValueError):
            load_factory({"entrypoint": "./strategy.py:create", "kwargs": {}})
        with self.assertRaises(TypeError):
            load_factory({"entrypoint": "json:__name__", "kwargs": {}})

    def test_reference_alias_is_lazy(self):
        script = "import sys; import vrlbench.memory; assert 'vex2' not in sys.modules"
        subprocess.run([sys.executable, "-c", script], check=True,
                       cwd=ROOT / "src",
                       env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))


@unittest.skipUnless(HAS_RUNTIME, "Install the runtime extra to execute reference-loop tests")
class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.counter = 0

    def request(self, **overrides):
        self.counter += 1
        result = {
            "benchmark": {"name": "demo", "cases": ["first", "second"], "params": {}, "factory": None},
            "memory": {"name": "demo_trace", "factory": {
                "entrypoint": "vrlbench.backends:create_demo_memory", "kwargs": {}}},
            "model": {"name": "scripted", "base_url": None, "api_key_env": "OPENAI_API_KEY",
                      "provider": "openai_compatible", "timeout_seconds": 120,
                      "transport": None, "request_options": {}, "allowed_response_models": []},
            "trials": 3, "max_environment_steps": 2,
            "output_dir": str(self.root / f"run-{self.counter}"),
            "shared_t1_dir": None, "timeout_seconds": 30, "max_total_tokens": None,
            "require_provider_usage": False, "runtime_env": {},
        }
        result.update(overrides)
        return result

    def run_worker(self, request):
        request_path = self.root / f"request-{self.counter}.json"
        request_path.write_text(json.dumps(request))
        environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
        environment.pop("OPENAI_API_KEY", None)
        command = subprocess.run([sys.executable, "-m", "vrlbench.worker", str(request_path)],
                                 cwd=self.root, env=environment, capture_output=True, text=True, timeout=30)
        path = Path(request["output_dir"]) / "worker_result.json"
        self.assertTrue(path.is_file(), command.stderr)
        return command, json.loads(path.read_text())

    def test_demo_uses_reference_loop_and_memory(self):
        command, result = self.run_worker(self.request())
        self.assertEqual(command.returncode, 0, result)
        self.assertEqual(result["status"], "complete")
        self.assertEqual([case["case_id"] for case in result["cases"]], ["first", "second"])
        for case in result["cases"]:
            first, second = case["trials"]
            self.assertFalse(first["success"])
            self.assertEqual(first["context"], "")
            self.assertTrue(second["success"])
            self.assertIn(first["trajectory"], second["context"])
            self.assertFalse(first["trial_metadata"]["task_performance_evaluated"])
            self.assertEqual(case["resource_usage"]["api_calls"], 0)

    def test_success_stop_and_final_failure_no_update(self):
        for after, total in ((1, 3), (10, 1)):
            request = self.request(trials=total)
            request["benchmark"]["params"] = {"success_after": after}
            command, result = self.run_worker(request)
            self.assertEqual(command.returncode, 0, result)
            self.assertTrue(all(len(case["trials"]) == 1 for case in result["cases"]))

    def test_each_case_gets_a_fresh_instance(self):
        plugin = self.root / "recording.py"
        log = self.root / "memory.jsonl"
        plugin.write_text(textwrap.dedent('''
            import json
            import uuid
            from pathlib import Path
            from vrlbench.memory import ERSlot
            class Memory(ERSlot):
                def __init__(self, path):
                    self.path = Path(path)
                    self.identity = str(uuid.uuid4())
                def reset(self):
                    self.state = []
                    with self.path.open('a') as output:
                        output.write(json.dumps({'identity': self.identity}) + '\\n')
                def on_failure(self, trajectory, model):
                    self.state.append(trajectory)
                def build_section(self):
                    return '\\n'.join(self.state)
            def create(path):
                return Memory(path)
        '''))
        request = self.request(memory={"name": "recording", "factory": {
            "entrypoint": f"{plugin}:create", "kwargs": {"path": str(log)}}})
        command, result = self.run_worker(request)
        self.assertEqual(command.returncode, 0, result)
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]["identity"], rows[1]["identity"])

    def test_protocol_fields_cannot_be_overridden(self):
        for options in ({"model": "another-model"}, {"extra_body": {"messages": []}}):
            request = self.request()
            request["model"]["request_options"] = options
            command, result = self.run_worker(request)
            self.assertEqual(command.returncode, 1)
            self.assertEqual(result["error"]["type"], "ValueError")
            self.assertIn("protocol fields", result["error"]["message"])

    def test_demo_cannot_accidentally_call_a_live_model(self):
        plugin = self.root / "requires_model.py"
        plugin.write_text(textwrap.dedent('''
            from vrlbench.memory import ERSlot
            class Memory(ERSlot):
                def reset(self): pass
                def build_section(self): return ''
                def on_failure(self, trajectory, model):
                    from vrlbench.runtime.llm import complete
                    complete(trajectory, model=model)
            def create(): return Memory()
        '''))
        request = self.request(memory={"name": "needs_model", "factory": {
            "entrypoint": f"{plugin}:create", "kwargs": {}}})
        command, result = self.run_worker(request)
        self.assertEqual(command.returncode, 1)
        self.assertEqual(result["error"]["type"], "DemoModelCallsDisabled")
        self.assertIn("disables model API calls", result["error"]["message"])

    def test_transport_mapping_and_memory_usage(self):
        plugin = self.root / "transport.py"
        calls = self.root / "requests.jsonl"
        plugin.write_text(textwrap.dedent('''
            import json
            from pathlib import Path
            from types import SimpleNamespace
            from vrlbench.memory import ERSlot
            class Completion:
                def __init__(self, path): self.path = path
                def create(self, **kwargs):
                    with Path(self.path).open('a') as output:
                        output.write(json.dumps(kwargs) + '\\n')
                    return SimpleNamespace(model='scripted-response',
                        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=5, total_tokens=12),
                        choices=[SimpleNamespace(message=SimpleNamespace(content='remember this failure'))])
            def create_client(path):
                return SimpleNamespace(chat=SimpleNamespace(completions=Completion(path)))
            class Memory(ERSlot):
                def reset(self): self.value = ''
                def build_section(self): return self.value
                def on_failure(self, trajectory, model):
                    from vrlbench.runtime.llm import complete
                    self.value = complete(trajectory, model=model)
            def create_memory(): return Memory()
        '''))
        request = self.request(memory={"name": "model_memory", "factory": {
            "entrypoint": f"{plugin}:create_memory", "kwargs": {}}})
        request["model"].update({
            "name": "DeepSeek-V4-Flash", "transport": {
                "entrypoint": f"{plugin}:create_client", "kwargs": {"path": str(calls)}},
            "request_options": {"seed": None, "temperature": 0.4, "max_tokens": 64,
                                "extra_body": {"example": True}},
            "allowed_response_models": ["scripted-response"],
        })
        command, result = self.run_worker(request)
        self.assertEqual(command.returncode, 0, result)
        rows = [json.loads(line) for line in calls.read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["model"], "DeepSeek-V4-Flash")
            self.assertNotIn("seed", row)
            self.assertEqual(row["max_tokens"], 64)
            self.assertEqual(row["extra_body"], {"thinking": {"type": "disabled"}, "example": True})
        for case in result["cases"]:
            self.assertEqual(case["resource_usage"]["api_calls"], 1)
            self.assertEqual(case["resource_usage"]["total_tokens"], 12)
            self.assertTrue(case["resource_usage"]["provider_usage_complete"])

    def test_custom_backend_and_record_validation(self):
        plugin = self.root / "backend.py"
        plugin.write_text(textwrap.dedent('''
            class Backend:
                def run(self, spec, runtime):
                    return [{'case_id': str(case), 'task': 'custom',
                             'trials': [{'trial': 0, 'success': True}], 'resource_usage': {}}
                            for case in spec['benchmark']['cases']]
            def create(): return Backend()
        '''))
        request = self.request()
        request["benchmark"]["factory"] = {"entrypoint": f"{plugin}:create", "kwargs": {}}
        command, result = self.run_worker(request)
        self.assertEqual(command.returncode, 0, result)
        self.assertEqual(result["cases"][0]["task"], "custom")

    def test_same_count_with_wrong_case_id_fails(self):
        plugin = self.root / "wrong_cases.py"
        plugin.write_text(textwrap.dedent('''
            class Backend:
                def run(self, spec, runtime):
                    return [{'case_id': 'wrong-' + str(case), 'task': 'custom',
                             'trials': [{'trial': 0, 'success': True}], 'resource_usage': {}}
                            for case in spec['benchmark']['cases']]
            def create(): return Backend()
        '''))
        request = self.request()
        request["benchmark"]["factory"] = {"entrypoint": f"{plugin}:create", "kwargs": {}}
        command, result = self.run_worker(request)
        self.assertEqual(command.returncode, 1)
        self.assertIn("case identities differ", result["error"]["message"])

    def test_backend_runtime_runs_fresh_memory_through_the_reference_loop(self):
        plugin = self.root / "loop_backend.py"
        plugin.write_text(textwrap.dedent('''
            from dataclasses import asdict
            class Backend:
                def run(self, spec, runtime):
                    cases = []
                    for case in spec['benchmark']['cases']:
                        count = [0]
                        def actor():
                            count[0] += 1
                            return 'executed failure' if count[0] == 1 else 'executed success'
                        episode = runtime.run_episode(actor, lambda trace: trace.endswith('success'))
                        cases.append({'case_id': str(case), 'task': 'plugin-task',
                                      'trials': [asdict(trial) for trial in episode.trials],
                                      'resource_usage': episode.resource_usage})
                    return cases
            def create(): return Backend()
        '''))
        request = self.request()
        request["benchmark"]["factory"] = {"entrypoint": f"{plugin}:create", "kwargs": {}}
        command, result = self.run_worker(request)
        self.assertEqual(command.returncode, 0, result)
        for case in result["cases"]:
            self.assertIn("executed failure", case["trials"][1]["context"])

    def test_sdk_shared_source_requires_matching_settings_and_selection(self):
        from vrlbench.backends import _validate_sdk_shared_source

        source = self.root / "baseline"
        source.mkdir()
        baseline = self.request(memory={"name": "retry", "factory": None})
        baseline["benchmark"] = {"name": "webshop", "cases": [0], "params": {}, "factory": None}
        (source / "request.json").write_text(json.dumps(baseline))
        (source / "evaluation.json").write_text(json.dumps({"status": "complete"}))
        comparison = dict(baseline, output_dir=str(self.root / "comparison"), shared_t1_dir=str(source),
                          memory={"name": "reflexion", "factory": None})
        _validate_sdk_shared_source(comparison)
        for field, value in (("trials", 5), ("max_environment_steps", 1), ("model", {"name": "other"})):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                _validate_sdk_shared_source({**comparison, field: value})
        wrong_cases = {**comparison, "benchmark": {**comparison["benchmark"], "cases": [1]}}
        with self.assertRaisesRegex(ValueError, "case selection"):
            _validate_sdk_shared_source(wrong_cases)

    def test_real_webshop_runner_with_scripted_environment_and_shared_transport(self):
        plugin = self.root / "synthetic_webshop.py"
        calls = self.root / "webshop_requests.jsonl"
        plugin.write_text(textwrap.dedent('''
            import json
            from pathlib import Path
            from types import SimpleNamespace
            from vrlbench.memory import ERSlot
            class Completion:
                def __init__(self, path): self.path = path
                def create(self, **kwargs):
                    with Path(self.path).open('a') as output:
                        output.write(json.dumps(kwargs) + '\\n')
                    prompt = kwargs['messages'][-1]['content']
                    answer = 'remember synthetic failure' if prompt.startswith('write memory:') else 'click[buy now]'
                    return SimpleNamespace(model='scripted',
                        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=5, total_tokens=12),
                        choices=[SimpleNamespace(message=SimpleNamespace(content=answer))])
            def create_client(path):
                return SimpleNamespace(chat=SimpleNamespace(completions=Completion(path)))
            class Memory(ERSlot):
                def reset(self): self.value = ''
                def build_section(self): return self.value
                def on_failure(self, trajectory, model):
                    from vrlbench.runtime.llm import complete
                    self.value = complete('write memory:' + trajectory, model=model)
            def create_memory(): return Memory()
            class Environment:
                def __init__(self): self.resets = 0
                def step(self, session, action):
                    if action == 'reset':
                        self.resets += 1
                        return 'Instruction: Buy synthetic sofa\\nAvailable actions: buy now', 0.0, False
                    return 'Synthetic purchase outcome', 1.0 if self.resets == 2 else 0.0, True
            class Backend:
                def run(self, spec, runtime):
                    from vrlbench.environments import webshop
                    webshop._load_default_env_factory = lambda: Environment
                    return runtime.run_reference()
            def create_backend(): return Backend()
        '''))
        request = self.request(memory={"name": "scripted_writer", "factory": {
            "entrypoint": f"{plugin}:create_memory", "kwargs": {}}})
        request["benchmark"] = {"name": "webshop", "cases": [0], "params": {},
                                "factory": {"entrypoint": f"{plugin}:create_backend", "kwargs": {}}}
        request["model"]["transport"] = {"entrypoint": f"{plugin}:create_client", "kwargs": {"path": str(calls)}}
        request["require_provider_usage"] = False
        command, result = self.run_worker(request)
        self.assertEqual(command.returncode, 0, result)
        case = result["cases"][0]
        self.assertEqual(case["case_id"], "sample_0000")
        self.assertEqual([trial["success"] for trial in case["trials"]], [False, True])
        self.assertIn("remember synthetic failure", case["trials"][1]["context"])
        self.assertEqual(case["resource_usage"]["api_calls"], 3)
        self.assertEqual(case["resource_usage"]["total_tokens"], 36)
        rows = [json.loads(line) for line in calls.read_text().splitlines()]
        self.assertEqual(len(rows), 3)
        self.assertIn("remember synthetic failure", rows[2]["messages"][0]["content"])

    def test_public_and_legacy_names_execute_the_same_native_method_through_sdk_and_cli(self):
        from vrlbench import SimulationSpec, simulate

        plugin = self.root / "reference_names.py"
        plugin.write_text(textwrap.dedent('''
            import json
            from pathlib import Path
            from types import SimpleNamespace
            class Completion:
                def __init__(self, path, responses):
                    self.path, self.responses = Path(path), iter(responses)
                def create(self, **kwargs):
                    with self.path.open('a') as output:
                        output.write(json.dumps(kwargs) + '\\n')
                    return SimpleNamespace(model='scripted',
                        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=5, total_tokens=12),
                        choices=[SimpleNamespace(message=SimpleNamespace(content=next(self.responses)))])
            def create_client(path, responses):
                return SimpleNamespace(chat=SimpleNamespace(completions=Completion(path, responses)))
            class Environment:
                def __init__(self): self.resets = 0
                def step(self, session, action):
                    if action == 'reset':
                        self.resets += 1
                        return 'Instruction: Buy synthetic sofa\\nAvailable actions: buy now', 0.0, False
                    return 'Synthetic purchase outcome', 1.0 if self.resets == 2 else 0.0, True
            class Backend:
                def run(self, spec, runtime):
                    from vrlbench.environments import webshop
                    webshop._load_default_env_factory = lambda: Environment
                    return runtime.run_reference()
            def create_backend(): return Backend()
        '''))
        raw = "  Joint raw reference response\n<policy> α & β </policy>\n\n"
        curator = json.dumps({"reasoning": "Synthetic curation", "operations": [
            {"type": "ADD", "section": "verification_checklist", "content": "Verify the sofa"}]})
        conditions = (
            ("retry", "retry", "none", "none", []),
            ("reflexion", "reflexion", "reflexion", "none", ["Remember the failure"]),
            ("dc_cu", "dynamic_cheatsheet_adapted", "dynamic_cheatsheet_adapted", "none",
             ["<cheatsheet>Verify the sofa</cheatsheet>"]),
            ("ace", "ace_adapted", "ace_adapted", "none", ["Synthetic reflection", curator]),
            ("vex2", "dsst", "none", "dsst", [raw]),
        )
        digest = hashlib.sha256(plugin.read_bytes()).hexdigest()
        for public, legacy, er, controller, updates in conditions:
            snapshots = []
            entrances = [("sdk", public), ("cli", legacy)]
            if public != legacy:
                entrances.append(("cli", public))
            for entrance, name in entrances:
                with self.subTest(entrance=entrance, name=name):
                    calls = self.root / f"calls-{self.counter}.jsonl"
                    responses = ["click[buy now]", *updates, "click[buy now]"]
                    request = self.request(memory={"name": name, "factory": None})
                    request["benchmark"] = {"name": "webshop", "cases": [0], "params": {},
                        "factory": {"entrypoint": f"{plugin}:create_backend", "kwargs": {}}}
                    request["model"]["transport"] = {"entrypoint": f"{plugin}:create_client",
                        "kwargs": {"path": str(calls), "responses": responses}}
                    spec = SimulationSpec.from_dict(request)
                    spec.validate()
                    if entrance == "sdk":
                        result = simulate(spec).require_complete().to_dict()
                    else:
                        path = self.root / "cli-method.json"
                        config = spec.to_dict()
                        config["workload"] = config.pop("benchmark")
                        path.write_text(json.dumps(config))
                        command = subprocess.run([sys.executable, "-m", "vrlbench", "simulate", "--config", str(path)],
                            cwd=self.root, env=dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1"),
                            capture_output=True, text=True, timeout=30)
                        self.assertEqual(command.returncode, 0, command.stderr + command.stdout)
                        result = json.loads((Path(request["output_dir"]) / "evaluation.json").read_text())
                    output = Path(request["output_dir"])
                    self.assertEqual(result["config"]["memory"], {"name": name, "factory": None})
                    self.assertEqual(json.loads((output / "request.json").read_text())["memory"]["name"], name)
                    native = json.loads((output / "reference" / "results.json").read_text())
                    self.assertEqual((native["er"], native["controller"]), (er, controller))
                    self.assertEqual(native["trials"], request["trials"])
                    self.assertEqual(native["max_environment_steps"], request["max_environment_steps"])
                    case = result["cases"][0]
                    self.assertEqual([trial["success"] for trial in case["trials"]], [False, True])
                    self.assertEqual(case["trials"][0]["context"], "")
                    self.assertEqual(case["resource_usage"]["api_calls"], len(responses))
                    self.assertEqual(case["resource_usage"]["total_tokens"], len(responses) * 12)
                    if public == "vex2":
                        self.assertEqual(case["trials"][1]["system_context"], raw)
                    for identity in result["identity"]["plugins"]:
                        self.assertEqual(identity["source_sha256"], digest)
                    rows = [json.loads(line) for line in calls.read_text().splitlines()]
                    self.assertEqual(len(rows), len(responses))
                    if public == "vex2":
                        payload = json.loads(rows[1]["messages"][-1]["content"])
                        self.assertEqual(payload["remaining_complete_attempts"], request["trials"] - 1)
                    snapshots.append((case, rows))
            self.assertTrue(all(snapshot == snapshots[0] for snapshot in snapshots))

    def test_parent_process_state_is_unchanged(self):
        before_path = list(sys.path)
        before_modules = set(sys.modules)
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://parent.invalid", "WORKER_MARKER": "parent"}):
            request = self.request(runtime_env={"WORKER_MARKER": "child"})
            command, result = self.run_worker(request)
            self.assertEqual(command.returncode, 0, result)
            self.assertEqual(os.environ["WORKER_MARKER"], "parent")
            self.assertEqual(os.environ["OPENAI_BASE_URL"], "https://parent.invalid")
        self.assertEqual(sys.path, before_path)
        self.assertEqual(set(sys.modules), before_modules)

    def test_worker_clears_inherited_limits_but_keeps_explicit_runtime_env(self):
        plugin = self.root / "inspect_runtime.py"
        plugin.write_text(textwrap.dedent('''
            import os
            class Backend:
                def run(self, spec, runtime):
                    assert 'VEX2_ALLOWED_RESPONSE_MODELS' not in os.environ
                    assert 'VEX2_LLM_REASONING_EFFORT' not in os.environ
                    assert os.environ['VEX2_REQUIRE_SYSTEM_FINGERPRINT'] == '0'
                    return [{'case_id': str(case), 'task': 'custom',
                             'trials': [{'trial': 0, 'success': True}], 'resource_usage': {}}
                            for case in spec['benchmark']['cases']]
            def create(): return Backend()
        '''))
        request = self.request(runtime_env={"VEX2_REQUIRE_SYSTEM_FINGERPRINT": "0"})
        request["benchmark"]["factory"] = {"entrypoint": f"{plugin}:create", "kwargs": {}}
        with patch.dict(os.environ, {"VEX2_ALLOWED_RESPONSE_MODELS": "old-model",
                                   "VEX2_LLM_REASONING_EFFORT": "old-effort",
                                   "VEX2_REQUIRE_SYSTEM_FINGERPRINT": "1"}):
            command, result = self.run_worker(request)
        self.assertEqual(command.returncode, 0, result)

    def test_invalid_factory_is_structured_failure(self):
        request = self.request(memory={"name": "bad", "factory": {
            "entrypoint": "builtins:object", "kwargs": {}}})
        command, result = self.run_worker(request)
        self.assertEqual(command.returncode, 1)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["type"], "TypeError")
        self.assertEqual(result["cases"], [])

    def test_singleton_memory_factory_is_rejected(self):
        plugin = self.root / "singleton.py"
        plugin.write_text(textwrap.dedent('''
            from vrlbench.backends import create_demo_memory
            singleton = create_demo_memory()
            def create(): return singleton
        '''))
        request = self.request(memory={"name": "singleton", "factory": {
            "entrypoint": f"{plugin}:create", "kwargs": {}}})
        command, result = self.run_worker(request)
        self.assertEqual(command.returncode, 1)
        self.assertIn("fresh instance", result["error"]["message"])


if __name__ == "__main__":
    unittest.main()
