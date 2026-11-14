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
from types import SimpleNamespace

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
    def test_responses_memory_preserves_images_usage_and_model_identity(self):
        from vrlbench.worker import _ClientProxy
        from vrlbench.provider_usage import ProviderUsageJournal, read_provider_usage
        from vrlbench.runtime import llm
        from vrlbench.runtime.budget import TokenUsageLedger
        from vrlbench.runtime.llm import ProviderIdentityMismatch
        calls = []
        def chat(**kwargs):
            raise AssertionError('memory must use the declared Responses API')
        def responses(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(id='resp-test', status='completed', output_text='A concrete plan',
                model='gpt-6.1-terra' if len(calls)==1 else 'gpt-6.1-terra-2026-10-26',
                usage=SimpleNamespace(input_tokens=11, output_tokens=7, total_tokens=18))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=chat)),
                                 responses=SimpleNamespace(create=responses))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            'VEX2_ALLOWED_RESPONSE_MODELS':'gpt-6.1-terra',
            'VEX2_ALLOWED_RESPONSE_MODEL_PREFIXES':''}):
            journal = ProviderUsageJournal(Path(tmp))
            proxy = _ClientProxy(lambda:client, {'api_mode':'responses', 'temperature':None,
                'reasoning_effort':'medium', 'capability_overrides':{'supports_vision':True}}, journal)
            ledger = TokenUsageLedger(require_provider_usage=True)
            with patch.object(llm, '_client', proxy), llm.token_budget_scope(ledger):
                self.assertEqual(llm.complete('Reflect', model='gpt-6.1-terra', max_tokens=4096,
                    images=[b'\x89PNG\r\n\x1a\noriginal']), 'A concrete plan')
                with self.assertRaises(ProviderIdentityMismatch):
                    llm.complete('Again', model='gpt-6.1-terra', max_tokens=4096)
            self.assertEqual(len(calls),2)
            self.assertEqual(calls[0]['max_output_tokens'],4096)
            self.assertEqual(calls[0]['reasoning'],{'effort':'medium'})
            self.assertEqual(calls[0]['input'][0]['content'][1]['type'],'input_image')
            self.assertNotIn('messages', calls[0])
            self.assertNotIn('capability_overrides', calls[0])
            self.assertNotIn('temperature', calls[0])
            self.assertEqual(ledger.total_tokens,36)
            usage=read_provider_usage(Path(tmp))
            self.assertEqual(usage['responses_received'],2)
            self.assertEqual(usage['total_tokens'],36)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.counter = 0

    def request(self, **overrides):
        self.counter += 1
        result = {
            "benchmark": {"name": "route_control", "cases": [
                {"case_id": "first", "target": "blue"}, {"case_id": "second", "target": "blue"}],
                "params": {}, "factory": {
                    "entrypoint": f"{ROOT}/examples/memory_study/fixture.py:create_backend", "kwargs": {}}},
            "memory": {"name": "feedback_memory", "factory": {
                "entrypoint": f"{ROOT}/examples/memory_study/fixture.py:create_memory", "kwargs": {}}},
            "model": {"name": "scripted", "base_url": None, "api_key_env": "OPENAI_API_KEY",
                      "provider": "openai_compatible", "timeout_seconds": 120,
                      "transport": {
                          "entrypoint": f"{ROOT}/examples/memory_study/fixture.py:create_no_model_client", "kwargs": {}},
                      "request_options": {}, "allowed_response_models": []},
            "trials": 2, "max_environment_steps": 1,
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
        for options in ({"model": "another-model"}, {"extra_body": {"messages": []}},
                        {"input": "replacement"}, {"extra_body": {"input": []}},
                        {"extra_body": {"instructions": "replace controlled system"}},
                        {"previous_response_id": "untracked-history"}):
            request = self.request()
            request["model"]["request_options"] = options
            command, result = self.run_worker(request)
            self.assertEqual(command.returncode, 1)
            self.assertEqual(result["error"]["type"], "ValueError")
            self.assertIn("protocol fields", result["error"]["message"])

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
                                "api_mode": "chat", "disable_vision": False,
                                "capability_overrides": {"supports_vision": True},
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
            for sdk_option in ('api_mode','disable_vision','capability_overrides'):
                self.assertNotIn(sdk_option,row)
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
                    return [{'case_id': case['case_id'], 'task': 'custom',
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
                    return [{'case_id': 'wrong-' + case['case_id'], 'task': 'custom',
                             'trials': [{'trial': 0, 'success': True}], 'resource_usage': {}}
                            for case in spec['benchmark']['cases']]
            def create(): return Backend()
        '''))
        request = self.request()
        request["benchmark"]["factory"] = {"entrypoint": f"{plugin}:create", "kwargs": {}}
        command, result = self.run_worker(request)
        self.assertEqual(command.returncode, 1)
        self.assertIn("case identities differ", result["error"]["message"])

    def test_backend_runtime_delivers_failure_feedback_to_the_next_action(self):
        command, result = self.run_worker(self.request())
        self.assertEqual(command.returncode, 0, result)
        for case in result["cases"]:
            first, second = case["trials"]
            self.assertEqual(json.loads(first["trajectory"])["action"], "amber")
            self.assertIn('Rejected controls JSON: ["amber"]', second["context"])
            self.assertEqual(json.loads(second["trajectory"])["action"], "blue")
            self.assertTrue(second["success"])

    def test_worker_clears_inherited_limits_but_keeps_explicit_runtime_env(self):
        plugin = self.root / "inspect_runtime.py"
        plugin.write_text(textwrap.dedent('''
            import os
            class Backend:
                def run(self, spec, runtime):
                    assert 'VEX2_ALLOWED_RESPONSE_MODELS' not in os.environ
                    assert 'VEX2_LLM_REASONING_EFFORT' not in os.environ
                    assert os.environ['VEX2_REQUIRE_SYSTEM_FINGERPRINT'] == '0'
                    return [{'case_id': case['case_id'], 'task': 'custom',
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
            from vrlbench.memory import ERSlot
            class Memory(ERSlot):
                def reset(self): pass
                def build_section(self): return ''
                def on_failure(self, trajectory, model): pass
            singleton = Memory()
            def create(): return singleton
        '''))
        request = self.request(memory={"name": "singleton", "factory": {
            "entrypoint": f"{plugin}:create", "kwargs": {}}})
        command, result = self.run_worker(request)
        self.assertEqual(command.returncode, 1)
        self.assertIn("fresh instance", result["error"]["message"])


if __name__ == "__main__":
    unittest.main()
