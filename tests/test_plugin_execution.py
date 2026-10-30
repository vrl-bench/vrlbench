"""Offline regressions for plugin imports and evaluation-process cleanup."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vrlbench import FactorySpec, MemorySpec, ModelSpec, SimulationSpec, WorkloadSpec, plan, simulate


class PluginExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, relative, source):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
        return path

    def run_python(self, source):
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(source)], cwd=self.root,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_local_module_is_visible_to_worker_without_importing_during_plan(self):
        marker = self.root / "imported"
        self.write("local_memory.py", f"""
            from pathlib import Path
            from vrlbench.plugins import load_factory
            Path({str(marker)!r}).write_text('worker imported plugin')
            def create():
                factory, _ = load_factory({{"entrypoint": {str(ROOT / "examples/memory_study/fixture.py") + ":create_memory"!r}}})
                return factory()
        """)
        spec = SimulationSpec(
            WorkloadSpec("teaching_route_control", [{"case_id": "case", "target": "blue"}],
                         factory=FactorySpec(f"{ROOT / 'examples/memory_study/fixture.py'}:create_backend")),
            ModelSpec("offline", transport=FactorySpec(
                f"{ROOT / 'examples/memory_study/fixture.py'}:create_no_model_client")),
            MemorySpec("local_memory", factory=FactorySpec("local_memory:create")),
            trials=2, max_environment_steps=1, output_dir=str(self.root / "module-run"), require_provider_usage=False,
        )
        with patch.object(sys, "path", [str(self.root), *sys.path]):
            self.assertEqual(plan(spec)["model_calls_executed"], 0)
            self.assertFalse(marker.exists())
            result = simulate(spec).require_complete()
        self.assertTrue(marker.exists())
        self.assertNotIn("local_memory", sys.modules)
        self.assertEqual(result.counts["completed"], 1)

    def test_same_directory_siblings_work_and_unused_conflicting_names_are_ignored(self):
        self.write("one/settings.py", "VALUE = 'one'\n")
        self.write("one/first.py", "from settings import VALUE\ndef create(): return VALUE\n")
        self.write("one/second.py", "from settings import VALUE\ndef create(): return VALUE\n")
        self.write("two/settings.py", "VALUE = 'unused'\n")
        self.write("two/third.py", "def create(): return 'three'\n")
        output = self.run_python("""
            from pathlib import Path
            from vrlbench.plugins import load_factory
            for path in ('one/first.py', 'one/second.py', 'two/third.py'):
                factory, _ = load_factory({'entrypoint': str(Path(path).resolve()) + ':create'})
                print(factory())
        """)
        self.assertEqual(output.splitlines(), ["one", "one", "three"])

    def test_deferred_sibling_conflict_fails_before_wrong_module_can_be_used(self):
        for name in ("one", "two"):
            self.write(f"{name}/settings.py", f"VALUE = {name!r}\n")
            self.write(f"{name}/candidate.py", """
                def create():
                    from settings import VALUE
                    return VALUE
            """)
        output = self.run_python("""
            from pathlib import Path
            from vrlbench.plugins import load_factory
            load_factory({'entrypoint': str(Path('one/candidate.py').resolve()) + ':create'})
            try:
                load_factory({'entrypoint': str(Path('two/candidate.py').resolve()) + ':create'})
            except ImportError as error:
                print(error)
            else:
                raise AssertionError('ambiguous deferred imports were accepted')
        """)
        self.assertIn("Conflicting local plugin import 'settings'", output)

    def test_transport_and_memory_sibling_conflict_is_a_structured_failure(self):
        self.write("transport/settings.py", "VALUE = 'transport'\n")
        self.write("memory/settings.py", "VALUE = 'memory'\n")
        transport = self.write("transport/client.py", """
            from settings import VALUE
            def create():
                raise AssertionError('the synthetic task must not call a model')
        """)
        memory = self.write("memory/candidate.py", """
            from settings import VALUE
            def create():
                raise AssertionError('the conflicting memory must not be initialized')
        """)
        spec = SimulationSpec(
            WorkloadSpec("teaching_route_control", [{"case_id": "case", "target": "blue"}],
                         factory=FactorySpec(f"{ROOT / 'examples/memory_study/fixture.py'}:create_backend")),
            ModelSpec("offline", transport=FactorySpec(f"{transport}:create")),
            MemorySpec("custom", factory=FactorySpec(f"{memory}:create")),
            trials=2, max_environment_steps=1, output_dir=str(self.root / "conflict-run"), require_provider_usage=False,
        )
        result = simulate(spec)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.to_dict()["error"]["type"], "ImportError")
        self.assertIn("Conflicting local plugin import 'settings'", result.to_dict()["error"]["message"])
        self.assertEqual(result.resources["provider_requests"]["requests_started"], 0)

    def test_unused_sibling_cannot_shadow_another_plugins_deferred_import(self):
        self.write("one/settings.py", "VALUE = 'one'\n")
        self.write("one/candidate.py", """
            def create():
                from settings import VALUE
                return VALUE
        """)
        self.write("two/settings.py", "VALUE = 'two'\n")
        self.write("two/candidate.py", "def create(): return 'unrelated'\n")
        output = self.run_python("""
            from pathlib import Path
            from vrlbench.plugins import load_factory
            first, _ = load_factory({'entrypoint': str(Path('one/candidate.py').resolve()) + ':create'})
            try:
                load_factory({'entrypoint': str(Path('two/candidate.py').resolve()) + ':create'})
            except ImportError as error:
                print(error)
            else:
                raise AssertionError('the new search root can shadow a declared deferred import')
            assert first() == 'one'
        """)
        self.assertIn("would shadow the deferred import", output)

    @unittest.skipUnless(os.name == "posix", "process-group cleanup requires POSIX")
    def test_all_worker_exit_paths_stop_only_their_own_descendants(self):
        plugin = self.write("helpers.py", """
            import json
            import pathlib
            import subprocess
            import sys
            import time
            class Backend:
                def __init__(self, mode):
                    self.mode = mode
                def run(self, spec, runtime):
                    code = ("import pathlib,signal,time; "
                            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                            "pathlib.Path('helper-started').touch(); time.sleep(20)")
                    child = subprocess.Popen([sys.executable, '-c', code])
                    pathlib.Path('helper.pid').write_text(str(child.pid))
                    while not pathlib.Path('helper-started').exists():
                        time.sleep(0.01)
                    if self.mode == 'failed':
                        raise RuntimeError('synthetic backend failure')
                    if self.mode == 'timeout':
                        time.sleep(20)
                    return [{'case_id':'case', 'task':'offline',
                             'trials':[{'trial':0, 'success':True}], 'resource_usage':{}}]
            def create(mode):
                return Backend(mode)
        """)
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        self.addCleanup(unrelated.wait)
        self.addCleanup(unrelated.terminate)
        for mode in ("complete", "failed", "timeout"):
            with self.subTest(mode=mode):
                output = self.root / mode
                result = simulate(SimulationSpec(
                    WorkloadSpec("offline", ["case"], factory=FactorySpec(f"{plugin}:create", {"mode": mode})),
                    ModelSpec("offline"), trials=1, max_environment_steps=15, output_dir=str(output),
                    timeout_seconds=4, require_provider_usage=False,
                ))
                self.assertEqual(result.status, mode)
                self.assertTrue((output / "helper-started").exists())
                helper_pid = int((output / "helper.pid").read_text())
                try:
                    state = subprocess.run(["ps", "-o", "stat=", "-p", str(helper_pid)],
                                           capture_output=True, text=True, check=False).stdout.strip()
                    self.assertTrue(not state or state.startswith("Z"), f"helper remains running: {state}")
                    self.assertIsNone(unrelated.poll())
                finally:
                    try:
                        os.kill(helper_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    unittest.main()
