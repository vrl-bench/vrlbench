"""Restart, pairing and cost invariants of the public per-case batch runner."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vrlbench import ActorSpec, FactorySpec, ModelSpec, SimulationSpec, WorkloadSpec
from vrlbench.batch import run_batch
from vrlbench.results import SimulationResult, aggregate_cases
from vrlbench.simulation import simulate

FIXTURE = Path(__file__).with_name('test_interactive.py').resolve()

def visual_session(*, case, params):
    from test_interactive import FakeSession, PNG
    return FakeSession(done=True, image=PNG)



@unittest.skipUnless(importlib.util.find_spec('openai'), 'requires runtime extra')
class BatchExecution(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.spec = SimulationSpec(WorkloadSpec('miniwob', [
            {'task': 'click-test', 'episode': 0}, {'task': 'click-test', 'episode': 1}],
            params={'session_factory': {'entrypoint': f'{Path(__file__).resolve()}:visual_session'}}),
            ModelSpec('fixture', transport=FactorySpec(f'{FIXTURE}:make_checkpoint_transport')),
            actor=ActorSpec('scripted', factory=FactorySpec(f'{FIXTURE}:make_actor')),
            trials=2, max_environment_steps=1, require_provider_usage=True,
            output_dir=str(self.root / 'unused'))

    def test_parallel_pairing_resume_and_new_call_accounting(self):
        root = self.root / 'batch'
        report = run_batch(self.spec, output_dir=root, jobs=2)
        self.assertEqual(report['status'], 'complete')
        baseline = SimulationResult.from_path(root / 'retry/evaluation.json').require_complete()
        child = SimulationResult.from_path(root / 'reflexion/evaluation.json').require_complete()
        self.assertEqual(baseline.counts['solved'], 0)
        from vrlbench.replay_viewer import replay_data
        replay = replay_data([root / 'retry', root / 'reflexion'])
        for run in replay['runs']:
            for case in run['cases']:
                for trial in case['trials']:
                    self.assertTrue(trial['initial_image'], trial['initial_snapshot_notice'])
        for left, right in zip(baseline.cases, child.cases):
            self.assertEqual(left['trials'][0]['policy_trajectory'], right['trials'][0]['policy_trajectory'])
            self.assertIn('shared_t1_source', right['trials'][0]['trial_metadata'])
        for result in (baseline, child):
            self.assertEqual(result.resources['api_calls'], result.resources['provider_requests']['requests_started'])
        records = {p: p.read_bytes() for p in root.glob('*/jobs/**/provider_requests.jsonl')}
        with patch('vrlbench.batch.simulate', side_effect=AssertionError('completed task failure rerolled')):
            self.assertEqual(run_batch(self.spec, output_dir=root, jobs=1, resume=True)['status'], 'complete')
        self.assertTrue(records)
        self.assertEqual(records, {p: p.read_bytes() for p in records})
        with self.assertRaisesRegex(ValueError, 'Cannot resume'):
            run_batch(replace(self.spec, model=replace(self.spec.model, name='different')),
                      output_dir=root, resume=True)

    def test_infrastructure_failure_retained_and_only_failed_job_restarted(self):
        root = self.root / 'recovery'
        fail = True
        def flaky(spec, **kwargs):
            nonlocal fail
            if fail:
                fail = False
                result = SimulationResult({'status':'failed', 'config':spec.to_dict(), 'cases':[],
                    'output_dir':spec.output_dir, **aggregate_cases([], spec.trials, 1)})
                result.write(Path(spec.output_dir) / 'evaluation.json')
                return result
            return simulate(spec, **kwargs)
        with patch('vrlbench.batch.simulate', side_effect=flaky):
            report = run_batch(self.spec, output_dir=root, jobs=1)
        self.assertEqual(report['status'], 'failed')
        self.assertIsNone(report['comparison'])
        originals = list(root.glob('retry/jobs/*/attempt-0000/evaluation.json'))
        self.assertEqual(len(originals), 2)
        calls = []
        def record(spec, **kwargs):
            calls.append(spec)
            return simulate(spec, **kwargs)
        with patch('vrlbench.batch.simulate', side_effect=record):
            recovered = run_batch(self.spec, output_dir=root, jobs=1, resume=True)
        self.assertEqual(recovered['status'], 'complete')
        self.assertEqual(len(calls), 2)  # failed Retry job plus its missing candidate
        self.assertEqual(len(list(root.glob('retry/jobs/*/attempt-0001/evaluation.json'))), 1)
        self.assertTrue(all(p.exists() for p in originals))

    def test_rejects_shared_vm_concurrency_before_creating_output(self):
        configured = replace(self.spec, workload=WorkloadSpec('osworld', []))
        with self.assertRaisesRegex(ValueError, 'Shared VM'):
            run_batch(configured, output_dir=self.root / 'blocked', jobs=2)
        self.assertFalse((self.root / 'blocked').exists())

    def test_cancelling_a_worker_stops_execution_and_keeps_an_incomplete_record(self):
        import threading
        import time
        backend = self.root / 'slow.py'
        backend.write_text('''import time
class Backend:
    def run(self, spec, runtime):
        time.sleep(60)
        raise AssertionError("cancelled work reached the end")
def create():
    return Backend()
''')
        cancelled = threading.Event()
        timer = threading.Timer(0.4, cancelled.set)
        timer.start()
        self.addCleanup(timer.cancel)
        configured = replace(self.spec, actor=ActorSpec(),
            workload=WorkloadSpec('slow', ['a'], factory=FactorySpec(f'{backend}:create')),
            output_dir=str(self.root / 'cancelled'), require_provider_usage=False)
        started = time.monotonic()
        result = simulate(configured, cancel_event=cancelled)
        self.assertEqual(result.status, 'cancelled')
        self.assertLess(time.monotonic() - started, 10)
        self.assertEqual(result.counts['completed'], 0)
        self.assertTrue((result.output_dir / 'evaluation.json').exists())
