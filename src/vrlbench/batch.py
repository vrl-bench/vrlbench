"""Bounded per-case evaluation, matched Retry trials, and restartable job artifacts."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys
from threading import Event
import time

from ._method_names import METHOD_NAMES
from .provider_usage import merge_provider_resources, read_provider_usage, TOKEN_FIELDS
from .results import SimulationResult, aggregate_cases, compare_results
from .simulation import _guard_output, simulate
from .specs import MemorySpec, SimulationSpec


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _fingerprint():
    """A resumed job must use the same installed harness, prompts and runtime."""
    root = Path(__file__).parent
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.rglob('*')) if p.is_file() and
             '__pycache__' not in p.parts and p.suffix in {'.py', '.txt', '.html'}}
    dependencies = {}
    for name in ('openhands-sdk', 'openai', 'packaging'):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    return {'source_sha256': hashlib.sha256(_canonical(files).encode()).hexdigest(),
            'python': list(sys.version_info[:3]), 'dependencies': dependencies}


def _write(path, value):
    # Reuse the same atomic writer as evaluation.json, including fsync.
    SimulationResult(value).write(path)


def _sum_journals(payloads):
    journals = [read_provider_usage(Path(p['output_dir'])) for p in payloads]
    if not all(journal is not None for journal in journals):
        return None
    result = {'basis': 'actual_sdk_requests_across_case_jobs',
              'journal_valid': all(j['journal_valid'] for j in journals),
              'provider_usage_complete': all(j['provider_usage_complete'] for j in journals)}
    for field in ('api_calls', *TOKEN_FIELDS, 'requests_started', 'responses_received',
                  'failed_requests', 'pending_requests', 'unknown_usage_calls'):
        values = [j.get(field) for j in journals]
        result[field] = sum(values) if all(type(v) is int for v in values) else None
    result['known_subtotals'] = {}
    for field in ('api_calls', *TOKEN_FIELDS):
        values = [j['known_subtotals'].get(field) for j in journals]
        known = [v for v in values if type(v) is int]
        result['known_subtotals'][field] = sum(known) if known else None
    return result


def _aggregate(spec, payloads, output, planned):
    cases = [case for payload in payloads for case in payload['cases']]
    aggregate = aggregate_cases(cases, spec.trials, planned_cases=planned)
    complete = len(payloads) == planned and all(p['status'] == 'complete' for p in payloads)
    complete = complete and aggregate['counts']['completed'] == planned
    aggregate['resources'] = merge_provider_resources(aggregate['resources'],
        _sum_journals(payloads), include_zero=spec.workload.factory is None)
    plugins = {_canonical(plugin): plugin for payload in payloads
               for plugin in payload.get('identity', {}).get('plugins', [])}
    identity = dict(payloads[0].get('identity', {})) if payloads else {}
    identity['plugins'] = list(plugins.values())
    result = SimulationResult({'schema_version': 1, 'status': 'complete' if complete else 'failed',
        'config': replace(spec, output_dir=str(output)).to_dict(), **aggregate, 'cases': cases,
        'identity': identity, 'output_dir': str(output),
        'error': None if complete else {'type': 'IncompleteBatch', 'message': 'Some case jobs did not complete'},
        'artifacts': {'case_jobs': [p['output_dir'] for p in payloads]}})
    result.write(output / 'evaluation.json')
    return result


def validate_batch(spec, methods, jobs):
    if type(jobs) is not int or jobs < 1:
        raise ValueError('jobs must be a positive integer')
    if not methods or len(set(methods)) != len(methods) or any(m not in METHOD_NAMES for m in methods):
        raise ValueError(f'methods must be distinct names from {METHOD_NAMES}')
    methods = ['retry', *(m for m in methods if m != 'retry')]
    if spec.workload.factory is None and spec.workload.name in {'osworld', 'webarena'} and jobs != 1:
        raise ValueError('Shared VM/site deployments require --jobs 1; provision independent deployments for concurrency')
    if spec.shared_t1_dir is not None or spec.memory != MemorySpec('retry'):
        raise ValueError('Batch input must be a memory-free Retry configuration without shared_t1_dir')
    return methods


def run_batch(spec: SimulationSpec, *, output_dir, methods=('retry', 'reflexion'),
              jobs=2, resume=False, on_progress=None):
    """Run each case in its own process; preserve failed attempts on resume.

    Retry runs first for each case. Other conditions consume that case's actual
    memory-free first trial. Completed task failures are final observations, not
    infrastructure failures to rerun until success.
    """
    methods = validate_batch(spec, methods, jobs)
    spec = spec.resolved()
    spec.validate()
    output = Path(output_dir).resolve()
    _guard_output(output)
    contract = {'schema_version': 1, 'config': replace(spec, output_dir=str(output)).to_dict(),
                'methods': methods, 'harness': _fingerprint()}
    # Include local plugin content in resume validation, not just its pathname.
    contract['local_plugins'] = {value: hashlib.sha256(Path(value.rsplit(':',1)[0]).read_bytes()).hexdigest()
        for value in _strings(spec.to_dict()) if '.py:' in value and Path(value.rsplit(':',1)[0]).is_file()}
    manifest = output / 'batch.json'
    if output.exists() and any(output.iterdir()):
        if not resume or not manifest.is_file():
            raise ValueError('Use a new batch directory or --resume with its existing batch.json')
        if json.loads(manifest.read_text()) != contract:
            raise ValueError('Cannot resume: configuration, methods, harness, or plugin source changed')
    elif resume:
        raise ValueError('--resume requires an existing batch')
    output.mkdir(parents=True, exist_ok=True)
    # Kernel lock releases on crash; it cannot become a stale lock after interruption.
    import fcntl
    with (output / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another process is using this batch directory') from exc
        # Recheck after acquiring the lock to exclude concurrent initialization.
        if manifest.exists() and json.loads(manifest.read_text()) != contract:
            raise ValueError('Batch contract changed during initialization')
        _write(manifest, contract)
        return _execute(spec, output, methods, jobs, on_progress)


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def _execute(spec, output, methods, jobs, on_progress):
    started = time.monotonic()
    cancelled = Event()
    def case_job(case):
        key = hashlib.sha256(_canonical(case).encode()).hexdigest()[:20]
        completed, source = {}, None
        for method in methods:
            if cancelled.is_set():
                break
            parent = output / method / 'jobs' / key
            previous = sorted(parent.glob('attempt-*'))
            attempt = previous[-1] if previous else None
            saved = attempt / 'evaluation.json' if attempt else None
            candidate = replace(spec, workload=replace(spec.workload, cases=[case]),
                memory=MemorySpec(method), shared_t1_dir=source)
            result = None
            if saved and saved.is_file():
                recorded = SimulationResult.from_path(saved)
                expected = replace(candidate, output_dir=str(attempt)).to_dict()
                if recorded.to_dict().get('config') != expected:
                    raise ValueError('Recorded job configuration differs from batch contract')
                try:
                    result = recorded.require_complete()
                except RuntimeError:
                    pass
            if result is None:
                attempt = parent / f'attempt-{len(previous):04d}'
                result = simulate(replace(candidate, output_dir=str(attempt)), cancel_event=cancelled)
            completed[method] = result.to_dict()
            if on_progress:
                on_progress({'case': case, 'method': method, 'status': result.status,
                             'counts': result.counts})
            if result.status != 'complete':
                break
            if method == 'retry':
                source = str(attempt)
        return completed
    results = []
    pool = ThreadPoolExecutor(max_workers=jobs)
    try:
        futures = [pool.submit(case_job, case) for case in spec.workload.cases]
        for future in as_completed(futures):
            results.append(future.result())
    except BaseException:
        cancelled.set()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    aggregates = []
    for method in methods:
        selected = [row[method] for row in results if method in row]
        selected.sort(key=lambda p: p['cases'][0]['case_id'] if p['cases'] else p['output_dir'])
        condition = replace(spec, memory=MemorySpec(method),
            shared_t1_dir=None if method == 'retry' else str(output / 'retry'))
        aggregates.append(_aggregate(condition, selected, output / method, len(spec.workload.cases)))
    complete = all(result.status == 'complete' for result in aggregates)
    report = {'schema_version': 1, 'status': 'complete' if complete else 'failed',
              'elapsed_seconds': time.monotonic() - started, 'output_dir': str(output),
              'conditions': {method: {'status': result.status, 'counts': result.counts,
                                     'metrics': result.metrics, 'resources': result.resources}
                             for method, result in zip(methods, aggregates)},
              'comparison': compare_results(aggregates) if complete else None,
              'execution_costs_including_previous_attempts': {
                  method: _sum_journals([{'output_dir': str(path)} for path in
                      sorted((output / method / 'jobs').glob('*/attempt-*'))])
                  for method in methods},
              'resume_policy': 'reuse complete jobs, retain and retry infrastructure failures; never reroll completed task failures'}
    _write(output / 'summary.json', report)
    return report
