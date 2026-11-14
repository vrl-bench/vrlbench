"""Provision an isolated, pinned MiniWoB environment without changing user projects."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import zipfile

from .environments.synapse_dependency import (
    SYNAPSE_SOURCE_COMMIT, SYNAPSE_SOURCE_URL, _runtime_overlay_root,
    synapse_dependency_contract, synapse_runtime_issues,
)

CHROME_VERSION = '146.0.7680.165'


def _run(args, log, *, env=None, timeout=900):
    try:
        completed = subprocess.run([str(v) for v in args], stdout=log, stderr=subprocess.STDOUT,
                                   env=env, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Setup command timed out; see {log.name}") from exc
    if completed.returncode:
        raise RuntimeError(f"Setup command failed ({completed.returncode}); see {log.name}")


def _browser(root, name, log):
    if platform.system() != 'Linux' or platform.machine() not in {'x86_64', 'AMD64'}:
        raise ValueError('Automatic browser setup supports Linux x86_64; supply --chrome and --chromedriver on other hosts')
    folder = root / 'browser'
    destination = folder / f'{name}-linux64' / name
    if not destination.is_file():
        folder.mkdir(exist_ok=True)
        archive = folder / f'{name}.zip'
        url = f'https://storage.googleapis.com/chrome-for-testing-public/{CHROME_VERSION}/linux64/{name}-linux64.zip'
        _run(['curl', '--fail', '--location', '--retry', '3', '--connect-timeout', '20',
              '--max-time', '600', '--output', archive, url], log)
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                if not (folder / member.filename).resolve().is_relative_to(folder.resolve()):
                    raise ValueError('Unsafe browser archive path')
            zipped.extractall(folder)
        archive.unlink()
        for executable in (destination, destination.parent / 'chrome_crashpad_handler'):
            if executable.exists():
                executable.chmod(executable.stat().st_mode | 0o111)
    return destination


def setup_miniwob(root, *, python=sys.executable, source=SYNAPSE_SOURCE_URL,
                  chrome=None, chromedriver=None):
    """Return a nonsecret environment manifest for subsequent suite/run commands."""
    synapse_dependency_contract()
    root = Path(root).expanduser().resolve()
    if (chrome is None) != (chromedriver is None):
        raise ValueError('Supply --chrome and --chromedriver together')
    marker = root / '.vrlbench-setup'
    if root.exists() and any(root.iterdir()) and not marker.is_file():
        raise ValueError(f'Refusing to provision a nonempty unmanaged directory: {root}')
    root.mkdir(parents=True, exist_ok=True)
    marker.write_text('miniwob-v1\n')
    with (root / 'setup.log').open('a') as log:
        _run([python, '-c', 'import sys; assert sys.version_info[:2] == (3,12), "Python 3.12 required"'], log)
        checkout = root / 'Synapse'
        if not (checkout / '.git').exists():
            checkout.mkdir(exist_ok=True)
            _run(['git', 'init', checkout], log)
        present = subprocess.run(['git', '-C', str(checkout), 'cat-file', '-e', SYNAPSE_SOURCE_COMMIT],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        if not present:
            for attempt in range(3):
                try:
                    _run(['git', '-C', checkout, '-c', 'http.version=HTTP/1.1', 'fetch',
                          '--depth', '1', source, SYNAPSE_SOURCE_COMMIT], log, timeout=180)
                    break
                except RuntimeError:
                    if attempt == 2:
                        raise
        if subprocess.run(['git', '-C', str(checkout), 'rev-parse', '--verify', 'HEAD'],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
            _run(['git', '-C', checkout, 'checkout', '--detach', SYNAPSE_SOURCE_COMMIT], log)
        head = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
        if head != SYNAPSE_SOURCE_COMMIT:
            raise ValueError('Provisioned Synapse source differs from the pinned contract; use a new setup root')
        # Recover a partially completed setup without overwriting user edits.
        for relative in synapse_dependency_contract()['runtime_overlay_file_sha256']:
            current = (checkout / relative).read_bytes()
            overlay = (_runtime_overlay_root() / relative).read_bytes()
            original = subprocess.check_output(['git', '-C', str(checkout), 'show', f'HEAD:{relative}'])
            if current not in (original, overlay):
                raise ValueError(f'Refusing to overwrite modified MiniWoB runtime: {relative}')
            shutil.copyfile(_runtime_overlay_root() / relative, checkout / relative)
        environment = root / 'venv'
        interpreter = environment / 'bin/python'
        if not interpreter.is_file():
            _run([python, '-m', 'venv', environment], log)
        lock = Path(__file__).parent / 'requirements/miniwob-py312.lock.txt'
        _run([interpreter, '-m', 'pip', 'install', '-r', lock], log)
        browser = Path(chrome).expanduser().resolve() if chrome else _browser(root, 'chrome', log)
        driver = Path(chromedriver).expanduser().resolve() if chromedriver else _browser(root, 'chromedriver', log)
        versions = {}
        for name, binary in [('chrome', browser), ('chromedriver', driver)]:
            versions[name] = subprocess.check_output([str(binary), '--version'], text=True, timeout=15).strip()
        # Selenium rejects mismatched major versions; fail during setup instead of after model billing starts.
        import re
        majors = [re.search(r'\d+\.', version).group().rstrip('.') for version in versions.values()]
        if majors[0] != majors[1]:
            raise ValueError('Chrome and ChromeDriver major versions differ')
        manifest = {'schema_version': 1, 'workload': 'miniwob',
                    'source_commit': head, 'browser_versions': versions,
                    'dependency_lock_sha256': hashlib.sha256(lock.read_bytes()).hexdigest(),
                    'environment_python': str(interpreter),
                    'runtime_env': {'SYNAPSE_ROOT': str(checkout), 'SE_CHROME': str(browser),
                                    'SE_CHROMEDRIVER': str(driver), 'PYTHONNOUSERSITE': '1'}}
        probe_env = {**os.environ, **manifest['runtime_env'],
                     'XDG_CACHE_HOME': str(root / 'cache'), 'XDG_CONFIG_HOME': str(root / 'config')}
        from .environments.import_path import isolated_package_path
        with isolated_package_path() as import_root:
            probe_env['PYTHONPATH'] = import_root
            _run([interpreter, '-c', '''
from vrlbench.environments.native_sessions import MiniWoBSession
s = MiniWoBSession({'task':'click-test','episode':0}, {'include_screenshot':True})
try:
    state = s.reset(trial_index=0)
    assert state.observation.text and state.observation.image
finally:
    s.close()
'''], log, env=probe_env)
        manifest['native_browser_reset_verified'] = True
        (root / 'environment.json').write_text(json.dumps(manifest, indent=2) + '\n')
        # A run can use this file through --environment; sourcing is optional.
        (root / 'activate.sh').write_text('\n'.join(f'export {key}={shlex.quote(value)}'
            for key, value in manifest['runtime_env'].items()) + '\n')
    return manifest


def apply_environment(spec, path):
    from dataclasses import replace
    manifest = json.loads(Path(path).read_text())
    if manifest.get('schema_version') != 1 or manifest.get('workload') != spec.workload.name:
        raise ValueError('Environment manifest does not match this workload')
    values = manifest.get('runtime_env')
    if not isinstance(values, dict):
        raise ValueError('Environment manifest requires runtime_env')
    updated = replace(spec, workload=replace(spec.workload,
        params={**spec.workload.params, 'environment_python': manifest['environment_python']}),
        runtime_env={**spec.runtime_env, **values})
    updated.validate()
    return updated
