from __future__ import annotations
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from vrlbench.runtime.er.base import ERSlot
from vrlbench.runtime.llm import complete
from vrlbench.runtime.trajectory import parse_policy_trajectory
_SOURCE_REPOSITORY = 'https://github.com/noahshinn/reflexion.git'
_SOURCE_COMMIT = '218cf0ef1df84b05ce379dd4a8e47f17766733a0'
_VISIBILITY_WINDOW = None
_VISIBILITY_PROTOCOL = 'all_failure_reflections'
_UPSTREAM_REFLECTION_MAX_TOKENS = 256
_EVALUATION_REFLECTION_MAX_TOKENS = _UPSTREAM_REFLECTION_MAX_TOKENS
_SUPPORTED_BENCHMARKS = frozenset({'miniwob', 'webshop', 'webarena', 'osworld'})
_PROMPT_DIR = Path(__file__).resolve().parent / 'source_prompts'
_PROMPT_ARTIFACTS: Mapping[str, tuple[str, str]] = {'miniwob_writer': ('reflexion_miniwob_writer.txt', '6445984018bb794c35d004138e488e11c89c37c2bf2a12e48924ddc68d3cd5b8'), 'webshop_writer': ('reflexion_webshop_writer.txt', '495f3492da8216d3a4ef14ee81945924a967ef266b99b2e00dc3e7e50a6152f2'), 'webshop_few_shot': ('reflexion_webshop_few_shot.txt', 'c0bd69af24fecc4d0b30f92e58af1e4f86320eb227b7002584bb04a88e913621')}
_UPSTREAM_ARTIFACTS: Mapping[str, str] = {'webshop_runs/generate_reflections.py': 'a01282c0e9079881e101aefbc249c540a8205d85817f1ca68b7d85e36a72498a', 'webshop_runs/reflection_few_shot_examples.txt': 'c0bd69af24fecc4d0b30f92e58af1e4f86320eb227b7002584bb04a88e913621', 'webshop_runs/env_history.py': '33c97e83ab70b713de1e85c9fb87395e5c71366c31ed2a53081dcab49f0e852b', 'alfworld_runs/generate_reflections.py': 'd7fbe4af7ff21ce9bc73ad2bcf16bf265682d1fb3554dc62985f3ada09997ef7', 'alfworld_runs/reflexion_few_shot_examples.txt': '5040995a3cdc88bce30fa26bec3ca4875ced8db98b7b2859f5c1614deec716e7', 'alfworld_runs/env_history.py': 'b41ee4eacbdb6f98b136f88ae2051de4bb65b583742c952ff100230090c87d0e'}

def _load_prompt_artifact(name: str) -> str:
    filename, expected_hash = _PROMPT_ARTIFACTS[name]
    path = _PROMPT_DIR / filename
    raw = path.read_bytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError(f'Reflexion prompt hash mismatch for {filename}: expected {expected_hash}, got {actual_hash}')
    return raw.decode('utf-8').strip()

def _benchmark_from_metadata(metadata: Mapping[str, Any] | None) -> str:
    benchmark = str((metadata or {}).get('benchmark') or '').strip().lower()
    if benchmark not in _SUPPORTED_BENCHMARKS:
        supported = ', '.join(sorted(_SUPPORTED_BENCHMARKS))
        raise ValueError(f'Reflexion requires explicit benchmark metadata and only supports the online families {{{supported}}}; got {benchmark!r}.')
    return benchmark

def _memory_block(entries: Sequence[str], *, writer: bool) -> str:
    if not entries:
        return ''
    if writer:
        block = '\n\nPlans from past attempts:\n'
        for index, reflection in enumerate(entries):
            block += f'Trial #{index}: {reflection.strip()}\n'
        return block.rstrip()
    block = 'Your memory for the task below:'
    for index, reflection in enumerate(entries):
        block += f'\nTrial {index}:\n{reflection.strip()}'
    return block

def _visible_entries(entries: Sequence[str]) -> Sequence[str]:
    return entries

def _source_interaction_log(benchmark: str, trajectory: str, *, actor_trajectory: str | None=None) -> str:
    """Render the interaction history exposed by the benchmark-native Actor."""
    evidence = parse_policy_trajectory(trajectory)
    if evidence is None:
        raise ValueError('Reflexion writer requires canonical executed A-O evidence')
    if benchmark == 'miniwob':
        source_log = str(actor_trajectory or '').strip()
        if not source_log:
            raise ValueError('MiniWoB Reflexion writer requires the source-native Actor decision-block trajectory')
        return f'{source_log}\n\nSTATUS: FAIL'
    elif benchmark in {'webshop', 'webarena', 'osworld'}:
        lines = [evidence['task'], 'Observation:', evidence['initial_observation']]
    else:
        raise AssertionError(f'unrouted Reflexion benchmark: {benchmark}')
    for step in evidence['steps']:
        lines.extend([f"> {step['action']}", 'Observation:', step['observation']])
    lines.append('STATUS: FAIL')
    return '\n'.join(lines).strip()

def _writer_prompt(benchmark: str, source_log: str, entries: Sequence[str]) -> str:
    if benchmark == 'webshop':
        template = _load_prompt_artifact('webshop_writer')
        template = template.replace('[[FEW_SHOT_EXAMPLES]]', _load_prompt_artifact('webshop_few_shot'))
        scenario = source_log
    elif benchmark == 'miniwob':
        template = _load_prompt_artifact('miniwob_writer')
        scenario = source_log
    elif benchmark in {'webarena', 'osworld'}:
        # A domain-neutral adaptation, not an upstream benchmark prompt.
        return ('The following trial failed. Reflect briefly on the executed actions and '
                'observed feedback, and write a concrete plan for the next attempt at the '
                'same task. Do not assume unobserved states or a hidden solution.\n\n'
                + source_log + _memory_block(entries, writer=True) + '\n\nReflection:')
    else:
        raise AssertionError(f'unrouted Reflexion benchmark: {benchmark}')
    prompt = template.replace('[[TRAJECTORY]]', scenario).replace('[[MEMORY_BLOCK]]', _memory_block(entries, writer=True))
    unresolved = re.findall('\\[\\[[A-Z_]+\\]\\]', prompt)
    if unresolved:
        raise RuntimeError(f'Unresolved Reflexion prompt fields: {unresolved}')
    return prompt

def _reflection_text(raw: str) -> str:
    """Preserve the official writer's direct append semantics."""
    return str(raw or '').strip()

class Reflexion(ERSlot):
    """Failure reflection with native or explicitly adapted writer prompts."""

    def __init__(self) -> None:
        self._entries = []
        self._benchmark = None

    def _bind_benchmark(self, metadata: Mapping[str, Any] | None) -> str:
        benchmark = _benchmark_from_metadata(metadata)
        if self._benchmark is not None and self._benchmark != benchmark:
            raise ValueError(f'A Reflexion instance cannot cross benchmark families: bound to {self._benchmark!r}, received {benchmark!r}.')
        self._benchmark = benchmark
        return benchmark

    def on_failure(self, trajectory: str, model: str) -> None:
        raise ValueError("Reflexion requires on_trial_end(..., metadata={'benchmark': workload_name}) so prompt routing is explicit.")

    def on_trial_end(self, trajectory: str, *, actor_trajectory: str | None=None, success: bool, model: str, trial_index: int, total_trials: int, metadata: dict[str, Any] | None=None) -> None:
        benchmark = self._bind_benchmark(metadata)
        if success:
            return
        if not bool((metadata or {}).get('has_future_trial', trial_index + 1 < total_trials)):
            return
        source_log = _source_interaction_log(benchmark, trajectory, actor_trajectory=actor_trajectory)
        prompt = _writer_prompt(benchmark, source_log, _visible_entries(self._entries))
        raw = complete(prompt, model=model, max_tokens=_EVALUATION_REFLECTION_MAX_TOKENS)
        self._entries.append(_reflection_text(raw))

    def read_for_trial(self, *, trial_index: int, total_trials: int, model: str, past_trajectories: Sequence[str] | None=None, metadata: dict[str, Any] | None=None) -> str:
        self._bind_benchmark(metadata)
        return self.build_section()

    def read_placement(self) -> str:
        return 'user_prefix'

    def build_section(self) -> str:
        if not self._entries:
            return ''
        return _memory_block(_visible_entries(self._entries), writer=False)

    def reset(self) -> None:
        self._entries.clear()
        self._benchmark = None
