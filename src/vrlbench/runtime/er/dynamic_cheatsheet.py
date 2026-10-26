from __future__ import annotations
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from vrlbench.runtime.er.base import ERSlot
from vrlbench.runtime.llm import complete
_CHEATSHEET_SYSTEM = None
_SOURCE_REPOSITORY = 'https://github.com/suzgunmirac/dynamic-cheatsheet.git'
_SOURCE_COMMIT = '5cfe3c37e8e52b1d858d0f3df46e7f17c50991b9'
_UPSTREAM_ARTIFACTS: Mapping[str, str] = {'dynamic_cheatsheet/language_model.py': '2ffd61a587a3effa8bbfa981f302e1a531526a8977cf419ae8c23e4103d4ba35', 'dynamic_cheatsheet/utils/extractor.py': '72cfc4e4c02d47da11cecf9ebd01b8a2fb4d08f65e904515ac9220b02548dbb0'}
_CURATOR_PROMPT_PATH = Path(__file__).resolve().parent / 'source_prompts' / 'dynamic_cheatsheet_cumulative.txt'
_CURATOR_PROMPT_SHA256 = '2b93bd3347b4878fb50c0674cb642d3ba3939cf0442680a3978ab1449bfe3dc0'
_CURATOR_MAX_TOKENS = 4096
_CHEATSHEET_BLOCK_RE = re.compile('<cheatsheet>(.*?)</cheatsheet>', re.DOTALL)
_TASK_LINE_RE = re.compile('^[ \\t]*(?:Task|Instruction):[^\\r\\n]*', re.MULTILINE)
_CURATOR_FIELD_RE = re.compile('\\[\\[(PREVIOUS_CHEATSHEET|QUESTION|MODEL_ANSWER)\\]\\]')

def _curator_prompt_template() -> str:
    raw = _CURATOR_PROMPT_PATH.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != _CURATOR_PROMPT_SHA256:
        raise RuntimeError(f'Dynamic Cheatsheet curator prompt hash mismatch: expected {_CURATOR_PROMPT_SHA256}, got {actual}')
    return raw.decode('utf-8').strip()

def _extract_cheatsheet_response(text: str) -> tuple[str | None, dict[str, Any]]:
    raw = (text if isinstance(text, str) else '').strip()
    if '<cheatsheet>' not in raw:
        return (None, {'status': 'rejected'})
    remainder = raw.split('<cheatsheet>', 1)[1].strip()
    candidate = remainder.split('</cheatsheet>', 1)[0].strip()
    return (candidate, {'status': 'accepted'})

def _extract_task(trajectory: str) -> str:
    match = _TASK_LINE_RE.search(trajectory)
    if match:
        return match.group(0).strip()
    return 'Current task (text unavailable)'

def _task_from_metadata(metadata: Mapping[str, Any] | None) -> str | None:
    values = metadata or {}
    for key in ('instruction', 'task'):
        value = values.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None

def _render_curator_prompt(*, previous_cheatsheet: str, task: str, model_answer: str) -> str:
    fields = {'PREVIOUS_CHEATSHEET': previous_cheatsheet or '(empty)', 'QUESTION': task, 'MODEL_ANSWER': model_answer}
    return _CURATOR_FIELD_RE.sub(lambda match: fields[match.group(1)], _curator_prompt_template())

class DynamicCheatsheetAdapted(ERSlot):
    """Dynamic Cheatsheet adapted to same-task, failure-only VRL retries."""

    def __init__(self) -> None:
        self._cheatsheet = ''

    def _update_from_failure(self, trajectory: str, model: str, *, task: str, task_source: str) -> None:
        del task_source
        prompt = _render_curator_prompt(previous_cheatsheet=self._cheatsheet, task=task, model_answer=trajectory)
        raw = complete(prompt, model=model, system=_CHEATSHEET_SYSTEM, max_tokens=_CURATOR_MAX_TOKENS)
        candidate, _ = _extract_cheatsheet_response(raw)
        if candidate is not None:
            self._cheatsheet = candidate

    def on_failure(self, trajectory: str, model: str) -> None:
        self._update_from_failure(trajectory, model, task=_extract_task(trajectory), task_source='trajectory_canonical_line')

    def on_trial_end(self, trajectory: str, *, actor_trajectory: str | None=None, success: bool, model: str, trial_index: int, total_trials: int, metadata: dict[str, Any] | None=None) -> None:
        if success:
            self.on_success(trajectory, model)
            return
        writer_trajectory = actor_trajectory if isinstance(actor_trajectory, str) and actor_trajectory.strip() else trajectory
        metadata_task = _task_from_metadata(metadata)
        self._update_from_failure(writer_trajectory, model, task=metadata_task or _extract_task(writer_trajectory), task_source='on_trial_end_metadata' if metadata_task is not None else 'trajectory_canonical_line')

    def read_for_trial(self, *, trial_index: int, total_trials: int, model: str, past_trajectories: Sequence[str] | None=None, metadata: dict[str, Any] | None=None) -> str:
        if not self._cheatsheet:
            return ''
        return "CHEATSHEET:\n'''\n" + self._cheatsheet.strip() + "\n'''"

    def read_placement(self) -> str:
        return 'user_prefix'

    def build_section(self) -> str:
        return self.read_for_trial(trial_index=0, total_trials=0, model='')

    def reset(self) -> None:
        self._cheatsheet = ''
