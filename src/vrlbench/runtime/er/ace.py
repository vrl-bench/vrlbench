from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from vrlbench.runtime.er.base import ERSlot
from vrlbench.runtime.llm import complete
_SOURCE_REPOSITORY = 'https://github.com/ace-agent/ace.git'
_SOURCE_COMMIT = 'bcb7cea0504afad6f55fec4845dd4864c9f9eee7'
_INTERACTIVE_SOURCE_REPOSITORY = 'https://github.com/ace-agent/ace-appworld.git'
_INTERACTIVE_SOURCE_COMMIT = '9f3e92155345a9159f3a8b25abc334eeca05b545'
_SOURCE_PAPER = 'https://arxiv.org/pdf/2510.04618v3'
_SOURCE_PAPER_SHA256 = '51050ced82df75c143b151262d5af8763916968ca50374bd8ff778f40552b0ad'
_UPSTREAM_ARTIFACTS: Mapping[str, str] = {'experiments/code/ace/adaptation_agent.py': 'ed870059788cecc99e37ceb80a2ff133c0b5815c706be300abfdf1d3f6863432', 'experiments/code/ace/adaptation_react.py': '4d714ab65205255842ffb6ce120e51e86e3aafaae7097e4f1f2e4e3a6c6bda43', 'experiments/code/ace/playbook.py': '84a9a7b999680b061cf37ed1bf32d2188d79684337ff44f71c5ba7c00e36d2a7', 'experiments/prompts/appworld_react_reflector_no_gt_prompt.txt': '7769bda89a9dd08044ee92823cde1d494bd8f06a562446f5ee6ff3821307c3c6', 'experiments/prompts/appworld_react_curator_prompt.txt': '27b7d9dd2dcfe85646bf19ea7be387760874b70c0f9f27b1bed38d8f3e11e2bc'}
_PROMPT_DIR = Path(__file__).resolve().parent / 'source_prompts'
_PROMPT_ARTIFACTS: Mapping[str, tuple[str, str]] = {'reflector': ('ace_interactive_reflector_no_gt.txt', '719c88875357fbf805b683b666c6118193748571147d519f1484df6619fad52f'), 'curator': ('ace_interactive_curator_no_gt.txt', 'e36273a5009cfd7e451f00161b7b4d3beba0551e5054af87d19d1ce59e83f4ce')}
_WRITER_MAX_TOKENS = 4096
_SECTIONS = ('strategies_and_hard_rules', 'apis_to_use_for_specific_information', 'useful_code_snippets_and_templates', 'common_mistakes_and_correct_strategies', 'problem_solving_heuristics_and_workflows', 'verification_checklist', 'troubleshooting_and_pitfalls', 'others')
_SECTION_TITLES = {'strategies_and_hard_rules': 'STRATEGIES AND HARD RULES', 'apis_to_use_for_specific_information': 'APIS TO USE FOR SPECIFIC INFORMATION', 'useful_code_snippets_and_templates': 'USEFUL CODE SNIPPETS AND TEMPLATES', 'common_mistakes_and_correct_strategies': 'COMMON MISTAKES AND CORRECT STRATEGIES', 'problem_solving_heuristics_and_workflows': 'PROBLEM SOLVING HEURISTICS AND WORKFLOWS', 'verification_checklist': 'VERIFICATION CHECKLIST', 'troubleshooting_and_pitfalls': 'TROUBLESHOOTING AND PITFALLS', 'others': 'OTHERS'}
_SECTION_SLUGS = {'strategies_and_hard_rules': 'shr', 'apis_to_use_for_specific_information': 'api', 'useful_code_snippets_and_templates': 'code', 'common_mistakes_and_correct_strategies': 'cms', 'problem_solving_heuristics_and_workflows': 'psw', 'verification_checklist': 'vc', 'troubleshooting_and_pitfalls': 'ts', 'others': 'misc'}
_REFLECTOR_FIELDS = frozenset({'reasoning', 'error_identification', 'root_cause_analysis', 'correct_approach', 'key_insight'})

def _load_prompt(name: str) -> str:
    filename, expected_hash = _PROMPT_ARTIFACTS[name]
    path = _PROMPT_DIR / filename
    raw = path.read_bytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError(f'ACE prompt hash mismatch for {filename}: expected {expected_hash}, got {actual_hash}')
    return raw.decode('utf-8').strip()

def _render(template: str, values: Mapping[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f'[[{key}]]', value)
    unresolved = sorted(set(re.findall('\\[\\[([A-Z_]+)\\]\\]', rendered)))
    if unresolved:
        raise RuntimeError(f'Unresolved ACE prompt fields: {unresolved}')
    return rendered

def _json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or '').strip()
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    fenced = re.search('```(?:json)?\\s*(\\{.*?\\})\\s*```', raw, re.DOTALL | re.I)
    candidates = [fenced.group(1)] if fenced else []
    depth = 0
    start: int | None = None
    in_string = False
    escaped = False
    for index, char in enumerate(raw):
        if in_string:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == '{':
            if depth == 0:
                start = index
            depth += 1
        elif char == '}' and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(raw[start:index + 1])
                start = None
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None

def _validate_reflector(raw: str) -> tuple[dict[str, str] | None, str | None]:
    value = _json_object(raw)
    if value is None:
        return (None, 'reflector output is not a JSON object')
    missing = sorted(_REFLECTOR_FIELDS - set(value))
    if missing:
        return (None, f'reflector output lacks fields: {missing}')
    result: dict[str, str] = {}
    for field in sorted(_REFLECTOR_FIELDS):
        item = value.get(field)
        if not isinstance(item, str) or not item.strip():
            return (None, f'reflector field {field!r} must be non-empty text')
        result[field] = item.strip()
    return (result, None)

def _normalize_section(section: str) -> str:
    return section.strip().lower().replace(' ', '_').replace('&', 'and').rstrip(':')

def _validate_curator(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    value = _json_object(raw)
    if value is None:
        return (None, 'curator output is not a JSON object')
    reasoning = value.get('reasoning')
    operations = value.get('operations')
    if not isinstance(reasoning, str):
        return (None, 'curator reasoning must be text')
    if not isinstance(operations, list):
        return (None, 'curator operations must be a list')
    normalized: list[dict[str, str]] = []
    for index, operation in enumerate(operations):
        if not isinstance(operation, Mapping):
            return (None, f'curator operation {index} is not an object')
        missing = {'type', 'section', 'content'} - set(operation)
        if missing:
            return (None, f'curator operation {index} lacks fields: {sorted(missing)}')
        if operation.get('type') != 'ADD':
            return (None, f'curator operation {index} is not ADD')
        section = _normalize_section(str(operation.get('section', '')))
        if section not in _SECTIONS:
            continue
        normalized.append({'type': 'ADD', 'section': section, 'content': str(operation.get('content', ''))})
    return ({'reasoning': reasoning, 'operations': normalized}, None)

def _task(metadata: Mapping[str, Any] | None, trajectory: str) -> str:
    values = metadata or {}
    for key in ('task', 'instruction'):
        value = values.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    match = re.search('^[ \\t]*(?:Task|Instruction):[^\\r\\n]*', trajectory, re.M)
    return match.group(0).strip() if match else 'Current interactive task'

class ACEAdapted(ERSlot):
    """ACE-AppWorld playbook evolution under the shared failure-only horizon."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._items = {section: [] for section in _SECTIONS}
        self._next_id = 1

    def read_placement(self) -> str:
        return 'user_prefix'

    def _has_items(self) -> bool:
        return any(self._items.values())

    def _playbook(self) -> str:
        lines: list[str] = []
        for section in _SECTIONS:
            lines.append(f'## {_SECTION_TITLES[section]}')
            for item in self._items[section]:
                lines.append(f"[{item['id']}] {item['content']}")
            lines.append('')
        return '\n'.join(lines).strip()

    def _apply_operations(self, operations: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
        accepted: list[dict[str, str]] = []
        for operation in operations:
            section = str(operation['section'])
            item = {'id': f'{_SECTION_SLUGS[section]}-{self._next_id:05d}', 'content': str(operation['content']).strip()}
            self._next_id += 1
            self._items[section].append(item)
            accepted.append({'type': 'ADD', 'section': section, **item})
        return accepted

    def build_section(self) -> str:
        if not self._has_items():
            return ''
        return 'ACE PLAYBOOK FROM PRIOR ATTEMPTS\n\n' + self._playbook()

    def read_for_trial(self, *, trial_index: int, total_trials: int, model: str, past_trajectories: Sequence[str] | None=None, metadata: dict[str, Any] | None=None) -> str:
        del trial_index, total_trials, model, past_trajectories, metadata
        return self.build_section()

    def on_trial_end(self, trajectory: str, *, actor_trajectory: str | None=None, success: bool, model: str, trial_index: int, total_trials: int, metadata: dict[str, Any] | None=None) -> None:
        if success:
            return
        has_future_trial = bool((metadata or {}).get('has_future_trial', trial_index + 1 < total_trials))
        if not has_future_trial:
            return
        writer_trajectory = actor_trajectory if isinstance(actor_trajectory, str) and actor_trajectory.strip() else trajectory
        task = _task(metadata, writer_trajectory)
        playbook_before = self._playbook()
        reflector_prompt = _render(_load_prompt('reflector'), {'TASK': task, 'TRAJECTORY': writer_trajectory, 'PLAYBOOK': playbook_before or '(empty)'})
        reflection_raw = complete(reflector_prompt, model=model, max_tokens=_WRITER_MAX_TOKENS).strip()
        curator_prompt = _render(_load_prompt('curator'), {'QUESTION_CONTEXT': task, 'CURRENT_PLAYBOOK': playbook_before or '(empty)', 'TRAJECTORY': writer_trajectory, 'RECENT_REFLECTION': reflection_raw})
        curator_raw = complete(curator_prompt, model=model, max_tokens=_WRITER_MAX_TOKENS).strip()
        curator_obj, _ = _validate_curator(curator_raw)
        if curator_obj is not None:
            self._apply_operations(curator_obj['operations'])
