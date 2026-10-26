from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import re
from typing import Any, Mapping
POLICY_AO_TRACE_VERSION = 'policy_ao_trace'
POLICY_SEARCH_TRACE_VERSION = 'policy_search_trace'
LOSSLESS_OBSERVATION_TRANSPORT_VERSION = 'lossless_repeated_observation_refs'
_VISUAL_ARTIFACT_RE = re.compile('^\\s*(?:Pre[-_ ]*|Post[-_ ]*)?(?:Screenshot|Observation image|Image|Frame)(?:\\s+\\d+)?\\s*:\\s*.+$', flags=re.I)

def strip_visual_artifact_lines(trajectory: str) -> str:
    """Remove local screenshot artifact paths before sending text to LLM memory."""
    return '\n'.join((line for line in str(trajectory or '').splitlines() if not _VISUAL_ARTIFACT_RE.match(line)))

def normalize_policy_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one actor-produced trace of executed A-O interactions only."""
    if not isinstance(value, Mapping):
        raise TypeError('policy evidence must be a mapping')
    if value.get('schema_version') != 1:
        raise ValueError('policy evidence must use schema_version=1')
    task = value.get('task')
    initial_observation = value.get('initial_observation')
    close_reason = value.get('trial_close_reason')
    steps = value.get('steps')
    if not isinstance(task, str) or not task.strip():
        raise ValueError('policy evidence task must be a non-empty string')
    if not isinstance(initial_observation, str):
        raise ValueError('policy evidence initial_observation must be a string')
    if not isinstance(close_reason, str) or not close_reason.strip():
        raise ValueError('policy evidence trial_close_reason must be non-empty')
    if not isinstance(steps, list):
        raise ValueError('policy evidence steps must be a list')
    normalized_steps: list[dict[str, str]] = []
    for index, raw_step in enumerate(steps):
        if not isinstance(raw_step, Mapping):
            raise ValueError(f'policy evidence step {index} must be a mapping')
        unknown = sorted(set(raw_step) - {'action', 'observation'})
        if unknown:
            raise ValueError(f'policy evidence step {index} has unknown fields: {unknown!r}')
        action = raw_step.get('action')
        observation = raw_step.get('observation')
        if not isinstance(action, str) or not action.strip():
            raise ValueError(f'policy evidence step {index} action must be non-empty')
        if not isinstance(observation, str):
            raise ValueError(f'policy evidence step {index} observation must be a string')
        step = {'action': action.strip(), 'observation': observation}
        normalized_steps.append(step)
    return {'schema_version': 1, 'task': ' '.join(task.split()), 'initial_observation': initial_observation, 'trial_close_reason': close_reason.strip(), 'steps': normalized_steps}

def render_policy_trajectory(value: Mapping[str, Any]) -> str:
    """Render validated policy evidence without admitting non-policy text."""
    evidence = normalize_policy_evidence(value)
    lines = [POLICY_AO_TRACE_VERSION, f"Task: {evidence['task']}", 'Initial observation JSON: ' + json.dumps(evidence['initial_observation'], ensure_ascii=True)]
    for index, step in enumerate(evidence['steps'], 1):
        lines.append(f'AO {index} JSON: ' + json.dumps({'action': step['action'], 'observation': step['observation']}, ensure_ascii=True, sort_keys=True, separators=(',', ':')))
    lines.append('Terminal JSON: ' + json.dumps({'trial_close_reason': evidence['trial_close_reason']}, ensure_ascii=True, sort_keys=True, separators=(',', ':')))
    return '\n'.join(lines)

def normalize_policy_search_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate executed A-O transitions from a recoverable-state search.

    A tree-search budget block is not a linear trajectory: the controller may
    restore a previously observed state before dispatching the next action.
    Each transition therefore names its actual source and target tree nodes and
    retains the actor-visible source observation used for that dispatch.
    """
    if not isinstance(value, Mapping):
        raise TypeError('policy search evidence must be a mapping')
    if value.get('schema_version') != 1:
        raise ValueError('policy search evidence must use schema_version=1')
    task = value.get('task')
    initial_observation = value.get('initial_observation')
    close_reason = value.get('trial_close_reason')
    transitions = value.get('transitions')
    if not isinstance(task, str) or not task.strip():
        raise ValueError('policy search evidence task must be non-empty')
    if not isinstance(initial_observation, str):
        raise ValueError('policy search initial observation must be a string')
    if not isinstance(close_reason, str) or not close_reason.strip():
        raise ValueError('policy search close reason must be non-empty')
    if not isinstance(transitions, list):
        raise ValueError('policy search transitions must be a list')
    expected_fields = {'restore_event_id', 'source_node_id', 'source_observation', 'action', 'observation', 'target_node_id'}
    normalized: list[dict[str, str]] = []
    seen_targets: set[str] = set()
    for index, raw_transition in enumerate(transitions):
        if not isinstance(raw_transition, Mapping):
            raise ValueError(f'policy search transition {index} must be a mapping')
        if set(raw_transition) != expected_fields:
            raise ValueError(f'policy search transition {index} has an invalid field set')
        transition: dict[str, str] = {}
        for field in expected_fields:
            item = raw_transition.get(field)
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f'policy search transition {index} {field} must be non-empty')
            transition[field] = item if field in {'source_observation', 'observation'} else item.strip()
        if transition['target_node_id'] in seen_targets:
            raise ValueError('policy search transition targets must be unique')
        seen_targets.add(transition['target_node_id'])
        normalized.append(transition)
    return {'schema_version': 1, 'task': ' '.join(task.split()), 'initial_observation': initial_observation, 'trial_close_reason': close_reason.strip(), 'transitions': normalized}

def render_policy_search_trajectory(value: Mapping[str, Any]) -> str:
    """Render one non-linear policy-search interaction trace."""
    evidence = normalize_policy_search_evidence(value)
    lines = [POLICY_SEARCH_TRACE_VERSION, f"Task: {evidence['task']}", 'Initial observation JSON: ' + json.dumps(evidence['initial_observation'], ensure_ascii=True)]
    for index, transition in enumerate(evidence['transitions'], 1):
        lines.append(f'Transition {index} JSON: ' + json.dumps(transition, ensure_ascii=True, sort_keys=True, separators=(',', ':')))
    lines.append('Terminal JSON: ' + json.dumps({'trial_close_reason': evidence['trial_close_reason']}, ensure_ascii=True, sort_keys=True, separators=(',', ':')))
    return '\n'.join(lines)

def parse_policy_search_trajectory(trajectory: str) -> dict[str, Any] | None:
    """Parse the canonical recoverable-search trace, if present."""
    lines = str(trajectory or '').splitlines()
    if not lines or lines[0].strip() != POLICY_SEARCH_TRACE_VERSION:
        return None
    if len(lines) < 4 or not lines[1].startswith('Task: '):
        raise ValueError('canonical policy search trace lacks its task')
    if not lines[2].startswith('Initial observation JSON: '):
        raise ValueError('canonical policy search trace lacks its initial observation')
    try:
        initial_observation = json.loads(lines[2].removeprefix('Initial observation JSON: '))
    except json.JSONDecodeError as exc:
        raise ValueError('canonical policy search trace has invalid initial JSON') from exc
    transitions: list[dict[str, str]] = []
    terminal_reason = ''
    terminal_seen = False
    for line_index, line in enumerate(lines[3:], 3):
        match = re.fullmatch('Transition (\\d+) JSON: (.+)', line)
        if match:
            if terminal_seen:
                raise ValueError('canonical policy search trace has a transition after close')
            if int(match.group(1)) != len(transitions) + 1:
                raise ValueError('canonical policy search transition indices are not contiguous')
            try:
                payload = json.loads(match.group(2))
            except json.JSONDecodeError as exc:
                raise ValueError('canonical policy search trace has invalid transition JSON') from exc
            transitions.append(payload)
            continue
        if line.startswith('Terminal JSON: '):
            if terminal_seen or line_index != len(lines) - 1:
                raise ValueError('canonical policy search terminal must occur once last')
            try:
                terminal = json.loads(line.removeprefix('Terminal JSON: '))
            except json.JSONDecodeError as exc:
                raise ValueError('canonical policy search trace has invalid terminal JSON') from exc
            if not isinstance(terminal, dict) or set(terminal) != {'trial_close_reason'}:
                raise ValueError('canonical policy search terminal has an invalid schema')
            terminal_reason = terminal.get('trial_close_reason')
            terminal_seen = True
            continue
        raise ValueError(f'canonical policy search trace has an unknown line: {line!r}')
    if not terminal_seen:
        raise ValueError('canonical policy search trace lacks its terminal record')
    return normalize_policy_search_evidence({'schema_version': 1, 'task': lines[1].removeprefix('Task: '), 'initial_observation': initial_observation, 'trial_close_reason': terminal_reason, 'transitions': deepcopy(transitions)})

def parse_policy_trajectory(trajectory: str) -> dict[str, Any] | None:
    """Parse the canonical policy trace; return ``None`` for other formats."""
    lines = str(trajectory or '').splitlines()
    if not lines or lines[0].strip() != POLICY_AO_TRACE_VERSION:
        return None
    if len(lines) < 4 or not lines[1].startswith('Task: '):
        raise ValueError('canonical policy trajectory lacks its task')
    if not lines[2].startswith('Initial observation JSON: '):
        raise ValueError('canonical policy trajectory lacks its initial observation')
    try:
        initial_observation = json.loads(lines[2].removeprefix('Initial observation JSON: '))
    except json.JSONDecodeError as exc:
        raise ValueError('canonical policy trajectory has invalid initial JSON') from exc
    steps: list[dict[str, str]] = []
    terminal_reason = ''
    terminal_seen = False
    for line_index, line in enumerate(lines[3:], 3):
        ao_match = re.fullmatch('AO (\\d+) JSON: (.+)', line)
        if ao_match:
            if terminal_seen:
                raise ValueError('canonical policy trajectory has AO after terminal')
            if int(ao_match.group(1)) != len(steps) + 1:
                raise ValueError('canonical policy trajectory AO indices are not contiguous')
            try:
                payload = json.loads(ao_match.group(2))
            except json.JSONDecodeError as exc:
                raise ValueError('canonical policy trajectory has invalid AO JSON') from exc
            if not isinstance(payload, dict):
                raise ValueError('canonical policy trajectory AO JSON must be an object')
            if set(payload) != {'action', 'observation'}:
                raise ValueError('canonical policy trajectory AO JSON has extra fields')
            steps.append({'action': payload.get('action'), 'observation': payload.get('observation')})
            continue
        if line.startswith('Terminal JSON: '):
            if terminal_seen or line_index != len(lines) - 1:
                raise ValueError('canonical policy trajectory terminal must occur once last')
            try:
                terminal = json.loads(line.removeprefix('Terminal JSON: '))
            except json.JSONDecodeError as exc:
                raise ValueError('canonical policy trajectory has invalid terminal JSON') from exc
            if not isinstance(terminal, dict):
                raise ValueError('canonical policy terminal JSON must be an object')
            if set(terminal) != {'trial_close_reason'}:
                raise ValueError('canonical policy terminal JSON has extra fields')
            terminal_reason = terminal.get('trial_close_reason')
            terminal_seen = True
            continue
        raise ValueError(f'canonical policy trajectory has an unknown line: {line!r}')
    if not terminal_seen:
        raise ValueError('canonical policy trajectory lacks its terminal record')
    return normalize_policy_evidence({'schema_version': 1, 'task': lines[1].removeprefix('Task: '), 'initial_observation': initial_observation, 'trial_close_reason': terminal_reason, 'steps': deepcopy(steps)})

def intern_repeated_policy_observations(text: str, policy_trajectory: str) -> tuple[str, dict[str, Any]]:
    """Replace exact repeated observations with reversible textual refs.

    The raw Actor and canonical policy records remain untouched. This view is
    only a lossless transport for LLM memory writers whose context would
    otherwise repeat the same large environment observation many times.
    """
    source = str(text or '')
    parsed = parse_policy_trajectory(policy_trajectory)
    if parsed is None:
        raise ValueError('lossless observation transport requires canonical policy evidence')
    observations = [str(parsed['initial_observation']), *(str(step['observation']) for step in parsed['steps'])]
    selected: list[tuple[str, str, int]] = []
    for observation in dict.fromkeys(observations):
        if not observation:
            continue
        occurrences = source.count(observation)
        if occurrences < 2:
            continue
        ref = f'OBS-{len(selected) + 1}'
        marker = f'<EXACT_OBSERVATION_REF:{ref}>'
        registry_overhead = len(ref) + len(marker) + 48
        original_chars = occurrences * len(observation)
        encoded_chars = len(observation) + occurrences * len(marker)
        if original_chars <= encoded_chars + registry_overhead:
            continue
        selected.append((observation, ref, occurrences))
    if not selected:
        digest = hashlib.sha256(source.encode('utf-8')).hexdigest()
        return (source, {'version': LOSSLESS_OBSERVATION_TRANSPORT_VERSION, 'active': False, 'raw_chars': len(source), 'encoded_chars': len(source), 'observation_refs': 0, 'replaced_occurrences': 0, 'raw_sha256': digest, 'encoded_sha256': digest})
    encoded_transcript = source
    for observation, ref, _ in sorted(selected, key=lambda row: len(row[0]), reverse=True):
        marker = f'<EXACT_OBSERVATION_REF:{ref}>'
        if marker in source:
            raise ValueError('observation reference marker collides with source text')
        encoded_transcript = encoded_transcript.replace(observation, marker)
    registry_lines = [f'{ref} JSON: {json.dumps(observation, ensure_ascii=True)}' for observation, ref, _ in selected]
    encoded = '\n'.join([LOSSLESS_OBSERVATION_TRANSPORT_VERSION, 'Each <EXACT_OBSERVATION_REF:OBS-n> below denotes the exact UTF-8 Observation registered here; references preserve every occurrence and its original order.', *registry_lines, 'ORDERED TRAJECTORY:', encoded_transcript])
    if len(encoded) >= len(source):
        digest = hashlib.sha256(source.encode('utf-8')).hexdigest()
        return (source, {'version': LOSSLESS_OBSERVATION_TRANSPORT_VERSION, 'active': False, 'raw_chars': len(source), 'encoded_chars': len(source), 'observation_refs': 0, 'replaced_occurrences': 0, 'raw_sha256': digest, 'encoded_sha256': digest})
    return (encoded, {'version': LOSSLESS_OBSERVATION_TRANSPORT_VERSION, 'active': True, 'raw_chars': len(source), 'encoded_chars': len(encoded), 'observation_refs': len(selected), 'replaced_occurrences': sum((row[2] for row in selected)), 'raw_sha256': hashlib.sha256(source.encode('utf-8')).hexdigest(), 'encoded_sha256': hashlib.sha256(encoded.encode('utf-8')).hexdigest()})
