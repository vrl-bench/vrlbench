from __future__ import annotations
from copy import deepcopy
import json
import logging
import os
import random
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional
from vrlbench.runtime import VEX2
from vrlbench.runtime.budget import EnvironmentBudgetExceeded, EnvironmentInteractionLedger
from vrlbench.runtime.controllers import make_controller
from vrlbench.runtime.er import make_er_slot
from vrlbench.runtime.loop import EpisodeResult, prepare_shared_t1, run_episode
from vrlbench.runtime.metrics import atomic_write_json, completion_metrics, per_trial_sr, resource_usage_summary, write_env_results
from vrlbench.runtime.shared_t1 import validate_actor_only_t1_usage, validate_shared_t1_environment_accounting, validate_shared_t1_source
from vrlbench.environments.webshop_dependency import WEBSHOP_FULL_DATA_FILES, WEBSHOP_OFFICIAL_TEST_GOAL_END, WEBSHOP_OFFICIAL_TEST_GOAL_START, webshop_observed_dependency_contract
logger = logging.getLogger(__name__)
DEFAULT_MAX_STEPS = 15
WEBSHOP_ACTION_PATTERN = '(?:search|click)\\[(?!\\s)(?![^\\]\\r\\n]*\\s\\])[^\\]\\r\\n]+\\]'
DEFAULT_WEBSHOP_ROOT = Path(os.environ['WEBSHOP_ROOT']).expanduser() if os.environ.get('WEBSHOP_ROOT', '').strip() else None
REFLEXION_ACTION_DEMONSTRATION_SOURCE = 'reflexion-original/webshop_runs/base_prompt.txt:6,24,41,44,47'
REFLEXION_ACTION_DEMONSTRATION = 'search[3 ounce bright citrus deodorant sensitive skin]\nclick[b078gwrc1j]\nclick[bright citrus]\nclick[3 ounce (pack of 1)]\nclick[buy now]'
ACTOR_PROMPT_TEMPLATE = '{state}Environment-feedback budget: {used_ao_steps} of {max_ao_steps} action-observation transitions used; {remaining_ao_steps} remain in this trial.\nReturn exactly one action.\nAction:'
FORMAT_REPAIR_PROMPT_TEMPLATE = '{state}Environment-feedback budget: {used_ao_steps} of {max_ao_steps} action-observation transitions used; {remaining_ao_steps} remain in this trial. This format repair does not consume an environment transition.\nThe previous response did not match the required format:\n{invalid_response}\nParser rejection: {rejection_reason}\n\nFormat repair only. Return exactly one search[...] or click[...] action with no\nprefix, code fence, extra whitespace inside the action name, or prose.\nIf method memory explicitly requires one non-environment auxiliary metadata\nline, append exactly that one line after the repaired action.\nPreserve the exact casing of visible labels as advertised. The official text\nenvironment advertises lowercase click targets, so copy those lowercase targets\nexactly.\nAction:'

class _OfficialWebShopTextAdapter:
    """Adapter for the official WebShop text Gym environment."""

    def __init__(self, env: Any):
        self.env = env
        self.observation = ''
        self._sample_index: Optional[int] = None
        self._last_action_invalid = False

    def step(self, session: int | str, action: str):
        if action == 'reset':
            observation, _info = self.env.reset(session=session)
            self.observation = str(observation)
            self._sample_index = _sample_index_from_session(session)
            self._last_action_invalid = False
            return (self._with_available_actions(self.observation), 0.0, False)
        self._last_action_invalid = not self._action_dispatch_is_valid(action)
        observation, reward, done, _info = self.env.step(action)
        self.observation = str(observation)
        return (self._with_available_actions(self.observation), float(reward), bool(done))

    @property
    def last_action_invalid(self) -> bool:
        """Whether the most recent parser-valid action was a WebShop no-op."""
        return self._last_action_invalid

    def _action_dispatch_is_valid(self, action: str) -> bool:
        match = re.fullmatch('(search|click)\\[([^\\]\\r\\n]+)\\]', str(action))
        if match is None:
            return False
        action_name, argument = match.groups()
        if action_name == 'search':
            return bool(argument)
        available = self.env.get_available_actions()
        clickables = {str(value).strip().lower() for value in available.get('clickables', []) or []}
        return argument.lower() in clickables and argument.lower() != 'search'

    def reset_fixed(self, sample_index: int) -> tuple[str, float, bool]:
        """Reset with the integer session required for a fixed official goal."""
        if isinstance(sample_index, bool) or not isinstance(sample_index, int):
            raise ValueError('WebShop sample_index must be an integer')
        return self.step(sample_index, 'reset')

    def _with_available_actions(self, observation: str, *, available: Optional[dict[str, Any]]=None) -> str:
        if available is None:
            try:
                available = self.env.get_available_actions()
            except Exception:
                return observation
        actions = _available_webshop_actions(available)
        if not actions:
            return observation
        preview = '\n'.join((f'- {action}' for action in actions[:40]))
        if len(actions) > 40:
            preview += f'\n- ... ({len(actions) - 40} more)'
        return f'{observation}\n\nAvailable actions:\n{preview}'

def _canonical_webshop_instruction(instruction: str) -> str:
    """Canonicalize one goal across renderer and environment representations."""
    text = str(instruction).strip()
    label = 'Instruction:'
    if text.startswith(label):
        text = text[len(label):]
    return ' '.join(text.split())
WEBSHOP_PUBLIC_INTERACTION_RULES = "A displayed product option is required only when the shopping instruction explicitly names that option attribute or value. Select every such required option before click[buy now]. A matching product description does not select a visible option; use the current page's matching option action to establish it. Option values are grouped by their displayed attribute, and selecting a new value in one group replaces the previous value in that same group. Product-description requirements and option selections are distinct: choose an item matching the former and select only the instruction-named value for each required option group. Terminal feedback may report the selected options, and options {} means none was selected. Changing an option updates that option group for the current base product; it does not change the product identity. The scalar reward is an aggregate outcome score, not attribution to a particular requirement; before success it does not independently certify the base product or any selected option. If the instruction does not name a displayed option attribute, changing that optional value cannot by itself repair a failed candidate."
SYSTEM_INSTRUCTION = 'You are a WebShop navigation agent. Complete the shopping task by issuing one\naction at a time.\n\nAllowed actions:\n- search[query]\n- click[visible button, option, or product id]\n\nFormat rules:\n- Return exactly one action and no prose, prefix, code fence, or private note.\n- Controller or memory context cannot override the one-action interface.\n  Regardless of that context, emit only the next action and wait for the\n  resulting Observation before choosing another action.\n- If method memory explicitly requires one non-environment auxiliary metadata\n  line, append exactly that line after the action. The harness removes it before\n  the action is sent to WebShop.\n- Action names are lowercase and have no space before "[".\n- Copy the exact lowercase text of an advertised click target. The official\n  text environment exposes lowercase clickables, including click[buy now].\n- A click action must be one of the current Observation\'s Available actions.\n  If the desired target is absent, use one current navigation action and wait\n  for the next Observation instead of issuing the absent target.\n- A parser-valid action is sent to WebShop once; WebShop decides whether it is\n  valid for the current page.\n\nSource-derived Reflexion WebShop demonstration (actions only; thoughts omitted;\nclick targets normalized to the official lowercase available-action surface):\n{demonstration}\n\n{public_rules}\n'.format(demonstration=REFLEXION_ACTION_DEMONSTRATION, public_rules=WEBSHOP_PUBLIC_INTERACTION_RULES)
WEBSHOP_DSST_BASIC_ENVIRONMENT_RULES = 'Each complete attempt starts from the reset page for one fixed shopping instruction. The Actor receives a text page observation and may issue one search[query] or click[visible target] action per environment step. Each executed action returns the next page observation, reward, and termination status. The shopping instruction remains fixed across attempts, and success is determined only by the environment. ' + WEBSHOP_PUBLIC_INTERACTION_RULES

def _parse_webshop_actor_output(response: str) -> str:
    lines = str(response or '').strip().splitlines()
    if len(lines) != 1 or not re.fullmatch(WEBSHOP_ACTION_PATTERN, lines[0]):
        return ''
    return lines[0]

def _sanitize_webshop_terminal_feedback(observation: str) -> str:
    """Remove evaluator-only target payloads from terminal WebShop pages."""
    text = str(observation or '')
    text = re.sub('\\[SEP\\]\\s*Target\\s*\\[SEP\\].*', '[SEP] Target details redacted', text, flags=re.I | re.S)
    text = re.sub('\\[SEP\\]\\s*Reward Details\\s*\\[SEP\\].*', '[SEP] Reward details redacted', text, flags=re.I | re.S)
    return text

def _shared_webshop_feedback(observation: str, reward: float, *, terminal: bool) -> str:
    """Expose the same public WebShop outcome signal to every method."""
    text = _sanitize_webshop_terminal_feedback(observation) if terminal else str(observation or '')
    if terminal or float(reward) != 0.0:
        text = f'{text}\nPublic environment reward: {float(reward)}'
    return text

def make_actor(env: Any, sample_index: int, model: str='gpt-4o-mini', *, llm_fn: Optional[Callable[..., str]]=None, max_steps: int=DEFAULT_MAX_STEPS, max_environment_steps: Optional[int]=None, shared_t1_trajectory: str='') -> Callable[[], str]:
    """Create a WebShop actor closure for one fixed session/index."""
    configured_action_horizon = int(max_steps)
    if configured_action_horizon <= 0:
        raise ValueError('max_steps must be positive')
    llm_usage_instrumented = llm_fn is None
    if llm_fn is None:
        from vrlbench.runtime.llm import complete as llm_fn
    assert llm_fn is not None
    trial_state: dict[str, Any] = {'trial_index': 0, 'total_trials': 0, 'prepared': False, 'task': '', 'initial_observation': '', 'format_repairs': 0, 'invalid_actor_responses': 0, 'policy_evidence_eligible': False, 'protocol_failure': False, 'protocol_failure_reason': '', 'environment_terminated': False, 'environment_truncated': False, 'trial_close_reason': '', 'policy_evidence': None, 'actor_messages': []}
    max_ao_steps = configured_action_horizon if max_environment_steps is None else int(max_environment_steps)
    interaction_ledger = EnvironmentInteractionLedger(max_steps_per_episode=max_ao_steps)
    max_actor_calls = 3 * max_ao_steps

    def set_trial_context(*, trial_index: int, total_trials: int) -> dict[str, Any]:
        trial_state['trial_index'] = int(trial_index)
        trial_state['total_trials'] = int(total_trials)
        trial_state['prepared'] = False
        trial_state['task'] = ''
        trial_state['initial_observation'] = ''
        return {'sample_index': int(sample_index), 'trial_index': int(trial_index), 'total_trials': int(total_trials)}

    def call_actor_model(prompt: str, *, phase: str, max_tokens: Optional[int]=None) -> str:
        if interaction_ledger.actor_turns >= max_actor_calls:
            raise RuntimeError('WebShop provider-call safety cap reached')
        interaction_ledger.charge_actor_turn()
        response = str(llm_fn(prompt, model=model, system=SYSTEM_INSTRUCTION, max_tokens=int(os.environ.get('WEBSHOP_ACTOR_MAX_TOKENS', '160')) if max_tokens is None else int(max_tokens)))
        messages = trial_state.get('actor_messages')
        if isinstance(messages, list):
            messages.append({'role': 'assistant', 'phase': phase, 'content': response})
        return response

    def compact_response(response: str) -> str:
        return ' '.join(str(response or '').split())[:500] or '<empty>'

    def prepare_trial() -> dict[str, str]:
        """Perform the trial's single public reset and cache its initial state."""
        if trial_state['prepared']:
            return {'task': str(trial_state['task']), 'initial_observation': str(trial_state['initial_observation'])}
        session = _fixed_webshop_session(sample_index)
        interaction_ledger.begin_episode(trial_state['trial_index'])
        trial_state['format_repairs'] = 0
        trial_state['invalid_actor_responses'] = 0
        trial_state['policy_evidence_eligible'] = False
        trial_state['protocol_failure'] = False
        trial_state['protocol_failure_reason'] = ''
        trial_state['environment_terminated'] = False
        trial_state['environment_truncated'] = False
        trial_state['trial_close_reason'] = ''
        trial_state['policy_evidence'] = None
        trial_state['actor_messages'] = []
        interaction_ledger.charge_reset()
        observation, _reward, _done = env.step(session, 'reset')
        trial_state['task'] = _extract_instruction(observation)
        trial_state['initial_observation'] = str(observation or '')
        if shared_t1_trajectory.strip():
            _assert_shared_t1_matches_current_goal(current_observation=observation, sample_index=sample_index, shared_t1_trajectory=shared_t1_trajectory)
        trial_state['prepared'] = True
        return {'task': str(trial_state['task']), 'initial_observation': str(observation or '')}

    def actor() -> str:
        prepared = prepare_trial()
        session = _fixed_webshop_session(sample_index)
        observation = prepared['initial_observation']
        transcript = ['WebShop', f'Sample index: {sample_index}', f"Trial index: {trial_state['trial_index']}", f'Observation:\n{observation}']
        prompt = f'WebShop task state:\nObservation:\n{observation}\n\n'
        trial_state['actor_messages'].append({'role': 'user', 'phase': 'initial', 'content': f"Task: {trial_state['task']}\nObservation:\n{observation}"})
        trial_state['policy_evidence'] = {'schema_version': 1, 'task': trial_state['task'], 'initial_observation': str(observation or ''), 'trial_close_reason': 'running', 'steps': []}

        def close_trial(reason: str, *, environment_terminated: bool=False, environment_truncated: bool=False) -> None:
            if environment_terminated and environment_truncated:
                raise ValueError('a trial cannot be both terminated and truncated')
            trial_state['trial_close_reason'] = reason
            policy_evidence = trial_state.get('policy_evidence')
            if isinstance(policy_evidence, dict):
                policy_evidence['trial_close_reason'] = reason
            trial_state['environment_terminated'] = environment_terminated
            trial_state['environment_truncated'] = environment_truncated
            executed_steps = policy_evidence.get('steps', []) if isinstance(policy_evidence, dict) else []
            trial_state['policy_evidence_eligible'] = bool(executed_steps) or reason == 'environment_success'

        def actor_model_or_protocol_failure(model_prompt: str, *, phase: str, max_tokens: Optional[int]=None) -> Optional[str]:
            return call_actor_model(model_prompt, phase=phase, max_tokens=max_tokens)
        while interaction_ledger.actor_turns < max_actor_calls:
            if interaction_ledger.steps >= interaction_ledger.max_steps_per_episode:
                close_trial('ao_horizon', environment_truncated=True)
                transcript.append('FAIL (environment step budget reached)')
                return '\n'.join(transcript)
            remaining_ao_steps = interaction_ledger.max_steps_per_episode - interaction_ledger.steps
            action_prompt = ACTOR_PROMPT_TEMPLATE.format(state=prompt, used_ao_steps=interaction_ledger.steps, max_ao_steps=interaction_ledger.max_steps_per_episode, remaining_ao_steps=remaining_ao_steps)
            response = actor_model_or_protocol_failure(action_prompt, phase='action')
            if response is None:
                return '\n'.join(transcript)
            action = _parse_webshop_actor_output(response)
            if not action and interaction_ledger.actor_turns < max_actor_calls:
                trial_state['invalid_actor_responses'] += 1
                trial_state['actor_messages'].append({'role': 'user', 'phase': 'parser_feedback', 'content': 'No environment step occurred. The response was rejected: ' + ('response must contain exactly one valid action' or 'response must contain exactly one valid action')})
                repair_prompt = FORMAT_REPAIR_PROMPT_TEMPLATE.format(state=prompt, used_ao_steps=interaction_ledger.steps, max_ao_steps=interaction_ledger.max_steps_per_episode, remaining_ao_steps=remaining_ao_steps, invalid_response=compact_response(response), rejection_reason='response must contain exactly one valid action' or 'response must contain exactly one valid action')
                trial_state['format_repairs'] += 1
                response = actor_model_or_protocol_failure(repair_prompt, phase='format_repair')
                if response is None:
                    return '\n'.join(transcript)
                action = _parse_webshop_actor_output(response)
            if not action:
                trial_state['invalid_actor_responses'] += 1
                trial_state['protocol_failure'] = True
                trial_state['protocol_failure_reason'] = 'format repair failed to produce one dispatchable action'
                transcript.append('[Actor format invalid; no environment feedback obtained]')
                close_trial('actor_protocol_failure')
                transcript.append('FAIL (actor protocol violation)')
                return '\n'.join(transcript)
            invalid_dispatched_action = False
            try:
                interaction_ledger.require_step()
                observation, reward, done = env.step(session, action)
                interaction_ledger.charge_step()
            except EnvironmentBudgetExceeded as e:
                close_trial('ao_horizon', environment_truncated=True)
                transcript.append(f'Environment budget exhausted: {e}')
                transcript.append('FAIL (environment step budget reached)')
                return '\n'.join(transcript)
            except AssertionError:
                if not bool(getattr(env, 'last_action_invalid', False)):
                    raise
                observation, reward, done = ('Invalid action!', 0.0, False)
                interaction_ledger.charge_step()
                invalid_dispatched_action = True
            invalid_dispatched_action = invalid_dispatched_action or bool(getattr(env, 'last_action_invalid', False))
            observation = _shared_webshop_feedback(observation, reward, terminal=bool(done or float(reward) > 0.0))
            transcript.append(f'> {action}')
            policy_evidence = trial_state.get('policy_evidence')
            if not isinstance(policy_evidence, dict):
                raise RuntimeError('WebShop policy evidence was not initialized')
            policy_evidence['steps'].append({'action': action, 'observation': str(observation or '')})
            transcript.append(f'Observation:\n{observation}')
            trial_state['actor_messages'].append({'role': 'user', 'phase': 'environment_feedback', 'content': f"Executed action: {action}\nObservation:\n{str(observation or '')}"})
            if done:
                close_trial('environment_success' if float(reward) >= 1.0 else 'environment_terminal_failure', environment_terminated=True)
                transcript.append(f'Reward: {float(reward)}')
                transcript.append('SUCCESS' if float(reward) >= 1.0 else 'FAIL')
                return '\n'.join(transcript)
            prompt += f'Action: {action}\nObservation:\n{observation}\n\n'
        if interaction_ledger.steps >= interaction_ledger.max_steps_per_episode:
            close_trial('ao_horizon', environment_truncated=True)
            transcript.append('FAIL (environment step budget reached)')
        else:
            trial_state['protocol_failure'] = True
            trial_state['protocol_failure_reason'] = 'provider call safety cap reached without a dispatchable action'
            close_trial('provider_call_safety_cap')
            transcript.append('FAIL (provider call safety cap reached)')
        return '\n'.join(transcript)

    def get_trial_metadata() -> dict[str, Any]:
        metadata = {**interaction_ledger.snapshot(), 'task': str(trial_state.get('task') or ''), 'llm_usage_instrumented': llm_usage_instrumented, 'format_repairs': int(trial_state.get('format_repairs') or 0), 'invalid_actor_responses': int(trial_state.get('invalid_actor_responses') or 0), 'protocol_failure': bool(trial_state.get('protocol_failure', False)), 'protocol_failure_reason': str(trial_state.get('protocol_failure_reason') or ''), 'environment_terminated': bool(trial_state.get('environment_terminated', False)), 'environment_truncated': bool(trial_state.get('environment_truncated', False)), 'trial_close_reason': str(trial_state.get('trial_close_reason') or ''), 'policy_evidence_eligible': bool(trial_state.get('policy_evidence_eligible', False))}
        return metadata

    def get_policy_evidence() -> dict[str, Any]:
        evidence = trial_state.get('policy_evidence')
        if not isinstance(evidence, dict):
            raise RuntimeError('WebShop actor has no completed policy evidence')
        return deepcopy(evidence)

    def get_actor_trajectory() -> str:
        messages = trial_state.get('actor_messages')
        if not isinstance(messages, list):
            raise RuntimeError('WebShop actor trajectory was not initialized')
        blocks = []
        for message in messages:
            role = str(message.get('role') or 'unknown').upper()
            phase = str(message.get('phase') or '').strip()
            label = f"[{role}{('/' + phase if phase else '')}]"
            blocks.append(f"{label}\n{message.get('content', '')}")
        return '\n\n'.join(blocks)
    actor.set_trial_context = set_trial_context
    actor.prepare_trial = prepare_trial
    actor.get_trial_metadata = get_trial_metadata
    actor.get_policy_evidence = get_policy_evidence
    actor.get_actor_trajectory = get_actor_trajectory
    return actor

def _fixed_webshop_session(sample_index: int) -> int:
    return int(sample_index)

def _extract_instruction(text: str) -> str:
    match = re.search('Instruction:\\s*(?:\\[SEP\\]\\s*)?([^\\r\\n]*?)(?=\\s*\\[SEP\\]|\\r?\\n|$)', str(text or ''))
    if not match:
        return ''
    return _canonical_webshop_instruction(match.group(1))

def _assert_shared_t1_matches_current_goal(*, current_observation: str, sample_index: int, shared_t1_trajectory: str) -> None:
    """Validate shared T1 against the observation from the formal reset.

    The official WebShop runtime binds goals to server-side session state. A shared T1
    artifact is only valid when the current reset target matches the stored
    trial-0 target for the same fixed session. The caller must pass the reset
    observation already obtained by the actor; this check never touches the
    environment itself.
    """
    expected = _extract_instruction(shared_t1_trajectory)
    current = _extract_instruction(current_observation)
    if not expected or not current:
        raise RuntimeError(f'WebShop shared T1 goal identity could not be recovered from both artifacts for sample_index={sample_index}')
    if expected != current:
        raise RuntimeError(f'WebShop shared T1 does not match the current reset target for sample_index={sample_index}. Rerun the WebShop baseline and all matched comparison cells in the same benchmark/server state. shared_t1={expected!r}; current_reset={current!r}')

def make_manager() -> Callable[[str], bool]:
    """Success when the trajectory terminates with SUCCESS."""

    def manager(trajectory: str) -> bool:
        for raw in reversed(trajectory.strip().splitlines()):
            line = raw.strip()
            if line == 'SUCCESS':
                return True
            if line in {'FAIL', 'FAIL (max steps reached)', 'FAIL (actor turn budget reached)', 'FAIL (environment step budget reached)'}:
                return False
        return False
    return manager

def _load_shared_t1(shared_t1_dir: str | Path | None, *, require_complete_accounting: bool=False, require_environment_accounting: bool=True, expected_max_environment_steps: Optional[int]=None) -> dict[int, dict[str, Any]]:
    if not shared_t1_dir:
        return {}
    path = Path(shared_t1_dir) / 'episodes.json'
    if not path.exists():
        raise FileNotFoundError(f'WebShop shared T1 requires episodes.json: {Path(shared_t1_dir)}')
    rows = json.loads(path.read_text())
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        sample_index = row.get('sample_index')
        if sample_index is None:
            match = re.search('sample_(\\d+)', str(row.get('name', '')))
            sample_index = int(match.group(1)) if match else None
        if sample_index is None:
            continue
        trial0 = next((trial for trial in row.get('trials', []) if isinstance(trial, dict) and int(trial.get('trial', -1)) == 0), None)
        if trial0 is None:
            continue
        sample_index = int(sample_index)
        if sample_index in out:
            raise ValueError(f'duplicate WebShop shared T1 sample_index={sample_index} in {path}')
        success = bool(trial0.get('success', False))
        trajectory = str(trial0.get('trajectory', ''))
        actor_trajectory = str(trial0.get('actor_trajectory', ''))
        if not success and (not trajectory.strip()):
            raise ValueError(f'WebShop shared T1 failed trial is missing trajectory evidence for sample_index={int(sample_index)} in {path}. Adapted memory baselines must update from the real failed T1 trace, not from a placeholder.')
        if not actor_trajectory.strip():
            raise ValueError(f'WebShop shared T1 is missing the raw Actor trajectory for sample_index={int(sample_index)} in {path}.')
        trial_metadata = trial0.get('trial_metadata')
        llm_usage = trial0.get('llm_usage')
        validate_actor_only_t1_usage(llm_usage if isinstance(llm_usage, dict) else None, source=f'{path} sample_index={sample_index}', require_complete_accounting=require_complete_accounting)
        if require_environment_accounting:
            if expected_max_environment_steps is None:
                raise ValueError('shared T1 requires a protocol-derived AO horizon')
            trial_metadata = validate_shared_t1_environment_accounting(trial_metadata if isinstance(trial_metadata, dict) else None, benchmark='webshop', source=f'{path} sample_index={sample_index}', expected_max_environment_steps=expected_max_environment_steps, success=success)
            if int(trial_metadata.get('sample_index', -1)) != sample_index:
                raise ValueError(f'{path} sample_index={sample_index}: trial metadata identity mismatch')
            task = str(trial_metadata.get('task') or '')
            if task != _extract_instruction(trajectory):
                raise ValueError(f'{path} sample_index={sample_index}: task metadata does not match the actor-visible trajectory')
        restored_metadata = dict(trial_metadata or {})
        out[sample_index] = {'success': success, 'trajectory': trajectory, 'policy_trajectory': str(trial0.get('policy_trajectory', '')), 'actor_trajectory': actor_trajectory, 'trial_metadata': restored_metadata, 'llm_usage': llm_usage}
    return out

def _normalize_webshop_controller(value: object) -> str:
    controller = str(value or 'none').strip().lower().replace('-', '_')
    if controller not in {'none', 'dsst'}:
        raise ValueError(f'controller must be one of none or dsst; got {value!r}')
    return controller

def _validate_shared_environment_identity(shared_t1_dir: str | Path | None, environment_identity: Any) -> None:
    """Reject an unknown or different native environment before any trial runs."""
    if environment_identity is None:
        return  # Synthetic factories do not claim a verified native identity.
    if (not isinstance(environment_identity, dict)
            or environment_identity.get('benchmark') != 'webshop'
            or not isinstance(environment_identity.get('source_data_index_root_sha256'), str)
            or re.fullmatch('[0-9a-f]{64}', environment_identity['source_data_index_root_sha256']) is None):
        raise ValueError('WebShop environment identity must identify a verified source/data/index')
    if not shared_t1_dir:
        return
    source = Path(shared_t1_dir) / 'results.json'
    try:
        summary = json.loads(source.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError('WebShop shared T1 is missing a readable environment identity') from exc
    if not isinstance(summary, dict) or summary.get('environment_identity') is None:
        raise ValueError('WebShop shared T1 is missing its verified environment identity')
    if summary['environment_identity'] != environment_identity:
        raise ValueError('WebShop shared T1 environment identity differs from the current source/data/index')

def run(*, num_envs: int=10, sample_indices: Optional[list[int]]=None, model: str='gpt-4o-mini', trials: int=6, er: str='none', controller: str='none', allow_gold_feedback: bool=False, output_dir: Optional[str]=None, env_factory: Optional[Callable[[], Any]]=None, llm_fn: Optional[Callable[..., str]]=None, max_steps: int=DEFAULT_MAX_STEPS, max_environment_steps: Optional[int]=None, shared_t1_dir: Optional[str]=None, max_total_tokens: Optional[int]=None, require_provider_usage: bool=False) -> list[EpisodeResult]:
    """Run WebShop under the unified VEX2 loop."""
    del allow_gold_feedback
    controller = _normalize_webshop_controller(controller)
    retry_source_method = er == 'none' and controller == 'none'
    if require_provider_usage:
        if retry_source_method and shared_t1_dir:
            raise ValueError('formal WebShop retry source cannot consume shared T1')
        if not retry_source_method and (not shared_t1_dir):
            raise ValueError('formal WebShop comparison methods must consume the exact retry T1')
    if output_dir is None:
        output_dir = str(Path.cwd() / 'runs' / 'webshop')
    os.makedirs(output_dir, exist_ok=True)
    resolved_max_environment_steps = int(max_steps) if max_environment_steps is None else int(max_environment_steps)
    if resolved_max_environment_steps <= 0:
        raise ValueError('max_environment_steps must be positive')
    selected = sorted(set((int(i) for i in sample_indices))) if sample_indices is not None else list(range(int(num_envs)))
    uses_official_runtime = env_factory is None
    if uses_official_runtime:
        invalid_test_indices = [index for index in selected if not WEBSHOP_OFFICIAL_TEST_GOAL_START <= index <= WEBSHOP_OFFICIAL_TEST_GOAL_END]
        if invalid_test_indices:
            raise ValueError(f'official WebShop evaluation uses human test goal indices {WEBSHOP_OFFICIAL_TEST_GOAL_START}..{WEBSHOP_OFFICIAL_TEST_GOAL_END}; got {invalid_test_indices}')
        env_factory = _load_default_env_factory()
    environment_identity = deepcopy(getattr(env_factory, '_vrlbench_environment_identity', None))
    _validate_shared_environment_identity(shared_t1_dir, environment_identity)
    vex = VEX2(er=er, controller=controller, allow_gold_feedback=False, model=model, trials=trials, max_total_tokens=max_total_tokens, require_provider_usage=require_provider_usage)
    manager = make_manager()
    if shared_t1_dir:
        validate_shared_t1_source(shared_t1_dir, benchmark='webshop', model=model, max_environment_steps=resolved_max_environment_steps, require_complete_accounting=require_provider_usage, expected_provider=os.environ.get('MODEL_PROVIDER') if require_provider_usage else None, expected_base_url=os.environ.get('OPENAI_BASE_URL', '') if require_provider_usage else None)
    shared_t1 = _load_shared_t1(shared_t1_dir, require_complete_accounting=require_provider_usage, expected_max_environment_steps=resolved_max_environment_steps)
    episodes: list[EpisodeResult] = []
    names: list[str] = []
    evidence_rows: list[dict[str, Any]] = []

    def _write_checkpoint(*, final: bool) -> None:
        write_env_results(episodes, output_dir, total_trials=trials, names=names)
        solved_metrics = completion_metrics(episodes, total_trials=trials)
        sr_curve = list(per_trial_sr(episodes, trials).values())
        summary = {'benchmark': 'webshop', 'vrl_task_sr': sum((ep.solved for ep in episodes)) / len(episodes) if episodes else 0.0, 'sr_at_t': solved_metrics['sr_at_t'], 'auc': sum(sr_curve) / len(sr_curve) if sr_curve else 0.0, 'avg_t_at_t': solved_metrics['avg_t_at_t'], 'n': len(episodes), 'n_planned': len(selected), 'num_total': len(episodes), 'num_solved': sum((1 for ep in episodes if ep.solved)), 'trials': trials, 'num_trials': trials, 'model': model, 'er': er, 'controller': controller, 'vex_method_layer': controller if controller != 'none' else None, 'controller_input': 'canonical_policy_ao_forest' if controller == 'dsst' else None, 'sample_indices': selected, 'max_steps': max_steps, 'max_environment_steps': resolved_max_environment_steps, 'official_runtime_verified': uses_official_runtime, 'max_total_tokens': max_total_tokens, 'require_provider_usage': bool(require_provider_usage), 'resource_usage': resource_usage_summary(episodes), 'resource_accounting_basis': 'logical_per_condition_including_shared_t1', 'checkpoint': not final, 'completed_episodes': len(episodes), 'infra_failures': []}
        summary['environment_identity'] = environment_identity
        atomic_write_json(Path(output_dir, 'episodes.json'), evidence_rows)
        atomic_write_json(Path(output_dir, 'results.json'), summary)
    for sample_index in selected:
        env = env_factory()
        t1 = shared_t1.get(sample_index)
        if t1 is None and shared_t1_dir:
            raise KeyError(f'WebShop sample_index={sample_index}: shared T1 missing')
        t1_trajectory = str(t1.get('trajectory') or '') if t1 is not None else ''
        if t1 is not None and (not bool(t1.get('success', False))) and (not t1_trajectory.strip()):
            raise ValueError(f'WebShop shared T1 failed trial is missing trajectory for sample_index={sample_index}; refusing to update adapted memory from placeholder evidence.')
        actor = make_actor(env, sample_index=sample_index, model=model, llm_fn=llm_fn, max_steps=max_steps, max_environment_steps=resolved_max_environment_steps, shared_t1_trajectory=t1_trajectory)
        if t1 is None:
            episode = vex.run(actor=actor, manager=manager, metadata={'benchmark': 'webshop', 'basic_environment_rules': WEBSHOP_DSST_BASIC_ENVIRONMENT_RULES, 'sample_index': sample_index, 'controller': controller})
        else:
            er_slot = make_er_slot(er)
            er_slot.reset()
            controller_slot = make_controller(controller, allow_gold_feedback=False, model=model)
            if controller_slot:
                controller_slot.reset()
            t1_success = bool(t1.get('success', False))
            initial_result, token_ledger = prepare_shared_t1(trajectory=t1_trajectory, policy_trajectory=str(t1.get('policy_trajectory') or ''), actor_trajectory=str(t1.get('actor_trajectory') or ''), success=t1_success, er_slot=er_slot, model=model, total_trials=trials, controller=controller_slot, metadata={'benchmark': 'webshop', 'basic_environment_rules': WEBSHOP_DSST_BASIC_ENVIRONMENT_RULES, 'sample_index': sample_index, 'controller': controller}, trial_metadata=t1.get('trial_metadata'), llm_usage=t1.get('llm_usage'), max_total_tokens=max_total_tokens, require_provider_usage=require_provider_usage)
            if t1_success:
                episode = initial_result
            else:
                episode = run_episode(actor=actor, manager=manager, er_slot=er_slot, model=model, total_trials=trials, controller=controller_slot, metadata={'benchmark': 'webshop', 'basic_environment_rules': WEBSHOP_DSST_BASIC_ENVIRONMENT_RULES, 'sample_index': sample_index, 'controller': controller}, initial_result=initial_result, reset_state=False, token_ledger=token_ledger)
        episodes.append(episode)
        name = f'sample_{sample_index:04d}'
        names.append(name)
        evidence_rows.append({'name': name, 'sample_index': sample_index, 'solved': episode.solved, 'solve_trial': episode.solve_trial, 'resource_usage': episode.resource_usage, 'trials': [{'trial': trial.trial, 'success': trial.success, 'trial_metadata': trial.trial_metadata, 'llm_usage': trial.llm_usage, 'context': trial.context, 'system_context': trial.system_context, 'user_context': trial.user_context, 'trajectory': trial.trajectory, 'policy_trajectory': trial.policy_trajectory, 'actor_trajectory': trial.actor_trajectory} for trial in episode.trials]})
        logger.info('webshop sample=%s solved=%s trial=%s', sample_index, episode.solved, episode.solve_trial)
        _write_checkpoint(final=False)
    _write_checkpoint(final=True)
    return episodes

def _sample_index_from_session(session: str) -> Optional[int]:
    fixed_match = re.match('^(?:vex2_)?fixed_(\\d+)$', str(session))
    if fixed_match:
        return int(fixed_match.group(1))
    match = re.match('^vex2_(\\d+)_\\d+$', str(session))
    if match:
        return int(match.group(1))
    text = str(session)
    return int(text) if text.isdigit() else None

def _available_webshop_actions(available: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    if available.get('has_search_bar'):
        actions.append('search[query]')
    for clickable in available.get('clickables', []) or []:
        text = str(clickable).strip().lower()
        if not text or text.lower() == 'search':
            continue
        actions.append(f'click[{text}]')
    return actions

def _load_default_env_factory() -> Callable[[], Any]:
    root = DEFAULT_WEBSHOP_ROOT
    if root is None:
        raise ImportError('Set WEBSHOP_ROOT to the prepared official WebShop checkout before running WebShop. The dependency is not installed inside vrlbench.')
    adapter = os.environ.get('WEBSHOP_ENV_ADAPTER', 'official_text').strip().lower()
    if adapter != 'official_text':
        raise ValueError('WebShop evaluation requires WEBSHOP_ENV_ADAPTER=official_text; alternative wrappers are not supported by this adapter')
    _validate_official_webshop_runtime_settings()
    root = Path(root).expanduser().resolve()
    observed = webshop_observed_dependency_contract(root)
    product_file = root / WEBSHOP_FULL_DATA_FILES[0]
    attribute_file = root / WEBSHOP_FULL_DATA_FILES[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    with _pushd(root):
        from web_agent_site.envs import WebAgentTextEnv
        from web_agent_site.engine import engine as webshop_engine
    webshop_engine.DEFAULT_ATTR_PATH = str(attribute_file)
    factory = _make_official_webshop_env_factory(WebAgentTextEnv, product_file)
    factory._vrlbench_environment_identity = {
        'benchmark': 'webshop',
        'source_data_index_root_sha256': observed['source_data_index_root_sha256'],
    }
    return factory

def _validate_official_webshop_runtime_settings() -> None:
    data_profile = os.environ.get('WEBSHOP_DATA_PROFILE', 'full').strip().lower()
    if data_profile != 'full':
        raise ValueError('official WebShop evaluation requires WEBSHOP_DATA_PROFILE=full')
    observation_mode = os.environ.get('WEBSHOP_OBSERVATION_MODE', 'text').strip()
    if observation_mode != 'text':
        raise ValueError('official WebShop evaluation requires WEBSHOP_OBSERVATION_MODE=text')
    human_goals = os.environ.get('WEBSHOP_HUMAN_GOALS', '1').strip().lower()
    if human_goals != '1':
        raise ValueError('official WebShop evaluation requires WEBSHOP_HUMAN_GOALS=1')
    goal_seed = os.environ.get('WEBSHOP_GOAL_SEED', '233').strip()
    if goal_seed != '233':
        raise ValueError('official WebShop evaluation requires WEBSHOP_GOAL_SEED=233')
    if os.environ.get('WEBSHOP_NUM_PRODUCTS', '').strip():
        raise ValueError('official WebShop full-data evaluation requires WEBSHOP_NUM_PRODUCTS unset')

def _make_official_webshop_env_factory(env_type: Callable[..., Any], product_file: Path) -> Callable[[], _OfficialWebShopTextAdapter]:
    """Create a lazy, run-scoped singleton official environment.

    WebShop loads the full 1.18M-product catalog and opens Lucene while building
    ``SimServer``. Runs are sequential, so one text environment can safely serve
    every sample. ``make_actor`` still performs the official integer-session
    reset at the start of every trial, which clears navigation state while
    preserving the fixed human test goal.
    """
    shared_adapter: dict[str, _OfficialWebShopTextAdapter] = {}
    common = {'observation_mode': 'text', 'session_prefix': os.environ.get('WEBSHOP_SESSION_PREFIX', 'vex2_')}

    def make_env() -> _OfficialWebShopTextAdapter:
        if 'adapter' not in shared_adapter:
            _seed_official_webshop_goal_generation()
            env = env_type(**common, file_path=str(product_file), num_products=None, human_goals=1)
            server = env.server
            goals = getattr(server, 'goals', ())
            if len(goals) <= WEBSHOP_OFFICIAL_TEST_GOAL_END:
                raise RuntimeError(f'official WebShop human-goal server does not contain the full test split 0..{WEBSHOP_OFFICIAL_TEST_GOAL_END}')
            shared_adapter['adapter'] = _OfficialWebShopTextAdapter(env)
        return shared_adapter['adapter']
    return make_env

def _seed_official_webshop_goal_generation() -> None:
    """Stabilize official WebShop product prices before SimServer creation.

    The pinned runtime samples missing product prices before applying its own
    fixed human-goal shuffle. Re-seeding once per condition keeps integer test
    goal indices stable across separate baseline and comparison processes.
    """
    random.seed(233)

@contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
