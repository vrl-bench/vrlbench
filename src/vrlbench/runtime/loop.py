from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
from typing import Any, Callable, List, Mapping, Optional
import vrlbench.runtime.prompts as P
from vrlbench.runtime import llm as _llm
from vrlbench.runtime.budget import TokenBudgetExceeded, TokenUsageLedger
from vrlbench.runtime.controllers import InterEpisodeController
from vrlbench.runtime.er.base import ERSlot
from vrlbench.runtime.shared_t1 import validate_formal_trial_environment_accounting, validate_shared_t1_environment_accounting
from vrlbench.runtime.trajectory import intern_repeated_policy_observations, normalize_policy_evidence, parse_policy_trajectory, render_policy_trajectory, strip_visual_artifact_lines

@contextmanager
def _public_environment_context(metadata: Mapping[str, Any]):
    """Expose the same public benchmark rules to every memory operation."""
    rules = str(metadata.get('basic_environment_rules') or '').strip()
    if not rules:
        yield
        return
    previous = _llm.get_context()
    context = '[Public benchmark environment]\nThe following interface rules are available equally to every method. They contain no hidden task state or outcome oracle.\n' + rules
    _llm.set_context(context)
    try:
        yield
    finally:
        _llm.set_context(previous)

@dataclass
class TrialResult:
    trial: int
    success: bool
    trajectory: str
    context: str = ''
    temporal_decision: Optional[dict[str, Any]] = None
    spatial_state_pre: Optional[dict[str, Any]] = None
    spatial_state: Optional[dict[str, Any]] = None
    controller_decision: Optional[dict[str, Any]] = None
    controller_state_pre: Optional[dict[str, Any]] = None
    controller_state: Optional[dict[str, Any]] = None
    trial_metadata: dict[str, Any] = field(default_factory=dict)
    llm_usage: dict[str, Any] = field(default_factory=dict)
    system_context: str = ''
    user_context: str = ''
    policy_trajectory: str = ''
    actor_trajectory: str = ''

@dataclass
class EpisodeResult:
    trials: List[TrialResult] = field(default_factory=list)
    resource_usage: dict[str, Any] = field(default_factory=dict)

    @property
    def solved(self) -> bool:
        return any((t.success for t in self.trials))

    @property
    def solve_trial(self) -> Optional[int]:
        for t in self.trials:
            if t.success:
                return t.trial
        return None

    @property
    def num_trials_used(self) -> int:
        return len(self.trials)

def snapshot_controller_after_success(controller: Any | None) -> Optional[dict[str, Any]]:
    """Return a post-success controller snapshot only when it allows it.

    DSST is failure-only: successful trajectories stop the episode
    and must not become cross-trial search-map evidence. Custom controllers
    keep their configured behavior unless they expose
    ``snapshot_after_success_enabled()``.
    """
    if controller is None:
        return None
    snapshot_gate = getattr(controller, 'snapshot_after_success_enabled', None)
    if callable(snapshot_gate) and (not bool(snapshot_gate())):
        return None
    return controller.snapshot()
_HARNESS_TRIAL_METADATA_KEYS = frozenset({'policy_ao_steps', 'policy_evidence_schema_version', 'policy_trajectory_sha256', 'policy_trajectory_source'})

def _bind_controller_task_scope(controller: Any | None, metadata: Mapping[str, Any]) -> None:
    """Bind a benchmark-declared decision problem without interpreting it."""
    setter = getattr(controller, 'set_task_scope', None)
    if not callable(setter):
        return
    scope = str(metadata.get('task_scope') or '').strip()
    if scope:
        setter(scope)

def _bind_controller_basic_environment_rules(controller: Any | None, metadata: Mapping[str, Any]) -> None:
    """Bind public benchmark mechanics without interpreting failure evidence."""
    setter = getattr(controller, 'set_basic_environment_rules', None)
    if not callable(setter):
        return
    rules = str(metadata.get('basic_environment_rules') or '').strip()
    if not rules:
        raise ValueError('DSST metadata lacks basic environment rules')
    setter(rules)

def _observe_controller_trial_task(controller: Any | None, task: object) -> None:
    observer = getattr(controller, 'observe_trial_task', None)
    if callable(observer):
        observer(str(task or ''))

def _merge_actor_metadata(target: dict[str, Any], incoming: Mapping[str, Any], *, source: str) -> None:
    overlap = sorted(set(incoming) & _HARNESS_TRIAL_METADATA_KEYS)
    if overlap:
        raise ValueError(f'{source} attempted to overwrite harness-owned trial metadata: {overlap!r}')
    target.update(incoming)

def _validated_policy_trajectory(*, evidence: Mapping[str, Any], trial_metadata: Mapping[str, Any]) -> tuple[str, int]:
    del trial_metadata
    normalized = normalize_policy_evidence(evidence)
    return (render_policy_trajectory(normalized), len(normalized['steps']))

def _notify_closed_controller_trial(controller: Any | None, *, trajectory: str,
                                    actor_trajectory: str, trial_index: int,
                                    close_reason: str, policy_evidence_eligible: bool,
                                    has_future_trial: bool, model: str,
                                    er_name: str, er_section: str='') -> None:
    """Report a closed opportunity without invoking a next-trial update.

    A valid final failure still has an executed trace. New controllers can
    consume that terminal record; old controllers retain their budget-only
    notification, without acquiring a new required protocol method.
    """
    if controller is None:
        return
    if policy_evidence_eligible:
        hook = getattr(controller, 'on_trial_closed_with_policy_evidence', None)
        if callable(hook):
            hook(trajectory, model, trial_index=trial_index, close_reason=close_reason,
                 has_future_trial=has_future_trial, actor_trajectory=actor_trajectory)
            return
    hook = getattr(controller, 'on_trial_closed_without_policy_evidence', None)
    if callable(hook):
        hook(trial_index=trial_index, close_reason=close_reason,
             has_future_trial=has_future_trial, model=model,
             er_name=er_name, er_section=er_section)

def prepare_shared_t1(*, trajectory: str, policy_trajectory: str | None=None, actor_trajectory: str | None=None, success: bool, er_slot: ERSlot, model: str, total_trials: int, controller: Optional[InterEpisodeController]=None, metadata: Optional[dict[str, Any]]=None, trial_metadata: Optional[Mapping[str, Any]]=None, llm_usage: Optional[Mapping[str, Any]]=None, max_total_tokens: Optional[int]=None, require_provider_usage: bool=False) -> tuple[EpisodeResult, TokenUsageLedger]:
    """Restore a shared first trial and replay its post-failure lifecycle.

    The shared trial itself is produced once by the common no-memory actor.
    This helper attaches that actor's resource ledgers, then performs exactly
    the memory read/write and decision-space updates that would have followed
    the same failed trial in :func:`run_episode`.  Keeping this path in the
    task-agnostic loop prevents benchmark adapters from silently diverging.
    """
    loop_metadata = dict(metadata or {})
    if type(total_trials) is not int or total_trials < 1:
        raise ValueError('total_trials must be a positive integer')
    shared_trial_metadata = dict(trial_metadata or {})
    benchmark = str(loop_metadata.get('benchmark') or '').strip().lower()
    formal_online_benchmark = benchmark in {'miniwob', 'webshop', 'webarena', 'osworld'}
    shared_policy_task = ''
    if formal_online_benchmark:
        max_environment_steps = shared_trial_metadata.get('max_environment_steps')
        if type(max_environment_steps) is not int:
            raise ValueError('shared T1 lacks a formal environment-step horizon')
        shared_trial_metadata = validate_shared_t1_environment_accounting(shared_trial_metadata, benchmark=benchmark, source='prepare_shared_t1', expected_max_environment_steps=max_environment_steps, success=bool(success))
    shared_policy_evidence_eligible = shared_trial_metadata.get('policy_evidence_eligible', True) is True
    if success and (not shared_policy_evidence_eligible):
        raise ValueError('a successful shared T1 cannot be ineligible policy evidence')
    if formal_online_benchmark:
        parsed_policy = parse_policy_trajectory(str(policy_trajectory or ''))
        if parsed_policy is None:
            raise ValueError('shared T1 lacks canonical executed policy A-O evidence')
        canonical_policy_trajectory, policy_ao_steps = _validated_policy_trajectory(evidence=parsed_policy, trial_metadata=shared_trial_metadata)
        expected_policy_hash = hashlib.sha256(canonical_policy_trajectory.encode('utf-8')).hexdigest()
        if shared_trial_metadata.get('policy_trajectory_sha256') != expected_policy_hash:
            raise ValueError('shared T1 policy trajectory hash is missing or invalid')
        if shared_trial_metadata.get('policy_ao_steps') != policy_ao_steps:
            raise ValueError('shared T1 policy A-O count is inconsistent')
        if shared_trial_metadata.get('policy_evidence_schema_version') != 1:
            raise ValueError('shared T1 policy evidence schema is inconsistent')
        if shared_trial_metadata.get('protocol_failure') is True and shared_policy_evidence_eligible and (policy_ao_steps == 0):
            raise ValueError('shared-T1 protocol failure cannot contribute empty policy evidence')
        policy_trajectory = canonical_policy_trajectory
        shared_policy_task = str(parsed_policy.get('task') or '')
    else:
        policy_trajectory = str(policy_trajectory or strip_visual_artifact_lines(trajectory))
    ledger = TokenUsageLedger(max_total_tokens=max_total_tokens, require_provider_usage=require_provider_usage)
    prefix_usage_complete = isinstance(llm_usage, Mapping) and int(llm_usage.get('api_calls') or 0) > 0 and bool(llm_usage.get('provider_usage_complete', False))
    if isinstance(llm_usage, Mapping):
        ledger.absorb(llm_usage)
    if not prefix_usage_complete:
        ledger.mark_unaccounted_prefix()
    prefix_snapshot = ledger.snapshot()
    if require_provider_usage and (not prefix_snapshot['provider_usage_complete']):
        raise ValueError('shared T1 is missing complete provider-reported LLM usage; configured runs require a newly accounted shared prefix')
    if max_total_tokens is not None and prefix_snapshot['total_tokens'] > max_total_tokens:
        raise TokenBudgetExceeded(f"shared T1 already exceeds the strict token budget: used={prefix_snapshot['total_tokens']}, max={max_total_tokens}")
    active_controller = controller
    if active_controller:
        active_controller.set_total_budget(total_trials)
        _bind_controller_task_scope(active_controller, loop_metadata)
        _bind_controller_basic_environment_rules(active_controller, loop_metadata)
        if shared_policy_task:
            _observe_controller_trial_task(active_controller, shared_policy_task)
    if formal_online_benchmark:
        if not isinstance(actor_trajectory, str) or not actor_trajectory.strip():
            raise ValueError('shared T1 lacks the raw Actor trajectory')
    actor_trajectory = str(actor_trajectory or trajectory)
    if formal_online_benchmark:
        expected_actor_hash = hashlib.sha256(actor_trajectory.encode('utf-8')).hexdigest()
        if shared_trial_metadata.get('actor_trajectory_sha256') != expected_actor_hash:
            raise ValueError('shared T1 raw Actor trajectory hash is inconsistent')
    if benchmark == 'miniwob' and shared_trial_metadata.get('actor_profile') == 'react_batch':
        memory_actor_trajectory, memory_transport = intern_repeated_policy_observations(actor_trajectory, policy_trajectory)
        shared_trial_metadata['memory_actor_trajectory_transport'] = memory_transport
    else:
        memory_actor_trajectory = actor_trajectory
    lifecycle_metadata = {**loop_metadata, **shared_trial_metadata}
    if not success and total_trials > 1:
        memory_trajectory = strip_visual_artifact_lines(policy_trajectory)
        with _llm.token_budget_scope(ledger):
            if shared_policy_evidence_eligible:
                with _public_environment_context(loop_metadata):
                    er_slot.on_trial_end(memory_trajectory, actor_trajectory=memory_actor_trajectory, success=False, model=model, trial_index=0, total_trials=total_trials, metadata={**lifecycle_metadata, 'phase': 'shared_t1', 'has_future_trial': True})
            if active_controller and shared_policy_evidence_eligible:
                with _public_environment_context(loop_metadata):
                    er_section = er_slot.read_for_trial(trial_index=1, total_trials=total_trials, model=model, past_trajectories=[memory_trajectory], metadata={**lifecycle_metadata, 'phase': 'post_failure_spatial'})
                active_controller.on_failure(policy_trajectory, model, er_name=er_slot.__class__.__name__, er_section=er_section, trial_index=0)
            elif active_controller:
                _notify_closed_controller_trial(active_controller, trajectory=policy_trajectory,
                    actor_trajectory=actor_trajectory, trial_index=0,
                    close_reason=str(shared_trial_metadata.get('trial_close_reason') or 'actor_protocol_failure'),
                    policy_evidence_eligible=False, has_future_trial=True,
                    model=model, er_name=er_slot.__class__.__name__)
    elif success and active_controller:
        active_controller.on_success(policy_trajectory, trial_index=0)
    elif not success and active_controller:
        with _llm.token_budget_scope(ledger):
            _notify_closed_controller_trial(active_controller, trajectory=policy_trajectory,
                actor_trajectory=actor_trajectory, trial_index=0,
                close_reason=str(shared_trial_metadata.get('trial_close_reason') or (
                    'trial_budget_exhausted' if shared_policy_evidence_eligible else 'actor_protocol_failure')),
                policy_evidence_eligible=shared_policy_evidence_eligible,
                has_future_trial=False, model=model, er_name=er_slot.__class__.__name__)
    usage_snapshot = ledger.snapshot()
    initial_trial = TrialResult(trial=0, success=bool(success), trajectory=trajectory, policy_trajectory=policy_trajectory, actor_trajectory=actor_trajectory, controller_state=snapshot_controller_after_success(active_controller) if success and active_controller else active_controller.snapshot() if active_controller else None, spatial_state=snapshot_controller_after_success(active_controller) if success and active_controller else active_controller.snapshot() if active_controller else None, trial_metadata=shared_trial_metadata, llm_usage=usage_snapshot)
    return (EpisodeResult(trials=[initial_trial], resource_usage=usage_snapshot), ledger)

def run_episode(actor: Callable[[], str], manager: Callable[[str], bool], er_slot: ERSlot, model: str, total_trials: int, controller: Optional[InterEpisodeController]=None, initial_result: Optional[EpisodeResult]=None, reset_state: bool=True, metadata: Optional[dict[str, Any]]=None, token_ledger: Optional[TokenUsageLedger]=None, max_total_tokens: Optional[int]=None, require_provider_usage: bool=False, on_trial: Callable[[EpisodeResult], None] | None=None) -> EpisodeResult:
    """Run one task instance while recording total provider tokens."""
    if type(total_trials) is not int or total_trials < 1:
        raise ValueError('total_trials must be a positive integer')
    if token_ledger is not None and max_total_tokens is not None:
        raise ValueError('pass either token_ledger or max_total_tokens, not both')
    ledger = token_ledger or TokenUsageLedger(max_total_tokens=max_total_tokens, require_provider_usage=require_provider_usage)
    if token_ledger is None and initial_result is not None:
        seeded = False
        for trial in initial_result.trials:
            if trial.llm_usage:
                ledger.absorb(trial.llm_usage)
                seeded = True
        if not seeded and initial_result.resource_usage:
            ledger.absorb(initial_result.resource_usage)
    with _llm.token_budget_scope(ledger):
        result = _run_episode_impl(actor=actor, manager=manager, er_slot=er_slot, model=model, total_trials=total_trials, controller=controller, initial_result=initial_result, reset_state=reset_state, metadata=metadata, usage_ledger=ledger, on_trial=on_trial)
    result.resource_usage = ledger.snapshot()
    return result

def _run_episode_impl(actor: Callable[[], str], manager: Callable[[str], bool], er_slot: ERSlot, model: str, total_trials: int, controller: Optional[InterEpisodeController]=None, initial_result: Optional[EpisodeResult]=None, reset_state: bool=True, metadata: Optional[dict[str, Any]]=None, usage_ledger: TokenUsageLedger | None=None, on_trial: Callable[[EpisodeResult], None] | None=None) -> EpisodeResult:
    """Run one episode under the Actor-Manager loop.

    Args:
        actor:    ``() -> trajectory: str``.
                  Working memory is injected via the LLM layer automatically.
        manager:  ``(trajectory: str) -> bool``.
        er_slot:  Replay slot instance (reset at start).
        model:    LLM deployment for ER / spatial memory generation.
    """
    loop_metadata = dict(metadata or {})
    benchmark = str(loop_metadata.get('benchmark') or '').strip().lower()
    formal_online_benchmark = benchmark in {'miniwob', 'webshop', 'webarena', 'osworld'}
    usage_ledger = usage_ledger or TokenUsageLedger()
    active_controller: Any | None = controller
    if reset_state:
        er_slot.reset()
        if active_controller:
            active_controller.reset()
    if active_controller:
        active_controller.set_total_budget(total_trials)
        _bind_controller_task_scope(active_controller, loop_metadata)
        _bind_controller_basic_environment_rules(active_controller, loop_metadata)
    result = EpisodeResult(trials=list(initial_result.trials) if initial_result is not None else [])
    if result.solved:
        return result

    def policy_trajectories() -> list[str]:
        return [strip_visual_artifact_lines(tr.policy_trajectory or tr.trajectory) for tr in result.trials if tr.trial_metadata.get('policy_evidence_eligible', True) is not False]
    for t in range(len(result.trials), total_trials):
        continuation_failure_hook = getattr(active_controller, 'continuation_failure', None)
        continuation_failure = continuation_failure_hook() if callable(continuation_failure_hook) else None
        if continuation_failure is not None:
            if not result.trials or not isinstance(continuation_failure, dict):
                raise RuntimeError('controller reported a continuation failure without a prior trial')
            last_trial = result.trials[-1]
            if 'controller_method_failure' in last_trial.trial_metadata:
                raise RuntimeError('controller method failure was recorded twice')
            last_trial.trial_metadata['controller_method_failure'] = {**continuation_failure, 'after_trial': last_trial.trial, 'prevented_trial': t}
            break
        usage_before = usage_ledger.snapshot()
        trial_metadata: dict[str, Any] = {}
        trial_hook = getattr(actor, 'set_trial_context', None)
        if callable(trial_hook):
            hook_value = trial_hook(trial_index=t, total_trials=total_trials)
            if isinstance(hook_value, dict):
                _merge_actor_metadata(trial_metadata, hook_value, source='actor.set_trial_context')
        if t > 0 and getattr(active_controller, 'requires_current_trial_state', False):
            prepare_hook = getattr(actor, 'prepare_trial', None)
            if not callable(prepare_hook):
                raise ValueError('controller requires an actor that exposes the current reset state')
            prepared = prepare_hook()
            if not isinstance(prepared, Mapping):
                raise ValueError('actor.prepare_trial must return a mapping')
            current_task = str(prepared.get('task') or '').strip()
            current_observation = str(prepared.get('initial_observation') or '')
            if not current_task or not current_observation.strip():
                raise ValueError('actor.prepare_trial must expose task and initial_observation')
            _observe_controller_trial_task(active_controller, current_task)
            current_state_hook = getattr(active_controller, 'set_current_trial_state', None)
            if not callable(current_state_hook):
                raise ValueError('controller requires current state but lacks its binding hook')
            current_state_hook(trial_index=t, task=current_task, initial_observation=current_observation)
        sections: List[str] = []
        trial_info = ''
        spatial_section = ''
        verbatim_system_context: str | None = None
        controller_decision = None
        er_section = ''
        user_context = ''
        if t > 0:
            with _public_environment_context(loop_metadata):
                er_section = er_slot.read_for_trial(trial_index=t, total_trials=total_trials, model=model, past_trajectories=policy_trajectories(), metadata={**loop_metadata, 'phase': 'trial_start'})
        else:
            er_section = ''
        if controller is not None:
            controller_view = controller.build_trial_context(trial_index=t, total_trials=total_trials, past_trajectories=policy_trajectories(), model=model, er_name=er_slot.__class__.__name__, er_section=er_section)
            trial_info = controller_view.trial_info
            spatial_section = controller_view.section
            verbatim_system_context = controller_view.verbatim_system_context
            controller_decision = controller_view.decision
        if t > 0 or spatial_section:
            if spatial_section:
                sections.append(spatial_section)
            if er_section:
                placement = er_slot.read_placement()
                if placement == 'system':
                    sections.append(er_section)
                elif placement == 'user_prefix':
                    user_context = er_section
                else:
                    raise ValueError(f'unsupported ER read placement {placement!r} for {er_slot.__class__.__name__}')
        if verbatim_system_context is not None:
            if sections or trial_info or user_context or spatial_section:
                raise ValueError('verbatim controller context cannot be combined with another cross-trial prompt')
            context = verbatim_system_context
        elif sections or trial_info:
            preamble = P.WM_PREAMBLE.format(trial_info=trial_info)
            inner = '\n\n'.join(sections) if sections else ''
            context = f'\n\n{P.WM_TAG_OPEN}\n{preamble}\n\n{inner}\n\n' if inner else f'\n\n{P.WM_TAG_OPEN}\n{preamble}\n\n'
            context += f'{P.working_memory_directive(spatial_section)}\n{P.WM_TAG_CLOSE}\n\n'
        else:
            context = ''
        controller_state_pre = active_controller.snapshot() if active_controller else None
        _llm.set_context(context)
        _llm.set_user_context(user_context)
        try:
            trajectory = actor()
        finally:
            _llm.set_context('')
            _llm.set_user_context('')
        actor_trajectory_hook = getattr(actor, 'get_actor_trajectory', None)
        if formal_online_benchmark and (not callable(actor_trajectory_hook)):
            raise ValueError('formal actor lacks the raw Actor trajectory hook')
        actor_trajectory = str(actor_trajectory_hook()) if callable(actor_trajectory_hook) else strip_visual_artifact_lines(trajectory)
        runtime_metadata_hook = getattr(actor, 'get_trial_metadata', None)
        if callable(runtime_metadata_hook):
            runtime_metadata = runtime_metadata_hook()
            if isinstance(runtime_metadata, dict):
                _merge_actor_metadata(trial_metadata, runtime_metadata, source='actor.get_trial_metadata')
        if formal_online_benchmark and (not actor_trajectory.strip()):
            raise ValueError('formal actor lacks a raw Actor trajectory')
        trial_metadata['actor_trajectory_sha256'] = hashlib.sha256(actor_trajectory.encode('utf-8')).hexdigest()
        if formal_online_benchmark:
            for field_name in ('protocol_failure', 'policy_evidence_eligible'):
                if type(trial_metadata.get(field_name)) is not bool:
                    raise ValueError(f'formal actor must report boolean {field_name} metadata')
        policy_evidence_eligible = trial_metadata.get('policy_evidence_eligible', True) is True
        success = manager(trajectory)
        if formal_online_benchmark:
            max_environment_steps = trial_metadata.get('max_environment_steps')
            if type(max_environment_steps) is not int:
                raise ValueError('formal actor lacks an environment-step horizon')
            trial_metadata = validate_formal_trial_environment_accounting(trial_metadata, benchmark=benchmark, source=f'live trial {t}', expected_max_environment_steps=max_environment_steps, success=bool(success), expected_trial_index=t)
        if success and (not policy_evidence_eligible):
            raise ValueError('a successful trial cannot be ineligible policy evidence')
        policy_evidence_hook = getattr(actor, 'get_policy_evidence', None)
        if callable(policy_evidence_hook):
            policy_trajectory, policy_ao_steps = _validated_policy_trajectory(evidence=policy_evidence_hook(), trial_metadata=trial_metadata)
            policy_trajectory_source = 'executed_actor_ao_hook'
        else:
            if formal_online_benchmark:
                raise ValueError('formal online benchmark actor lacks executed policy A-O evidence')
            policy_trajectory = strip_visual_artifact_lines(trajectory)
            policy_ao_steps = -1
            policy_trajectory_source = 'plain_trajectory_fallback'
        if trial_metadata.get('protocol_failure') is True and policy_evidence_eligible and (policy_ao_steps == 0):
            raise ValueError('actor protocol failure cannot contribute empty policy evidence')
        _observe_controller_trial_task(active_controller, trial_metadata.get('task') or (parse_policy_trajectory(policy_trajectory) or {}).get('task'))
        trial_metadata.update({'policy_ao_steps': policy_ao_steps, 'policy_evidence_schema_version': 1 if policy_trajectory_source == 'executed_actor_ao_hook' else None, 'policy_trajectory_sha256': hashlib.sha256(policy_trajectory.encode('utf-8')).hexdigest(), 'policy_trajectory_source': policy_trajectory_source})
        if benchmark == 'miniwob' and trial_metadata.get('actor_profile') == 'react_batch':
            memory_actor_trajectory, memory_transport = intern_repeated_policy_observations(actor_trajectory, policy_trajectory)
            trial_metadata['memory_actor_trajectory_transport'] = memory_transport
        else:
            memory_actor_trajectory = actor_trajectory
        trial_lifecycle_metadata = {**loop_metadata, **trial_metadata}
        memory_trajectory = strip_visual_artifact_lines(policy_trajectory)
        temporal_decision = controller_decision
        has_future_trial = t + 1 < total_trials
        if policy_evidence_eligible and (not success) and has_future_trial:
            with _public_environment_context(loop_metadata):
                er_slot.on_trial_end(memory_trajectory, actor_trajectory=memory_actor_trajectory, success=success, model=model, trial_index=t, total_trials=total_trials, metadata={**trial_lifecycle_metadata, 'phase': 'trial_end', 'has_future_trial': has_future_trial})
        if success:
            if active_controller:
                active_controller.on_success(policy_trajectory, trial_index=t)
        elif has_future_trial and policy_evidence_eligible:
            if active_controller:
                with _public_environment_context(loop_metadata):
                    next_er_section = er_slot.read_for_trial(trial_index=t + 1, total_trials=total_trials, model=model, past_trajectories=policy_trajectories() + [memory_trajectory], metadata={**trial_lifecycle_metadata, 'phase': 'post_failure_spatial'})
                active_controller.on_failure(policy_trajectory, model, er_name=er_slot.__class__.__name__, er_section=next_er_section, trial_index=t)
        elif not success and (not policy_evidence_eligible or not has_future_trial):
            _notify_closed_controller_trial(active_controller, trajectory=policy_trajectory,
                actor_trajectory=actor_trajectory, trial_index=t,
                close_reason=str(trial_metadata.get('trial_close_reason') or (
                    'trial_budget_exhausted' if policy_evidence_eligible else 'actor_protocol_failure')),
                policy_evidence_eligible=policy_evidence_eligible,
                has_future_trial=has_future_trial, model=model,
                er_name=er_slot.__class__.__name__, er_section=er_section)
        controller_state = snapshot_controller_after_success(active_controller) if success and active_controller else active_controller.snapshot() if active_controller else None
        result.trials.append(TrialResult(trial=t, success=success, trajectory=trajectory, policy_trajectory=policy_trajectory, actor_trajectory=actor_trajectory, context='\n\n'.join((value for value in (context, user_context) if value)), temporal_decision=temporal_decision, spatial_state_pre=controller_state_pre, spatial_state=controller_state, controller_decision=controller_decision, controller_state_pre=controller_state_pre, controller_state=controller_state, trial_metadata=trial_metadata, llm_usage=usage_ledger.delta(usage_before), system_context=context, user_context=user_context))
        if on_trial is not None:
            on_trial(EpisodeResult(trials=list(result.trials), resource_usage=usage_ledger.snapshot()))
        if success:
            break
    return result
