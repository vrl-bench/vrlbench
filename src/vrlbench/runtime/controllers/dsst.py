from __future__ import annotations
import hashlib
import json
from typing import Any, Sequence
from vrlbench.runtime.controllers.base import ControllerContext
import vrlbench.runtime.llm as _llm
import vrlbench.runtime.prompts as P
from vrlbench.runtime.trajectory import parse_policy_trajectory
DSST_AO_FOREST_ENCODING = 'ordered_raw_canonical_policy_ao_trace'

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def serialize_dsst_ao_forest(rows: Sequence[dict[str, Any]]) -> dict[str, object]:
    """Pass the ordered canonical A-O traces to the scheduler unchanged."""
    attempts: list[dict[str, object]] = []
    for row in rows:
        trace = str(row['ao_trace'])
        if parse_policy_trajectory(trace) is None:
            raise ValueError('formal DSST requires the canonical policy A-O trace')
        attempts.append({'attempt': int(row['trial']) + 1, 'canonical_policy_ao_trace': trace})
    return {'encoding': DSST_AO_FOREST_ENCODING, 'attempts': attempts}

class DSSTController:
    """Implement the decision-space search controller: collect the ordered
    failure-only A--O forest, request one frozen-model DSST update, and pass its
    raw response unchanged to the next Actor.
    """
    method_id = 'dsst'
    requires_current_trial_state = False

    def __init__(self, *, allow_gold_feedback: bool=False, model: str | None=None) -> None:
        del model
        if allow_gold_feedback:
            raise ValueError('formal DSST does not admit gold or oracle feedback')
        self.total_budget = 0
        self.task_scope = ''
        self.task_scope_source = ''
        self.basic_environment_rules = ''
        self.last_update_for_trial = -1
        self.failure_ao_forest: list[dict[str, Any]] = []
        self.joint_output: dict[str, Any] = {}

    def reset(self) -> None:
        self.task_scope = ''
        self.task_scope_source = ''
        self.basic_environment_rules = ''
        self.last_update_for_trial = -1
        self.failure_ao_forest.clear()
        self.joint_output = {}

    def set_total_budget(self, total_budget: int) -> None:
        budget = int(total_budget)
        if budget < 1:
            raise ValueError('DSST requires at least one complete attempt')
        self.total_budget = budget

    def set_task_scope(self, task_scope: str) -> None:
        """Bind the stable decision problem shared by all complete attempts."""
        scope = str(task_scope or '').strip()
        if not scope:
            raise ValueError('DSST task scope must be non-empty')
        if self.task_scope and self.task_scope != scope:
            raise ValueError('DSST task scope changed within one episode')
        self.task_scope = scope
        self.task_scope_source = 'benchmark_episode_scope'

    def set_basic_environment_rules(self, rules: str) -> None:
        """Bind public mechanics shared by every attempt in the benchmark."""
        normalized = str(rules or '').strip()
        if not normalized:
            raise ValueError('DSST basic environment rules must be non-empty')
        if self.basic_environment_rules and self.basic_environment_rules != normalized:
            raise ValueError('DSST basic environment rules changed within one episode')
        self.basic_environment_rules = normalized

    def observe_trial_task(self, task: str) -> None:
        """Use an observed task as scope only when the benchmark did not bind one."""
        observed = str(task or '').strip()
        if not observed:
            raise ValueError('DSST observed an empty task')
        if not self.task_scope:
            self.task_scope = observed
            self.task_scope_source = 'first_observed_task'
        elif self.task_scope_source == 'first_observed_task' and self.task_scope != observed:
            raise ValueError('DSST task changed within one fixed-task episode')

    def _update_for_trial(self, *, trial: int, model: str) -> None:
        if trial <= self.last_update_for_trial:
            raise ValueError('DSST updates must follow counted trial order')
        remaining = self.total_budget - trial
        if remaining < 1:
            raise ValueError('DSST update requested outside the complete-attempt horizon')
        if not self.task_scope:
            raise ValueError('DSST update requires a stable task scope')
        if not self.basic_environment_rules:
            raise ValueError('DSST update requires basic environment rules')
        payload = {'task_or_task_family_scope': self.task_scope, 'basic_environment_rules': self.basic_environment_rules, 'remaining_complete_attempts': remaining, 'complete_ordered_failure_only_ao_forest': serialize_dsst_ao_forest(self.failure_ao_forest)}
        user_input = json.dumps(payload, ensure_ascii=False, indent=2)
        raw = str(_llm.complete(user_input, model=model, system=P.DSST_ICL_PROMPT, max_tokens=P.DSST_MAX_TOKENS) or '')
        self.joint_output = {'method': 'single_llm_joint_induction_and_allocation', 'for_trial': trial, 'after_failure_trial': trial - 1, 'remaining_complete_attempts': remaining, 'raw': raw}
        self.last_update_for_trial = trial

    def on_failure(self, trajectory: str, model: str, *, er_name: str='', er_section: str='', trial_index: int | None=None) -> None:
        del er_name, er_section, model
        parsed = parse_policy_trajectory(trajectory)
        if parsed is None:
            raise ValueError('formal DSST requires the canonical policy A-O trace')
        self.observe_trial_task(str(parsed['task']))
        trial = int(trial_index) if trial_index is not None else int(self.failure_ao_forest[-1]['trial']) + 1 if self.failure_ao_forest else 0
        if self.failure_ao_forest and trial <= int(self.failure_ao_forest[-1]['trial']):
            raise ValueError('DSST failure A-O traces must arrive in trial order')
        self.failure_ao_forest.append({'trial': trial, 'ao_trace': trajectory})

    def build_trial_context(self, *, trial_index: int, total_trials: int, past_trajectories: Sequence[str], model: str, er_name: str, er_section: str) -> ControllerContext:
        del total_trials, past_trajectories, er_name, er_section
        if int(trial_index) + 1 <= 1:
            return ControllerContext()
        self._update_for_trial(trial=int(trial_index), model=model)
        raw = str(self.joint_output.get('raw') or '')
        return ControllerContext(verbatim_system_context=raw, decision={'method': self.joint_output['method']})

    def continuation_failure(self) -> dict[str, Any] | None:
        return None

    def on_success(self, trajectory: str, *, trial_index: int | None=None) -> None:
        del trajectory, trial_index

    def on_trial_closed_without_policy_evidence(self, *, trial_index: int, close_reason: str, has_future_trial: bool, model: str, er_name: str='', er_section: str='') -> None:
        del trial_index, close_reason, has_future_trial
        del model, er_name, er_section

    def snapshot_after_success_enabled(self) -> bool:
        return True

    def snapshot(self) -> dict[str, object]:
        return {'schema_version': 1, 'method': 'dsst', 'total_budget': self.total_budget, 'task_scope': self.task_scope, 'task_scope_source': self.task_scope_source, 'basic_environment_rules': self.basic_environment_rules, 'last_update_for_trial': self.last_update_for_trial, 'failure_only_ao_forest': [dict(row) for row in self.failure_ao_forest], 'joint_dsst_output': dict(self.joint_output)}
