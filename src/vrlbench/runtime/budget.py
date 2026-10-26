from __future__ import annotations
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Mapping
TOKEN_ACCOUNTING_VERSION = 'provider_resource_usage'
AO_BUDGET_UNIT = 'action_observation_transition'
ACTION_ATTEMPT_BUDGET_UNIT = 'environment_action_or_invalid_response'

class EnvironmentBudgetExceeded(RuntimeError):
    """Raised before an actor would exceed its declared interaction budget."""

class TokenBudgetExceeded(RuntimeError):
    """Raised before an LLM call would exceed its declared token budget."""

@dataclass
class EnvironmentInteractionLedger:
    """Enforce and report one complete episode's environment budget."""
    max_steps_per_episode: int
    max_resets_per_episode: int = 1
    trial_index: int = -1
    resets: int = 0
    steps: int = 0
    restore_count: int = 0
    actor_turns: int = 0
    active: bool = False
    count_invalid_actions: bool = False
    invalid_action_attempts: int = 0

    def __post_init__(self) -> None:
        if int(self.max_steps_per_episode) <= 0:
            raise ValueError('max_steps_per_episode must be positive')
        if int(self.max_resets_per_episode) <= 0:
            raise ValueError('max_resets_per_episode must be positive')
        self.max_steps_per_episode = int(self.max_steps_per_episode)
        self.max_resets_per_episode = int(self.max_resets_per_episode)

    def begin_episode(self, trial_index: int) -> None:
        self.trial_index = int(trial_index)
        self.resets = 0
        self.steps = 0
        self.restore_count = 0
        self.actor_turns = 0
        self.invalid_action_attempts = 0
        self.active = True

    def _require_active(self) -> None:
        if not self.active:
            raise RuntimeError('begin_episode must be called before accounting')

    def charge_reset(self) -> None:
        self._require_active()
        if self.resets >= self.max_resets_per_episode:
            raise EnvironmentBudgetExceeded(f'environment reset budget exhausted for trial {self.trial_index}: max={self.max_resets_per_episode}')
        self.resets += 1

    def require_step(self, count: int=1) -> None:
        """Raise before ``env.step`` if the requested AO cannot fit."""
        self._require_active()
        count = int(count)
        if count <= 0:
            raise ValueError('environment step admission must be positive')
        if self.action_attempts + count > self.max_steps_per_episode:
            raise EnvironmentBudgetExceeded(f'action budget exhausted for trial {self.trial_index}: used={self.action_attempts}, requested={count}, max={self.max_steps_per_episode}')

    def charge_step(self, count: int=1) -> None:
        """Record actual post-action feedback after a successful admission."""
        self.require_step(count)
        self.steps += count

    def charge_invalid_action(self) -> None:
        """Spend one action opportunity without inventing an environment step."""
        if not self.count_invalid_actions:
            raise RuntimeError('invalid-action accounting was not enabled')
        self.require_step()
        self.invalid_action_attempts += 1

    @property
    def action_attempts(self) -> int:
        return self.steps + self.invalid_action_attempts

    def charge_restore(self) -> None:
        """Record one state restore without charging an environment step."""
        self._require_active()
        self.restore_count += 1

    def charge_actor_turn(self) -> None:
        self._require_active()
        self.actor_turns += 1

    @property
    def remaining_steps(self) -> int:
        """Return unused action opportunities, including rejected responses."""
        self._require_active()
        return max(0, self.max_steps_per_episode - self.action_attempts)

    def snapshot(self) -> dict[str, Any]:
        result = {'trial_index': self.trial_index, 'environment_resets': self.resets, 'environment_steps': self.steps, 'remaining_environment_steps': self.remaining_steps, 'restore_count': self.restore_count, 'actor_turns': self.actor_turns, 'max_environment_resets': self.max_resets_per_episode, 'max_environment_steps': self.max_steps_per_episode, 'budget_unit': ACTION_ATTEMPT_BUDGET_UNIT if self.count_invalid_actions else AO_BUDGET_UNIT, 'environment_budget_active': self.active}
        if self.count_invalid_actions:
            result.update(action_attempts=self.action_attempts,
                          invalid_action_attempts=self.invalid_action_attempts)
        return result

def _nonnegative_int(value: Any, *, default: int=0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default

def provider_token_count(value: Any) -> int | None:
    """Accept finite, nonnegative whole counts without coercing other values."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value >= 0 and value.is_integer():
        return int(value)
    return None

def provider_usage_total(usage: Mapping[str, Any]) -> int | None:
    """Return an accountable total, or None for incomplete/contradictory usage.

    A provider total is sufficient even without both components. Without it,
    both components are required. Supplied invalid or contradictory counts
    are never silently replaced by a computed total.
    """
    raw = tuple(usage.get(key) for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
    counts = tuple(provider_token_count(value) for value in raw)
    if any(value is not None and count is None for value, count in zip(raw, counts)):
        return None
    prompt, completion, total = counts
    if total is not None:
        if any(count is not None and count > total for count in (prompt, completion)):
            return None
        if prompt is not None and completion is not None and prompt + completion != total:
            return None
        return total
    if prompt is not None and completion is not None:
        return prompt + completion
    return None

@dataclass
class TokenUsageLedger:
    max_total_tokens: int | None = None
    require_provider_usage: bool = False
    total_tokens: int = 0
    api_calls: int = 0
    missing_usage_calls: int = 0
    api_failures: int = 0
    unaccounted_prefixes: int = 0
    hard_cap_violations: int = 0
    _lock: Lock = field(default_factory=Lock, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.max_total_tokens is not None:
            self.max_total_tokens = int(self.max_total_tokens)
            if self.max_total_tokens <= 0:
                raise ValueError('max_total_tokens must be positive when set')

    @property
    def remaining_tokens(self) -> int | None:
        if self.max_total_tokens is None:
            return None
        return max(self.max_total_tokens - self.total_tokens, 0)

    def admit_call(self, *, estimated_prompt_tokens: int, requested_completion_tokens: int | None) -> int | None:
        estimated_prompt_tokens = _nonnegative_int(estimated_prompt_tokens)
        if requested_completion_tokens is not None:
            requested_completion_tokens = _nonnegative_int(requested_completion_tokens)
            if requested_completion_tokens <= 0:
                raise ValueError('requested_completion_tokens must be positive')
        with self._lock:
            if self.max_total_tokens is None:
                return requested_completion_tokens
            available = self.max_total_tokens - self.total_tokens - estimated_prompt_tokens
            if available <= 0:
                raise TokenBudgetExceeded('token budget exhausted before LLM call')
            if requested_completion_tokens is None:
                return available
            return min(requested_completion_tokens, available)

    def record_response(self, usage: Mapping[str, Any]) -> None:
        total = provider_usage_total(usage)
        with self._lock:
            if total is not None:
                self.total_tokens += total
            self.api_calls += 1
            if total is None:
                self.missing_usage_calls += 1
            if self.max_total_tokens is not None and self.total_tokens > self.max_total_tokens:
                self.hard_cap_violations += 1

    def record_failure(self) -> None:
        with self._lock:
            self.api_failures += 1

    def mark_unaccounted_prefix(self) -> None:
        with self._lock:
            self.unaccounted_prefixes += 1

    def absorb(self, snapshot: Mapping[str, Any] | None) -> None:
        if not isinstance(snapshot, Mapping):
            return
        with self._lock:
            self.total_tokens += _nonnegative_int(snapshot.get('total_tokens'))
            self.api_calls += _nonnegative_int(snapshot.get('api_calls'))
            if snapshot.get('provider_usage_complete') is not True:
                self.unaccounted_prefixes += 1
            if self.max_total_tokens is not None and self.total_tokens > self.max_total_tokens:
                self.hard_cap_violations += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {'total_tokens': self.total_tokens, 'api_calls': self.api_calls, 'provider_usage_complete': self.missing_usage_calls == 0 and self.api_failures == 0 and (self.unaccounted_prefixes == 0)}

    def delta(self, before: Mapping[str, Any]) -> dict[str, Any]:
        after = self.snapshot()
        return {'total_tokens': self.total_tokens - _nonnegative_int(before.get('total_tokens')), 'api_calls': self.api_calls - _nonnegative_int(before.get('api_calls')), 'provider_usage_complete': after['provider_usage_complete']}
