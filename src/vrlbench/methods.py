"""Use reference memory methods inside an existing agent's trial loop.

The API owns only cross-trial state: callers run and reset their environment,
execute the Actor, and report its real outcome. Reference prompts and writers
are shipped with the shared runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Iterator, Mapping, Sequence

from ._method_names import METHOD_NAMES, canonical_method_name, reference_method_binding


@dataclass
class CaseConfig:
    """Per-case limits and public information; trial indices start at zero."""

    total_trials: int = 6
    model: str = "gpt-4o-mini"
    task_scope: str = ""
    environment_rules: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialRecord:
    """One executed trial, not a proposed plan or synthetic success label."""

    trial_index: int
    trajectory: str
    success: bool
    actor_trajectory: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActorContext:
    """Append ``system`` to the Actor system prompt; prepend ``user`` to its task."""

    system: str = ""
    user: str = ""


class MemoryMethod(ABC):
    """One complete method, including integrated scheduling when applicable."""

    @abstractmethod
    def reset_case(self, config: CaseConfig) -> None:
        """Discard prior-case state and bind this case's configuration."""

    @abstractmethod
    def context_for_trial(self, trial_index: int) -> ActorContext:
        """Return cached Actor-visible context for the next counted trial."""

    @abstractmethod
    def observe_trial(self, record: TrialRecord) -> None:
        """Consume a completed trial and perform the method's original update."""

    def snapshot(self) -> dict[str, Any]:
        """Return optional inspectable state without invoking a model."""
        return {}


_REFERENCE_CALL_LOCK = RLock()


def list_methods() -> tuple[str, ...]:
    """List the five reference conditions without importing their runtime."""
    return METHOD_NAMES


def _enable_reference() -> None:
    try:
        import vrlbench.runtime  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name == "openai":
            raise ModuleNotFoundError(
                "Reference methods require: pip install 'vrlbench[runtime]'"
            ) from exc
        raise


def render_trajectory(*, task: str, initial_observation: str,
                      steps: Sequence[Mapping[str, str]], close_reason: str) -> str:
    """Render executed A-O transitions using the reference trace format."""
    _enable_reference()
    from vrlbench.runtime.trajectory import render_policy_trajectory

    return render_policy_trajectory({
        "schema_version": 1, "task": task, "initial_observation": initial_observation,
        "steps": list(steps), "trial_close_reason": close_reason,
    })


def load_method(name: str, *, client: Any = None) -> MemoryMethod:
    """Create a lazy reference-method adapter with an optional compatible client.

    All calls through this API are serialized while binding the reference LLM
    transport. This does not isolate concurrent direct runtime calls.
    No model client is constructed until an actual memory operation needs one.
    """
    key = canonical_method_name(name)
    if key is None:
        raise ValueError(f"Unknown method {name!r}; choose from {', '.join(METHOD_NAMES)}")
    return _ReferenceMethod(key, client=client)


class _ReferenceMethod(MemoryMethod):
    def __init__(self, name: str, *, client: Any = None) -> None:
        self.name = name
        self._client = client
        self._default_client: Any = None
        self._config: CaseConfig | None = None

    def reset_case(self, config: CaseConfig) -> None:
        if not isinstance(config, CaseConfig):
            raise TypeError("config must be a CaseConfig")
        if type(config.total_trials) is not int or config.total_trials < 1:
            raise ValueError("total_trials must be a positive integer")
        if not isinstance(config.model, str) or not config.model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(config.metadata, dict):
            raise TypeError("metadata must be a dictionary")
        _enable_reference()
        from vrlbench.runtime.er import make_er_slot

        bound = deepcopy(config)
        if bound.environment_rules:
            bound.metadata["basic_environment_rules"] = bound.environment_rules
        slot = make_er_slot(reference_method_binding(self.name)[0])
        controller = None
        if self.name == "reflexion":
            # The source writer has benchmark-specific prompts, not a generic
            # fallback prompt for arbitrary external tasks.
            slot.read_for_trial(trial_index=0, total_trials=bound.total_trials,
                                model=bound.model, metadata=bound.metadata)
        if self.name == "vex2":
            from vrlbench.runtime.controllers.dsst import DSSTController

            controller = DSSTController()
            controller.set_total_budget(bound.total_trials)
            scope = bound.task_scope or bound.metadata.get("task_scope", "")
            if scope:
                controller.set_task_scope(scope)
            controller.set_basic_environment_rules(
                bound.metadata.get("basic_environment_rules", ""))
        self._slot, self._controller = slot, controller
        self._config = bound
        self._history: list[str] = []
        self._contexts: dict[int, ActorContext] = {}
        self._next_trial = 0
        self._stopped = False

    def _require_active(self, trial_index: int) -> CaseConfig:
        if self._config is None:
            raise RuntimeError("Call reset_case(config) before using a method")
        if self._stopped:
            raise RuntimeError("This case has stopped; call reset_case for a new case")
        if type(trial_index) is not int or trial_index != self._next_trial:
            raise ValueError(f"Expected trial_index={self._next_trial} (zero-based)")
        return self._config

    @contextmanager
    def _reference_scope(self, *, public_rules: bool) -> Iterator[None]:
        """Restore caller state even on errors; retain any active usage ledger."""
        from vrlbench.runtime import llm
        from vrlbench.runtime.loop import _public_environment_context

        with _REFERENCE_CALL_LOCK:
            previous_client = llm._client
            previous_system, previous_user = llm.get_context(), llm.get_user_context()
            borrowed = self._client is None and llm._get_token_ledger() is not None and previous_client is not None
            try:
                llm._client = self._client if self._client is not None else (
                    previous_client if borrowed else self._default_client)
                llm.set_context("")
                llm.set_user_context("")
                if public_rules:
                    with _public_environment_context(self._config.metadata):
                        yield
                else:
                    yield
            finally:
                if self._client is None and not borrowed:
                    self._default_client = llm._client
                llm._client = previous_client
                llm.set_context(previous_system)
                llm.set_user_context(previous_user)

    def context_for_trial(self, trial_index: int) -> ActorContext:
        config = self._require_active(trial_index)
        if trial_index in self._contexts:
            return self._contexts[trial_index]
        context = ActorContext()
        if trial_index > 0:
            with self._reference_scope(public_rules=self._controller is None):
                if self._controller is not None:
                    view = self._controller.build_trial_context(
                        trial_index=trial_index, total_trials=config.total_trials,
                        past_trajectories=self._history, model=config.model,
                        er_name=self._slot.__class__.__name__, er_section="")
                    context = ActorContext(system=view.verbatim_system_context or "")
                else:
                    section = self._slot.read_for_trial(
                        trial_index=trial_index, total_trials=config.total_trials,
                        past_trajectories=self._history, model=config.model,
                        metadata={**config.metadata, "phase": "trial_start"})
                    if section:
                        placement = self._slot.read_placement()
                        if placement == "user_prefix":
                            context = ActorContext(user=section)
                        elif placement == "system":
                            import vrlbench.runtime.prompts as prompts

                            context = ActorContext(system=(
                                f"\n\n{prompts.WM_TAG_OPEN}\n"
                                f"{prompts.WM_PREAMBLE.format(trial_info='')}\n\n{section}\n\n"
                                f"{prompts.working_memory_directive('')}\n{prompts.WM_TAG_CLOSE}\n\n"))
                        else:
                            raise ValueError(f"Unsupported reference memory placement {placement!r}")
        self._contexts[trial_index] = context
        return context

    def observe_trial(self, record: TrialRecord) -> None:
        if not isinstance(record, TrialRecord):
            raise TypeError("record must be a TrialRecord")
        config = self._require_active(record.trial_index)
        if type(record.success) is not bool:
            raise TypeError("success must be a boolean")
        if not isinstance(record.trajectory, str):
            raise TypeError("trajectory must be a string")
        if record.actor_trajectory is not None and not isinstance(record.actor_trajectory, str):
            raise TypeError("actor_trajectory must be a string when provided")
        if not isinstance(record.metadata, dict):
            raise TypeError("metadata must be a dictionary")
        eligible = record.metadata.get("policy_evidence_eligible", True)
        if type(eligible) is not bool:
            raise TypeError("policy_evidence_eligible must be a boolean")
        if record.success and not eligible:
            raise ValueError("A successful trial cannot be ineligible policy evidence")
        if eligible:
            from vrlbench.runtime.trajectory import parse_policy_trajectory

            evidence = parse_policy_trajectory(record.trajectory)
            if evidence is None:
                raise ValueError("Use render_trajectory for the canonical executed A-O trace")
            if self._controller is not None:
                self._controller.observe_trial_task(evidence["task"])
        has_future = record.trial_index + 1 < config.total_trials
        if eligible and not record.success and has_future:
            if self._controller is not None:
                self._controller.on_failure(record.trajectory, config.model,
                                            trial_index=record.trial_index)
            else:
                with self._reference_scope(public_rules=True):
                    self._slot.on_trial_end(
                        record.trajectory, actor_trajectory=record.actor_trajectory,
                        success=False, model=config.model, trial_index=record.trial_index,
                        total_trials=config.total_trials, metadata={
                            **config.metadata, **record.metadata,
                            "phase": "trial_end", "has_future_trial": has_future})
        if eligible:
            self._history.append(record.trajectory)
        self._next_trial += 1
        self._stopped = record.success or not has_future

    def snapshot(self) -> dict[str, Any]:
        if self._config is None:
            return {}
        state = {"method": self.name, "completed_trials": self._next_trial,
                 "total_trials": self._config.total_trials, "stopped": self._stopped,
                 "history": list(self._history)}
        state["state"] = (self._controller.snapshot() if self._controller is not None
                          else {"memory": self._slot.build_section()})
        return deepcopy(state)
