"""Workload adapters that execute the shared trial loop and reference runners."""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterator, Mapping
import weakref

from vrlbench._method_names import (
    is_reserved_memory_name, reference_method_binding, simulation_method_name,
)
from vrlbench.plugins import create_plugin, load_factory

_METHOD_FACTORIES: dict[str, Callable[..., Any]] = {}


class _MethodController:
    """Adapt one whole-method instance to the shared trial loop.

    The reference controller channel is a verbatim system-text channel. User
    text is rejected explicitly rather than silently dropped or repositioned.
    """

    def __init__(self, method: Any, *, name: str, model: str, native_rules: bool) -> None:
        self.method = method
        self.method_id = name
        self.model = model
        self._native_rules = native_rules
        self._metadata: dict[str, Any] = {}
        self.reset()

    def reset(self) -> None:
        self._ready = False
        self._total_trials: int | None = None
        self._task_scope = str(self._metadata.get("task_scope") or "")
        self._environment_rules = str(self._metadata.get("basic_environment_rules") or "")
        self._contexts: dict[int, Any] = {}

    def set_total_budget(self, total_budget: int) -> None:
        if type(total_budget) is not int or total_budget < 1:
            raise ValueError("total_budget must be a positive integer")
        self._total_trials = total_budget

    def set_task_scope(self, scope: str) -> None:
        self._task_scope = scope

    @property
    def set_basic_environment_rules(self) -> Any:
        # Only the native runners require this hook. A custom workload may
        # intentionally have no additional public environment rules.
        return self._set_environment_rules if self._native_rules else None

    def _set_environment_rules(self, rules: str) -> None:
        self._environment_rules = rules

    def bind_case_metadata(self, metadata: Mapping[str, Any]) -> None:
        self._metadata = dict(metadata)
        self._task_scope = str(metadata.get("task_scope") or "")
        self._environment_rules = str(metadata.get("basic_environment_rules") or "")

    def _ensure_case(self) -> None:
        if not self._ready:
            if self._total_trials is None:
                raise RuntimeError("set_total_budget must be called before using a memory method")
            from vrlbench.methods import CaseConfig

            self.method.reset_case(CaseConfig(
                total_trials=self._total_trials, model=self.model,
                task_scope=self._task_scope, environment_rules=self._environment_rules,
                metadata=deepcopy(self._metadata),
            ))
            self._ready = True

    def build_trial_context(self, *, trial_index: int, **_: Any) -> Any:
        from vrlbench.runtime.controllers.base import ControllerContext
        from vrlbench.methods import ActorContext

        self._ensure_case()
        if trial_index not in self._contexts:
            context = self.method.context_for_trial(trial_index)
            if not isinstance(context, ActorContext):
                raise TypeError("MemoryMethod.context_for_trial must return ActorContext")
            if not isinstance(context.system, str) or not isinstance(context.user, str):
                raise TypeError("ActorContext system and user must be strings")
            if context.user:
                raise ValueError("The reference trial loop supports whole-method system context only; ActorContext.user must be empty")
            if trial_index == 0 and context.system:
                raise ValueError("The first trial must not contain cross-trial memory context")
            self._contexts[trial_index] = context
        return ControllerContext(verbatim_system_context=self._contexts[trial_index].system)

    def observe_trial(
        self, *, trajectory: str, actor_trajectory: str | None,
        trial_index: int, success: bool, metadata: Mapping[str, Any],
    ) -> None:
        """Pass one completed trial to the method without changing its contents."""
        from vrlbench.methods import TrialRecord

        self._ensure_case()
        self.method.observe_trial(TrialRecord(
            trial_index=trial_index, trajectory=trajectory, success=success,
            actor_trajectory=actor_trajectory, metadata=deepcopy(dict(metadata)),
        ))

    def on_failure(self, trajectory: str, model: str, *, trial_index: int | None = None, **_: Any) -> None:
        # Keep the legacy controller entry points for direct callers. The trial
        # loop uses observe_trial to include the actual transcript and metadata.
        self.observe_trial(
            trajectory=trajectory, actor_trajectory=None, trial_index=int(trial_index),
            success=False, metadata={"policy_evidence_eligible": True, "has_future_trial": True},
        )

    def on_success(self, trajectory: str, *, trial_index: int | None = None) -> None:
        self.observe_trial(
            trajectory=trajectory, actor_trajectory=None, trial_index=int(trial_index),
            success=True, metadata={"policy_evidence_eligible": True, "has_future_trial": False},
        )

    def snapshot(self) -> dict[str, Any]:
        self._ensure_case()
        state = self.method.snapshot()
        if not isinstance(state, dict):
            raise TypeError("MemoryMethod.snapshot must return a dictionary")
        json.dumps(state, allow_nan=False)
        return deepcopy(state)

    def snapshot_after_success_enabled(self) -> bool:
        return True

    def continuation_failure(self) -> None:
        return None


@contextmanager
def _temporary_attribute(owner: Any, name: str, value: Any) -> Iterator[None]:
    previous = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, previous)


@contextmanager
def memory_bindings(spec: Mapping[str, Any]) -> Iterator[tuple[str, str, list[dict[str, Any]]]]:
    """Bind a candidate only within one worker execution, restoring all hooks."""
    from vrlbench.runtime.er import _SLOTS

    original_slots = set(_SLOTS)
    original_methods = set(_METHOD_FACTORIES)
    try:
        binding = register_memory(spec["memory"], native_rules=spec["benchmark"]["name"] in {"miniwob", "webshop"})
        er_name, controller, identity = binding
        with ExitStack() as stack:
            if controller in _METHOD_FACTORIES:
                from vrlbench.runtime import controllers, framework, loop

                create = _METHOD_FACTORIES[controller]
                original_factory = controllers.make_controller

                def make_controller(name: str = "none", **kwargs: Any) -> Any:
                    if name == controller:
                        return create(model=kwargs.get("model", spec["model"]["name"]))
                    return original_factory(name, **kwargs)

                def allow_name(original: Callable[..., str]) -> Callable[..., str]:
                    def normalize(value: object) -> str:
                        return controller if value == controller else original(value)
                    return normalize

                def bind_case(original: Callable[..., Any], *, shared: bool = False) -> Callable[..., Any]:
                    def call(*args: Any, **kwargs: Any) -> Any:
                        candidate = kwargs.get("controller")
                        if isinstance(candidate, _MethodController):
                            metadata = {
                                "benchmark": spec["benchmark"]["name"],
                                "max_environment_steps": spec["max_environment_steps"],
                                "max_total_tokens": spec.get("max_total_tokens"),
                                "require_provider_usage": bool(spec.get("require_provider_usage", False)),
                                **dict(kwargs.get("metadata") or {}),
                            }
                            if shared:
                                metadata["shared_t1_trial_metadata"] = dict(kwargs.get("trial_metadata") or {})
                                metadata["shared_t1_llm_usage"] = dict(kwargs.get("llm_usage") or {})
                            elif candidate._ready:
                                # A continuation has already consumed the same
                                # shared first trial; retain its source identity.
                                metadata = {**candidate._metadata, **metadata}
                            candidate.bind_case_metadata(metadata)
                        return original(*args, **kwargs)
                    return call

                run_episode = bind_case(loop.run_episode)
                prepare_shared_t1 = bind_case(loop.prepare_shared_t1, shared=True)

                stack.enter_context(_temporary_attribute(controllers, "make_controller", make_controller))
                stack.enter_context(_temporary_attribute(framework, "make_controller", make_controller))
                stack.enter_context(_temporary_attribute(framework, "_controller_name", allow_name(framework._controller_name)))
                stack.enter_context(_temporary_attribute(loop, "run_episode", run_episode))
                stack.enter_context(_temporary_attribute(loop, "prepare_shared_t1", prepare_shared_t1))
                stack.enter_context(_temporary_attribute(framework, "run_episode", run_episode))
            gate = spec.get("replay_gate")
            if gate is not None:
                from vrlbench.runtime import framework, loop
                from vrlbench.replay import wrap_run_episode

                policy_factory = None
                if gate.get("factory") is not None:
                    factory, source = load_factory(gate["factory"])
                    policy_kwargs = dict(gate["factory"].get("kwargs", {}))
                    policy_factory = lambda: factory(**policy_kwargs)
                    identity.append({"role": "replay_policy", **source})
                gated = wrap_run_episode(loop.run_episode, gate=gate,
                                         benchmark=spec["benchmark"]["name"], policy_factory=policy_factory)
                stack.enter_context(_temporary_attribute(loop, "run_episode", gated))
                stack.enter_context(_temporary_attribute(framework, "run_episode", gated))
            yield binding
    finally:
        for name in set(_SLOTS) - original_slots:
            _SLOTS.pop(name, None)
        for name in set(_METHOD_FACTORIES) - original_methods:
            _METHOD_FACTORIES.pop(name, None)


@dataclass(frozen=True)
class BackendRuntime:
    """Process-local bindings supplied to a trusted benchmark plugin.

    Each ``run_episode`` creates fresh memory and controller instances, then
    executes the original trial loop under the evaluation's declared budgets.
    """

    spec: Mapping[str, Any]
    er_name: str
    controller: str

    @property
    def model(self) -> str:
        return str(self.spec["model"]["name"])

    @property
    def trials(self) -> int:
        return int(self.spec["trials"])

    @property
    def max_environment_steps(self) -> int:
        return int(self.spec["max_environment_steps"])

    def make_memory(self) -> Any:
        from vrlbench.runtime.er import make_er_slot

        return make_er_slot(self.er_name, **self.spec.get("memory", {}).get("params", {}))

    def make_controller(self) -> Any:
        from vrlbench.runtime.controllers import make_controller

        return make_controller(self.controller, allow_gold_feedback=False, model=self.model)

    def run_episode(self, actor: Any, manager: Any, *, metadata: Mapping[str, Any] | None = None,
                    on_trial: Callable[[Any], None] | None = None,
                    shared_trial: Mapping[str, Any] | None = None) -> Any:
        from vrlbench.runtime.loop import prepare_shared_t1, run_episode

        loop_metadata = {"benchmark": self.spec["benchmark"]["name"], **dict(metadata or {})}
        controller = self.make_controller()
        if isinstance(controller, _MethodController):
            controller.bind_case_metadata(loop_metadata)
        memory = self.make_memory()
        common = dict(controller=controller, er_slot=memory, model=self.model,
                      total_trials=self.trials, metadata=loop_metadata,
                      require_provider_usage=bool(self.spec.get("require_provider_usage", False)))
        if shared_trial is None:
            return run_episode(actor=actor, manager=manager, **common,
                               max_total_tokens=self.spec.get("max_total_tokens"), on_trial=on_trial)
        memory.reset()
        if controller is not None:
            controller.reset()
        prefix, ledger = prepare_shared_t1(
            **common, trajectory=shared_trial["trajectory"],
            policy_trajectory=shared_trial["policy_trajectory"],
            actor_trajectory=shared_trial["actor_trajectory"],
            success=shared_trial["success"], trial_metadata=shared_trial["trial_metadata"],
            llm_usage=shared_trial["llm_usage"], max_total_tokens=self.spec.get("max_total_tokens"),
            on_trial=on_trial,
        )
        if on_trial is not None:
            on_trial(prefix)
        return run_episode(actor=actor, manager=manager, **common, initial_result=prefix,
                           reset_state=False, token_ledger=ledger, on_trial=on_trial)

def register_memory(spec: Mapping[str, Any], *, native_rules: bool = False) -> tuple[str, str, list[dict[str, Any]]]:
    """Bind one candidate factory in this worker, never in the caller process."""
    from vrlbench.runtime.er import _SLOTS
    from vrlbench.runtime.er.base import ERSlot

    name = str(spec["name"])
    factory_spec = spec.get("factory")
    method_spec = spec.get("method_factory")
    if factory_spec is not None and method_spec is not None:
        raise ValueError("provide factory or method_factory, not both")
    if method_spec is not None:
        from vrlbench.methods import MemoryMethod

        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
            raise ValueError("memory.name must be a simple method identifier")
        if is_reserved_memory_name(name) or name in _SLOTS:
            raise ValueError(f"A custom factory cannot replace reference memory {name!r}")
        factory, identity = load_factory(method_spec)
        kwargs = dict(method_spec.get("kwargs", {}))
        instances: weakref.WeakValueDictionary[int, Any] = weakref.WeakValueDictionary()

        def create_controller(*, model: str) -> _MethodController:
            method = factory(**kwargs)
            if not isinstance(method, MemoryMethod):
                raise TypeError("A method_factory must return a vrlbench.methods.MemoryMethod instance")
            if id(method) in instances:
                raise ValueError("A method_factory must create a fresh instance for every evaluation case")
            instances[id(method)] = method
            return _MethodController(method, name=name, model=model, native_rules=native_rules)

        runtime_name = "community_" + name.lower().replace("-", "_")
        if runtime_name in _METHOD_FACTORIES:
            raise ValueError("The worker already has a candidate method registration")
        _METHOD_FACTORIES[runtime_name] = create_controller
        return "none", runtime_name, [{"role": "memory_method", "name": name, **identity}]
    if factory_spec is None:
        public_name = simulation_method_name(name)
        if public_name is None:
            raise ValueError(f"Custom memory {name!r} requires a factory")
        er_name, controller = reference_method_binding(public_name)
        return er_name, controller, []
    if is_reserved_memory_name(name) or name in _SLOTS:
        raise ValueError(f"A custom factory cannot replace reference memory {name!r}")
    factory, identity = load_factory(factory_spec)
    kwargs = dict(factory_spec.get("kwargs", {}))
    instances: weakref.WeakValueDictionary[int, Any] = weakref.WeakValueDictionary()

    def create_memory() -> ERSlot:
        memory = factory(**kwargs)
        if not isinstance(memory, ERSlot):
            raise TypeError("A memory factory must return a vrlbench.memory.MemorySystem instance (ERSlot is compatible)")
        if id(memory) in instances:
            raise ValueError("A memory factory must create a fresh instance for every evaluation case")
        instances[id(memory)] = memory
        return memory

    # Registration and model binding are deliberately process-local. The SDK
    # always starts a new worker for another candidate or evaluation.
    runtime_name = "community_" + identity.get("source_sha256", name)[:24].lower()
    if runtime_name in _SLOTS:
        raise ValueError("The worker already has a candidate memory registration")
    _SLOTS[runtime_name] = create_memory
    return runtime_name, "none", [{"role": "memory", **identity}]


def run_backend(spec: Mapping[str, Any], er_name: str, controller: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    benchmark = spec["benchmark"]
    if benchmark.get("factory") is not None:
        backend, identity = create_plugin(benchmark["factory"])
        method = getattr(backend, "run", None)
        if not callable(method):
            raise TypeError("A benchmark factory must return an object with run(spec, runtime)")
        runtime = BackendRuntime(spec=spec, er_name=er_name, controller=controller)
        return method(dict(spec), runtime), [{"role": "benchmark", **identity}]
    return run_interactive(spec, er_name, controller)


def run_interactive(spec: Mapping[str, Any], er_name: str,
                    controller: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run every built-in workload through the same actor/session interface."""
    import os
    import sys
    from importlib.metadata import version

    actor = spec.get("actor") or {"name": "openhands", "params": {}}
    actor_factory = None
    if actor.get("factory") is not None:
        factory, source = load_factory(actor["factory"])
        factory_kwargs = dict(actor["factory"].get("kwargs", {}))

        def actor_factory(**kwargs: Any) -> Any:
            if set(kwargs) & set(factory_kwargs):
                raise ValueError("actor factory kwargs cannot replace runtime-supplied arguments")
            return factory(**kwargs, **factory_kwargs)

        actor_identity = {"role": "actor", "name": actor["name"], **source}
    else:
        if actor["name"] != "openhands":
            raise ValueError("the built-in actor is openhands; custom actors require actor.factory")
        if sys.version_info < (3, 12):
            raise RuntimeError("Built-in simulations require Python 3.12+ and vrlbench[runtime,openhands]")
        # Set before importing the SDK: all auxiliary state belongs to this run.
        os.environ["OH_PERSISTENCE_DIR"] = str(Path(spec["output_dir"]) / "reference" / "openhands-state")
        try:
            sdk_version = version("openhands-sdk")
        except ImportError as exc:
            raise ModuleNotFoundError("Install the simulation runtime: pip install 'vrlbench[runtime,openhands]'") from exc
        actor_identity = {"role": "actor", "name": "openhands", "sdk_version": sdk_version}

    from .environments.interactive import run_cases
    from .environments.native_sessions import make_session

    condenser_spec = actor.get("params", {}).get("condenser") if actor["name"] == "openhands" else None
    condenser_identity = None
    if condenser_spec is not None:
        _, condenser_identity = load_factory(condenser_spec)
    if spec.get("shared_t1_dir") is not None:
        from ._shared_t1 import load_shared_first_trials

        load_shared_first_trials(spec, actor_identity=actor_identity,
                                 condenser_identity=condenser_identity)

    benchmark = spec["benchmark"]
    name = benchmark["name"]
    if name == "miniwob":
        cases = [{**case, "case_id": f"{case['task']}_ep{case['episode']}"}
                 for case in benchmark["cases"]]
    elif name == "webshop":
        cases = [{"case_id": f"sample_{index:04d}", "sample_index": index}
                 for index in benchmark["cases"]]
    else:
        cases = benchmark["cases"]

    def create_session(case: Mapping[str, Any], params: Mapping[str, Any]) -> Any:
        return make_session(name, case, params, model_config=spec["model"],
                            environment_pythonpath=spec.get("runtime_env", {}).get("PYTHONPATH"))

    runtime = BackendRuntime(spec=spec, er_name=er_name, controller=controller)
    rows = run_cases(spec, runtime, create_session, cases=cases, actor_factory=actor_factory)
    identity = [actor_identity]
    if condenser_identity is not None:
        for row in rows:
            for trial in row["trials"]:
                if trial["trial_metadata"].get("condenser_identity") != condenser_identity:
                    raise ValueError("The OpenHands condenser source changed during evaluation")
        identity.append({"role": "actor_condenser", **condenser_identity})
    return rows, identity
