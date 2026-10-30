"""Workload adapters that execute the shared trial loop and reference runners."""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
from copy import deepcopy
from dataclasses import dataclass
import importlib
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
    """Adapt one whole-method instance without changing the reference loop.

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

    def on_failure(self, trajectory: str, model: str, *, trial_index: int | None = None, **_: Any) -> None:
        from vrlbench.methods import TrialRecord

        self._ensure_case()
        self.method.observe_trial(TrialRecord(
            trial_index=int(trial_index), trajectory=trajectory, success=False,
            metadata={"policy_evidence_eligible": True, "has_future_trial": True},
        ))

    def on_success(self, trajectory: str, *, trial_index: int | None = None) -> None:
        from vrlbench.methods import TrialRecord

        self._ensure_case()
        self.method.observe_trial(TrialRecord(
            trial_index=int(trial_index), trajectory=trajectory, success=True,
            metadata={"policy_evidence_eligible": True, "has_future_trial": False},
        ))

    def on_trial_closed_with_policy_evidence(
        self, trajectory: str, model: str, *, trial_index: int,
        close_reason: str, has_future_trial: bool,
        actor_trajectory: str | None = None,
    ) -> None:
        """Report a valid final failure without asking for another policy."""
        from vrlbench.methods import TrialRecord

        self._ensure_case()
        self.method.observe_trial(TrialRecord(
            trial_index=trial_index, trajectory=trajectory, success=False,
            actor_trajectory=actor_trajectory,
            metadata={"policy_evidence_eligible": True, "trace_available": True,
                      "has_future_trial": has_future_trial, "trial_close_reason": close_reason},
        ))

    def on_trial_closed_without_policy_evidence(
        self, *, trial_index: int, close_reason: str, has_future_trial: bool, **_: Any,
    ) -> None:
        from vrlbench.methods import TrialRecord

        self._ensure_case()
        # The loop deliberately supplies no trace for this notification. Do
        # not manufacture a failure observation for a closed opportunity.
        self.method.observe_trial(TrialRecord(
            trial_index=trial_index, trajectory="", success=False,
            metadata={"policy_evidence_eligible": False, "trace_available": False,
                      "has_future_trial": has_future_trial, "trial_close_reason": close_reason},
        ))

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
                benchmark = spec["benchmark"]["name"]
                if benchmark in {"miniwob", "webshop"}:
                    runner = importlib.import_module(f"vrlbench.environments.{benchmark}")
                    normalizer = f"_normalize_{benchmark}_controller"
                    stack.enter_context(_temporary_attribute(runner, normalizer, allow_name(getattr(runner, normalizer))))
                    # MiniWoB imports its factory locally; WebShop keeps a
                    # module binding. The former sees controllers above.
                    if hasattr(runner, "make_controller"):
                        stack.enter_context(_temporary_attribute(runner, "make_controller", make_controller))
                    for attribute, replacement in (("run_episode", run_episode), ("prepare_shared_t1", prepare_shared_t1)):
                        if hasattr(runner, attribute):
                            stack.enter_context(_temporary_attribute(runner, attribute, replacement))
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
                benchmark = spec["benchmark"]["name"]
                if benchmark in {"miniwob", "webshop"}:
                    runner = importlib.import_module(f"vrlbench.environments.{benchmark}")
                    if hasattr(runner, "run_episode"):
                        stack.enter_context(_temporary_attribute(runner, "run_episode", gated))
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

        return make_er_slot(self.er_name)

    def make_controller(self) -> Any:
        from vrlbench.runtime.controllers import make_controller

        return make_controller(self.controller, allow_gold_feedback=False, model=self.model)

    def run_episode(self, actor: Any, manager: Any, *, metadata: Mapping[str, Any] | None = None,
                    on_trial: Callable[[Any], None] | None = None) -> Any:
        from vrlbench.runtime.loop import run_episode

        loop_metadata = {"benchmark": self.spec["benchmark"]["name"], **dict(metadata or {})}
        controller = self.make_controller()
        if isinstance(controller, _MethodController):
            controller.bind_case_metadata(loop_metadata)
        return run_episode(actor=actor, manager=manager, er_slot=self.make_memory(),
                           controller=controller, model=self.model,
                           total_trials=self.trials, metadata=loop_metadata,
                           max_total_tokens=self.spec.get("max_total_tokens"),
                           require_provider_usage=bool(self.spec.get("require_provider_usage", False)),
                           on_trial=on_trial)

    def run_reference(self) -> list[dict[str, Any]]:
        """Execute an original MiniWoB/WebShop runner with these bindings."""
        return run_reference(self.spec, self.er_name, self.controller)


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


def _case_rows(path: Path, *, task: str | None = None) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise TypeError(f"Expected a list of cases at {path}")
    converted = []
    for row in rows:
        if "case_id" in row:
            converted.append(dict(row))
            continue
        converted.append({
            "case_id": str(row.get("name", row.get("sample_index", row.get("episode", "")))),
            "task": str(task or row.get("task") or "webshop"),
            "trials": row["trials"],
            "resource_usage": row.get("resource_usage", {}),
        })
    return converted


def _validate_sdk_shared_source(spec: Mapping[str, Any]) -> None:
    """Check SDK selection/settings in addition to native shared-T1 validation."""
    value = spec.get("shared_t1_dir")
    if not value:
        return
    source = Path(value).resolve()
    candidates = [source]
    if source.name == "reference":
        candidates.append(source.parent)
    elif source.parent.name == "reference":
        candidates.append(source.parent.parent)
    root = next((path for path in candidates if (path / "request.json").is_file()), None)
    if root is None:
        return  # Original runner artifacts still receive their native checks.
    original = json.loads((root / "request.json").read_text())
    status_path = root / "evaluation.json"
    if not status_path.is_file() or json.loads(status_path.read_text()).get("status") != "complete":
        raise ValueError("Shared T1 must come from a completed SDK evaluation")
    if (original.get("memory", {}).get("name") != "retry"
            or original.get("memory", {}).get("factory") is not None
            or original.get("memory", {}).get("method_factory") is not None):
        raise ValueError("Shared T1 must come from the reference memory-free Retry condition")
    from vrlbench.worker import expected_case_ids

    if original["benchmark"]["name"] != spec["benchmark"]["name"]:
        raise ValueError("Shared T1 SDK benchmark differs")
    if original["benchmark"].get("factory") != spec["benchmark"].get("factory"):
        raise ValueError("Shared T1 SDK benchmark factory differs")
    if expected_case_ids(original["benchmark"]) != expected_case_ids(spec["benchmark"]):
        raise ValueError("Shared T1 SDK case selection differs")
    for key in ("model", "trials", "max_environment_steps", "max_total_tokens", "runtime_env"):
        if original.get(key) != spec.get(key):
            raise ValueError(f"Shared T1 SDK {key} differs")
    name = spec["benchmark"]["name"]
    defaults = {"headless": True, "trial_seed_mode": "paired_trial"} if name == "miniwob" else {"max_actor_steps": None}
    original_params = {**defaults, **original["benchmark"].get("params", {})}
    current_params = {**defaults, **spec["benchmark"].get("params", {})}
    if original_params != current_params:
        raise ValueError("Shared T1 SDK benchmark parameters differ")


def run_reference(spec: Mapping[str, Any], er_name: str, controller: str) -> list[dict[str, Any]]:
    """Reuse the two original runners without rewriting their task lifecycle."""
    benchmark = spec["benchmark"]
    name = benchmark["name"]
    params = dict(benchmark.get("params", {}))
    cases = benchmark["cases"]
    output = Path(spec["output_dir"]) / "reference"
    _validate_sdk_shared_source(spec)
    shared_root = spec.get("shared_t1_dir")
    if shared_root and (Path(shared_root) / "reference").is_dir():
        shared_root = str(Path(shared_root) / "reference")
    common = {
        "model": spec["model"]["name"], "trials": spec["trials"],
        "er": er_name, "controller": controller,
        "shared_t1_dir": shared_root,
        "max_environment_steps": spec["max_environment_steps"],
        "max_total_tokens": spec.get("max_total_tokens"),
        "require_provider_usage": bool(spec.get("require_provider_usage", False)),
        "capture_observations": params.get("capture_observations", False),
    }
    if type(common["capture_observations"]) is not bool:
        raise ValueError("capture_observations must be a boolean")
    if name == "webshop":
        unknown = set(params) - {"max_actor_steps", "capture_observations"}
        if unknown:
            raise ValueError(f"Unknown WebShop parameters: {sorted(unknown)}")
        if any(type(case) is not int for case in cases):
            raise TypeError("WebShop cases must be integer goal indices")
        from vrlbench.environments import webshop

        webshop.run(**common, sample_indices=list(cases), num_envs=len(cases),
                    max_steps=params.get("max_actor_steps"), output_dir=str(output))
        _require_finished(output)
        return _case_rows(output / "episodes.json")
    if name != "miniwob":
        raise ValueError(f"Unsupported reference benchmark {name!r}")
    unknown = set(params) - {"headless", "trial_seed_mode", "capture_observations"}
    if unknown:
        raise ValueError(f"Unknown MiniWoB parameters: {sorted(unknown)}")
    grouped: dict[str, list[int]] = {}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("task"), str) or type(case.get("episode")) is not int:
            raise TypeError("MiniWoB cases must contain task and integer episode")
        grouped.setdefault(case["task"], []).append(case["episode"])
    from vrlbench.environments import miniwob

    result = []
    task_index = {}
    for index, (task, episodes) in enumerate(grouped.items()):
        task_output = output / f"task-{index:04d}"
        task_index[task] = task_output.name
        task_common = dict(common)
        if shared_root and (Path(shared_root) / "task_index.json").is_file():
            source_index = json.loads((Path(shared_root) / "task_index.json").read_text())
            relative = source_index.get(task)
            if not isinstance(relative, str):
                raise KeyError(f"Shared T1 source does not contain MiniWoB task {task!r}")
            source_dir = (Path(shared_root) / relative).resolve()
            if Path(shared_root).resolve() not in source_dir.parents:
                raise ValueError("Shared T1 task index points outside its source root")
            task_common["shared_t1_dir"] = str(source_dir)
        miniwob.run(**task_common, tasks=[task], episode_indices=episodes,
                    episodes_per_task=len(episodes), output_dir=str(task_output),
                    headless=params.get("headless", True),
                    trial_seed_mode=params.get("trial_seed_mode", "paired_trial"))
        _require_finished(task_output)
        result.extend(_case_rows(task_output / "episodes.json", task=task))
        output.mkdir(parents=True, exist_ok=True)
        (output / "task_index.json").write_text(json.dumps(task_index, indent=2) + "\n")
    return result


def _require_finished(output: Path) -> None:
    summary = json.loads((output / "results.json").read_text())
    if summary.get("checkpoint") is not False:
        raise RuntimeError("The benchmark runner produced an incomplete checkpoint")
    if summary.get("infra_failures"):
        raise RuntimeError("The benchmark runner reported infrastructure failures")


def run_backend(spec: Mapping[str, Any], er_name: str, controller: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    benchmark = spec["benchmark"]
    factory_spec = benchmark.get("factory")
    if factory_spec is not None:
        backend, identity = create_plugin(factory_spec)
        method = getattr(backend, "run", None)
        if not callable(method):
            raise TypeError("A benchmark factory must return an object with run(spec, runtime)")
        runtime = BackendRuntime(spec=spec, er_name=er_name, controller=controller)
        return method(dict(spec), runtime), [{"role": "benchmark", **identity}]
    if benchmark["name"] == "webarena":
        from .environments.webarena import run
        return run(spec, BackendRuntime(spec, er_name, controller)), []
    if benchmark["name"] == "osworld":
        from .environments.osworld import run
        return run(spec, BackendRuntime(spec, er_name, controller)), []
    return run_reference(spec, er_name, controller), []
