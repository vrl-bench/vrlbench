"""Memory-system simulation inputs and the version-1 worker wire format."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import urlsplit

from ._method_names import is_reserved_memory_name, simulation_method_name


def _json_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    try:
        json.dumps(value, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} must contain finite JSON values") from exc
    return value


def _construct(cls: type, value: Mapping[str, Any]):
    if not isinstance(value, Mapping):
        raise ValueError(f"{cls.__name__} must be an object")
    unknown = set(value) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**dict(value))


def _positive_integer(value: Any, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_single_completion_count(request: Mapping[str, Any]) -> None:
    """The runtime consumes one choice, so neither request layer may request more."""
    extra_body = request.get("extra_body")
    for layer in (request, extra_body):
        if isinstance(layer, Mapping) and "n" in layer:
            if type(layer["n"]) is not int or layer["n"] != 1:
                raise ValueError("model requests support only one completion; n must be the integer 1")


@dataclass(frozen=True)
class FactorySpec:
    """Explicit trusted plugin entry point: ``module:factory`` or ``file.py:factory``."""

    entrypoint: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.entrypoint, str) or ":" not in self.entrypoint:
            raise ValueError("factory entrypoint must be module:factory or file.py:factory")
        location, attribute = self.entrypoint.rsplit(":", 1)
        if not location or not all(part.isidentifier() for part in attribute.split(".")):
            raise ValueError("factory must name a module/file and a callable attribute")
        if location.endswith(".py"):
            if not Path(location).is_file():
                raise ValueError(f"factory file does not exist: {location}")
        elif not all(part.isidentifier() for part in location.split(".")):
            raise ValueError(f"invalid factory module: {location}")
        _json_object(self.kwargs, "factory kwargs")

    def resolved(self, base: Path) -> FactorySpec:
        if ":" not in self.entrypoint:
            return self
        location, attribute = self.entrypoint.rsplit(":", 1)
        if location.endswith(".py"):
            location = str((base / location).resolve())
        return FactorySpec(f"{location}:{attribute}", dict(self.kwargs))


@dataclass(frozen=True)
class MemorySpec:
    name: str = "retry"
    factory: FactorySpec | None = None
    method_factory: FactorySpec | None = None


@dataclass(frozen=True)
class ReplayGateSpec:
    """Optional trial-level exposure policy, independent of memory writes."""

    policy: str = "epsilon"
    epsilon: float | None = None
    seed: int = 0
    factory: FactorySpec | None = None

    def validate(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("replay_gate.seed must be a nonnegative integer")
        if self.policy == "epsilon":
            if (isinstance(self.epsilon, bool) or not isinstance(self.epsilon, (int, float))
                    or not math.isfinite(self.epsilon) or not 0 <= self.epsilon <= 1):
                raise ValueError("replay_gate.epsilon must be a finite probability")
            if self.factory is not None:
                raise ValueError("an epsilon gate cannot also use a custom factory")
        elif self.policy == "custom":
            if not isinstance(self.factory, FactorySpec) or self.epsilon is not None:
                raise ValueError("a custom replay gate requires a factory and no epsilon")
            self.factory.validate()
        else:
            raise ValueError("replay_gate.policy must be epsilon or custom")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    provider: str = "openai_compatible"
    timeout_seconds: float = 120.0
    transport: FactorySpec | None = None
    request_options: dict[str, Any] = field(default_factory=dict)
    allowed_response_models: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WorkloadSpec:
    """Task cases and environment settings that exercise a memory system."""

    name: str
    cases: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    factory: FactorySpec | None = None


@dataclass(frozen=True, init=False)
class SimulationSpec:
    """Compose one memory system, model, workload, and execution budget.

    ``workload`` is the public task-selection field. The ``benchmark`` keyword
    and property remain compatible with earlier SDK callers; serialization
    keeps that version-1 key for the unchanged worker and stored results.
    Both trial and action budgets must be supplied explicitly.
    """

    workload: WorkloadSpec
    model: ModelSpec
    trials: int
    max_environment_steps: int
    memory: MemorySpec = field(default_factory=MemorySpec)
    output_dir: str = "runs/evaluation"
    shared_t1_dir: str | None = None
    timeout_seconds: float = 600.0
    max_total_tokens: int | None = None
    require_provider_usage: bool = True
    runtime_env: dict[str, str] = field(default_factory=dict)
    replay_gate: ReplayGateSpec | None = None

    def __init__(
        self,
        workload: WorkloadSpec | None = None,
        model: ModelSpec | None = None,
        memory: MemorySpec | None = None,
        *,
        trials: int,
        max_environment_steps: int,
        output_dir: str = "runs/evaluation",
        shared_t1_dir: str | None = None,
        timeout_seconds: float = 600.0,
        max_total_tokens: int | None = None,
        require_provider_usage: bool = True,
        runtime_env: dict[str, str] | None = None,
        replay_gate: ReplayGateSpec | None = None,
        benchmark: WorkloadSpec | None = None,
    ) -> None:
        if workload is not None and benchmark is not None:
            raise TypeError("provide workload or the compatibility keyword benchmark, not both")
        selected = workload if workload is not None else benchmark
        if selected is None or model is None:
            raise TypeError("SimulationSpec requires workload and model")
        values = {
            "workload": selected, "model": model,
            "memory": MemorySpec() if memory is None else memory,
            "trials": trials, "max_environment_steps": max_environment_steps,
            "output_dir": output_dir, "shared_t1_dir": shared_t1_dir,
            "timeout_seconds": timeout_seconds, "max_total_tokens": max_total_tokens,
            "require_provider_usage": require_provider_usage,
            "runtime_env": {} if runtime_env is None else runtime_env,
            "replay_gate": replay_gate,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def benchmark(self) -> WorkloadSpec:
        """Compatibility view; new code uses ``workload``."""
        return self.workload

    def to_dict(self) -> dict[str, Any]:
        """Return the stable version-1 worker/result configuration."""
        value = asdict(self)
        value["benchmark"] = value.pop("workload")
        # Preserve the existing wire representation for ER-slot callers.
        if self.memory.method_factory is None:
            value["memory"].pop("method_factory")
        if self.replay_gate is None:
            value.pop("replay_gate")
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SimulationSpec:
        if not isinstance(data, Mapping):
            raise ValueError("simulation config must be a JSON object")
        value = dict(data)
        missing = [name for name in ("trials", "max_environment_steps") if name not in value]
        if missing:
            raise ValueError("simulation config requires explicit " + ", ".join(missing))
        if "workload" in value and "benchmark" in value:
            raise ValueError("provide workload or benchmark in a config, not both")
        if "benchmark" in value:
            value["workload"] = value.pop("benchmark")
        workload = dict(value.get("workload", {}))
        model = dict(value.get("model", {}))
        memory = dict(value.get("memory", {}))
        for section, key in ((workload, "factory"), (memory, "factory"),
                             (memory, "method_factory"), (model, "transport")):
            if section.get(key) is not None:
                section[key] = _construct(FactorySpec, section[key])
        if value.get("replay_gate") is not None:
            gate = dict(value["replay_gate"])
            if gate.get("factory") is not None:
                gate["factory"] = _construct(FactorySpec, gate["factory"])
            value["replay_gate"] = _construct(ReplayGateSpec, gate)
        value["workload"] = _construct(WorkloadSpec, workload)
        value["model"] = _construct(ModelSpec, model)
        value["memory"] = _construct(MemorySpec, memory)
        return _construct(cls, value)

    @classmethod
    def from_path(cls, path: str | Path) -> SimulationSpec:
        source = Path(path).resolve()
        return cls.from_dict(json.loads(source.read_text(encoding="utf-8"))).resolved(source.parent)

    def resolved(self, base: Path | None = None) -> SimulationSpec:
        root = Path.cwd() if base is None else Path(base)
        value = self.to_dict()
        for section, key in (("benchmark", "factory"), ("memory", "factory"),
                             ("memory", "method_factory"), ("model", "transport")):
            original = value[section].get(key)
            if original is not None:
                value[section][key] = asdict(_construct(FactorySpec, original).resolved(root))
        if self.replay_gate is not None and self.replay_gate.factory is not None:
            value["replay_gate"]["factory"] = asdict(self.replay_gate.factory.resolved(root))
        value["output_dir"] = str((root / self.output_dir).resolve())
        if self.workload.factory is None and self.workload.name in {"webarena", "osworld"}:
            params = value["benchmark"]["params"]
            if self.workload.name == "osworld" and isinstance(params.get("env_kwargs"), dict):
                vm = params["env_kwargs"].get("path_to_vm")
                if isinstance(vm, str):
                    params["env_kwargs"]["path_to_vm"] = str((root / Path(vm).expanduser()).resolve())
            if isinstance(params.get("reset_factory"), dict):
                params["reset_factory"] = asdict(_construct(FactorySpec, params["reset_factory"]).resolved(root))
            for case in value["benchmark"]["cases"]:
                if isinstance(case, dict) and isinstance(case.get("config_file"), str):
                    case["config_file"] = str((root / Path(case["config_file"]).expanduser()).resolve())
        if self.shared_t1_dir is not None:
            value["shared_t1_dir"] = str((root / self.shared_t1_dir).resolve())
        return self.from_dict(value)

    def validate(self) -> None:
        if self.replay_gate is not None:
            if not isinstance(self.replay_gate, ReplayGateSpec):
                raise ValueError("replay_gate must use ReplayGateSpec")
            self.replay_gate.validate()
        _positive_integer(self.trials, "trials")
        _positive_integer(self.max_environment_steps, "max_environment_steps")
        if self.max_total_tokens is not None:
            _positive_integer(self.max_total_tokens, "max_total_tokens")
        for label, value in (("timeout_seconds", self.timeout_seconds),
                             ("model.timeout_seconds", self.model.timeout_seconds)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{label} must be finite and positive")
        if type(self.require_provider_usage) is not bool:
            raise ValueError("require_provider_usage must be a boolean")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("output_dir must be a nonempty path")
        if not isinstance(self.model.name, str) or not self.model.name.strip():
            raise ValueError("model.name is required")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", self.model.api_key_env):
            raise ValueError("model.api_key_env must name an environment variable, not contain a key")
        if self.model.base_url is not None:
            parsed = urlsplit(self.model.base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("base_url must be an HTTP(S) endpoint without credentials, query, or fragment")
        _json_object(self.model.request_options, "model.request_options")
        forbidden = {"model", "messages", "stream", "api_key", "base_url"} & set(self.model.request_options)
        if forbidden:
            raise ValueError(f"request_options cannot override {sorted(forbidden)}")
        _validate_single_completion_count(self.model.request_options)
        if not isinstance(self.model.allowed_response_models, list) or any(not isinstance(x, str) or not x for x in self.model.allowed_response_models):
            raise ValueError("allowed_response_models must be a list of model names")
        for plugin in (self.memory.factory, self.memory.method_factory,
                       self.workload.factory, self.model.transport):
            if plugin is not None:
                if not isinstance(plugin, FactorySpec):
                    raise ValueError("plugins must use FactorySpec")
                plugin.validate()
        if not isinstance(self.memory.name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", self.memory.name):
            raise ValueError("memory.name must be a simple method identifier")
        if self.memory.factory is not None and self.memory.method_factory is not None:
            raise ValueError("provide factory or method_factory, not both")
        custom_factory = self.memory.factory or self.memory.method_factory
        if custom_factory is None and simulation_method_name(self.memory.name) is None:
            raise ValueError("custom memory requires a factory or method_factory")
        if custom_factory is not None and is_reserved_memory_name(self.memory.name):
            raise ValueError("custom factories cannot override a reference method name")
        if self.workload.factory is None and self.workload.name not in {"demo", "miniwob", "webshop", "webarena", "osworld"}:
            raise ValueError("a new workload requires a factory")
        if not isinstance(self.workload.cases, list) or not self.workload.cases:
            raise ValueError("workload.cases must explicitly select at least one case")
        _json_object(self.workload.params, "workload.params")
        if self.workload.factory is None:
            allowed_params = {"webshop": {"max_actor_steps", "capture_observations"},
                              "miniwob": {"headless", "trial_seed_mode", "capture_observations"},
                              "demo": {"success_after"},
                              "webarena": {"observation_type", "headless", "reset_factory", "capture_observations", "judge_model"},
                              "osworld": {"env_kwargs", "pause"}}[self.workload.name]
            unknown = set(self.workload.params) - allowed_params
            if unknown:
                raise ValueError(f"unknown {self.workload.name} params: {sorted(unknown)}")
            for key in ("max_actor_steps", "success_after"):
                if key in self.workload.params:
                    _positive_integer(self.workload.params[key], f"workload.params.{key}")
            if "headless" in self.workload.params and type(self.workload.params["headless"]) is not bool:
                raise ValueError("headless must be a boolean")
            if "capture_observations" in self.workload.params and type(self.workload.params["capture_observations"]) is not bool:
                raise ValueError("capture_observations must be a boolean")
            if self.workload.params.get("trial_seed_mode", "paired_trial") not in {"paired_trial", "fixed_instance"}:
                raise ValueError("trial_seed_mode must be paired_trial or fixed_instance")
            if self.workload.name in {"miniwob", "webshop"} and self.require_provider_usage:
                if self.memory.name != "retry" and self.shared_t1_dir is None:
                    raise ValueError("accounted reference comparisons require shared_t1_dir from a completed Retry run")
                if self.memory.name == "retry" and self.shared_t1_dir is not None:
                    raise ValueError("a Retry source run must not consume another shared first trial")
            if self.workload.name in {"webarena", "osworld"}:
                self._validate_interactive_workload()
        identities = []
        for case in self.workload.cases:
            if self.workload.name == "webshop" and (type(case) is not int or case < 0):
                raise ValueError("WebShop cases must be nonnegative goal indices")
            if self.workload.name == "miniwob":
                if not isinstance(case, dict) or set(case) != {"task", "episode"} or not isinstance(case["task"], str) or not case["task"].strip() or type(case["episode"]) is not int or case["episode"] < 0:
                    raise ValueError("MiniWoB cases must be {task: name, episode: nonnegative index}")
            try:
                identities.append(json.dumps(case, sort_keys=True, allow_nan=False))
            except (ValueError, TypeError) as exc:
                raise ValueError("case identifiers must be finite JSON values") from exc
        if len(set(identities)) != len(identities):
            raise ValueError("case selection contains duplicates")
        if self.shared_t1_dir is not None and not Path(self.shared_t1_dir).is_dir():
            raise ValueError("shared_t1_dir must be an existing completed Retry directory")
        if not isinstance(self.runtime_env, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in self.runtime_env.items()):
            raise ValueError("runtime_env must map environment names to strings")
        for key in self.runtime_env:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", key):
                raise ValueError("invalid runtime environment variable name")
            if key.endswith(("API_KEY", "ACCESS_TOKEN", "PASSWORD", "SECRET")):
                raise ValueError("credentials belong in the calling environment, not runtime_env")
        json.dumps(self.to_dict(), allow_nan=False)

    def _validate_interactive_workload(self) -> None:
        name, params = self.workload.name, self.workload.params
        if self.shared_t1_dir is not None:
            raise ValueError(f"{name} does not yet import shared first trials; run independent trials with matched budgets")
        if name == "webarena":
            if "judge_model" in params and (not isinstance(params["judge_model"], str) or not params["judge_model"].strip()):
                raise ValueError("WebArena judge_model must be a nonempty model name")
            reset = params.get("reset_factory")
            if not isinstance(reset, dict):
                raise ValueError("WebArena requires reset_factory to restore the website backends before each trial")
            _construct(FactorySpec, reset).validate()
            if params.get("observation_type", "accessibility_tree") not in {"accessibility_tree", "html"}:
                raise ValueError("WebArena supports text observations: accessibility_tree or html")
        if name == "osworld":
            kwargs = _json_object(params.get("env_kwargs", {}), "OSWorld env_kwargs")
            if not isinstance(kwargs.get("path_to_vm"), str) or not kwargs["path_to_vm"].strip():
                raise ValueError("OSWorld requires env_kwargs.path_to_vm pointing to a provisioned VM; the adapter does not download images")
            if kwargs.get("action_space", "pyautogui") != "pyautogui":
                raise ValueError("OSWorld adapter uses action_space=pyautogui")
            snapshot = kwargs.get("snapshot_name", "init_state")
            if not isinstance(snapshot, str) or not snapshot.strip():
                raise ValueError("OSWorld requires a nonempty reset snapshot")
            pause = params.get("pause", 2.0)
            if isinstance(pause, bool) or not isinstance(pause, (int, float)) or not math.isfinite(pause) or pause < 0:
                raise ValueError("OSWorld pause must be finite and nonnegative")
            if self.max_total_tokens is not None:
                raise ValueError("OSWorld screenshot requests do not support a strict max_total_tokens ceiling")
        seen = set()
        for case in self.workload.cases:
            if (not isinstance(case, dict) or set(case) != {"case_id", "config_file"}
                    or any(not isinstance(case.get(key), str) or not case[key].strip() for key in ("case_id", "config_file"))):
                raise ValueError(f"{name} cases must be {{case_id: string, config_file: local path}}")
            if case["case_id"] in seen:
                raise ValueError("duplicate canonical case_id")
            seen.add(case["case_id"])


# Public compatibility names share one implementation and one data model.
BenchmarkSpec = WorkloadSpec
EvaluationSpec = SimulationSpec
