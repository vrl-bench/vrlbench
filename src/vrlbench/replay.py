"""Optional exposure policies around the unchanged complete-trial loop.

Memory and controller preparation still run normally. Only the two cross-trial
context channels presented to the actor are gated; records retain both prepared
and delivered text. No policy is installed when replay_gate is absent.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import wraps
import hashlib
import json
import weakref
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class ReplayContext:
    """Input to a custom per-case policy; return True to expose memory."""

    case_id: str
    trial_index: int
    total_trials: int
    system_context: str
    user_context: str
    seed: int = 0


def epsilon_draw(seed: int, case_id: str, trial_index: int) -> float:
    """Stable shared draws, independent of method, epsilon, case order or paths."""
    key = json.dumps([seed, case_id, trial_index], ensure_ascii=False, separators=(",", ":"))
    bits = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") >> 11
    return bits / (1 << 53)


def _case_id(metadata: Mapping[str, Any], benchmark: str) -> str:
    explicit = metadata.get("case_id")
    if isinstance(explicit, str) and explicit:
        return explicit
    if benchmark == "webshop" and type(metadata.get("sample_index")) is int:
        return f"sample_{metadata['sample_index']:04d}"
    if benchmark == "miniwob" and isinstance(metadata.get("task"), str) and type(metadata.get("episode")) is int:
        return f"{metadata['task']}_ep{metadata['episode']}"
    if benchmark == "demo" and isinstance(metadata.get("task"), str):
        return metadata["task"]
    raise ValueError("A replay gate requires stable case identity; custom workloads must pass metadata.case_id")


class _ActorProxy:
    def __init__(self, actor: Any, *, case_id: str, total_trials: int,
                 gate: Mapping[str, Any], policy: Callable[[ReplayContext], bool] | None) -> None:
        self.actor = actor
        self.case_id = case_id
        self.total_trials = total_trials
        self.gate = gate
        self.policy = policy
        self.index = 0
        self.reads: dict[int, dict[str, Any]] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.actor, name)

    def set_trial_context(self, *, trial_index: int, total_trials: int) -> Any:
        self.index = trial_index
        hook = getattr(self.actor, "set_trial_context", None)
        return hook(trial_index=trial_index, total_trials=total_trials) if callable(hook) else None

    def __call__(self) -> Any:
        from vrlbench.runtime import llm

        system, user = llm.get_context(), llm.get_user_context()
        if self.index not in self.reads:
            context = ReplayContext(self.case_id, self.index, self.total_trials, system, user,
                                    seed=int(self.gate["seed"]))
            decision: dict[str, Any] = {"policy": self.gate["policy"], "case_id": self.case_id,
                                        "trial_index": self.index, "applied": self.index > 0}
            if self.index == 0:
                exposed = False
            elif self.policy is not None:
                exposed = self.policy(context)
                if type(exposed) is not bool:
                    raise TypeError("A replay policy must return bool")
            else:
                draw = epsilon_draw(int(self.gate["seed"]), self.case_id, self.index)
                exposed = draw >= self.gate["epsilon"]
                decision.update({"epsilon": self.gate["epsilon"], "seed": self.gate["seed"], "draw": draw})
            decision.update({"exposed": exposed, "prepared_system_context": system,
                             "prepared_user_context": user})
            self.reads[self.index] = decision
        exposed = self.reads[self.index]["exposed"]
        try:
            llm.set_context(system if exposed else "")
            llm.set_user_context(user if exposed else "")
            return self.actor()
        finally:
            llm.set_context(system)
            llm.set_user_context(user)


def _record_delivered_context(result: Any, decisions: Mapping[int, dict[str, Any]]) -> Any:
    """Use actual exposure decisions for both live checkpoints and final records."""
    for index, trial in enumerate(result.trials):
        decision = decisions.get(trial.trial)
        if decision is None:
            continue  # A copied shared first trial was not executed by this proxy.
        system = decision["prepared_system_context"] if decision["exposed"] else ""
        user = decision["prepared_user_context"] if decision["exposed"] else ""
        result.trials[index] = replace(
            trial, system_context=system, user_context=user,
            context="\n\n".join(value for value in (system, user) if value),
            trial_metadata={**trial.trial_metadata, "replay_gate": decision},
        )
    return result


def wrap_run_episode(original: Callable[..., Any], *, gate: Mapping[str, Any],
                     benchmark: str, policy_factory: Callable[[], Any] | None = None) -> Callable[..., Any]:
    """Wrap one SDK-bound loop, without editing the reference implementation."""
    policies: list[Callable[[], Any]] = []

    @wraps(original)
    def run(*args: Any, **kwargs: Any) -> Any:
        # All public backends use the keyword actor contract. Preserve positional
        # callers too: the reference signature starts with actor.
        actor = kwargs.get("actor") if "actor" in kwargs else args[0]
        metadata = dict(kwargs.get("metadata", args[8] if len(args) > 8 else None) or {})
        case_id = _case_id(metadata, benchmark)
        policy = policy_factory() if policy_factory is not None else None
        if policy_factory is not None and not callable(policy):
            raise TypeError("A replay policy factory must return a callable")
        if policy is not None:
            policies[:] = [reference for reference in policies if reference() is not None]
            if any(reference() is policy for reference in policies):
                raise ValueError("A replay policy factory must create a fresh instance for every case")
            try:
                policies.append(weakref.ref(policy))
            except TypeError:
                # Slotted callables may not support weak references. Retain
                # those objects for this run instead of imposing a new protocol.
                policies.append(lambda policy=policy: policy)
        proxy = _ActorProxy(actor, case_id=case_id,
                            total_trials=int(kwargs.get("total_trials", args[4] if len(args) > 4 else 6)),
                            gate=gate, policy=policy)
        if "actor" in kwargs:
            kwargs["actor"] = proxy
        else:
            args = (proxy, *args[1:])
        on_trial = kwargs.get("on_trial")
        if on_trial is not None:
            def checkpoint(result: Any) -> None:
                on_trial(_record_delivered_context(result, proxy.reads))
            kwargs["on_trial"] = checkpoint
        result = original(*args, **kwargs)
        return _record_delivered_context(result, proxy.reads)
    return run
