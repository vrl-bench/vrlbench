"""Conservative equivalence for independent cases in built-in workloads."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Mapping


def comparable_actor(value: Any) -> dict[str, Any]:
    """Compare effective actor settings, treating older records as ReAct."""
    from .specs import ActorSpec, FactorySpec, _construct, _json_object

    actor = ActorSpec() if value is None else _construct(ActorSpec, value)
    params = deepcopy(_json_object(actor.params, "actor.params"))
    if actor.name == "openhands":
        params.setdefault("max_output_tokens", 1024)
        if "condenser" in params:
            condenser = _construct(FactorySpec, params["condenser"])
            params["condenser"] = {"entrypoint": condenser.entrypoint, "kwargs": condenser.kwargs}
    return {"name": actor.name, "params": params}


def recorded_condenser_identity(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Read the actual condenser source recorded by an executed actor."""
    identities = [plugin for plugin in ((payload.get("identity") or {}).get("plugins") or [])
                  if isinstance(plugin, Mapping) and plugin.get("role") == "actor_condenser"]
    identities.extend(
        identity for case in payload.get("cases", []) for trial in case.get("trials", [])
        if (identity := (trial.get("trial_metadata") or {}).get("condenser_identity")) is not None
    )
    digests = set()
    for identity in identities:
        if not isinstance(identity, Mapping):
            raise ValueError("recorded actor_condenser identity must be a mapping")
        digest = identity.get("source_sha256")
        if digest is not None:
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(char not in "0123456789abcdefABCDEF" for char in digest)):
                raise ValueError("recorded actor_condenser source_sha256 must be a SHA-256 digest")
            digests.add(digest.lower())
    if len(digests) > 1:
        raise ValueError("results have different actor_condenser plugin source identities")
    return next((identity for identity in identities if identity.get("source_sha256")), None)


def comparable_workload(value: Any) -> Any:
    """Ignore selection order only for the built-in per-case runners.

    Custom adapters may assign meaning to execution order, including adapters
    that reuse a built-in name. All settings other than this known-independent
    selection order remain unchanged.
    """
    normalized = deepcopy(value)
    if (not isinstance(value, Mapping) or value.get("factory") is not None
            or value.get("name") not in {"miniwob", "webshop"}
            or not isinstance(value.get("cases"), list)):
        return normalized
    normalized = dict(normalized)
    normalized["cases"] = sorted(
        normalized["cases"],
        key=lambda case: json.dumps(case, sort_keys=True, allow_nan=False),
    )
    return normalized
