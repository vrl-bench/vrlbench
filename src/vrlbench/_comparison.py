"""Conservative equivalence for independent cases in built-in workloads."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Mapping


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
