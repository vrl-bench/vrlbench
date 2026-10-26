"""Memory-system interface backed by the shared runtime lifecycle.

``from vrlbench.memory import MemorySystem`` loads the runtime lazily
and requires the ``runtime`` installation extra. ``ERSlot`` remains a name for
the same base class; no replacement abstraction or second contract is introduced.
"""

from __future__ import annotations

from typing import Any

__all__ = ["MemorySystem", "ERSlot"]


def __getattr__(name: str) -> Any:
    if name not in {"MemorySystem", "ERSlot"}:
        raise AttributeError(name)
    try:
        from vrlbench.runtime.er.base import ERSlot
    except ModuleNotFoundError as exc:
        if exc.name == "openai":
            raise ModuleNotFoundError(
                "Memory strategies require the runtime extra: pip install 'vrlbench[runtime]'"
            ) from exc
        raise
    return ERSlot
