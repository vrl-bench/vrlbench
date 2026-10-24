"""Community names and unchanged reference execution keys, without runtime imports."""

from __future__ import annotations


# Public name: (stored reference execution key, additional load_method aliases).
# Simulation accepts public names and execution keys, not presentation labels.
_METHODS = {
    "retry": ("retry", ("none",)),
    "reflexion": ("reflexion", ()),
    "dc_cu": ("dynamic_cheatsheet_adapted", ("dc-cu", "dc-cu*")),
    "ace": ("ace_adapted", ("ace*",)),
    "vex2": ("dsst", ("vex²",)),
}
METHOD_NAMES = tuple(_METHODS)
_SIMULATION_NAMES = {name: public for public, (native, _) in _METHODS.items()
                     for name in (public, native)}
_ALIASES = {name: public for public, (native, aliases) in _METHODS.items()
            for name in (public, native, *aliases)}


def canonical_method_name(name: object) -> str | None:
    """Resolve the existing case-insensitive method API aliases."""
    return _ALIASES.get(str(name).strip().lower())


def simulation_method_name(name: str) -> str | None:
    """Resolve public identifiers and legacy keys without rewriting stored input."""
    return _SIMULATION_NAMES.get(name)


def reference_method_binding(name: str) -> tuple[str, str]:
    """Return the original ER slot and controller for a canonical public name."""
    native = _METHODS[name][0]
    if native == "dsst":
        return "none", "dsst"
    return "none" if native == "retry" else native, "none"


def is_reserved_memory_name(name: str) -> bool:
    """Keep reference aliases unavailable to custom factories, including DC-CU."""
    return canonical_method_name(name.replace("-", "_")) is not None
