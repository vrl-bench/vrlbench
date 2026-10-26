
from vrlbench.runtime.er.ace import ACEAdapted
from vrlbench.runtime.er.base import ERSlot
from vrlbench.runtime.er.dynamic_cheatsheet import DynamicCheatsheetAdapted
from vrlbench.runtime.er.no_er import EmptyReplay
from vrlbench.runtime.er.reflexion import Reflexion


_SLOTS = {
    "none": EmptyReplay,
    "reflexion": Reflexion,
    "dynamic_cheatsheet_adapted": DynamicCheatsheetAdapted,
    "ace_adapted": ACEAdapted,
}


def canonical_er_name(name: str) -> str:
    key = str(name).strip().lower()
    if key == "retry":
        key = "none"
    if key not in _SLOTS:
        raise ValueError(f"Unknown replay method: {name}")
    return key


def make_er_slot(name: str) -> ERSlot:
    return _SLOTS[canonical_er_name(name)]()


__all__ = [
    "ERSlot",
    "EmptyReplay",
    "Reflexion",
    "DynamicCheatsheetAdapted",
    "ACEAdapted",
    "canonical_er_name",
    "make_er_slot",
]
