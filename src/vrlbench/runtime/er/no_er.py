
from vrlbench.runtime.er.base import ERSlot


class EmptyReplay(ERSlot):
    """Use an empty replay slot across attempts."""

    def on_failure(self, trajectory: str, model: str) -> None:
        pass

    def build_section(self) -> str:
        return ""                       

    def reset(self) -> None:
        pass
