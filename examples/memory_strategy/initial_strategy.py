"""An editable memory-interface example, not a reported paper method."""

from vrlbench.memory import MemorySystem


class MemoryStrategy(MemorySystem):
    """Replay the two most recent failed trial traces."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.failures: list[str] = []

    def on_failure(self, trajectory: str, model: str) -> None:
        self.failures.append(trajectory)
        self.failures = self.failures[-2:]

    def build_section(self) -> str:
        if not self.failures:
            return ""
        return "Previous unsuccessful attempts:\n\n" + "\n\n".join(self.failures)


def create_memory() -> MemorySystem:
    return MemoryStrategy()
