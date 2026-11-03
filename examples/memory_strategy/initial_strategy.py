"""An editable memory-interface example, not a reported paper method."""

import json

from vrlbench.memory import MemorySystem
from vrlbench.runtime.trajectory import parse_policy_trajectory, serialize_policy_history


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
        # Keep original traces in state; only the model-facing view shares
        # repeated page observations. Other memory plugins are not rewritten.
        if all(parse_policy_trajectory(trace) is not None for trace in self.failures):
            history = serialize_policy_history([
                {"trial": index, "ao_trace": trace}
                for index, trace in enumerate(self.failures)])
            text = json.dumps(history, ensure_ascii=False, indent=2)
            return "Previous unsuccessful attempts (numbered within this retained window):\n\n" + text
        else:
            text = "\n\n".join(self.failures)
        return "Previous unsuccessful attempts:\n\n" + text


def create_memory() -> MemorySystem:
    return MemoryStrategy()
