from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

@dataclass(frozen=True)
class ControllerContext:
    """Actor-visible material produced by one integrated controller."""
    trial_info: str = ''
    section: str = ''
    verbatim_system_context: str | None = None
    decision: dict[str, Any] | None = None

@runtime_checkable
class InterEpisodeController(Protocol):
    """Lifecycle for VEX² and whole-method extensions between complete trials."""
    method_id: str

    def reset(self) -> None:
        ...

    def set_total_budget(self, total_budget: int) -> None:
        ...

    def build_trial_context(self, *, trial_index: int, total_trials: int, past_trajectories: Sequence[str], model: str, er_name: str, er_section: str) -> ControllerContext:
        ...

    def on_failure(self, trajectory: str, model: str, *, er_name: str='', er_section: str='', trial_index: int | None=None) -> None:
        ...

    def on_success(self, trajectory: str, *, trial_index: int | None=None) -> None:
        ...

    def snapshot(self) -> dict[str, object]:
        ...

    def snapshot_after_success_enabled(self) -> bool:
        ...

    def continuation_failure(self) -> dict[str, Any] | None:
        ...
