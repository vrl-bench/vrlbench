from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Sequence

class ERSlot(ABC):
    """Abstract base for an interchangeable memory/replay method."""

    def on_failure(self, trajectory: str, model: str) -> None:
        """Process a failed trial trajectory and update internal state."""
        return None

    def on_success(self, trajectory: str, model: str) -> None:
        """Process a successful trial trajectory and update internal state."""
        return None

    def on_trial_end(self, trajectory: str, *, actor_trajectory: str | None=None, success: bool, model: str, trial_index: int, total_trials: int, metadata: dict[str, Any] | None=None) -> None:
        """Dispatch a completed trial to the method-specific writer.

        ``trajectory`` is the canonical executed A-O record used for shared
        accounting. ``actor_trajectory`` preserves every Actor response and
        actor-visible feedback in order; exact repeated observations may use a
        reversible reference transport while the raw execution record remains on
        the trial. A named method chooses the factual view required by its
        source writer; the harness adds no method-specific interpretation.

        Subclasses can override this when the original method writes memory
        through a different flow than the simple success/failure callbacks.
        """
        del actor_trajectory
        if success:
            self.on_success(trajectory, model)
        else:
            self.on_failure(trajectory, model)

    def read_for_trial(self, *, trial_index: int, total_trials: int, model: str, past_trajectories: Sequence[str] | None=None, metadata: dict[str, Any] | None=None) -> str:
        """Return the memory context for a trial using the method's read flow.

        The default uses the framework context-block interface.  Prior
        method adapters can override this to perform retrieval, load persisted
        memory, synthesize a cheatsheet, or follow any original read protocol.
        """
        return self.build_section()

    def read_placement(self) -> str:
        """Return where actor-visible memory is injected.

        Framework-native replay slots use system-context placement. Named prior-method adapters override this when their source
        implementation reads memory inside the user task prompt.
        """
        return 'system'

    @abstractmethod
    def build_section(self) -> str:
        """Return the context block for the next trial (empty if no memory)."""

    @abstractmethod
    def reset(self) -> None:
        """Reset per-episode adapter state."""
