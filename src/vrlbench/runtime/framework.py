
from __future__ import annotations

from typing import Callable, Optional

from vrlbench.runtime.controllers import make_controller
from vrlbench.runtime.er import canonical_er_name, make_er_slot
from vrlbench.runtime.loop import EpisodeResult, run_episode


def _controller_name(name: object) -> str:
    normalized = str(name or "none").strip().lower()
    if normalized not in {"none", "dsst"}:
        raise ValueError(
            "controller must be one of none or dsst; "
            f"got {name!r}"
        )
    return normalized


class VEX2:
    """Compose memory and an optional VEX² controller with the shared loop."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        trials: int,
        er: str = "none",
        controller: str = "dsst",
        allow_gold_feedback: bool = False,
        max_total_tokens: Optional[int] = None,
        require_provider_usage: bool = False,
    ) -> None:
        if type(trials) is not int or trials < 1:
            raise ValueError("trials must be a positive integer")
        self.model = model
        self.trials = trials
        self.er = canonical_er_name(er)
        self.controller = _controller_name(controller)
        self.allow_gold_feedback = bool(allow_gold_feedback)
        self.max_total_tokens = max_total_tokens
        self.require_provider_usage = bool(require_provider_usage)

    def run(
        self,
        actor: Callable[[], str],
        manager: Callable[[str], bool],
        *,
        metadata: Optional[dict] = None,
    ) -> EpisodeResult:
        """Run one finite-horizon task instance."""

        if self.controller == "dsst":
            if self.er != "none":
                raise ValueError("formal DSST requires er='none'")
            if self.allow_gold_feedback:
                raise ValueError("formal DSST does not admit gold or oracle feedback")
            controller_slot = make_controller(
                "dsst",
                allow_gold_feedback=False,
                model=self.model,
            )
        else:
            controller_slot = make_controller(
                self.controller,
                allow_gold_feedback=self.allow_gold_feedback,
                model=self.model,
            )
        return run_episode(
            actor=actor,
            manager=manager,
            er_slot=make_er_slot(self.er),
            model=self.model,
            total_trials=self.trials,
            controller=controller_slot,
            metadata=metadata,
            max_total_tokens=self.max_total_tokens,
            require_provider_usage=self.require_provider_usage,
        )

    def __repr__(self) -> str:
        budget = (
            ""
            if self.max_total_tokens is None
            else f", max_total_tokens={self.max_total_tokens}"
        )
        return (
            f"VEX2(model={self.model}, T={self.trials}, er={self.er}, "
            f"controller={self.controller}{budget})"
        )
