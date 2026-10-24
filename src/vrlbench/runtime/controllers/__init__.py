
from vrlbench.runtime.controllers.base import ControllerContext, InterEpisodeController
from vrlbench.runtime.controllers.dsst import DSSTController


def make_controller(
    name: str | None,
    **kwargs: object,
) -> InterEpisodeController | None:
    normalized = str(name or "none").strip().lower()
    if normalized == "none":
        return None
    if normalized == "dsst":
        return DSSTController(**kwargs)
    raise ValueError(
        "controller must be one of none or dsst; "
        f"got {name!r}"
    )


__all__ = [
    "ControllerContext",
    "InterEpisodeController",
    "DSSTController",
    "make_controller",
]
