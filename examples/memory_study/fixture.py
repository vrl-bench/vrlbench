"""A deterministic teaching task: outcomes follow actions, never trial numbers."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any


CONTROLS = ("amber", "blue", "green")
MEMORY_PREFIX = "Rejected controls JSON: "
TASK = "Open the route by activating one control. Rejected controls remain invalid after reset."


class RouteEnvironment:
    """One public observation and one state-changing action per fresh trial."""

    def __init__(self, target: str) -> None:
        if target not in CONTROLS:
            raise ValueError("target must be one of the available controls")
        self._target = target
        self._active = False
        self.success = False

    def reset(self) -> dict[str, Any]:
        self._active, self.success = True, False
        return {"task": TASK, "controls": list(CONTROLS), "screen": "route closed"}

    def step(self, action: str) -> dict[str, Any]:
        if not self._active:
            raise RuntimeError("the trial is over; reset before another action")
        if action not in CONTROLS:
            raise ValueError("action must name a visible control")
        self._active = False
        self.success = action == self._target
        return {
            "screen": "route open" if self.success else "route closed",
            "feedback": "Route opened." if self.success else f"Rejected control: {action}.",
            "success": self.success,
            "terminal": True,
        }


def select_action(observation: dict[str, Any], context: str) -> str:
    """The same stateless rule policy in both conditions; no hidden task input."""
    rejected: list[str] = []
    for line in context.splitlines():
        if line.startswith(MEMORY_PREFIX):
            rejected = json.loads(line[len(MEMORY_PREFIX):])
    controls = observation["controls"]
    return next((control for control in controls if control not in rejected), controls[0])


class RuleActor:
    def __init__(self, environment: RouteEnvironment) -> None:
        self.environment = environment
        self.record: dict[str, Any] = {}

    def __call__(self) -> str:
        from vrlbench.runtime import llm

        observation = self.environment.reset()
        context = "\n\n".join(filter(None, (llm.get_context(), llm.get_user_context())))
        action = select_action(observation, context)
        feedback = self.environment.step(action)
        self.record = {"initial_observation": observation, "action": action,
                       "observation": feedback}
        return json.dumps(self.record, sort_keys=True)

    def get_trial_metadata(self) -> dict[str, Any]:
        return {"environment_steps": 1, "max_environment_steps": 1,
                "synthetic_teaching_fixture": True, "llm_performance_evaluated": False}


def create_memory(*, keep_last: int = 2, events_file: str | None = None) -> Any:
    """Retain rejected controls from actual failed-trial feedback, without a model."""
    from vrlbench.memory import MemorySystem

    if type(keep_last) is not int or keep_last < 1:
        raise ValueError("keep_last must be a positive integer")
    if events_file is not None and (not events_file or Path(events_file).name != events_file
                                    or events_file in {".", ".."}):
        raise ValueError("events_file must be a file name within the worker's run directory")
    events_path = Path.cwd() / events_file if events_file is not None else None

    class FeedbackMemory(MemorySystem):
        def __init__(self) -> None:
            self.reset()

        def reset(self) -> None:
            self.rejected: list[str] = []

        def on_failure(self, trajectory: str, model: str) -> None:
            # The original loop passes the exact JSON action/observation trace.
            feedback = json.loads(trajectory)["observation"]["feedback"]
            prefix = "Rejected control: "
            if feedback.startswith(prefix) and feedback.endswith("."):
                self.rejected.append(feedback[len(prefix):-1])
                self.rejected = self.rejected[-keep_last:]

        def build_section(self) -> str:
            return MEMORY_PREFIX + json.dumps(self.rejected) if self.rejected else ""

        def record_event(self, event: dict[str, Any]) -> None:
            if events_path is not None:
                with events_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, sort_keys=True) + "\n")

        def on_trial_end(self, trajectory: str, *, actor_trajectory: str | None = None,
                         success: bool, model: str, trial_index: int, total_trials: int,
                         metadata: dict[str, Any] | None = None) -> None:
            before = list(self.rejected)
            super().on_trial_end(trajectory, actor_trajectory=actor_trajectory, success=success,
                                 model=model, trial_index=trial_index, total_trials=total_trials,
                                 metadata=metadata)
            self.record_event({"operation": "write", "case_id": (metadata or {}).get("case_id"),
                               "trial": trial_index, "input_trajectory": trajectory,
                               "retained_before": before, "retained_after": list(self.rejected)})

        def read_for_trial(self, *, trial_index: int, total_trials: int, model: str,
                           past_trajectories: Any = None,
                           metadata: dict[str, Any] | None = None) -> str:
            returned = super().read_for_trial(trial_index=trial_index, total_trials=total_trials,
                                             model=model, past_trajectories=past_trajectories,
                                             metadata=metadata)
            self.record_event({"operation": "read", "case_id": (metadata or {}).get("case_id"),
                               "trial": trial_index, "retained": list(self.rejected),
                               "returned_context": returned})
            return returned

    return FeedbackMemory()


def create_no_model_client() -> Any:
    """Fail closed: even an accidental model call cannot initialize networking."""
    from vrlbench.runtime.budget import TokenBudgetExceeded

    def reject_call(**kwargs: Any) -> Any:
        raise TokenBudgetExceeded("This teaching study permits zero model API calls")

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=reject_call)))


class RouteBackend:
    def run(self, spec: dict[str, Any], runtime: Any) -> list[dict[str, Any]]:
        if runtime.max_environment_steps != 1:
            raise ValueError("this teaching task uses exactly one environment action per trial")
        records = []
        for case in spec["benchmark"]["cases"]:
            environment = RouteEnvironment(case["target"])
            actor = RuleActor(environment)
            episode = runtime.run_episode(actor, lambda trajectory: environment.success,
                                          metadata={"case_id": case["case_id"]})
            records.append({"case_id": case["case_id"], "task": "teaching_route_control",
                            "trials": [asdict(trial) for trial in episode.trials],
                            "resource_usage": episode.resource_usage})
        return records


def create_backend() -> RouteBackend:
    return RouteBackend()
