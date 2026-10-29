"""Shared actor and runner for resettable interactive task adapters.

Environment adapters own native actions, resets, and outcome verification. This
module owns the bounded actor interaction; the common runtime owns memory and
the outer trial loop.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Observation:
    text: str
    image: bytes | None = None


@dataclass(frozen=True)
class TaskState:
    task: str
    observation: Observation
    identity: str


@dataclass(frozen=True)
class Transition:
    observation: Observation
    terminated: bool = False
    truncated: bool = False


class Session(Protocol):
    system_prompt: str
    basic_rules: str

    def reset(self, *, trial_index: int) -> TaskState: ...
    def step(self, action: str) -> Transition: ...
    def evaluate(self) -> bool: ...
    def close(self) -> None: ...


class InteractiveActor:
    """One actor, with no cross-trial state except the verified case identity."""

    def __init__(self, session: Session, *, model: str,
                 max_environment_steps: int, artifact_dir: str | Path,
                 capture_observations: bool = False,
                 llm_fn: Callable[..., str] | None = None,
                 on_progress: Callable[[dict[str, Any]], None] | None = None) -> None:
        from vrlbench.runtime.budget import EnvironmentInteractionLedger

        if type(max_environment_steps) is not int or max_environment_steps < 1:
            raise ValueError("max_environment_steps must be a positive integer")
        if type(capture_observations) is not bool:
            raise ValueError("capture_observations must be a boolean")
        self.llm_usage_instrumented = llm_fn is None
        if llm_fn is None:
            from vrlbench.runtime.llm import complete
            llm_fn = complete
        self.session = session
        self.model = model
        self.llm_fn = llm_fn
        self.artifact_dir = Path(artifact_dir)
        self.capture_observations = capture_observations
        self.on_progress = on_progress
        self.ledger = EnvironmentInteractionLedger(max_environment_steps, count_invalid_actions=True)
        self.identity: str | None = None
        self.task: str | None = None
        self.success = False
        self.set_trial_context(trial_index=0, total_trials=1)

    def set_trial_context(self, *, trial_index: int, total_trials: int) -> dict[str, Any]:
        if type(trial_index) is not int or type(total_trials) is not int or not 0 <= trial_index < total_trials:
            raise ValueError("invalid trial index or budget")
        self.trial_index, self.total_trials = trial_index, total_trials
        self.prepared: TaskState | None = None
        self.success = False
        self.evidence: dict[str, Any] | None = None
        self.messages: list[str] = []
        self.screenshots: list[dict[str, Any]] = []
        self.replay_observations: list[dict[str, Any]] = []
        self.invalid_action_events: list[dict[str, Any]] = []
        self.close_reason = "running"
        self.protocol_failure = False
        self.terminated = self.truncated = False
        self.system_context: str | None = None
        self.user_context: str | None = None
        self.progress_phase = "reset"
        self.progress_status = "in_progress"
        self.pending_action: dict[str, Any] | None = None
        self.environment_result: dict[str, Any] | None = None
        self.progress_error: dict[str, str] | None = None
        self.ledger.begin_episode(trial_index)
        return {"trial_index": trial_index, "total_trials": total_trials}

    def _checkpoint(self, phase: str) -> None:
        self.progress_phase = phase
        if self.on_progress is not None:
            self.on_progress(self.get_trial_progress())

    def get_trial_progress(self) -> dict[str, Any]:
        """Snapshot an unfinished trial without inventing its outcome or feedback."""
        context = (None if self.system_context is None or self.user_context is None
                   else "\n\n".join(value for value in (self.system_context, self.user_context) if value))
        return {"trial": self.trial_index, "success": None,
                "status": self.progress_status, "phase": self.progress_phase,
                "context": context, "system_context": self.system_context,
                "user_context": self.user_context,
                "actor_trajectory": self.get_actor_trajectory(),
                "policy_evidence": deepcopy(self.evidence),
                "trial_metadata": self._metadata_snapshot(),
                "pending_action": deepcopy(self.pending_action),
                "environment_result": deepcopy(self.environment_result),
                "error": deepcopy(self.progress_error)}

    def checkpoint_interrupted(self, error: Exception) -> None:
        self.progress_status = "interrupted"
        self.progress_error = {"type": type(error).__name__}
        self._checkpoint(self.progress_phase)

    def _save_image(self, image: bytes, *, index: int) -> dict[str, Any]:
        if not isinstance(image, bytes) or not image:
            raise TypeError("an observation image must be nonempty bytes")
        if image.startswith(b"\x89PNG\r\n\x1a\n"):
            suffix = ".png"
        elif image.startswith(b"\xff\xd8\xff"):
            suffix = ".jpg"
        else:
            raise ValueError("observation images must be PNG or JPEG")
        directory = self.artifact_dir / f"trial_{self.trial_index:03d}"
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = directory / f"observation_{index:04d}{suffix}"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(image)
        return {"observation_index": index, "path": str(path),
                "sha256": hashlib.sha256(image).hexdigest()}

    def _observation(self, value: Observation, *, index: int) -> str:
        if not isinstance(value, Observation) or not isinstance(value.text, str):
            raise TypeError("adapter observations must use Observation(text, image)")
        text = value.text
        if value.image is not None:
            entry = self._save_image(value.image, index=index)
            self.screenshots.append(entry)
            # This is a reference to the actual actor input, not a generated
            # textual interpretation of the image. Memory gets text traces.
            text += f"\nScreenshot: {entry['path']} (sha256={entry['sha256']})"
        elif self.capture_observations:
            # A text actor's POV recording must not change its model input.
            capture = getattr(self.session, "capture_observation", None)
            try:
                if not callable(capture):
                    raise NotImplementedError("session has no screenshot capture hook")
                entry = {**self._save_image(capture(), index=index), "kind": "image"}
            except Exception as exc:
                entry = {"observation_index": index, "kind": "unavailable",
                         "reason": "capture failed: " + type(exc).__name__}
            self.replay_observations.append(entry)
        return text

    def prepare_trial(self) -> dict[str, str]:
        if self.prepared is not None:
            assert self.evidence is not None
            return {"task": self.prepared.task,
                    "initial_observation": self.evidence["initial_observation"]}
        self.ledger.charge_reset()
        state = self.session.reset(trial_index=self.trial_index)
        if (not isinstance(state, TaskState) or not isinstance(state.task, str)
                or not state.task.strip() or not isinstance(state.identity, str)
                or not state.identity.strip()):
            raise ValueError("adapter reset must identify a nonempty task and case")
        if self.identity is not None and (state.identity != self.identity or state.task != self.task):
            raise ValueError("adapter reset changed the task identity between trials")
        self.identity, self.task = state.identity, state.task
        initial = self._observation(state.observation, index=0)
        self.evidence = {"schema_version": 1, "task": state.task,
                         "initial_observation": initial,
                         "trial_close_reason": "running", "steps": []}
        self.messages = [f"[SYSTEM/base]\n{self.session.system_prompt}",
                         f"[USER/initial]\nTask: {state.task}\nObservation:\n{initial}"]
        self.prepared = state
        self._checkpoint("reset")
        return {"task": state.task, "initial_observation": initial}

    def _finish(self, *, terminated: bool, truncated: bool) -> None:
        outcome = self.session.evaluate()
        if type(outcome) is not bool:
            raise TypeError("adapter evaluate() must return a boolean verified outcome")
        self.success = outcome
        if outcome:
            self.close_reason = "environment_success"
            self.terminated, self.truncated = True, False
        else:
            self.terminated, self.truncated = terminated, truncated
            self.close_reason = ("environment_terminal_failure" if terminated else
                                 "ao_horizon" if self.ledger.remaining_steps == 0 else
                                 "environment_truncated")
        assert self.evidence is not None
        self.evidence["trial_close_reason"] = self.close_reason
        self.environment_result = {"success": outcome,
                                   "score": getattr(self.session, "evaluation_score", None)}
        self._checkpoint("environment_complete")

    def _action_from_response(self, raw: str) -> str:
        action = raw.strip()
        if not action:
            raise ValueError("empty action response")
        # Adapters may validate native syntax before any environment mutation.
        # ValueError rejects a response; transport and adapter faults propagate.
        parse = getattr(self.session, "parse_response", None)
        if callable(parse):
            action = parse(action)
            if not isinstance(action, str) or not action.strip():
                raise TypeError("adapter parse_response() must return a nonempty action string")
        return action

    def __call__(self) -> str:
        # The loop binds memory immediately before calling the actor. A
        # controller may have prepared the reset state earlier, without it.
        if self.llm_usage_instrumented:
            from vrlbench.runtime import llm
            self.system_context, self.user_context = llm.get_context(), llm.get_user_context()
        already_prepared = self.prepared is not None
        prepared = self.prepare_trial()
        if already_prepared:
            self._checkpoint("reset")
        assert self.prepared is not None and self.evidence is not None
        observation = self.prepared.observation
        history = f"Task: {prepared['task']}\nObservation:\n{prepared['initial_observation']}"
        while self.ledger.remaining_steps:
            prompt = (f"{history}\n\nEnvironment actions remaining: {self.ledger.remaining_steps}.\n"
                      "Return exactly one action in the environment's action format.")
            self.ledger.charge_actor_turn()
            kwargs: dict[str, Any] = {"model": self.model, "system": self.session.system_prompt,
                                      "max_tokens": 1024, "retry_on_error": False}
            if observation.image is not None:
                kwargs["images"] = [observation.image]
            self.messages.append(f"[USER/action_prompt]\n{prompt}")
            raw = self.llm_fn(prompt, **kwargs)
            if not isinstance(raw, str):
                raise TypeError("actor model must return text")
            self.messages.append(f"[ASSISTANT/action]\n{raw}")
            try:
                action = self._action_from_response(raw)
            except ValueError as exc:
                reason = str(exc)
                before = self.ledger.action_attempts
                self.ledger.charge_invalid_action()
                self.invalid_action_events.append({
                    "after_environment_step": self.ledger.steps,
                    "raw_response": raw, "reason": reason,
                    "budget_used_before": before,
                    "budget_used_after": self.ledger.action_attempts})
                feedback = (f"Invalid action response: {reason}. No environment action was executed; "
                            "the observation is unchanged. One action opportunity was consumed; "
                            f"{self.ledger.remaining_steps} remain. Return one valid environment action.")
                self.messages.append(f"[USER/parser_feedback]\n{feedback}")
                history += f"\nRejected response: {raw}\n{feedback}"
                self._checkpoint("invalid_action")
                continue
            self.ledger.require_step()
            self.pending_action = {
                "action": action, "raw_response": raw,
                "before_observation": (self.evidence["steps"][-1]["observation"]
                                       if self.evidence["steps"] else
                                       self.evidence["initial_observation"]),
                "budget_used_before": self.ledger.action_attempts,
                "environment_steps_before": self.ledger.steps,
            }
            self._checkpoint("awaiting_environment_feedback")
            transition = self.session.step(action)
            if (not isinstance(transition, Transition)
                    or type(transition.terminated) is not bool
                    or type(transition.truncated) is not bool
                    or (transition.terminated and transition.truncated)):
                raise ValueError("adapter step must return a valid Transition")
            observation = transition.observation
            text = self._observation(observation, index=self.ledger.steps + 1)
            self.ledger.charge_step()
            self.evidence["steps"].append({"action": action, "observation": text})
            self.messages.append(f"[USER/environment_feedback]\nExecuted action: {action}\nObservation:\n{text}")
            history += f"\nAction: {action}\nObservation:\n{text}"
            self.pending_action = None
            self._checkpoint("environment_feedback")
            if transition.terminated or transition.truncated or not self.ledger.remaining_steps:
                self._finish(terminated=transition.terminated,
                             truncated=(transition.truncated or not self.ledger.remaining_steps)
                             and not transition.terminated)
                break
        if self.close_reason == "running":
            self._finish(terminated=False, truncated=True)
        outcome = "SUCCESS" if self.success else "FAIL"
        return f"{history}\nSTATUS: {outcome}\nClose reason: {self.close_reason}"

    def get_trial_metadata(self) -> dict[str, Any]:
        if self.evidence is None or self.close_reason == "running":
            raise RuntimeError("the trial has not completed")
        if self.ledger.resets != 1 or len(self.evidence["steps"]) != self.ledger.steps:
            raise RuntimeError("interactive actor trace and environment accounting differ")
        if self.terminated and self.truncated:
            raise RuntimeError("a completed trial cannot both terminate and truncate")
        if self.success != (self.close_reason == "environment_success"):
            raise RuntimeError("verified outcome and trial close state differ")
        return self._metadata_snapshot()

    def _metadata_snapshot(self) -> dict[str, Any]:
        return {**self.ledger.snapshot(), "task": self.task, "case_identity": self.identity,
                "source_identity": deepcopy(getattr(self.session, "source_identity", {})),
                "llm_usage_instrumented": self.llm_usage_instrumented,
                "protocol_failure": self.protocol_failure,
                "invalid_actor_responses": len(self.invalid_action_events),
                "invalid_action_policy": "consume_one_action_and_continue",
                "invalid_action_events": deepcopy(self.invalid_action_events),
                "policy_evidence_eligible": bool(self.evidence and self.evidence["steps"]) or self.success,
                "environment_terminated": self.terminated,
                "environment_truncated": self.truncated,
                "environment_score": getattr(self.session, "evaluation_score", None),
                "trial_close_reason": self.close_reason,
                "screenshots": deepcopy(self.screenshots),
                "replay_observations": deepcopy(self.replay_observations)}

    def get_policy_evidence(self) -> dict[str, Any]:
        if self.evidence is None or self.close_reason == "running":
            raise RuntimeError("the trial has not completed")
        return deepcopy(self.evidence)

    def get_actor_trajectory(self) -> str:
        return "\n\n".join(self.messages)


def make_actor(session: Session, *, model: str, max_environment_steps: int,
               artifact_dir: str | Path, capture_observations: bool = False,
               llm_fn: Callable[..., str] | None = None,
               on_progress: Callable[[dict[str, Any]], None] | None = None) -> InteractiveActor:
    return InteractiveActor(session, model=model, max_environment_steps=max_environment_steps,
                            artifact_dir=artifact_dir, capture_observations=capture_observations,
                            llm_fn=llm_fn, on_progress=on_progress)


def run_cases(spec: Mapping[str, Any], runtime: Any,
              make_session: Callable[[Mapping[str, Any], dict[str, Any]], Session]) -> list[dict[str, Any]]:
    """Run explicitly selected cases with fresh memory and reliable cleanup."""
    if spec.get("shared_t1_dir") is not None:
        raise ValueError("these adapters currently execute independent first trials; shared_t1_dir is unsupported")
    benchmark = spec["benchmark"]
    params = {**dict(benchmark.get("params", {})),
              "max_environment_steps": int(spec["max_environment_steps"])}
    output = Path(spec["output_dir"]) / "reference"
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    def write_checkpoint(records: list[dict[str, Any]]) -> None:
        temporary = output / "episodes.json.tmp"
        temporary.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output / "episodes.json")

    identifiers = []
    for case in benchmark["cases"]:
        if not isinstance(case, Mapping) or not isinstance(case.get("case_id"), str) or not case["case_id"].strip():
            raise ValueError("interactive cases must contain a nonempty case_id")
        identifiers.append(case["case_id"])
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("interactive selection contains duplicate case_id values")
    for case in benchmark["cases"]:
        case_id = case["case_id"]
        artifact_dir = output / "observations" / hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:20]
        session = make_session(case, dict(params))
        record = {"case_id": case_id,
                  "task": str(getattr(session, "task_group", benchmark["name"])),
                  "trials": [], "resource_usage": {},
                  "source_identity": deepcopy(getattr(session, "source_identity", {}))}

        def checkpoint_progress(progress: dict[str, Any]) -> None:
            if (progress["trial"] != len(record["trials"])
                    or progress["trial"] >= int(spec["trials"])):
                raise ValueError("unfinished trial must follow the committed trial prefix within T")
            record["in_progress_trial"] = progress
            write_checkpoint([*rows, record])

        actor = None
        failure = None
        try:
            actor = make_actor(session, model=runtime.model,
                               max_environment_steps=int(spec["max_environment_steps"]),
                               artifact_dir=artifact_dir,
                               capture_observations=params.get("capture_observations", False),
                               on_progress=checkpoint_progress)
            metadata = {"benchmark": benchmark["name"], "case_id": case_id,
                        "task_scope": str(getattr(session, "task_scope", "Complete this task through repeated attempts.")),
                        "basic_environment_rules": session.basic_rules,
                        "source_identity": deepcopy(getattr(session, "source_identity", {}))}
            json.dumps(metadata, allow_nan=False)

            def case_record(episode: Any) -> dict[str, Any]:
                return {"case_id": case_id,
                        "task": str(getattr(session, "task_group", benchmark["name"])),
                        "trials": [asdict(trial) for trial in episode.trials],
                        "resource_usage": episode.resource_usage,
                        "source_identity": deepcopy(getattr(session, "source_identity", {}))}

            def checkpoint_trial(episode: Any) -> None:
                # Retain completed trials if a later action or memory call fails.
                # Completeness still depends on success or reaching the trial limit.
                record.clear()
                record.update(case_record(episode))
                write_checkpoint([*rows, record])

            episode = runtime.run_episode(actor, lambda _trace: actor.success,
                                          metadata=metadata, on_trial=checkpoint_trial)
            row = case_record(episode)
        except Exception as exc:
            failure = exc
            if actor is not None:
                try:
                    actor.checkpoint_interrupted(exc)
                except Exception as checkpoint_error:
                    logger.warning("Could not save interrupted trial after %s: %s",
                                   type(exc).__name__, type(checkpoint_error).__name__)
            raise
        finally:
            try:
                session.close()
            except Exception as close_error:
                if failure is None:
                    raise
                logger.warning("Could not close session after %s: %s",
                               type(failure).__name__, type(close_error).__name__)
        rows.append(row)
        write_checkpoint(rows)
    return rows


__all__ = ["Observation", "TaskState", "Transition", "Session", "InteractiveActor",
           "make_actor", "run_cases"]
