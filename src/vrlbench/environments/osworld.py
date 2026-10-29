"""OSWorld desktop tasks through the upstream VM-backed ``DesktopEnv``.

API reference: xlang-ai/OSWorld, commit
044bc33901cb893808e3f91ebc9a17ed8e54b44c, desktop_env/desktop_env.py.
The VM, snapshot, task files, and upstream dependencies are provisioned outside
VRL-Bench. Importing this module never starts or downloads a VM.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import inspect
import json
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from .interactive import Observation, TaskState, Transition, run_cases


SYSTEM_PROMPT = """Operate the desktop to complete the given task.
Read the current screenshot and any provided accessibility or terminal text.
Use absolute pixel coordinates in the current screenshot's native dimensions,
provided with each observation. The origin (0, 0) is the top-left corner; x
increases to the right and y increases downward. Do not normalize coordinates
or use the dimensions of a scaled image preview.
Return exactly one self-contained Python program, either as bare Python or in
one fully closed Python code block, or the standalone marker DONE, FAIL, or WAIT.
Do not mix Python with a special marker or add explanations outside the code.
The program executes in a fresh process in the task VM; variables and functions
do not persist between turns. Import pyautogui when using it, and insert short
waits between consecutive GUI actions so the interface can respond. Do not
describe an action instead of executing it. Use DONE when the task is complete.
""".strip()

BASIC_RULES = (
    "The environment is an OSWorld desktop VM. Each trial restores the task "
    "through DesktopEnv.reset(task_config=...). Actions are pyautogui Python "
    "code executed in that VM, or DONE, FAIL, WAIT. DONE and FAIL end the trial; "
    "the task's upstream evaluator determines success."
)


def _action(raw: str) -> str:
    """Remove a single presentation wrapper without rewriting the program."""
    text = raw.strip()
    match = re.fullmatch(r"```(?:python|py)?\s*\n(.*?)\n```", text, re.S | re.I)
    return match.group(1).strip() if match and text.count("```") == 2 else text


def _observation(value: Any) -> Observation:
    if not isinstance(value, Mapping):
        raise TypeError("OSWorld must return an observation mapping")
    screenshot = value.get("screenshot")
    if not isinstance(screenshot, bytes) or not screenshot:
        raise RuntimeError("OSWorld returned no screenshot; a visual observation is required")
    # Native OSWorld screenshots are PNG. Read IHDR without decoding or
    # changing the image supplied to the actor.
    if (len(screenshot) < 33
            or screenshot[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"):
        raise RuntimeError("OSWorld screenshots must be native PNG with a complete IHDR header")
    width = int.from_bytes(screenshot[16:20], "big")
    height = int.from_bytes(screenshot[20:24], "big")
    if not (0 < width < 2**31 and 0 < height < 2**31):
        raise RuntimeError("OSWorld PNG screenshot dimensions must be in [1, 2**31 - 1]")
    parts = [f"Current desktop screenshot attached: {width} x {height} pixels.\n"
             f"Pixel coordinate bounds: 0 <= x < {width}; 0 <= y < {height}."]
    for field, label in (("accessibility_tree", "Accessibility tree"),
                         ("terminal", "Terminal output")):
        text = value.get(field)
        if text is not None:
            if not isinstance(text, str):
                raise TypeError(f"OSWorld {field} must be text or null")
            if text.strip():
                parts.append(f"{label}:\n{text}")
    return Observation(text="\n\n".join(parts), image=screenshot)


class OSWorldSession:
    """One fixed task and VM, reset by the upstream environment each trial."""

    system_prompt = SYSTEM_PROMPT
    basic_rules = BASIC_RULES
    task_scope = "Complete the same desktop task after each VM reset."
    task_group = "osworld"

    def __init__(self, case: Mapping[str, Any], params: Mapping[str, Any], *,
                 env_factory: Callable[..., Any] | None = None) -> None:
        if not isinstance(case, Mapping) or not isinstance(case.get("case_id"), str):
            raise ValueError("OSWorld cases require case_id and config_file")
        config_path = Path(case["config_file"]).expanduser().resolve()
        raw = config_path.read_bytes()
        config = json.loads(raw)
        if (not isinstance(config, dict) or not isinstance(config.get("id"), str)
                or not isinstance(config.get("instruction"), str)
                or not config["instruction"].strip()
                or not isinstance(config.get("evaluator"), dict)):
            raise ValueError("OSWorld config requires id, instruction, and evaluator")
        self.case_id = case["case_id"]
        self._config = config
        self._task = config["instruction"]
        kwargs = dict(params.get("env_kwargs", {}))
        if not isinstance(kwargs.get("path_to_vm"), str) or not kwargs["path_to_vm"].strip():
            raise ValueError("OSWorld requires an explicitly provisioned env_kwargs.path_to_vm")
        if kwargs.get("action_space", "pyautogui") != "pyautogui":
            raise ValueError("The OSWorld adapter uses action_space='pyautogui'")
        kwargs["action_space"] = "pyautogui"
        snapshot = kwargs.get("snapshot_name", "init_state")
        if not isinstance(snapshot, str) or not snapshot.strip():
            raise ValueError("OSWorld snapshot_name must be nonempty")
        pause = params.get("pause", 2.0)
        if (isinstance(pause, bool) or not isinstance(pause, (int, float))
                or not math.isfinite(pause) or pause < 0):
            raise ValueError("OSWorld pause must be a finite nonnegative number")
        self._pause = float(pause)
        self.source_identity = {
            "case_id": self.case_id, "task_id": config["id"],
            "config_sha256": hashlib.sha256(raw).hexdigest(),
            "snapshot_name": snapshot, "provider_name": kwargs.get("provider_name", "vmware"),
        }
        self._identity = json.dumps(self.source_identity, sort_keys=True)
        if env_factory is None:
            try:
                from desktop_env.desktop_env import DesktopEnv
                from desktop_env.controllers.python import PythonController
            except ImportError as exc:
                raise RuntimeError(
                    "OSWorld requires the upstream desktop_env package and a prepared VM; "
                    "see docs/SIMULATION.md#additional-workloads"
                ) from exc
            if "strict" not in inspect.signature(PythonController.get_file).parameters:
                raise RuntimeError(
                    "Unsupported OSWorld file API: get_file(strict=True) is required. "
                    "Use the checked upstream source revision in docs/QUICKSTART.md#use-osworld; "
                    "desktop-env 1.0.2 does not provide this API."
                )
            env_factory = DesktopEnv
        self._env = env_factory(**kwargs)
        self._active = False
        self._closed = False
        self._score: float | None = None

    def reset(self, *, trial_index: int) -> TaskState:
        del trial_index
        if self._closed:
            raise RuntimeError("OSWorld session is closed")
        # DesktopEnv owns snapshot restoration and task setup. Passing the full
        # same config each time also reinstalls the upstream task evaluator.
        value = self._env.reset(task_config=deepcopy(self._config))
        self._active = True
        self._score = None
        return TaskState(task=self._task, observation=_observation(value),
                         identity=self._identity)

    def parse_response(self, raw: str) -> str:
        """Validate one actor response without executing or repairing its code."""
        action = _action(raw)
        if action in {"DONE", "FAIL", "WAIT"}:
            return action
        try:
            program = ast.parse(action)
            compile(program, "<OSWorld action>", "exec")
        except SyntaxError as exc:
            if "```" in raw:
                raise ValueError(
                    "OSWorld response must contain one fully closed Python code block "
                    "with no text outside it, or one complete bare Python program"
                ) from exc
            raise ValueError(
                "OSWorld response must be one complete Python program or a standalone "
                f"DONE, FAIL, or WAIT marker; Python syntax error: {exc.msg}"
            ) from exc
        if not program.body:
            raise ValueError("OSWorld response must contain a nonempty Python program or a special marker")
        if any(isinstance(node, ast.Expr) and isinstance(node.value, ast.Name)
               and node.value.id in {"DONE", "FAIL", "WAIT"}
               for node in ast.walk(program)):
            raise ValueError(
                "OSWorld special markers DONE, FAIL, and WAIT must be standalone "
                "responses, not mixed with Python code"
            )
        return action

    def step(self, action: str) -> Transition:
        if not self._active or self._closed:
            raise RuntimeError("Reset the OSWorld task before stepping")
        value, _reward, done, _info = self._env.step(_action(action), pause=self._pause)
        if type(done) is not bool:
            raise TypeError("OSWorld step termination flag must be boolean")
        # Upstream reward is not task completion. Even FAIL can be the correct
        # terminal action for an infeasible task, so evaluation stays separate.
        return Transition(observation=_observation(value), terminated=done)

    def evaluate(self) -> bool:
        if not self._active or self._closed:
            raise RuntimeError("Reset the OSWorld task before evaluating")
        if self._score is None:
            # Native file getters catch transport errors and may turn them into
            # a zero score. Strict reads keep 404 as an absent task artifact,
            # but preserve connection/HTTP failures even if a getter catches them.
            controller = getattr(self._env, "controller", None)
            get_file = getattr(controller, "get_file", None)
            failures: list[Exception] = []

            def checked_file(*args, **kwargs):
                kwargs["strict"] = True
                try:
                    return get_file(*args, **kwargs)
                except Exception as exc:
                    failures.append(exc)
                    raise

            if callable(get_file):
                controller.get_file = checked_file
            try:
                score = self._env.evaluate()
            finally:
                if callable(get_file):
                    controller.get_file = get_file
            if failures:
                raise RuntimeError("OSWorld file retrieval failed during task evaluation") from failures[0]
            if (isinstance(score, bool) or not isinstance(score, Real)
                    or not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError("OSWorld evaluator must return a finite score in [0, 1]")
            self._score = float(score)
        return self._score == 1.0

    @property
    def evaluation_score(self) -> float | None:
        """Preserve the native score, including partial task completion."""
        return self._score

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._env.close()


def make_session(case: Mapping[str, Any], params: Mapping[str, Any]) -> OSWorldSession:
    return OSWorldSession(case, params)


def run(spec: Mapping[str, Any], runtime: Any) -> list[dict[str, Any]]:
    return run_cases(spec, runtime, make_session)
