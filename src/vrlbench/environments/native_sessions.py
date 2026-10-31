"""Single-step sessions for external actors using the native task adapters.

These wrappers do not call a model or own a trial/action budget. The actor owns
those limits and calls ``step`` once for each environment action. Native
loaders, action parsers, reset seeds and public feedback remain authoritative.
"""

from __future__ import annotations

from copy import deepcopy
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

from .interactive import Observation, ObservationUnavailable, TaskState, Transition


class MiniWoBSession:
    """One MiniWoB task family, with the native reset schedule and actions."""

    task_group = "miniwob"
    allows_task_variation = True

    def __init__(self, case: Mapping[str, Any], params: Mapping[str, Any], *,
                 env: Any = None) -> None:
        from . import miniwob
        from .synapse_dependency import SYNAPSE_SOURCE_COMMIT

        task = case.get("task")
        seed = case.get("episode")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("MiniWoB cases require a nonempty task name")
        if type(seed) is not int or seed < 0:
            raise ValueError("MiniWoB cases require a nonnegative integer episode")
        mode = params.get("trial_seed_mode", "paired_trial")
        if mode not in {"paired_trial", "fixed_instance"}:
            raise ValueError("MiniWoB trial_seed_mode must be paired_trial or fixed_instance")
        headless = params.get("headless", True)
        if type(headless) is not bool:
            raise ValueError("MiniWoB headless must be a boolean")
        self._native = miniwob
        self._env = env if env is not None else miniwob._load_mw_stack()[0](
            subdomain=task, headless=headless,
        )
        self._seed, self._mode = seed, mode
        self.task_group = task
        self.task_scope = f"Complete tasks in the MiniWoB {task} task family across browser resets."
        self.basic_rules = miniwob.MINIWOB_DSST_BASIC_ENVIRONMENT_RULES
        self.system_prompt = (
            "Control the MiniWoB browser through the Agent API. Return exactly one "
            "primitive action, without a Thought prefix or explanation. "
            "Use agent.click_xpath, agent.type, agent.press, agent.click_option, "
            "or agent.movemouse with one string literal argument. "
            + miniwob.MINIWOB_PUBLIC_INTERACTION_RULES
        )
        self.source_identity = {
            "benchmark": "miniwob", "task": task, "base_seed": seed,
            "trial_seed_mode": mode, "source_commit": SYNAPSE_SOURCE_COMMIT,
        }
        self._identity = json.dumps(self.source_identity, sort_keys=True)
        self._actions: Any = None
        self._closed = False
        self._clock_started = False
        self._reset_seed: int | None = None

    def reset(self, *, trial_index: int) -> TaskState:
        from vrlbench.runtime.reset_protocol import miniwob_reset_seed

        if self._closed:
            raise RuntimeError("MiniWoB session is closed")
        if type(trial_index) is not int or trial_index < 0:
            raise ValueError("trial_index must be a nonnegative integer")
        self._reset_seed = miniwob_reset_seed(self._seed, trial_index, mode=self._mode)
        text = self._env.reset(seed=self._reset_seed)
        # The native action-budget track suspends the browser's wall-clock
        # timeout. Model latency must not consume an unreported action budget.
        self._env.instance.driver.execute_script(
            "if (core.EP_TIMER !== null) { clearTimeout(core.EP_TIMER); }"
            "core.EP_TIMER = setTimeout(function () {}, 2147483647);"
            "core.clearTimer();"
        )
        self._actions = self._native._AgentActions(self._env, trial_index=trial_index)
        self._clock_started = False
        task = self._native._task_from_observation_html(str(text or ""), self._env.get_task())
        return TaskState(task=task, observation=Observation(text=str(text or "")),
                         identity=self._identity)

    def parse_response(self, raw: str) -> str:
        actions = self._native._parse_agent_actions(raw)
        if len(actions) != 1:
            raise ValueError("MiniWoB session.step accepts exactly one primitive Agent API action")
        return self._native._canonical_agent_code(actions)

    def step(self, action: str) -> Transition:
        if self._closed or self._actions is None or self._actions.done:
            raise RuntimeError("Reset MiniWoB before acting, and do not act after termination")
        parsed = self._native._parse_agent_actions(self.parse_response(action))
        if not self._clock_started:
            self._env.instance.start_time = time.time()
            self._clock_started = True
        method, argument = parsed[0]
        getattr(self._actions, method)(argument)
        return Transition(observation=Observation(text=str(self._actions.last_obs or "")),
                          terminated=bool(self._actions.done))

    def evaluate(self) -> bool:
        if self._closed or self._actions is None:
            raise RuntimeError("Cannot evaluate MiniWoB before reset or after close")
        return bool(self._actions.done and self._actions.reward > 0)

    @property
    def evaluation_score(self) -> float | None:
        return float(self._actions.reward) if self._actions is not None and self._actions.done else None

    def get_trial_metadata(self) -> dict[str, Any]:
        return {"base_seed": self._seed, "reset_seed": self._reset_seed,
                "trial_seed_mode": self._mode,
                "environment_action_clock": "ao_horizon_only_native_timeout_suspended_episode_active"}

    def capture_observation(self) -> bytes:
        if self._closed or self._actions is None:
            raise RuntimeError("Reset MiniWoB before capturing an observation")
        image = self._env.instance.driver.get_screenshot_as_png()
        if not isinstance(image, bytes) or not image:
            raise ValueError("MiniWoB screenshot must be nonempty PNG bytes")
        return image

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self._actions is not None or getattr(self._env, "instance", None) is not None:
                self._env.close()


class WebShopSession:
    """One fixed official shopping goal with native one-action transitions."""

    task_group = "webshop"
    task_scope = "Complete the same WebShop shopping goal after each reset."

    def __init__(self, case: int | Mapping[str, Any], params: Mapping[str, Any], *,
                 env: Any = None) -> None:
        from . import webshop
        from .webshop_dependency import WEBSHOP_OFFICIAL_TEST_GOAL_END, WEBSHOP_OFFICIAL_TEST_GOAL_START

        del params
        sample_index = case.get("sample_index") if isinstance(case, Mapping) else case
        if type(sample_index) is not int:
            raise ValueError("WebShop cases require an integer goal index or sample_index")
        if not WEBSHOP_OFFICIAL_TEST_GOAL_START <= sample_index <= WEBSHOP_OFFICIAL_TEST_GOAL_END:
            raise ValueError("WebShop goal index must belong to the official human test split")
        self._native = webshop
        self.sample_index = sample_index
        if env is None:
            factory = webshop._load_default_env_factory()
            self._env = factory()
            environment_identity = deepcopy(getattr(factory, "_vrlbench_environment_identity", {}))
        else:
            self._env = env
            environment_identity = {}
        self.source_identity = {**environment_identity, "benchmark": "webshop", "sample_index": sample_index}
        self._identity = json.dumps(self.source_identity, sort_keys=True)
        self.system_prompt = webshop.SYSTEM_INSTRUCTION
        self.basic_rules = webshop.WEBSHOP_DSST_BASIC_ENVIRONMENT_RULES
        self._active = self._closed = self._terminated = self._exclude_html = False
        self._reward = 0.0
        self._task: str | None = None

    def reset(self, *, trial_index: int) -> TaskState:
        if self._closed:
            raise RuntimeError("WebShop session is closed")
        if type(trial_index) is not int or trial_index < 0:
            raise ValueError("trial_index must be a nonnegative integer")
        observation, _reward, _done = self._env.step(
            self._native._fixed_webshop_session(self.sample_index), "reset",
        )
        task = self._native._extract_instruction(observation)
        if not task or (self._task is not None and task != self._task):
            raise ValueError("WebShop reset must preserve a nonempty fixed shopping goal")
        self._task = task
        self._active, self._terminated, self._exclude_html = True, False, False
        self._reward = 0.0
        return TaskState(task=task, observation=Observation(text=str(observation or "")),
                         identity=self._identity)

    def parse_response(self, raw: str) -> str:
        action = self._native._parse_webshop_actor_output(raw)
        if not action:
            raise ValueError("WebShop requires exactly one search[query] or click[target] action")
        return action

    def step(self, action: str) -> Transition:
        if not self._active or self._closed or self._terminated:
            raise RuntimeError("Reset WebShop before acting, and do not act after termination")
        action = self.parse_response(action)
        try:
            observation, reward, done = self._env.step(
                self._native._fixed_webshop_session(self.sample_index), action,
            )
        except AssertionError:
            if not bool(getattr(self._env, "last_action_invalid", False)):
                raise
            observation, reward, done = "Invalid action!", 0.0, False
        self._reward, self._terminated = float(reward), bool(done)
        self._exclude_html = bool(done or self._reward > 0.0)
        feedback = self._native._shared_webshop_feedback(
            observation, self._reward, terminal=self._exclude_html,
        )
        return Transition(observation=Observation(text=feedback), terminated=self._terminated)

    def evaluate(self) -> bool:
        if not self._active or self._closed:
            raise RuntimeError("Cannot evaluate WebShop before reset or after close")
        return self._terminated and self._reward >= 1.0

    @property
    def evaluation_score(self) -> float | None:
        return self._reward if self._active else None

    def get_trial_metadata(self) -> dict[str, Any]:
        return {"sample_index": self.sample_index}

    def capture_observation(self) -> str:
        if not self._active or self._closed:
            raise RuntimeError("Reset WebShop before capturing an observation")
        if self._exclude_html:
            raise ObservationUnavailable("Terminal page omitted to exclude evaluator-only details; public outcome text is retained.")
        html = self._env.capture_observation()
        if not isinstance(html, str) or not html.strip():
            raise ValueError("WebShop capture must be nonempty native HTML")
        return html

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            close = getattr(getattr(self._env, "env", self._env), "close", None)
            if callable(close):
                close()


def make_miniwob_session(case: Mapping[str, Any], params: Mapping[str, Any]) -> MiniWoBSession:
    return MiniWoBSession(case, params)


def make_webshop_session(case: int | Mapping[str, Any], params: Mapping[str, Any]) -> WebShopSession:
    return WebShopSession(case, params)


def make_session(benchmark: str, case: Any, params: Mapping[str, Any], *,
                 model_config: Mapping[str, Any] | None = None,
                 environment_pythonpath: str | None = None) -> Any:
    """Select the native session, optionally in a separate Python environment."""
    settings = dict(params)
    executable = settings.pop("environment_python", None)
    if executable is not None:
        return SubprocessSession(benchmark, case, settings, python_executable=executable,
                                 model_config=model_config,
                                 environment_pythonpath=environment_pythonpath)
    factory_spec = settings.pop("session_factory", None)
    if factory_spec is not None:
        from vrlbench.plugins import load_factory

        factory, identity = load_factory(factory_spec)
        session = factory(case=case, params=settings, **dict(factory_spec.get("kwargs", {})))
        if not all(callable(getattr(session, name, None)) for name in ("reset", "step", "evaluate", "close")):
            raise TypeError("A session_factory must return a Session with reset, step, evaluate, and close")
        session.source_identity = {**dict(getattr(session, "source_identity", {})),
                                   "session_factory": identity}
        return session
    if benchmark == "miniwob":
        return make_miniwob_session(case, settings)
    if benchmark == "webshop":
        return make_webshop_session(case, settings)
    if benchmark == "webarena":
        from .webarena import make_session as create
    elif benchmark == "osworld":
        from .osworld import make_session as create
    else:
        raise ValueError(f"Unsupported interactive benchmark {benchmark!r}")
    return create(case, settings)


class SubprocessSession:
    """Keep an environment's Python dependencies outside the actor process.

    This is a single sequential session over stdin/stdout, not a service. The
    simulation worker owns its lifetime and receives all environment failures.
    """

    def __init__(self, benchmark: str, case: Any, params: Mapping[str, Any], *,
                 python_executable: str, model_config: Mapping[str, Any] | None = None,
                 environment_pythonpath: str | None = None) -> None:
        if not isinstance(python_executable, str) or not python_executable.strip():
            raise ValueError("environment_python must be a nonempty Python executable")
        env = os.environ.copy()
        source_root = str(Path(__file__).resolve().parents[2])
        # Do not leak the actor interpreter's stdlib/site-packages into an
        # environment running another Python version. Preserve only explicit
        # environment import roots supplied by the simulation configuration.
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (source_root, environment_pythonpath)))
        env.pop("PYTHONHOME", None)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        self._process = subprocess.Popen(
            [python_executable, "-m", "vrlbench.environments.session_worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, encoding="utf-8", env=env,
        )
        self._closed = False
        self._score: float | None = None
        try:
            journal_path = None
            if model_config is not None:
                from vrlbench.runtime import llm

                journal = getattr(llm.get_client(), "journal", None)
                journal_path = str(journal.path) if journal is not None else None
            metadata = self._request(
                "init", benchmark=benchmark, case=case, params=dict(params),
                model_config=dict(model_config) if model_config is not None else None,
                journal_path=journal_path,
            )
            self.system_prompt = metadata["system_prompt"]
            self.basic_rules = metadata["basic_rules"]
            self.task_group = metadata["task_group"]
            self.task_scope = metadata["task_scope"]
            self.source_identity = metadata["source_identity"]
            self.allows_task_variation = metadata["allows_task_variation"]
        except BaseException:
            try:
                self.close()
            except Exception:
                pass  # Preserve the initialization error after cleanup.
            raise

    def _request(self, method: str, **arguments: Any) -> Any:
        if self._closed:
            raise RuntimeError("environment subprocess session is closed")
        assert self._process.stdin is not None and self._process.stdout is not None
        # The caller's ledger is scoped to the current trial/case. The child
        # owns native judge requests but returns their usage, not a second copy
        # of the durable journal entries.
        llm = sys.modules.get("vrlbench.runtime.llm")
        ledger = llm._get_token_ledger() if llm is not None else None
        request = json.dumps({"method": method, **arguments,
                              "require_provider_usage": bool(ledger and ledger.require_provider_usage)},
                             allow_nan=False)
        try:
            self._process.stdin.write(request + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError("environment subprocess stopped before handling the request") from exc
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError(f"environment subprocess exited without a {method} response")
        response = json.loads(line)
        if not isinstance(response, dict) or type(response.get("ok")) is not bool:
            raise RuntimeError("environment subprocess returned an invalid response")
        if ledger is not None:
            ledger.absorb(response.get("model_usage"))
        if not response["ok"]:
            error = response.get("error", {})
            if error.get("type") == "ObservationUnavailable":
                raise ObservationUnavailable(str(error.get("message", "Snapshot was deliberately omitted")))
            message = f"environment {method} failed: {error.get('type', 'Error')}: {error.get('message', '')}"
            native_error = {"ValueError": ValueError, "MiniWoBActionParseError": ValueError,
                            "TypeError": TypeError, "KeyError": KeyError, "RuntimeError": RuntimeError,
                            "FileNotFoundError": FileNotFoundError, "ConnectionError": ConnectionError,
                            "TimeoutError": TimeoutError, "ImportError": ImportError,
                            "ModuleNotFoundError": ModuleNotFoundError}.get(error.get("type"))
            if native_error is not None:
                raise native_error(message)
            if llm is not None:
                from vrlbench.runtime.budget import TokenBudgetExceeded

                error_type = {"TokenBudgetExceeded": TokenBudgetExceeded,
                              "ProviderUsageMissing": llm.ProviderUsageMissing,
                              "ProviderIdentityMismatch": llm.ProviderIdentityMismatch,
                              "ProviderCallUncertain": llm.ProviderCallUncertain}.get(error.get("type"))
                if error_type is not None:
                    raise error_type(message)
            raise RuntimeError(message)
        return response.get("result")

    @staticmethod
    def _observation(value: Mapping[str, Any]) -> Observation:
        image = value.get("image_base64")
        return Observation(text=value["text"], image=base64.b64decode(image, validate=True) if image is not None else None)

    def reset(self, *, trial_index: int) -> TaskState:
        result = self._request("reset", trial_index=trial_index)
        self._score = None
        return TaskState(task=result["task"], observation=self._observation(result["observation"]),
                         identity=result["identity"])

    def parse_response(self, raw: str) -> str:
        return self._request("parse_response", raw=raw)

    def step(self, action: str) -> Transition:
        result = self._request("step", action=action)
        return Transition(observation=self._observation(result["observation"]),
                          terminated=result["terminated"], truncated=result["truncated"])

    def evaluate(self) -> bool:
        result = self._request("evaluate")
        self._score = result["score"]
        return result["success"]

    @property
    def evaluation_score(self) -> float | None:
        return self._score

    def get_trial_metadata(self) -> dict[str, Any]:
        return self._request("get_trial_metadata")

    def capture_observation(self) -> bytes | str:
        result = self._request("capture_observation")
        if result["kind"] == "image":
            return base64.b64decode(result["data"], validate=True)
        if result["kind"] == "html":
            return result["data"]
        raise RuntimeError("environment subprocess returned an unsupported capture kind")

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._process.poll() is None:
                self._request("close")
        finally:
            self._closed = True
            if self._process.stdin is not None:
                self._process.stdin.close()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
            if self._process.stdout is not None:
                self._process.stdout.close()


__all__ = ["MiniWoBSession", "WebShopSession", "SubprocessSession", "make_session",
           "make_miniwob_session", "make_webshop_session"]
