"""WebArena's accessibility-tree environment and official task evaluator.

Uses the libwebarena 0.0.5 compatibility package also used by BrowserGym.
The API was checked against its wheel (SHA256
5cfdf8abc2b40cc4442905a29af1c45c5fd0ed57bbfe8e3a6421a2e53cb6cb1b),
and web-arena-x/webarena dce04686a56253aefba7b18a4fa0937cf1dc987b.
The browser reset does not restore the websites: a deployment-provided site
reset callback is required and runs before every trial, including the first.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from ..plugins import create_plugin


ACTION_RULES = (
    "Actions: click [id]; hover [id]; type [id] [text] [0 or 1] "
    "(1 presses Enter); press [key combination]; scroll [up or down]; "
    "goto [url]; new_tab; tab_focus [index]; close_tab; go_back; "
    "go_forward; stop [answer]. Keep the square brackets around every argument: "
    "for example, type [12] [example text] [1]. "
    "Use only element IDs from the current observation."
)
# Both native fuzzy graders use this model in libwebarena 0.0.5.
NATIVE_JUDGE_MODEL = "gpt-4-1106-preview"


def _upstream():
    try:
        from webarena.browser_env import ScriptBrowserEnv
        from webarena.browser_env.actions import (
            ActionParsingError, ActionTypes, create_id_based_action,
            create_none_action, create_stop_action,
        )
        from webarena.evaluation_harness.evaluators import evaluator_router
    except ImportError as exc:
        raise ImportError(
            "WebArena requires libwebarena==0.0.5, Playwright "
            "Chromium, configured website URLs, and self-hosted benchmark sites; "
            "see docs/SIMULATION.md."
        ) from exc
    return (ScriptBrowserEnv, create_id_based_action, create_none_action,
            create_stop_action, ActionTypes, ActionParsingError, evaluator_router)


@contextmanager
def _native_judge_transport(model_override: str | None):
    """Keep upstream grading, but account its model requests through the SDK."""
    from webarena.evaluation_harness import helper_functions
    from vrlbench.runtime.llm import complete_messages

    def generate(**request):
        if model_override is not None:
            request["model"] = model_override
        return complete_messages(**request)

    original = helper_functions.generate_from_openai_chat_completion
    helper_functions.generate_from_openai_chat_completion = generate
    try:
        yield
    finally:
        helper_functions.generate_from_openai_chat_completion = original


class WebArenaSession:
    """One task, with a clean website and browser state for each complete trial."""

    system_prompt = (
        "Control the browser to complete the user's task. Return exactly one "
        "WebArena action, without explanation or Markdown fences. " + ACTION_RULES
    )
    basic_rules = (
        "Each trial starts from the same WebArena task after website and browser "
        "reset. A trial ends on stop or its action limit. Task success is checked "
        "by the official WebArena task evaluator, not browser action execution. "
        + ACTION_RULES
    )

    def __init__(self, case: Mapping[str, Any], params: Mapping[str, Any]):
        if not isinstance(case, Mapping) or not case.get("case_id") or not case.get("config_file"):
            raise ValueError("WebArena cases require case_id and config_file")
        self.case = dict(case)
        self.path = Path(str(case["config_file"])).resolve(strict=True)
        source = self.path.read_bytes()
        self.config = json.loads(source)
        if not isinstance(self.config, dict) or not isinstance(self.config.get("intent"), str):
            raise ValueError("WebArena task config must contain an intent string")
        self._source_sha256 = hashlib.sha256(source).hexdigest()
        evaluation = self.config.get("eval")
        if not isinstance(evaluation, dict) or not evaluation.get("eval_types"):
            raise ValueError("WebArena task config must declare eval.eval_types")
        if set(evaluation["eval_types"]) - {"string_match", "url_match", "program_html"}:
            raise ValueError("Unsupported WebArena evaluator type")
        self._uses_judge = False
        if "string_match" in evaluation["eval_types"]:
            references = evaluation.get("reference_answers")
            if not isinstance(references, dict) or not references:
                raise ValueError("WebArena string_match requires reference_answers")
            if set(references) - {"exact_match", "must_include", "fuzzy_match"}:
                raise ValueError("Unsupported WebArena string matching rule")
            self._uses_judge = "fuzzy_match" in references
        self._judge_model = params.get("judge_model")
        if "judge_model" in params and (not isinstance(self._judge_model, str) or not self._judge_model.strip()):
            raise ValueError("WebArena judge_model must be a nonempty model name")
        reset_spec = params.get("reset_factory")
        if not isinstance(reset_spec, dict):
            raise ValueError(
                "WebArena requires reset_factory: a trusted factory returning "
                "a callback(case) that restores the task websites before each trial"
            )
        self._reset_sites, self.reset_identity = create_plugin(reset_spec)
        if not callable(self._reset_sites):
            raise TypeError("WebArena reset_factory must return a callable reset(case)")
        self.source_identity = {
            "case_id": case["case_id"], "config_sha256": self._source_sha256,
            "reset_source_sha256": self.reset_identity.get("source_sha256"),
            "judge_model": self._judge_model or NATIVE_JUDGE_MODEL,
        }
        self._identity = json.dumps(self.source_identity, sort_keys=True)
        observation_type = params.get("observation_type", "accessibility_tree")
        if observation_type not in {"accessibility_tree", "html"}:
            raise ValueError("WebArena supports accessibility_tree or html observations")
        headless = params.get("headless", True)
        if type(headless) is not bool:
            raise ValueError("WebArena headless must be a boolean")
        (environment, self._parse, self._none, self._stop, self._types,
         self._parse_error, self._evaluator_router) = _upstream()
        self.env = environment(headless=headless, observation_type=observation_type)
        self.trajectory: list[Any] = []
        self._observation = None
        self._terminated = False
        self._score: float | None = None

    def _observation_from(self, obs, info):
        from .interactive import Observation

        if not isinstance(obs, dict) or not isinstance(obs.get("text"), str):
            raise RuntimeError("WebArena did not provide a text observation")
        page = info.get("page") if isinstance(info, dict) else None
        url = getattr(page, "url", "")
        text = f"URL: {url}\n{obs['text']}" if url else obs["text"]
        error = info.get("fail_error") if isinstance(info, dict) else None
        if error:
            text += f"\nBrowser action error: {error}"
        return Observation(text=text)

    def reset(self, *, trial_index: int):
        from .interactive import TaskState

        self._check_config()
        # The callback may return None on success, but explicit failure is fatal.
        if self._reset_sites(dict(self.case)) is False:
            raise RuntimeError("WebArena website reset failed")
        self._check_config()
        obs, info = self.env.reset(options={"config_file": str(self.path)})
        self.trajectory = [{"observation": obs, "info": info}]
        self._observation = self._observation_from(obs, info)
        self._terminated = False
        self._score = None
        return TaskState(task=self.config["intent"], observation=self._observation,
                         identity=self._identity)

    def parse_response(self, raw: str) -> str:
        """Validate the native grammar without rewriting or executing an action."""
        try:
            self._parse(raw)
        except self._parse_error as exc:
            raise ValueError(f"WebArena action syntax: {exc}. {ACTION_RULES}") from exc
        return raw

    def step(self, action: str):
        from .interactive import Transition

        if self._observation is None or self._terminated:
            raise RuntimeError("Reset WebArena before acting, and do not act after stop")
        try:
            parsed = self._parse(action)
        except self._parse_error as exc:
            # Invalid outputs consume one action just like other browser actions.
            parsed = self._none()
            parsed["raw_prediction"] = action
            self.trajectory.append(parsed)
            obs, _, terminated, truncated, info = self.env.step(parsed)
            self.trajectory.append({"observation": obs, "info": info})
            self._observation = self._observation_from(obs, info)
            from .interactive import Observation
            self._observation = Observation(text=(
                self._observation.text + f"\nInvalid action format: {exc}. {ACTION_RULES}"
            ))
            self._terminated = bool(terminated or truncated)
            return Transition(observation=self._observation,
                              terminated=bool(terminated), truncated=bool(truncated))
        parsed["raw_prediction"] = action
        self.trajectory.append(parsed)
        if parsed["action_type"] == self._types.STOP:
            self._terminated = True
            return Transition(observation=self._observation, terminated=True)
        obs, _, terminated, truncated, info = self.env.step(parsed)
        self.trajectory.append({"observation": obs, "info": info})
        self._observation = self._observation_from(obs, info)
        self._terminated = bool(terminated or truncated)
        return Transition(observation=self._observation,
                          terminated=bool(terminated), truncated=bool(truncated))

    def evaluate(self) -> bool:
        if not self.trajectory:
            raise RuntimeError("Cannot evaluate a WebArena task before reset")
        self._check_config()
        # The official evaluator expects a final action, including on budget exhaustion.
        if "observation" in self.trajectory[-1]:
            self.trajectory.append(self._stop(""))
        evaluator = self._evaluator_router(str(self.path))
        with _native_judge_transport(self._judge_model) if self._uses_judge else nullcontext():
            score = evaluator(trajectory=self.trajectory, config_file=str(self.path),
                              page=self.env.page, client=self.env.get_page_client(self.env.page))
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise RuntimeError(f"WebArena evaluator returned an invalid score: {score!r}")
        self._score = float(score)
        return self._score == 1.0

    @property
    def evaluation_score(self) -> float | None:
        return self._score

    def _check_config(self):
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != self._source_sha256:
            raise RuntimeError("WebArena task config changed during evaluation")

    def capture_observation(self) -> bytes:
        """Record the live browser viewport without changing text actor input."""
        if self._observation is None:
            raise RuntimeError("Reset WebArena before capturing an observation")
        return self.env.page.screenshot(type="png")

    def close(self):
        self.env.close()


def make_session(case, params):
    return WebArenaSession(case, params)


def run(spec, runtime):
    from .interactive import run_cases

    return run_cases(spec, runtime, make_session)
