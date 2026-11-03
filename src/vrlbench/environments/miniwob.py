"""Native MiniWoB observations and actions; agent execution lives in the actor backend."""

from __future__ import annotations

import ast
import io
import re
import tokenize
from html import unescape
from typing import Any, Callable, List, Optional, Tuple

from vrlbench.environments.synapse_dependency import configure_synapse_import_path
from vrlbench.runtime.budget import EnvironmentInteractionLedger

MINIWOB_PUBLIC_INTERACTION_RULES = "agent.movemouse only hovers. agent.type and agent.press act on the currently focused control. An HTML control marked readonly cannot accept direct typing. XPath tag tests for SVG elements are namespace-sensitive; target an SVG element with a namespace-stable selector such as `//*[@id='target']` or `//*[local-name()='circle' and @id='target']`, not an unprefixed tag selector such as `//circle`. Each click XPath is resolved against the current DOM when that primitive executes, after any earlier action in the same block."
MINIWOB_DSST_BASIC_ENVIRONMENT_RULES = (
    "Each complete attempt starts from a browser reset in the named MiniWoB task family. The Actor receives a natural-language task and the current HTML observation. It may interact only through click_xpath, type, press, click_option, and movemouse. Each executed primitive action returns environment feedback and consumes one A-O step. Reset-local task values and DOM identifiers may change across attempts; success is determined only by the environment. "
    + MINIWOB_PUBLIC_INTERACTION_RULES
)
MINIWOB_ALLOWED_AGENT_METHODS = (
    "click_xpath",
    "type",
    "press",
    "click_option",
    "movemouse",
)
MINIWOB_ALLOWED_PRESS_KEYS = frozenset(
    {
        "enter",
        "space",
        "arrowleft",
        "arrowright",
        "backspace",
        "arrowup",
        "arrowdown",
        "command+a",
        "command+c",
        "command+v",
    }
)
_mw_stack = None


def _task_from_observation_html(obs: str, fallback: str) -> str:
    """Use the visible MiniWoB query when env utterance and DOM drift apart."""
    match = re.search(
        "<div\\s+id=[\\\"']query[\\\"'][^>]*>(.*?)</div>",
        obs,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return fallback
    text = re.sub("<[^>]+>", "", match.group(1))
    text = unescape(re.sub("\\s+", " ", text)).strip()
    return text or fallback


def _load_mw_stack():
    global _mw_stack
    if _mw_stack is None:
        configure_synapse_import_path()
        from synapse.envs.miniwob.environment import MiniWoBEnv
        from synapse.envs.miniwob.action import (
            MiniWoBType,
            MiniWoBElementClickXpath,
            MiniWoBElementClickOption,
            MiniWoBMoveXpath,
        )
        from selenium.webdriver.common.keys import Keys

        _mw_stack = (
            MiniWoBEnv,
            MiniWoBType,
            MiniWoBElementClickXpath,
            MiniWoBElementClickOption,
            MiniWoBMoveXpath,
            Keys,
        )
    return _mw_stack


class _AgentActions:
    """Thin wrapper that translates Python API calls into Synapse actions."""

    def __init__(
        self,
        env: Any,
        *,
        trial_index: int = 0,
        interaction_ledger: Optional[EnvironmentInteractionLedger] = None,
        after_step: Optional[Callable[[int], None]] = None,
    ) -> None:
        self.env = env
        self.done = False
        self.reward = 0.0
        self.last_obs = ""
        self.step_log: List[Tuple[str, str]] = []
        self.trial_index = int(trial_index)
        self._step_index = 0
        self.interaction_ledger = interaction_ledger
        self.after_step = after_step

    def _step(self, action: Any, action_repr: str) -> None:
        if self.done:
            return
        if self.interaction_ledger is not None:
            self.interaction_ledger.require_step()
        self._step_index += 1
        obs, reward, done, _ = self.env.step(action)
        if self.interaction_ledger is not None:
            self.interaction_ledger.charge_step()
        self.last_obs = obs if obs else ""
        self.step_log.append((action_repr, self.last_obs))
        if done:
            self.done = True
            self.reward = reward
        if self.after_step is not None:
            self.after_step(self._step_index)

    def click_xpath(self, xpath: str):
        _, _, ClickXpath, _, _, _ = _load_mw_stack()
        self._step(ClickXpath(xpath), f"agent.click_xpath({xpath!r})")

    def type(self, characters: str):
        _, Type, _, _, _, _ = _load_mw_stack()
        self._step(Type(characters), f"agent.type({characters!r})")

    def press(self, key_type: str):
        _, Type, _, _, _, Keys = _load_mw_stack()
        key_map = {
            "enter": "\n",
            "space": " ",
            "arrowleft": Keys.LEFT,
            "arrowright": Keys.RIGHT,
            "backspace": Keys.BACKSPACE,
            "arrowup": Keys.UP,
            "arrowdown": Keys.DOWN,
        }
        if key_type in key_map:
            self._step(Type(key_map[key_type]), f"agent.press({key_type!r})")
        elif key_type in ("command+a", "command+c", "command+v"):
            self._step(Type(key_type), f"agent.press({key_type!r})")
        else:
            raise ValueError(f"Invalid key: {key_type}")

    def click_option(self, xpath: str):
        _, _, _, ClickOption, _, _ = _load_mw_stack()
        self._step(ClickOption(xpath), f"agent.click_option({xpath!r})")

    def movemouse(self, xpath: str):
        _, _, _, _, MoveXpath, _ = _load_mw_stack()
        self._step(MoveXpath(xpath), f"agent.movemouse({xpath!r})")


class MiniWoBActionParseError(ValueError):
    """Raised when an actor response violates the shared action grammar."""


def _unwrap_agent_code(response: str) -> str:
    """Accept raw Python or exactly one complete Python code fence."""
    text = str(response or "").strip()
    if not text:
        raise MiniWoBActionParseError("empty actor response")
    if text.startswith("```"):
        match = re.fullmatch(
            "```(?:python)?[ \\t]*\\r?\\n(?P<code>.*?)\\r?\\n```",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if not match:
            raise MiniWoBActionParseError(
                "response must be exactly one complete Python code fence"
            )
        text = match.group("code").strip()
    elif "```" in text:
        raise MiniWoBActionParseError("code fences cannot be mixed with other text")
    if not text:
        raise MiniWoBActionParseError("actor response contains no calls")
    return text


def _parse_agent_actions(response: str) -> List[Tuple[str, str]]:
    """Accept only direct native Agent API calls, with no executable Python."""
    code = _unwrap_agent_code(response)
    try:
        for token in tokenize.generate_tokens(io.StringIO(code).readline):
            if token.type == tokenize.COMMENT or (
                token.type == tokenize.OP and token.string == ";"
            ):
                raise MiniWoBActionParseError("comments and semicolons are not allowed")
        tree = ast.parse(code, mode="exec")
    except (SyntaxError, IndentationError, tokenize.TokenError) as exc:
        raise MiniWoBActionParseError(f"invalid action syntax: {exc}") from exc
    actions: List[Tuple[str, str]] = []
    for statement in tree.body:
        if not isinstance(statement, ast.Expr) or not isinstance(
            statement.value, ast.Call
        ):
            raise MiniWoBActionParseError("expected one direct agent method call")
        call = statement.value
        if (
            not isinstance(call.func, ast.Attribute)
            or not isinstance(call.func.value, ast.Name)
            or call.func.value.id != "agent"
            or call.func.attr not in MINIWOB_ALLOWED_AGENT_METHODS
        ):
            raise MiniWoBActionParseError("unknown or non-agent method")
        if (
            call.keywords
            or len(call.args) != 1
            or not isinstance(call.args[0], ast.Constant)
            or not isinstance(call.args[0].value, str)
        ):
            raise MiniWoBActionParseError(
                "agent actions require exactly one string literal argument"
            )
        argument = call.args[0].value
        if call.func.attr == "press" and argument not in MINIWOB_ALLOWED_PRESS_KEYS:
            raise MiniWoBActionParseError(f"unsupported press key: {argument!r}")
        actions.append((call.func.attr, argument))
    if not actions:
        raise MiniWoBActionParseError("action contains no calls")
    return actions


def _canonical_agent_code(actions: List[Tuple[str, str]]) -> str:
    return "\n".join((f"agent.{method}({argument!r})" for method, argument in actions))
