"""Official WebShop text environment, native actions and read-only page capture."""

from __future__ import annotations

import os
import random
import re
import sys
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional

from vrlbench.environments.webshop_dependency import (
    WEBSHOP_FULL_DATA_FILES,
    WEBSHOP_OFFICIAL_TEST_GOAL_END,
    WEBSHOP_OFFICIAL_TEST_GOAL_START,
    webshop_observed_dependency_contract,
)

WEBSHOP_ACTION_PATTERN = (
    "(?:search|click)\\[(?!\\s)(?![^\\]\\r\\n]*\\s\\])[^\\]\\r\\n]+\\]"
)
DEFAULT_WEBSHOP_ROOT = (
    Path(os.environ["WEBSHOP_ROOT"]).expanduser()
    if os.environ.get("WEBSHOP_ROOT", "").strip()
    else None
)
WEBSHOP_PUBLIC_INTERACTION_RULES = "A displayed product option is required only when the shopping instruction explicitly names that option attribute or value. Select every such required option before click[buy now]. A matching product description does not select a visible option; use the current page's matching option action to establish it. Option values are grouped by their displayed attribute, and selecting a new value in one group replaces the previous value in that same group. Product-description requirements and option selections are distinct: choose an item matching the former and select only the instruction-named value for each required option group. Terminal feedback may report the selected options, and options {} means none was selected. Changing an option updates that option group for the current base product; it does not change the product identity. The scalar reward is an aggregate outcome score, not attribution to a particular requirement; before success it does not independently certify the base product or any selected option. If the instruction does not name a displayed option attribute, changing that optional value cannot by itself repair a failed candidate."
WEBSHOP_DSST_BASIC_ENVIRONMENT_RULES = (
    "Each complete attempt starts from the reset page for one fixed shopping instruction. The Actor receives a text page observation and may issue one search[query] or click[visible target] action per environment step. Each executed action returns the next page observation, reward, and termination status. The shopping instruction remains fixed across attempts, and success is determined only by the environment. "
    + WEBSHOP_PUBLIC_INTERACTION_RULES
)
SYSTEM_INSTRUCTION = (
    "Complete the WebShop shopping task using one search[query] or click[visible target] action. "
    "Return the native action only. Use exact lowercase click labels from the current observation, "
    "including click[buy now], and wait for feedback before choosing another action. "
    "A parser-valid action is dispatched once; the environment decides its validity. "
    + WEBSHOP_PUBLIC_INTERACTION_RULES
)


class _OfficialWebShopTextAdapter:
    """Adapter for the official WebShop text Gym environment."""

    def __init__(self, env: Any):
        self.env = env
        self.observation = ""
        self._sample_index: Optional[int] = None
        self._last_action_invalid = False

    def step(self, session: int | str, action: str):
        if action == "reset":
            observation, _info = self.env.reset(session=session)
            self.observation = str(observation)
            self._sample_index = _sample_index_from_session(session)
            self._last_action_invalid = False
            return (self._with_available_actions(self.observation), 0.0, False)
        self._last_action_invalid = not self._action_dispatch_is_valid(action)
        observation, reward, done, _info = self.env.step(action)
        self.observation = str(observation)
        return (
            self._with_available_actions(self.observation),
            float(reward),
            bool(done),
        )

    @property
    def last_action_invalid(self) -> bool:
        """Whether the most recent parser-valid action was a WebShop no-op."""
        return self._last_action_invalid

    def capture_observation(self) -> str:
        """Read current native HTML without changing the actor's text input."""
        page_source = getattr(getattr(self.env, "browser", None), "page_source", None)
        if not isinstance(page_source, str) or not page_source.strip():
            raise RuntimeError("WebShop has no current native HTML")
        return page_source

    def _action_dispatch_is_valid(self, action: str) -> bool:
        match = re.fullmatch("(search|click)\\[([^\\]\\r\\n]+)\\]", str(action))
        if match is None:
            return False
        action_name, argument = match.groups()
        if action_name == "search":
            return bool(argument)
        available = self.env.get_available_actions()
        clickables = {
            str(value).strip().lower()
            for value in available.get("clickables", []) or []
        }
        return argument.lower() in clickables and argument.lower() != "search"

    def reset_fixed(self, sample_index: int) -> tuple[str, float, bool]:
        """Reset with the integer session required for a fixed official goal."""
        if isinstance(sample_index, bool) or not isinstance(sample_index, int):
            raise ValueError("WebShop sample_index must be an integer")
        return self.step(sample_index, "reset")

    def _with_available_actions(
        self, observation: str, *, available: Optional[dict[str, Any]] = None
    ) -> str:
        if available is None:
            try:
                available = self.env.get_available_actions()
            except Exception:
                return observation
        actions = _available_webshop_actions(available)
        if not actions:
            return observation
        preview = "\n".join((f"- {action}" for action in actions[:40]))
        if len(actions) > 40:
            preview += f"\n- ... ({len(actions) - 40} more)"
        return f"{observation}\n\nAvailable actions:\n{preview}"


def _canonical_webshop_instruction(instruction: str) -> str:
    """Canonicalize one goal across renderer and environment representations."""
    text = str(instruction).strip()
    label = "Instruction:"
    if text.startswith(label):
        text = text[len(label) :]
    return " ".join(text.split())


def _parse_webshop_actor_output(response: str) -> str:
    lines = str(response or "").strip().splitlines()
    if len(lines) != 1 or not re.fullmatch(WEBSHOP_ACTION_PATTERN, lines[0]):
        return ""
    return lines[0]


def _sanitize_webshop_terminal_feedback(observation: str) -> str:
    """Remove evaluator-only target payloads from terminal WebShop pages."""
    text = str(observation or "")
    text = re.sub(
        "\\[SEP\\]\\s*Target\\s*\\[SEP\\].*",
        "[SEP] Target details redacted",
        text,
        flags=re.I | re.S,
    )
    text = re.sub(
        "\\[SEP\\]\\s*Reward Details\\s*\\[SEP\\].*",
        "[SEP] Reward details redacted",
        text,
        flags=re.I | re.S,
    )
    return text


def _shared_webshop_feedback(observation: str, reward: float, *, terminal: bool) -> str:
    """Expose the same public WebShop outcome signal to every method."""
    text = (
        _sanitize_webshop_terminal_feedback(observation)
        if terminal
        else str(observation or "")
    )
    if terminal or float(reward) != 0.0:
        text = f"{text}\nPublic environment reward: {float(reward)}"
    return text


def _fixed_webshop_session(sample_index: int) -> int:
    return int(sample_index)


def _extract_instruction(text: str) -> str:
    match = re.search(
        "Instruction:\\s*(?:\\[SEP\\]\\s*)?([^\\r\\n]*?)(?=\\s*\\[SEP\\]|\\r?\\n|$)",
        str(text or ""),
    )
    if not match:
        return ""
    return _canonical_webshop_instruction(match.group(1))


def _sample_index_from_session(session: str) -> Optional[int]:
    fixed_match = re.match("^(?:vex2_)?fixed_(\\d+)$", str(session))
    if fixed_match:
        return int(fixed_match.group(1))
    match = re.match("^vex2_(\\d+)_\\d+$", str(session))
    if match:
        return int(match.group(1))
    text = str(session)
    return int(text) if text.isdigit() else None


def _available_webshop_actions(available: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    if available.get("has_search_bar"):
        actions.append("search[query]")
    for clickable in available.get("clickables", []) or []:
        text = str(clickable).strip().lower()
        if not text or text.lower() == "search":
            continue
        actions.append(f"click[{text}]")
    return actions


def _load_default_env_factory() -> Callable[[], Any]:
    root = DEFAULT_WEBSHOP_ROOT
    if root is None:
        raise ImportError(
            "Set WEBSHOP_ROOT to the prepared official WebShop checkout before running WebShop. The dependency is not installed inside vrlbench."
        )
    adapter = os.environ.get("WEBSHOP_ENV_ADAPTER", "official_text").strip().lower()
    if adapter != "official_text":
        raise ValueError(
            "WebShop evaluation requires WEBSHOP_ENV_ADAPTER=official_text; alternative wrappers are not supported by this adapter"
        )
    _validate_official_webshop_runtime_settings()
    root = Path(root).expanduser().resolve()
    observed = webshop_observed_dependency_contract(root)
    product_file = root / WEBSHOP_FULL_DATA_FILES[0]
    attribute_file = root / WEBSHOP_FULL_DATA_FILES[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    with _pushd(root):
        from web_agent_site.envs import WebAgentTextEnv
        from web_agent_site.engine import engine as webshop_engine
    webshop_engine.DEFAULT_ATTR_PATH = str(attribute_file)
    factory = _make_official_webshop_env_factory(WebAgentTextEnv, product_file)
    factory._vrlbench_environment_identity = {
        "benchmark": "webshop",
        "source_data_index_root_sha256": observed["source_data_index_root_sha256"],
    }
    return factory


def _validate_official_webshop_runtime_settings() -> None:
    data_profile = os.environ.get("WEBSHOP_DATA_PROFILE", "full").strip().lower()
    if data_profile != "full":
        raise ValueError(
            "official WebShop evaluation requires WEBSHOP_DATA_PROFILE=full"
        )
    observation_mode = os.environ.get("WEBSHOP_OBSERVATION_MODE", "text").strip()
    if observation_mode != "text":
        raise ValueError(
            "official WebShop evaluation requires WEBSHOP_OBSERVATION_MODE=text"
        )
    human_goals = os.environ.get("WEBSHOP_HUMAN_GOALS", "1").strip().lower()
    if human_goals != "1":
        raise ValueError("official WebShop evaluation requires WEBSHOP_HUMAN_GOALS=1")
    goal_seed = os.environ.get("WEBSHOP_GOAL_SEED", "233").strip()
    if goal_seed != "233":
        raise ValueError("official WebShop evaluation requires WEBSHOP_GOAL_SEED=233")
    if os.environ.get("WEBSHOP_NUM_PRODUCTS", "").strip():
        raise ValueError(
            "official WebShop full-data evaluation requires WEBSHOP_NUM_PRODUCTS unset"
        )


def _make_official_webshop_env_factory(
    env_type: Callable[..., Any], product_file: Path
) -> Callable[[], _OfficialWebShopTextAdapter]:
    """Create a lazy, run-scoped singleton official environment.

    WebShop loads the full 1.18M-product catalog and opens Lucene while building
    ``SimServer``. Runs are sequential, so one text environment can safely serve
    every sample. ``make_actor`` still performs the official integer-session
    reset at the start of every trial, which clears navigation state while
    preserving the fixed human test goal.
    """
    shared_adapter: dict[str, _OfficialWebShopTextAdapter] = {}
    common = {
        "observation_mode": "text",
        "session_prefix": os.environ.get("WEBSHOP_SESSION_PREFIX", "vex2_"),
    }

    def make_env() -> _OfficialWebShopTextAdapter:
        if "adapter" not in shared_adapter:
            _seed_official_webshop_goal_generation()
            env = env_type(
                **common, file_path=str(product_file), num_products=None, human_goals=1
            )
            server = env.server
            goals = getattr(server, "goals", ())
            if len(goals) <= WEBSHOP_OFFICIAL_TEST_GOAL_END:
                raise RuntimeError(
                    f"official WebShop human-goal server does not contain the full test split 0..{WEBSHOP_OFFICIAL_TEST_GOAL_END}"
                )
            shared_adapter["adapter"] = _OfficialWebShopTextAdapter(env)
        return shared_adapter["adapter"]

    return make_env


def _seed_official_webshop_goal_generation() -> None:
    """Stabilize official WebShop product prices before SimServer creation.

    The pinned runtime samples missing product prices before applying its own
    fixed human-goal shuffle. Re-seeding once per condition keeps integer test
    goal indices stable across separate baseline and comparison processes.
    """
    random.seed(233)


@contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
