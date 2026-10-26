from __future__ import annotations
import ast
from copy import deepcopy
import hashlib
import io
import json
import logging
import os
import re
import signal
import time
import tokenize
from html import unescape
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from vrlbench.environments.synapse_dependency import configure_synapse_import_path
from vrlbench.runtime.budget import EnvironmentBudgetExceeded, EnvironmentInteractionLedger
from vrlbench.runtime.miniwob_task_scope import MINIWOB_TASKS
from vrlbench.runtime.reset_protocol import miniwob_reset_seed
from vrlbench.runtime.shared_t1 import validate_actor_only_t1_usage, validate_shared_t1_environment_accounting, validate_shared_t1_source
logger = logging.getLogger(__name__)

def _task_from_observation_html(obs: str, fallback: str) -> str:
    """Use the visible MiniWoB query when env utterance and DOM drift apart."""
    match = re.search('<div\\s+id=[\\"\']query[\\"\'][^>]*>(.*?)</div>', obs, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return fallback
    text = re.sub('<[^>]+>', '', match.group(1))
    text = unescape(re.sub('\\s+', ' ', text)).strip()
    return text or fallback

def _resolve_shared_t1_task_dir(shared_t1_dir: str, task_name: str, model: str, trials: int) -> Optional[Path]:
    root = Path(shared_t1_dir)
    direct = root / 'env_results_trial_0.json'
    if direct.exists():
        return root
    candidates = [root / f'{task_name}_base_{model}_T1', root / f'{task_name}_base_{model}_T{trials}', root / f'{task_name}_base_{model}']
    for cand in candidates:
        if (cand / 'env_results_trial_0.json').exists():
            return cand
    return None

def _load_shared_t1(task_dir: Path, task_name: Optional[str]=None, *, require_failure_trajectory: bool=True, require_environment_accounting: bool=True, require_complete_accounting: bool=False, expected_max_environment_steps: Optional[int]=None, expected_trial_seed_mode: Optional[str]=None) -> Dict[int, Dict[str, object]]:
    del require_failure_trajectory
    episodes_path = task_dir / 'episodes.json'
    if not episodes_path.is_file():
        raise FileNotFoundError(f'shared first-trial episodes are missing: {episodes_path}')
    episodes = json.loads(episodes_path.read_text(encoding='utf-8'))
    out: Dict[int, Dict[str, object]] = {}
    for episode in episodes:
        if task_name is not None and episode.get('task') != task_name:
            continue
        seed = int(episode.get('episode', len(out)))
        trial0 = next((trial for trial in episode.get('trials', []) if trial.get('trial') == 0), None)
        if trial0 is None:
            continue
        if seed in out:
            raise ValueError(f'duplicate shared first-trial episode {seed}')
        trial_metadata = trial0.get('trial_metadata')
        llm_usage = trial0.get('llm_usage')
        validate_actor_only_t1_usage(llm_usage if isinstance(llm_usage, dict) else None, source=f'{episodes_path} episode {seed}', require_complete_accounting=require_complete_accounting)
        if require_environment_accounting:
            if expected_max_environment_steps is None:
                raise ValueError('shared first trial requires an A-O horizon')
            trial_metadata = validate_shared_t1_environment_accounting(trial_metadata if isinstance(trial_metadata, dict) else None, benchmark='miniwob', source=f'{episodes_path} episode {seed}', expected_max_environment_steps=expected_max_environment_steps, success=bool(trial0.get('success', False)))
            if int(trial_metadata.get('base_seed', -1)) != seed:
                raise ValueError('shared first-trial base seed differs')
            if int(trial_metadata.get('reset_seed', -1)) != seed:
                raise ValueError('shared first-trial reset seed differs')
            if expected_trial_seed_mode is not None and trial_metadata.get('trial_seed_mode') != expected_trial_seed_mode:
                raise ValueError('shared first-trial seed mode differs')
        restored_metadata = dict(trial_metadata or {})
        row = {'success': bool(trial0.get('success', False)), 'trajectory': str(trial0.get('trajectory', '')), 'policy_trajectory': str(trial0.get('policy_trajectory', '')), 'actor_trajectory': str(trial0.get('actor_trajectory', '')), 'trial_metadata': restored_metadata, 'llm_usage': llm_usage}
        if not row['success'] and (not str(row['trajectory']).strip()):
            raise ValueError(f'shared first-trial failure lacks trajectory: {seed}')
        if not str(row['actor_trajectory']).strip():
            raise ValueError(f'shared first trial lacks Actor trajectory: {seed}')
        out[seed] = row
    if not out:
        raise ValueError(f'no shared first-trial episodes found in {episodes_path}')
    return out
_mw_stack = None

def _load_mw_stack():
    global _mw_stack
    if _mw_stack is None:
        configure_synapse_import_path()
        from synapse.envs.miniwob.environment import MiniWoBEnv
        from synapse.envs.miniwob.action import MiniWoBType, MiniWoBElementClickXpath, MiniWoBElementClickOption, MiniWoBMoveXpath
        from selenium.webdriver.common.keys import Keys
        _mw_stack = (MiniWoBEnv, MiniWoBType, MiniWoBElementClickXpath, MiniWoBElementClickOption, MiniWoBMoveXpath, Keys)
    return _mw_stack
MINIWOB_BOUNDED_REPEAT_MAX = 128
MINIWOB_BOUNDED_REPEAT_VARIABLE = '_'
DEFAULT_MAX_ENVIRONMENT_STEPS = 128
MINIWOB_PUBLIC_INTERACTION_RULES = "agent.movemouse only hovers. agent.type and agent.press act on the currently focused control. An HTML control marked readonly cannot accept direct typing. XPath tag tests for SVG elements are namespace-sensitive; target an SVG element with a namespace-stable selector such as `//*[@id='target']` or `//*[local-name()='circle' and @id='target']`, not an unprefixed tag selector such as `//circle`. Each click XPath is resolved against the current DOM when that primitive executes, after any earlier action in the same block."
MINIWOB_DSST_BASIC_ENVIRONMENT_RULES = 'Each complete attempt starts from a browser reset in the named MiniWoB task family. The Actor receives a natural-language task and the current HTML observation. It may interact only through click_xpath, type, press, click_option, and movemouse. Each executed primitive action returns environment feedback and consumes one A-O step. Reset-local task values and DOM identifiers may change across attempts; success is determined only by the environment. ' + MINIWOB_PUBLIC_INTERACTION_RULES
REACT_BATCH_REMAINING_ENVIRONMENT_BUDGET_PROMPT_TEMPLATE = 'Current trial action budget: {remaining_steps} action opportunity/opportunities remain. Each executed primitive Agent API call consumes one opportunity. A rejected response with no dispatchable action also consumes one opportunity, leaves the environment unchanged, and is followed by another decision if budget remains. Generate actions until choosing the next action genuinely requires a new Observation; do not request more actions than remain.'
REACT_BATCH_TURN_SUFFIX = 'Respond now using the required Thought/Action format.'
SYNAPSE_LIGHTWEIGHT_REMAINING_ENVIRONMENT_BUDGET_PROMPT_TEMPLATE = REACT_BATCH_REMAINING_ENVIRONMENT_BUDGET_PROMPT_TEMPLATE
MINIWOB_REMAINING_ENVIRONMENT_BUDGET_PROMPT_TEMPLATE = REACT_BATCH_REMAINING_ENVIRONMENT_BUDGET_PROMPT_TEMPLATE
REACT_BATCH_SYSTEM_INSTRUCTION = 'Interact with a web browser to complete a task. You will be given the task and the current HTML observation. At each turn, use the provided Agent API to emit an action block. The block may contain one or more actions whose order is already determined by the current Observation. After the block executes, you will receive the latest Observation before choosing another block.\n\nAgent API:\n  agent.click_xpath(xpath: str)     # Click an element by xpath\n  agent.type(text: str)             # Type text via keyboard\n  agent.press(key: str)             # Press key: enter, space, arrowleft, arrowright, backspace, arrowup, arrowdown, command+a, command+c, command+v\n  agent.click_option(xpath: str)    # Click an option in a dropdown\n  agent.movemouse(xpath: str)       # Move mouse to element\n\nUse only these Agent API calls. Do not use JavaScript, DOM mutation, or\nXPath expressions that read attributes instead of clicking elements.\n' + MINIWOB_PUBLIC_INTERACTION_RULES + '\nBriefly reason, then generate one or more actions until choosing the next action genuinely requires a new Observation. Your answer must contain exactly one ``Thought:`` field followed by one ``Action:`` field. The Action field must contain exactly one Python code block with one or more direct top-level Agent API calls, one call per line. Each call must have exactly one positional string literal. Do not include bullet lists or variable binding notes. Do not use semicolons, keyword arguments, nested calls, variables, conditionals, comprehensions, helper functions, imports, or generated calls. Repeated identical actions may use only ``for _ in range(N):`` with a nonnegative integer literal ``N`` and exactly one direct Agent API call in the loop body. Each expanded action consumes one environment step. Do not include a later action when choosing it depends on feedback from an earlier action in the same block; wait for the next Observation instead.\nIf method memory explicitly requires one non-environment auxiliary metadata\nline, append exactly that line after the code block. The harness validates and\nremoves it before parsing or dispatching executable actions.\n\nWrap your code in ``` blocks. Here are format examples:\n\nTask: Click on the "Submit" button.\nObservation:\n<div id="wrap"><div id="query">Click on the "Submit" button.</div><div id="area"><button id="subbtn">Submit</button><button id="other">Cancel</button></div></div>\nThought: The requested button is visible and uniquely identified by id.\nAction:\n```\nagent.click_xpath("//button[@id=\'subbtn\']")\n```\n\nTask: Enter "hello" into the text field and press submit.\nObservation:\n<div id="wrap"><div id="query">Enter "hello" into the text field and press submit.</div><div id="area"><input type="text" id="input"/><button id="subbtn">Submit</button></div></div>\nThought: The field, requested text, and submit control are all known from the current Observation, so these actions form one safe block.\nAction:\n```\nagent.click_xpath("//input[@id=\'input\']")\nagent.type("hello")\nagent.click_xpath("//button[@id=\'subbtn\']")\n```\n\nTask: Open the menu and select "Target".\nObservation:\n<div id="wrap"><button id="menu">Open</button><div role="menuitem">Target</div><div role="menuitem">Other</div></div>\nThought: Opening changes visibility, but the requested target and its selector are already determined by this Observation, so both actions form one block.\nAction:\n```\nagent.click_xpath("//button[@id=\'menu\']")\nagent.click_xpath("//*[@role=\'menuitem\' and text()=\'Target\']")\n```\n\nTask: Click the SVG item whose id is "target".\nObservation:\n<div id="wrap"><svg><circle id="target" r="4"/><circle id="other" r="4"/></svg></div>\nThought: SVG tag names are namespace-sensitive in XPath, so I will use the visible target id with a namespace-stable wildcard selector.\nAction:\n```\nagent.click_xpath("//*[@id=\'target\']")\n```\n'
SYSTEM_INSTRUCTION = REACT_BATCH_SYSTEM_INSTRUCTION
SYNAPSE_LIGHTWEIGHT_SYSTEM_INSTRUCTION = 'Interact with a web browser to complete a task. You will be given the task and the current HTML observation. Generate Python code using the provided Agent API to perform actions.\n\nAgent API:\n  agent.click_xpath(xpath: str)     # Click an element by xpath\n  agent.type(text: str)             # Type text via keyboard\n  agent.press(key: str)             # Press key: enter, space, arrowleft, arrowright, backspace, arrowup, arrowdown, command+a, command+c, command+v\n  agent.click_option(xpath: str)    # Click an option in a dropdown\n  agent.movemouse(xpath: str)       # Move mouse to element\n\nUse only these Agent API calls. Do not use JavaScript, DOM mutation, or\nXPath expressions that read attributes instead of clicking elements. If an\nearlier Action Constraint says an element/action was rejected or disabled, do\nnot retry that action through hover, key presses, alternate XPath, or other\nworkarounds unless the current observation explicitly shows the state changed.\nYour answer must be executable code only: one fenced block containing only\nagent.* calls, one call per line. Do not include explanations, bullet lists,\nvariable binding notes, or prose inside the code block.\nDo not use Python variables, loops, conditionals, comprehensions, helper\nfunctions, imports, or generated calls. When a task requires several actions,\nwrite every concrete agent.* call literally on its own line in execution order;\nperform all visible prerequisites before the terminal Submit action.\nWrap your code in ``` blocks. Here are two examples:\n\nTask: Click on the "Submit" button.\nObservation:\n<div id="wrap"><div id="query">Click on the "Submit" button.</div><div id="area"><button id="subbtn">Submit</button><button id="other">Cancel</button></div></div>\nAction:\n```\nagent.click_xpath("//button[@id=\'subbtn\']")\n```\n\nTask: Enter "hello" into the text field and press submit.\nObservation:\n<div id="wrap"><div id="query">Enter "hello" into the text field and press submit.</div><div id="area"><input type="text" id="input"/><button id="subbtn">Submit</button></div></div>\nAction:\n```\nagent.type("hello")\nagent.click_xpath("//button[@id=\'subbtn\']")\n```\n'
ACTION_GRAMMAR_SUMMARY = 'Allowed executable actions: agent.click_xpath(xpath), agent.type(text), agent.press("enter"|"space"|"arrowleft"|"arrowright"|"backspace"|"arrowup"|"arrowdown"|"command+a"|"command+c"|"command+v"), agent.click_option(xpath), agent.movemouse(xpath).'
SYNAPSE_LIGHTWEIGHT_ACTION_GRAMMAR_SUMMARY = 'Allowed executable actions: agent.click_xpath(xpath), agent.type(text), agent.press("enter"|"space"|"arrowleft"|"arrowright"|"backspace"|"arrowup"|"arrowdown"|"command+a"|"command+c"|"command+v"), agent.click_option(xpath), agent.movemouse(xpath).'
MINIWOB_ALLOWED_AGENT_METHODS = ('click_xpath', 'type', 'press', 'click_option', 'movemouse')
MINIWOB_ALLOWED_PRESS_KEYS = frozenset({'enter', 'space', 'arrowleft', 'arrowright', 'backspace', 'arrowup', 'arrowdown', 'command+a', 'command+c', 'command+v'})
MINIWOB_ACTOR_EXAMPLES = ({'task': 'Click on the "Submit" button.', 'observation': '<div id="wrap"><div id="query">Click on the "Submit" button.</div><div id="area"><button id="subbtn">Submit</button><button id="other">Cancel</button></div></div>', 'actions': ('agent.click_xpath("//button[@id=\'subbtn\']")',), 'thought': 'The requested button is visible and uniquely identified by id.'}, {'task': 'Enter "hello" into the text field and press submit.', 'observation': '<div id="wrap"><div id="query">Enter "hello" into the text field and press submit.</div><div id="area"><input type="text" id="input"/><button id="subbtn">Submit</button></div></div>', 'actions': ('agent.click_xpath("//input[@id=\'input\']")', 'agent.type("hello")', 'agent.click_xpath("//button[@id=\'subbtn\']")'), 'thought': 'The field, text, and submit control are all known, so the actions form one safe block.'}, {'task': 'Open the menu and select "Target".', 'observation': '<div id="wrap"><button id="menu">Open</button><div role="menuitem">Target</div><div role="menuitem">Other</div></div>', 'actions': ('agent.click_xpath("//button[@id=\'menu\']")', 'agent.click_xpath("//*[@role=\'menuitem\' and text()=\'Target\']")'), 'thought': 'Opening changes visibility, but the target and selector are already known, so both actions form one safe block.'}, {'task': 'Click the SVG item whose id is "target".', 'observation': '<div id="wrap"><svg><circle id="target" r="4"/><circle id="other" r="4"/></svg></div>', 'actions': ('agent.click_xpath("//*[@id=\'target\']")',), 'thought': 'SVG tag names are namespace-sensitive in XPath, so the visible target id should use a namespace-stable wildcard selector.'})
MINIWOB_ACTION_GRAMMAR_SUMMARY = 'Parser contract: one Thought field followed by one Action field containing one Python code fence with direct top-level agent.<allowlist>(string literal) calls or a top-level for _ in range(N) bounded repeat with one such call in its body. The response is accepted or rejected before dispatch. Allowed methods: ' + ', '.join((f'agent.{name}' for name in MINIWOB_ALLOWED_AGENT_METHODS)) + '.'
SYNAPSE_LIGHTWEIGHT_PARSER_GRAMMAR_SUMMARY = 'Parser contract: one raw Python response or one Python code fence with one or more direct top-level agent.<allowlist>(string literal) calls, accepted or rejected atomically before sequential dispatch. Allowed methods: ' + ', '.join((f'agent.{name}' for name in MINIWOB_ALLOWED_AGENT_METHODS)) + '.'

def _normalize_actor_profile(value: str) -> str:
    profile = str(value or '').strip().lower()
    if profile not in {'react_batch', 'synapse_lightweight'}:
        raise ValueError('MINIWOB_ACTOR_PROFILE must be react_batch or synapse_lightweight')
    return profile

def _actor_profile() -> str:
    return _normalize_actor_profile(os.environ.get('MINIWOB_ACTOR_PROFILE', 'react_batch'))

def _system_instruction_for_profile(profile: str) -> str:
    if profile == 'synapse_lightweight':
        return SYNAPSE_LIGHTWEIGHT_SYSTEM_INSTRUCTION
    return SYSTEM_INSTRUCTION

def _action_grammar_for_profile(profile: str) -> str:
    if profile == 'synapse_lightweight':
        return SYNAPSE_LIGHTWEIGHT_PARSER_GRAMMAR_SUMMARY
    return MINIWOB_ACTION_GRAMMAR_SUMMARY

def miniwob_dsst_basic_environment_rules(task_name: str, *, profile: str) -> str:
    """Expose the public within-attempt horizon without supplying task strategy."""
    return MINIWOB_DSST_BASIC_ENVIRONMENT_RULES + ' One action block may contain multiple sequential primitive actions. Each primitive consumes one action opportunity. A rejected response consumes one opportunity without executing an environment action; the Actor may continue while the trial action budget remains. There is no separate task-specific model-decision cap.'

class _AgentActions:
    """Thin wrapper that translates Python API calls into Synapse actions."""

    def __init__(self, env: Any, *, trial_index: int=0, interaction_ledger: Optional[EnvironmentInteractionLedger]=None, after_step: Optional[Callable[[int], None]]=None) -> None:
        self.env = env
        self.done = False
        self.reward = 0.0
        self.last_obs = ''
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
        self.last_obs = obs if obs else ''
        self.step_log.append((action_repr, self.last_obs))
        if done:
            self.done = True
            self.reward = reward
        if self.after_step is not None:
            self.after_step(self._step_index)

    def click_xpath(self, xpath: str):
        _, _, ClickXpath, _, _, _ = _load_mw_stack()
        self._step(ClickXpath(xpath), f'agent.click_xpath({xpath!r})')

    def type(self, characters: str):
        _, Type, _, _, _, _ = _load_mw_stack()
        self._step(Type(characters), f'agent.type({characters!r})')

    def press(self, key_type: str):
        _, Type, _, _, _, Keys = _load_mw_stack()
        key_map = {'enter': '\n', 'space': ' ', 'arrowleft': Keys.LEFT, 'arrowright': Keys.RIGHT, 'backspace': Keys.BACKSPACE, 'arrowup': Keys.UP, 'arrowdown': Keys.DOWN}
        if key_type in key_map:
            self._step(Type(key_map[key_type]), f'agent.press({key_type!r})')
        elif key_type in ('command+a', 'command+c', 'command+v'):
            self._step(Type(key_type), f'agent.press({key_type!r})')
        else:
            raise ValueError(f'Invalid key: {key_type}')

    def click_option(self, xpath: str):
        _, _, _, ClickOption, _, _ = _load_mw_stack()
        self._step(ClickOption(xpath), f'agent.click_option({xpath!r})')

    def movemouse(self, xpath: str):
        _, _, _, _, MoveXpath, _ = _load_mw_stack()
        self._step(MoveXpath(xpath), f'agent.movemouse({xpath!r})')

class MiniWoBActionParseError(ValueError):
    """Raised when an actor response violates the shared action grammar."""

def _unwrap_agent_code(response: str) -> str:
    """Accept raw Python or exactly one complete Python code fence."""
    text = str(response or '').strip()
    if not text:
        raise MiniWoBActionParseError('empty actor response')
    if text.startswith('```'):
        match = re.fullmatch('```(?:python)?[ \\t]*\\r?\\n(?P<code>.*?)\\r?\\n```', text, flags=re.DOTALL | re.IGNORECASE)
        if not match:
            raise MiniWoBActionParseError('response must be exactly one complete Python code fence')
        text = match.group('code').strip()
    elif '```' in text:
        raise MiniWoBActionParseError('code fences cannot be mixed with other text')
    if not text:
        raise MiniWoBActionParseError('actor response contains no calls')
    return text

def _unwrap_react_step(response: str) -> str:
    """Return the Action payload from one minimal Thought-Action turn."""
    text = str(response or '').strip()
    match = re.fullmatch('Thought:\\s*(?P<thought>.+?)\\r?\\nAction:\\s*(?P<action>.+)', text, flags=re.DOTALL)
    if match is None or not match.group('thought').strip():
        raise MiniWoBActionParseError('response must contain one non-empty Thought followed by one Action')
    return match.group('action').strip()

def _parse_agent_actions(response: str, *, allow_bounded_repeat: bool=False, allow_comments: bool=False) -> List[Tuple[str, str]]:
    """Parse and validate the complete response before any action can run."""
    code = _unwrap_agent_code(response)
    try:
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)
        for token in tokens:
            if token.type == tokenize.COMMENT and (not allow_comments):
                raise MiniWoBActionParseError('comments are not allowed')
            if token.type == tokenize.OP and token.string == ';':
                raise MiniWoBActionParseError('semicolons are not allowed')
    except (IndentationError, tokenize.TokenError) as exc:
        raise MiniWoBActionParseError(f'invalid Python token stream: {exc}') from exc
    try:
        tree = ast.parse(code, mode='exec')
    except SyntaxError as exc:
        raise MiniWoBActionParseError(f'invalid Python syntax: {exc.msg}') from exc
    allowed = set(MINIWOB_ALLOWED_AGENT_METHODS)

    def parse_agent_call(statement: ast.stmt) -> Tuple[str, str]:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            raise MiniWoBActionParseError('expected one allowed agent method call')
        call = statement.value
        if not isinstance(call.func, ast.Attribute) or not isinstance(call.func.value, ast.Name) or call.func.value.id != 'agent' or (call.func.attr not in allowed):
            raise MiniWoBActionParseError('unknown or non-agent method')
        if call.keywords:
            raise MiniWoBActionParseError('keyword arguments are not allowed')
        if len(call.args) != 1:
            raise MiniWoBActionParseError('each agent method requires exactly one positional argument')
        argument = call.args[0]
        if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
            raise MiniWoBActionParseError('agent method arguments must be string literals')
        if call.func.attr == 'press' and argument.value not in MINIWOB_ALLOWED_PRESS_KEYS:
            raise MiniWoBActionParseError(f'unsupported press key: {argument.value!r}')
        return (call.func.attr, argument.value)
    actions: List[Tuple[str, str]] = []
    for statement in tree.body:
        if isinstance(statement, ast.Expr):
            actions.append(parse_agent_call(statement))
            continue
        if not allow_bounded_repeat:
            raise MiniWoBActionParseError('expected one allowed agent method call')
        if not isinstance(statement, ast.For):
            raise MiniWoBActionParseError('only direct agent calls and bounded repeat loops are allowed')
        if not isinstance(statement.target, ast.Name) or statement.target.id != MINIWOB_BOUNDED_REPEAT_VARIABLE:
            raise MiniWoBActionParseError("bounded repeat loops must use '_' as the loop variable")
        range_call = statement.iter
        if not isinstance(range_call, ast.Call) or not isinstance(range_call.func, ast.Name) or range_call.func.id != 'range' or range_call.keywords or (len(range_call.args) != 1):
            raise MiniWoBActionParseError('bounded repeat loops require range(<integer literal>)')
        bound_node = range_call.args[0]
        if not isinstance(bound_node, ast.Constant) or isinstance(bound_node.value, bool) or (not isinstance(bound_node.value, int)) or (bound_node.value < 0):
            raise MiniWoBActionParseError('bounded repeat range must be a nonnegative integer literal')
        if bound_node.value > MINIWOB_BOUNDED_REPEAT_MAX:
            raise MiniWoBActionParseError(f'bounded repeat range cannot exceed {MINIWOB_BOUNDED_REPEAT_MAX}')
        if statement.orelse:
            raise MiniWoBActionParseError('bounded repeat loops cannot have an else clause')
        if len(statement.body) != 1:
            raise MiniWoBActionParseError('bounded repeat loops require exactly one body call')
        action = parse_agent_call(statement.body[0])
        actions.extend([action] * bound_node.value)
    if not actions:
        raise MiniWoBActionParseError('actor response contains no calls')
    return actions

def _parse_miniwob_actor_output(response: str, *, require_react_envelope: bool=False) -> List[Tuple[str, str]]:
    action_response = _unwrap_react_step(response) if require_react_envelope else response
    return _parse_agent_actions(action_response, allow_bounded_repeat=require_react_envelope, allow_comments=require_react_envelope)

def _canonical_agent_code(actions: List[Tuple[str, str]]) -> str:
    return '\n'.join((f'agent.{method}({argument!r})' for method, argument in actions))

def _react_decision_history_entry(response: str, actions: List[Tuple[str, str]]) -> str:
    action_response = str(response or '').strip()
    match = re.fullmatch('Thought:\\s*(?P<thought>.+?)\\r?\\nAction:\\s*(?P<action>.+)', action_response, flags=re.DOTALL)
    if match is None or not match.group('thought').strip():
        raise MiniWoBActionParseError('accepted ReAct response has no recoverable Thought')
    return f"Thought: {match.group('thought').strip()}\nAction:\n```\n{_canonical_agent_code(actions)}\n```"
_EPISODE_TIMEOUT = int(os.environ.get('MW_EPISODE_TIMEOUT', '0'))
if _EPISODE_TIMEOUT < 0:
    raise ValueError('MW_EPISODE_TIMEOUT must be non-negative')
_INFRA_RETRIES = int(os.environ.get('MW_INFRA_RETRIES', '1'))

def _resolve_max_environment_steps(value: Optional[int]) -> int:
    if value is None:
        value = int(os.environ.get('MINIWOB_MAX_ENVIRONMENT_STEPS', str(DEFAULT_MAX_ENVIRONMENT_STEPS)))
    value = int(value)
    if value <= 0:
        raise ValueError('max_environment_steps must be positive')
    return value

def _remaining_environment_steps(interaction_ledger: EnvironmentInteractionLedger) -> int:
    remaining = getattr(interaction_ledger, 'remaining_steps', None)
    if remaining is None:
        remaining = interaction_ledger.max_steps_per_episode - interaction_ledger.steps
    return max(0, int(remaining))

def _with_remaining_environment_budget(prompt: str, interaction_ledger: EnvironmentInteractionLedger, *, actor_profile: str='react_batch', decision_index: Optional[int]=None, max_decisions: Optional[int]=None) -> str:
    template = SYNAPSE_LIGHTWEIGHT_REMAINING_ENVIRONMENT_BUDGET_PROMPT_TEMPLATE if actor_profile == 'synapse_lightweight' else REACT_BATCH_REMAINING_ENVIRONMENT_BUDGET_PROMPT_TEMPLATE
    budget_text = template.format(remaining_steps=_remaining_environment_steps(interaction_ledger))
    if actor_profile == 'react_batch':
        if decision_index is None or max_decisions is None:
            raise ValueError('formal actor calls require the model decision budget')
        if not 1 <= int(decision_index) <= int(max_decisions):
            raise ValueError('actor decision index must lie within the trial cap')
    stripped = str(prompt).rstrip()
    if actor_profile == 'react_batch' and stripped.endswith(REACT_BATCH_TURN_SUFFIX):
        stripped = stripped[:-len(REACT_BATCH_TURN_SUFFIX)].rstrip()
        return f'{stripped}\n\n{budget_text}\n{REACT_BATCH_TURN_SUFFIX}'
    if stripped.endswith('Action:'):
        stripped = stripped[:-len('Action:')].rstrip()
        return f'{stripped}\n\n{budget_text}\nAction:'
    return f'{stripped}\n\n{budget_text}'

def _episode_timeout_seconds(*, trials: int) -> int:
    return _EPISODE_TIMEOUT

def _atomic_write_json(path: str | Path, payload: Any, *, indent: int=2) -> None:
    from vrlbench.runtime.metrics import atomic_write_json
    atomic_write_json(path, payload, indent=indent)

class _EpisodeTimeout(Exception):
    pass

def _alarm_handler(signum, frame):
    raise _EpisodeTimeout('timeout')

def make_actor(env: Any, seed: int, model: str='gpt-4o-mini', max_steps: Optional[int]=None, max_environment_steps: Optional[int]=None, trial_seed_mode: str='fixed_instance', llm_fn: Optional[Callable[..., str]]=None, artifact_dir: str | Path | None=None) -> Callable[[], str]:
    """Create a MiniWoB actor: ``() -> trajectory: str``.

    The actor is a plain closure — VEX² injects working memory via
    the LLM layer automatically on every ``generate()`` call while
    the trial context is active.

    Args:
        env:       Synapse MiniWoBEnv instance (shared across episodes).
        seed:      Episode seed (determines the first task instance).
        model:     LLM deployment name.
        max_steps: Compatibility fallback for the per-trial action budget when
                  ``max_environment_steps`` is not supplied.
        max_environment_steps:
                  Hard limit on action opportunities in one trial. A rejected
                  response spends one opportunity without an ``env.step`` call.
                  Every method uses the same model-neutral environment ledger;
                  one Actor response may contain multiple primitive actions.
        trial_seed_mode:
                  ``fixed_instance`` reuses ``seed`` for every trial.
                  ``paired_trial`` deterministically changes the reset seed
                  with the trial index, while remaining matched across methods.
        artifact_dir:
                  Optional screenshot directory for human replay. Images never
                  enter actor prompts or memory. The legacy real-time profile
                  reports unavailable images without adding screenshot latency.
    """
    from vrlbench.runtime import llm as _llm
    llm_usage_instrumented = llm_fn is None
    if llm_fn is None:
        llm_fn = _llm.generate
    assert llm_fn is not None
    if trial_seed_mode not in {'fixed_instance', 'paired_trial'}:
        raise ValueError("trial_seed_mode must be 'fixed_instance' or 'paired_trial'")
    actor_profile = _actor_profile()
    formal_actor = actor_profile == 'react_batch'
    system_instruction = _system_instruction_for_profile(actor_profile)
    action_grammar = _action_grammar_for_profile(actor_profile)
    trial_state: Dict[str, Any] = {'trial_index': 0, 'total_trials': 0, 'prepared': False, 'reset_seed': None, 'task': '', 'initial_observation': '', 'action_clock_started': False, 'format_repairs': 0, 'invalid_actor_responses': 0, 'action_decisions': 0, 'protocol_failure': False, 'protocol_failure_reason': '', 'policy_evidence_eligible': False, 'environment_terminated': False, 'environment_truncated': False, 'trial_close_reason': '', 'policy_evidence': None, 'actor_messages': []}
    interaction_ledger = EnvironmentInteractionLedger(
        max_steps_per_episode=_resolve_max_environment_steps(
            max_environment_steps if max_environment_steps is not None else max_steps),
        count_invalid_actions=True)
    capture_root = Path(artifact_dir).resolve() if artifact_dir is not None else None

    def capture_observation(index: int) -> None:
        if capture_root is None:
            return
        entry: Dict[str, Any] = {'observation_index': index}
        if not formal_actor:
            entry.update(kind='unavailable', reason='capture disabled for the active native real-time clock')
        else:
            try:
                driver = getattr(getattr(env, 'instance', None), 'driver', None)
                capture = getattr(driver, 'get_screenshot_as_png', None)
                if not callable(capture):
                    raise RuntimeError('MiniWoB has no current screenshot interface')
                raw = capture()
                if not isinstance(raw, bytes) or not raw.startswith(b'\x89PNG\r\n\x1a\n'):
                    raise ValueError('MiniWoB screenshot must contain PNG bytes')
                directory = capture_root / f"trial_{trial_state['trial_index']:03d}"
                directory.mkdir(parents=True, exist_ok=True, mode=0o700)
                path = directory / f'observation_{index:04d}.png'
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, 'wb') as stream:
                    stream.write(raw)
                entry.update(kind='image', path=str(path), sha256=hashlib.sha256(raw).hexdigest())
            except _EpisodeTimeout:
                raise
            except Exception as exc:
                entry.update(kind='unavailable', reason='capture failed: ' + type(exc).__name__)
        trial_state['replay_observations'].append(entry)

    def _seed_for_trial(trial_index: int) -> int:
        return miniwob_reset_seed(seed, trial_index, mode=trial_seed_mode)

    def set_trial_context(*, trial_index: int, total_trials: int) -> Dict[str, Any]:
        trial_state['trial_index'] = int(trial_index)
        trial_state['total_trials'] = int(total_trials)
        trial_state['prepared'] = False
        trial_state['reset_seed'] = None
        trial_state['task'] = ''
        trial_state['initial_observation'] = ''
        trial_state['action_clock_started'] = False
        reset_seed = _seed_for_trial(int(trial_index))
        return {'reset_seed': reset_seed, 'base_seed': seed, 'trial_seed_mode': trial_seed_mode, 'total_trials': int(total_trials)}

    def _call_actor_model(prompt: str, *, phase: str, decision_index: int, max_decisions: int, max_tokens: Optional[int]=None) -> str:
        interaction_ledger.charge_actor_turn()
        prompt_with_budget = _with_remaining_environment_budget(prompt, interaction_ledger, actor_profile=actor_profile, decision_index=decision_index, max_decisions=max_decisions)
        response = str(llm_fn(prompt_with_budget, model=model, retry_on_empty=False, max_tokens=int(os.environ.get('MINIWOB_ACTOR_MAX_TOKENS', '512')) if max_tokens is None else int(max_tokens)))
        messages = trial_state.get('actor_messages')
        if isinstance(messages, list):
            messages.append({'role': 'assistant', 'phase': phase, 'content': response})
        return response

    def prepare_trial() -> Dict[str, str]:
        """Perform the trial's single public reset and cache its initial state."""
        if trial_state['prepared']:
            return {'task': str(trial_state['task']), 'initial_observation': str(trial_state['initial_observation'])}
        reset_seed = _seed_for_trial(trial_state['trial_index'])
        interaction_ledger.begin_episode(trial_state['trial_index'])
        trial_state['format_repairs'] = 0
        trial_state['invalid_actor_responses'] = 0
        trial_state['invalid_action_events'] = []
        trial_state['action_decisions'] = 0
        trial_state['protocol_failure'] = False
        trial_state['protocol_failure_reason'] = ''
        trial_state['policy_evidence_eligible'] = False
        trial_state['environment_terminated'] = False
        trial_state['environment_truncated'] = False
        trial_state['trial_close_reason'] = ''
        trial_state['policy_evidence'] = None
        trial_state['actor_messages'] = []
        trial_state['replay_observations'] = []
        interaction_ledger.charge_reset()
        obs = env.reset(seed=reset_seed)
        env.instance.driver.execute_script('\n            if (core.EP_TIMER !== null) { clearTimeout(core.EP_TIMER); }\n            core.EP_TIMER = setTimeout(function () {}, 2147483647);\n            core.clearTimer();\n            ')
        task = _task_from_observation_html(obs, env.get_task())
        trial_state['reset_seed'] = reset_seed
        trial_state['task'] = task
        trial_state['initial_observation'] = str(obs or '')
        capture_observation(0)
        trial_state['prepared'] = True
        return {'task': task, 'initial_observation': str(obs or '')}

    def _start_action_clock() -> None:
        if trial_state['action_clock_started']:
            return
        if not formal_actor:
            env.instance.driver.execute_script("\n                core.ept0 = new Date().getTime();\n                core.countdownTimer(core.EPISODE_MAX_TIME);\n                if (core.EP_TIMER !== null) { clearTimeout(core.EP_TIMER); }\n                core.EP_TIMER = setTimeout(function () {\n                  core.endEpisode(-1, false, 'timed out');\n                }, core.EPISODE_MAX_TIME);\n                ")
        env.instance.start_time = time.time()
        trial_state['action_clock_started'] = True

    def actor() -> str:
        prepared = prepare_trial()
        _start_action_clock()
        task = prepared['task']
        obs = prepared['initial_observation']
        steps = interaction_ledger.max_steps_per_episode
        init_ob = f'Task: {task}\nObservation:\n{obs}'
        trial_state['actor_messages'].append({'role': 'user', 'phase': 'initial', 'content': init_ob})
        init_prompt = system_instruction + f'\nHere is the task:\n{init_ob}'
        decision_history: List[str] = []

        def decision_prompt(current_observation: str) -> str:
            sections = [system_instruction]
            sections.append(f'Here is the task:\nTask: {task}\nObservation:\n{current_observation}')
            if decision_history:
                sections.append('Previous Thought/Action decisions and parser feedback, oldest first:\n' + '\n\n'.join(decision_history))
            return '\n\n'.join(sections)
        prompt = decision_prompt(obs) if formal_actor else init_prompt + '\n'
        trajectory: list = [f'Task: {task}', f'Observation:\n{obs}', action_grammar]
        trial_state['policy_evidence'] = {'schema_version': 1, 'task': task, 'initial_observation': str(obs or ''), 'trial_close_reason': 'running', 'steps': []}
        agent = _AgentActions(env, trial_index=trial_state['trial_index'], interaction_ledger=interaction_ledger,
                              after_step=capture_observation if capture_root is not None else None)
        action_decisions = 0

        def close_trial(reason: str, *, environment_terminated: bool=False, environment_truncated: bool=False) -> None:
            if environment_terminated and environment_truncated:
                raise ValueError('a trial cannot be both terminated and truncated')
            trial_state['trial_close_reason'] = reason
            policy_evidence = trial_state.get('policy_evidence')
            if isinstance(policy_evidence, dict):
                policy_evidence['trial_close_reason'] = reason
            trial_state['environment_terminated'] = environment_terminated
            trial_state['environment_truncated'] = environment_truncated
            executed_steps = policy_evidence.get('steps', []) if isinstance(policy_evidence, dict) else []
            trial_state['policy_evidence_eligible'] = bool(executed_steps) or interaction_ledger.invalid_action_attempts > 0 or reason == 'environment_success'

        while interaction_ledger.remaining_steps > 0:
            turn_suffix = REACT_BATCH_TURN_SUFFIX if formal_actor else 'Action:'
            raw = _call_actor_model(prompt + turn_suffix, phase='action', decision_index=action_decisions + 1, max_decisions=steps)
            accepted_raw = raw
            action_decisions += 1
            trial_state['action_decisions'] = action_decisions
            try:
                actions = _parse_miniwob_actor_output(raw, require_react_envelope=formal_actor)
            except MiniWoBActionParseError as initial_error:
                trial_state['invalid_actor_responses'] += 1
                before = interaction_ledger.action_attempts
                interaction_ledger.charge_invalid_action()
                feedback = (f'Invalid action response: {initial_error}. No environment action was executed; '
                            f'the observation is unchanged. One action opportunity was consumed; '
                            f'{interaction_ledger.remaining_steps} remain. Return a valid action using the Agent API.')
                trial_state['invalid_action_events'].append({
                    'after_environment_step': interaction_ledger.steps,
                    'raw_response': raw, 'reason': str(initial_error),
                    'budget_used_before': before,
                    'budget_used_after': interaction_ledger.action_attempts})
                trajectory.append(feedback)
                trial_state['actor_messages'].append({'role': 'user', 'phase': 'parser_feedback', 'content': feedback})
                decision_history.append(f'Rejected response:\n{raw}\nParser feedback: {feedback}')
                prompt = decision_prompt(obs) if formal_actor else init_prompt + '\n' + '\n'.join(trajectory)
                continue
            code = _canonical_agent_code(actions)
            trajectory.append(f'Proposed code:\n{code}')
            budget_error: Optional[EnvironmentBudgetExceeded] = None
            try:
                for method, argument in actions:
                    getattr(agent, method)(argument)
            except EnvironmentBudgetExceeded as e:
                budget_error = e
            executed_steps = list(agent.step_log)
            agent.step_log.clear()
            for action_repr, step_obs in executed_steps:
                policy_evidence = trial_state.get('policy_evidence')
                if not isinstance(policy_evidence, dict):
                    raise RuntimeError('MiniWoB policy evidence was not initialized')
                policy_evidence['steps'].append({'action': action_repr, 'observation': str(step_obs or '')})
                trajectory.append(f'> {action_repr}')
                if step_obs:
                    trajectory.append(f'Observation:\n{step_obs}')
            if executed_steps:
                trial_state['actor_messages'].append({'role': 'user', 'phase': 'environment_feedback', 'content': f"Observation after the accepted action block:\n{str(executed_steps[-1][1] or '')}"})
            if formal_actor:
                decision_history.append(_react_decision_history_entry(accepted_raw, actions))
            if agent.done:
                if not executed_steps and agent.last_obs:
                    trajectory.append(f'Observation:\n{agent.last_obs}')
                close_trial('environment_success' if agent.reward > 0 else 'environment_terminal_failure', environment_terminated=True)
                trajectory.append('SUCCESS' if agent.reward > 0 else 'FAIL')
                return '\n'.join(trajectory)
            if budget_error is not None:
                close_trial('ao_horizon', environment_truncated=True)
                trajectory.append(f'Environment budget exhausted: {budget_error}')
                trajectory.append('FAIL (action budget reached)')
                return '\n'.join(trajectory)
            new_obs = executed_steps[-1][1] if executed_steps else agent.last_obs
            if new_obs:
                obs = str(new_obs)
                prompt = decision_prompt(obs) if formal_actor else init_prompt + '\n' + '\n'.join(trajectory)
        close_trial('ao_horizon', environment_truncated=True)
        trajectory.append('FAIL (action budget reached)')
        return '\n'.join(trajectory)

    def get_trial_metadata() -> Dict[str, Any]:
        metadata = {**interaction_ledger.snapshot(), 'task': str(trial_state.get('task') or ''), 'total_trials': int(trial_state.get('total_trials') or 0), 'llm_usage_instrumented': llm_usage_instrumented, 'action_decisions': int(trial_state.get('action_decisions') or 0), 'format_repairs': int(trial_state.get('format_repairs') or 0), 'invalid_actor_responses': int(trial_state.get('invalid_actor_responses') or 0), 'protocol_failure': bool(trial_state.get('protocol_failure', False)), 'protocol_failure_reason': str(trial_state.get('protocol_failure_reason') or ''), 'environment_terminated': bool(trial_state.get('environment_terminated', False)), 'environment_truncated': bool(trial_state.get('environment_truncated', False)), 'trial_close_reason': str(trial_state.get('trial_close_reason') or ''), 'environment_action_clock': 'ao_horizon_only_native_timeout_suspended_episode_active' if formal_actor else 'actor_start_to_terminal', 'policy_evidence_eligible': bool(trial_state.get('policy_evidence_eligible', False)), 'actor_profile': actor_profile}
        metadata['invalid_action_policy'] = 'consume_one_action_and_continue'
        metadata['invalid_action_events'] = deepcopy(trial_state.get('invalid_action_events', []))
        if capture_root is not None:
            metadata['replay_observations'] = deepcopy(trial_state.get('replay_observations', []))
        return metadata

    def get_policy_evidence() -> Dict[str, Any]:
        evidence = trial_state.get('policy_evidence')
        if not isinstance(evidence, dict):
            raise RuntimeError('MiniWoB actor has no completed policy evidence')
        return deepcopy(evidence)

    def get_actor_trajectory() -> str:
        messages = trial_state.get('actor_messages')
        if not isinstance(messages, list):
            raise RuntimeError('MiniWoB actor trajectory was not initialized')
        blocks = []
        for message in messages:
            role = str(message.get('role') or 'unknown').upper()
            phase = str(message.get('phase') or '').strip()
            label = f"[{role}{('/' + phase if phase else '')}]"
            blocks.append(f"{label}\n{message.get('content', '')}")
        return '\n\n'.join(blocks)
    actor.set_trial_context = set_trial_context
    actor.prepare_trial = prepare_trial
    actor.get_trial_metadata = get_trial_metadata
    actor.get_policy_evidence = get_policy_evidence
    actor.get_actor_trajectory = get_actor_trajectory
    actor.trial_seed_mode = trial_seed_mode
    actor.actor_profile = actor_profile
    return actor

def make_manager() -> callable:
    """Create a trajectory-based success judge.

    Convention: the actor appends ``"SUCCESS"`` or ``"FAIL"`` as the last
    meaningful line of the trajectory.  The manager checks this.
    """

    def manager(trajectory: str) -> bool:
        for line in reversed(trajectory.strip().splitlines()):
            stripped = line.strip()
            if stripped == 'SUCCESS':
                return True
            if stripped in ('FAIL', 'FAIL (max steps reached)', 'FAIL (environment step budget reached)', 'FAIL (action budget reached)', 'FAIL (actor decision cap reached)', 'FAIL (actor protocol violation)', '[Repeated action, stopping]'):
                return False
        return False
    return manager

def _normalize_miniwob_controller(value: object) -> str:
    controller = str(value or 'none').strip().lower().replace('-', '_')
    if controller not in {'none', 'dsst'}:
        raise ValueError(f'controller must be one of none or dsst; got {value!r}')
    return controller

def run(tasks: Optional[List[str]]=None, episodes_per_task: int=10, model: str='gpt-4o-mini', trials: int=6, er: str='none', controller: str='none', allow_gold_feedback: bool=False, output_dir: Optional[str]=None, headless: bool=True, shared_t1_dir: Optional[str]=None, episode_indices: Optional[List[int]]=None, trial_seed_mode: str='fixed_instance', max_environment_steps: Optional[int]=None, max_total_tokens: Optional[int]=None, require_provider_usage: bool=False, capture_observations: bool=False) -> None:
    """Run the MiniWoB experiment using VEX².

    For each task x episode, calls ``VEX2.run()`` with a fresh actor and
    manager. The benchmark owns task I/O, reset pairing, A-O accounting, and
    result collection; the selected controller owns cross-trial information.

    Output: ``{output_dir}/results.json`` with per-task VRL Task SR.
           ``{output_dir}/env_results_trial_{t}.json`` for analysis scripts.
    """
    from vrlbench.runtime import VEX2
    from vrlbench.runtime.controllers import make_controller
    from vrlbench.runtime.er import make_er_slot
    from vrlbench.runtime.metrics import completion_metrics, write_env_results, resource_usage_summary
    from vrlbench.runtime.loop import EpisodeResult, prepare_shared_t1, run_episode
    if type(capture_observations) is not bool:
        raise ValueError('capture_observations must be a boolean')
    controller = _normalize_miniwob_controller(controller)
    actor_profile = _actor_profile()
    retry_source_method = er == 'none' and controller == 'none'
    if require_provider_usage:
        if actor_profile != 'react_batch':
            raise ValueError('formal MiniWoB comparison requires MINIWOB_ACTOR_PROFILE=react_batch')
        if retry_source_method and shared_t1_dir:
            raise ValueError('formal MiniWoB retry source cannot consume shared T1')
        if not retry_source_method and (not shared_t1_dir):
            raise ValueError('formal MiniWoB comparison methods must consume the exact retry T1')
    if output_dir is None:
        output_dir = str(Path.cwd() / 'runs' / 'miniwob')
    task_list = tasks or MINIWOB_TASKS
    planned_episode_ids = list(episode_indices) if episode_indices is not None else list(range(episodes_per_task))
    os.makedirs(output_dir, exist_ok=True)
    if trial_seed_mode not in {'fixed_instance', 'paired_trial'}:
        raise ValueError("trial_seed_mode must be 'fixed_instance' or 'paired_trial'")
    resolved_max_environment_steps = _resolve_max_environment_steps(max_environment_steps)
    vex = VEX2(er=er, controller=controller, allow_gold_feedback=allow_gold_feedback, model=model, trials=trials, max_total_tokens=max_total_tokens, require_provider_usage=require_provider_usage)
    logger.info('VEX² config: %s', vex)
    task_sr: Dict[str, float] = {}
    all_episodes: List[EpisodeResult] = []
    all_names: List[str] = []
    evidence_rows: List[Dict[str, Any]] = []
    infra_failures: List[Dict[str, Any]] = []
    shared_inputs = set()

    def _write_checkpoint(*, current_task: Optional[str]=None, current_solved: int=0, current_total: int=0, final: bool=False) -> None:
        partial_sr = dict(task_sr)
        if current_task is not None and current_total:
            partial_sr[current_task] = current_solved / current_total
        macro_sr = sum(partial_sr.values()) / len(partial_sr) if partial_sr else 0.0
        partial_completion = completion_metrics(all_episodes, total_trials=trials)
        solved_episodes = sum((1 for episode in all_episodes if episode.solved))
        summary = {'benchmark': 'miniwob', 'vrl_task_sr': macro_sr, 'sr_at_t': partial_completion['sr_at_t'], 'avg_t_at_t': partial_completion['avg_t_at_t'], 'per_task': partial_sr, 'num_solved': solved_episodes, 'num_total': len(all_episodes), 'n': len(all_episodes), 'n_planned': len(task_list) * len(planned_episode_ids), 'trials': trials, 'num_trials': trials, 'model': model, 'er': er, 'controller': controller, 'vex_method_layer': controller if controller != 'none' else None, 'config': repr(vex), 'controller_input': 'canonical_policy_ao_forest' if controller == 'dsst' else None, 'allow_gold_feedback': allow_gold_feedback, 'trial_seed_mode': trial_seed_mode, 'max_environment_steps': resolved_max_environment_steps, 'max_total_tokens': max_total_tokens, 'require_provider_usage': bool(require_provider_usage), 'resource_usage': resource_usage_summary(all_episodes), 'resource_accounting_basis': 'logical_per_condition_including_shared_t1', 'actor_profile': actor_profile, 'checkpoint': not final, 'completed_episodes': len(evidence_rows), 'infra_failures': list(infra_failures)}
        write_env_results(all_episodes, output_dir, total_trials=trials, names=all_names)
        _atomic_write_json(os.path.join(output_dir, 'episodes.json'), evidence_rows)
        _atomic_write_json(os.path.join(output_dir, 'results.json'), summary)
    for task_name in task_list:
        shared_rows = None
        if shared_t1_dir:
            task_t1_dir = _resolve_shared_t1_task_dir(shared_t1_dir, task_name, model, trials)
            if task_t1_dir is None:
                raise FileNotFoundError(f'{task_name}: no shared T1 found under {shared_t1_dir}')
            else:
                source_key = str(task_t1_dir.resolve())
                if source_key not in shared_inputs:
                    validate_shared_t1_source(task_t1_dir, benchmark='miniwob', model=model, max_environment_steps=resolved_max_environment_steps, require_complete_accounting=require_provider_usage, expected_provider=os.environ.get('MODEL_PROVIDER') if require_provider_usage else None, expected_base_url=os.environ.get('OPENAI_BASE_URL', '') if require_provider_usage else None, expected_trial_seed_mode=trial_seed_mode)
                    shared_inputs.add(source_key)
                env_only_t1_allowed = er == 'none' and controller == 'none'
                shared_rows = _load_shared_t1(task_t1_dir, task_name=task_name, require_failure_trajectory=not env_only_t1_allowed, require_complete_accounting=require_provider_usage, expected_max_environment_steps=resolved_max_environment_steps, expected_trial_seed_mode=trial_seed_mode)
        MiniWoBEnv = _load_mw_stack()[0]
        env = MiniWoBEnv(subdomain=task_name, headless=headless)
        dsst_basic_environment_rules = miniwob_dsst_basic_environment_rules(task_name, profile=actor_profile)
        episode_ids = planned_episode_ids
        solved_count = 0
        for episode_offset, ep in enumerate(episode_ids, start=1):
            seed = ep
            sample_name = f'{task_name}_ep{ep}'
            evidence_row: Dict[str, Any] = {'task': task_name, 'episode': ep}
            manager = make_manager()
            result = None
            last_error = ''
            for attempt in range(_INFRA_RETRIES + 1):
                actor = make_actor(env, seed=seed, model=model, max_environment_steps=resolved_max_environment_steps, trial_seed_mode=trial_seed_mode,
                                   artifact_dir=Path(output_dir) / 'observations' / sample_name / f'attempt_{attempt:03d}' if capture_observations else None)
                prev = signal.signal(signal.SIGALRM, _alarm_handler)
                signal.alarm(_episode_timeout_seconds(trials=trials))
                try:
                    t1 = (shared_rows or {}).get(seed)
                    if t1 is None:
                        if shared_t1_dir:
                            raise KeyError(f'{task_name} ep={ep}: shared T1 missing for requested episode')
                        result = vex.run(actor=actor, manager=manager, metadata={'benchmark': 'miniwob', 'task': task_name, 'task_scope': f'MiniWoB task family: {task_name}', 'basic_environment_rules': dsst_basic_environment_rules, 'episode': ep, 'controller': controller})
                    else:
                        t1_success = bool(t1['success'])
                        t1_traj = str(t1['trajectory'] or ('SHARED_T1_SUCCESS' if t1_success else 'SHARED_T1_FAIL'))
                        if t1_traj == 'SHARED_T1_FAIL':
                            if er != 'none' or controller != 'none':
                                raise ValueError(f'{task_name} ep={ep}: shared T1 failure lacks trajectory; refusing to update memory from placeholder evidence.')
                        er_slot = make_er_slot(er)
                        if controller == 'dsst':
                            controller_slot = make_controller(controller, allow_gold_feedback=False, model=model)
                        else:
                            controller_slot = make_controller(controller, allow_gold_feedback=allow_gold_feedback, model=model)
                        er_slot.reset()
                        if controller_slot:
                            controller_slot.reset()
                        initial_result, token_ledger = prepare_shared_t1(trajectory=t1_traj, policy_trajectory=str(t1.get('policy_trajectory') or ''), actor_trajectory=str(t1.get('actor_trajectory') or ''), success=t1_success, er_slot=er_slot, model=model, total_trials=trials, controller=controller_slot, metadata={'benchmark': 'miniwob', 'task': task_name, 'task_scope': f'MiniWoB task family: {task_name}', 'basic_environment_rules': dsst_basic_environment_rules, 'episode': ep, 'controller': controller}, trial_metadata=t1.get('trial_metadata'), llm_usage=t1.get('llm_usage'), max_total_tokens=max_total_tokens, require_provider_usage=require_provider_usage)
                        if t1_success:
                            result = initial_result
                        else:
                            result = run_episode(actor=actor, manager=manager, er_slot=er_slot, model=model, total_trials=trials, controller=controller_slot, metadata={'benchmark': 'miniwob', 'task': task_name, 'task_scope': f'MiniWoB task family: {task_name}', 'basic_environment_rules': dsst_basic_environment_rules, 'episode': ep, 'controller': controller}, initial_result=initial_result, reset_state=False, token_ledger=token_ledger)
                        evidence_row['shared_t1_trajectory_present'] = bool(t1_traj and t1_traj != 'SHARED_T1_FAIL')
                        guidance_entries = list(getattr(controller_slot, '_constraints', [])) if controller_slot else []
                        controller_state = controller_slot.snapshot() if controller_slot else None
                        evidence_row['controller'] = controller
                        evidence_row['controller_state'] = controller_state
                        evidence_row['spatial_guidance_entries'] = guidance_entries
                        evidence_row['spatial_deadends'] = guidance_entries
                        evidence_row['spatial_state'] = controller_state
                    last_error = ''
                    break
                except _EpisodeTimeout as e:
                    last_error = str(e)
                    logger.error('%s ep=%d: timeout', task_name, ep)
                except (ValueError, KeyError) as e:
                    logger.error('%s ep=%d: evaluation contract violation: %s', task_name, ep, e)
                    raise
                except Exception as e:
                    last_error = str(e)
                    logger.error('%s ep=%d: %s', task_name, ep, e)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, prev)
                if last_error and require_provider_usage:
                    break
                if attempt < _INFRA_RETRIES:
                    evidence_row.setdefault('infra_retries', []).append({'attempt': attempt + 1, 'error': last_error})
                    try:
                        env.close()
                    except Exception:
                        pass
                    env = MiniWoBEnv(subdomain=task_name, headless=headless)
                    logger.info('%s ep=%d: retrying after MiniWoB infrastructure error (%d/%d)', task_name, ep, attempt + 1, _INFRA_RETRIES)
            if result is None:
                infra_failure = {'task': task_name, 'episode': ep, 'error': last_error, 'failure_type': 'infrastructure_or_provider_failure'}
                infra_failures.append(infra_failure)
                _write_checkpoint(current_task=task_name, current_solved=solved_count, current_total=episode_offset - 1)
                try:
                    env.close()
                except Exception:
                    pass
                raise RuntimeError(f'MiniWoB execution stopped before the current episode could be committed: {infra_failure}')
            evidence_row['infra_error'] = False
            all_episodes.append(result)
            all_names.append(sample_name)
            evidence_row['name'] = sample_name
            evidence_row['solved'] = result.solved
            evidence_row['solve_trial'] = result.solve_trial
            evidence_row['resource_usage'] = result.resource_usage
            evidence_row['trials'] = [{'trial': tr.trial, 'success': tr.success, 'trial_metadata': tr.trial_metadata, 'llm_usage': tr.llm_usage, 'context': tr.context, 'system_context': tr.system_context, 'user_context': tr.user_context, 'controller_decision': tr.controller_decision, 'controller_state_pre': tr.controller_state_pre, 'controller_state': tr.controller_state, 'temporal_decision': tr.temporal_decision, 'spatial_state_pre': tr.spatial_state_pre, 'spatial_state': tr.spatial_state, 'trajectory': tr.trajectory, 'policy_trajectory': tr.policy_trajectory, 'actor_trajectory': tr.actor_trajectory} for tr in result.trials]
            evidence_rows.append(evidence_row)
            if result.solved:
                solved_count += 1
                logger.info('%s ep=%d: SOLVED (trial %d)', task_name, ep, result.solve_trial)
            else:
                logger.info('%s ep=%d: FAIL', task_name, ep)
            _write_checkpoint(current_task=task_name, current_solved=solved_count, current_total=episode_offset)
        denominator = len(episode_ids)
        sr = solved_count / denominator if denominator else 0.0
        task_sr[task_name] = sr
        logger.info('task=%-30s  vrl_sr=%.3f', task_name, sr)
        try:
            env.close()
        except Exception:
            pass
    macro_sr = sum(task_sr.values()) / len(task_sr) if task_sr else 0.0
    logger.info('MiniWoB macro VRL Task SR = %.3f  (%d tasks)', macro_sr, len(task_sr))
    _write_checkpoint(final=True)
