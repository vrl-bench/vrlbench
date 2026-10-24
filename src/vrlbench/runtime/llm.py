import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator, List, Mapping, Optional
from openai import OpenAI
from vrlbench.runtime.budget import TokenBudgetExceeded, TokenUsageLedger, provider_token_count, provider_usage_total
logger = logging.getLogger(__name__)
_client: Optional[OpenAI] = None

class ProviderUsageMissing(RuntimeError):
    """Raised when a configured response has no complete, valid token total."""

class ProviderIdentityMismatch(RuntimeError):
    """Raised when a provider response violates the frozen model identity."""

class ProviderCallUncertain(RuntimeError):
    """Raised when a strict provider call fails with an uncertain charge state."""

def _api_key_for_base_url(base_url: Optional[str]) -> Optional[str]:
    """Return the API key that matches the configured OpenAI-compatible endpoint."""
    if base_url and 'bigmodel.cn' in base_url and os.environ.get('BIGMODEL_API_KEY'):
        return os.environ.get('BIGMODEL_API_KEY')
    return os.environ.get('OPENAI_API_KEY')

def get_client() -> OpenAI:
    global _client
    if _client is None:
        timeout = os.environ.get('VEX2_LLM_TIMEOUT')
        timeout_kwargs = {'timeout': float(timeout)} if timeout else {}
        base_url = os.environ.get('OPENAI_BASE_URL')
        _client = OpenAI(api_key=_api_key_for_base_url(base_url), base_url=base_url, max_retries=0, **timeout_kwargs)
    return _client
_ctx = threading.local()
_user_ctx = threading.local()
_token_budget_ctx = threading.local()
_provider_call_spacing_lock = threading.Lock()
_provider_last_call_started_monotonic = 0.0

def set_context(context: str) -> None:
    """Set working memory context for the current trial."""
    _ctx.value = context

def _get_context() -> str:
    """Return the current context (persistent within a trial)."""
    return getattr(_ctx, 'value', '')

def _respect_provider_call_spacing() -> None:
    """Pace provider calls without retrying or changing request semantics."""
    raw_interval = os.environ.get('VEX2_LLM_MIN_INTERVAL_SECONDS', '').strip()
    if not raw_interval:
        return
    interval = float(raw_interval)
    if interval <= 0:
        return
    global _provider_last_call_started_monotonic
    with _provider_call_spacing_lock:
        now = time.monotonic()
        wait_seconds = interval - (now - _provider_last_call_started_monotonic)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        _provider_last_call_started_monotonic = time.monotonic()

def get_context() -> str:
    """Public read-only access to the active working-memory context."""
    return _get_context()

def set_user_context(context: str) -> None:
    """Set source-method memory prepended to actor user prompts this trial."""
    _user_ctx.value = context

def _get_user_context() -> str:
    return getattr(_user_ctx, 'value', '')

def get_user_context() -> str:
    """Public read-only access to active user-prompt memory."""
    return _get_user_context()

def _actor_user_prompt(prompt: str) -> str:
    context = _get_user_context()
    return f'{context}\n\n{prompt}' if context else prompt

def _get_token_ledger() -> TokenUsageLedger | None:
    value = getattr(_token_budget_ctx, 'value', None)
    return value if isinstance(value, TokenUsageLedger) else None

@contextmanager
def token_budget_scope(ledger: TokenUsageLedger | None) -> Iterator[None]:
    """Account all nested LLM calls against one task-instance ledger."""
    previous = getattr(_token_budget_ctx, 'value', None)
    _token_budget_ctx.value = ledger
    try:
        yield
    finally:
        if previous is None:
            if hasattr(_token_budget_ctx, 'value'):
                delattr(_token_budget_ctx, 'value')
        else:
            _token_budget_ctx.value = previous

def _uses_bigmodel_thinking_control(model: str) -> bool:
    return model.startswith(('glm-5', 'glm-4.7'))

def _is_reasoning_model(model: str) -> bool:
    return model.startswith('gpt-5.4') or _uses_bigmodel_thinking_control(model)

def _token_kwarg(model: str, max_tokens: int) -> dict:
    if model.startswith('gpt-5.4'):
        return {'max_completion_tokens': max_tokens}
    return {'max_tokens': max_tokens}

def _provider_extra_body(model: str) -> Optional[dict]:
    if _uses_bigmodel_thinking_control(model) or model == 'DeepSeek-V4-Flash':
        return {'thinking': {'type': 'disabled'}}
    return None

def _fixed_reasoning_effort(model: str) -> str | None:
    return 'none' if model.startswith('gpt-5.4') else None

def _reasoning_controls_sampling(model: str, extra_body: Optional[Mapping[str, Any]], request_reasoning_effort: str | None=None) -> bool:
    """Whether the active reasoning mode restricts ordinary sampling kwargs."""
    if str(request_reasoning_effort or '').strip().lower() == 'none':
        return False
    if _uses_bigmodel_thinking_control(model):
        thinking = extra_body.get('thinking') if isinstance(extra_body, Mapping) else None
        if isinstance(thinking, Mapping) and thinking.get('type') == 'disabled':
            return False
    return _is_reasoning_model(model)

def _supports_temperature(model: str) -> bool:
    return not model.startswith('gpt-5.4')

def _supports_seed(model: str) -> bool:
    return True

def _provider_stop(model: str, stop: str | list[str] | None) -> str | list[str] | None:
    if stop is None or model.startswith('gpt-5.4'):
        return None
    return stop

def _apply_local_stop(text: str, stop: str | list[str] | None) -> str:
    """Apply an exact stop-token truncation when the provider cannot do it."""
    if not stop or not text:
        return text
    stops = [stop] if isinstance(stop, str) else list(stop)
    best: int | None = None
    for marker in stops:
        if not marker:
            continue
        index = text.find(marker)
        if index >= 0 and (best is None or index < best):
            best = index
    return text[:best].strip() if best is not None else text

def _provider_for_base_url(base_url: str) -> str:
    lowered = str(base_url or '').lower()
    if 'bigmodel.cn' in lowered:
        return 'bigmodel'
    if 'deepseek' in lowered:
        return 'deepseek'
    if not lowered:
        return 'openai'
    return 'openai_compatible'

def _field(value: object, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)

def _int_field(value: object, name: str) -> int | None:
    return provider_token_count(_field(value, name))

def _usage_payload(response: object) -> dict[str, Any]:
    usage = getattr(response, 'usage', None)
    prompt_details = _field(usage, 'prompt_tokens_details') if usage else None
    completion_details = _field(usage, 'completion_tokens_details') if usage else None
    # Preserve missing and invalid core fields until the shared validator sees them.
    return {'prompt_tokens': _field(usage, 'prompt_tokens'), 'completion_tokens': _field(usage, 'completion_tokens'), 'total_tokens': _field(usage, 'total_tokens'), 'cached_prompt_tokens': _int_field(prompt_details, 'cached_tokens') if prompt_details else None, 'reasoning_tokens': _int_field(completion_details, 'reasoning_tokens') if completion_details else None}

def _response_choices(response: object) -> list[Any]:
    choices = _field(response, 'choices')
    if choices is None:
        return []
    try:
        return list(choices)
    except TypeError:
        return []

def _validate_provider_identity(response: object) -> None:
    """Fail immediately when a paper run reaches an unexpected backend."""
    raw_allowed = os.environ.get('VEX2_ALLOWED_RESPONSE_MODELS')
    raw_allowed_prefixes = os.environ.get('VEX2_ALLOWED_RESPONSE_MODEL_PREFIXES')
    if raw_allowed is not None:
        allowed = {value.strip() for value in raw_allowed.split(',') if value.strip()}
        allowed_prefixes = {value.strip() for value in (raw_allowed_prefixes or '').split(',') if value.strip()}
        if not allowed:
            raise ProviderIdentityMismatch('VEX2_ALLOWED_RESPONSE_MODELS does not name any model')
        observed = str(getattr(response, 'model', '') or '').strip()
        if observed not in allowed and (not any((observed.startswith(prefix) for prefix in allowed_prefixes))):
            raise ProviderIdentityMismatch(f'provider response model is outside the configured interface: observed={observed!r}, allowed={sorted(allowed)!r}, allowed_prefixes={sorted(allowed_prefixes)!r}')
    require_fingerprint = os.environ.get('VEX2_REQUIRE_SYSTEM_FINGERPRINT', '').strip().lower()
    if require_fingerprint in {'1', 'true', 'yes', 'on'}:
        fingerprint = str(getattr(response, 'system_fingerprint', '') or '').strip()
        if not fingerprint:
            raise ProviderIdentityMismatch('provider response omitted the required system_fingerprint')

def _estimate_message_tokens(messages: list[dict[str, str]], model: str) -> int:
    """Conservatively estimate chat input tokens for strict call admission."""
    try:
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding('o200k_base')
        count = 3
        for message in messages:
            count += 3
            count += len(encoding.encode(str(message.get('role', ''))))
            count += len(encoding.encode(str(message.get('content', ''))))
    except Exception:
        count = 3 + sum((4 + max((len(str(message.get('content', ''))) + 2) // 3, 1) for message in messages))
    margin = max(int(os.environ.get('VEX2_TOKEN_ESTIMATE_MARGIN', '16')), 0)
    return count + margin

def _admit_llm_call(messages: list[dict[str, str]], *, model: str, requested_max_tokens: int | None) -> int | None:
    ledger = _get_token_ledger()
    if ledger is None:
        return int(requested_max_tokens) if requested_max_tokens is not None else None
    return ledger.admit_call(estimated_prompt_tokens=_estimate_message_tokens(messages, model), requested_completion_tokens=requested_max_tokens)

def _record_resource_usage(response: object) -> None:
    usage = _usage_payload(response)
    ledger = _get_token_ledger()
    missing_required_usage = False
    if ledger is not None:
        ledger.record_response(usage)
        missing_required_usage = bool(ledger.require_provider_usage and provider_usage_total(usage) is None)
    if missing_required_usage:
        raise ProviderUsageMissing('provider response has incomplete, invalid, or contradictory token usage required by this run')
    _validate_provider_identity(response)

def _record_resource_failure() -> None:
    ledger = _get_token_ledger()
    if ledger is not None:
        ledger.record_failure()

def complete(prompt: str, model: str, system: Optional[str]=None, temperature: float=0.0, max_tokens: int | None=1024, seed: int=42, stop: str | list[str] | None=None) -> str:
    """LLM call (simple retry, no temperature escalation).

    Auto-injects working memory context into system prompt if active.
    """
    prompt = _actor_user_prompt(prompt)
    ctx = _get_context()
    if ctx:
        system = system + '\n\n' + ctx if system else ctx
    c = get_client()
    messages: list = []
    if system:
        messages.append({'role': 'system', 'content': system})
    messages.append({'role': 'user', 'content': prompt})
    extra_body = _provider_extra_body(model)
    reasoning_effort = _fixed_reasoning_effort(model)
    reasoning = _reasoning_controls_sampling(model, extra_body, reasoning_effort)
    provider_stop = _provider_stop(model, stop)
    local_stop = stop if stop is not None and provider_stop is None else None
    last_error: Exception | None = None
    for attempt in range(6):
        admitted_max_tokens = _admit_llm_call(messages, model=model, requested_max_tokens=max_tokens)
        _respect_provider_call_spacing()
        response_committed = False
        try:
            api_kwargs = dict(model=model, messages=messages)
            if _supports_seed(model):
                api_kwargs['seed'] = seed
            if admitted_max_tokens is not None:
                api_kwargs.update(_token_kwarg(model, admitted_max_tokens))
            if reasoning_effort is not None:
                api_kwargs['reasoning_effort'] = reasoning_effort
            if not reasoning and _supports_temperature(model):
                api_kwargs['temperature'] = temperature
            if provider_stop is not None:
                api_kwargs['stop'] = provider_stop
            if extra_body is not None:
                api_kwargs['extra_body'] = extra_body
            resp = c.chat.completions.create(**api_kwargs)
            response_committed = True
            _record_resource_usage(resp)
            text = resp.choices[0].message.content or ''
            return _apply_local_stop(text, local_stop)
        except TokenBudgetExceeded:
            raise
        except ProviderUsageMissing:
            raise
        except ProviderIdentityMismatch:
            raise
        except Exception as e:
            last_error = e
            if not response_committed:
                _record_resource_failure()
            ledger = _get_token_ledger()
            if ledger is not None and ledger.require_provider_usage:
                if response_committed:
                    raise
                raise ProviderCallUncertain('strict provider call failed with uncertain charge; automatic retry is disabled') from e
            logger.warning('complete attempt %d failed: %s', attempt + 1, e)
            time.sleep(2 ** attempt)
    ledger = _get_token_ledger()
    if ledger is not None and ledger.require_provider_usage:
        raise RuntimeError('LLM complete failed after 6 provider attempts') from last_error
    return ''

def generate(prompt: str, model: str, stop: Optional[List[str]]=None, temperature: float=0.0, max_tokens: int=256, seed: int=42, retry_on_empty: bool=True) -> str:
    """LLM call for action generation.

    Auto-injects working memory into system prompt on EVERY call within
    a trial.  The LM treats WM as persistent meta-cognitive context
    (what it knows from past attempts), not as an instruction to act on.
    """
    prompt = _actor_user_prompt(prompt)
    ctx = _get_context()
    system = ctx if ctx else None
    c = get_client()
    messages: list = []
    if system:
        messages.append({'role': 'system', 'content': system})
    messages.append({'role': 'user', 'content': prompt})
    extra_body = _provider_extra_body(model)
    reasoning_effort = _fixed_reasoning_effort(model)
    reasoning = _reasoning_controls_sampling(model, extra_body, reasoning_effort)
    provider_stop = _provider_stop(model, stop)
    local_stop = stop if stop is not None and provider_stop is None else None
    last_error: Exception | None = None
    max_attempts = 6 if retry_on_empty else 1
    for attempt in range(max_attempts):
        admitted_max_tokens = _admit_llm_call(messages, model=model, requested_max_tokens=max_tokens)
        api_kwargs: dict = dict(model=model, messages=messages)
        if admitted_max_tokens is None:
            raise AssertionError('generate requires a finite completion allowance')
        api_kwargs.update(_token_kwarg(model, admitted_max_tokens))
        if reasoning_effort is not None:
            api_kwargs['reasoning_effort'] = reasoning_effort
        if extra_body is not None:
            api_kwargs['extra_body'] = extra_body
        if not reasoning and _supports_temperature(model):
            api_kwargs['temperature'] = temperature
        if provider_stop is not None:
            api_kwargs['stop'] = provider_stop
        if _supports_seed(model):
            api_kwargs['seed'] = seed
        _respect_provider_call_spacing()
        response_committed = False
        try:
            resp = c.chat.completions.create(**api_kwargs)
            response_committed = True
            _record_resource_usage(resp)
            text = resp.choices[0].message.content or ''
            text = _apply_local_stop(text, local_stop)
            if text.strip():
                return text
            ledger = _get_token_ledger()
            if ledger is not None and ledger.require_provider_usage:
                return ''
            temperature = min(temperature + 0.2, 1.0)
            if not reasoning and _supports_temperature(model):
                api_kwargs['temperature'] = temperature
        except TokenBudgetExceeded:
            raise
        except ProviderUsageMissing:
            raise
        except ProviderIdentityMismatch:
            raise
        except Exception as e:
            last_error = e
            if not response_committed:
                _record_resource_failure()
            ledger = _get_token_ledger()
            if ledger is not None and ledger.require_provider_usage:
                if response_committed:
                    raise
                raise ProviderCallUncertain('strict provider call failed with uncertain charge; automatic retry is disabled') from e
            logger.warning('generate attempt %d failed: %s', attempt + 1, e)
            time.sleep(2 ** attempt)
    ledger = _get_token_ledger()
    if ledger is not None and ledger.require_provider_usage and (last_error is not None):
        raise RuntimeError(f'LLM generate failed after {max_attempts} provider attempt(s)') from last_error
    return ''
