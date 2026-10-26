"""Child-process entry point for trusted VRL-Bench simulation components.

The simulator starts a fresh worker for every run. This provides
process-state isolation, not a security sandbox for hostile Python programs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping

from vrlbench.plugins import runtime_root
from vrlbench.provider_usage import ProviderUsageJournal
from vrlbench.specs import _validate_single_completion_count


def _persist_request_event(callback: Any, *args: Any) -> Any:
    try:
        return callback(*args)
    except OSError as exc:
        from vrlbench.runtime.budget import TokenBudgetExceeded

        class ProviderUsagePersistenceError(TokenBudgetExceeded):
            pass

        # The reference loop never retries budget-stop exceptions. A disk
        # failure must stop further calls, not look like a retryable API error.
        raise ProviderUsagePersistenceError(
            "Could not persist model-request accounting; stopped further model calls"
        ) from exc


class _CompletionProxy:
    def __init__(self, owner: "_ClientProxy") -> None:
        self.owner = owner

    def create(self, **kwargs: Any) -> Any:
        options = self.owner.options
        forwarded = dict(kwargs)
        for key, value in options.items():
            if value is None:
                forwarded.pop(key, None)
            elif key == "extra_body" and isinstance(value, dict):
                forwarded[key] = {**dict(forwarded.get(key) or {}), **value}
            else:
                forwarded[key] = value
        _validate_single_completion_count(forwarded)
        # Reference admission has already reserved a completion allowance. A
        # transport mapping must not enlarge or remove that limit when a
        # strict per-case token ceiling is active.
        from vrlbench.runtime import llm

        ledger = llm._get_token_ledger()
        if ledger is not None and ledger.max_total_tokens is not None:
            admitted = kwargs.get("max_completion_tokens", kwargs.get("max_tokens"))
            effective = {**forwarded, **dict(forwarded.get("extra_body") or {})}
            declared = effective.get("max_completion_tokens", effective.get("max_tokens"))
            if admitted is not None and (type(declared) is not int or declared > admitted or declared < 1):
                raise ValueError("request_options cannot increase or remove a strict completion-token allowance")
        create = self.owner.client.chat.completions.create
        journal = self.owner.journal
        request_id = _persist_request_event(journal.begin) if journal is not None else None
        try:
            response = create(**forwarded)
        except BaseException as exc:
            if journal is not None:
                _persist_request_event(journal.failure, request_id, exc)
            raise
        if journal is not None:
            _persist_request_event(journal.response, request_id, response)
        return response


class _ClientProxy:
    """Apply one declared request mapping to actor and memory calls alike."""
    def __init__(self, factory: Any, options: Mapping[str, Any],
                 journal: ProviderUsageJournal | None = None) -> None:
        self._factory = factory
        self._client: Any = None
        self.options = dict(options)
        self.journal = journal
        self.chat = SimpleNamespace(completions=_CompletionProxy(self))

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._factory()
            create = getattr(getattr(getattr(self._client, "chat", None), "completions", None), "create", None)
            if not callable(create):
                raise TypeError("A model transport factory must return an OpenAI-compatible client")
        return self._client


def configure_model(model: Mapping[str, Any], *, offline_demo: bool = False,
                    journal: ProviderUsageJournal | None = None) -> list[dict[str, Any]]:
    """Bind a lazy, process-local transport shared by all reference calls."""
    from vrlbench.runtime import llm

    options = model.get("request_options", {})
    if not isinstance(options, dict):
        raise TypeError("model.request_options must be an object")
    if options.get("extra_body") is not None and not isinstance(options["extra_body"], dict):
        raise TypeError("request_options.extra_body must be an object or null")
    forbidden = {"model", "messages", "stream"} & set(options)
    if isinstance(options.get("extra_body"), dict):
        forbidden |= {"model", "messages", "stream"} & set(options["extra_body"])
    if forbidden:
        raise ValueError(f"request_options cannot override protocol fields: {sorted(forbidden)}")
    _validate_single_completion_count(options)

    base_url = model.get("base_url")
    if base_url:
        os.environ["OPENAI_BASE_URL"] = str(base_url)
    api_key_env = str(model.get("api_key_env", "OPENAI_API_KEY"))
    secret = os.environ.get(api_key_env)
    if secret:
        os.environ["OPENAI_API_KEY"] = secret
    else:
        os.environ.pop("OPENAI_API_KEY", None)
    os.environ["MODEL_PROVIDER"] = str(model.get("provider", "openai_compatible"))
    os.environ["VEX2_LLM_TIMEOUT"] = str(model.get("timeout_seconds", 120))
    allowed = model.get("allowed_response_models", [])
    if allowed:
        os.environ["VEX2_ALLOWED_RESPONSE_MODELS"] = ",".join(allowed)
    identity = []
    transport = model.get("transport")
    if transport is not None:
        # Instantiate lazily too: the demo must not initialize a network client
        # merely because a model configuration is present.
        from vrlbench.plugins import load_factory

        factory, source = load_factory(transport)
        kwargs = dict(transport.get("kwargs", {}))
        client_factory = lambda: factory(**kwargs)
        identity.append({"role": "model_transport", **source})
    elif offline_demo:
        def client_factory() -> Any:
            from vrlbench.runtime.budget import TokenBudgetExceeded

            class DemoModelCallsDisabled(TokenBudgetExceeded):
                pass

            def reject_call(**kwargs: Any) -> Any:
                # The reference loop does not retry budget exceptions. Treat
                # this deterministic example as having a zero model-call budget.
                raise DemoModelCallsDisabled(
                    "The synthetic demo disables model API calls; supply an explicit test transport "
                    "or select a real task environment for a model-based memory strategy"
                )

            return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=reject_call)))
    else:
        def client_factory() -> Any:
            from openai import OpenAI

            return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),
                          base_url=os.environ.get("OPENAI_BASE_URL"),
                          timeout=float(model.get("timeout_seconds", 120)), max_retries=0)
    llm._client = _ClientProxy(client_factory, options,
                              journal if transport is not None or not offline_demo else None)
    return identity


def _validate_cases(cases: Any, total_trials: int) -> list[dict[str, Any]]:
    if not isinstance(cases, list):
        raise TypeError("A benchmark backend must return a list of case records")
    identifiers = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("case_id"), str) or not case["case_id"]:
            raise TypeError("Each case must have a nonempty string case_id")
        if case["case_id"] in identifiers:
            raise ValueError(f"Duplicate returned case_id: {case['case_id']}")
        identifiers.add(case["case_id"])
        if not isinstance(case.get("task"), str) or not isinstance(case.get("resource_usage"), dict):
            raise TypeError("Each case requires task and resource_usage")
        trials = case.get("trials")
        if not isinstance(trials, list) or not 1 <= len(trials) <= total_trials:
            raise ValueError("A completed case requires between one and T executed trials")
        for index, trial in enumerate(trials):
            if not isinstance(trial, dict) or type(trial.get("trial")) is not int or trial["trial"] != index:
                raise ValueError("Trial indices must be contiguous and zero based")
            if type(trial.get("success")) is not bool:
                raise TypeError("Trial success must be boolean")
            if trial["success"] and index != len(trials) - 1:
                raise ValueError("A success-stop case cannot contain trials after success")
    return cases


def expected_case_ids(benchmark: Mapping[str, Any]) -> set[str]:
    """Resolve the public selection to canonical returned case identifiers."""
    name = benchmark["name"]
    selected = benchmark["cases"]
    if name == "webshop":
        identifiers = [f"sample_{case:04d}" for case in selected]
    elif name == "miniwob":
        identifiers = [f"{case['task']}_ep{case['episode']}" for case in selected]
    else:
        identifiers = [str(case["case_id"]) if isinstance(case, dict) else str(case) for case in selected]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("The selected cases have duplicate canonical identifiers")
    return set(identifiers)


def execute_request(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Execute in the worker only; the public SDK does not call this in-process."""
    journal = ProviderUsageJournal(Path(spec["output_dir"]))
    runtime = runtime_root()
    for key in ("OPENAI_BASE_URL", "VEX2_ALLOWED_RESPONSE_MODELS", "VEX2_ALLOWED_RESPONSE_MODEL_PREFIXES",
                "VEX2_REQUIRE_SYSTEM_FINGERPRINT", "VEX2_LLM_REASONING_EFFORT",
                "VEX2_EXPECTED_RUN_META_FILE"):
        os.environ.pop(key, None)
    for key, value in spec.get("runtime_env", {}).items():
        os.environ[str(key)] = str(value)
    try:
        offline_demo = spec["benchmark"]["name"] == "demo" and spec["benchmark"].get("factory") is None
        identity = configure_model(spec["model"], offline_demo=offline_demo, journal=journal)
        from vrlbench.runtime import llm

        llm.set_context("")
        llm.set_user_context("")
        from vrlbench.backends import memory_bindings, run_backend

        with memory_bindings(spec) as (er_name, controller, memory_identity):
            cases, backend_identity = run_backend(spec, er_name, controller)
        cases = _validate_cases(cases, int(spec["trials"]))
        json.dumps(cases, allow_nan=False)
        requested = expected_case_ids(spec["benchmark"])
        observed = {case["case_id"] for case in cases}
        if observed != requested:
            missing, extra = sorted(requested - observed), sorted(observed - requested)
            raise ValueError(f"Returned case identities differ from the selection: missing={missing}, extra={extra}")
        run_identity = {"runtime_root": str(runtime), "reference_root": str(runtime),
                        "plugins": identity + memory_identity + backend_identity}
        native_summary = Path(spec["output_dir"]) / "reference" / "results.json"
        if spec["benchmark"]["name"] == "webshop" and native_summary.is_file():
            environment_identity = json.loads(native_summary.read_text()).get("environment_identity")
            if environment_identity is not None:
                run_identity["environment"] = environment_identity
        return {"schema_version": 1, "status": "complete", "cases": cases, "error": None,
                "identity": run_identity}
    except ModuleNotFoundError as exc:
        if exc.name == "openai":
            raise ModuleNotFoundError(
                "Evaluation requires the runtime extra: pip install 'vrlbench[runtime]'"
            ) from exc
        raise


def _redact(message: str) -> str:
    for name, value in os.environ.items():
        if value and len(value) >= 4 and any(marker in name.upper() for marker in ("API_KEY", "TOKEN", "SECRET", "PASSWORD")):
            message = message.replace(value, "[REDACTED]")
    return message


def _partial_cases(output_dir: Path) -> list[dict[str, Any]]:
    """Keep readable native checkpoints after failure, without calling them complete."""
    from vrlbench.backends import _case_rows

    result = []
    seen = set()
    for path in sorted((output_dir / "reference").glob("**/episodes.json")):
        try:
            for case in _case_rows(path):
                if case["case_id"] not in seen:
                    result.append(case)
                    seen.add(case["case_id"])
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return result


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("Usage: python -m vrlbench.worker request.json", file=sys.stderr)
        return 2
    try:
        spec = json.loads(Path(args[0]).read_text())
        output_dir = Path(spec["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"Invalid evaluation request: {_redact(str(exc))}", file=sys.stderr)
        return 2
    try:
        result = execute_request(spec)
    except Exception as exc:
        result = {"schema_version": 1, "status": "failed", "cases": _partial_cases(output_dir),
                  "error": {"type": type(exc).__name__, "message": _redact(str(exc))}}
    target = output_dir / "worker_result.json"
    temporary = target.with_suffix(".json.tmp")
    try:
        serialized = json.dumps(result, indent=2, allow_nan=False)
    except (TypeError, ValueError) as exc:
        result = {"schema_version": 1, "status": "failed", "cases": [],
                  "error": {"type": type(exc).__name__, "message": _redact(str(exc))}}
        serialized = json.dumps(result, indent=2, allow_nan=False)
    temporary.write_text(serialized + "\n")
    temporary.replace(target)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
