"""Sequential native-environment worker for an external actor's Session bridge."""

from __future__ import annotations

import base64
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
from typing import Any


def _observation(value: Any) -> dict[str, Any]:
    return {"text": value.text,
            "image_base64": base64.b64encode(value.image).decode("ascii") if value.image is not None else None}


def _dispatch(session: Any, request: dict[str, Any]) -> Any:
    method = request["method"]
    if method == "reset":
        value = session.reset(trial_index=request["trial_index"])
        return {"task": value.task, "observation": _observation(value.observation), "identity": value.identity}
    if method == "parse_response":
        return session.parse_response(request["raw"])
    if method == "step":
        value = session.step(request["action"])
        return {"observation": _observation(value.observation),
                "terminated": value.terminated, "truncated": value.truncated}
    if method == "evaluate":
        success = session.evaluate()
        if type(success) is not bool:
            raise TypeError("environment evaluate must return a boolean")
        return {"success": success, "score": getattr(session, "evaluation_score", None)}
    if method == "get_trial_metadata":
        hook = getattr(session, "get_trial_metadata", None)
        return hook() if callable(hook) else {}
    if method == "capture_observation":
        value = session.capture_observation()
        if isinstance(value, bytes):
            return {"kind": "image", "data": base64.b64encode(value).decode("ascii")}
        if isinstance(value, str):
            return {"kind": "html", "data": value}
        raise TypeError("environment capture must return image bytes or HTML text")
    if method == "close":
        session.close()
        return None
    raise ValueError(f"Unsupported session method {method!r}")


def main() -> int:
    # Preserve a private copy of the protocol pipe, then route Python, native
    # libraries and inherited subprocess stdout to stderr before loading an env.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    session = None
    ledger = None
    scopes = ExitStack()
    try:
        for line in sys.stdin:
            closing = False
            before = ledger.snapshot() if ledger is not None else {"total_tokens": 0, "api_calls": 0}
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise TypeError("session request must be an object")
                method = request.get("method")
                if method == "init":
                    if session is not None:
                        raise RuntimeError("session was already initialized")
                    from .native_sessions import make_session
                    from vrlbench.provider_usage import ProviderUsageJournal
                    from vrlbench.runtime import llm
                    from vrlbench.runtime.budget import TokenUsageLedger
                    from vrlbench.worker import configure_model

                    ledger = TokenUsageLedger(require_provider_usage=request.get("require_provider_usage") is True)
                    scopes.enter_context(llm.token_budget_scope(ledger))
                    model_config = request.get("model_config")
                    if model_config is not None:
                        path = request.get("journal_path")
                        journal = ProviderUsageJournal.attach(Path(path).parent) if path is not None else None
                        configure_model(model_config, journal=journal)
                    params = dict(request["params"])
                    params.pop("environment_python", None)
                    session = make_session(request["benchmark"], request["case"], params)
                    result = {"system_prompt": session.system_prompt, "basic_rules": session.basic_rules,
                              "task_group": getattr(session, "task_group", request["benchmark"]),
                              "task_scope": getattr(session, "task_scope", "Complete this task through repeated attempts."),
                              "source_identity": getattr(session, "source_identity", {}),
                              "allows_task_variation": getattr(session, "allows_task_variation", False)}
                else:
                    if session is None:
                        raise RuntimeError("initialize the session before using it")
                    ledger.require_provider_usage = request.get("require_provider_usage") is True
                    result = _dispatch(session, request)
                    closing = method == "close"
                    if closing:
                        session = None
                response = {"ok": True, "result": result}
                # Validate the native result before writing a protocol frame.
                json.dumps(response, allow_nan=False)
            except Exception as exc:
                response = {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}}
            response["model_usage"] = (ledger.delta(before) if ledger is not None else
                                       {"api_calls": 0, "total_tokens": 0, "provider_usage_complete": True})
            encoded = json.dumps(response, allow_nan=False)
            try:
                protocol.write(encoded + "\n")
                protocol.flush()
            except BrokenPipeError:
                break
            if closing:
                break
    finally:
        try:
            if session is not None:
                session.close()
        finally:
            scopes.close()
            protocol.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
