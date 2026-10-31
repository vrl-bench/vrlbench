"""Durable accounting for SDK-managed model requests, separate from outcomes.

The append-only journal contains counters and request states only. It never
stores credentials, headers, prompts, response text, or exception messages.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from threading import Lock
import time
from typing import Any, Mapping
from uuid import uuid4


JOURNAL_NAME = "provider_requests.jsonl"
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def _count(value: Any) -> int | None:
    if type(value) is int and value >= 0:
        return value
    if type(value) is float and math.isfinite(value) and value >= 0 and value.is_integer():
        return int(value)
    return None


def _usage(response: Any) -> tuple[dict[str, int | None], bool]:
    raw = response.get("usage") if isinstance(response, Mapping) else getattr(response, "usage", None)
    values = {field: (raw.get(field) if isinstance(raw, Mapping) else getattr(raw, field, None))
              for field in TOKEN_FIELDS}
    for field, alias in (("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens")):
        if values[field] is None:
            values[field] = raw.get(alias) if isinstance(raw, Mapping) else getattr(raw, alias, None)
    counts = {field: _count(value) for field, value in values.items()}
    invalid = any(value is not None and counts[field] is None for field, value in values.items())
    prompt, completion, total = (counts[field] for field in TOKEN_FIELDS)
    if total is not None:
        invalid = invalid or any(value is not None and value > total for value in (prompt, completion))
        invalid = invalid or (prompt is not None and completion is not None and prompt + completion != total)
    elif prompt is not None and completion is not None:
        counts["total_tokens"] = prompt + completion
    return ({field: None for field in TOKEN_FIELDS} if invalid else counts,
            not invalid and counts["total_tokens"] is not None)


class ProviderUsageJournal:
    """Flush a request's start before invoking the transport, then its outcome."""

    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / JOURNAL_NAME
        self._lock = Lock()
        output_dir.mkdir(parents=True, exist_ok=True)
        # Workers are never resumed into an existing run directory.
        with self.path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"schema_version": 1, "event": "journal_started"}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    @classmethod
    def attach(cls, output_dir: Path) -> ProviderUsageJournal:
        """Append from an environment worker without creating another header."""
        path = Path(output_dir) / JOURNAL_NAME
        with path.open(encoding="utf-8") as handle:
            header = json.loads(handle.readline())
        if header != {"schema_version": 1, "event": "journal_started"}:
            raise ValueError("cannot attach to an invalid provider usage journal")
        journal = cls.__new__(cls)
        journal.path, journal._lock = path, Lock()
        return journal

    def _append(self, event: dict[str, Any]) -> None:
        line = json.dumps({"schema_version": 1, "time_ns": time.time_ns(), **event}, allow_nan=False)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def begin(self) -> str:
        request_id = uuid4().hex
        self._append({"event": "request_started", "request_id": request_id})
        return request_id

    def response(self, request_id: str, response: Any) -> None:
        usage, complete = _usage(response)
        self._append({"event": "request_completed", "request_id": request_id,
                      "usage": usage, "provider_usage_complete": complete})

    def failure(self, request_id: str, error: BaseException) -> None:
        self._append({"event": "request_failed", "request_id": request_id,
                      "error_type": type(error).__name__})


def read_provider_usage(output_dir: Path) -> dict[str, Any] | None:
    """Recover known usage even if the worker never committed a case/result.

    An unmatched start, malformed trailing write, failed request, or invalid
    provider usage keeps the final total unknown. Reported subtotals survive.
    """
    path = output_dir / JOURNAL_NAME
    if not path.is_file():
        return None
    starts: set[str] = set()
    endings: dict[str, dict[str, Any]] = {}
    valid = True
    header = False
    try:
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict) or event.get("schema_version") != 1:
                        raise ValueError("invalid event")
                    name = event.get("event")
                    if index == 0 and name == "journal_started":
                        header = True
                        continue
                    request_id = event.get("request_id")
                    if not isinstance(request_id, str) or not request_id:
                        raise ValueError("missing request identity")
                    if name == "request_started" and request_id not in starts:
                        starts.add(request_id)
                    elif (name in {"request_completed", "request_failed"}
                          and request_id in starts and request_id not in endings):
                        endings[request_id] = event
                    else:
                        raise ValueError("invalid request sequence")
                except (ValueError, TypeError):
                    valid = False
    except (OSError, UnicodeError):
        valid = False
    valid = valid and header
    completed = [event for event in endings.values() if event["event"] == "request_completed"]
    reported_usage = []
    complete_responses = 0
    for event in completed:
        usage, complete = _usage({"usage": event.get("usage")})
        reported_usage.append(usage)
        if complete and event.get("provider_usage_complete") is True:
            complete_responses += 1
    unknown = len(starts) - complete_responses
    all_known = valid and unknown == 0
    subtotals: dict[str, int | None] = {"api_calls": len(starts)}
    totals: dict[str, int | None] = {"api_calls": len(starts) if valid else None}
    for field in TOKEN_FIELDS:
        reported = [usage[field] for usage in reported_usage if usage[field] is not None]
        subtotals[field] = sum(reported) if reported or not starts and valid else None
        totals[field] = (sum(reported) if all_known and len(reported) == len(starts) else None)
    return {"basis": "actual_sdk_model_requests_in_this_run", **totals,
            "provider_usage_complete": all_known, "known_subtotals": subtotals,
            "requests_started": len(starts), "responses_received": len(completed),
            "failed_requests": sum(event["event"] == "request_failed" for event in endings.values()),
            "pending_requests": len(starts) - len(endings), "unknown_usage_calls": unknown,
            "journal_valid": valid}


def merge_provider_resources(resources: Mapping[str, Any], journal: Mapping[str, Any] | None,
                             *, include_zero: bool) -> dict[str, Any]:
    """Use actual new calls, not copied shared-T1 usage or case subtotals twice.

    Custom backends that do not use the SDK model transport retain their own
    accounting. Native workloads also accept a zero-request journal: replaying
    a successful first trial makes no new model request.
    """
    result = dict(resources)
    if journal is None:
        return result
    result["provider_requests"] = dict(journal)
    if not include_zero and not journal["requests_started"]:
        return result
    for field in ("api_calls", *TOKEN_FIELDS, "provider_usage_complete"):
        result[field] = journal[field]
    result["known_subtotals"] = {**dict(resources.get("known_subtotals", {})),
                                 **journal["known_subtotals"]}
    # These counters describe case checkpoints, not the request journal. Keep
    # only the environment coverage here; cases retain their original ledgers.
    result["known_cases"] = {key: value for key, value in dict(resources.get("known_cases", {})).items()
                             if key not in {"api_calls", *TOKEN_FIELDS}}
    result["basis"] = "actual_sdk_model_requests_and_received_case_environment_steps"
    return result
