"""Export recorded actions, observations and memory as a local step viewer.

This module reads results only. It never loads an agent plugin, restores an
environment or calls a model. Page snapshots are optional recorded artifacts;
missing snapshots stay missing rather than being reconstructed from text.
"""
from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
from typing import Any, Iterable

from .inspection import inspect_case
from .results import SimulationResult

_MAX_ASSET_BYTES = 16 * 1024 * 1024
_MAX_REPORT_BYTES = 64 * 1024 * 1024
_ACTION_ATTEMPT_BUDGET_UNIT = "environment_action_or_invalid_response"


def _linear_trace(source: str) -> dict | None:
    """Read the canonical linear record without importing a model runtime."""
    lines = source.splitlines()
    if not lines or lines[0] != "policy_ao_trace":
        return None
    try:
        if not lines[1].startswith("Task: "):
            raise ValueError("task header is missing")
        if not lines[2].startswith("Initial observation JSON: "):
            raise ValueError("initial observation is missing")
        initial = json.loads(lines[2].removeprefix("Initial observation JSON: "))
        if not isinstance(initial, str):
            raise ValueError("initial observation is not text")
        steps = []
        for number, line in enumerate(lines[3:-1], 1):
            prefix = f"AO {number} JSON: "
            if not line.startswith(prefix):
                raise ValueError("action indices are not contiguous")
            step = json.loads(line.removeprefix(prefix))
            if (not isinstance(step, dict) or set(step) != {"action", "observation"}
                    or not isinstance(step["action"], str) or not step["action"].strip()
                    or not isinstance(step["observation"], str)):
                raise ValueError("invalid action-observation transition")
            steps.append(step)
        if not lines[-1].startswith("Terminal JSON: "):
            raise ValueError("terminal record is missing")
        terminal = json.loads(lines[-1].removeprefix("Terminal JSON: "))
        if (not isinstance(terminal, dict) or set(terminal) != {"trial_close_reason"}
                or not isinstance(terminal["trial_close_reason"], str)):
            raise ValueError("invalid terminal record")
        return {"task": lines[1].removeprefix("Task: "),
                "initial_observation": initial, "steps": steps,
                "trial_close_reason": terminal["trial_close_reason"]}
    except (IndexError, ValueError, TypeError) as exc:
        raise ValueError(f"Cannot replay malformed canonical trace: {exc}") from exc


class _StaticHTML(HTMLParser):
    """Keep page structure but not executable code, forms or network resources."""

    tags = frozenset("html body div span p br hr h1 h2 h3 h4 h5 h6 b strong i em small pre code ul ol li table thead tbody tfoot tr td th caption button label input select option textarea section article header footer main nav a img dl dt dd fieldset legend".split())
    blocked = frozenset({"script", "style", "iframe", "object", "embed", "svg", "math", "template"})
    void = frozenset({"br", "hr", "input", "img"})
    attrs = frozenset({"class", "id", "title", "type", "value", "placeholder", "colspan", "rowspan", "selected", "checked", "alt"})

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.blocked_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.blocked:
            self.blocked_depth += 1
            return
        if self.blocked_depth or tag not in self.tags:
            return
        safe = [(key, value or "") for key, value in attrs if key in self.attrs]
        if tag in {"button", "input", "select", "textarea"}:
            safe.append(("disabled", ""))
        self.parts.append("<" + tag + "".join(
            f' {key}="{escape(value, quote=True)}"' for key, value in safe) + ">")

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.void:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag in self.blocked:
            self.blocked_depth = max(0, self.blocked_depth - 1)
        elif not self.blocked_depth and tag in self.tags and tag not in self.void:
            self.parts.append(f"</{tag}>")

    def handle_data(self, text):
        if not self.blocked_depth:
            self.parts.append(escape(text))


def static_page_html(source: str) -> str:
    """Create an inert offline view; this is a DOM snapshot, not a screenshot."""
    parser = _StaticHTML()
    parser.feed(source)
    parser.close()
    return ('<!doctype html><meta charset="utf-8">'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; form-action \'none\'; base-uri \'none\'">'
            '<style>body{font:16px/1.6 system-ui,sans-serif;color:#172a40;padding:20px;overflow-wrap:anywhere}button,input,select,textarea{font:inherit;padding:6px 10px;margin:4px;border:1px solid #bcc9d7;border-radius:4px;color:#172a40;background:#f7f9fc}a{color:#2553a6}table{border-collapse:collapse}td,th{padding:6px 10px}pre{white-space:pre-wrap}img{display:none}</style>'
            + "".join(parser.parts))


def _snapshot(root: Path, original_root: str | None, item: dict, *,
              other_runs: Iterable[tuple[Path, Path | None]] = ()) -> tuple[str | None, str | None, str | None]:
    """Read assets only from runs explicitly selected by the caller.

    A shared first trial can reference the source Retry run. Its assets are
    readable only when that run was also supplied, including after relocation.
    """
    if item.get("kind") == "unavailable":
        return None, None, str(item.get("reason") or "Snapshot was not recorded")
    raw_path, expected = item.get("path"), item.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(expected, str):
        return None, None, "Snapshot path or checksum was not recorded"
    raw = Path(raw_path)
    path = None
    for directory, original in [(root, Path(original_root) if original_root else None), *other_runs]:
        candidate = raw if raw.is_absolute() else root / raw
        if raw.is_absolute() and original and raw.is_relative_to(original):
            candidate = directory / raw.relative_to(original)
        candidate = candidate.resolve()
        if candidate.is_relative_to(directory):
            path = candidate
            break
    if path is None:
        return None, None, "Snapshot is outside the selected run directories; include its source run to view it"
    if not path.is_file():
        return None, None, "Recorded snapshot file is missing"
    if path.stat().st_size > _MAX_ASSET_BYTES:
        return None, None, "Snapshot exceeds the 16 MiB embedding limit"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        return None, None, "Snapshot checksum does not match its record"
    if item.get("kind") == "html":
        try:
            return None, static_page_html(raw.decode("utf-8")), None
        except UnicodeError:
            return None, None, "HTML snapshot is not valid UTF-8"
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif raw.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    else:
        return None, None, "Snapshot is not a supported PNG, JPEG or HTML artifact"
    return "data:" + mime + ";base64," + base64.b64encode(raw).decode("ascii"), None, None


def _invalid_events(metadata: dict, environment_steps: int) -> dict[int, list[dict]]:
    """Index explicit rejected responses, never infer them from actor prose."""
    events = metadata.get("invalid_action_events", [])
    if not isinstance(events, list):
        raise ValueError("Cannot replay malformed invalid-action events: expected a list")
    grouped: dict[int, list[dict]] = {}
    previous = 0
    for rejected, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError("Cannot replay malformed invalid-action event")
        position = event.get("after_environment_step")
        before, after = event.get("budget_used_before"), event.get("budget_used_after")
        if (type(position) is not int or not previous <= position <= environment_steps
                or not isinstance(event.get("raw_response"), str)
                or not isinstance(event.get("reason"), str)
                or type(before) is not int or type(after) is not int
                or before != position + rejected or after != before + 1):
            raise ValueError("Cannot replay malformed invalid-action event position or budget")
        grouped.setdefault(position, []).append(event)
        previous = position
    return grouped


def replay_data(paths: Iterable[str | Path], *, case_id: str | None = None) -> dict[str, Any]:
    """Build a view from canonical executed A-O records without changing them."""
    saved_runs = []
    for given in paths:
        path = Path(given).resolve()
        path = path / "evaluation.json" if path.is_dir() else path
        saved_runs.append((path, SimulationResult.from_path(path)))
    source_roots = [(path.parent, result.output_dir) for path, result in saved_runs]
    runs = []
    for path, result in saved_runs:
        payload = result.to_dict()
        index = inspect_case(result)
        config = payload.get("config", {})
        run = {"label": path.parent.name, "status": result.status, "error": payload.get("error"),
               "method": (index.get("memory") or {}).get("name", "unknown"),
               "model": (index.get("model") or {}).get("name", "unknown"),
               "trial_budget": index.get("trial_budget"),
               "max_environment_steps": config.get("max_environment_steps"), "cases": []}
        if config.get("recording_mode") == "reexecuted_actions_from_saved_run":
            run["recording_notice"] = ("Re-executed recorded actions; HTML captured now, "
                                       "not during the original model run. No model was called.")
        selected = [case_id] if case_id is not None else [c["case_id"] for c in index["cases"]]
        for identity in selected:
            view = inspect_case(result, identity)["case"]
            case = {"case_id": identity, "task": view.get("task"), "trials": []}
            for row in view["trials"]:
                evidence = _linear_trace(row.get("policy_trajectory") or "")
                metadata = row.get("trial_metadata") or {}
                memory = {key: deepcopy(row.get(key)) for key in
                          ("context", "system_context", "user_context")}
                # Controller state is distinct from the exact actor input.
                original = next(c for c in payload["cases"] if c["case_id"] == identity)
                stored = next(t for t in original["trials"] if t["trial"] == row["trial"])
                for key in ("controller_state_pre", "controller_state"):
                    memory[key] = deepcopy(stored.get(key))
                memory["state_pre"] = deepcopy(stored.get("memory_state_pre"))
                memory["state_after"] = deepcopy(stored.get("memory_state"))
                trial = {"trial_index": row["trial"], "trial_number": row["trial_number"],
                         "success": row["success"], "close_reason": metadata.get("trial_close_reason"),
                         "metadata": deepcopy(metadata), "memory": memory, "frames": [],
                         "has_action_record": evidence is not None,
                         "initial_observation": None, "initial_image": None,
                         "initial_html": None, "initial_snapshot_notice": None,
                         "raw_actor_trajectory": row.get("actor_trajectory"),
                         "frame_notice": None}
                if evidence is None:
                    trial["frame_notice"] = "No canonical per-action observation record is available."
                else:
                    case["task"] = evidence["task"]
                    trial["initial_observation"] = evidence["initial_observation"]
                    trial["close_reason"] = metadata.get("trial_close_reason") or evidence["trial_close_reason"]
                    observations = [evidence["initial_observation"], *(s["observation"] for s in evidence["steps"])]
                    snapshots = {}
                    items = [*metadata.get("replay_observations", []),
                             *({**s, "kind": "image"} for s in metadata.get("screenshots", []))]
                    for item in items:
                        if isinstance(item, dict) and type(item.get("observation_index")) is int:
                            snapshots[item["observation_index"]] = _snapshot(
                                path.parent, payload.get("output_dir"), item, other_runs=source_roots)
                    initial = snapshots.get(0, (None, None, None))
                    trial.update(initial_image=initial[0], initial_html=initial[1],
                                 initial_snapshot_notice=initial[2])
                    rejected = _invalid_events(metadata, len(evidence["steps"]))
                    rejected_count = 0
                    for number in range(len(evidence["steps"]) + 1):
                        before = snapshots.get(number, (None, None, None))
                        for event in rejected.get(number, []):
                            trial["frames"].append({"kind": "invalid_action",
                                "after_environment_step": number, "action": None,
                                "raw_response": event["raw_response"], "reason": event["reason"],
                                "budget_used_before": event["budget_used_before"],
                                "budget_used_after": event["budget_used_after"],
                                "environment_steps_before": number, "environment_steps_after": number,
                                "before_observation": observations[number], "after_observation": observations[number],
                                "before_image": before[0], "after_image": before[0],
                                "before_html": before[1], "after_html": before[1],
                                "before_snapshot_notice": before[2], "after_snapshot_notice": before[2]})
                            rejected_count += 1
                        if number == len(evidence["steps"]):
                            break
                        step = evidence["steps"][number]
                        after = snapshots.get(number + 1, (None, None, None))
                        frame = {"step_index": number, "action": step["action"],
                            "before_observation": observations[number], "after_observation": observations[number + 1],
                            "before_image": before[0], "after_image": after[0],
                            "before_html": before[1], "after_html": after[1],
                            "before_snapshot_notice": before[2], "after_snapshot_notice": after[2]}
                        if metadata.get("budget_unit") == _ACTION_ATTEMPT_BUDGET_UNIT or rejected:
                            frame.update(kind="environment_action", environment_steps_before=number,
                                environment_steps_after=number + 1,
                                budget_used_before=number + rejected_count,
                                budget_used_after=number + rejected_count + 1)
                        trial["frames"].append(frame)
                case["trials"].append(trial)
            run["cases"].append(case)
        runs.append(run)
    if not runs:
        raise ValueError("replay requires at least one saved run")
    return {"schema_version": 1, "runs": runs}


def export_replay(paths: Iterable[str | Path], output: str | Path, *, case_id: str | None = None) -> Path:
    """Write a self-contained HTML report without overwriting any existing file."""
    data = replay_data(paths, case_id=case_id)
    content = json.dumps(data, ensure_ascii=False, allow_nan=False)
    # A transcript can contain </script>; JSON quoting alone is insufficient.
    for source, replacement in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                                ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        content = content.replace(source, replacement)
    template = Path(__file__).with_name("_replay.html").read_text(encoding="utf-8")
    rendered = template.replace("__VRLBENCH_REPLAY_DATA__", content)
    if len(rendered.encode("utf-8")) > _MAX_REPORT_BYTES:
        raise ValueError("Replay exceeds 64 MiB; select one case with --case")
    destination = Path(output).resolve()
    if destination.suffix.lower() != ".html":
        raise ValueError("replay output must be an .html file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(rendered)
    return destination
