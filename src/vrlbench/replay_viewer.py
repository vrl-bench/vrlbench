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
from urllib.parse import quote

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


def _snapshot(root: Path, original_root: str | Path | None, item: dict, *,
              other_runs: Iterable[tuple[Path, Path | None]] = ()) -> tuple[str | None, str | None, str | None]:
    """Read assets only from selected runs or a verified shared-trial source."""
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


def _shared_snapshot_run(path: Path, case_id: str, trial: dict, *,
                         selected: list[tuple[Path, SimulationResult]],
                         cache: dict[Path, tuple[dict, str]]) -> tuple[Path, Path | None] | None:
    """Authorize this trial's recorded source, without following source chains.

    The result checksum binds the source metadata; matching the first trial
    binds the requested images to that experience. Individual image checksums
    and resolved-path containment are still checked by ``_snapshot``.
    """
    reference = (trial.get("trial_metadata") or {}).get("shared_t1_source")
    if reference is None:
        return None
    if (not isinstance(reference, dict) or reference.get("case_id") != case_id
            or not isinstance(reference.get("evaluation_path"), str)
            or not isinstance(reference.get("result_sha256"), str)
            or trial.get("trial") != 0):
        raise ValueError("Cannot replay malformed shared first-trial source")
    recorded_path = Path(reference["evaluation_path"])
    if not recorded_path.is_absolute() or recorded_path.name != "evaluation.json":
        raise ValueError("Shared first-trial source must name an absolute evaluation.json")
    # Explicitly supplied runs also provide relocation mappings for old paths.
    source = next((given for given, result in selected
                   if given == recorded_path or (result.output_dir is not None
                       and result.output_dir / "evaluation.json" == recorded_path)), recorded_path).resolve()
    if source == path:
        raise ValueError("A shared first trial cannot refer to its own run")
    if not source.is_file():
        return None  # Keep the text trace and the existing missing-source notice.
    if source not in cache:
        if source.stat().st_size > _MAX_REPORT_BYTES:
            raise ValueError("Shared first-trial source exceeds the 64 MiB reading limit")
        payload = SimulationResult.from_path(source).require_complete().to_dict()
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                          ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        cache[source] = payload, digest
    payload, digest = cache[source]
    if digest != reference["result_sha256"].lower():
        raise ValueError("Shared first-trial source checksum does not match its record")
    cases = [case for case in payload.get("cases", []) if case.get("case_id") == case_id]
    if len(cases) != 1 or not cases[0].get("trials"):
        raise ValueError("Shared first-trial source does not contain the recorded case")
    original = cases[0]["trials"][0]
    for field in ("trial", "success", "trajectory", "policy_trajectory", "actor_trajectory"):
        if original.get(field) != trial.get(field):
            raise ValueError(f"Shared first-trial source has a different {field}")
    for field in ("replay_observations", "screenshots"):
        if ((original.get("trial_metadata") or {}).get(field)
                != (trial.get("trial_metadata") or {}).get(field)):
            raise ValueError("Shared first-trial snapshot references differ from their source")
    original_root = payload.get("output_dir")
    return source.parent, Path(original_root) if original_root is not None else None


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
    source_cache: dict[Path, tuple[dict, str]] = {}
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
            original = next(c for c in payload["cases"] if c["case_id"] == identity)
            partial = view.get("in_progress_trial")
            for row in [*view["trials"], *([partial] if partial is not None else [])]:
                incomplete = row is partial
                evidence = (deepcopy(row.get("policy_evidence")) if incomplete else
                            _linear_trace(row.get("policy_trajectory") or ""))
                if incomplete and evidence is not None:
                    if (not isinstance(evidence, dict) or not isinstance(evidence.get("task"), str)
                            or not isinstance(evidence.get("initial_observation"), str)
                            or not isinstance(evidence.get("steps"), list)
                            or any(not isinstance(step, dict) or set(step) != {"action", "observation"}
                                   or not isinstance(step["action"], str) or not step["action"].strip()
                                   or not isinstance(step["observation"], str)
                                   for step in evidence["steps"])):
                        raise ValueError("Cannot replay malformed in-progress action observations")
                metadata = row.get("trial_metadata") or {}
                shared_root = _shared_snapshot_run(path, identity, row, selected=saved_runs,
                                                   cache=source_cache)
                snapshot_root, original_root = shared_root or (path.parent, payload.get("output_dir"))
                memory = {key: deepcopy(row.get(key)) for key in
                          ("context", "system_context", "user_context")}
                # Controller state is distinct from the exact actor input.
                stored = row if incomplete else next(
                    t for t in original["trials"] if t["trial"] == row["trial"])
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
                if incomplete:
                    trial.update(incomplete=True, status=row["status"], phase=row["phase"],
                                 error=deepcopy(row.get("error")),
                                 environment_result=deepcopy(row.get("environment_result")))
                if evidence is None:
                    trial["frame_notice"] = "No canonical per-action observation record is available."
                else:
                    case["task"] = evidence["task"]
                    trial["initial_observation"] = evidence["initial_observation"]
                    trial["close_reason"] = metadata.get("trial_close_reason") or evidence.get("trial_close_reason")
                    observations = [evidence["initial_observation"], *(s["observation"] for s in evidence["steps"])]
                    snapshots = {}
                    items = [*metadata.get("replay_observations", []),
                             *({**s, "kind": "image"} for s in metadata.get("screenshots", []))]
                    for item in items:
                        if isinstance(item, dict) and type(item.get("observation_index")) is int:
                            snapshots[item["observation_index"]] = _snapshot(
                                snapshot_root, original_root, item, other_runs=source_roots)
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
                pending = row.get("pending_action") if incomplete else None
                if pending is not None:
                    if evidence is None:
                        raise ValueError("Cannot replay a pending action without a recorded observation")
                    native = len(evidence["steps"])
                    rejected = sum(len(events) for events in _invalid_events(metadata, native).values())
                    if (not isinstance(pending, dict)
                            or not isinstance(pending.get("action"), str) or not pending["action"].strip()
                            or not isinstance(pending.get("raw_response"), str)
                            or type(pending.get("environment_steps_before")) is not int
                            or pending["environment_steps_before"] != native
                            or type(pending.get("budget_used_before")) is not int
                            or pending["budget_used_before"] != native + rejected
                            or pending.get("before_observation") != observations[-1]):
                        raise ValueError("Cannot replay malformed pending environment action")
                    before = snapshots.get(native, (None, None, None))
                    trial["frames"].append({
                        "kind": "pending_action", "action": pending["action"],
                        "raw_response": pending["raw_response"],
                        "step_index": native, "budget_used_before": pending["budget_used_before"],
                        "budget_used_after": None, "environment_steps_before": native,
                        "environment_steps_after": None,
                        "before_observation": pending["before_observation"], "after_observation": None,
                        "before_image": before[0], "after_image": None,
                        "before_html": before[1], "after_html": None,
                        "before_snapshot_notice": before[2],
                        "after_snapshot_notice": "Environment feedback was not recorded; execution result is unknown."})
                case["trials"].append(trial)
            run["cases"].append(case)
        runs.append(run)
    if not runs:
        raise ValueError("replay requires at least one saved run")
    return {"schema_version": 1, "runs": runs}


def export_replay(paths: Iterable[str | Path], output: str | Path, *,
                  case_id: str | None = None, external_images: bool = False) -> Path:
    """Write a replay, embedding images by default, without overwriting outputs."""
    if type(external_images) is not bool:
        raise TypeError("external_images must be a boolean")
    requested = Path(output)
    destination = requested.resolve()
    if destination.suffix.lower() != ".html":
        raise ValueError("replay output must be an .html file")
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(f"Replay output already exists: {requested}")
    asset_directory = destination.with_name(destination.stem + ".assets")
    if external_images and (asset_directory.exists() or asset_directory.is_symlink()):
        raise FileExistsError(f"Replay asset directory already exists: {asset_directory}")

    data = replay_data(paths, case_id=case_id)
    assets: dict[str, bytes] = {}
    if external_images:
        # Only these fields contain snapshots already checked by _snapshot.
        # Actor text and metadata are not resource references.
        for run in data["runs"]:
            for case in run["cases"]:
                for trial in case["trials"]:
                    images = [(trial, "initial_image")]
                    images.extend((frame, field) for frame in trial["frames"]
                                  for field in ("before_image", "after_image"))
                    for record, field in images:
                        source = record[field]
                        if source is None:
                            continue
                        header, encoded = source.split(",", 1)
                        suffix = {"data:image/png;base64": ".png",
                                  "data:image/jpeg;base64": ".jpg"}[header]
                        raw = base64.b64decode(encoded, validate=True)
                        filename = hashlib.sha256(raw).hexdigest() + suffix
                        assets.setdefault(filename, raw)
                        record[field] = quote(asset_directory.name, safe="") + "/" + filename

    content = json.dumps(data, ensure_ascii=False, allow_nan=False)
    # A transcript can contain </script>; JSON quoting alone is insufficient.
    for source, replacement in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                                ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        content = content.replace(source, replacement)
    template = Path(__file__).with_name("_replay.html").read_text(encoding="utf-8")
    if external_images:
        template = template.replace("<head>", '<head>\n<meta http-equiv="Content-Security-Policy" '
                                    'content="img-src \'self\' data:">', 1)
    rendered = template.replace("__VRLBENCH_REPLAY_DATA__", content)
    if len(rendered.encode("utf-8")) > _MAX_REPORT_BYTES:
        raise ValueError("Replay exceeds 64 MiB; use --external-images / external_images=True "
                         "or select one case with --case")

    destination.parent.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    created_directory = False
    try:
        if assets:
            asset_directory.mkdir()
            created_directory = True
            for filename, raw in assets.items():
                path = asset_directory / filename
                with path.open("xb") as stream:
                    created.append(path)
                    stream.write(raw)
        with destination.open("x", encoding="utf-8") as stream:
            created.append(destination)
            stream.write(rendered)
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        if created_directory:
            asset_directory.rmdir()
        raise
    return destination
