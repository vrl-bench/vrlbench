"""Execute memory-system simulations in isolated workers with one trial loop.

The simulator composes memory, model and workload configurations, records the
resulting trial processes, and summarizes their outcomes and resource use.
The worker and reference adapters execute the actual memory lifecycle; this
module owns planning, process isolation, timeouts and result construction.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

from .specs import SimulationSpec
from .results import SimulationResult, aggregate_cases
from .provider_usage import JOURNAL_NAME, merge_provider_resources, read_provider_usage
from ._version import __version__
from ._comparison import comparable_workload


def plan(spec: SimulationSpec) -> dict[str, Any]:
    """Resolve and validate input without loading a plugin or making model calls."""
    resolved = spec.resolved()
    resolved.validate()
    _validate_shared_source(resolved)
    return {"schema_version": 1, "config": resolved.to_dict(),
            "planned_cases": len(resolved.workload.cases),
            "max_complete_trials": len(resolved.workload.cases) * resolved.trials,
            "execution": "fresh_process", "plugins": "trusted_local_code",
            "model_calls_executed": 0}


def _validate_shared_source(spec: SimulationSpec) -> None:
    if spec.shared_t1_dir is None:
        return
    sdk_result = Path(spec.shared_t1_dir) / "evaluation.json"
    if not sdk_result.is_file():
        return  # Original native sources are checked by the unchanged reference loader.
    baseline = SimulationResult.from_path(sdk_result).require_complete().to_dict().get("config", {})
    if baseline.get("memory", {}).get("name") != "retry" or baseline.get("memory", {}).get("factory") is not None:
        raise ValueError("shared_t1_dir must identify a completed memory-free Retry run")
    current = spec.to_dict()
    for field in ("model", "benchmark", "trials", "max_environment_steps", "runtime_env", "require_provider_usage"):
        previous, requested = baseline.get(field), current[field]
        if field == "benchmark":
            previous, requested = comparable_workload(previous), comparable_workload(requested)
        if previous != requested:
            raise ValueError(f"shared first-trial source has different {field}")


def _guard_output(path: Path) -> None:
    source_root = Path(__file__).resolve().parents[2]
    candidates = [Path(__file__).resolve().parent]
    if (source_root / "pyproject.toml").is_file() and (source_root / "src" / "vrlbench").is_dir():
        candidates.extend(source_root / p for p in ("src", "docs", "tests", "examples", "scripts", ".git"))
        if path == source_root:
            raise ValueError("output_dir must not be the repository root")
    for protected in candidates:
        if path == protected or protected in path.parents:
            raise ValueError(f"output_dir must not overwrite source or reference files: {protected}")


def _stop_worker(process: subprocess.Popen) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait()
    # A worker may exit on TERM while one of its descendants ignores TERM.
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _available_checkpoints(output: Path) -> list[dict[str, Any]]:
    """Read native snapshots after a killed worker, without importing its runtime.

    A checkpoint never upgrades a failed or timed-out evaluation to a complete
    score. The separate provider-request journal also retains model usage from
    interrupted cases whose trial records have not been committed.
    """
    found: dict[str, dict[str, Any]] = {}
    for path in sorted((output / "reference").glob("**/episodes.json")):
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict) or not isinstance(row.get("trials"), list):
                    continue
                identity = row.get("name", row.get("sample_index", row.get("episode")))
                if identity is None:
                    continue
                case = {"case_id": str(identity), "task": str(row.get("task") or "webshop"),
                        "trials": row["trials"], "resource_usage": row.get("resource_usage", {})}
                previous = found.get(case["case_id"])
                if previous is None or len(case["trials"]) > len(previous["trials"]):
                    found[case["case_id"]] = case
        except (OSError, ValueError, TypeError):
            continue
    return list(found.values())


def simulate(spec: SimulationSpec) -> SimulationResult:
    """Execute one memory system on a workload and retain its trial process.

    Inspect ``status`` or call ``require_complete()`` before comparing metrics.

    Existing nonempty output directories are rejected, never overwritten. A
    worker exception or timeout produces a failed result with available recorded costs,
    rather than a synthetic task failure. Plugins are trusted Python code, not
    sandboxed code. This call may incur model API charges on online benchmarks.
    """
    resolved = spec.resolved()
    resolved.validate()
    _validate_shared_source(resolved)
    output = Path(resolved.output_dir)
    _guard_output(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"output_dir is not empty; choose a new run directory: {output}")
    # Exclusive creation also detects two callers racing for the same run.
    request_path = output / "request.json"
    with request_path.open("x", encoding="utf-8") as handle:
        json.dump(resolved.to_dict(), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    stdout_path, stderr_path = output / "stdout.log", output / "stderr.log"
    env = os.environ.copy()
    # Preserve the caller's import roots even though the worker runs in its
    # output directory. Resolving paths is read-only and never imports plugins.
    package_root = str(Path(__file__).resolve().parent.parent)
    import_roots = [package_root]
    for entry in [*sys.path, *env.get("PYTHONPATH", "").split(os.pathsep)]:
        if isinstance(entry, str):
            import_roots.append(str(Path(entry or os.curdir).resolve()))
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(import_roots))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    start = time.monotonic()
    status, error = "complete", None
    try:
        with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
            process = subprocess.Popen([sys.executable, "-m", "vrlbench.worker", str(request_path)],
                                       env=env, stdout=out, stderr=err, cwd=output,
                                       start_new_session=os.name == "posix")
            try:
                returncode = process.wait(timeout=resolved.timeout_seconds)
            except subprocess.TimeoutExpired:
                status, error = "timeout", {"type": "EvaluationTimeout", "message": "worker exceeded timeout_seconds"}
            finally:
                # The leader may finish normally or catch its own exception
                # while browser/server helpers remain alive in this run's group.
                # Every exit path cleans only the new session created above.
                _stop_worker(process)
                returncode = process.returncode
    except OSError as exc:
        returncode = None
        status, error = "failed", {"type": type(exc).__name__, "message": str(exc)}
    worker_path = output / "worker_result.json"
    worker: dict[str, Any] = {}
    if worker_path.is_file():
        try:
            worker = json.loads(worker_path.read_text(encoding="utf-8"))
            if not isinstance(worker, dict):
                raise ValueError("worker result is not an object")
        except (ValueError, OSError) as exc:
            status, error = "failed", {"type": "InvalidWorkerResult", "message": str(exc)}
            worker = {}
    if status == "complete" and (returncode != 0 or worker.get("status") != "complete"):
        status = "failed"
        error = worker.get("error") or {"type": "WorkerError", "message": f"worker exited with code {returncode}; see stderr.log"}
    cases = worker.get("cases", [])
    if status != "complete" and isinstance(cases, list):
        # SIGTERM/timeout does not execute the worker's Python exception path.
        # Recover its native snapshots in the parent, keeping any newer records
        # that the worker did manage to return.
        if all(isinstance(case, dict) and isinstance(case.get("case_id"), str) for case in cases):
            merged = {case["case_id"]: case for case in cases}
            for case in _available_checkpoints(output):
                previous = merged.get(case["case_id"])
                if previous is None or len(case["trials"]) > len(previous.get("trials", [])):
                    merged[case["case_id"]] = case
            cases = list(merged.values())
    try:
        aggregate = aggregate_cases(cases, trials=resolved.trials, planned_cases=len(resolved.workload.cases))
    except (ValueError, TypeError, KeyError) as exc:
        status, error = "failed", {"type": "InvalidCaseRecords", "message": str(exc)}
        aggregate = aggregate_cases([], trials=resolved.trials, planned_cases=len(resolved.workload.cases))
    if status == "complete" and aggregate["counts"]["completed"] != aggregate["counts"]["planned"]:
        status, error = "failed", {"type": "IncompleteEvaluation", "message": "not all selected cases completed"}
    aggregate["resources"] = merge_provider_resources(
        aggregate["resources"], read_provider_usage(output),
        include_zero=resolved.workload.factory is None,
    )
    payload = {"schema_version": 1, "status": status, "config": resolved.to_dict(),
               **aggregate, "cases": cases, "error": error,
               "elapsed_seconds": time.monotonic() - start,
               "identity": {**worker.get("identity", {}), "sdk_version": __version__},
               "artifacts": {"request": str(request_path), "stdout": str(stdout_path),
                             "stderr": str(stderr_path), "worker_result": str(worker_path),
                             "provider_requests": str(output / JOURNAL_NAME)},
               "output_dir": str(output)}
    result = SimulationResult(payload)
    result.write(output / "evaluation.json")
    return result
