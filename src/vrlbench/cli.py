"""Simulate memory systems, inspect execution, and compare recorded outcomes."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from .simulation import plan, simulate
from .inspection import inspect_case, render_inspection
from .results import SimulationResult, compare_results
from .specs import ActorSpec, SimulationSpec
from ._version import __version__

def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _configured_spec(args: argparse.Namespace) -> SimulationSpec:
    if args.preset == "gui-smoke":
        from .environments.browser_smoke import example_spec
        spec = example_spec()
    elif args.preset == "miniwob-gui-v1":
        from .presets import miniwob_gui
        spec = miniwob_gui()
    elif args.preset == "miniwob-visual-v1":
        from .presets import miniwob_visual
        spec = miniwob_visual()
    else:
        spec = SimulationSpec.from_path(args.config)
    overrides = {name: getattr(args, name) for name in ("base_url", "api_key_env")
                 if getattr(args, name) is not None}
    if args.model is not None:
        overrides["name"] = args.model
    if overrides:
        spec = replace(spec, model=replace(spec.model, **overrides))
    if args.output_dir is not None:
        spec = replace(spec, output_dir=str(args.output_dir.resolve()))
    if args.environment is not None:
        from .setup import apply_environment
        spec = apply_environment(spec, args.environment)
    return spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VRL-Bench: A Unified Simulator for Agent Memory Systems")
    parser.add_argument("--version", action="version", version=f"VRL-Bench {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("simulate", aliases=["evaluate", "run"],
                                  help="simulate a memory system on a workload (online model calls may incur charges)")
    execute.add_argument("--output-dir", type=Path, help="override the config's run directory")
    execute.add_argument("--actor", choices=["openhands"], help="use the built-in OpenHands actor")
    execute.add_argument("--plan", action="store_true", help="validate inputs without importing plugins or calling a model")
    doctor = commands.add_parser("doctor", help="check local prerequisites without model calls or environment startup")
    doctor.add_argument("--output-dir", type=Path, help="check the intended run directory")
    batch = commands.add_parser("batch", help="run matched memory conditions with per-case workers and resumable artifacts")
    batch.add_argument("--output-dir", type=Path, required=True)
    batch.add_argument("--methods", nargs="+", default=["retry", "reflexion"])
    batch.add_argument("--jobs", type=int, default=1)
    batch.add_argument("--resume", action="store_true")
    batch.add_argument("--plan", action="store_true")
    setup = commands.add_parser("setup", help="provision a pinned native benchmark environment")
    setup.add_argument("workload", choices=["miniwob"])
    setup.add_argument("--root", type=Path, default=Path("external/miniwob"))
    setup.add_argument("--python", default=sys.executable)
    setup.add_argument("--source", help="Synapse Git URL or local checkout for offline installation")
    setup.add_argument("--chrome", type=Path)
    setup.add_argument("--chromedriver", type=Path)
    for command in (execute, doctor, batch):
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--config", type=Path)
        source.add_argument("--preset", choices=["miniwob-gui-v1", "miniwob-visual-v1", "gui-smoke"], help="bundled native tasks or integration check")
        command.add_argument("--model", help="override the configured model name")
        command.add_argument("--base-url", help="override the model endpoint")
        command.add_argument("--api-key-env", help="environment variable holding the model credential")
        command.add_argument("--environment", type=Path, help="environment.json produced by vrlbench setup")
    inspect = commands.add_parser("inspect", help="read recorded memory contexts, actions, and outcomes; no execution")
    inspect.add_argument("path", type=Path, help="evaluation.json or its run directory")
    inspect.add_argument("--case", help="case ID; omit to list available cases")
    inspect.add_argument("--format", choices=["text", "json"], default="text")
    replay = commands.add_parser("replay", help="export a step-by-step viewer from saved results; no execution")
    replay.add_argument("paths", type=Path, nargs="+", help="evaluation.json files or run directories")
    replay.add_argument("--case", help="include one recorded case ID")
    replay.add_argument("--output", type=Path, help="new HTML file; default: first run's replay.html")
    replay.add_argument("--external-images", action="store_true",
                        help="store original images once in a sibling <stem>.assets directory")
    compare = commands.add_parser("compare", help="compare completed simulation runs with matching cases, model, and budgets")
    compare.add_argument("paths", type=Path, nargs="+")
    compare.add_argument("--statistics", action="store_true", help="add paired intervals, exact tests and conditional power; requires [analysis]")
    compare.add_argument("--analysis-unit", choices=["case", "task"], help="independent cases or equally weighted task-family means")
    compare.add_argument("--resamples", type=int, help="paired-bootstrap draws (default: 10000)")
    compare.add_argument("--seed", type=int, help="bootstrap seed (default: 0)")
    compare.add_argument("--alpha", type=float, help="test level and interval error probability (default: 0.05)")
    args = parser.parse_args(argv)
    try:
        if args.command == "setup":
            from .setup import setup_miniwob
            kwargs = {"python": args.python, "chrome": args.chrome, "chromedriver": args.chromedriver}
            if args.source is not None:
                kwargs["source"] = args.source
            _print_json(setup_miniwob(args.root, **kwargs))
            return 0
        if args.command == "batch":
            from .batch import run_batch, validate_batch
            spec = _configured_spec(args)
            methods = validate_batch(spec, args.methods, args.jobs)
            if args.plan:
                value = plan(spec)
                value.update(methods=methods, jobs=args.jobs, execution="per_case_fresh_process")
                _print_json(value)
                return 0
            report = run_batch(spec, output_dir=args.output_dir, methods=args.methods,
                               jobs=args.jobs, resume=args.resume,
                               on_progress=lambda event: print(json.dumps(event), file=sys.stderr, flush=True))
            _print_json(report)
            return 0 if report["status"] == "complete" else 1
        if args.command == "doctor":
            from .diagnostics import diagnose
            report = diagnose(_configured_spec(args))
            _print_json(report)
            return 0 if report["status"] == "ready" else 1
        if args.command == "replay":
            from .replay_viewer import export_replay
            first = args.paths[0]
            root = first if first.is_dir() else first.parent
            output = args.output or root / "replay.html"
            path = export_replay(args.paths, output, case_id=args.case,
                                 external_images=args.external_images)
            print(f"Replay saved: {path}")
            assets = path.with_name(path.stem + ".assets")
            if assets.is_dir():
                print(f"Keep this asset directory beside the HTML file: {assets}")
            print("Open this HTML file in a browser. No model or environment was executed.")
            return 0
        if args.command == "inspect":
            path = args.path / "evaluation.json" if args.path.is_dir() else args.path
            view = inspect_case(SimulationResult.from_path(path), args.case)
            if args.format == "json":
                _print_json(view)
            else:
                print(render_inspection(view))
            return 0
        if args.command in {"simulate", "evaluate", "run"}:
            spec = _configured_spec(args)
            if args.actor is not None:
                params = spec.actor.params if args.actor == spec.actor.name else {}
                spec = replace(spec, actor=ActorSpec(args.actor, params))
            if args.plan:
                _print_json(plan(spec))
                return 0
            result = simulate(spec)
            _print_json({"status": result.status, "counts": result.counts,
                         "metrics": result.metrics, "output_dir": str(result.output_dir),
                         "error": result.to_dict().get("error")})
            return 0 if result.status == "complete" else 1
        if args.command == "compare":
            if args.statistics and args.analysis_unit is None:
                raise ValueError("--statistics requires an explicit --analysis-unit case or task")
            if not args.statistics and any(value is not None for value in
                                           (args.analysis_unit, args.resamples, args.seed, args.alpha)):
                raise ValueError("statistical options require --statistics")
            loaded = [SimulationResult.from_path(path / "evaluation.json" if path.is_dir() else path)
                      for path in args.paths]
            if args.statistics:
                from .statistics import analyze_results
                value = analyze_results(loaded, analysis_unit=args.analysis_unit,
                                        resamples=args.resamples if args.resamples is not None else 10_000,
                                        seed=args.seed if args.seed is not None else 0,
                                        alpha=args.alpha if args.alpha is not None else 0.05)
            else:
                value = compare_results(loaded)
            _print_json(value)
            return 0
    except KeyboardInterrupt:
        print("vrlbench: interrupted; active workers stopped and attempt records retained. Use batch --resume to continue.", file=sys.stderr)
        return 130
    except (ValueError, TypeError, OSError, KeyError, RuntimeError) as exc:
        print(f"vrlbench: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
