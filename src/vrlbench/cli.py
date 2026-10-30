"""Simulate memory systems, inspect execution, and compare recorded outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .simulation import plan, simulate
from .inspection import inspect_case, render_inspection
from .results import SimulationResult, compare_results
from .specs import WorkloadSpec, SimulationSpec, FactorySpec, MemorySpec, ModelSpec
from ._version import __version__

def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VRL-Bench: A Unified Simulator for Agent Memory Systems")
    parser.add_argument("--version", action="version", version=f"VRL-Bench {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("simulate", aliases=["evaluate"],
                                  help="simulate a memory system on a workload (online model calls may incur charges)")
    execute.add_argument("--config", type=Path, required=True)
    execute.add_argument("--output-dir", type=Path, help="override the config's run directory")
    execute.add_argument("--plan", action="store_true", help="validate inputs without importing plugins or calling a model")
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
    demo = commands.add_parser("demo", help="run the real trial loop with a deterministic local fixture; no API calls")
    demo.add_argument("--trials", type=int, required=True, help="maximum complete trials per case")
    demo.add_argument("--max-environment-steps", type=int, required=True,
                      help="action budget within each trial")
    demo.add_argument("--output-dir", type=Path, default=Path("runs/demo"))
    demo.add_argument("--memory-factory", help="trusted module:factory or file.py:factory to exercise")
    demo.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "replay":
            from .replay_viewer import export_replay
            first = args.paths[0]
            root = first if first.is_dir() else first.parent
            output = args.output or root / "replay.html"
            path = export_replay(args.paths, output, case_id=args.case,
                                 external_images=args.external_images)
            print(f"Replay saved: {path}")
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
        if args.command in {"simulate", "evaluate", "demo"}:
            if args.command == "demo":
                factory = FactorySpec(args.memory_factory or "vrlbench.backends:create_demo_memory").resolved(Path.cwd())
                spec = SimulationSpec(workload=WorkloadSpec("demo", cases=["demo-0", "demo-1"]),
                                      model=ModelSpec("scripted-local"),
                                      memory=MemorySpec("demo_memory", factory), trials=args.trials,
                                      max_environment_steps=args.max_environment_steps,
                                      output_dir=str(args.output_dir), require_provider_usage=False)
            else:
                spec = SimulationSpec.from_path(args.config)
                if args.output_dir is not None:
                    value = spec.to_dict()
                    value["output_dir"] = str(args.output_dir.resolve())
                    spec = SimulationSpec.from_dict(value)
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
    except (ValueError, TypeError, OSError, KeyError, RuntimeError) as exc:
        print(f"vrlbench: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
