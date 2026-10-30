"""Configure, simulate, inspect and compare a complete local memory intervention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vrlbench import SimulationSpec, compare_results, simulate


CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def study_spec(condition: str, output: Path, *, trials: int,
               max_environment_steps: int, keep_last: int = 2) -> SimulationSpec:
    """Reuse the CLI configurations rather than maintaining another task setup."""
    if condition not in {"retry", "feedback"}:
        raise ValueError("unknown local study condition")
    value = SimulationSpec.from_path(CONFIGS / f"local_{condition}.json").to_dict()
    value.update(output_dir=str(output), trials=trials,
                 max_environment_steps=max_environment_steps)
    if condition == "feedback":
        value["memory"]["factory"]["kwargs"]["keep_last"] = keep_last
    return SimulationSpec.from_dict(value)


def public_configuration(result: Any) -> dict[str, Any]:
    value = result.to_dict()["config"]
    if "benchmark" in value:
        value["workload"] = value.pop("benchmark")
    return value


def markdown_report(report: dict[str, Any]) -> str:
    lines = ["# Study an agent memory system", "", report["scope"], "",
             "Configure memory system + model + workload → simulate → inspect reads, writes and actions "
             "→ compare → modify the memory system.", "",
             "## Controlled comparison", "", "Fixed factors:", ""]
    lines.extend(f"- {factor}" for factor in report["fixed_factors"])
    lines.extend(["", f"Changed factor: {report['changed_factor']}", "",
                  f"Reset behavior: {report['reset_behavior']}", "",
                  "The following are recorded contexts and action/observation traces, not inferred thoughts."])
    for condition, record in report["conditions"].items():
        lines.extend(["", f"## {condition}", "",
                      f"Source configuration: `{record['source_config']}`", "",
                      f"Run: [{condition}/evaluation.json]({condition}/evaluation.json)", "",
                      "Resolved configuration:", "", "```json",
                      json.dumps(record["configuration"], indent=2, sort_keys=True), "```", ""])
        for case in record["cases"]:
            lines.extend([f"### {case['case_id']}", ""])
            for trial in case["trials"]:
                outcome = "success" if trial["success"] else "failure"
                events = [event for event in record["memory_events"] or []
                          if event["case_id"] == case["case_id"] and event["trial"] == trial["trial"]]
                reads = [event for event in events if event["operation"] == "read"]
                writes = [event for event in events if event["operation"] == "write"]
                lines.extend([f"Trial {trial['trial'] + 1}: {outcome}", ""])
                if reads:
                    lines.extend(["Actual memory read:", "", "```json",
                                  json.dumps(reads, indent=2, sort_keys=True), "```", ""])
                lines.extend(["Injected context (verbatim):", "", "```text",
                              trial["context"], "```", "",
                              "Executed action/observation trace (verbatim):", "", "```json",
                              trial["trajectory"], "```", ""])
                if writes:
                    lines.extend(["Actual memory write, including retained state before/after:", "",
                                  "```json", json.dumps(writes, indent=2, sort_keys=True), "```", ""])
    lines.extend(["## Descriptive fixture counts", "",
                  "These counts explain this hand-written task, not LLM or benchmark performance.", "",
                  "| Condition | Solved cases | Executed trials | Model API calls |",
                  "| --- | ---: | ---: | ---: |"])
    for condition, record in report["conditions"].items():
        executed = sum(len(case["trials"]) for case in record["cases"])
        lines.append(f"| {condition} | {record['counts']['solved']}/{record['counts']['planned']} "
                     f"| {executed} | {record['resources']['api_calls']} |")
    lines.extend(["", "## Modify and test again", "",
                  "Change `memory.factory.kwargs.keep_last` in the feedback configuration, or run the "
                  "report script with `--keep-last 1` and a new output directory. Compare the changed "
                  "read/write events and executed actions, not only the final count.", "",
                  "The original loop does not read cross-trial memory on the first trial and does not "
                  "write after success or after the final allowed failure. A missing memory-event "
                  "log is recorded as unknown, not reconstructed from actions. The native Retry "
                  "condition has no event logger; its actual injected context is recorded above."])
    return "\n".join(lines) + "\n"


def run_study(output: Path, *, trials: int, max_environment_steps: int,
              keep_last: int = 2) -> dict[str, Any]:
    if type(trials) is not int or trials < 1 or type(keep_last) is not int or keep_last < 1:
        raise ValueError("trials and keep_last must be positive integers")
    if type(max_environment_steps) is not int or max_environment_steps != 1:
        raise ValueError("this fixture requires max_environment_steps=1")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"study output is not empty: {output}")
    results = {name: simulate(study_spec(name, output / name, trials=trials,
                                       max_environment_steps=max_environment_steps,
                                       keep_last=keep_last)).require_complete()
               for name in ("retry", "feedback")}
    memory_events = {}
    for name in results:
        path = output / name / "memory_events.jsonl"
        memory_events[name] = ([json.loads(line) for line in path.read_text().splitlines()]
                               if path.exists() else None)
    report = {
        "schema_version": 1,
        "scope": "A deterministic mechanism-teaching fixture for an agent memory system. "
                 "Its hand-written actor and task make the intervention inspectable; "
                 "the results do not measure LLM performance or predict improvements on real tasks.",
        "fixed_factors": [
            "The same three cases, public controls, initial observations and stateless rule actor.",
            f"At most {trials} fresh trials per case, one environment action per trial, success-stop.",
            "The same environment transition and feedback rules; rejected controls remain invalid within each case.",
            "Zero model API calls; the declared transport rejects accidental completion requests.",
        ],
        "changed_factor": f"Retry discards failure feedback; feedback memory retains the most recent "
                          f"{keep_last} rejected-control observations and replays them on later trials.",
        "reset_behavior": "Every trial resets the environment to the same closed route. "
                          "The actor selects from only that public observation and the original loop's "
                          "injected context, never trial indices, condition names, case IDs or hidden targets. "
                          "Only the memory slot retains failure information between trials; "
                          "a fresh memory instance is created for every case.",
        "conditions": {name: {"source_config": f"examples/configs/local_{name}.json",
                              "configuration": public_configuration(result),
                              "result_path": f"{name}/evaluation.json", "counts": result.counts,
                              "resources": result.resources, "cases": result.cases,
                              "memory_events": memory_events[name]}
                       for name, result in results.items()},
        "comparison": compare_results(list(results.values())),
    }
    with (output / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    with (output / "report.md").open("x", encoding="utf-8") as handle:
        handle.write(markdown_report(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--max-environment-steps", type=int, required=True,
                        help="action budget per trial; this fixture requires 1")
    parser.add_argument("--keep-last", type=int, default=2)
    args = parser.parse_args()
    run_study(args.output_dir, trials=args.trials,
              max_environment_steps=args.max_environment_steps, keep_last=args.keep_last)
    print(args.output_dir.resolve() / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
