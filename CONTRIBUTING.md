# Contributing to VRL-Bench

Contributions to memory methods, workloads, model compatibility,
and simulation analysis are welcome. You can use and extend the methods
independently of the simulator; see the [development guide](docs/DEVELOPMENT.md).

## What to contribute

- **Memory methods:** reusable implementations of extraction, retention,
  retrieval, replay, or integrated scheduling. Use `MemoryMethod` for a complete
  next-trial method, or retain the `MemorySystem` lifecycle for replay slots.
- **Workloads:** interactive tasks with explicit resets,
  complete-trial outcomes, interaction budgets, and resource records.
- **Model compatibility:** documented settings or transports that preserve
  actor and memory usage accounting.
- **Simulation analysis:** memory-process inspection, result analysis, and examples
  that help explain when memory helps or hurts subsequent task performance.

## Develop and test

Use Python 3.12 for development:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[runtime,analysis,dev]'
python -m unittest discover -s tests -v
python examples/agent_loop/run.py
python examples/agent_loop/run.py --method vex2
python examples/memory_study/run.py --output-dir runs/contribution-study
git diff --check
```

These checks make no model calls. Online evaluations need benchmark assets
and may incur API charges.
For packaging changes, also run `python -m build --outdir output/dist` and
check that the installed package works from outside the checkout.

Tests for a new memory strategy should cover reset between cases, updates
after failure, next-trial reads, success stopping, terminal-trial notifications,
and no generation for a nonexistent next trial. Changes to execution or results should also test
error and timeout statuses, resource accounting, and compatibility checks.

## Submit a change

Include a runnable configuration, a focused test, and the commands you ran.
For experiment results, describe the changed mechanism and comparison settings,
and report task outcomes alongside resource use. The
[comparison walkthrough](docs/QUICKSTART.md) provides a worked example.

### Contribute a published or new memory method

We welcome contributions from method authors and independent implementers.
Keep the core mechanism easy to find and reuse. Include the paper citation,
original code and license, and a short description of any adaptation to the
trial-and-error setting. Preserve the method's defining behavior rather than
forcing every method into a reflection summary or separate pipeline stages.

Provide a small offline lifecycle test and an example of direct use. For a
simulation integration, also demonstrate matched tasks, actor, and trial and
interaction limits, with model calls and tokens accounted for. New methods
should use the community API and include their own runnable configurations.

If your research builds on VRL-Bench, cite the framework and the original
methods you use. Share the paper and implementation so we can link to real
community work; contributions retain their authorship and source attribution.

Record the framework version, model and environment settings, configurations,
task selection, and results for a new study.

## Keep contributions safe to share

Do not commit credentials, `.env` files, benchmark downloads, virtual
environments, or generated runs. Respect third-party licenses and document
any additional setup requirements.
