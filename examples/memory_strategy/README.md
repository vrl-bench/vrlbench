# Recent failure traces

[Back to the README](../../README.md) · [Development guide](../../docs/DEVELOPMENT.md)

This strategy replays the two most recent failed trial traces. It implements
the same `MemorySystem` interface as the reference memory methods, and its
memory updates make no model calls. Use it as a trace-replay baseline or replace
its retention and context-building logic with your own implementation.

## Evaluate on tasks

Prepare the [WebShop environment and model](../../docs/QUICKSTART.md#webshop-setup),
then complete the matched Retry run. The supplied configuration connects
[`initial_strategy.py`](initial_strategy.py) through `memory.factory.entrypoint`:

```bash
vrlbench simulate --config examples/configs/webshop_trace_memory.json --output-dir runs/recent-traces-example
vrlbench inspect runs/recent-traces-example
vrlbench compare runs/webshop_retry runs/recent-traces-example
```

The actor executes WebShop tasks with the retained traces. The configuration's
`shared_t1_dir` points to the completed Retry source; keep its model, cases, and
budgets matched. Inspect subsequent contexts and actions alongside the paired
outcomes to understand how the memory changes behavior.

File paths in JSON configurations
are relative to the configuration file; SDK specifications resolve relative
paths from the caller's current directory. Use a new output directory for
each evaluation; existing results are never overwritten.

## Check the interface locally

After installing the SDK runtime, check write/read/reset behavior without a
model key or benchmark download:

```bash
python examples/memory_strategy/check.py
```
