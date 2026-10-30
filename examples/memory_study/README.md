# Local memory-behavior example

Recording a failure is useful only if its later retrieval changes an action in a useful way. This local example follows the whole process:

> configure memory system + model + workload → simulate → inspect memory reads, writes, actions and feedback → compare → modify the memory system

The rule-based actor and local environment make each transition easy to inspect. This is an interface example; for language-model task evaluation, use the [WebShop walkthrough](../../docs/QUICKSTART.md#run-a-memory-comparison).

## 1. Configure and run two systems

Install from the repository root:

```bash
python -m pip install -e '.[runtime]'
vrlbench simulate --config examples/configs/local_retry.json
vrlbench simulate --config examples/configs/local_feedback.json
```

The two JSON files declare the same workload, model, trial budget and interaction budget. Only `memory` changes:

```json
{
  "name": "feedback_memory",
  "factory": {
    "entrypoint": "../memory_study/fixture.py:create_memory",
    "kwargs": {"keep_last": 2, "events_file": "memory_events.jsonl"}
  }
}
```

Each run creates `evaluation.json` under `runs/local_retry` or `runs/local_feedback`. The feedback system also writes actual memory-hook events to `memory_events.jsonl` inside its run directory. No model key is needed: the configured transport rejects accidental model calls without creating a network client. Nonempty output directories are never overwritten; use a fresh `--output-dir` to rerun.

## 2. Follow what happened, not just the score

```bash
vrlbench inspect runs/local_retry/evaluation.json --case case_b
vrlbench inspect runs/local_feedback/evaluation.json --case case_b
vrlbench compare runs/local_retry/evaluation.json runs/local_feedback/evaluation.json
```

Both systems use the same stateless rule actor. It chooses the first visible control not listed in the injected memory. It receives no trial number, method name, case ID or hidden target. Every trial resets to the same closed-route screen, presents the same three controls and permits one action. The environment checks that action against its internal target and returns feedback.

In `case_b`, both conditions first choose `amber` and see the same rejection. Retry discards that feedback and chooses `amber` again. The feedback system records the rejected control; the original loop injects its memory on the next trial. The same actor then chooses `blue`, receives another rejection and finally chooses `green`. The environment evaluates each action as it happens. `case_a` requires fewer attempts; `case_c` succeeds on the first action in both conditions.

The feedback log records the input passed to `on_trial_end`, retained values before and after that call, and the string returned by `read_for_trial`. Events use the same case ID and zero-based trial index as the run records. The loop starts with no cross-trial memory and skips writes after success or the final allowed failure.

For one report joining the configurations, actual memory events and verbatim trial records, run:

```bash
python examples/memory_study/run.py --output-dir runs/memory-study --trials 3 --max-environment-steps 1
```

Open `runs/memory-study/report.md`; `report.json` contains the same case records and SDK comparison. The script reads the same two JSON configurations and requires both budget arguments. This fixture supports exactly one action per trial, so `--max-environment-steps` must be `1`. It also works from outside the checkout when invoked by its absolute path with the installed environment.

## 3. Modify the memory system and test again

Change `memory.factory.kwargs.keep_last` in a copy of the feedback configuration and select a new output directory. Or produce another complete report:

```bash
python examples/memory_study/run.py --output-dir runs/memory-study-window1 --trials 3 --max-environment-steps 1 --keep-last 1
```

In `case_b`, keeping only one rejection makes the system forget `amber` after rejecting `blue`. The actor chooses `amber` again and exhausts its three-trial budget, whereas the two-entry memory recovers. Inspect the write and read records to follow how the retention change affects this actor's next action.

## 4. Ask the same question on real tasks

Use the same configure–execute–inspect–revise process to investigate when memory helps or harms a language agent. Replace the local workload with WebShop or your own tasks and configure the actor model. Hold cases and budgets fixed, start with Retry, reuse its shared first trial for the memory condition, and inspect the resulting contexts and action/observation traces. The [real-task walkthrough](../../docs/QUICKSTART.md#run-a-memory-comparison) explains that path.
