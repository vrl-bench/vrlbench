# Develop and integrate memory systems

[README](../README.md) · [Run a comparison](QUICKSTART.md) · [Analyze results](STATISTICS.md)

Implement a memory system, evaluate the complete implementation on tasks, then
connect it to your agent's trial loop. Memory state, update logic, retrieval,
context construction, and scheduling are yours to implement—not a fixed set of
components whose parameters are the only things you can change.

- [Implement a memory system](#bring-your-own-memory) or [an integrated method](#add-an-integrated-method).
- [Evaluate candidate code](#evaluate-a-complete-implementation) and [read task feedback](#feed-task-results-back-into-development).
- [Connect your agent loop](#connect-your-agent-loop).
- [Configure a model](#use-another-model-or-transport), [budgets](#select-tasks-and-budgets), or [a task environment](#add-a-task-environment).

## Bring your own memory

Complete the [WebShop comparison](QUICKSTART.md#run-a-memory-comparison), then
save this `MemorySystem` implementation as `my_memory.py`:

```python
from vrlbench.memory import MemorySystem


class RecentFailures(MemorySystem):
    def __init__(self, keep=2):
        self.keep = keep
        self.reset()

    def reset(self):
        self.failures = []

    def on_failure(self, trajectory, model):
        self.failures.append(trajectory)
        self.failures = self.failures[-self.keep:]

    def build_section(self):
        if not self.failures:
            return ""
        return "Previous unsuccessful attempts:\n\n" + "\n\n".join(self.failures)


def create_memory(keep=2):
    return RecentFailures(keep=keep)
```

Evaluate its factory on the same tasks, using the configured Retry run as the
shared first-trial source:

```python
from dataclasses import replace
from pathlib import Path
from vrlbench import SimulationSpec, FactorySpec, MemorySpec, simulate

base = SimulationSpec.from_path("examples/configs/webshop_retry.json")
spec = replace(
    base,
    memory=MemorySpec(
        name="recent_failures",
        factory=FactorySpec(
            entrypoint=f"{Path('my_memory.py').resolve()}:create_memory",
            kwargs={"keep": 2},
        ),
    ),
    shared_t1_dir=base.output_dir,
    output_dir="runs/recent-failures",
)
result = simulate(spec).require_complete()
print(result.metrics)
```

This runs the configured actor in WebShop with your memory implementation;
model calls may incur API charges. Keep the completed Retry directory and use
a new output directory for each candidate. Change the memory implementation
while holding the actor, tasks, and budgets fixed, then
[inspect and compare](#inspect-and-compare-results) the resulting behavior.
For a local interface check before preparing a benchmark, use the
[optional examples](QUICKSTART.md#local-interface-check).

Factories accept JSON-compatible keyword arguments and can be named as
`package.module:create_memory` or `/path/to/file.py:create_memory`. Relative file
paths in JSON resolve from the configuration file's directory. Use a distinct
`memory.name`; a custom factory cannot replace a built-in reference name.

For local files, prefer an explicit file entrypoint. Module entrypoints must be
installed or visible on the caller's import path; the worker preserves those
roots when changing to the run directory. Ordinary sibling imports work, but
conflicting statically declared local dependencies fail instead of reusing
another component's module. For larger plugins, use distinct package names and
package-relative imports.

### Memory lifecycle

| Hook | Responsibility |
|---|---|
| `reset()` | Clear per-case memory. |
| `on_trial_end(...)` | Receive the executed trace, outcome, trial index, total budget, and metadata. |
| `on_failure(...)` / `on_success(...)` | Default callbacks selected by `on_trial_end()`. |
| `read_for_trial(...)` | Build or retrieve the memory context for a trial. |
| `build_section()` | Supply the default memory representation. |
| `read_placement()` | Return `"system"` or `"user_prefix"` for context placement. |

`reset()` and `build_section()` are abstract. The default `read_for_trial()`
returns `build_section()`; override the full read/write hooks for retrieval,
selective replay, or budget-aware context construction. `on_trial_end()` receives
the canonical executed action–observation trace and, when available, the original
`actor_trajectory` separately.

The built-in loop resets memory between cases, stops on success, and invokes
the memory writer after failure only when another trial remains. A factory must
return a fresh instance for each case; do not keep case-specific state in module
globals or return a singleton. The definitions are in
[`MemorySystem`](../src/vrlbench/runtime/er/base.py) and
[`run_episode`](../src/vrlbench/runtime/loop.py).

## Add an integrated method

Use `MemoryMethod` when one implementation owns the complete next-trial decision,
including policy selection and allocation of remaining attempts. Save this
example as `my_method.py`, then replace its trace-replay logic with your method:

```python
from vrlbench import ActorContext, CaseConfig, MemoryMethod, TrialRecord


class MyMethod(MemoryMethod):
    def reset_case(self, config: CaseConfig):
        self.config = config
        self.context = ""

    def context_for_trial(self, trial_index: int) -> ActorContext:
        return ActorContext(system=self.context if trial_index else "")

    def observe_trial(self, record: TrialRecord):
        if (not record.success
                and record.trial_index + 1 < self.config.total_trials
                and record.metadata.get("policy_evidence_eligible") is not False):
            self.context = record.trajectory


def create_method():
    return MyMethod()
```

Select the factory with:

```python
from pathlib import Path
from vrlbench import FactorySpec, MemorySpec

memory = MemorySpec(
    name="my_method",
    method_factory=FactorySpec(f"{Path('my_method.py').resolve()}:create_method"),
)
```

`method_factory` and `factory` are mutually exclusive. Both create fresh
per-case state. A complete method need not split its implementation into an
extractor, retriever, and scheduler.

The simulator passes `ActorContext.system` unchanged. Its whole-method adapter
rejects nonempty `ActorContext.user`; use `MemorySystem` / `factory` for
user-prefix memory in a simulation. The standalone agent API below supports
both context channels.

The first trial is memory-free. The method receives completed traces and the
remaining trial horizon, not hidden answers. For visual actors,
`record.image_observations()` returns the original `(observation_index, image_bytes)`
pairs after checking their trial directory and SHA-256. This excludes screenshots
recorded only for replay alongside a text actor.

In the simulator, `observe_trial()`
also receives the successful trial and the final-budget failure, once each.
For a valid final failure, its actual trace is retained and
`metadata.has_future_trial` is false. A protocol closure without usable policy
evidence consumes an opportunity but arrives with an empty trace,
`policy_evidence_eligible=False`, and `trace_available=False`; do not turn it
into invented failure experience. A method can record terminal outcomes, while
the reference writers generate no new next-trial memory after success or the
last allowed trial. The example follows that update rule.

For exposure experiments, configure the optional
[Replay gate](SIMULATION.md#replay-gate) rather than rewriting memory preparation
or updates. Epsilon is one policy; a policy factory can supply another. Custom
workloads using it must pass a stable `metadata.case_id` to `runtime.run_episode`.

Use this memory binding in the matched task configuration above. The supplied
[`whole_method/method.py`](../examples/whole_method/method.py) is another complete
factory you can evaluate and revise. An
[offline lifecycle check](QUICKSTART.md#local-interface-check) is available
before running tasks.

## Reference implementations

Use `MemorySpec(name="...")` without a factory in a simulation, or
`load_method(name, client=...)` in your agent. The five public keys select the
same core implementations:

| Key | Behavior and implementation | Original work |
|---|---|---|
| `retry` | [No cross-trial memory](../src/vrlbench/runtime/er/no_er.py). | Memory-free baseline. |
| `reflexion` | [Write and replay failure reflections](../src/vrlbench/runtime/er/reflexion.py). | [Reflexion — NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html) |
| `dc_cu` | [DC-Cu*: maintain a cumulative cheatsheet](../src/vrlbench/runtime/er/dynamic_cheatsheet.py). | [Dynamic Cheatsheet — EACL 2026](https://aclanthology.org/2026.eacl-long.333/) |
| `ace` | [ACE*: update a playbook from failure reflections](../src/vrlbench/runtime/er/ace.py). | [ACE — ICLR 2026](https://arxiv.org/abs/2510.04618) |
| `vex2` | [VEX²: jointly select policy directions and allocate remaining trials](../src/vrlbench/runtime/controllers/dsst.py). | [VRL-Bench](https://arxiv.org/abs/2609.12404) |

DC-Cu* and ACE* adapt the original update mechanisms to failure-only, same-task
trial-and-error evaluation, not the original papers' cross-task settings.
Their source prompts and licenses are in
[`source_prompts/`](../src/vrlbench/runtime/er/source_prompts/); upstream revisions
are recorded in the
[prompt notices](../src/vrlbench/runtime/er/source_prompts/THIRD_PARTY_NOTICES)
and [`THIRD_PARTY_NOTICES`](../THIRD_PARTY_NOTICES).

VEX² makes its joint decision in one LM response, not through separate plugin
components. The model-based methods need model calls and a task environment.
The SDK, method loader, and CLI also accept the legacy
names `dynamic_cheatsheet_adapted`, `ace_adapted`, and `dsst`, selecting the same
implementations as the public keys.

## Evaluate a complete implementation

`evaluate_candidate()` accepts a Python file whose factory returns a complete
`MemoryMethod` or `MemorySystem`. A person, coding agent, or search program can
submit an implementation, read task feedback, revise the code, and evaluate it
again. The interface accepts changes to the method itself, not just parameters.

After completing the WebShop comparison, evaluate the supplied trace-replay
method against its Retry source:

```python
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from vrlbench import SimulationSpec, evaluate_candidate

base = SimulationSpec.from_path("examples/configs/webshop_retry.json")
root = Path("runs/development") / uuid4().hex

feedback = evaluate_candidate(
    "examples/whole_method/method.py",
    replace(
        base,
        shared_t1_dir=base.output_dir,
        output_dir=str(root / "candidate"),
    ),
    reference=base.output_dir,
)
print(feedback.status, feedback.scorable)
print(feedback.metrics)
print(feedback.regressed_case_ids)
print(feedback.evaluation_path)
```

The supplied method replays the latest failed trace as an interface example,
not a required starting algorithm. An empty memory implementation can be the
starting version. Read the returned contexts, actions, outcomes, and regressions,
revise the implementation, and compare the next version with its parent:

```python
# After editing the method, keep the workload, model, and budgets unchanged.
next_feedback = evaluate_candidate(
    "examples/whole_method/method.py",
    replace(base, shared_t1_dir=base.output_dir,
            output_dir=str(root / "next-version")),
    reference=feedback.evaluation_path,
)
print(next_feedback.comparison["paired_effects"][0]["case_deltas"])
```

The comparison reference can change from one version to the next; it is
independent of the shared first-trial source. Keep the original memory-free run
as a fixed reference for cumulative progress. For an offline introduction to
the feedback fields, use the
[local interface example](QUICKSTART.md#local-interface-check).

The default is `factory="create_method"`, returning `MemoryMethod`. For a memory
slot, set `interface="memory"`, `factory="create_memory"`, and optionally
`factory_kwargs={...}`. Only the memory binding changes: the workload, model,
trial and interaction limits, replay gate, and shared-first-trial source remain
as supplied. Provider-accounted MiniWoB and WebShop memory comparisons require
a compatible completed Retry source. WebArena and OSWorld also support
`shared_t1_dir`: keep it pointing to the original V0 run while changing the
comparison `reference` to each version's actual parent. V0 can be an empty
`MemoryMethod` implementation. The simulator reuses its recorded first-trial
actions, observations, outcome, and actor usage, then applies each candidate's
own memory update before a fresh T2 reset. It does not copy V0's memory state.
New runs record actor usage separately from post-trial memory calls; older
custom-method sources without that separation must be rerun.

A supplied `reference` can be a `SimulationResult`, run directory, or result
file. It must be complete and compatible; another memory method can be the
reference. Obvious model, environment, and budget mismatches are rejected before
candidate execution. The final comparison also checks recorded non-memory
component identities.

## Feed task results back into development

Each call returns `CandidateFeedback` and saves `feedback.json` beside
`evaluation.json`:

| Field | Meaning |
|---|---|
| `status`, `scorable` | Execution status, and whether completion, candidate identity, and any requested comparison were verified. |
| `metrics`, `counts` | Success, recovery, censored trials-to-success, and case counts. Unknown metrics remain `None`. |
| `cases` | Recorded contexts, actions, observations, and outcomes supplied by the workload. |
| `comparison` | Paired comparison with the supplied compatible reference, including per-case changes in each effect's `case_deltas`; `None` when unavailable. |
| `reference` | The supplied reference result's path (when passed by path), canonical result SHA-256, memory configuration, and recorded memory-source identities; `None` when no reference was supplied. |
| `regressed_case_ids` | Cases losing success or needing more censored trials than the reference; `None` means no valid comparison, not no regressions. |
| `resources` | Available model and environment usage, including costs retained after incomplete execution. |
| `candidate_sha256`, `candidate_identity_verified` | Starting source hash and whether the worker's recorded candidate matches it. |
| `candidate_path`, `candidate_source_path`, `output_dir`, `evaluation_path`, `error` | Original source, saved source snapshot, run and result locations, and any execution or verification error. |

Check `scorable` before ranking candidates. Failed or timed-out execution is
not a score of zero. Incomplete runs and failed identity or compatibility checks
expose `None` metrics while retaining available records and costs.

Each case delta contains both success outcomes, actual trial counts,
`success_difference` (target minus reference), and
`censored_trials_difference`. The latter uses `T + 1` for an unsolved case, as
does `AvgT@T`; a negative difference is better. An executed trial count is not
a trials-to-success value when the case remains unsolved. Reference hashes
use UTF-8 JSON with sorted keys, compact separators, and `ensure_ascii=False`,
so the same result passed in memory or loaded from a file has the same identity.

Read the feedback, change the implementation, and use a fresh output directory
for the next evaluation. The framework neither chooses the next candidate nor
imposes a composite fitness function. If task success is the objective, optimize
it while keeping recovery, trial counts, resource use, and individual regressions
visible. Screen on small development workloads, then evaluate selected methods
on held-out cases with repeated or [paired analysis](STATISTICS.md).

### Inspect and compare results

`simulate(spec)` returns `SimulationResult` and saves `evaluation.json`. Call
`result.require_complete()` before treating the run as a completed score.

```bash
vrlbench inspect runs/my-memory
vrlbench inspect runs/my-memory --case CASE_ID
vrlbench compare runs/retry runs/my-memory
```

Choose a case ID listed by the first command. Inspection reads saved contexts,
traces, outcomes, and usage; missing fields stay unknown. Comparison reports
descriptive paired differences by default, with any compatible memory method
as the first, reference run. Add `--statistics --analysis-unit case` or `task`
for intervals, exact tests, and conditional power.

For Python analysis, load `SimulationResult.from_path(...)`, then use
`compare_results([...])` with the reference first, or
`analyze_results([...], analysis_unit="case")`. See [simulation](SIMULATION.md)
for result fields and [statistics](STATISTICS.md) for analysis choices.

## Connect your agent loop

The method API can run without MiniWoB, WebShop, or the simulator. Your runner
owns tools, environment resets, and complete attempts; the method receives
completed traces and supplies the next trial's context.

### Select a method

```python
from vrlbench.methods import list_methods, load_method

print(list_methods())
method = load_method("retry")  # No client or cross-trial memory.
```

For a model-based method, supply your configured client:

```python
from openai import OpenAI
from vrlbench.methods import CaseConfig, TrialRecord, load_method, render_trajectory

client = OpenAI()  # Uses your usual OpenAI-compatible client configuration.
method = load_method("reflexion", client=client)
method.reset_case(CaseConfig(
    total_trials=6,
    model="your-model",
    metadata={"benchmark": "webshop"},
))

for trial_index in range(6):
    context = method.context_for_trial(trial_index)
    # Your runner resets the environment and executes one complete attempt.
    outcome = run_your_agent(context=context)
    trace = render_trajectory(
        task=outcome.task,
        initial_observation=outcome.initial_observation,
        steps=outcome.steps,  # [{"action": str, "observation": str}, ...]
        close_reason=outcome.close_reason,
    )
    method.observe_trial(TrialRecord(
        trial_index=trial_index,
        trajectory=trace,
        success=outcome.success,
    ))
    if outcome.success:
        break
```

`run_your_agent` and its outcome object belong to your application. Send only
executed actions and environment observations to `render_trajectory`. For
MiniWoB Reflexion, also pass the original actor decision-block text as
`TrialRecord(actor_trajectory=...)`.

The reference Reflexion writer selects MiniWoB or WebShop prompts through
`CaseConfig.metadata["benchmark"]`. This example selects the WebShop prompt;
changing the label does not adapt it to an unrelated domain. For a new domain,
implement its prompt adapter or a custom method.

### Place the returned context correctly

`ActorContext` separates system context from a user-prompt prefix:

```python
messages = [{"role": "system", "content": your_actor_system_prompt}]
if context.system:
    messages.append({"role": "system", "content": context.system})
user_prompt = (
    context.user + "\n\n" + your_actor_user_prompt
    if context.user else your_actor_user_prompt
)
messages.append({"role": "user", "content": user_prompt})
```

Use this placement in every actor call throughout the trial. Reflexion uses the
user prefix. VEX² supplies system context: pass its raw response, including
whitespace, without parsing a policy, selecting a branch, or adding a
generated-plan header. Keep ordinary actor instructions and tool definitions
unchanged across compared methods.

### Configure VEX²

Bind the stable task description and public environment mechanics once per case:

```python
method = load_method("vex2", client=client)
method.reset_case(CaseConfig(
    total_trials=6,
    model="your-model",
    task_scope="The task or task family shared across these attempts.",
    environment_rules="The public tool interface and reset/stopping rules.",
))
```

VEX² conditions one joint response on these inputs, ordered failed-trial traces,
and the remaining complete-trial budget. The API does not split that decision
into extraction, retrieval, and scheduling components. Do not put hidden answers
or task-specific solution rules in the public environment description.

### Lifecycle and API notes

- Trial indices are zero-based. Read context, execute a complete trial, then
  report its result in order. The first trial has no cross-trial context.
- Call `reset_case` for every independent case and reset the environment for
  every attempt. Report the terminal trial too, then stop on success or the
  trial limit. Reference methods do not generate a new memory update for success
  or final-budget failure.
- `context_for_trial` caches each trial's context, so repeated reads do not
  repeat reference-method model calls. Custom methods should likewise cache
  expensive reads. `snapshot()` exposes state for inspection, not checkpoint loading.
- A client must implement `chat.completions.create(**kwargs)`. `load_method()`
  is lazy: if no client is supplied, one is created only when a memory operation
  needs it. Reference-method calls are serialized while temporarily binding the
  client; the previous client and context are restored in `finally`, without
  changing `os.environ`. Do not interleave concurrent direct calls to the
  underlying reference runtime in the same process.

Keep the actor, tasks, trial and interaction limits fixed when comparing
methods, and include a memory-free Retry condition when that comparison is
needed. See [matched runs and accounting](SIMULATION.md).

### Check the API lifecycle locally

The [agent-loop example](../examples/agent_loop/README.md) exercises context
placement and failure updates with fixed traces and a local response fixture:

```bash
python examples/agent_loop/run.py
python examples/agent_loop/run.py --method vex2
```

It requires no API key or benchmark installation. Use it to check the method
lifecycle; use your actor's executed task results to evaluate memory quality.

## Use another model or transport

For the simulator, set the model and OpenAI-compatible endpoint explicitly:

```python
from vrlbench import ModelSpec

model = ModelSpec(
    name="my-model",
    base_url="https://provider.example/v1",
    api_key_env="MY_PROVIDER_API_KEY",
    timeout_seconds=120,
    request_options={"temperature": 0},
)
```

Set `MY_PROVIDER_API_KEY` in the calling environment, not JSON. Actor and built-in
memory calls share the declared transport and request options. Endpoint settings
are worker-local; they do not modify the caller's environment or cached client.

Use `request_options` for supported provider-specific fields. A `null` value
removes a field supplied by the reference caller when the provider rejects it.
Protocol fields such as `model`, `messages`, and `stream` cannot be overridden;
`n`, if supplied directly or in `extra_body`, must be the integer `1`. Declare
different response-model names in `allowed_response_models`, and test a small
task selection before a larger run.

For a custom transport, provide a factory returning an OpenAI-compatible
`client.chat.completions.create(**kwargs)`. For example, save this client factory
as `my_transport.py` and replace its client construction with your adapter:

```python
import os
from openai import OpenAI


def create_client(base_url, api_key_env, timeout_seconds=120):
    return OpenAI(
        api_key=os.environ[api_key_env],
        base_url=base_url,
        timeout=timeout_seconds,
        max_retries=0,
    )
```

Bind it with:

```python
from pathlib import Path
from vrlbench import FactorySpec, ModelSpec

model = ModelSpec(
    name="my-model",
    transport=FactorySpec(
        f"{Path('my_transport.py').resolve()}:create_client",
        kwargs={
            "base_url": "https://provider.example/v1",
            "api_key_env": "MY_PROVIDER_API_KEY",
            "timeout_seconds": 120,
        },
    ),
)
```

The worker creates the client lazily. Responses must provide completion text
and the usage fields consumed by the reference runtime. Routing actor and
memory calls through this shared transport preserves request accounting; a
custom transport owns its client timeout and retry configuration.

## Select tasks and budgets

Select cases explicitly with `WorkloadSpec`:

```python
from vrlbench import WorkloadSpec

webshop = WorkloadSpec(name="webshop", cases=[0, 1, 2])
miniwob = WorkloadSpec(
    name="miniwob",
    cases=[{"task": "click-button", "episode": 0}],
    params={"headless": True, "trial_seed_mode": "paired_trial"},
)
```

| Setting | Limit or requirement |
|---|---|
| `SimulationSpec.trials` | Required positive integer: complete attempts per case, stopping on success. |
| `max_environment_steps` | Required positive integer: action budget within each trial. |
| `workload.params.max_actor_steps` | Legacy WebShop field; `max_environment_steps` takes precedence. |
| `max_total_tokens` | Cumulative model tokens per case, across its trials. |
| `timeout_seconds` | Evaluation worker wall-clock limit, distinct from the model-call timeout. |
| `require_provider_usage` | Require provider-reported model usage for accounted runs. |
| `runtime_env` | Nonsecret worker-local settings, such as external benchmark paths. |

Keep credentials in the calling environment. Check a configuration with
`plan(spec)` or `vrlbench simulate --config ... --plan`, which does not import
plugins or call a model. With provider usage accounting enabled, non-Retry
MiniWoB and WebShop evaluations require a completed matched Retry source via
`shared_t1_dir`. Start with
[`webshop_retry.json`](../examples/configs/webshop_retry.json), preserve its
native first-trial records, and keep the model, cases, environment, and budgets
fixed when changing the memory method. The
[online walkthrough](QUICKSTART.md#5-prepare-an-online-environment) gives the setup
and command sequence.

## Add a task environment

Set `WorkloadSpec.factory` to a factory returning an object with
`run(spec, runtime)`. Use `runtime.run_episode(actor, manager, metadata=...)` to
create fresh method state per case and apply the declared model, trial budget,
token budget, and usage requirement. Do not instantiate the memory candidate
again inside the backend.

This complete source-checkout example reuses the action-dependent route task
from the local walkthrough. Save it as `my_workload.py`:

```python
from dataclasses import asdict
from examples.memory_study.fixture import RouteEnvironment, RuleActor


def make_task(case, max_environment_steps):
    if max_environment_steps != 1:
        raise ValueError("This route task permits one environment action per trial")
    environment = RouteEnvironment(case["target"])
    return RuleActor(environment), lambda trace: environment.success


class MyBenchmark:
    def run(self, spec, runtime):
        records = []
        for case in spec["benchmark"]["cases"]:
            actor, manager = make_task(case, runtime.max_environment_steps)
            episode = runtime.run_episode(
                actor, manager, metadata={"case_id": case["case_id"], "task": "route"}
            )
            records.append({
                "case_id": case["case_id"],
                "task": "route",
                "trials": [asdict(trial) for trial in episode.trials],
                "resource_usage": episode.resource_usage,
            })
        return records


def create_benchmark():
    return MyBenchmark()
```

Run it without model calls:

```python
from pathlib import Path
from vrlbench import (
    FactorySpec, MemorySpec, ModelSpec, SimulationSpec, WorkloadSpec, simulate,
)

fixture = Path("examples/memory_study/fixture.py").resolve()
spec = SimulationSpec(
    workload=WorkloadSpec(
        name="route",
        cases=[{"case_id": "route-blue", "target": "blue"}],
        factory=FactorySpec(f"{Path('my_workload.py').resolve()}:create_benchmark"),
    ),
    model=ModelSpec("rule-actor", transport=FactorySpec(f"{fixture}:create_no_model_client")),
    memory=MemorySpec("route_feedback", factory=FactorySpec(f"{fixture}:create_memory")),
    trials=3,
    max_environment_steps=1,
    require_provider_usage=False,
    output_dir="runs/my-route",
)
print(simulate(spec).require_complete().metrics)
```

For your environment, replace `make_task` and the rule actor. The actor resets
and runs a complete attempt, enforces the interaction limit, and returns the
executed trace; the manager reads the actual environment outcome. Follow the
reference actor's `set_trial_context` and trial-metadata hooks to record budget
use. The generic loop cannot infer steps in an unknown environment: the task
adapter must enforce and report them. The example's environment allows exactly
one action after each reset; success depends on that action, not the trial number.

Each returned case has this shape:

```python
{
    "case_id": "unique-case-id",
    "task": "task-family",
    "trials": [
        {
            "trial": 0,
            "success": False,
            "trajectory": "Executed actions and observations",
            "trial_metadata": {"environment_steps": 1},
            "llm_usage": {},
        },
        # More trials, up to T; stop immediately after success.
    ],
    "resource_usage": {},
}
```

The SDK validates case records and computes common metrics. Trial indices are
zero-based and contiguous; cases finish on success or after the full trial
budget. Return actual usage when available; missing usage stays unknown, not
zero. Infrastructure exceptions must remain execution failures, not ordinary
unsuccessful cases. A new environment can choose its interaction format, but
must define reset, real task outcomes, stopping, and resource accounting.

## Execution boundary

Factories execute trusted Python in a fresh worker, not in the caller during
planning. Each case receives fresh memory state; each evaluation needs a new
or empty output directory. For `evaluate_candidate()`, the starting source is
saved as `candidate_source.py` with its SHA-256. If the worker reports different
candidate source bytes, the candidate is not scorable. This identifies the
entrypoint file, not every imported dependency.

Process isolation separates runtime state and model configuration; it is not a
security sandbox. Run generated or untrusted implementations in an isolated
environment, without credentials or sensitive files and without write access to
the evaluator, task definitions, scoring rules, or budgets. Candidates must not
change the actor or use task answers. An external development driver can use the
JSON configuration, `evaluate_candidate()`, and recorded feedback without
changing the task runners or memory interface.
