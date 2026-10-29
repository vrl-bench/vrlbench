# Quick start

[Back to the README](../README.md)

Choose an interactive environment, model, and memory system, then use one workflow:
`simulate` → `inspect` → `compare` → revise the memory implementation.

- **[MiniWoB](#use-miniwob):** browser and UI tasks, using the
  [MiniWoB configuration](../examples/configs/miniwob_retry.json).
- **[WebShop](#webshop-setup):** interactive shopping, using the
  [WebShop configuration](../examples/configs/webshop_retry.json).
- **[WebArena](#use-webarena):** tasks on self-hosted websites.
- **[OSWorld](#use-osworld):** desktop tasks in a resettable VM.
- **[Your tasks](DEVELOPMENT.md#add-a-task-environment):** connect a workload
  adapter while retaining the same memory and analysis interfaces.

The worked example below compares memory-free Retry, recent failure traces,
and Reflexion on WebShop. Inspect how memory changes later actions, compare
task outcomes, then improve your own memory system and reuse it in your agent.

This walkthrough executes real task environments and hosted models, so model
calls may incur API charges. [MiniWoB setup](#use-miniwob) and optional
[offline interface checks](#optional-offline-interface-checks) are also provided.

## 1. Install the simulator

The SDK supports Python 3.10+. This WebShop walkthrough uses Python 3.10 and
Java 11 on a Linux execution host. MiniWoB uses a separate Python 3.12 environment.
Model inference uses hosted APIs; no local model weights or GPU are required.

```bash
git clone https://github.com/vrl-bench/vrlbench.git
cd vrlbench
python3.10 -m venv .venv-webshop
. .venv-webshop/bin/activate
python -m pip install -e '.[runtime]'
vrlbench --help
```

For configuration plans, saved-run comparison, and inspection only,
`pip install -e .` is sufficient.
The `runtime` extra does not install benchmark assets or browsers.

<a id="5-prepare-an-online-environment"></a>
<a id="webshop-setup"></a>
<a id="webshop"></a>

## 2. Prepare WebShop

Run the following commands from the repository root. Keep benchmark downloads
under `external/` and evaluation outputs under `runs/`; neither belongs in
version control.

| Environment | Runtime | Dependency lock |
|---|---|---|
| WebShop | Python 3.10; Java 11; full product catalog and search index | [`webshop-py310.lock.txt`](../src/vrlbench/requirements/webshop-py310.lock.txt) |

Ensure `java -version` reports Java 11 in the active shell. WebShop also needs
`git`, `curl`, and the native runtime libraries checked by its setup script.

```bash
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
```

```bash
WEBSHOP_ROOT="$PWD/external/WebShop" bash scripts/setup_webshop.sh
export WEBSHOP_ROOT="$PWD/external/WebShop"
```

The script installs the locked Python dependencies, checks out WebShop commit
`64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd`, downloads and verifies the product
and human-goal files, and builds the complete search index. The catalog contains
1,181,430 products; allow for substantial downloads and disk use. Setup refuses
conflicting tracked changes or an unmarked existing index rather than replacing
them silently.

The full WebShop lock uses the same OpenAI SDK version as VRL-Bench's `runtime`
extra. To add paired statistics without upgrading its retrieval dependencies,
follow the [constrained analysis install](STATISTICS.md#run-a-paired-analysis).

The runtime checks the prepared checkout, data, index, Python 3.10, Java 11, and
required native packages. Compatible patch releases and unrelated build tools
do not need to match the machine that built the index. Inspect the setup
requirements without downloading data:

```bash
bash scripts/setup_webshop.sh --print-contract
```

This inspection needs the installed SDK's Python dependencies but does not run
the benchmark. Set `WEBSHOP_ROOT` explicitly in the shell used for online runs,
even when using the script's default destination.

Each index build records its own file hashes. New builds need not reproduce an
older index's bytes. Shared-first-trial runs must use the same recorded
source/data/index identity; `compare` also checks this identity when available.

## 3. Configure the comparison

The supplied WebShop configurations compare:

| Condition | What reaches a later attempt | Configuration |
|---|---|---|
| Retry | No cross-trial memory | [`webshop_retry.json`](../examples/configs/webshop_retry.json) |
| Recent failure traces | Up to two complete failed traces | [`webshop_trace_memory.json`](../examples/configs/webshop_trace_memory.json) |
| Reflexion | The reference method's failure reflections | [`webshop_reflexion.json`](../examples/configs/webshop_reflexion.json) |

The configurations select the same two goals, six trials per goal, and at most
15 actor steps per trial. The recent-trace implementation is also an editable
starting point for your own memory system.

For page snapshots in the [step-by-step viewer](#inspect-and-compare), set
`workload.params.capture_observations` to `true` in each configuration before
running. Recording is off by default; saved text observations remain available.

In all three files, replace `YOUR_MODEL_NAME` and choose your model's
`name`, `base_url`, `api_key_env`, and supported `request_options`. Set credentials
in the calling shell, not JSON files. For another credential variable, set
`model.api_key_env` to that variable's name. Use an explicit `model.base_url` for
a nondefault endpoint; do not carry an unrelated `OPENAI_BASE_URL` into a
different provider's run. See
[model and transport options](DEVELOPMENT.md#use-another-model-or-transport).
The two goals provide a small first run. Choose a larger task selection for
your study and keep it identical across methods.

Keep the following fixed:

- The actor model, endpoint, request options, task cases, and environment.
- The complete-trial and interaction budgets, reset rules, and success stopping.
- The first-trial records used by the memory-enabled conditions.

Memory persists within a case and resets between cases; the environment resets
between attempts. Memory updates after failure only when another attempt remains.

<a id="6-run-three-online-memory-conditions"></a>
<a id="run-a-memory-comparison"></a>

## 4. Run Retry and the memory methods

Check the Retry configuration without importing plugins or executing tasks:

```bash
vrlbench simulate --config examples/configs/webshop_retry.json --plan
```

After preparing WebShop, run Retry first. This command executes the configured
actor and can incur API charges:

```bash
export OPENAI_API_KEY=your-api-key
vrlbench simulate --config examples/configs/webshop_retry.json
```

Wait for Retry to report `status: complete`. Both memory conditions set
`shared_t1_dir` to that run. With provider usage accounting enabled, non-Retry
reference evaluations require this completed matched source before planning or
execution. Preserve its native trial records, not just `evaluation.json`.

Then plan and run the memory conditions:

```bash
vrlbench simulate --config examples/configs/webshop_trace_memory.json --plan
vrlbench simulate --config examples/configs/webshop_trace_memory.json
vrlbench simulate --config examples/configs/webshop_reflexion.json --plan
vrlbench simulate --config examples/configs/webshop_reflexion.json
```

Use new or empty output directories. When changing the model, cases, or budgets,
create a new matched Retry source and update both shared-source paths. For later
experiments, copy the example configurations within `examples/configs/` and set
new output directories; factory and run paths resolve from the configuration
file's directory. Keep the configuration, framework version, and environment
settings with your results.

<a id="7-inspect-and-compare-online-runs"></a>
<a id="inspect-and-compare"></a>

## 5. Inspect and compare the results

List the recorded case IDs, then inspect the same case in each condition:

```bash
vrlbench inspect runs/webshop_trace_memory
vrlbench inspect runs/webshop_retry --case sample_0000
vrlbench inspect runs/webshop_trace_memory --case sample_0000
vrlbench inspect runs/webshop_reflexion --case sample_0000
```

Use a listed case ID if you change the case selection. `inspect` reads saved
contexts, system/user memory channels, action-observation traces, outcomes,
metadata, and usage. Use `--format json` for structured output. Displayed trial
numbers start at one; stored indices are zero-based. Missing fields remain
unknown, distinct from recorded empty strings.

To walk through the saved execution in a browser:

```bash
vrlbench replay runs/webshop_retry runs/webshop_reflexion --output runs/webshop-replay.html
```

Open the generated HTML file. Select a run and case, then use the trial list,
Previous/Next, or arrow keys to move through actions. Before/After shows the
observation before an action and its resulting feedback. Memory context and
evaluation state follow the selected action; recorded internal memory state
is displayed separately from the context delivered to the actor.
For shared first trials, include the source Retry run to load its snapshots.

With `workload.params.capture_observations: true`, MiniWoB's `react_batch`
profile saves PNGs after reset and each executed primitive action. Official
WebShop runs save nonterminal native HTML; the viewer removes scripts and
remote resources. Terminal WebShop observations, older runs, and MiniWoB's
legacy real-time profile use recorded text when no snapshot is available.
Snapshots do not change actor inputs, scoring, or action budgets.

The export reads saved results and snapshots only: it makes no model calls
and does not rerun the environment. Add `--case sample_0000` to select one case.
Without `--output`, the file is saved as `replay.html` in the first run's
directory. An existing output file is never overwritten. The HTML contains
task observations and memory, so review it before sharing.

Compare task outcomes and resource use:

```bash
vrlbench compare runs/webshop_retry runs/webshop_trace_memory runs/webshop_reflexion
```

The first run is the reference. Output includes success, recovery, censored
trials-to-success, recorded resources, and paired wins/losses/ties. Unsolved
cases count as `T+1` in AvgT; unreported usage stays unknown, not zero.

`compare` requires complete, compatible runs. A timed-out or failed evaluation
is incomplete, not a zero task score. Any memory method can be the first,
reference run, so you can also compare memory methods directly.

For [paired confidence intervals, exact tests, and conditional power](STATISTICS.md),
install the analysis extra while retaining WebShop's dependency constraints:

```bash
python -m pip install -e '.[runtime,analysis]' \
  -c src/vrlbench/requirements/webshop-py310.lock.txt
vrlbench compare runs/webshop_retry runs/webshop_trace_memory runs/webshop_reflexion \
  --statistics --analysis-unit case
```

WebShop goals use independent case pairs as the analysis unit. MiniWoB uses
`--analysis-unit task` when cases are grouped within task families. The two-goal
first run exercises this workflow; expand the task set to evaluate a method's
effects and uncertainty.

Results and usage are saved in `evaluation.json`. The
[simulation reference](SIMULATION.md) defines the runtime, metrics, and status fields.

<a id="8-develop-your-own-memory-system"></a>

## 6. Develop your own memory system

Change retention, extraction, retrieval, context construction, or scheduling
without replacing the surrounding execution loop. The
[development guide](DEVELOPMENT.md) covers both a `MemorySystem` through
`memory.factory` and a complete `MemoryMethod` through `memory.method_factory`.
Retain the matched Retry source and other comparison settings when replacing
the memory binding. Preserve each implementation's configuration and records,
and report additional model calls and tokens when it uses an LM.

For an iterative development workflow, use
[`evaluate_candidate()`](DEVELOPMENT.md): submit a complete candidate file,
inspect its task metrics, recorded cases and paired feedback, modify the code,
and evaluate again in a fresh run directory. After evaluating a method on your
task family, [reuse its implementation in your own agent](DEVELOPMENT.md#connect-your-agent-loop).

<a id="miniwob"></a>

## Use MiniWoB

Prepare MiniWoB in a separate Python 3.12 environment on Linux, with Chromium
and a matching ChromeDriver. The supplied setup uses browser major version 137
and the locked dependencies in
[`miniwob-py312.lock.txt`](../src/vrlbench/requirements/miniwob-py312.lock.txt).
Install the browser and driver binaries separately.

From the repository root, deactivate another virtual environment before starting:

```bash
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
python3.12 -m venv .venv-miniwob
. .venv-miniwob/bin/activate
python -m pip install -r src/vrlbench/requirements/miniwob-py312.lock.txt
python -m pip install -e '.[runtime]'

mkdir -p external
git clone https://github.com/ltzheng/Synapse.git external/Synapse
git -C external/Synapse checkout 89dbbf8f617fdfb6d97b27324719db95da6ccdff
for name in action.py environment.py instance.py; do
  cp "src/vrlbench/environments/synapse_overlay/synapse/envs/miniwob/$name" \
    "external/Synapse/synapse/envs/miniwob/$name"
done
export SYNAPSE_ROOT="$PWD/external/Synapse"
export SE_CHROME=/absolute/path/to/chromium-137
export SE_CHROMEDRIVER=/absolute/path/to/chromedriver-137
```

Replace the browser paths with your installed binaries. The adapter checks the
Synapse base commit and all three overlaid files; keep those files unchanged
after setup. MiniWoB HTML tasks load from that dependency checkout.
`SYNAPSE_ROOT` must explicitly identify this prepared external checkout.

Set your model in [`miniwob_retry.json`](../examples/configs/miniwob_retry.json),
then plan and run the supplied `click-test` case:

```bash
export OPENAI_API_KEY=your-api-key
vrlbench simulate --config examples/configs/miniwob_retry.json --plan
vrlbench simulate --config examples/configs/miniwob_retry.json
vrlbench inspect runs/miniwob_retry
```

For memory comparisons, use the same model, tasks, and budgets, then set
`shared_t1_dir` to the completed Retry run as in the WebShop workflow. The
[development guide](DEVELOPMENT.md) covers method configuration.

## Additional workloads

WebArena and OSWorld use the same multi-trial memory lifecycle,
interaction accounting, inspection, and online metrics. Prepare each native
environment in its own Python environment, then install `vrlbench[runtime]`
there. Their dependencies and task assets are not installed by the SDK.
The adapters have offline interface tests; native installations still need an
end-to-end check on the execution host before collecting results.

The configurations below contain paths and task identifiers to replace with
your local selection. Paths resolve from the configuration file's directory.
Run `--plan` to check the declaration without starting an environment or model.
For memory comparisons, copy a Retry configuration, change `memory` and
`output_dir`, and keep the model, cases, reset rules, and budgets fixed.
These two adapters execute independent first trials and do not import
`shared_t1_dir`. Reflexion uses a domain-neutral writer adaptation here;
the existing MiniWoB and WebShop prompt routes are unchanged.
An empty or syntactically invalid response consumes one action opportunity
without executing an environment action. The actor receives the rejection
reason and continues while its per-trial budget remains.
Saved identities check the recorded task files and settings; they do not
fingerprint a website deployment or VM image. Preserve the same prepared
environment and snapshot when comparing memory systems.

### Use WebArena

Prepare self-hosted sites, task configurations, authentication state, and
site URL environment variables using the
[WebArena setup](https://github.com/web-arena-x/webarena).
The adapter uses the `libwebarena==0.0.5` environment API and Playwright
Chromium. Install environment dependencies in a dedicated environment rather
than applying the original agent's full requirements, which include an older
OpenAI client.

```bash
python -m pip install libwebarena==0.0.5 'numpy<2' 'transformers<5' 'huggingface-hub<1'
python -m playwright install chromium
python -m pip install -e '.[runtime]'
```

Some `must_include` checks use NLTK's tokenizer. Prepare its English resources
in a private directory rather than letting a grading call download them:

```bash
export NLTK_DATA="$PWD/external/nltk_data"
(
  umask 077
  python -m nltk.downloader -d "$NLTK_DATA" punkt_tab
)
```

Export the native site URL variables required by your tasks: `REDDIT`,
`SHOPPING`, `SHOPPING_ADMIN`, `GITLAB`, `WIKIPEDIA`, `MAP`, and `HOMEPAGE`.
Use your self-hosted URLs and authentication files, not public demo sites.

Set the task config paths in
[`webarena_retry.json`](../examples/configs/webarena_retry.json).
Use absolute paths for `storage_state` inside native task configurations;
Playwright resolves that field from the worker's working directory.
The `reset_factory` is a required deployment hook: replace
`your_package.environments:make_webarena_reset` with an installed factory that
returns a callable `reset(case)`. That callable must restore the websites and
their data to the task's initial state before returning; raising an exception
or returning `False` stops the run. It runs before every trial, including T1.
Reopening the browser or returning to the start URL is not a website reset.

The actor receives accessibility-tree text or HTML. The official task evaluator
determines success, not the reward indicating whether a browser action executed.
Supported evaluators are `string_match`, `url_match`, and `program_html`.
Native fuzzy matching and unachievable-task judgments use the SDK transport
and usage accounting without exposing grading prompts to the actor. Their
default model remains the upstream `gpt-4-1106-preview`; an explicit
`params.judge_model` selects another model if your endpoint does not serve it.
Keep that setting identical across compared methods.
Set `params.capture_observations: true` to record the browser viewport for
per-action replay without changing the text actor's input.

```bash
vrlbench simulate --config examples/configs/webarena_retry.json --plan
vrlbench simulate --config examples/configs/webarena_retry.json
vrlbench inspect runs/webarena_retry
```

### Use OSWorld

Prepare the native `desktop_env` package, a local VM, a clean snapshot, and task
files following the [OSWorld setup](https://github.com/xlang-ai/OSWorld).
The adapter's checked API is upstream revision
[`044bc339`](https://github.com/xlang-ai/OSWorld/tree/044bc33901cb893808e3f91ebc9a17ed8e54b44c).
Use its source checkout and official environment setup, then install
`vrlbench[runtime]` in that environment. The PyPI `desktop-env==1.0.2` package
lacks `get_file(strict=True)` and is rejected before a VM starts.
The adapter does not download VM images.

In [`osworld_retry.json`](../examples/configs/osworld_retry.json), set the task
config paths and `params.env_kwargs.path_to_vm`. Other `env_kwargs` are passed
to `DesktopEnv`, including `provider_name` and `snapshot_name`. Use a
vision-capable model: the current screenshot is sent with the task text and
visible accessibility or terminal output. Actions execute through the native
`pyautogui` interface inside the VM, not on the host.
Each observation includes the current PNG's native dimensions. Actions use
absolute pixels from the top-left origin; the adapter does not resize the image
or rescale the actor's coordinates.
Each response is one self-contained Python program (bare or in one closed code
block), or a standalone `DONE`, `FAIL`, or `WAIT`. A program mixed with a control
marker is rejected before execution, rather than partially executed.
The example enables accessibility text so verbal memory can retain observed
state changes; the actor also receives the current screenshot. Memory writers
receive the recorded text trace, not the screenshot bytes.

Every trial calls `DesktopEnv.reset(task_config=...)` to restore the snapshot
and task setup. The native `evaluate()` result determines success; `DONE`,
`FAIL`, or a terminal flag alone does not. Provider usage is recorded, but a
strict `max_total_tokens` ceiling is not supported for screenshot requests.
Set trial/step limits and a run timeout explicitly.
For a Docker deployment, explicitly check writable VM storage and host port
bindings; the native provider's defaults do not confine both to your task
directory and loopback interface.

```bash
vrlbench simulate --config examples/configs/osworld_retry.json --plan
vrlbench simulate --config examples/configs/osworld_retry.json
vrlbench inspect runs/osworld_retry
```

<a id="3-inspect-memory-and-task-behavior"></a>
<a id="2-simulate-two-memory-systems"></a>
<a id="local-interface-check"></a>

## Optional offline interface checks

These examples let you inspect interfaces without an API key or benchmark
download. Use a Python 3.10+ environment with `.[runtime]` installed and a new
output directory for each run.

### Rule-based memory example

```bash
vrlbench simulate --config examples/configs/local_retry.json
vrlbench simulate --config examples/configs/local_feedback.json
vrlbench inspect runs/local_feedback --case case_b
vrlbench compare runs/local_retry runs/local_feedback
```

This example runs a small environment and a rule-based actor. The environment
evaluates the selected actions; retaining a rejection changes what the actor
tries next. It illustrates memory reads, writes, and context injection, not
language-model performance.

<a id="4-change-the-retention-window"></a>

The [example guide](../examples/memory_study/README.md) also shows how to produce
a report and change the retention window:

```bash
python examples/memory_study/run.py --output-dir runs/memory-study
python examples/memory_study/run.py --output-dir runs/memory-study-window1 --keep-last 1
```

Each report joins configurations, memory-hook events, and trial records in
`report.md` and `report.json`. Compare `case_b` to follow how the changed
retention window affects later actions.

### Scripted lifecycle checks

```bash
vrlbench demo --output-dir runs/demo
python examples/memory_strategy/check.py
python examples/agent_loop/run.py
```

These check the trial loop, memory write/read/reset lifecycle, and method
integration with fixed traces and responses. They do not evaluate task
performance. See the [memory interface example](../examples/memory_strategy/README.md)
and [agent-loop example](../examples/agent_loop/README.md) for details.
