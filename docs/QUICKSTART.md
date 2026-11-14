# Quick start

## Run a native benchmark comparison

On Linux x86_64 with Python 3.12, Git, curl, and Chrome's system libraries:

```bash
python -m pip install -r src/vrlbench/requirements/runner-py312.lock.txt
python -m pip install --no-deps -e .
vrlbench setup miniwob --root external/miniwob
export OPENAI_API_KEY=your-key
vrlbench doctor --preset miniwob-visual-v1 \
  --environment external/miniwob/environment.json --model YOUR_VISION_MODEL_NAME
vrlbench batch --preset miniwob-visual-v1 \
  --environment external/miniwob/environment.json --model YOUR_VISION_MODEL_NAME \
  --methods retry reflexion --jobs 2 --output-dir runs/miniwob-visual
vrlbench compare runs/miniwob-visual/retry runs/miniwob-visual/reflexion \
  --statistics --analysis-unit task
vrlbench replay runs/miniwob-visual/retry runs/miniwob-visual/reflexion \
  --output runs/miniwob-visual/replay.html
```

Replace the model placeholder; use `--base-url` on `doctor` and `batch` for a
compatible endpoint. Credentials remain in the shell, outside the configuration.
Setup fetches the pinned Synapse revision, applies the released runtime overlay,
installs the locked environment dependencies in an isolated Python 3.12 venv,
and downloads Chrome/ChromeDriver 146.0.7680.165. It performs an actual native
browser reset before writing `environment.json`. Logs stay in the setup root.
The simulator's OpenHands dependencies remain separate from environment dependencies.
If matching browser binaries are already installed, pass both `--chrome PATH`
and `--chromedriver PATH` to setup. Their versions are recorded. On hosts
without Chrome's system libraries, install those through the system package
manager or use the container workflow below. Setup does not install system packages.

The installed `miniwob-visual-v1` preset selects **16 native cases: eight task families and two episode
seeds per family**, with two trials and twelve action opportunities per trial.
It includes typing, options, checkboxes, numeric feedback, and a slider.
These are original MiniWoB tasks and native grades, not replacement fixtures.
The equivalent editable config is `examples/configs/miniwob_visual.json`.
This selection verifies the workflow; expand `workload.cases` and the declared
budgets for a research evaluation. It is not the full MiniWoB task collection.
`include_screenshot: true` sends the native browser PNG alongside HTML to the
model; `capture_observations` separately controls replay recording.

Each case runs in its own fresh worker. A case's Retry run completes before
other memory conditions reuse its actual memory-free first trial. Successful
T1 cases make no additional actor calls for the other condition. Aggregate
`evaluation.json` files use the normal inspect, compare, statistics, and replay
tools. `summary.json` records coverage, per-condition results, and execution
costs including previous attempts. Provider calls copied from T1 are not billed
again in the new condition's request ledger.

To recover an interrupted batch, repeat exactly the same command with `--resume`.
Completed jobs, including completed **unsolved** cases, are reused. Infrastructure
failures get a new `attempt-*` directory; earlier attempts and logs remain intact.
Changes to the model, cases, methods, budgets, environment binding, harness source,
or local plugin source require a new output directory. Concurrency may change on
resume. Shared WebArena sites and OSWorld VMs require `--jobs 1`.

For a custom memory implementation, retain the same environment manifest and
use the [existing factory workflow](DEVELOPMENT.md#bring-your-own-memory) with
`run --environment external/miniwob/environment.json`. The built-in batch
method list currently selects released reference methods.

## Container workflow

Build the self-contained MiniWoB runner from the repository root. Docker is
required; model inference still uses your existing API endpoint.

```bash
docker build -f docker/miniwob/Dockerfile -t vrlbench:miniwob .
mkdir -p runs
docker run --rm --init --shm-size=1g \
  --user "$(id -u):$(id -g)" -e OPENAI_API_KEY \
  -v "$PWD/runs:/runs" vrlbench:miniwob \
  batch --preset miniwob-visual-v1 \
  --environment /opt/miniwob/environment.json --model YOUR_VISION_MODEL_NAME \
  --methods retry reflexion --jobs 2 --output-dir /runs/miniwob-visual
```

The image contains the pinned environment, Chrome/ChromeDriver, and locked
simulator dependencies. Build does not make model calls. Keep the resolved image
ID with results (`docker image inspect vrlbench:miniwob --format '{{.Id}}'`).
For a fixed base image, build with `--build-arg BASE_IMAGE=python@sha256:YOUR_DIGEST`.
The default Python base tag and Debian security updates can change; do not claim
that separate rebuilds are byte-identical. The Docker build context uses an
allowlist and excludes local credentials, output data, and external checkouts.

Use the same image, volume mount and user options to `inspect`, `compare`, and
`replay` the `/runs/miniwob-visual/retry` and `reflexion` directories. These
commands do not need `OPENAI_API_KEY`. Add `--resume` to the original batch
command to continue in the same output directory.

## Screenshot-only GUI evaluation

`miniwob-gui-v1` uses the same 16 native cases as `miniwob-visual-v1`, but
supplies only task text and 640 × 480 screenshots. The actor has structured
`click(x,y)`, `type(text)`, `press(key)`, `drag(x,y,to_x,to_y)`, and `stop()` tools; HTML
and XPath are excluded. Every pointer action goes through Chrome's native
input dispatcher and MiniWoB's native step and reward. Typing acts on the
focused control: the agent must locate and click an input before typing.
`stop()` ends an unsuccessful attempt through MiniWoB's native termination;
the agent cannot declare its own success or reward.
This action-budget track uses MiniWoB's native, time-independent raw reward.
The wall-clock timeout is suspended; the native raw and discounted terminal
rewards are both recorded so model latency cannot turn a completed task into
a failure. This reward mode is part of the environment identity.
Chrome session creation is attempted at most three times during a native
reset, before acting, with the same seed. The attempt count is recorded
in `environment_metadata.native_browser_start_attempts`; other reset failures
and provider failures propagate without automatic retry.

```bash
vrlbench batch --preset miniwob-gui-v1 \
  --environment external/miniwob/environment.json --model YOUR_VISION_MODEL_NAME \
  --methods retry reflexion --jobs 2 --output-dir runs/pixel-gui
vrlbench replay runs/pixel-gui/retry runs/pixel-gui/reflexion \
  --output runs/pixel-gui/replay.html
```

This is a distinct observation/action track. Use a fresh Retry baseline for
memory comparisons; the earlier HTML/XPath results are not interchangeable.
The preset has two trials and twenty action opportunities per trial. The
OpenHands GUI tool requests one named environment tool call and disables
parallel tool calls at the actor request layer;
memory-writer requests retain their own protocol settings. A supported
vision/tool-calling model is required; protocol validity does not guarantee
task success. Inspect actual screenshots, executed actions and native scores.

For GPT-6 vision models, select the Responses API for tool calling. If the
installed OpenHands/LiteLLM model catalog does not yet recognize the model,
declare its verified capabilities explicitly so the SDK preserves images:

```json
{
  "name": "gpt-6.1-terra",
  "request_options": {
    "api_mode": "responses",
    "reasoning_effort": "medium",
    "temperature": null,
    "capability_overrides": {
      "supports_vision": true,
      "supports_responses_api": true,
      "supports_reasoning_effort": true,
      "supports_sampling_params": false
    }
  }
}
```

This is the `model` section of a simulation configuration. Set
`actor.params.max_output_tokens` to 4096 to leave room for reasoning and action
arguments. SDK capability fields are kept out of native memory-writer requests;
`api_mode: "responses"` also selects Responses for those calls. Native task
evaluators retain their independently configured model and protocol.
Check your endpoint with an actual screenshot and tool call before scaling up;
listing a model in `/models` alone does not establish compatibility. See
[OpenAI's model migration guidance](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6-astra).

For long GUI trials, configure the same bounded context policy in both arms
before starting. The default OpenHands view retains all prior observations;
this can exceed endpoint image limits and accumulate large input costs.

```json
{
  "actor": {
    "name": "openhands",
    "params": {
      "max_output_tokens": 4096,
      "condenser": {
        "entrypoint": "vrlbench.actors.observation_context:preserve_latest_context",
        "kwargs": {"recent_observations": 2, "history_chars": 1000, "image_history": 2}
      }
    }
  }
}
```

This deterministic SDK view keeps the task, action history, latest real
observation, and recent rejection feedback. It retains two recent screenshot
events and short head/tail snippets of older tool observations; raw SDK events
and screenshots remain intact. No extra summarization model calls are needed.
The actor checks that the exact latest observation text and image are present
before each action request, and that vision is active when images are required.
Missing current state is an infrastructure failure, not a scored task failure.
Generic summarizers can forget the latest observation under token pressure;
the pre-request check rejects this instead of acting on an old screen.
Set `runtime_env.OH_PERSISTENCE_DIR` to a writable run-specific directory
when the host's home quota is limited.

Reflexion verifies all original screenshots, then attaches at most 8 uniformly
sampled frames, including the first and last, to its writer. The prompt lists
their original observation indices and explicitly states any omissions; it
retains the textual interaction log. Configure a different endpoint limit with
`memory.params.max_images` (integer, at least 2), and keep it fixed throughout
the comparison. Short trajectories retain every frame. Its trajectory text
defaults to a 60,000-character budget: long inputs keep indexed step order and
bounded head/tail snippets of observations and long actions, with omissions
explicitly marked. Original evidence is retained. Set
`memory.params.max_trajectory_chars` to `null` for the full original transcript
only when the endpoint can accept it. This is a bounded-input adaptation of
the reference writer for long interactive trajectories.

For reasoning models, give the writer its own sufficient output allowance:

```json
{
  "memory": {
    "name": "reflexion",
    "params": {"max_tokens": 4096, "max_images": 8, "max_trajectory_chars": 60000}
  }
}
```

## First multimodal GUI evaluation

The bundled browser smoke task runs a real Chromium page, sends native
screenshots to the OpenHands actor, executes coordinate clicks, and grades the
resulting page state. It exercises the vision, tool, reset, accounting, and
recording pipeline with two small tasks before provisioning a benchmark site
or desktop VM. Use a tool-calling vision model and Python 3.12+.
The `gui-smoke` preset is included in the installed package and selects two
cases with at most two trials and four actions per trial. The JSON equivalent
is `examples/configs/gui_smoke.json` for editing these limits.

```bash
python -m pip install -e '.[runtime,gui]'
python -m playwright install chromium
export OPENAI_API_KEY=your-key
vrlbench doctor --preset gui-smoke --model YOUR_VISION_MODEL_NAME
vrlbench run --preset gui-smoke \
  --model YOUR_VISION_MODEL_NAME --output-dir runs/my-gui-smoke
vrlbench inspect runs/my-gui-smoke
vrlbench replay runs/my-gui-smoke
```

Replace `YOUR_VISION_MODEL_NAME` with your model name in both commands.
Use `--base-url` for a compatible model endpoint.
On Linux, install the Chromium system libraries using your system's package
manager if Playwright reports missing shared libraries. The two tasks verify
installation and execution, while MiniWoB, WebShop, WebArena, and OSWorld
provide the benchmark workloads described below.

Before a new evaluation, check the selected configuration from the same shell
and Python environment that will run it:

```bash
vrlbench doctor --config examples/configs/osworld_retry.json
vrlbench run --config examples/configs/osworld_retry.json \
  --model YOUR_VISION_MODEL_NAME --api-key-env OPENAI_API_KEY \
  --output-dir runs/my-desktop-evaluation
```

`doctor` reports missing local prerequisites without starting a VM, importing
an environment, or calling a model. Exit code 1 means prerequisites are missing;
exit code 2 means the configuration is invalid. `run` uses the same evaluator,
budgets, result format, and exit codes as `simulate`. `--model`, `--base-url`,
and `--api-key-env` override model settings for `doctor`, `run`, and `simulate`. Credentials are
read from the named shell variable. Each run needs a new or empty output directory.
An online smoke run is still needed to verify endpoint vision/tool support and
native task resets after the local checks pass.

For visual WebArena agents, set `workload.params.include_screenshot` to `true`.
Each reset and action then supplies the native browser PNG alongside the text
observation. Action IDs continue to come from the text. Compare visual methods
using the same observation setting and a newly generated matching Retry source.
OSWorld already supplies native screenshots and their pixel dimensions.

The OpenHands actor preserves complete text for environment messages through
Chat and Responses serialization. Before provider admission, it checks that the
serialized current observation contains the full text and its matching image.
An explicit context policy may shorten older observations; the recorded SDK
events retain their original content.

[Back to the README](../README.md)

Choose an interactive environment, model, and memory system, then use one workflow:
`simulate` → `inspect` → `compare` → revise the memory implementation.
Built-in environments share the OpenHands actor. Run the simulator in Python
3.12+; an environment with older dependencies can run in its own interpreter
through `workload.params.environment_python`.
Set `trials` and `max_environment_steps` explicitly in every simulation
configuration. Both must be positive integers; neither has a default.

- **[MiniWoB](#use-miniwob):** browser and UI tasks, using the
  [MiniWoB configuration](../examples/configs/miniwob_retry.json).
- **[WebShop](#webshop-setup):** interactive shopping, using the
  [WebShop configuration](../examples/configs/webshop_retry.json).
- **[WebArena](#use-webarena):** tasks on self-hosted websites.
- **[OSWorld](#use-osworld):** desktop tasks in a resettable VM.
- **[Your tasks](DEVELOPMENT.md#add-a-task-environment):** connect a workload
  adapter while retaining the same memory and analysis interfaces.

Start with MiniWoB to run a real browser task without a product catalog,
search index, self-hosted website, or VM. WebShop provides a longer comparison
walkthrough below; WebArena and OSWorld use the same simulation commands after
their native environments are prepared. Model calls may incur API charges;
replaying saved results does not call a model.

Keep benchmark downloads under `external/` and evaluation outputs under
`runs/`; neither belongs in version control. Run commands from the repository
root. The evaluator does not download benchmark assets or browser binaries during a run;
`setup miniwob` is the explicit provisioning step.

<a id="miniwob"></a>

## Use MiniWoB

Use Linux, Python 3.12, Chromium or Chrome, and a
[compatible ChromeDriver](https://developer.chrome.com/docs/chromedriver/downloads/version-selection).
Install the locked Python dependencies in
[`miniwob-py312.lock.txt`](../src/vrlbench/requirements/miniwob-py312.lock.txt)
and prepare the browser and driver binaries separately. Model inference uses
a hosted API; no local model weights or GPU are needed.

Clone the repository and prepare a dedicated environment. If you already have
the checkout, start from its root and deactivate any other virtual environment:

```bash
git clone https://github.com/vrl-bench/vrlbench.git
cd vrlbench
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
python3.12 -m venv .venv-miniwob
. .venv-miniwob/bin/activate
python -m pip install -r src/vrlbench/requirements/miniwob-py312.lock.txt
python -m pip install -e '.[runtime,openhands]'

mkdir -p external
git clone https://github.com/ltzheng/Synapse.git external/Synapse
git -C external/Synapse checkout 89dbbf8f617fdfb6d97b27324719db95da6ccdff
for name in action.py environment.py instance.py; do
  cp "src/vrlbench/environments/synapse_overlay/synapse/envs/miniwob/$name" \
    "external/Synapse/synapse/envs/miniwob/$name"
done
export SYNAPSE_ROOT="$PWD/external/Synapse"
export SE_CHROME=/absolute/path/to/chrome
export SE_CHROMEDRIVER=/absolute/path/to/chromedriver
```

Replace the browser paths with your installed binaries. The adapter checks the
Synapse base commit and all three overlaid files; keep those files unchanged
after setup. MiniWoB HTML tasks load from that checkout. Set `SYNAPSE_ROOT` in
the shell used for runs.

### Run a comparison and replay it

Set the same model in these three configurations, replacing `YOUR_MODEL_NAME`.
For a nondefault endpoint, set `model.base_url` explicitly; keep credentials in
the shell variable named by `model.api_key_env`, not in a configuration file.
See [model options](DEVELOPMENT.md#use-another-model-or-transport).

| Condition | Configuration |
|---|---|
| Memory-free Retry | [`miniwob_retry.json`](../examples/configs/miniwob_retry.json) |
| Recent failure traces | [`miniwob_trace_memory.json`](../examples/configs/miniwob_trace_memory.json) |
| Reflexion | [`miniwob_reflexion.json`](../examples/configs/miniwob_reflexion.json) |

The examples select the same `click-test` case, with explicit limits of six
trials and 128 action opportunities per trial. Page recording is enabled in
all three configurations, so the first run already saves native screenshots.
Both memory conditions reuse Retry's actual first trial. Run Retry to completion
before starting them:

```bash
export OPENAI_API_KEY=your-api-key
vrlbench simulate --config examples/configs/miniwob_retry.json --plan
vrlbench simulate --config examples/configs/miniwob_retry.json
vrlbench simulate --config examples/configs/miniwob_trace_memory.json
vrlbench simulate --config examples/configs/miniwob_reflexion.json
vrlbench inspect runs/miniwob_reflexion
vrlbench replay runs/miniwob_retry runs/miniwob_trace_memory runs/miniwob_reflexion \
  --output runs/miniwob-replay.html
vrlbench compare runs/miniwob_retry runs/miniwob_trace_memory runs/miniwob_reflexion
```

Open `runs/miniwob-replay.html` to step through the recorded pages, actions,
memory, and budgets. `--plan` checks the declaration without starting an
environment or calling a model; a completed simulation reports `status: complete`.
Use new output directories when rerunning a comparison.

The task may succeed on its first trial, in which case all conditions stop
without generating memory. For memory experiments, select a broader set of
tasks and episodes in all three configurations. Keep the model, cases, reset
rules, budgets, and recording settings matched, and generate a new Retry source
after changing them.

To try your own implementation, continue with
[Bring your own memory](DEVELOPMENT.md#bring-your-own-memory): keep the same
configuration and commands, replacing only the memory binding and run paths.

## Use WebShop

This walkthrough compares Retry, recent failure traces, and Reflexion on two
shopping goals. Preparing the environment requires the full product catalog
and a search index even when you select only a few goals.

### 1. Install the simulator

Use Python 3.12+ for OpenHands and Python 3.10 with Java 11 for the WebShop
environment. The simulator connects to that environment through a subprocess.
Model inference uses hosted APIs; no local model weights or GPU are required.

```bash
git clone https://github.com/vrl-bench/vrlbench.git
cd vrlbench
python3.12 -m venv .venv-simulator
. .venv-simulator/bin/activate
python -m pip install -e '.[runtime,openhands]'
vrlbench --help
```

For configuration plans, saved-run comparison, and inspection only,
`pip install -e .` is sufficient.
The `runtime` extra does not install benchmark assets or browsers.

<a id="5-prepare-an-online-environment"></a>
<a id="webshop-setup"></a>
<a id="webshop"></a>

### 2. Prepare WebShop

Run the following commands from the repository root. Keep benchmark downloads
under `external/` and evaluation outputs under `runs/`; neither belongs in
version control.

| Environment | Runtime | Dependency lock |
|---|---|---|
| WebShop | Python 3.10; Java 11; full product catalog and search index | [`webshop-py310.lock.txt`](../src/vrlbench/requirements/webshop-py310.lock.txt) |

Ensure `java -version` reports Java 11 in the active shell. WebShop also needs
`git`, `curl`, and the native runtime libraries checked by its setup script.

```bash
python3.10 -m venv .venv-webshop
. .venv-webshop/bin/activate
python -m pip install -e '.[runtime]'
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
1,181,430 products. Allow about 20 GiB for the catalog, converted documents,
and index, plus space for Python dependencies and run outputs. Download and
indexing time depend on the network and host; both finish before model calls
begin. Setup refuses conflicting tracked changes or an unmarked existing index
rather than replacing them silently.

Setup is complete when the command exits successfully after
`[webshop-contract] post-index dependency checks passed` and prints the
`WEBSHOP_ROOT` export. To check an already prepared installation without
rebuilding its index:

```bash
python -m vrlbench.environments.webshop_dependency \
  --root "$WEBSHOP_ROOT" --phase post-index
```

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

### 3. Configure the comparison

The supplied WebShop configurations compare:

| Condition | What reaches a later attempt | Configuration |
|---|---|---|
| Retry | No cross-trial memory | [`webshop_retry.json`](../examples/configs/webshop_retry.json) |
| Recent failure traces | Up to two complete failed traces | [`webshop_trace_memory.json`](../examples/configs/webshop_trace_memory.json) |
| Reflexion | The reference method's failure reflections | [`webshop_reflexion.json`](../examples/configs/webshop_reflexion.json) |

The configurations select the same two goals, six trials per goal, and at most
15 actor steps per trial. The recent-trace implementation is also an editable
starting point for your own memory system.

Set `workload.params.environment_python` to the absolute path of
`.venv-webshop/bin/python` in every compared configuration, and reactivate
`.venv-simulator` before running `vrlbench`. The two environments must remain
separate; do not install OpenHands into WebShop's Python 3.10 environment.

These example configurations already set
`workload.params.capture_observations` to `true`, so their first runs save
native pages for the [step-by-step viewer](#inspect-and-compare).

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

### 4. Run Retry and the memory methods

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

### 5. Inspect and compare the results

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
The exporter also loads snapshots from a verified shared first-trial source.
Including Retry lets you inspect that run beside the memory condition.

With `workload.params.capture_observations: true`, MiniWoB saves PNGs after
reset and each executed primitive action. Official
WebShop runs save nonterminal native HTML; the viewer removes scripts and
remote resources. Terminal WebShop observations and older runs use recorded
text when no snapshot is available.
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
install the analysis extra in the simulator environment:

```bash
python -m pip install -e '.[analysis]'
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

### 6. Develop your own memory system

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

## Additional workloads

WebArena and OSWorld use the same multi-trial memory lifecycle,
interaction accounting, inspection, and online metrics. Prepare each native
environment in its own Python environment with `vrlbench[runtime]` installed.
Run the simulator from a separate Python 3.12+ environment with
`vrlbench[runtime,openhands]`, setting `workload.params.environment_python` to
the native interpreter. Their dependencies and task assets are not installed by the SDK.
The adapters have offline interface tests; native installations still need an
end-to-end check on the execution host before collecting results.

The configurations below contain paths and task identifiers to replace with
your local selection. Paths resolve from the configuration file's directory.
Run `--plan` to check the declaration without starting an environment or model.
For memory comparisons, copy a Retry configuration, change `memory` and
`output_dir`, and keep the model, cases, reset rules, and budgets fixed.
Set `shared_t1_dir` to the completed Retry run to reuse its actual first trials
and start each method's new environment interactions at T2. Reflexion uses a
domain-neutral writer adaptation here; the existing MiniWoB and WebShop prompt
routes are unchanged.
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
python3.10 -m venv .venv-webarena
. .venv-webarena/bin/activate
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
Set `workload.params.environment_python` to the absolute path of
`.venv-webarena/bin/python`, then reactivate the Python 3.12+ simulator
environment before running the commands below.
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
config paths, `params.environment_python` to the native environment's Python,
and `params.env_kwargs.path_to_vm`. Run `vrlbench` from the separate Python
3.12+ simulator environment. Other `env_kwargs` are passed
to `DesktopEnv`, including `provider_name` and `snapshot_name`. Use a
vision-capable model: the current screenshot is sent with the task text and
visible accessibility or terminal output. Actions execute through the native
`pyautogui` interface inside the VM, not on the host.
To restrict programs to direct mouse, keyboard and sleep calls, set
`params.gui_only` to `true`. This optional policy supports module and function
import aliases, such as `import pyautogui as p` or
`from pyautogui import click as tap`, and executes the admitted
program unchanged. It excludes other imports, programmatic file calls and
terminal-launch hotkeys; its source hash is recorded in the environment
identity. Omitting this option retains the native Python action interface.
Each observation includes the current PNG's native dimensions. Actions use
absolute pixels from the top-left origin; the adapter does not resize the image
or rescale the actor's coordinates.
Each response is one self-contained Python program (bare or in one closed code
block), or a standalone `DONE`, `FAIL`, or `WAIT`. A program mixed with a control
marker is rejected before execution, rather than partially executed.
The example enables accessibility text so verbal memory can retain observed
state changes; the actor also receives the current screenshot. Native XML is
formatted as a compact table using OSWorld's visible-node filter: Ubuntu nodes
must be showing and visible, with an enabled/editable/expandable/checkable state,
a name or text, nonnegative positions, and positive dimensions. The table
preserves roles, names, text, states, and screen bounds without a token or
row-count cutoff. Memory writers receive this recorded text trace, not the
screenshot bytes.
For image-heavy runs, export with `vrlbench replay RUN --external-images --output replay.html`;
keep the HTML beside its `replay.assets/` directory, which stores each original image once.

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
python examples/memory_study/run.py --output-dir runs/memory-study --trials 3 --max-environment-steps 1
python examples/memory_study/run.py --output-dir runs/memory-study-window1 --trials 3 --max-environment-steps 1 --keep-last 1
```

Each report joins configurations, memory-hook events, and trial records in
`report.md` and `report.json`. Compare `case_b` to follow how the changed
retention window affects later actions.

### Scripted lifecycle checks

```bash
python examples/memory_strategy/check.py
python examples/agent_loop/run.py
```

These check the memory write/read/reset lifecycle and method integration
with fixed traces and responses. They do not evaluate task
performance. See the [memory interface example](../examples/memory_strategy/README.md)
and [agent-loop example](../examples/agent_loop/README.md) for details.
