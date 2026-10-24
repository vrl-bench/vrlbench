# Statistical comparison of memory systems

[README](../README.md) · [Simulation](SIMULATION.md) · [Comparison walkthrough](QUICKSTART.md#7-inspect-and-compare-online-runs)

Compare any memory methods under the same tasks, actor, trial budget and
interaction limits. The **first run is the reference**; it can be Retry,
Reflexion, or another memory method. The remaining runs are compared with it.
This analysis reads completed results without running models.

## Run a paired analysis

Install the optional analysis dependencies, then select the statistical unit:

```bash
python -m pip install -e '.[analysis]'
vrlbench compare runs/reflexion runs/ace runs/vex2 \
  --statistics --analysis-unit case --resamples 10000 --seed 0
```

In the Python 3.10 WebShop environment, retain its locked dependency stack:

```bash
python -m pip install -e '.[runtime,analysis]' \
  -c src/vrlbench/requirements/webshop-py310.lock.txt
```

This keeps WebShop's NumPy 1.22.4 and SciPy 1.13.1 while enabling the same
analysis API. The SDK also supports newer compatible NumPy/SciPy versions in
other environments; the WebShop constraints do not apply to those installs.

Use `case` for independent paired task instances, such as WebShop goals.
Use `task` when cases are grouped within task families, such as MiniWoB:

```bash
vrlbench compare runs/miniwob_retry runs/miniwob_reflexion \
  --statistics --analysis-unit task
```

The unit is required explicitly. Case analysis gives each case equal weight;
task analysis first averages cases within each task, then gives each task
equal weight. At least two units are required. The existing `compare` command without
`--statistics` retains its descriptive output and needs no analysis packages.

## Read the output

Start with the estimated success-rate gain and its paired interval. Inspect
the positive and negative pairs for regressions that an average can hide.
Conditional power then describes how sensitive the exact test is to the
observed imbalance at the available number of non-tied units.

The added `statistics` object reports, for each target relative to the reference:

- The success-rate difference in percentage points and its paired percentile
  bootstrap interval, with the analysis unit, resample count and seed.
- Positive, negative and tied paired units; an exact two-sided McNemar test
  for binary case outcomes, or an exact sign test for task-mean differences.
- Conditional power under the observed alternative, using a two-sided
  calculation. For case outcomes, let `b` be target-only successes
  and `c` reference-only successes. Conditional on `b+c` discordant pairs,
  power is the probability of rejection under `p_alt = b/(b+c)`. Task analysis
  applies the corresponding directional calculation to non-tied task means.

The interval estimates the **mean paired success difference**
(`effect_estimand`). The exact test instead asks whether a non-tied unit is
equally likely to favor either method
(`exact_test_null = positive_probability_given_non_tie=0.5`). With binary case
outcomes, these nulls coincide. With task means, they need not: a large gain on
one task can outweigh small losses on many others. The task sign-test p-value
and its power describe directional consistency, **not the significance or
power of the macro-average gain**. `power.estimand` identifies this distinction
in machine-readable output.

`power.basis` is `observed_alternative_conditional`; the test and conditioned
unit count accompany the estimate. With no discordant units, power is zero
and the alternative probability is unknown (`null`). This is post-hoc
conditional power, not independent evidence of improvement or a sample-size
planner. Intervals and tests are unadjusted across multiple comparisons, and
the selected units must support the independence assumption.

The two-sided test and alternative probabilities use SciPy's
[`binomtest`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html)
and [`binom`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html).
The rejection rule is `p_value < alpha`.
The default `alpha=0.05` produces 95% intervals.

## Python API

```python
from vrlbench import SimulationResult, analyze_results, posthoc_power_mcnemar

reference = SimulationResult.from_path("runs/reflexion/evaluation.json")
candidate = SimulationResult.from_path("runs/ace/evaluation.json")
report = analyze_results([reference, candidate], analysis_unit="case")
power = posthoc_power_mcnemar(n_disc=10, p_alt=0.8)
```

`analyze_results` reuses the complete-run, case-identity, task-assignment,
model and budget checks of `compare_results`. It does not change saved
results or the default comparison API.

## Check shared component versions

Comparisons also use recorded source hashes for custom task adapters and model
transports. Different hashes for either shared component reject the comparison;
memory and replay-policy implementations may differ intentionally.

`component_identity_check` reports the scope `recorded_non_memory_plugin_sources`
and per-component status: `verified`, `unverified` when a participating run lacks
a hash, or `not_applicable` when no such plugin is used. Legacy results without
these identities remain readable and are not presented as hash-verified. This
check covers recorded plugin entrypoint files, not all built-in code or external
benchmark dependencies.

For native WebShop runs, `environment_identity_check` separately compares the
recorded source/data/index hash. Different hashes reject the comparison; missing
identities in older results are reported as `unverified`. This permits new index
builds while keeping a paired comparison on the same prepared environment.
