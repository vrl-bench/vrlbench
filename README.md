<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/brand/logo-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/brand/logo-light.png">
    <img src="docs/assets/brand/logo-light.png" width="560" alt="VRL-Bench">
  </picture>
</h1>

<h3 align="center">A Unified Simulator for Agent Memory Systems</h3>

<p align="center">English · <a href="README.zh-CN.md">简体中文</a></p>

<p align="center">
  <a href="#about">About</a> ·
  <a href="#quick-start">Get Started</a> ·
  <a href="#memory-library">Memory Library</a> ·
  <a href="#documentation">Documentation</a> ·
  <a href="https://arxiv.org/abs/2609.12404">Paper</a> ·
  <a href="#contributing">Contribute</a>
</p>

## About

VRL-Bench is an open systems framework for **building, simulating, and comparing agent memory** from NASP, Tsinghua University.

<a id="memory-can-hurt"></a>

**Memory can hurt the agents it is meant to improve.** Our
[research](https://arxiv.org/abs/2609.12404) revealed an
exploration–exploitation trade-off in verbal reinforcement learning.
VRL-Bench uses trial and error to evaluate how memory changes an agent's actions
and task success.

An effective memory system must decide what to retain, when to use it, and how
it should guide the next attempt. VRL-Bench makes these design choices testable
through controlled, repeatable task execution.

**Start from your real tasks, develop and compare memory systems in simulation,
then reuse the evaluated implementation in the same real agent.**

<a id="the-simulator"></a>

- **Develop memory systems.** Test your implementation with
  [`evaluate_candidate()`](docs/DEVELOPMENT.md) and use task metrics and
  case-level feedback to improve it.
- **[Simulate real tasks](docs/SIMULATION.md).** Develop and improve memory systems
  through repeatable trial-and-error experiments grounded in your real tasks,
  then bring the evaluated implementation back to your agent.
- **Control experience exposure.** Use a
  [Replay gate](docs/SIMULATION.md#replay-gate) to control experience replay and study
  the tension between exploiting past experience and continued exploration.
- **Compare memory methods.** Measure what memory actually contributes under
  matched trial-and-error conditions, using
  [memory-free Retry as a surprisingly strong baseline](docs/RESEARCH.md#memory-can-helpor-hurt).
- **Inspect behavior and performance.** Examine memory contexts, actions, and
  outcomes; track task success, recovery after an initial failure, trials to
  solve, and model calls.
- **Evaluate results statistically.** Compare methods with
  [paired confidence intervals, exact tests, and conditional power](docs/STATISTICS.md)
  to assess outcome differences and uncertainty.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/simulator-overview-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/simulator-overview.svg">
    <img src="docs/assets/simulator-overview.svg" width="100%" alt="Real tasks, tools, and an agent on the left connect to VRL-Bench simulation on the right. Develop and compare memory systems through controlled trial-and-error execution and statistical analysis, then reuse the evaluated implementation in the same real agent.">
  </picture>
</p>

<p align="center">
  <strong>1.0</strong> · MiniWoB + WebShop · Reusable memory library ·
  Extensible methods, models, and workloads
</p>

## Quick start

The SDK supports Python 3.10+. The WebShop workflow below uses Python 3.10
and Java 11; MiniWoB uses a separate Python 3.12 environment.

```bash
git clone https://github.com/vrl-bench/vrlbench.git
cd vrlbench
python3.10 -m venv .venv-webshop
. .venv-webshop/bin/activate
python -m pip install -e '.[runtime]'
```

### Run, inspect, and compare

Follow the [WebShop setup](docs/QUICKSTART.md#webshop-setup) to prepare the
environment, set the same model in the three example configurations, and supply
your API key through an environment variable. The configurations select two
shopping goals, with up to six trials per goal. Runs execute the configured
models and may incur API charges.

Run memory-free Retry first, then compare recent failure traces and Reflexion:

```bash
vrlbench simulate --config examples/configs/webshop_retry.json
vrlbench simulate --config examples/configs/webshop_trace_memory.json
vrlbench simulate --config examples/configs/webshop_reflexion.json
vrlbench inspect runs/webshop_reflexion --case sample_0000
vrlbench compare runs/webshop_retry runs/webshop_trace_memory runs/webshop_reflexion
```

The memory conditions reuse Retry's completed first trial under matched cases
and budgets. Inspect the recorded contexts and actions to see how each method
uses failure feedback, then compare success, recovery, and trials to solve.
Add [paired statistics](docs/STATISTICS.md) to assess differences and uncertainty.

Continue with the [complete walkthrough](docs/QUICKSTART.md), or use
[`evaluate_candidate()`](docs/DEVELOPMENT.md#evaluate-a-complete-implementation)
to evaluate and improve your own memory system on the same tasks.
For an installation and API check without benchmark downloads, use the
[local interface example](docs/QUICKSTART.md#local-interface-check).

## Memory library

VRL-Bench integrates memory strategies from leading conferences, spanning
[Reflexion (2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html)
to [ACE* (2026)](https://arxiv.org/abs/2510.04618).

The library provides [method implementations, prompts, and original sources](docs/DEVELOPMENT.md#reference-implementations)
for use as research baselines or as memory components in your own agents.

Stars mark failure-only trial-and-error adaptations. Benchmark-specific prompt
adapters are described in the [method reference](docs/DEVELOPMENT.md#reference-implementations).

### Use a method in your agent

Load a method and connect it to your agent loop:

```python
from vrlbench import list_methods, load_method

print(list_methods())  # retry, reflexion, dc_cu, ace, vex2
memory = load_method("reflexion")
```

Follow the [integration guide](docs/DEVELOPMENT.md#connect-your-agent-loop) to
connect your model client, pass trial feedback, and retrieve memory context
for the next attempt.

## Research

<a id="why-trial-and-error"></a>
<a id="reference-results"></a>
<a id="explorationexploitation-in-verbal-reinforcement-learning"></a>

**Trial and error, a classical research setting in RL, connects retained
experience to later actions, failure recovery, and eventual success.** Repeated
attempts at the same task help expose memory effects that are hard to observe
in one-off use or cross-task experience reuse.

Our [controlled reflection-exposure experiments](docs/RESEARCH.md#experience-exposure-creates-an-explorationexploitation-decision)
show that replaying more reflection can reduce task success, revealing an
empirical exploration–exploitation trade-off in verbal reinforcement learning:
when should an agent rely on experience, and when should it keep exploring?

<p align="center">
  <img src="docs/assets/trial-and-error.svg" width="100%" alt="Visualization of a recorded WebShop case: after the same failed first trial, Reflexion replays advice and remains unsolved through trial six, while memory-free Retry explores another product and succeeds on trial two.">
</p>

*One recorded WebShop case: after the same failed first trial, Reflexion remains
unsolved through trial six, while memory-free Retry explores another product
and succeeds on trial two.*

The [paper](https://arxiv.org/abs/2609.12404) also compares memory systems across
three models on MiniWoB and WebShop. See the [research overview](docs/RESEARCH.md)
for the experiments and analyses, and the [walkthrough](docs/QUICKSTART.md) to
design your own comparison.

## Documentation

- [Get started](docs/QUICKSTART.md) — Install, run a simulation, inspect behavior,
  and complete a memory-system comparison.
- [Develop a memory system](docs/DEVELOPMENT.md) — Implement a method, evaluate it,
  and improve it from task feedback.
- [Analyze results](docs/STATISTICS.md) — Compare task outcomes with paired intervals,
  exact tests, and conditional power.

## Contributing

Contribute memory methods, model integrations, interactive workloads, or
analysis tools. Include a runnable example, tests, attribution for the original
methods, and a description of any task adaptations.

For research using VRL-Bench, cite the framework and the original methods you
use; share your paper and implementation through Issues so others can build
on the work.

[Contribute a method](CONTRIBUTING.md) ·
[Extension interfaces](docs/DEVELOPMENT.md) ·
[Discuss an integration](https://github.com/vrl-bench/vrlbench/issues)

## Citation

```bibtex
@misc{bai2026vrlbenchbenchmarkingagentscomputer,
      title={VRL-Bench: Benchmarking agents on computer control tasks under finite trial budgets},
      author={Yu Bai and Yukai Miao and Dawei Wang and Li Chen and Yanyu Ren and Yuqian Shi and Dan Li and Ying Xiong and Chengqiu Tan and Run Zhou and Li Li},
      year={2026},
      eprint={2609.12404},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2609.12404},
}
```

[Download BibTeX](CITATION.bib) · [arXiv citation export](https://arxiv.org/bibtex/2609.12404)

## License

First-party code is provided under the [MIT License](LICENSE). Third-party
code, prompts, and benchmark assets retain their original terms; see
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
