<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/brand/logo-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/brand/logo-light.png">
    <img src="docs/assets/brand/logo-light.png" width="560" alt="VRL-Bench">
  </picture>
</h1>

<h3 align="center">面向智能体记忆系统的统一仿真器</h3>

<p align="center"><a href="README.md">English</a> · 简体中文</p>

<p align="center">
  <a href="#项目简介">项目简介</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="#记忆方法库">记忆方法库</a> ·
  <a href="#文档">文档</a> ·
  <a href="https://arxiv.org/abs/2609.12404">论文</a> ·
  <a href="#参与贡献">参与贡献</a>
</p>

## 项目简介

VRL-Bench 是清华大学 NASP 团队的开源系统框架，用于**开发、仿真和比较智能体记忆系统**。

<a id="memory-can-hurt"></a>

**记忆可能反而损害智能体的表现。** 我们的[研究](https://arxiv.org/abs/2609.12404)揭示了语言强化学习中的探索—利用权衡。VRL-Bench 通过多轮试错（trial and error），评估记忆如何改变智能体的后续行动和任务成功率。

记忆系统的关键不只是存储和检索，还包括保留什么经验、何时使用、怎样引导下一次尝试。开发者可以通过可控、可重复的任务执行，检验这些设计的实际效果。

**从真实任务出发，在仿真中开发和比较记忆系统，再将评估过的同一实现接回原来的真实智能体。**

<a id="仿真器"></a>

- **开发记忆系统。** 通过 [`evaluate_candidate()`](docs/DEVELOPMENT.md) 运行自己的记忆实现，根据任务指标和逐案例反馈改进策略。
- **[仿真真实任务](docs/SIMULATION.md)。** 围绕真实任务开展可重复的多轮试错实验，开发和改进记忆系统，再将评估过的同一实现接回原来的智能体。
- **控制经验暴露。** 通过 [Replay gate](docs/SIMULATION.md#replay-gate) 控制经验重放，研究利用已有经验与继续探索之间的张力。
- **公平比较记忆方法。** 在相同的多轮试错条件下衡量记忆的实际作用，并以[比预想更强的无记忆 Retry](docs/RESEARCH.md#memory-can-helpor-hurt) 为基线。
- **观察行为与表现。** 检查各轮记忆上下文、动作和结果，统计成功率、首轮失败后的恢复率、解决任务所需的尝试数与模型调用量。
- **进行统计分析。** 使用[配对置信区间、精确检验与条件功效分析](docs/STATISTICS.md)比较不同方法，评估效果差异及其不确定性。

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/simulator-overview-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/simulator-overview.svg">
    <img src="docs/assets/simulator-overview.svg" width="100%" alt="左侧的真实任务、工具与智能体接入右侧的 VRL-Bench 仿真。通过受控的多轮试错与统计分析，开发、比较和改进记忆系统，再将评估后的同一实现接回原来的真实智能体。">
  </picture>
</p>

<p align="center">
  <strong>1.0</strong> · MiniWoB + WebShop · 可复用的记忆方法库 ·
  可扩展的方法、模型与任务
</p>

## 快速开始

SDK 支持 Python 3.10 及以上版本。下面的 WebShop 流程使用 Python 3.10 和 Java 11；MiniWoB 使用独立的 Python 3.12 环境。

```bash
git clone https://github.com/vrl-bench/vrlbench.git
cd vrlbench
python3.10 -m venv .venv-webshop
. .venv-webshop/bin/activate
python -m pip install -e '.[runtime]'
```

### 运行、观察与比较

先按 [WebShop 安装指南](docs/QUICKSTART.md#webshop-setup)准备环境，在三份示例配置中设置相同的模型，并通过环境变量提供 API key。配置选择两个购物目标，每个目标最多尝试六轮。运行会实际调用模型，可能产生 API 费用。

先完成无记忆 Retry，再运行保留近期失败轨迹的方法与 Reflexion：

```bash
vrlbench simulate --config examples/configs/webshop_retry.json
vrlbench simulate --config examples/configs/webshop_trace_memory.json
vrlbench simulate --config examples/configs/webshop_reflexion.json
vrlbench inspect runs/webshop_reflexion --case sample_0000
vrlbench compare runs/webshop_retry runs/webshop_trace_memory runs/webshop_reflexion
```

记忆方法复用 Retry 已完成的首轮尝试，并保持任务与预算一致。检查各轮上下文和动作，观察不同方法怎样利用失败反馈，再比较成功率、失败恢复和解决任务所需的尝试数。通过[配对统计分析](docs/STATISTICS.md)进一步判断差异及其不确定性。

按[完整教程](docs/QUICKSTART.md)继续，或用 [`evaluate_candidate()`](docs/DEVELOPMENT.md#evaluate-a-complete-implementation)在同样的任务上评估、改进自己的记忆系统。如需先检查安装与接口，可运行[不依赖基准下载的本地示例](docs/QUICKSTART.md#local-interface-check)。

## 记忆方法库

从 [Reflexion（2023）](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html)到 [ACE*（2026）](https://arxiv.org/abs/2510.04618)，VRL-Bench 集成了多篇高水平会议论文提出的记忆策略。

方法库提供[各方法的实现、提示词和原始来源](docs/DEVELOPMENT.md#reference-implementations)，可用作研究基线，也可直接接入自己的智能体。

星号表示在仅有失败信息的多轮试错条件下进行的适配。不同任务环境的提示词适配见[方法参考](docs/DEVELOPMENT.md#reference-implementations)。

### 在自己的智能体中使用

加载记忆方法，接入智能体循环：

```python
from vrlbench import list_methods, load_method

print(list_methods())  # retry, reflexion, dc_cu, ace, vex2
memory = load_method("reflexion")
```

按[接入指南](docs/DEVELOPMENT.md#connect-your-agent-loop)连接模型客户端，传入各轮反馈，并读取下一轮的记忆上下文。

## 研究

<a id="reference-results"></a>
<a id="explorationexploitation-in-verbal-reinforcement-learning"></a>
<a id="为什么研究多轮试错"></a>

**多轮试错是强化学习（RL）中的经典研究方式，能把保留的经验、后续行动、失败恢复与最终成功联系起来。** 对同一任务进行连续尝试，让我们更容易看清单次使用或跨任务复用中不易观察的记忆影响。

通过[反思暴露控制的参数扫描实验](docs/RESEARCH.md#experience-exposure-creates-an-explorationexploitation-decision)，我们发现，更多地重放反思有时反而会降低任务成功率。这揭示了语言强化学习中经验利用与继续探索之间的权衡：智能体何时应当依靠已有经验，何时应当尝试其他方向？

<p align="center">
  <img src="docs/assets/trial-and-error.svg" width="100%" alt="一个已记录 WebShop 案例的可视化：两种条件共享同一次首轮失败，Reflexion 重放反思后直到第六轮仍未成功，而无记忆 Retry 探索了另一件商品，在第二轮成功。">
</p>

*一个已记录的 WebShop 案例：在同一次首轮失败后，Reflexion 直到第六轮仍未解决任务，而无记忆 Retry 探索了另一件商品，在第二轮成功。*

基于 VRL-Bench 的首批研究还比较了三种模型在 MiniWoB 和 WebShop 上使用不同记忆系统的表现。实验与分析见[研究介绍](docs/RESEARCH.md)和[论文](https://arxiv.org/abs/2609.12404)，也可以按[完整教程](docs/QUICKSTART.md)设计自己的比较实验。

## 文档

以下详细指南目前为英文。

- [开始使用](docs/QUICKSTART.md)：安装、运行仿真、观察行为，完成一次记忆系统比较。
- [开发记忆系统](docs/DEVELOPMENT.md)：实现、评估并根据任务反馈改进记忆方法。
- [分析结果](docs/STATISTICS.md)：比较任务效果，计算配对区间、精确检验和条件功效。

## 参与贡献

欢迎贡献记忆方法、模型接入、交互任务或分析工具。请附上可运行的示例、测试、原始方法来源和任务适配说明。

使用 VRL-Bench 开展研究时，请引用框架及实际采用的原始方法，也欢迎通过 Issues 分享论文与实现，方便其他人在此基础上继续研究。

[贡献方法](CONTRIBUTING.md) ·
[扩展接口](docs/DEVELOPMENT.md) ·
[讨论接入](https://github.com/vrl-bench/vrlbench/issues)

## 引用

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

[下载 BibTeX](CITATION.bib) · [arXiv 引用导出](https://arxiv.org/bibtex/2609.12404)

## 许可

本项目自行编写的代码采用 [MIT 许可](LICENSE)。第三方代码、提示词和基准资源仍遵循各自的原有条款，详见 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES)。
