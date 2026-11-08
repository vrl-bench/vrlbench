# Maintainer operating plan

Maintainer working draft, as of **2026-11-09 (Asia/Shanghai)**. Priorities below
are proposed work, not delivery dates or commitments to users. Update this
snapshot after remote changes rather than treating it as live repository state.

## Baseline and evidence

- Source: [`main` at `2bc7afa76183886c112e8df877ee7c86c24c6c84`](https://github.com/vrl-bench/vrlbench/commit/2bc7afa76183886c112e8df877ee7c86c24c6c84).
- Public snapshot: 0 releases, 0 tags, 0 open issues, 0 open pull requests,
  and 0 milestones. These counts do not measure adoption or user satisfaction.
- Package metadata, runtime version and citation metadata identify `0.1.0`.
  **The first `v0.1.0` GitHub release is a proposed target; it is not published.**
- Latest source checks succeeded:
  [simulator and distributions](https://github.com/vrl-bench/vrlbench/actions/runs/41958594274/job/127019310154),
  [WebShop SDK dependencies](https://github.com/vrl-bench/vrlbench/actions/runs/41958594274/job/127019309993).
  [Website deployment](https://github.com/vrl-bench/vrlbench/actions/runs/41958601277/job/127019337504)
  also succeeded. These results apply to the recorded source, not subsequent edits.
- CI covers offline behavior and packaging. It does not establish live benchmark
  reproducibility, provider compatibility, or new research results for a release.

## Preparation completed in this change

The release checklist, unreleased changelog, bug and feature forms, and PR
template are ready for review. Local validation used Python 3.12 on macOS:
wheel/sdist builds, `twine check`, six existing distribution and installed-wheel
tests, dependency consistency, YAML/form checks, version consistency, and
relative documentation links. Full Linux CI must still run on the proposed
changes; no live benchmark evaluation was performed.

## Backlog

| Priority | Work | Status | Next step / completion evidence |
|---|---|---|---|
| P0 | Prepare first GitHub release | Ready for review | Review changelog and release runbook; validate the final candidate commit through CI. |
| P0 | Publish `v0.1.0` and release assets | Deferred remote operation | After reviewing the candidate, create the matching tag and GitHub release; verify links and asset checksums. |
| P0 | Enable useful feedback | Ready for review | Review bug and feature/integration forms plus PR guidance; merge and verify their GitHub rendering. |
| P1 | Open three contributor tasks | Drafts below; no remote issues | Check that each task is still needed, create scoped issues and invite contributors without assigning anyone. |
| P1 | Share project announcement | Draft; publication gated | Publish only after the release exists; add its verified URL and choose appropriate channels. |
| P1 | Review first-user friction | Pending feedback | Turn reproducible setup problems and confusing documentation into focused changes. |
| P1 | Decide package distribution channel | Undecided | Choose GitHub assets only or PyPI; configure ownership and trusted publishing before any PyPI upload. |

No remote issues, labels, milestones, releases, announcements, invitations,
or scheduled automations are created by this document. Local drafts become
effective only when their relevant repository changes are published.

## Working cadence

At each maintainer work session, inspect new issues and PRs and failing checks.
Separate reproducible defects, setup questions, feature proposals and method
contributions. Ask for the smallest missing reproduction detail, record the next
action, and link duplicates. Never interpret an incomplete evaluation as a zero
success score. Review logs and shared runs for credentials and private task data.

Review the backlog weekly when practical: update status and blockers, inspect
release evidence, choose the next small contributor task, and record decisions
here or in linked issues. This is a suggested maintainer habit, not a scheduled
automation or response-time SLA. Do not promise fixes or dates without an owner
and an agreed scope.

## Contributor issue drafts

These are proposed issue bodies, not existing issues. Confirm current code and
open work before publishing; omit tasks that have already been completed.

### 1. Reproduce the MiniWoB setup on a clean Linux environment

Suggested labels: `documentation`, `help wanted`.

Goal: identify setup friction in the current [quick start](QUICKSTART.md#use-miniwob).
Keep the initial contribution to a setup report and any necessary documentation
fixes; an online model evaluation is a separately agreed extension.

Acceptance criteria:

- Record the tested commit, Linux distribution, Python, browser and driver versions.
- Follow the documented environment setup and record commands, deviations and
  dependency checks; verify the MiniWoB configuration with `--plan`.
- Run an existing offline interface example and distinguish that evidence from
  successful browser execution or an online benchmark result.
- Fix any reproduced documentation mismatch and include redacted failure logs
  where relevant; no credentials, downloaded assets or generated runs in the PR.

### 2. Add a Python 3.11 base-wheel smoke check

Suggested labels: `enhancement`, `good first issue` (after confirming scope).

Goal: exercise the declared SDK Python 3.11 support separately from the
Python 3.12+ OpenHands simulator.

Acceptance criteria:

- Build and install the wheel in a fresh Python 3.11 environment outside the
  checkout, without the `openhands` extra or a source-path fallback.
- Check base-package import, the reported version and `vrlbench --help`; record
  that this verifies the SDK and CLI entry point, not an online simulator run.
- Preserve existing Python 3.12 and WebShop Python 3.10 checks; make no model calls.
- CI passes and the PR states exactly which installation surface it covers.

### 3. Detect release-version metadata drift

Suggested labels: `enhancement`, `good first issue` (after confirming scope).

Goal: catch disagreement between `pyproject.toml`, `src/vrlbench/_version.py`
and `CITATION.cff` before a tag or distribution is published.

Acceptance criteria:

- Add a small offline check using the existing development tooling.
- It passes for the current consistent values and clearly identifies the file
  when any one of the three versions differs.
- Document how maintainers run it before a release; no network or model calls.
- If release tooling already enforces this invariant, document that evidence
  and close the proposed task instead of duplicating the check.

## Announcement drafts — publish only after the release

**Do not post either draft yet.** Before publication, verify `v0.1.0`, insert
the actual published release URL, and confirm every linked setup instruction
matches the release. The descriptions below state existing framework behavior;
they do not claim that features ship today. Posting to any channel is a separate
remote action.

### English

VRL-Bench is an open framework for building, simulating and comparing agent
memory under controlled trial-and-error budgets. It provides reusable memory
methods, recorded-trial replay and paired analysis, with memory-free Retry as a
comparison baseline. We welcome setup reports, documentation improvements and
method contributions with runnable examples.

[Repository](https://github.com/vrl-bench/vrlbench) ·
[Quick start](https://github.com/vrl-bench/vrlbench/blob/main/docs/QUICKSTART.md) ·
[Contributing](https://github.com/vrl-bench/vrlbench/blob/main/CONTRIBUTING.md) ·
[Paper](https://arxiv.org/abs/2609.12404)

Release: **insert verified published release URL before posting**.

### 中文

VRL-Bench 是用于构建、仿真和比较智能体记忆的开源框架。它在受控的多轮试错预算下，
提供可复用的记忆方法、执行记录回放与配对分析，并以无记忆 Retry 作为比较基线。
欢迎反馈安装和使用问题、改进文档，或提交带可运行示例的记忆方法。

[项目仓库](https://github.com/vrl-bench/vrlbench) ·
[中文介绍](https://github.com/vrl-bench/vrlbench/blob/main/README.zh-CN.md) ·
[贡献指南](https://github.com/vrl-bench/vrlbench/blob/main/CONTRIBUTING.md) ·
[论文](https://arxiv.org/abs/2609.12404)

发布版本：**发布后填入已验证的 GitHub release 链接，再对外发送**。
