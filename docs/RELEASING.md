# Releasing VRL-Bench

This is the maintainer checklist for a versioned GitHub Release. The proposed
first version is `0.1.0`; this document does not announce a published release.
PyPI distribution is a separate decision and is not configured by this change.

## Prepare the candidate

- [ ] Choose the release scope and record the candidate's full commit SHA.
  For the first release, review all included features rather than implying
  that they were added since an earlier tag.
- [ ] Keep `pyproject.toml` (`project.version`),
  `src/vrlbench/_version.py` (`__version__`), and `CITATION.cff` (`version`)
  equal to the release version. Use `v0.1.0` as the proposed first tag.
- [ ] Finalize [CHANGELOG.md](../CHANGELOG.md), replacing `Unreleased` with
  the version and actual publication date. Keep unreleased work separate.
- [ ] Review README links, the English and Chinese setup instructions,
  [third-party notices](../THIRD_PARTY_NOTICES), and bundled package resources.
- [ ] Record which workloads were actually exercised, their environment
  versions, and any remaining limitations. Do not equate an offline package
  check with a completed online benchmark evaluation.
- [ ] Get the candidate reviewed and merged. Require the existing
  [distribution checks](../.github/workflows/check.yml) to pass on the exact
  commit selected for release.

## Validate and build

Use a clean Linux checkout of the reviewed commit and Python 3.12, matching
the main CI job. The
following validation is offline with respect to models and benchmarks;
installing dependencies still needs network access. Use new output paths
when repeating simulations, because existing run directories are not replaced.
The commands create a fresh temporary output root for each candidate. Before building, confirm
`git status --short` is empty and record `git rev-parse HEAD`; release artifacts
must come from that committed source, not an uncommitted working tree.

```bash
python3.12 -m venv .venv-release
. .venv-release/bin/activate
python -m pip install -e '.[runtime,openhands,analysis,dev]' twine
python -m pip check
release_root="$(mktemp -d)"
python -m build --outdir "$release_root/dist"
python -m twine check "$release_root/dist/"*.whl "$release_root/dist/"*.tar.gz
export VRLBENCH_TEST_WHEEL="$(find "$release_root/dist" -maxdepth 1 -name '*.whl' -print -quit)"
export VRLBENCH_TEST_SDIST="$(find "$release_root/dist" -maxdepth 1 -name '*.tar.gz' -print -quit)"
python -m unittest discover -s tests -v
python examples/agent_loop/run.py
python examples/agent_loop/run.py --method vex2
python examples/memory_study/run.py --output-dir "$release_root/memory-study" --trials 3 --max-environment-steps 1
vrlbench simulate --config examples/configs/local_feedback.json --output-dir "$release_root/feedback"
vrlbench simulate --config examples/configs/local_retry.json --output-dir "$release_root/retry"
vrlbench compare "$release_root/retry" "$release_root/feedback" --statistics --analysis-unit case --resamples 1000
vrlbench inspect "$release_root/feedback" --case case_b --format json
git diff --check
```

- [ ] Check version consistency and record the source identity:

```bash
python - <<'PY'
from pathlib import Path
import re
import tomllib

version = tomllib.loads(Path('pyproject.toml').read_text())['project']['version']
runtime = re.search(r'^__version__ = "([^"]+)"',
                    Path('src/vrlbench/_version.py').read_text(), re.M).group(1)
citation = re.search(r'^version: (\S+)', Path('CITATION.cff').read_text(), re.M).group(1)
assert version == runtime == citation, (version, runtime, citation)
print(version)
PY
git rev-parse HEAD
git status --short
```

- [ ] Confirm there are exactly two fresh distributions for the candidate
  version, without stale builds. Use the existing CI's isolated wheel-install
  checks as well as its source tests. Those checks cover base installation,
  packaged resources, optional dependencies, and a local simulation outside
  the checkout.
- [ ] Validate the separately constrained Python 3.10 WebShop dependencies
  using the existing `webshop-sdk-dependencies` CI job. State the tested
  Python versions explicitly; metadata alone is not a compatibility test.
- [ ] Generate checksums for the two release assets after validation:

```bash
python - "$release_root/dist" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys

directory = Path(sys.argv[1])
files = sorted([*directory.glob('*.whl'), *directory.glob('*.tar.gz')])
assert len(files) == 2, 'Expected one wheel and one source distribution'
lines = [f'{sha256(path.read_bytes()).hexdigest()}  {path.name}\n' for path in files]
(directory / 'SHA256SUMS').write_text(''.join(lines))
PY
```

## Draft and publish on GitHub

- [ ] Open [a new GitHub Release](https://github.com/vrl-bench/vrlbench/releases/new)
  as a draft. Select the reviewed commit SHA as the target; verify it rather
  than accepting a moving branch's current tip.
- [ ] Set the tag and title to the agreed version, for example `v0.1.0`.
  Check prerelease/latest status deliberately. A draft is not a publication.
- [ ] Copy the finalized changelog scope and limitations into the release
  notes. Include installation instructions, the full source SHA, validation
  evidence, supported environment setup links, and the paper citation.
- [ ] Attach the validated wheel, source distribution, and `SHA256SUMS`
  from that same commit. Review the archive contents and release notes.
- [ ] Publish the reviewed draft. Verify the tag target, downloadable
  assets, checksums, and installation from the published wheel in a fresh
  environment. Do not advertise `pip install vrlbench` unless a matching
  PyPI package has separately been published and verified.
- [ ] Add the actual release URL to the English and Chinese announcement
  drafts in [the management plan](PROJECT_MANAGEMENT.md), update the README
  news, and share only through channels selected by a maintainer.

## Corrections and follow-up

Correct prose in release notes when appropriate and describe material
corrections. If an artifact is broken, explain the affected version, prepare
a patch version with matching metadata, and repeat validation. Keep published
tags tied to their original source instead of silently moving them.

After publication, triage installation reports and collect reproducible
examples for the next patch. Use the [management backlog](PROJECT_MANAGEMENT.md)
to record remaining work. Add PyPI publishing only after confirming package
ownership, the intended release channel, and its authentication setup.
