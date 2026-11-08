# Changelog

User-facing changes will be recorded here for each tagged release.
The initial `0.1.0` release is being prepared; the scope below is unreleased.
At release time, replace `Unreleased` with the version and publication date.

## Unreleased

### Initial release scope

- A shared `simulate` → `inspect` → `compare` workflow for evaluating agent
  memory under explicit trial and interaction budgets.
- Reusable Retry, Reflexion, DC-Cu, ACE, and VEX² implementations, with
  documented trial-and-error adaptations and source attribution.
- MiniWoB, WebShop, WebArena, and OSWorld workload integrations using an
  OpenHands actor, plus interfaces for custom actors, memory, and workloads.
- Saved-run replay of observations, actions, memory, and remaining budgets.
- Paired outcome analysis and resource accounting, and offline examples for
  trying the interfaces without benchmark downloads or model calls.
- Maintainer release instructions, structured issue forms, and a pull request
  checklist for reproducible contributions.

### Requirements and limitations

- The simulator with OpenHands requires Python 3.12+ and a compatible
  tool-calling model endpoint; OSWorld also requires vision. Native
  environments may require a separate interpreter, external
  assets, browsers, self-hosted services, or a resettable VM. Follow the
  [quick start](docs/QUICKSTART.md) for each workload.
- Online runs may incur model API charges. The SDK does not include benchmark
  assets, browser binaries, or VM images.
- Workload availability is distinct from published experimental coverage.
  The accompanying study's six main settings use three models on MiniWoB and
  WebShop; it does not establish that memory improves every task or model.
- First-party code is MIT licensed. Third-party code, prompts, and assets
  retain their original terms; see [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).

See [the release checklist](docs/RELEASING.md) for validation and publication.
