# Changelog

All notable changes to this project are documented here.

## [0.19.0] - 2026-08-16

### Added

- Complete adaptive pipeline from discovery through repeated final validation.
- Hermes-oriented objective for recurring growing-context phases.
- Simple `run_autotune.py` entry point and offline `analyze_autotune.py` report.
- Deterministic recommendations followed by optional local-model explanation.
- Public project documentation, contribution guidance, security policy, issue
  templates, and GPU-free continuous integration.

### Changed

- Generated deployment commands now bind to `127.0.0.1` by default.
- Network-wide deployment requires explicit `--deployment-host 0.0.0.0`.
- Public project scope excludes the earlier Unsloth-specific legacy benchmark.

### Security

- Expanded Git ignore rules for models, environment files, keys, and run data.
- Documented the risks and operator responsibilities of network exposure.
