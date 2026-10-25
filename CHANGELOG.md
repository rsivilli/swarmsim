# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] - 2026-10-25

### Added

- RF emitter simulation and SigMF IQ synthesis (`modsim rf`, preview): emission scheduling, link
  budgets, truth tables, protocol-like waveforms, channel effects, receiver impairments and
  direction-finding arrays. See `docs/design/rf-sigmf.md`.
- Helm chart: `timeline.existingConfigMap` to mount a timeline ConfigMap you created yourself.
  `modsim helm-values` output now uses it.
- Helm chart: post-install notes with the port-forward command, plus a warning when no timeline
  source is set.
- CONTRIBUTING, SECURITY, Code of Conduct, issue/PR templates and Dependabot configuration.
- CI job that lints and renders the Helm chart and builds the Locust image.
- `CLAUDE.md` with development commands and architecture notes for Claude Code.

### Changed

- The package version now comes from the installed package metadata instead of being hardcoded in
  three places.
- `embed_timeline.sh` sets `timeline.path` to match the input file's extension.

### Fixed

- Docker and Kubernetes instructions mounted the timeline somewhere the Locust image doesn't look.
- `modsim version` linked to the wrong repository.

### Removed

- The unimplemented `modsim export` command (use `modsim run --export --export-mode ...`).
- Deprecated `main.py` entry point (use `modsim run`).
- The scenario YAMLs in `examples/` (duplicates of `scenarios/`), `sample_output/`, and the ad-hoc
  top-level test scripts.

## [0.2.0] - 2026-10-20

- Initial public release.
