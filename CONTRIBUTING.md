# Contributing to swarmsim

Thanks for your interest in improving swarmsim! Bug reports, scenario contributions, docs fixes and
code changes are all welcome.

## Getting Set Up

You need Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/rsivilli/swarmsim.git
cd swarmsim
uv sync
```

## Before Opening a Pull Request

Run the same checks CI runs:

```bash
uv run ruff check .
uv run ruff format --check .
MPLBACKEND=Agg uv run pytest
for f in scenarios/*.yaml; do uv run modsim validate "$f"; done
```

If you change anything under `locust/`, also check the image and chart:

```bash
docker build -t modsim-locust:ci locust/
helm lint locust/helm-chart
helm template test locust/helm-chart --set timeline.existingConfigMap=locust-timeline
```

## Guidelines

- **Keep PRs focused.** One fix or feature per PR makes review much faster.
- **Add tests** under `tests/` for behavior changes and bug fixes.
- **Scenarios**: new files in `scenarios/` must pass `modsim validate`. Keep them fast to run, and
  note in the file what real-world event or concept they're based on, if any.
- **Dependencies**: add them with `uv add` so `uv.lock` stays in sync (CI uses `uv sync --locked`).
- **User-facing changes**: add a line under "Unreleased" in [CHANGELOG.md](CHANGELOG.md).

## Reporting Bugs and Requesting Features

Use the [issue templates](https://github.com/rsivilli/swarmsim/issues/new/choose). For bugs, the
scenario YAML and the output of `modsim version` are the most useful things to include.

Security issues should be reported privately; see [SECURITY.md](SECURITY.md).

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By participating you agree to
uphold it.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
