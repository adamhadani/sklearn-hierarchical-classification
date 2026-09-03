# Contributing

**Find this project useful?** Bug reports, improvement suggestions and pull requests are welcome.
Please use the issue templates (bug report, feature request) so that reports come with what is
needed to act on them, and see [SECURITY.md](./SECURITY.md) for reporting vulnerabilities privately.
Everyone taking part is expected to follow the [code of conduct](./CODE_OF_CONDUCT.md).


## Development setup

The project is managed with [uv](https://docs.astral.sh/uv/) and requires Python 3.11+.

    git clone https://github.com/promptromp/sklearn-hierarchical-classification.git
    cd sklearn-hierarchical-classification
    uv sync --dev
    uv run pre-commit install

`pre-commit` runs ruff (lint + format), mypy and the fast test suite on every commit. CI runs the
same hooks plus the full test suite across all supported Python versions, so a change that passes
`uv run pre-commit run --all-files` locally should pass CI.

The documentation is a Jekyll site under `docs/`; see the
[Development](https://promptromp.github.io/sklearn-hierarchical-classification/development) page
for previewing it locally. Hand-written pages such as the API reference must be updated together
with the code they describe.


## Pull requests

This project uses [git-flow](https://github.com/nvie/gitflow): `develop` is the integration branch
and `master` holds releases. Submit pull requests against `develop`. Every change reaches `develop`
through a pull request with passing CI, and is squash-merged. Add tests for behaviour changes; the
existing tests live under `sklearn_hierarchical_classification/tests/`.


## Releases

Versions are derived from git tags by setuptools-scm. Maintainers release by pushing a tag of the
form `X.Y.Z`, which publishes to PyPI and creates a GitHub Release automatically.


## License

By contributing you agree that your contributions are licensed under the project's
[Apache License 2.0](./LICENSE).
