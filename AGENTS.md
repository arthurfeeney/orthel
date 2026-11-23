# AGENTS.md

## Project goals

`orthel` is a PyTorch utility library for constant-coefficient elliptic PDE solvers
and orthogonal transforms. 

Code should remain easy to use in machine learning workflows, preserve
PyTorch autograd support, and work with `torch.compile` where practical.

## Repository layout

- `python/orthel/`: importable `orthel` package.
- `python/orthel/transforms/`: orthogonal transform implementations.
- `python/tests/`: unit tests collected by pytest.
- `python/tests/integration/`: grid-convergence studies not collected by pytest.
- `pyproject.toml`: uv build and pytest configuration.
- `.github/workflows/tests.yml`: CI test workflow.

The package source root is `python/`; `python/orthel` is the `orthel` module.

## Development

Use uv for environment management, dependency installation, builds, and commands:

```sh
uv sync --locked --group dev
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv build
```

Unit tests live in `python/tests/`. Integration convergence studies are excluded
from pytest and run directly:

```sh
uv run python python/tests/integration/convergence.py
```

## Testing expectations

tests for transforms and solvers should be tested in several ways
1. relative forward error against known solutions. I.e., cases where an analytical solution is known.
2. relative backwards error against challenging cases without known analytical solutions.
3. compare against scipy references when useful.
4. test different dimensions, data types, boundary types, and autograd paths.

Do not use dense matrices for production transforms or solvers unless the
operation is explicitly small or intended as a test reference.

Run the full pytest suite after package or solver changes. New features should always be tested.
