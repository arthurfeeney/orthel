# orthel

This is a utility library to solve n-dimensional PDEs of the form

$$
\sum_{i=1}^{d} a_i \frac{\partial^2 u}{\partial x_i^2} + c\,u = f
\quad \text{on } \Omega = [0, L_1) \times \cdots \times [0, L_d),
$$

This includes the Poisson ($a_i = 1, c=0$), screened Poisson ($c < 0$) Helmholtz ($c > 0$) equations.
The `a_i` should be constant per-axis. Variable-coefficient poisson is not supported.

This library is mainly intended to be easy to use in machine learning models. 
Everything should be compatible with PyTorch's compiler and autodiff.

The project supports autodiff through the solver's source term:

$$
x = A^{-1}f \\
\partial f = A^{-1} \partial x
$$

## Build

```sh
uv venv
source .venv/bin/activate
uv sync
```

## Test

```sh
uv run pytest python/tests/  # unit tests; python/tests/integration is excluded from collection
uv run python python/tests/integration/convergence.py  # grid-convergence studies
```

## Accuracy

Unit tests check forward-error for problems with known solutions. We also check problems without a convenient solution. 
in these cases the unit tests check the backward-error. (The tolerance is always some factor times machine epsilon.)

