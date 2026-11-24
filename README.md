# orthel

`orthel` is a library with orthogonal transforms, interpolation, and
some elliptic PDE solvers:

1. Orthogonal transforms
    - dct, dst and their inverses 
2. Basic functional operators
    - curl, divergence, gradient
    - Helmholtz decomposition and Leray projections
3. Solvers for a class of elliptic PDEs (using the orthogonal transforms!)
    - Specifically, $\sum_{i=1}^{d} a_i \frac{\partial^2 u}{\partial x_i^2} + c\,u = f
\quad \text{on } \Omega = [0, L_1) \times \cdots \times [0, L_d),$
    - Poisson ($a = 1, c=0$), Helmholtz ($c > 0$), screened Poisson ($c < 0$).
    - Support autodiff through the source term `f`: $x = A^{-1}f \rightarrow \partial f = A^{-1} \partial x$
3. Interpolation schemes for converting between different types of grids.

The operations provided in this library are intended to be easy to plug into machine learning models. 
I.e., all operators are made to work with `torch.compile` and PyTorch's autodiff.

## Developer setup 

### Build

```sh
uv venv
source .venv/bin/activate
uv sync
```

### Test

```sh
uv run pytest python/tests/  # unit tests; python/tests/integration is excluded from collection
uv run python python/tests/integration/convergence.py  # grid-convergence studies
