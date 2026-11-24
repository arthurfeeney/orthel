"""UniformGridSolver against a sparse direct solve built from the stencils.

* Cell grid: ghost cells are eliminated. Dirichlet ``u_ghost = 2g - u_0``
  adds ``-1`` to the diagonal and ``2g / h²`` to the lift; Neumann
  ``u_ghost = u_0 + h q`` (outward ``q``) adds ``+1`` and ``q / h``.
* Node grid: every node is an unknown. Dirichlet wall nodes get identity rows
  ``u = g`` (a corner shared by two Dirichlet walls takes the later axis's
  value); Neumann wall nodes use ``u_{-1} = u_1 + 2 h q``, which doubles the
  inward neighbour and adds ``2 q / h`` to the lift.
"""

import itertools

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from orthel.boundary import Dirichlet, Neumann, Periodic
from orthel.uniform_grid_solver import UniformGridSolver

AXES = {
    "p": ("periodic", "periodic"),
    "nn": ("neumann", "neumann"),
    "dd": ("dirichlet", "dirichlet"),
    "nd": ("neumann", "dirichlet"),
    "dn": ("dirichlet", "neumann"),
}
LENGTHS = (1.1, 1.4, 0.9)
COEFFICIENTS = (1.3, 0.6, 2.1)
BATCH = 2


def axis_size(n, kinds, grid):
    return n + 1 if grid == "node" and kinds[0] != "periodic" else n


def second_difference_1d(n, h, kinds, grid):
    """The 1D operator on one axis and the lift weight of each side's value."""
    m = axis_size(n, kinds, grid)
    t = (
        np.diag(-2.0 * np.ones(m))
        + np.diag(np.ones(m - 1), 1)
        + np.diag(np.ones(m - 1), -1)
    )
    weights = [0.0, 0.0]
    if kinds[0] == "periodic":
        t[0, -1] += 1
        t[-1, 0] += 1
    for side, kind in enumerate(kinds if kinds[0] != "periodic" else ()):
        row = 0 if side == 0 else m - 1
        if grid == "cell":
            # Ghost 2g - u (Dirichlet) or u + h q (Neumann) folded into row.
            t[row, row] += -1 if kind == "dirichlet" else 1
            weights[side] = 2 / h**2 if kind == "dirichlet" else 1 / h
        elif kind == "neumann":
            # Ghost u_1 + 2 h q: the inward neighbour counts twice.
            t[row, 1 if side == 0 else m - 2] += 1
            weights[side] = 2 / h
        # Node Dirichlet rows become identity rows in reference_system.
    return t / h**2, weights


def reference_system(shape, kinds, values, coefficients, shift, grid):
    """Sparse ``A`` and a function mapping ``f`` to the right-hand side."""
    ndim = len(shape)
    spacings = [LENGTHS[i] / shape[i] for i in range(ndim)]
    sizes = [axis_size(n, k, grid) for n, k in zip(shape, kinds)]
    total = int(np.prod(sizes))
    a_matrix = shift * sp.identity(total, format="csr")
    weights = []
    for axis in range(ndim):
        t, w = second_difference_1d(shape[axis], spacings[axis], kinds[axis], grid)
        weights.append(w)
        factors = [sp.identity(s, format="csr") for s in sizes]
        factors[axis] = sp.csr_matrix(t)
        term = factors[0]
        for factor in factors[1:]:
            term = sp.kron(term, factor, format="csr")
        a_matrix = a_matrix + coefficients[axis] * term

    # Node grid: Dirichlet wall nodes are known, u = g.
    known = np.zeros(sizes, dtype=bool)
    known_values = np.zeros(sizes)
    if grid == "node":
        for axis in range(ndim):
            for side, kind in enumerate(kinds[axis]):
                if kind != "dirichlet":
                    continue
                index = [slice(None)] * ndim
                index[axis] = 0 if side == 0 else sizes[axis] - 1
                known[tuple(index)] = True
                known_values[tuple(index)] = values[axis][side]
        mask = sp.diags(known.ravel().astype(float))
        a_matrix = (sp.identity(total) - mask) @ a_matrix + mask

    def rhs(f):
        b = f.copy()
        for axis in range(ndim):
            for side, kind in enumerate(kinds[axis]):
                if kind == "periodic" or (grid == "node" and kind == "dirichlet"):
                    continue
                index = [slice(None)] * ndim
                index[axis] = 0 if side == 0 else sizes[axis] - 1
                b[tuple(index)] -= (
                    coefficients[axis] * weights[axis][side] * values[axis][side]
                )
        b[known] = known_values[known]
        return b.ravel()

    return a_matrix.tocsr(), rhs, sizes


def boundary_values(shape, kinds, grid, rng):
    """Random values varying along each wall (face shape = field minus axis)."""
    sizes = [axis_size(n, k, grid) for n, k in zip(shape, kinds)]
    values = []
    for axis, axis_kinds in enumerate(kinds):
        face = sizes[:axis] + sizes[axis + 1 :]
        values.append(
            [
                None if kind == "periodic" else rng.standard_normal(face)
                for kind in axis_kinds
            ]
        )
    return values


def solver_bc(kinds, values):
    bc = []
    for axis_kinds, axis_values in zip(kinds, values):
        if axis_kinds[0] == "periodic":
            bc.append(Periodic())
            continue
        bc.append(
            tuple(
                (Dirichlet if kind == "dirichlet" else Neumann)(torch.from_numpy(v))
                for kind, v in zip(axis_kinds, axis_values)
            )
        )
    return tuple(bc)


def singular(kinds, shift):
    return shift == 0 and not any("dirichlet" in k for k in kinds)


def compare(shape, names, grid, shift, dtype=torch.float64, seed=0):
    kinds = [AXES[name] for name in names]
    ndim = len(shape)
    coefficients = COEFFICIENTS[:ndim]
    rng = np.random.default_rng(seed)
    values = boundary_values(shape, kinds, grid, rng)
    a_matrix, rhs, sizes = reference_system(
        shape, kinds, values, coefficients, shift, grid
    )

    if singular(kinds, shift):
        # A compatible f: rhs(f) = A u0. Fix the constant with a bordered system.
        fs = []
        for _ in range(BATCH):
            u0 = rng.standard_normal(int(np.prod(sizes)))
            fs.append((a_matrix @ u0 - rhs(np.zeros(sizes))).reshape(sizes))
        ones = sp.csr_matrix(np.ones((a_matrix.shape[0], 1)))
        bordered = sp.bmat([[a_matrix, ones], [ones.T, None]], format="csc")
        refs = [
            spla.spsolve(bordered, np.append(rhs(f), 0.0))[:-1].reshape(sizes)
            for f in fs
        ]
    else:
        fs = [rng.standard_normal(sizes) for _ in range(BATCH)]
        refs = [spla.spsolve(a_matrix.tocsc(), rhs(f)).reshape(sizes) for f in fs]

    solver = UniformGridSolver(
        shape,
        LENGTHS[:ndim],
        solver_bc(kinds, values),
        coefficients=coefficients,
        shift=shift,
        grid=grid,
    )
    f = torch.from_numpy(np.stack(fs)).to(dtype)
    u = solver(f).double().numpy()
    expected = np.stack(refs)

    error = u - expected
    if singular(kinds, shift):
        error = error - error.reshape(BATCH, -1).mean(axis=1).reshape(
            (BATCH,) + (1,) * ndim
        )
    scale = max(np.abs(expected).max(), 1.0)
    tolerance = (1e-10 if dtype == torch.float64 else 1e-4) * scale
    assert u.shape == expected.shape
    assert np.abs(error).max() <= tolerance, (
        f"max error {np.abs(error).max():.2e} > {tolerance:.2e}"
    )


CASES_1D = [
    ((n,), (name,), grid, shift)
    for n in (1, 2, 3, 8)
    for name in AXES
    for grid in ("cell", "node")
    for shift in (0.0, -1.7, 0.37)
]
CASES_2D = [
    (shape, names, grid, shift)
    for shape in ((1, 3), (3, 1), (2, 2), (5, 6))
    for names in itertools.product(AXES, repeat=2)
    for grid in ("cell", "node")
    for shift in (0.0, 0.37)
]
CASES_3D = [
    (shape, names, grid, shift)
    for shape in ((1, 2, 3), (3, 4, 5))
    for names in (
        ("p", "nd", "dd"),
        ("nn", "p", "dn"),
        ("dd", "nn", "nd"),
        ("nn", "nn", "nn"),
        ("dn", "dd", "p"),
    )
    for grid in ("cell", "node")
    for shift in (0.0, -1.7)
]


def _id(case):
    shape, names, grid, shift = case
    return f"{grid}-{'x'.join(map(str, shape))}-{'-'.join(names)}-c{shift}"


@pytest.mark.parametrize("case", CASES_1D + CASES_2D + CASES_3D, ids=_id)
def test_matches_scipy_spsolve(case):
    compare(*case)


@pytest.mark.parametrize("grid", ["cell", "node"])
@pytest.mark.parametrize("names", list(itertools.product(AXES, repeat=2)), ids="-".join)
def test_matches_scipy_spsolve_float32(names, grid):
    compare((5, 6), names, grid, 0.37, dtype=torch.float32)
