"""UniformGridSolver with ``grid="node"``: unknowns at cell corners."""

import math

import pytest
import torch
from orthel.boundary import Dirichlet, Neumann, Periodic
from orthel.uniform_grid_solver import UniformGridSolver

DTYPES = [torch.float32, torch.float64]

# (shape, lengths, bc, coefficients, shift). Covers every (low, high) wall pair,
# periodic axes, 1D and 3D, unequal coefficients and a screening shift.
CASES = {
    "dd_dd": ((6, 7), (1.2, 1.5), (Dirichlet(1.0), Dirichlet(-0.5)), 1.0, 0.0),
    "nn_nn": ((6, 7), (1.2, 1.5), (Neumann(), Neumann()), 1.0, 0.0),
    "nd_dn": (
        (6, 7),
        (1.2, 1.5),
        ((Neumann(0.5), Dirichlet(2.0)), (Dirichlet(-1.0), Neumann(0.25))),
        (1.0, 2.5),
        0.0,
    ),
    "periodic_nn": ((6, 8), (1.2, 1.5), (Periodic(), Neumann(0.75)), 1.0, -3.0),
    "periodic": ((6, 8), (1.2, 1.5), Periodic(), 1.0, 0.0),
    "1d_dn": ((9,), 2.0, ((Dirichlet(1.5), Neumann(-2.0)),), 1.0, 0.0),
    "3d_mixed": (
        (3, 4, 5),
        (1.0, 1.1, 1.3),
        (Neumann(), Periodic(), (Dirichlet(0.5), Neumann())),
        (1.0, 0.5, 2.0),
        0.0,
    ),
}


def _random(shape, dtype, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, dtype=torch.float64, generator=generator).to(dtype)


def _make(name, **kwargs):
    shape, lengths, bc, coefficients, shift = CASES[name]
    return UniformGridSolver(
        shape,
        lengths,
        bc,
        coefficients=coefficients,
        shift=shift,
        grid="node",
        **kwargs,
    )


def node_residual(solver, u, f):
    """Independent node stencil: ``A u - f`` on the unknown nodes.

    Also returns the largest error in the Dirichlet wall nodes, skipping
    corners, which take the later axis's value. Neumann wall nodes use the
    ghost ``u_1 + 2 h q`` (outward ``q``); periodic axes wrap. Boundary values
    broadcast to the face as in the solver.
    """
    ndim = len(solver.shape)
    dims = [u.ndim - ndim + i for i in range(ndim)]
    au = solver.shift * u
    keep = [slice(None)] * u.ndim
    wall_error = torch.zeros((), dtype=u.dtype)
    for axis, dim in enumerate(dims):
        h, a = solver.spacings[axis], solver.coefficients[axis]
        n = u.shape[dim]
        low, high = solver.bc[axis]
        if low.kind == "periodic":
            au = au + a * (u.roll(1, dim) - 2 * u + u.roll(-1, dim)) / h**2
            continue
        face = u.shape[:dim] + u.shape[dim + 1 :]
        ghosts = []
        for side, boundary in ((0, low), (1, high)):
            value = torch.as_tensor(boundary.value, dtype=u.dtype)
            value = torch.broadcast_to(value, face).unsqueeze(dim)
            edge = u.narrow(dim, 0 if side == 0 else n - 1, 1)
            if boundary.kind == "neumann":
                ghosts.append(
                    u.narrow(dim, 1 if side == 0 else n - 2, 1) + 2 * h * value
                )
                continue
            ghosts.append(torch.zeros_like(edge))  # not an unknown
            index = [slice(None)] * u.ndim
            for later in range(axis + 1, ndim):
                lo, hi = solver.bc[later]
                index[dims[later]] = slice(
                    1 if lo.kind == "dirichlet" else 0,
                    -1 if hi.kind == "dirichlet" else None,
                )
            error = (edge - value)[tuple(index)]
            if error.numel():
                wall_error = torch.maximum(wall_error, error.abs().max())
        lower = torch.cat([ghosts[0], u.narrow(dim, 0, n - 1)], dim)
        upper = torch.cat([u.narrow(dim, 1, n - 1), ghosts[1]], dim)
        au = au + a * (lower - 2 * u + upper) / h**2
        keep[dim] = slice(
            1 if low.kind == "dirichlet" else 0,
            n - 1 if high.kind == "dirichlet" else n,
        )
    return (au - f)[tuple(keep)], wall_error


def _singular(solver):
    return solver.shift == 0 and not any(
        "dirichlet" in b for b in solver.boundary_types
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("name", CASES)
def test_solution_satisfies_node_stencil(name, dtype):
    solver = _make(name)
    f = _random((2, *solver.field_shape), dtype)
    if _singular(solver):
        # Pure Neumann/periodic: use a compatible f = A u (every node is unknown).
        f = node_residual(solver, _random(f.shape, dtype, 1), torch.zeros_like(f))[0]

    u = solver(f)

    residual, wall_error = node_residual(solver, u, f)
    assert u.shape == f.shape and u.dtype == dtype
    # Round-off from the solve, amplified by the 1/h^2 stencil: up to a few
    # hundred eps on these small grids, depending on the transform backend.
    scale = 1e3
    tolerance = scale * torch.finfo(dtype).eps * f.abs().max().item()
    assert residual.abs().max() <= tolerance
    assert wall_error <= 10 * torch.finfo(dtype).eps


def test_tensor_boundary_values_and_batch():
    # Values vary along the wall and, for q, across the batch.
    g = torch.linspace(-1.0, 1.0, 8, dtype=torch.float64)  # along the x nodes
    q = _random((2, 6), torch.float64, 3)  # (batch, y nodes)
    bc = ((Dirichlet(g), Dirichlet(0.5)), (Neumann(q), Neumann()))
    solver = UniformGridSolver((5, 7), (1.0, 1.4), bc, grid="node")
    f = _random((2, 6, 8), torch.float64)

    u = solver(f)

    residual, wall_error = node_residual(solver, u, f)
    assert residual.abs().max() <= 1e-12 * f.abs().max() * 1e2
    assert wall_error == 0
    torch.testing.assert_close(u[:, 0], g.expand(2, -1))


@pytest.mark.parametrize("name", ["nn_nn", "periodic"])
def test_null_space_gives_solution_up_to_constant(name):
    solver = _make(name)
    u = _random((2, *solver.field_shape), torch.float64)
    f = node_residual(solver, u, torch.zeros_like(u))[0]  # A u

    recovered = solver(f)

    offset = recovered - u
    offset = offset - offset.mean(dim=(-2, -1), keepdim=True)
    assert offset.abs().max() <= 1e-12 * f.abs().max()


@pytest.mark.parametrize(
    "bc",
    [
        (Dirichlet(), Dirichlet()),
        (Neumann(), Dirichlet()),
        ((Dirichlet(), Neumann()), Periodic()),
    ],
    ids=["dd", "nd", "dn_periodic"],
)
def test_analytic_solution_is_second_order(bc):
    # u = cos(a y + p) cos(b x + r) with exact boundary values at the nodes.
    ly, lx = 1.0, 1.5
    a, b, p, r = 1.3 * math.pi, 2 * math.pi / lx, 0.4, 0.3

    def exact(y, x):
        return torch.cos(a * y + p) * torch.cos(b * x + r)

    errors = []
    for n in (16, 32, 64):
        solver = UniformGridSolver((n, n), (ly, lx), bc, grid="node")
        y, x = solver.coords(dtype=torch.float64)
        u = exact(y[:, None], x[None, :])
        f = -(a**2 + b**2) * u

        def wall(axis, side):
            boundary = solver.bc[axis][side]
            if boundary.kind == "periodic":
                return boundary
            coord = (ly, lx)[axis] * side
            if axis == 0:
                yy, xx = torch.tensor(coord, dtype=torch.float64), x
                normal = -a * torch.sin(a * yy + p) * torch.cos(b * xx + r)
            else:
                yy, xx = y, torch.tensor(coord, dtype=torch.float64)
                normal = -b * torch.cos(a * yy + p) * torch.sin(b * xx + r)
            if boundary.kind == "dirichlet":
                return Dirichlet(exact(yy, xx))
            return Neumann(normal if side == 1 else -normal)

        exact_bc = tuple((wall(i, 0), wall(i, 1)) for i in range(2))
        errors.append((solver(f, exact_bc) - u).abs().max().item())

    rates = [math.log2(e0 / e1) for e0, e1 in zip(errors, errors[1:])]
    assert min(rates) > 1.9, errors
    assert errors[-1] < 1e-3


@pytest.mark.parametrize("name", ["dd_dd", "nd_dn", "periodic_nn", "1d_dn"])
def test_gradcheck(name):
    solver = _make(name)
    f = _random((2, *solver.field_shape), torch.float64).requires_grad_()

    assert torch.autograd.gradcheck(solver, (f,))


def test_gradcheck_boundary_value():
    solver = UniformGridSolver((4, 5), 1.0, (Dirichlet(), Neumann()), grid="node")
    f = _random(solver.field_shape, torch.float64)
    g = _random((6,), torch.float64).requires_grad_()
    q = _random((5,), torch.float64).requires_grad_()

    def solve(g, q):
        return solver(f, ((Dirichlet(g), Dirichlet()), (Neumann(q), Neumann())))

    assert torch.autograd.gradcheck(solve, (g, q))


def test_single_cell_dirichlet_axis_has_no_unknowns():
    solver = UniformGridSolver((1, 4), 1.0, (Dirichlet(2.0), Periodic()), grid="node")
    u = solver(torch.randn(2, 4, dtype=torch.float64))

    torch.testing.assert_close(u, torch.full_like(u, 2.0))


def test_field_shape_and_coords():
    solver = UniformGridSolver(
        (4, 6), (2.0, 3.0), (Dirichlet(), Periodic()), grid="node"
    )

    assert solver.field_shape == (5, 6)
    y, x = solver.coords()
    torch.testing.assert_close(y, torch.arange(5) * 0.5, check_dtype=False)
    torch.testing.assert_close(x, torch.arange(6) * 0.5, check_dtype=False)
    with pytest.raises(ValueError, match=r"\(\*batch, \*\(5, 6\)\)"):
        solver(torch.zeros(4, 6))


def test_node_grid_rejects_unsupported_options():
    with pytest.raises(ValueError, match="grid"):
        UniformGridSolver((4,), 1.0, Dirichlet(), grid="edge")
    solver = UniformGridSolver((4,), 1.0, Dirichlet(), grid="node")
    with pytest.raises(NotImplementedError):
        solver.ghost_cells(torch.zeros(5))


SMALL_AXES = {
    "nn": (Neumann(-0.4), Neumann(0.3)),
    "dd": (Dirichlet(0.7), Dirichlet(-1.2)),
    "nd": (Neumann(-0.4), Dirichlet(-1.2)),
    "dn": (Dirichlet(0.7), Neumann(0.3)),
    "p": Periodic(),
}
SMALL_GRIDS = [((n,), (a,)) for n in (1, 2, 3) for a in SMALL_AXES] + [
    ((ny, nx), (a, b))
    for ny in (1, 2, 3)
    for nx in (1, 2, 3)
    for a in SMALL_AXES
    for b in SMALL_AXES
]


@pytest.mark.parametrize("shift", [0.0, -1.5])
@pytest.mark.parametrize(
    "shape, axes", SMALL_GRIDS, ids=lambda v: "x".join(map(str, v)) if v else ""
)
def test_small_grids_satisfy_node_stencil(shape, axes, shift):
    # One-cell axes are the edge case: with a Neumann and a Dirichlet side the
    # Neumann ghost reflects onto the Dirichlet wall node itself.
    bc = tuple(SMALL_AXES[a] for a in axes)
    solver = UniformGridSolver(shape, 1.3, bc, shift=shift, grid="node")
    f = _random((2, *solver.field_shape), torch.float64, 20)
    if _singular(solver):
        # Pure Neumann/periodic: use a compatible f = A u (every node is unknown).
        f = node_residual(solver, _random(f.shape, torch.float64, 21), 0 * f)[0]

    u = solver(f)

    residual, wall_error = node_residual(solver, u, f)
    error = residual.abs().max().item() if residual.numel() else 0.0
    assert error <= 1e-10 * f.abs().max().item()
    assert wall_error <= 1e-14
