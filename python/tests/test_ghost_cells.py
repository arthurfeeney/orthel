import itertools
import math

import pytest
import torch

from orthel.boundary import Dirichlet, Neumann, Periodic
from orthel.uniform_grid_solver import UniformGridSolver
from test_uniform_grid_solver import CASES, OPERATORS, apply_operator, random_bc

WALL_TYPES = [Neumann(), Dirichlet(), (Neumann(), Dirichlet()), (Dirichlet(), Neumann())]
GHOSTS = ["standard", "pde"]
DTYPES = [torch.float64, torch.float32]


def _bc_id(bc):
    return "-".join("/".join(s.kind for s in b) if isinstance(b, tuple) else b.kind for b in bc)


def _case_id(case):
    return f"{len(case[0])}d-{_bc_id(case[2])}"


def apply_with_ghosts(solver, u, ghosts):
    """``Σ_i a_i ∂²u/∂x_i² + c u`` with 3-point stencils, padding ``u`` with ``ghosts``."""
    ndim = len(solver.shape)
    out = solver.shift * u
    for axis in range(ndim):
        dim, n, h = u.ndim - ndim + axis, solver.shape[axis], solver.spacings[axis]
        padded = torch.cat([ghosts[axis][0], u, ghosts[axis][1]], dim=dim)
        a = solver.coefficients[axis]
        out = out + a * (padded.narrow(dim, 0, n) - 2 * u + padded.narrow(dim, 2, n)) / h**2
    return out


def has_null_mode(solver):
    return solver.shift == 0 and not any("dirichlet" in b for b in solver.boundary_types)


def remove_mean(solver, x):
    spatial = tuple(range(-len(solver.shape), 0))
    return x - x.mean(dim=spatial, keepdim=True)


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_standard_ghost_cells_match_stencil_oracle(case):
    # The default ghost cells are the ones the independent oracle in
    # test_uniform_grid_solver uses.
    shape, lengths, bc = case
    solver = UniformGridSolver(shape, lengths, bc)
    generator = torch.Generator().manual_seed(0)
    u = torch.randn(2, *shape, dtype=torch.float64, generator=generator)
    boundaries = random_bc(solver, (2,), torch.float64, generator)

    ghosts = solver.ghost_cells(u, bc=boundaries)

    torch.testing.assert_close(
        apply_with_ghosts(solver, u, ghosts), apply_operator(solver, u, boundaries)
    )


OPERATOR_CASES = [
    (case, coefficients, shift)
    for case in CASES
    for coefficients, shift in OPERATORS[len(case[0])]
]


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("ghost", GHOSTS)
@pytest.mark.parametrize(
    "operator_case",
    OPERATOR_CASES,
    ids=lambda oc: f"{_case_id(oc[0])}-a{'_'.join(map(str, oc[1]))}-c{oc[2]}",
)
def test_solution_satisfies_stencil_with_its_ghost_cells(operator_case, ghost, dtype):
    # Padding the solution with ghost_cells and applying the stencil gives back
    # f, with a normwise backward error at rounding level.
    (shape, lengths, bc), coefficients, shift = operator_case
    solver = UniformGridSolver(
        shape, lengths, bc, coefficients=coefficients, shift=shift, ghost_rule=ghost
    )
    generator = torch.Generator().manual_seed(1)
    f = torch.randn(2, *shape, dtype=dtype, generator=generator)
    boundaries = random_bc(solver, (2,), dtype, generator)

    u = solver(f, boundaries)

    # S(u) = A u + k, where k is the stencil applied to the known ghost parts.
    zero = torch.zeros_like(u)
    known = apply_with_ghosts(solver, zero, solver.ghost_cells(zero, f, boundaries))
    residual = f - apply_with_ghosts(solver, u, solver.ghost_cells(u, f, boundaries))
    if has_null_mode(solver):
        residual = remove_mean(solver, residual)
    norm_a = sum(4 * a / h**2 for a, h in zip(solver.coefficients, solver.spacings)) + abs(shift)
    eta = residual.abs().max() / (norm_a * u.abs().max() + (f - known).abs().max())
    tolerance = 10 * torch.finfo(dtype).eps
    assert eta <= tolerance, f"backward error {eta:.2e}"


def quadratic(x, y):
    return 1 + 0.3 * x - 0.7 * y + 0.5 * x**2 - 0.4 * x * y + 0.8 * y**2


def quadratic_gradient(x, y):
    return 0.3 + x - 0.4 * y, -0.7 - 0.4 * x + 1.6 * y


def exact_bc(solver, solution, gradient):
    """The solver's boundaries with the values of a 2D solution."""
    xs = solver.coords(dtype=torch.float64)
    bc = []
    for axis, sides in enumerate(solver.boundary_types):
        if sides[0] == "periodic":
            bc.append(Periodic())
            continue
        pair = []
        for side, kind in enumerate(sides):
            face = list(xs)
            face[axis] = torch.tensor([0.0 if side == 0 else solver.lengths[axis]], dtype=torch.float64)
            x, y = torch.meshgrid(*face, indexing="ij")
            if kind == "dirichlet":
                pair.append(Dirichlet(solution(x, y).squeeze(axis)))
            else:
                normal = (1 if side == 1 else -1) * gradient(x, y)[axis]
                pair.append(Neumann(normal.squeeze(axis)))
        bc.append(tuple(pair))
    return bc


@pytest.mark.parametrize("ghost", GHOSTS)
@pytest.mark.parametrize("bc", list(itertools.product(WALL_TYPES, repeat=2)), ids=_bc_id)
def test_quadratic_solution(bc, ghost):
    # With a constant f, the "pde" ghost cells are exact for a quadratic, so the
    # discrete solution matches it to rounding. The standard Dirichlet ghost is
    # off by (h²/4) ∂²u/∂n²; the standard Neumann ghost is exact here because
    # ∂³u/∂n³ = 0.
    coefficients = (2.0, 0.5)
    solver = UniformGridSolver((24, 18), (1.0, 1.5), bc, coefficients=coefficients, ghost_rule=ghost)
    x, y = torch.meshgrid(*solver.coords(dtype=torch.float64), indexing="ij")
    expected = quadratic(x, y)
    f = torch.full_like(expected, coefficients[0] * 1.0 + coefficients[1] * 1.6)

    u = solver(f, exact_bc(solver, quadratic, quadratic_gradient))

    error = u - expected
    if has_null_mode(solver):
        error = remove_mean(solver, error)
    error = error.abs().max().item()
    if ghost == "pde" or not any("dirichlet" in b for b in solver.boundary_types):
        assert error <= 1e-11, f"max error {error:.2e}"
    else:
        assert error > 1e-4, "standard Dirichlet ghost should not be exact for quadratics"


def trig_solution(lengths, bc):
    """A smooth product of cosines, periodic along periodic axes, and its gradient.

    The wavenumbers differ between axes: with equal ones, ∂²u/∂x² = ∂²u/∂y²
    and the first-order corner errors of the standard ghost cells cancel.
    """
    ks = [
        2 * math.pi / length if axis_bc[0] == "periodic" else k
        for length, axis_bc, k in zip(lengths, bc, (1.7, 2.9))
    ]
    phases = (0.3, 1.1)

    def solution(x, y):
        return torch.cos(ks[0] * x + phases[0]) * torch.cos(ks[1] * y + phases[1])

    def gradient(x, y):
        cx, cy = torch.cos(ks[0] * x + phases[0]), torch.cos(ks[1] * y + phases[1])
        sx, sy = torch.sin(ks[0] * x + phases[0]), torch.sin(ks[1] * y + phases[1])
        return -ks[0] * sx * cy, -ks[1] * cx * sy

    return solution, gradient, ks


def solve_manufactured(bc, coefficients, shift, ghost, n):
    lengths = (1.0, 1.5)
    solver = UniformGridSolver(
        (n, 3 * n // 2), lengths, bc, coefficients=coefficients, shift=shift, ghost_rule=ghost
    )
    solution, gradient, ks = trig_solution(lengths, solver.boundary_types)
    x, y = torch.meshgrid(*solver.coords(dtype=torch.float64), indexing="ij")
    expected = solution(x, y)
    f = (shift - coefficients[0] * ks[0] ** 2 - coefficients[1] * ks[1] ** 2) * expected
    boundaries = exact_bc(solver, solution, gradient)
    u = solver(f, boundaries)

    # ∂u/∂x on every x-face, using the ghost cells across the walls.
    low, high = solver.ghost_cells(u, f, boundaries)[0]
    padded = torch.cat([low, u, high], dim=0)
    du = (padded[1:] - padded[:-1]) / solver.spacings[0]
    x_faces = torch.arange(n + 1, dtype=torch.float64) * solver.spacings[0]
    xf, yf = torch.meshgrid(x_faces, solver.coords(dtype=torch.float64)[1], indexing="ij")
    if solver.boundary_types[0][0] == "periodic":
        du, xf, yf = du[:-1], xf[:-1], yf[:-1]
    return (u - expected).abs().max().item(), (du - gradient(xf, yf)[0]).abs().max().item()


CONVERGENCE_CASES = [
    ((Dirichlet(), Dirichlet()), (1.0, 1.0), 0.0),
    (((Neumann(), Dirichlet()), (Dirichlet(), Neumann())), (2.0, 0.5), -5.0),
    ((Dirichlet(), Periodic()), (1.0, 1.5), 7.3),
    ((Neumann(), (Dirichlet(), Dirichlet())), (1.0, 1.0), 0.0),
]


@pytest.mark.parametrize("ghost", GHOSTS)
@pytest.mark.parametrize(
    "case", CONVERGENCE_CASES, ids=lambda c: f"{_bc_id(c[0])}-c{c[2]}"
)
def test_convergence_order(case, ghost):
    # The solution is second order with either rule. With "pde" ghost cells
    # the gradient is too, including across the walls and at the corners.
    bc, coefficients, shift = case
    errors = [solve_manufactured(bc, coefficients, shift, ghost, n) for n in (32, 64, 128)]
    u_rates = [math.log2(a[0] / b[0]) for a, b in zip(errors, errors[1:])]
    du_rates = [math.log2(a[1] / b[1]) for a, b in zip(errors, errors[1:])]

    assert min(u_rates) > 1.9, f"solution rates {u_rates}"
    sides = [b if isinstance(b, tuple) else (b, b) for b in bc]
    dirichlet_corner = all(any(isinstance(s, Dirichlet) for s in pair) for pair in sides)
    if ghost == "standard" and dirichlet_corner:
        # Where two Dirichlet walls meet, the standard ghost cells' O(1)
        # truncation errors make the gradient first order.
        assert max(du_rates) < 1.5, f"gradient rates {du_rates}"
    else:
        assert min(du_rates) > 1.9, f"gradient rates {du_rates}"


def test_pde_rule_matches_standard_without_walls():
    standard = UniformGridSolver((16, 12), 1.0, Periodic(), shift=-2.0)
    pde = UniformGridSolver((16, 12), 1.0, Periodic(), shift=-2.0, ghost_rule="pde")
    f = torch.randn(16, 12, dtype=torch.float64, generator=torch.Generator().manual_seed(2))

    torch.testing.assert_close(pde(f), standard(f), rtol=0, atol=0)


@pytest.mark.parametrize(
    "bc", [(Dirichlet(), (Neumann(), Dirichlet())), (Periodic(), Dirichlet()), (Neumann(), Periodic())], ids=_bc_id
)
def test_pde_ghost_gradcheck(bc):
    solver = UniformGridSolver((6, 5), (1.0, 1.5), bc, coefficients=(2.0, 0.5), shift=-3.0, ghost_rule="pde")
    generator = torch.Generator().manual_seed(3)
    f = torch.randn(6, 5, dtype=torch.float64, generator=generator, requires_grad=True)
    walls = [(axis, pair) for axis, pair in enumerate(random_bc(solver, (), torch.float64, generator))
             if isinstance(pair, tuple)]
    values = [side.value.requires_grad_() for _, pair in walls for side in pair]

    def solve(f, *values):
        bc = list(solver.bc)
        for i, (axis, (low, high)) in enumerate(walls):
            bc[axis] = (type(low)(values[2 * i]), type(high)(values[2 * i + 1]))
        return solver(f, bc)

    assert torch.autograd.gradcheck(solve, (f, *values))


def test_pde_ghost_cells_require_f():
    solver = UniformGridSolver((6, 5), 1.0, (Neumann(), Periodic()), ghost_rule="pde")
    with pytest.raises(ValueError, match="f is required"):
        solver.ghost_cells(torch.zeros(6, 5))


def test_invalid_dirichlet_ghost_raises():
    with pytest.raises(ValueError, match="ghost_rule"):
        UniformGridSolver((6, 5), 1.0, Dirichlet(), ghost_rule="quadratic")
