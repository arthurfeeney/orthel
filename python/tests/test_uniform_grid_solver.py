import itertools
import math

import pytest
import torch

from orthel.boundary import Dirichlet, Neumann, Periodic, parse_bc
from orthel.uniform_grid_solver import UniformGridSolver

AXIS_TYPES = [
    Periodic(),
    Neumann(),
    Dirichlet(),
    (Neumann(), Dirichlet()),
    (Dirichlet(), Neumann()),
]

CASES = (
    [((257,), (1.0,), (bc,)) for bc in AXIS_TYPES]
    + [((48, 37), (1.0, 1.5), bc) for bc in itertools.product(AXIS_TYPES, repeat=2)]
    + [
        ((16, 12, 20), (1.0, 1.5, 2.0), bc)
        for bc in [
            (Periodic(), (Neumann(), Dirichlet()), Dirichlet()),
            (Neumann(), Neumann(), Neumann()),
            ((Dirichlet(), Neumann()), Periodic(), Neumann()),
            (Periodic(), Periodic(), Periodic()),
        ]
    ]
)

DTYPES = [torch.float64, torch.float32]


def _case_id(case):
    shape, _, bc = case
    axes = ["/".join(s.kind for s in b) if isinstance(b, tuple) else b.kind for b in bc]
    return f"{len(shape)}d-" + "-".join(axes)


def apply_operator(solver, u, bc=None):
    """Apply ``Σ_i a_i ∂²/∂x_i² + c`` with 3-point stencils and ghost cells.

    This is independent of the transforms used by the solver, so it serves as
    an oracle: ``f - apply_operator(u, bc)`` is the residual of
    ``A u = f - b``, where ``b`` holds the wall-value contributions.
    """
    ndim = len(solver.shape)
    pairs = solver.bc if bc is None else parse_bc(bc, ndim)
    out = solver.shift * u
    for axis in range(ndim):
        dim, n, h = u.ndim - ndim + axis, solver.shape[axis], solver.spacings[axis]
        edges = (u.narrow(dim, 0, 1), u.narrow(dim, n - 1, 1))
        ghosts = []
        for side, edge in enumerate(edges):
            kind = solver.boundary_types[axis][side]
            if kind == "periodic":
                ghosts.append(edges[1 - side])
                continue
            g = torch.as_tensor(pairs[axis][side].value, dtype=u.dtype)
            g = torch.broadcast_to(g, edge.squeeze(dim).shape).unsqueeze(dim)
            ghosts.append(2 * g - edge if kind == "dirichlet" else edge + h * g)
        padded = torch.cat([ghosts[0], u, ghosts[1]], dim=dim)
        a = solver.coefficients[axis]
        out = out + a * (padded.narrow(dim, 0, n) - 2 * u + padded.narrow(dim, 2, n)) / h**2
    return out


def backward_error(solver, u, f, bc=None):
    """Normwise backward error ``‖(f - b) - A u‖ / (‖A‖ ‖u‖ + ‖f - b‖)`` in the ∞-norm."""
    residual = f - apply_operator(solver, u, bc)
    if solver.shift == 0 and not any("dirichlet" in bc for bc in solver.boundary_types):
        # Without Dirichlet sides or a shift, the solver discards the mean as a uniform source.
        spatial = tuple(range(-len(solver.shape), 0))
        residual = residual - residual.mean(dim=spatial, keepdim=True)
    b = -apply_operator(solver, torch.zeros_like(u), bc)
    norm_a = sum(4 * a / h**2 for a, h in zip(solver.coefficients, solver.spacings))
    norm_a += abs(solver.shift)
    return (residual.abs().max() / (norm_a * u.abs().max() + (f - b).abs().max())).item()


def random_bc(solver, batch, dtype, generator):
    """The solver's boundary types with random wall values."""
    types = {"dirichlet": Dirichlet, "neumann": Neumann}
    bc = []
    for axis, kinds in enumerate(solver.boundary_types):
        if kinds[0] == "periodic":
            bc.append(Periodic())
            continue
        face = solver.shape[:axis] + solver.shape[axis + 1 :]
        bc.append(tuple(
            types[kind](torch.randn(*batch, *face, dtype=dtype, generator=generator))
            for kind in kinds
        ))
    return bc


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_solve_backward_error(case, dtype):
    shape, lengths, bc = case
    solver = UniformGridSolver(shape, lengths, bc)
    generator = torch.Generator().manual_seed(0)
    f = torch.randn(2, *shape, dtype=dtype, generator=generator)
    boundaries = random_bc(solver, (2,), dtype, generator)

    u = solver(f, boundaries)

    assert u.shape == f.shape and u.dtype == dtype
    eta = backward_error(solver, u, f, boundaries)
    tolerance = 10 * torch.finfo(dtype).eps
    assert eta <= tolerance, f"backward error {eta:.2e}"


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_homogeneous_solve_backward_error(case, dtype):
    shape, lengths, bc = case
    solver = UniformGridSolver(shape, lengths, bc)
    f = torch.randn(*shape, dtype=dtype, generator=torch.Generator().manual_seed(1))

    u = solver(f)

    eta = backward_error(solver, u, f)
    tolerance = 10 * torch.finfo(dtype).eps
    assert eta <= tolerance, f"backward error {eta:.2e}"


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_gradient_backward_error(case, dtype):
    # The gradient λ = ∂⟨g, u⟩/∂f should solve A λ = g with homogeneous walls.
    shape, lengths, bc = case
    solver = UniformGridSolver(shape, lengths, bc)
    generator = torch.Generator().manual_seed(2)
    f = torch.randn(2, *shape, dtype=dtype, generator=generator, requires_grad=True)
    boundaries = random_bc(solver, (2,), dtype, generator)
    g = torch.randn(2, *shape, dtype=dtype, generator=generator)

    (lam,) = torch.autograd.grad(solver(f, boundaries), f, g)

    eta = backward_error(solver, lam, g)
    tolerance = 10 * torch.finfo(dtype).eps
    assert eta <= tolerance, f"backward error {eta:.2e}"


def test_backward_error_detects_perturbed_solution():
    # Guards against a vacuous check: a small perturbation must exceed the tolerance.
    solver = UniformGridSolver((48, 37), (1.0, 1.5), (Periodic(), (Dirichlet(), Neumann())))
    generator = torch.Generator().manual_seed(3)
    f = torch.randn(48, 37, dtype=torch.float64, generator=generator)
    boundaries = random_bc(solver, (), torch.float64, generator)
    u = solver(f, boundaries)

    u_bad = u + 1e-12 * u.abs().max() * torch.randn(u.shape, dtype=u.dtype, generator=generator)

    tolerance = 10 * torch.finfo(torch.float64).eps
    assert backward_error(solver, u_bad, f, boundaries) > tolerance


# (coefficients, shift) per spatial dimension count: anisotropic Poisson,
# screened Poisson and a non-resonant Helmholtz shift.
OPERATORS = {
    1: [((0.3,), 0.0), ((2.0,), -5.0), ((1.0,), 7.0)],
    2: [((0.3, 4.0), 0.0), ((2.0, 0.5), -5.0), ((1.0, 1.5), 7.0)],
    3: [((0.3, 4.0, 1.0), 0.0), ((2.0, 0.5, 1.0), -5.0), ((1.0, 1.5, 0.7), 7.0)],
}
OPERATOR_CASES = [
    (case, coefficients, shift)
    for case in CASES
    for coefficients, shift in OPERATORS[len(case[0])]
]


def _operator_case_id(operator_case):
    case, coefficients, shift = operator_case
    return f"{_case_id(case)}-a{'_'.join(map(str, coefficients))}-c{shift}"


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("operator_case", OPERATOR_CASES, ids=_operator_case_id)
def test_constant_coefficient_backward_error(operator_case, dtype):
    (shape, lengths, bc), coefficients, shift = operator_case
    solver = UniformGridSolver(shape, lengths, bc, coefficients=coefficients, shift=shift)
    generator = torch.Generator().manual_seed(4)
    f = torch.randn(2, *shape, dtype=dtype, generator=generator, requires_grad=True)
    boundaries = random_bc(solver, (2,), dtype, generator)
    g = torch.randn(2, *shape, dtype=dtype, generator=generator)

    u = solver(f, boundaries)
    (lam,) = torch.autograd.grad(u, f, g)

    tolerance = 10 * torch.finfo(dtype).eps
    eta = backward_error(solver, u.detach(), f.detach(), boundaries)
    assert eta <= tolerance, f"solve backward error {eta:.2e}"
    eta = backward_error(solver, lam, g)
    assert eta <= tolerance, f"gradient backward error {eta:.2e}"


def test_negative_shift_removes_null_space():
    # All-Neumann with c < 0 is non-singular, so the mean of f must not be discarded.
    solver = UniformGridSolver((16, 12), 1.0, Neumann(), shift=-3.0)
    f = torch.full((16, 12), 2.0, dtype=torch.float64)

    u = solver(f)

    torch.testing.assert_close(u, torch.full_like(f, -2.0 / 3.0))


def test_resonant_shift_raises():
    # c equal to minus an eigenvalue makes the operator singular.
    shape, length = (8,), 1.0
    h = length / shape[0]
    shift = (2 / h * math.sin(math.pi / 8)) ** 2  # periodic mode 1
    with pytest.raises(ValueError, match="singular"):
        UniformGridSolver(shape, length, Periodic(), shift=shift)


@pytest.mark.parametrize("coefficients", [0.0, -1.0, (1.0, 0.0), (1.0,)])
def test_invalid_coefficients_raise(coefficients):
    with pytest.raises(ValueError, match="coefficients"):
        UniformGridSolver((4, 4), 1.0, Neumann(), coefficients=coefficients)


@pytest.mark.parametrize("lengths", [0.0, -1.0, (1.0, 0.0), (1.0, -2.0), float("inf"), float("nan")])
def test_invalid_lengths_raise(lengths):
    with pytest.raises(ValueError, match="lengths must be positive"):
        UniformGridSolver((4, 4), lengths, Neumann())
