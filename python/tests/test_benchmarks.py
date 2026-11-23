import itertools
import math

import pytest
import torch
from orthel.boundary import Dirichlet, Neumann, Periodic
from orthel.eigenvalues import second_difference_eigenvalues
from orthel.uniform_grid_solver import UniformGridSolver

AXIS_TYPES = [
    Periodic(),
    Neumann(),
    Dirichlet(),
    (Neumann(), Dirichlet()),
    (Dirichlet(), Neumann()),
]

CASES = (
    [((12,), (1.0,), (bc,)) for bc in AXIS_TYPES]
    + [((12, 9), (1.0, 1.5), bc) for bc in itertools.product(AXIS_TYPES, repeat=2)]
    + [
        ((6, 5, 7), (1.0, 1.5, 2.0), bc)
        for bc in [
            (Periodic(), (Neumann(), Dirichlet()), Dirichlet()),
            (Neumann(), Neumann(), Neumann()),
            ((Dirichlet(), Neumann()), Periodic(), Neumann()),
            (Periodic(), Periodic(), Periodic()),
        ]
    ]
)

# (coefficients, shift) for each spatial dimension count: Poisson, anisotropic
# screened Poisson and a non-resonant Helmholtz shift.
OPERATORS = {
    1: [((1.0,), 0.0), ((2.0,), -5.0), ((1.0,), 7.3)],
    2: [((1.0, 1.0), 0.0), ((2.0, 0.5), -5.0), ((1.0, 1.5), 7.3)],
    3: [((1.0, 1.0, 1.0), 0.0), ((2.0, 0.5, 1.0), -5.0), ((1.0, 1.5, 0.7), 7.3)],
}
OPERATOR_IDS = ["poisson", "screened", "helmholtz"]

DTYPES = [torch.float64, torch.float32]


def _case_id(case):
    shape, _, bc = case
    axes = ["/".join(s.kind for s in b) if isinstance(b, tuple) else b.kind for b in bc]
    return f"{len(shape)}d-" + "-".join(axes)


def discrete_mode(n, length, bc, m, periodic_phase):
    """Return mode ``m`` of the 1D second difference and its eigenvalue.

    The mode is the analytic function sampled at cell centres, so it is
    independent of the transforms: a sine or cosine of ``2πm x / L`` on a
    periodic axis, and on a wall axis a sine (Dirichlet low side) or cosine
    (Neumann low side) of ``π (m + offset) x / L``, with ``offset`` 1/2 per
    Dirichlet side.
    """
    h = length / n
    x = (torch.arange(n, dtype=torch.float64) + 0.5) * h
    if bc == ("periodic", "periodic"):
        theta = 2 * math.pi * m / n
        wave = torch.sin if periodic_phase == "sin" else torch.cos
    else:
        theta = math.pi * (m + 0.5 * bc.count("dirichlet")) / n
        wave = torch.sin if bc[0] == "dirichlet" else torch.cos
    eigenvalue = -(((2 / h) * math.sin(theta / 2)) ** 2)
    return wave(theta / h * x), eigenvalue


def choose_mode(n, bc, axis, selection):
    """Pick a low or high mode on each axis; ``"mixed"`` alternates them by axis."""
    high = selection == "high" or (selection == "mixed" and axis % 2 == 1)
    if bc != ("periodic", "periodic"):
        return (n - 1 if high else 1), "cos"
    # At cell centres the Nyquist mode (n // 2, n even) is sin(π(j + 1/2)) = ±1;
    # its cosine vanishes there, so the high mode uses the sine.
    return (n // 2, "sin") if high else (1, "cos")


def smallest_nonzero_eigenvalue(solver):
    """``min |Σ a_i λ_i + c|`` over the operator's non-null modes."""
    total = torch.full((), solver.shift, dtype=torch.float64)
    ndim = len(solver.shape)
    for i, (n, h, a, bc) in enumerate(
        zip(solver.shape, solver.spacings, solver.coefficients, solver.boundary_types)
    ):
        view = [1] * ndim
        view[i] = -1
        total = total + a * second_difference_eigenvalues(n, h, bc).reshape(view)
    magnitude = total.abs()
    return magnitude[magnitude > 0].min().item()


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("selection", ["low", "high", "mixed"])
@pytest.mark.parametrize("operator", range(3), ids=OPERATOR_IDS)
@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_single_eigenvector_is_solved_exactly(case, operator, selection, dtype):
    shape, lengths, bc = case
    coefficients, shift = OPERATORS[len(shape)][operator]
    solver = UniformGridSolver(
        shape, lengths, bc, coefficients=coefficients, shift=shift
    )

    f = torch.ones((), dtype=torch.float64)
    eigenvalue = shift
    for axis, (n, length, axis_bc, a) in enumerate(
        zip(shape, lengths, solver.boundary_types, coefficients)
    ):
        m, phase = choose_mode(n, axis_bc, axis, selection)
        values, lam = discrete_mode(n, length, axis_bc, m, phase)
        f = f[..., None] * values
        eigenvalue += a * lam
    expected = f / eigenvalue
    assert f.abs().max() > 0.5, "mode vanishes on the grid"

    u = solver(f.to(dtype))

    # Forward-error bound of a backward-stable solve: eps ‖f‖ ‖A⁻¹‖.
    tolerance = (
        10
        * torch.finfo(dtype).eps
        * f.abs().max()
        / smallest_nonzero_eigenvalue(solver)
    )
    error = (u.double() - expected).abs().max()
    assert error <= tolerance, f"max error {error:.2e} exceeds {tolerance:.2e}"


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("shift", [0.0, -5.0, 7.3])
@pytest.mark.parametrize(
    "bc",
    [(Periodic(),), (Neumann(), Periodic()), (Neumann(), Neumann(), Periodic())],
    ids=lambda bc: "-".join(b.kind for b in bc),
)
def test_constant_mode(bc, shift, dtype):
    # The constant is the m = 0 mode when no side is Dirichlet. With c = 0 it is
    # the null mode, so the zero-mean solution u = 0 is returned; otherwise u = f / c.
    shape = (8, 6, 5)[: len(bc)]
    solver = UniformGridSolver(shape, 1.0, bc, shift=shift)
    f = torch.full(shape, 3.0, dtype=dtype)

    u = solver(f)

    expected = torch.zeros_like(f) if shift == 0 else f / shift
    torch.testing.assert_close(u, expected, atol=10 * torch.finfo(dtype).eps, rtol=0)
