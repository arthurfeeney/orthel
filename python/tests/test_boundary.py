import pytest
import torch

from orthel.boundary import Dirichlet, Neumann, Periodic, parse_bc
from orthel.uniform_grid_solver import UniformGridSolver


def kinds(pairs):
    return tuple((low.kind, high.kind) for low, high in pairs)


@pytest.mark.parametrize(
    "bc, expected",
    [
        (Periodic(), (("periodic", "periodic"),) * 2),
        (Dirichlet(), (("dirichlet", "dirichlet"),) * 2),
        ([Neumann(), Periodic()], (("neumann", "neumann"), ("periodic", "periodic"))),
        ([(Dirichlet(1.0), Neumann()), Periodic()], (("dirichlet", "neumann"), ("periodic", "periodic"))),
    ],
)
def test_parse_bc(bc, expected):
    assert kinds(parse_bc(bc, 2)) == expected


def test_parse_bc_keeps_values():
    g = torch.ones(3)
    (low, high), = parse_bc([(Dirichlet(g), Neumann(2.0))], 1)

    assert low.value is g and high.value == 2.0


@pytest.mark.parametrize(
    "bc, error, match",
    [
        ("periodic", TypeError, "expected a boundary object"),
        ([(Dirichlet(), "neumann"), Periodic()], TypeError, "expected a boundary object"),
        ([1.0, Neumann()], TypeError, "expected a boundary object"),
        ([(Periodic(), Neumann()), Neumann()], ValueError, "periodic must apply to both sides"),
        ([Periodic()], ValueError, "expected 2 boundary entries"),
    ],
)
def test_parse_bc_rejects_invalid(bc, error, match):
    with pytest.raises(error, match=match):
        parse_bc(bc, 2)


def test_solver_uses_values_given_at_construction():
    # A 1D Dirichlet problem u'' = 0 with u(0) = 1, u(1) = 3 is linear.
    solver = UniformGridSolver((8,), 1.0, [(Dirichlet(1.0), Dirichlet(3.0))])
    x = solver.coords(dtype=torch.float64)[0]

    u = solver(torch.zeros(8, dtype=torch.float64))

    torch.testing.assert_close(u, 1 + 2 * x)


def test_call_bc_replaces_values():
    solver = UniformGridSolver((8,), 1.0, [(Dirichlet(1.0), Dirichlet(3.0))])
    x = solver.coords(dtype=torch.float64)[0]

    u = solver(torch.zeros(8, dtype=torch.float64), [(Dirichlet(-1.0), Dirichlet(1.0))])

    torch.testing.assert_close(u, -1 + 2 * x)


def test_call_bc_must_match_types():
    solver = UniformGridSolver((8, 6), 1.0, [Periodic(), Dirichlet()])
    with pytest.raises(ValueError, match="do not match"):
        solver(torch.zeros(8, 6), [Periodic(), Neumann()])
