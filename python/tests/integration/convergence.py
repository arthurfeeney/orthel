"""Convergence-rate integration tests for UniformGridSolver.

Each study solves a manufactured problem on a sequence of refined grids and
measures the observed order of three errors against the exact solution of
the continuous problem:

* ``u``: the solution at cell centres.
* ``grad``: ``∂u/∂x_i`` on every face, using the solver's ghost cells across walls.
* ``adjoint``: the autograd gradient of ``J(u) = Σ w u`` with respect to ``f``,
  which approximates the solution ``z`` of the adjoint problem ``A z = w``
  with homogeneous boundaries, in every cell.
* ``adj-int``: the same, two or more cells away from the walls.
* ``adj-dir``: the directional derivative along a smooth ``δf``, against
  ``∫ z δf``.

All are second order, with two known exceptions:

* With ``ghost_rule="standard"``, the face gradient is first order where two
  Dirichlet walls meet.
* With ``ghost_rule="pde"``, ``f`` in the cells next to a wall also sets the
  wall's normal derivatives, so the gradient in those cells is off by a fixed
  fraction of ``z``: O(h) next to Dirichlet walls (where ``z`` vanishes) and
  O(1) next to Neumann walls. Away from the walls and along smooth directions
  it is still second order.

Run from the project root (these are not collected by pytest):

    uv run python python/tests/integration/convergence.py            # all studies
    uv run python python/tests/integration/convergence.py --quick    # fewer refinements
    uv run python python/tests/integration/convergence.py -k 2d      # studies whose name contains "2d"

The exit status is 1 if any observed rate is below its expected rate minus
``--slack``.
"""

import argparse
import itertools
import math
import sys
from dataclasses import dataclass

import torch

from orthel.boundary import Dirichlet, Neumann, Periodic
from orthel.uniform_grid_solver import UniformGridSolver

DTYPE = torch.float64

# Distinct wavenumbers per axis: with equal ones the first-order corner errors
# of the standard Dirichlet ghost cells cancel by symmetry.
WAVENUMBERS = (1.7, 2.9, 2.3)
PHASES = (0.3, 1.1, -0.4)

AXIS_TYPES = {
    "P": Periodic(),
    "N": Neumann(),
    "D": Dirichlet(),
    "N/D": (Neumann(), Dirichlet()),
    "D/N": (Dirichlet(), Neumann()),
}

# name -> (coefficients per axis, shift)
OPERATORS = {
    "poisson": ((1.0, 1.0, 1.0), 0.0),
    "screened": ((2.0, 0.5, 1.0), -5.0),
    "helmholtz": ((1.0, 1.5, 0.7), 7.3),
}

GRIDS = {
    1: dict(base=(16,), lengths=(1.0,), levels=6),
    2: dict(base=(16, 24), lengths=(1.0, 1.5), levels=5),
    3: dict(base=(8, 10, 12), lengths=(1.0, 1.25, 1.5), levels=4),
}

BC_CASES = {
    1: [("P",), ("N",), ("D",), ("N/D",), ("D/N",)],
    2: [
        ("P", "P"),
        ("N", "N"),
        ("D", "D"),
        ("D/N", "P"),
        ("N/D", "D/N"),
        ("D", "N"),
        ("P", "D"),
    ],
    3: [("P", "N/D", "D"), ("N", "N", "N"), ("D/N", "P", "D")],
}


@dataclass
class Study:
    name: str
    bc: tuple
    coefficients: tuple
    shift: float
    ghost_rule: str
    lengths: tuple
    base: tuple
    levels: int

    @property
    def types(self):
        return [b if isinstance(b, tuple) else (b, b) for b in self.bc]

    @property
    def dirichlet_corner(self):
        return sum(any(isinstance(s, Dirichlet) for s in pair) for pair in self.types) >= 2

    @property
    def null_mode(self):
        return self.shift == 0 and not any(isinstance(s, Dirichlet) for pair in self.types for s in pair)

    @property
    def has_neumann(self):
        return any(isinstance(s, Neumann) for pair in self.types for s in pair)

    @property
    def has_walls(self):
        return not all(isinstance(pair[0], Periodic) for pair in self.types)

    def expected(self):
        grad = 1.0 if self.ghost_rule == "standard" and self.dirichlet_corner else 2.0
        adjoint = 2.0
        if self.ghost_rule == "pde" and self.has_walls:
            adjoint = 0.0 if self.has_neumann else 1.0
        return {"u": 2.0, "grad": grad, "adjoint": adjoint, "adj-int": 2.0, "adj-dir": 2.0}


def manufactured(study):
    """Per-axis factors of ``u = Π_i cos(k_i x_i + φ_i)``; periodic axes use ``k = 2π / L``."""
    ks = []
    for i, (pair, length) in enumerate(zip(study.types, study.lengths)):
        periodic = isinstance(pair[0], Periodic)
        ks.append(2 * math.pi / length if periodic else WAVENUMBERS[i])
    return ks, PHASES[: len(ks)]


def adjoint_factors(study):
    """Per-axis ``(k, φ)`` of ``z = Π_i cos(k_i x_i + φ_i)`` meeting homogeneous boundaries."""
    factors = []
    for i, (pair, length) in enumerate(zip(study.types, study.lengths)):
        low, high = (type(s) for s in pair)
        if low is Periodic:
            factors.append((2 * math.pi / length, PHASES[i]))
        elif (low, high) == (Neumann, Neumann):
            factors.append((math.pi / length, 0.0))
        elif (low, high) == (Dirichlet, Dirichlet):
            factors.append((math.pi / length, -math.pi / 2))
        elif (low, high) == (Neumann, Dirichlet):
            factors.append((math.pi / (2 * length), 0.0))
        else:
            factors.append((math.pi / (2 * length), -math.pi / 2))
    return factors


def product(factors, points):
    grids = torch.meshgrid(*points, indexing="ij")
    out = torch.ones((), dtype=DTYPE)
    for (k, phase), x in zip(factors, grids):
        out = out * torch.cos(k * x + phase)
    return out


def derivative(factors, points, axis):
    grids = torch.meshgrid(*points, indexing="ij")
    out = torch.ones((), dtype=DTYPE)
    for i, ((k, phase), x) in enumerate(zip(factors, grids)):
        out = out * (-k * torch.sin(k * x + phase) if i == axis else torch.cos(k * x + phase))
    return out


def symbol(study, factors):
    return study.shift - sum(a * k**2 for a, (k, _) in zip(study.coefficients, factors))


def exact_bc(study, solver, factors):
    points = solver.coords(dtype=DTYPE)
    bc = []
    for axis, pair in enumerate(solver.bc):
        if isinstance(pair[0], Periodic):
            bc.append(Periodic())
            continue
        sides = []
        for side, boundary in enumerate(pair):
            face = list(points)
            face[axis] = torch.tensor([0.0 if side == 0 else study.lengths[axis]], dtype=DTYPE)
            if isinstance(boundary, Dirichlet):
                sides.append(Dirichlet(product(factors, face).squeeze(axis)))
            else:
                sign = 1 if side == 1 else -1
                sides.append(Neumann(sign * derivative(factors, face, axis).squeeze(axis)))
        bc.append(tuple(sides))
    return bc


def remove_mean(x):
    return x - x.mean()


def errors_at(study, level):
    shape = tuple(n * 2**level for n in study.base)
    solver = UniformGridSolver(
        shape, study.lengths, study.bc, coefficients=study.coefficients,
        shift=study.shift, ghost_rule=study.ghost_rule,
    )
    ndim = len(shape)
    points = solver.coords(dtype=DTYPE)
    factors = list(zip(*manufactured(study)))
    expected = product(factors, points)
    bc = exact_bc(study, solver, factors)

    f = (symbol(study, factors) * expected).requires_grad_()
    u = solver(f, bc)
    error = u.detach() - expected
    u_error = (remove_mean(error) if study.null_mode else error).abs().max().item()

    # Face gradients along each axis, padding with the solver's ghost cells.
    ghosts = solver.ghost_cells(u.detach(), f.detach(), bc)
    grad_error = 0.0
    for axis in range(ndim):
        low, high = ghosts[axis]
        padded = torch.cat([low, u.detach(), high], dim=axis)
        du = torch.diff(padded, dim=axis) / solver.spacings[axis]
        faces = list(points)
        faces[axis] = torch.arange(shape[axis] + 1, dtype=DTYPE) * solver.spacings[axis]
        exact = derivative(factors, faces, axis)
        if isinstance(solver.bc[axis][0], Periodic):
            du, exact = du.narrow(axis, 0, shape[axis]), exact.narrow(axis, 0, shape[axis])
        grad_error = max(grad_error, (du - exact).abs().max().item())

    # ∂J/∂f for J = Σ w u approximates the adjoint solution z, where A z = w.
    z_factors = adjoint_factors(study)
    z = product(z_factors, points)
    w = symbol(study, z_factors) * z
    (adjoint,) = torch.autograd.grad((w * u).sum(), f)
    adjoint_error = adjoint - z
    if study.null_mode:
        adjoint_error = remove_mean(adjoint_error)
    interior = adjoint_error[tuple(slice(2, -2) for _ in shape)]

    # Directional derivatives along smooth δf, against ∫ z δf (cell sums are
    # O(h²) quadrature). Errors from the interior and from the cells next to
    # walls can cancel along one direction, so take the worst of several.
    grids = torch.meshgrid(*points, indexing="ij")
    cell_volume = math.prod(solver.spacings)
    directional = 0.0
    for direction in (
        math.prod(torch.exp(-x) for x in grids),
        math.prod(1 + x**2 for x in grids),
        math.prod(torch.cos(3 * x + 0.5) for x in grids),
    ):
        if study.null_mode:
            direction = remove_mean(direction)
        error = abs(((adjoint - z) * direction).sum().item()) * cell_volume
        directional = max(directional, error)

    return shape, {
        "u": u_error,
        "grad": grad_error,
        "adjoint": adjoint_error.abs().max().item(),
        "adj-int": interior.abs().max().item(),
        "adj-dir": directional,
    }


def run(study):
    rows = [errors_at(study, level) for level in range(study.levels)]
    rates = {
        metric: math.log2(rows[-2][1][metric] / rows[-1][1][metric]) for metric in rows[0][1]
    }
    return rows, rates


def build_studies(quick):
    studies = []
    for ndim, cases in BC_CASES.items():
        grid = GRIDS[ndim]
        levels = grid["levels"] - (1 if quick else 0)
        for names, (op, (coefficients, shift)), rule in itertools.product(
            cases, OPERATORS.items(), ("standard", "pde")
        ):
            bc = tuple(AXIS_TYPES[n] for n in names)
            studies.append(Study(
                name=f"{ndim}d {'-'.join(names)} {op} {rule}",
                bc=bc, coefficients=coefficients[:ndim], shift=shift, ghost_rule=rule,
                lengths=grid["lengths"], base=grid["base"], levels=levels,
            ))
    return studies


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="one refinement fewer per study")
    parser.add_argument("-k", metavar="TEXT", default="", help="only run studies whose name contains TEXT")
    parser.add_argument("--slack", type=float, default=0.15, help="allowed shortfall below the expected rate")
    parser.add_argument("-v", "--verbose", action="store_true", help="print the errors at every level")
    args = parser.parse_args()

    studies = [s for s in build_studies(args.quick) if args.k in s.name]
    failures = []
    metrics = list(build_studies(True)[0].expected())
    print(f"{'study':40s} {'finest':>14s} " + "".join(f"{m:>9s}" for m in metrics))
    for study in studies:
        rows, rates = run(study)
        expected = study.expected()
        failed = [m for m in rates if rates[m] < expected[m] - args.slack]
        marks = {m: ("!" if m in failed else " ") for m in rates}
        finest = "x".join(map(str, rows[-1][0]))
        print(f"{study.name:40s} {finest:>14s} " + "".join(f"{rates[m]:8.2f}{marks[m]}" for m in metrics))
        if args.verbose:
            for shape, errors in rows:
                print(f"    {str(shape):>16s}  " + "  ".join(f"{m} {e:.2e}" for m, e in errors.items()))
        if failed:
            failures.append((study, rates, expected, failed))

    print(f"\n{len(studies) - len(failures)}/{len(studies)} studies at their expected rates")
    for study, rates, expected, failed in failures:
        detail = ", ".join(f"{m} {rates[m]:.2f} < {expected[m]:.1f}" for m in failed)
        print(f"FAIL {study.name}: {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
