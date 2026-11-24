"""
Solve various PDEs using PyTorch and orthel's UniformGridSolver, which works
for a range of elliptic PDEs: poisson, screened poisson, helmholz, etc.

Run from the project root with

    uv run --group examples python examples/toy_problems.py

or pass ``--save toy_problems.png`` to write the figure instead of showing it.
"""

import argparse

import matplotlib.pyplot as plt
import torch

from orthel.boundary import Dirichlet, Neumann, Periodic
from orthel.uniform_grid_solver import UniformGridSolver

SHAPE = (128, 128)
LENGTHS = (1.0, 1.0)


def gaussian(solver, center, width):
    x, y = torch.meshgrid(*solver.coords(dtype=torch.float64), indexing="ij")
    return torch.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / (2 * width**2))


def periodic_dipole():
    """Positive and negative charge on a torus: ∇²u = f."""
    solver = UniformGridSolver(SHAPE, LENGTHS, Periodic())
    f = gaussian(solver, (0.35, 0.5), 0.05) - gaussian(solver, (0.65, 0.5), 0.05)
    return solver, f, solver(f)


def heated_lid():
    """Laplace's equation in a box held at u = 0 with the top wall at u = 1."""
    solver = UniformGridSolver(SHAPE, LENGTHS, [Dirichlet(), (Dirichlet(), Dirichlet(1.0))])
    f = torch.zeros(SHAPE, dtype=torch.float64)
    return solver, f, solver(f)


def insulated_source_sink():
    """A source and a sink in an insulated box (zero normal flux on every wall)."""
    solver = UniformGridSolver(SHAPE, LENGTHS, Neumann())
    f = gaussian(solver, (0.25, 0.25), 0.06) - gaussian(solver, (0.75, 0.75), 0.06)
    return solver, f, solver(f)


def screened_source():
    """Screened Poisson ∇²u - κ²u = f: the response decays over a length 1/κ."""
    solver = UniformGridSolver(SHAPE, LENGTHS, Dirichlet(), shift=-(20.0**2))
    f = -gaussian(solver, (0.5, 0.5), 0.02)
    return solver, f, solver(f)


def channel_flow_pressure():
    """Channel periodic in x with walls in y, a wall-normal inflow on the bottom
    balanced by an outflow on the top."""
    solver = UniformGridSolver(SHAPE, LENGTHS, (Periodic(), Neumann()))
    x = solver.coords(dtype=torch.float64)[0]
    flux = torch.sin(2 * torch.pi * x)
    f = torch.zeros(SHAPE, dtype=torch.float64)
    # Values that depend on the grid can be passed per call, with the same types.
    return solver, f, solver(f, [Periodic(), (Neumann(-flux), Neumann(flux))])


def helmholtz_waves():
    """Helmholtz ∇²u + k²u = f driven by a small source in a Dirichlet box.

    The box has no damping or outgoing boundary, so the response is a pattern of
    standing waves rather than rings spreading from the source.
    """
    solver = UniformGridSolver(SHAPE, LENGTHS, Dirichlet(), shift=30.0**2)
    f = gaussian(solver, (0.5, 0.5), 0.015)
    return solver, f, solver(f)


PROBLEMS = [
    ("Periodic dipole", periodic_dipole),
    ("Heated lid (Laplace)", heated_lid),
    ("Insulated source/sink", insulated_source_sink),
    ("Screened Poisson", screened_source),
    ("Channel with wall flux", channel_flow_pressure),
    ("Helmholtz waves", helmholtz_waves),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--save", metavar="PATH", help="write the figure instead of showing it")
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), constrained_layout=True)
    for ax, (title, problem) in zip(axes.flat, PROBLEMS):
        solver, _, u = problem()
        extent = (0, solver.lengths[0], 0, solver.lengths[1])
        limit = u.abs().max().item()
        # Signed fields use a diverging map centred on zero.
        if u.min() < 0 < u.max():
            image = ax.imshow(u.T, origin="lower", extent=extent, cmap="RdBu_r", vmin=-limit, vmax=limit)
        else:
            image = ax.imshow(u.T, origin="lower", extent=extent, cmap="viridis")
        ax.contour(u.T, levels=12, colors="k", linewidths=0.5, alpha=0.5, extent=extent)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, shrink=0.8)

    if args.save:
        fig.savefig(args.save, dpi=120)
    else:
        plt.show()


if __name__ == "__main__":
    main()
