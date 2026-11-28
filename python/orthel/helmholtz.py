"""Helmholtz decomposition of 2D MAC-grid velocities."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .boundary import AxisBC, Dirichlet, Periodic
from .mac_operators import face_shape, mac_curl, mac_divergence, mac_gradient
from .uniform_grid_solver import UniformGridSolver


def helmholtz(
    velx: torch.Tensor,
    vely: torch.Tensor,
    *,
    lengths: float | Sequence[float] = 1.0,
    bc: AxisBC | Sequence[AxisBC] = (Periodic(), Periodic()),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose 2D face velocities into streamfunction and potential fields.

    The Helmholtz convention used here is
    ``u = (∂psi/∂y, -∂psi/∂x) + (∂phi/∂x, ∂phi/∂y) = curl(psi) + grad(phi)``.

    Args:
        velx: x-normal face velocity with shape ``(*batch, ny, nx + 1)``.
        vely: y-normal face velocity with shape ``(*batch, ny + 1, nx)``.
        lengths: physical lengths ``(Ly, Lx)``.
        bc: boundary entries ``(y_bc, x_bc)``.

    Returns:
        ``(psi, phi)``. ``phi`` is cell-centred with shape ``(*batch, ny, nx)``.
        ``psi`` lives on the grid nodes (cell corners, see
        :mod:`orthel.mac_operators`) with shape ``(*batch, ny + 1, nx + 1)`` for
        two wall axes, and ``ny`` / ``nx`` along periodic axes, so that
        ``mac_vel_from_streamfunction(psi) + mac_gradient(phi)`` rebuilds the input.

    ``phi`` solves ``Δphi = div(u)`` with ``bc``. ``psi`` solves the node
    Laplacian ``Δpsi = -curl(u - grad(phi))`` with ``psi = 0`` on walls and zero
    mean when both axes are periodic.

    Limitation: the harmonic part of ``u`` (a uniform flow on a periodic
    domain, net flow along a channel, or wall flux not captured by ``phi``) is
    in neither field, so the reconstruction is exact only when ``u`` has none.
    """
    ny, nx = face_shape(velx, vely, bc)
    solver = UniformGridSolver(
        (ny, nx),
        lengths,
        bc,
        coefficients=1.0,
        shift=0.0,
        ghost_rule="standard",
    )
    spacings = solver.spacings

    phi = solver(mac_divergence(velx, vely, spacings, bc))
    gx, gy = mac_gradient(phi, spacings, bc)
    vorticity = mac_curl(velx - gx, vely - gy, spacings, bc)

    psi_solver = UniformGridSolver(
        (ny, nx),
        solver.lengths,
        tuple(
            Periodic() if low == "periodic" else Dirichlet()
            for low, _ in solver.boundary_types
        ),
        coefficients=1.0,
        shift=0.0,
        grid="node",
    )
    # curl only covers the unknown nodes; the solver ignores f on wall nodes.
    pad = []
    for low, _ in reversed(solver.boundary_types):
        pad += [0, 0] if low == "periodic" else [1, 1]
    psi = psi_solver(-torch.nn.functional.pad(vorticity, pad))
    return psi, phi
