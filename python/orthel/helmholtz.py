from __future__ import annotations

from collections.abc import Sequence

import torch

from .boundary import AxisBC, Dirichlet, Periodic, parse_bc
from .operators import divergence, gradient
from .uniform_grid_solver import UniformGridSolver


def _spatial_shape(
    velx: torch.Tensor,
    vely: torch.Tensor,
    bc: AxisBC | Sequence[AxisBC],
) -> tuple[int, int]:
    if velx.ndim < 2 or vely.ndim < 2:
        raise ValueError("velx and vely must include two spatial dimensions")
    boundary = parse_bc(bc, 2)
    (low_y, _), (low_x, _) = boundary
    ny = velx.shape[-2]
    nx = vely.shape[-1]
    if low_y.kind == "periodic":
        if vely.shape[-2] != ny:
            raise ValueError(
                "periodic y boundary requires vely shape (*, ny, nx); "
                f"got {tuple(vely.shape)}"
            )
    elif vely.shape[-2] != ny + 1:
        raise ValueError(
            "wall y boundary requires vely shape (*, ny + 1, nx); "
            f"got {tuple(vely.shape)}"
        )
    if low_x.kind == "periodic":
        if velx.shape[-1] != nx:
            raise ValueError(
                "periodic x boundary requires velx shape (*, ny, nx); "
                f"got {tuple(velx.shape)}"
            )
    elif velx.shape[-1] != nx + 1:
        raise ValueError(
            "wall x boundary requires velx shape (*, ny, nx + 1); "
            f"got {tuple(velx.shape)}"
        )
    if velx.shape[:-2] != vely.shape[:-2]:
        raise ValueError(
            f"velx batch shape {tuple(velx.shape[:-2])} does not match "
            f"vely batch shape {tuple(vely.shape[:-2])}"
        )
    return ny, nx


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
        ``(psi, phi)`` on the cell-centred grid, each with shape
        ``(*batch, ny, nx)``.

    ``phi`` is the scalar dilational potential. ``psi`` is reconstructed from
    the solenoidal residual and uses periodic boundaries on periodic axes and
    Dirichlet boundaries on wall axes.
    """
    ny, nx = _spatial_shape(velx, vely, bc)
    solver = UniformGridSolver(
        (ny, nx),
        lengths,
        bc,
        coefficients=1.0,
        shift=0.0,
        ghost_rule="standard",
    )
    spacings = solver.spacings
    boundary = solver.boundary_types

    div = divergence(velx, vely, spacings, bc)
    phi = solver(div)
    gx, gy = gradient(phi, spacings, bc)
    solenoidal_x = velx - gx
    solenoidal_y = vely - gy

    if boundary[1][0] == "periodic":
        ux_center = solenoidal_x
    else:
        ux_center = 0.5 * (solenoidal_x[..., :, :-1] + solenoidal_x[..., :, 1:])
    if boundary[0][0] == "periodic":
        uy_center = solenoidal_y
    else:
        uy_center = 0.5 * (solenoidal_y[..., :-1, :] + solenoidal_y[..., 1:, :])

    if boundary[1][0] == "periodic":
        duydx = (uy_center.roll(-1, -1) - uy_center) / spacings[1]
    elif nx > 1:
        duydx = torch.empty_like(uy_center)
        duydx[..., :, 0] = (uy_center[..., :, 1] - uy_center[..., :, 0]) / spacings[1]
        duydx[..., :, -1] = (uy_center[..., :, -1] - uy_center[..., :, -2]) / spacings[
            1
        ]
        if nx > 2:
            duydx[..., :, 1:-1] = (uy_center[..., :, 2:] - uy_center[..., :, :-2]) / (
                2 * spacings[1]
            )
    else:
        duydx = torch.zeros_like(uy_center)

    if boundary[0][0] == "periodic":
        duxdy = (ux_center.roll(-1, -2) - ux_center) / spacings[0]
    elif ny > 1:
        duxdy = torch.empty_like(ux_center)
        duxdy[..., 0, :] = (ux_center[..., 1, :] - ux_center[..., 0, :]) / spacings[0]
        duxdy[..., -1, :] = (ux_center[..., -1, :] - ux_center[..., -2, :]) / spacings[
            0
        ]
        if ny > 2:
            duxdy[..., 1:-1, :] = (ux_center[..., 2:, :] - ux_center[..., :-2, :]) / (
                2 * spacings[0]
            )
    else:
        duxdy = torch.zeros_like(ux_center)

    curl = duydx - duxdy

    psi_boundary = tuple(
        Periodic() if low == "periodic" else Dirichlet() for low, _ in boundary
    )
    psi_solver = UniformGridSolver(
        (ny, nx),
        solver.lengths,
        psi_boundary,
        coefficients=1.0,
        shift=0.0,
        ghost_rule="standard",
    )
    psi = psi_solver(-curl)
    return psi, phi
