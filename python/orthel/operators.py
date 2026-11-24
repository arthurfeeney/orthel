from __future__ import annotations

from collections.abc import Sequence

import torch

from .boundary import AxisBC, parse_bc


def divergence(
    facex: torch.Tensor,
    facey: torch.Tensor,
    spacings: float | Sequence[float],
    bc: AxisBC | Sequence[AxisBC] | None = None,
) -> torch.Tensor:
    """Return ``∂ux/∂x + ∂uy/∂y`` on a 2D cell-centred grid.

    ``facex`` has shape ``(*batch, ny, nx + 1)`` and ``facey`` has shape
    ``(*batch, ny + 1, nx)``. ``spacings`` is ``(dy, dx)``.
    Pass ``bc`` as ``(y_bc, x_bc)`` for periodic axes. If omitted, wall
    face shapes are assumed.
    """
    if facex.ndim < 2 or facey.ndim < 2:
        raise ValueError("facex and facey must include two spatial dimensions")

    if isinstance(spacings, (int, float)):
        dy = dx = float(spacings)
    else:
        if len(spacings) != 2:
            raise ValueError(f"expected spacings (dy, dx), got {spacings}")
        dy, dx = (float(h) for h in spacings)
    if dy <= 0 or dx <= 0:
        raise ValueError(f"spacings must be positive, got {(dy, dx)}")

    ny = facex.shape[-2]
    nx = facey.shape[-1]
    if bc is None:
        boundary = ((None, None), (None, None))
    else:
        boundary = parse_bc(bc, 2)

    (low_y, high_y), (low_x, high_x) = boundary
    expected_facex = nx + 1 if low_x is None or low_x.kind != "periodic" else nx
    expected_facey = ny + 1 if low_y is None or low_y.kind != "periodic" else ny

    if facex.shape[-2:] != (ny, expected_facex):
        raise ValueError(
            f"expected facex spatial shape ({ny}, {expected_facex}), "
            f"got {tuple(facex.shape[-2:])}"
        )
    if facey.shape[-2:] != (expected_facey, nx):
        raise ValueError(
            f"expected facey spatial shape ({expected_facey}, {nx}), "
            f"got {tuple(facey.shape[-2:])}"
        )
    if facex.shape[:-2] != facey.shape[:-2]:
        raise ValueError(
            f"face batch shapes differ: {tuple(facex.shape[:-2])} "
            f"and {tuple(facey.shape[:-2])}"
        )

    if low_x is not None and low_x.kind == "periodic":
        du_dx = (facex.roll(-1, -1) - facex) / dx
    else:
        du_dx = (facex[..., :, 1:] - facex[..., :, :-1]) / dx

    if low_y is not None and low_y.kind == "periodic":
        du_dy = (facey.roll(-1, -2) - facey) / dy
    else:
        du_dy = (facey[..., 1:, :] - facey[..., :-1, :]) / dy
    return du_dx + du_dy


def curl(
    facex: torch.Tensor,
    facey: torch.Tensor,
    spacings: float | Sequence[float],
    bc: AxisBC | Sequence[AxisBC] | None = None,
) -> torch.Tensor:
    """Return ``∂uy/∂x - ∂ux/∂y`` on a 2D cell-centred grid.

    ``facex`` has shape ``(*batch, ny, nx + 1)`` and ``facey`` has shape
    ``(*batch, ny + 1, nx)`` for wall axes. Periodic axes use ``nx`` and
    ``ny`` faces respectively. ``spacings`` is ``(dy, dx)``.
    """
    if facex.ndim < 2 or facey.ndim < 2:
        raise ValueError("facex and facey must include two spatial dimensions")

    if isinstance(spacings, (int, float)):
        dy = dx = float(spacings)
    else:
        if len(spacings) != 2:
            raise ValueError(f"expected spacings (dy, dx), got {spacings}")
        dy, dx = (float(h) for h in spacings)
    if dy <= 0 or dx <= 0:
        raise ValueError(f"spacings must be positive, got {(dy, dx)}")

    ny = facex.shape[-2]
    nx = facey.shape[-1]
    if bc is None:
        boundary = ((None, None), (None, None))
    else:
        boundary = parse_bc(bc, 2)

    (low_y, _), (low_x, _) = boundary
    if low_x is not None and low_x.kind == "periodic":
        ux_center = facex
    else:
        if facex.shape[-2:] != (ny, nx + 1):
            raise ValueError(
                f"wall x boundary requires facex shape (*, {ny}, {nx + 1}), "
                f"got {tuple(facex.shape[-2:])}"
            )
        ux_center = 0.5 * (facex[..., :, :-1] + facex[..., :, 1:])

    if low_y is not None and low_y.kind == "periodic":
        uy_center = facey
    else:
        if facey.shape[-2:] != (ny + 1, nx):
            raise ValueError(
                f"wall y boundary requires facey shape (*, {ny + 1}, {nx}), "
                f"got {tuple(facey.shape[-2:])}"
            )
        uy_center = 0.5 * (facey[..., :-1, :] + facey[..., 1:, :])

    if facex.shape[:-2] != facey.shape[:-2]:
        raise ValueError(
            f"face batch shapes differ: {tuple(facex.shape[:-2])} "
            f"and {tuple(facey.shape[:-2])}"
        )

    if low_x is not None and low_x.kind == "periodic":
        duydx = (uy_center - uy_center.roll(1, -1)) / dx
    elif nx > 1:
        duydx = torch.empty_like(uy_center)
        duydx[..., :, 0] = (uy_center[..., :, 1] - uy_center[..., :, 0]) / dx
        duydx[..., :, -1] = (uy_center[..., :, -1] - uy_center[..., :, -2]) / dx
        if nx > 2:
            duydx[..., :, 1:-1] = (uy_center[..., :, 2:] - uy_center[..., :, :-2]) / (
                2 * dx
            )
    else:
        duydx = torch.zeros_like(uy_center)

    if low_y is not None and low_y.kind == "periodic":
        duxdy = (ux_center - ux_center.roll(1, -2)) / dy
    elif ny > 1:
        duxdy = torch.empty_like(ux_center)
        duxdy[..., 0, :] = (ux_center[..., 1, :] - ux_center[..., 0, :]) / dy
        duxdy[..., -1, :] = (ux_center[..., -1, :] - ux_center[..., -2, :]) / dy
        if ny > 2:
            duxdy[..., 1:-1, :] = (ux_center[..., 2:, :] - ux_center[..., :-2, :]) / (
                2 * dy
            )
    else:
        duxdy = torch.zeros_like(ux_center)

    return duydx - duxdy


def gradient(
    grid: torch.Tensor,
    spacings: float | Sequence[float],
    bc: AxisBC | Sequence[AxisBC],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map a 2D cell-centred grid to x/y face-normal gradients.

    ``grid`` has shape ``(*batch, ny, nx)``. Returns ``(facex, facey)`` with
    shapes ``(*batch, ny, nx + 1)`` and ``(*batch, ny + 1, nx)``.

    ``bc`` follows :func:`orthel.boundary.parse_bc`: ``(y_bc, x_bc)``.
    Neumann wall faces receive zero normal gradient. Dirichlet walls receive
    the same anti-mirrored ghost-cell gradient used for pressure fields.
    """
    if grid.ndim < 2:
        raise ValueError("grid must include two spatial dimensions")

    if isinstance(spacings, (int, float)):
        dy = dx = float(spacings)
    else:
        if len(spacings) != 2:
            raise ValueError(f"expected spacings (dy, dx), got {spacings}")
        dy, dx = (float(h) for h in spacings)
    if dy <= 0 or dx <= 0:
        raise ValueError(f"spacings must be positive, got {(dy, dx)}")

    boundary = parse_bc(bc, 2)
    (low_y, high_y), (low_x, high_x) = boundary

    if low_y.kind == "periodic":
        facey = (grid - grid.roll(1, -2)) / dy
    else:
        interior = (grid[..., 1:, :] - grid[..., :-1, :]) / dy
        low_face = (
            2 * grid[..., :1, :] / dy
            if low_y.kind == "dirichlet"
            else torch.zeros_like(grid[..., :1, :])
        )
        high_face = (
            -2 * grid[..., -1:, :] / dy
            if high_y.kind == "dirichlet"
            else torch.zeros_like(grid[..., -1:, :])
        )
        facey = torch.cat([low_face, interior, high_face], dim=-2)

    if low_x.kind == "periodic":
        facex = (grid - grid.roll(1, -1)) / dx
    else:
        interior = (grid[..., :, 1:] - grid[..., :, :-1]) / dx
        low_face = (
            2 * grid[..., :, :1] / dx
            if low_x.kind == "dirichlet"
            else torch.zeros_like(grid[..., :, :1])
        )
        high_face = (
            -2 * grid[..., :, -1:] / dx
            if high_x.kind == "dirichlet"
            else torch.zeros_like(grid[..., :, -1:])
        )
        facex = torch.cat([low_face, interior, high_face], dim=-1)

    return facex, facey


def vel_from_streamfunction(
    streamfunction: torch.Tensor,
    spacings: float | Sequence[float],
    bc: AxisBC | Sequence[AxisBC],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map a 2D cell-centred streamfunction to face-normal velocities.

    Uses ``ux = ∂ψ/∂y`` and ``uy = -∂ψ/∂x``.

    ``streamfunction`` has shape ``(*batch, ny, nx)``. Returns ``(facex, facey)``
    with shapes ``(*batch, ny, nx + 1)`` and ``(*batch, ny + 1, nx)``.
    """
    if streamfunction.ndim < 2:
        raise ValueError("streamfunction must include two spatial dimensions")

    if isinstance(spacings, (int, float)):
        dy = dx = float(spacings)
    else:
        if len(spacings) != 2:
            raise ValueError(f"expected spacings (dy, dx), got {spacings}")
        dy, dx = (float(h) for h in spacings)
    if dy <= 0 or dx <= 0:
        raise ValueError(f"spacings must be positive, got {(dy, dx)}")

    boundary = parse_bc(bc, 2)
    (low_y, _), (low_x, _) = boundary

    if low_y.kind == "periodic":
        dpsi_dy = (streamfunction.roll(-1, -2) - streamfunction) / dy
    else:
        dpsi_dy = torch.empty_like(streamfunction)
        dpsi_dy[..., 0, :] = (
            streamfunction[..., 1, :] - streamfunction[..., 0, :]
        ) / dy
        dpsi_dy[..., -1, :] = (
            streamfunction[..., -1, :] - streamfunction[..., -2, :]
        ) / dy
        if streamfunction.shape[-2] > 2:
            dpsi_dy[..., 1:-1, :] = (
                streamfunction[..., 2:, :] - streamfunction[..., :-2, :]
            ) / (2 * dy)

    if low_x.kind == "periodic":
        dpsi_dx = (streamfunction.roll(-1, -1) - streamfunction) / dx
    else:
        dpsi_dx = torch.empty_like(streamfunction)
        dpsi_dx[..., :, 0] = (
            streamfunction[..., :, 1] - streamfunction[..., :, 0]
        ) / dx
        dpsi_dx[..., :, -1] = (
            streamfunction[..., :, -1] - streamfunction[..., :, -2]
        ) / dx
        if streamfunction.shape[-1] > 2:
            dpsi_dx[..., :, 1:-1] = (
                streamfunction[..., :, 2:] - streamfunction[..., :, :-2]
            ) / (2 * dx)

    if low_x.kind == "periodic":
        facex = dpsi_dy
    else:
        facex = torch.cat(
            [
                dpsi_dy[..., :, :1],
                0.5 * (dpsi_dy[..., :, :-1] + dpsi_dy[..., :, 1:]),
                dpsi_dy[..., :, -1:],
            ],
            dim=-1,
        )

    if low_y.kind == "periodic":
        facey = -dpsi_dx
    else:
        facey = torch.cat(
            [
                -dpsi_dx[..., :1, :],
                -0.5 * (dpsi_dx[..., :-1, :] + dpsi_dx[..., 1:, :]),
                -dpsi_dx[..., -1:, :],
            ],
            dim=-2,
        )

    return facex, facey
