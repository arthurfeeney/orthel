"""Staggered (MAC) finite-difference operators on a 2D uniform grid.

Grid layout, with axis order ``(y, x)`` throughout:

* cell-centred scalars (``phi``) have shape ``(*batch, ny, nx)``;
* x-normal face values (``facex``) sit on the left face of each cell and have
  shape ``(*batch, ny, nx + 1)`` when x is a wall axis and ``(*batch, ny, nx)``
  when x is periodic (the face at ``x = Lx`` is the one at ``x = 0``);
* y-normal face values (``facey``) likewise have ``ny + 1`` rows on a wall y
  axis and ``ny`` on a periodic one;
* node (cell-corner) values, used for the streamfunction, sit at the lower-left
  corner of each cell and have ``n + 1`` entries along a wall axis, including
  both walls, and ``n`` along a periodic one.

With this layout ``mac_divergence(mac_vel_from_streamfunction(psi)) == 0`` and
``mac_curl(mac_gradient(phi)) == 0`` hold exactly for every boundary type.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .boundary import AxisBC, Boundary, parse_bc


def _parse_spacings(spacings: float | Sequence[float]) -> tuple[float, float]:
    if isinstance(spacings, (int, float)):
        dy = dx = float(spacings)
    else:
        if len(spacings) != 2:
            raise ValueError(f"expected spacings (dy, dx), got {spacings}")
        dy, dx = (float(h) for h in spacings)
    if not (0 < dy < float("inf") and 0 < dx < float("inf")):
        raise ValueError(f"spacings must be positive and finite, got {(dy, dx)}")
    return dy, dx


def _periodic_axes(bc: AxisBC | Sequence[AxisBC] | None) -> tuple[bool, bool]:
    """Whether the ``(y, x)`` axes are periodic. ``None`` means walls."""
    if bc is None:
        return False, False
    (low_y, _), (low_x, _) = parse_bc(bc, 2)
    return low_y.kind == "periodic", low_x.kind == "periodic"


def face_shape(
    facex: torch.Tensor,
    facey: torch.Tensor,
    bc: AxisBC | Sequence[AxisBC] | None = None,
) -> tuple[int, int]:
    """Validate a pair of face fields and return the cell grid shape ``(ny, nx)``."""
    if facex.ndim < 2 or facey.ndim < 2:
        raise ValueError("facex and facey must include two spatial dimensions")
    periodic_y, periodic_x = _periodic_axes(bc)
    ny = facex.shape[-2]
    nx = facey.shape[-1]
    expected_facex = (ny, nx if periodic_x else nx + 1)
    expected_facey = (ny if periodic_y else ny + 1, nx)
    if tuple(facex.shape[-2:]) != expected_facex:
        kind = "periodic" if periodic_x else "wall"
        raise ValueError(
            f"{kind} x boundary requires facex spatial shape {expected_facex}, "
            f"got {tuple(facex.shape[-2:])}"
        )
    if tuple(facey.shape[-2:]) != expected_facey:
        kind = "periodic" if periodic_y else "wall"
        raise ValueError(
            f"{kind} y boundary requires facey spatial shape {expected_facey}, "
            f"got {tuple(facey.shape[-2:])}"
        )
    if facex.shape[:-2] != facey.shape[:-2]:
        raise ValueError(
            f"face batch shapes differ: {tuple(facex.shape[:-2])} "
            f"and {tuple(facey.shape[:-2])}"
        )
    if ny < 1 or nx < 1:
        raise ValueError(f"grid must have at least one cell, got {(ny, nx)}")
    return ny, nx


def mac_divergence(
    facex: torch.Tensor,
    facey: torch.Tensor,
    spacings: float | Sequence[float],
    bc: AxisBC | Sequence[AxisBC] | None = None,
) -> torch.Tensor:
    """Return ``∂ux/∂x + ∂uy/∂y`` at cell centres.

    ``facex`` and ``facey`` follow the face layout in the module docstring.
    ``spacings`` is ``(dy, dx)``. ``bc`` (``(y_bc, x_bc)``) only selects
    periodic or wall layouts; if omitted, both axes are walls.
    """
    dy, dx = _parse_spacings(spacings)
    face_shape(facex, facey, bc)
    periodic_y, periodic_x = _periodic_axes(bc)

    if periodic_x:
        du_dx = (facex.roll(-1, -1) - facex) / dx
    else:
        du_dx = (facex[..., :, 1:] - facex[..., :, :-1]) / dx

    if periodic_y:
        du_dy = (facey.roll(-1, -2) - facey) / dy
    else:
        du_dy = (facey[..., 1:, :] - facey[..., :-1, :]) / dy
    return du_dx + du_dy


def mac_curl(
    facex: torch.Tensor,
    facey: torch.Tensor,
    spacings: float | Sequence[float],
    bc: AxisBC | Sequence[AxisBC] | None = None,
) -> torch.Tensor:
    """Return the vorticity ``∂uy/∂x - ∂ux/∂y`` at the grid nodes.

    The circulation around a node needs the faces on all four sides of it, so
    nodes on a wall (where face-normal velocities carry no tangential
    information) are left out: the result has shape
    ``(*batch, ny - 1, nx - 1)`` for two wall axes and ``ny`` / ``nx`` in place
    of ``ny - 1`` / ``nx - 1`` along periodic axes. Entry ``[j, i]`` is the
    node at ``(j + 1) dy, (i + 1) dx`` on wall axes and ``j dy, i dx`` on
    periodic axes.

    ``facex`` and ``facey`` follow the face layout in the module docstring.
    ``spacings`` is ``(dy, dx)`` and ``bc`` is ``(y_bc, x_bc)``; if omitted,
    both axes are walls.
    """
    dy, dx = _parse_spacings(spacings)
    face_shape(facex, facey, bc)
    periodic_y, periodic_x = _periodic_axes(bc)

    # Differences of uy across x and of ux across y, both landing on nodes.
    if periodic_x:
        duy = facey - facey.roll(1, -1)
    else:
        duy = facey[..., :, 1:] - facey[..., :, :-1]
    if periodic_y:
        dux = facex - facex.roll(1, -2)
    else:
        dux = facex[..., 1:, :] - facex[..., :-1, :]

    # Drop the wall nodes along the other axis.
    if not periodic_y:
        duy = duy[..., 1:-1, :]
    if not periodic_x:
        dux = dux[..., :, 1:-1]
    return duy / dx - dux / dy


def _wall_value(
    boundary: Boundary, grid: torch.Tensor, dim: int
) -> torch.Tensor | None:
    """The boundary value broadcast to the face, or ``None`` if it is the number 0.

    Matches :class:`orthel.UniformGridSolver`: ``value`` must broadcast to the
    grid shape with ``dim`` removed.
    """
    value = boundary.value
    if isinstance(value, (int, float)) and value == 0:
        return None
    face = grid.shape[:dim] + grid.shape[dim + 1 :]
    value = torch.as_tensor(value, dtype=grid.dtype, device=grid.device)
    return torch.broadcast_to(value, face).unsqueeze(dim)


def _wall_face_gradient(
    boundary: Boundary,
    edge: torch.Tensor,
    grid: torch.Tensor,
    dim: int,
    h: float,
    side: int,
) -> torch.Tensor:
    """Normal-axis derivative on a wall face from the ghost cell the solver uses.

    Dirichlet ghosts are ``2g - u_edge`` and Neumann ghosts ``u_edge + h q``,
    where ``q`` is the outward normal derivative.
    """
    value = _wall_value(boundary, grid, dim)
    sign = 1.0 if side == 0 else -1.0  # (inside - outside) direction along the axis
    if boundary.kind == "dirichlet":
        face = edge if value is None else edge - value
        return sign * 2 * face / h
    # Neumann: ∂u/∂x = -q on the low wall and +q on the high wall.
    if value is None:
        return torch.zeros_like(edge)
    return -sign * value


def mac_gradient(
    grid: torch.Tensor,
    spacings: float | Sequence[float],
    bc: AxisBC | Sequence[AxisBC],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map a cell-centred field to face-normal gradients ``(∂φ/∂x, ∂φ/∂y)``.

    ``grid`` has shape ``(*batch, ny, nx)``. Returns ``(facex, facey)`` in the
    face layout of the module docstring.

    ``bc`` follows :func:`orthel.boundary.parse_bc`: ``(y_bc, x_bc)``. Wall
    faces use the same ghost cells as :class:`orthel.UniformGridSolver` with
    ``ghost_rule="standard"``, including the boundary values: ``Dirichlet(g)``
    gives ``±2(φ_edge - g)/h`` and ``Neumann(q)`` gives ``∓q`` on the low and
    high faces. So ``mac_divergence(mac_gradient(solver(f)))`` reproduces ``f``.
    """
    if grid.ndim < 2:
        raise ValueError("grid must include two spatial dimensions")
    dy, dx = _parse_spacings(spacings)
    boundary = parse_bc(bc, 2)

    faces = []
    for (low, high), dim, h in zip(boundary, (-2, -1), (dy, dx)):
        dim = grid.ndim + dim
        if low.kind == "periodic":
            faces.append((grid - grid.roll(1, dim)) / h)
            continue
        n = grid.shape[dim]
        interior = (grid.narrow(dim, 1, n - 1) - grid.narrow(dim, 0, n - 1)) / h
        low_face = _wall_face_gradient(low, grid.narrow(dim, 0, 1), grid, dim, h, 0)
        high_face = _wall_face_gradient(
            high, grid.narrow(dim, n - 1, 1), grid, dim, h, 1
        )
        faces.append(torch.cat([low_face, interior, high_face], dim=dim))

    facey, facex = faces
    return facex, facey


def mac_vel_from_streamfunction(
    streamfunction: torch.Tensor,
    spacings: float | Sequence[float],
    bc: AxisBC | Sequence[AxisBC],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map a node streamfunction to face-normal velocities ``(∂ψ/∂y, -∂ψ/∂x)``.

    ``streamfunction`` lives on the grid nodes (see the module docstring): its
    shape is ``(*batch, ny + 1, nx + 1)`` for two wall axes, with ``ny`` /
    ``nx`` in place of ``ny + 1`` / ``nx + 1`` along periodic axes. Returns
    ``(facex, facey)`` in the matching face layout. The result is exactly
    divergence free, and its flux through a wall face is the change of ``ψ``
    along that face, so ``ψ`` constant on a wall means no flow through it.
    """
    if streamfunction.ndim < 2:
        raise ValueError("streamfunction must include two spatial dimensions")
    dy, dx = _parse_spacings(spacings)
    periodic_y, periodic_x = _periodic_axes(bc)
    for name, n, periodic in (
        ("y", streamfunction.shape[-2], periodic_y),
        ("x", streamfunction.shape[-1], periodic_x),
    ):
        minimum = 1 if periodic else 2
        if n < minimum:
            kind = "periodic" if periodic else "wall"
            raise ValueError(
                f"{kind} {name} axis needs at least {minimum} nodes, got {n}"
            )

    psi = streamfunction
    if periodic_y:
        dpsi_dy = psi.roll(-1, -2) - psi
    else:
        dpsi_dy = psi[..., 1:, :] - psi[..., :-1, :]
    if periodic_x:
        dpsi_dx = psi.roll(-1, -1) - psi
    else:
        dpsi_dx = psi[..., :, 1:] - psi[..., :, :-1]
    return dpsi_dy / dy, -dpsi_dx / dx
