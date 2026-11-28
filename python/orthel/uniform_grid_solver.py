"""Spectral solver for ``Σ_i a_i ∂²u/∂x_i² + c u = f`` on a uniform grid."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Union, get_args

import torch

from .boundary import AxisBC, Boundary, parse_bc
from .eigenvalues import second_difference_eigenvalues
from .transforms.dtt import (
    dct1,
    dct2,
    dct3,
    dct4,
    dst1,
    dst2,
    dst3,
    dst4,
    idct1,
    idct2,
    idct3,
    idct4,
    idst1,
    idst2,
    idst3,
    idst4,
)

GhostRule = Literal["standard"]
Grid = Literal["cell", "node"]
BC = Union[AxisBC, Sequence[AxisBC]]
Pairs = tuple[tuple[Boundary, Boundary], ...]
Transform = Callable[[torch.Tensor, int], torch.Tensor]

_GHOST_RULES = get_args(GhostRule)
_GRIDS = get_args(Grid)

# (low, high) wall types -> (forward, inverse) transforms that diagonalise the
# 3-point operator on the unknown points, with eigenvalues ordered as in
# second_difference_eigenvalues.
_WALL_TRANSFORMS: dict[str, dict[tuple[str, str], tuple[Transform, Transform]]] = {
    # Cell centres, ghost cells mirrored (Neumann) or anti-mirrored (Dirichlet).
    "cell": {
        ("neumann", "neumann"): (dct2, idct2),
        ("dirichlet", "dirichlet"): (dst2, idst2),
        ("neumann", "dirichlet"): (dct4, idct4),
        ("dirichlet", "neumann"): (dst4, idst4),
    },
    # Nodes, Dirichlet wall nodes left out, Neumann wall nodes with u_{-1} = u_1.
    "node": {
        ("neumann", "neumann"): (dct1, idct1),
        ("dirichlet", "dirichlet"): (dst1, idst1),
        ("neumann", "dirichlet"): (dct3, idct3),
        ("dirichlet", "neumann"): (dst3, idst3),
    },
}


@dataclass(frozen=True)
class _Axis:
    """The 1D discretisation along one axis: ``n`` cells of width ``h``.

    ``kinds`` is the ``(low, high)`` boundary types and ``a`` the axis's
    coefficient.
    """

    n: int
    h: float
    a: float
    kinds: tuple[str, str]
    grid: str

    @property
    def periodic(self) -> bool:
        return self.kinds[0] == "periodic"

    @property
    def size(self) -> int:
        """Points along the axis: cells, or nodes including both wall nodes."""
        return self.n + 1 if self.grid == "node" and not self.periodic else self.n

    @property
    def unknowns(self) -> tuple[int, int]:
        """``(start, length)`` of the unknown points; Dirichlet nodes are known."""
        if self.grid == "node":
            return int(self.kinds[0] == "dirichlet"), self.size - self.kinds.count(
                "dirichlet"
            )
        return 0, self.size

    def known_sides(self) -> tuple[int, ...]:
        """Sides whose wall point is known (Dirichlet nodes)."""
        if self.grid != "node":
            return ()
        return tuple(side for side in (0, 1) if self.kinds[side] == "dirichlet")

    def eigenvalues(self, half_spectrum: bool) -> torch.Tensor:
        return second_difference_eigenvalues(
            self.n, self.h, self.kinds, half_spectrum=half_spectrum, grid=self.grid
        )

    @property
    def transforms(self) -> tuple[Transform, Transform]:
        return _WALL_TRANSFORMS[self.grid][self.kinds]

    def ghost_known(self, side: int, value: torch.Tensor) -> torch.Tensor:
        """The part of a cell-grid ghost cell set by the wall value.

        ``2g`` for Dirichlet (ghost ``2g - u_0``) and ``h q`` for Neumann
        (ghost ``u_0 + h q``).
        """
        return 2 * value if self.kinds[side] == "dirichlet" else self.h * value

    def lift(self, side: int, value: torch.Tensor) -> torch.Tensor:
        """What a wall value adds to the stencil of the unknown next to it.

        Cells: ``a / h²`` times the ghost's known part. Nodes: a Dirichlet
        node enters its neighbour's stencil with weight ``a / h²``; a Neumann
        ghost ``u_1 + 2 h q`` adds ``2 a q / h`` to the wall node's.
        """
        a, h = self.a, self.h
        if self.grid == "cell":
            return a / h**2 * self.ghost_known(side, value)
        if self.kinds[side] == "neumann":
            return 2 * a / h * value
        # With a single unknown that is a Neumann wall node, its ghost
        # u_1 + 2 h q reflects onto this Dirichlet node too, so g enters twice.
        twice = self.unknowns[1] == 1 and self.kinds[1 - side] == "neumann"
        return (2 if twice else 1) * a / h**2 * value


def _per_axis(
    value: float | Sequence[float], ndim: int, name: str
) -> tuple[float, ...]:
    """One float per axis from a scalar or a sequence of ``ndim`` values."""
    if isinstance(value, (int, float)):
        value = (value,) * ndim
    if len(value) != ndim:
        raise ValueError(f"expected {ndim} {name}, got {len(value)}")
    return tuple(float(v) for v in value)


class UniformGridSolver:
    """Solve ``Σ_i a_i ∂²u/∂x_i² + c u = f`` on a uniform grid.

    Args:
        shape: Number of cells along each axis.
        lengths: Domain length, one value for all axes or one per axis.
        bc: :class:`Periodic`, :class:`Dirichlet` or :class:`Neumann`
            boundaries, one for every side or one entry per axis (a boundary
            or a ``(low, high)`` pair).
        coefficients: Positive ``a_i``, one value for all axes or one per axis.
        shift: The constant ``c``.
        ghost_rule: How wall ghost cells are filled. ``"standard"`` (the only
            rule for now) uses ``2g - u_0`` (Dirichlet) and ``u_0 + h q``
            (Neumann) on cells.
        grid: ``"cell"`` puts ``u`` and ``f`` at cell centres, with shape
            ``shape``. ``"node"`` puts them at cell corners: ``n + 1`` values
            along a wall axis (both wall nodes included) and ``n`` along a
            periodic axis.
            On nodes a Dirichlet wall fixes ``u = g`` at its wall nodes and a
            Neumann wall uses a centered ghost cell ``u_{-1} = u_1 + 2 h q``.
    """

    def __init__(
        self,
        shape: Sequence[int],
        lengths: float | Sequence[float],
        bc: BC,
        coefficients: float | Sequence[float] = 1.0,
        shift: float = 0.0,
        ghost_rule: GhostRule = "standard",
        grid: Grid = "cell",
    ) -> None:
        self.shape = tuple(int(n) for n in shape)
        ndim = len(self.shape)
        if ndim == 0 or any(n < 1 for n in self.shape):
            raise ValueError(f"shape must be non-empty and positive, got {shape}")

        self.lengths = _per_axis(lengths, ndim, "lengths")
        if any(not 0 < length < float("inf") for length in self.lengths):
            raise ValueError(f"lengths must be positive and finite, got {self.lengths}")

        self.bc = parse_bc(bc, ndim)
        self.boundary_types = tuple((low.kind, high.kind) for low, high in self.bc)

        self.coefficients = _per_axis(coefficients, ndim, "coefficients")
        if any(not (math.isfinite(a) and a > 0) for a in self.coefficients):
            raise ValueError(
                f"coefficients must be positive and finite, got {self.coefficients}"
            )
        self.shift = float(shift)
        if not math.isfinite(self.shift):
            raise ValueError(f"shift must be finite, got {self.shift}")

        if ghost_rule not in _GHOST_RULES:
            raise ValueError(
                f"ghost_rule must be one of {_GHOST_RULES}, got {ghost_rule!r}"
            )
        self.ghost_rule = ghost_rule
        if grid not in _GRIDS:
            raise ValueError(f"grid must be one of {_GRIDS}, got {grid!r}")
        self.grid = grid

        self.spacings = tuple(L / n for L, n in zip(self.lengths, self.shape))
        self._axes = tuple(
            _Axis(n, h, a, kinds, grid)
            for n, h, a, kinds in zip(
                self.shape, self.spacings, self.coefficients, self.boundary_types
            )
        )
        self.field_shape = tuple(axis.size for axis in self._axes)
        self._periodic_axes = tuple(i for i, ax in enumerate(self._axes) if ax.periodic)
        self._wall_axes = tuple(i for i, ax in enumerate(self._axes) if not ax.periodic)

        self._inv_eigenvalues = self._inverse_eigenvalues()
        self._inv_eigenvalues_cache: dict[
            tuple[torch.device, torch.dtype], torch.Tensor
        ] = {}

    # Public API.

    def __call__(self, f: torch.Tensor, bc: BC | None = None) -> torch.Tensor:
        """Solve for ``u`` given ``f`` of shape ``(*batch, *self.field_shape)``.

        Args:
            f: Right-hand side at cell centres, or at nodes with ``grid="node"``.
            bc: Boundaries to use instead of the solver's, with the same
                types. Values must broadcast to the face, including batch
                dimensions.
        """
        self._check_field(f, "f")
        bc = self._call_bc(bc)
        dims = self._spatial_dims(f)

        rhs = self._lifted_rhs(f, bc, dims)
        u = rhs.clone() if rhs.numel() == 0 else self._apply_inverse(rhs, dims)
        return self._add_known_points(u, f, bc, dims)

    def coords(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Return the coordinates along each axis as 1D tensors."""
        offset = 0.5 if self.grid == "cell" else 0.0
        return tuple(
            (torch.arange(n, device=device, dtype=dtype) + offset) * h
            for n, h in zip(self.field_shape, self.spacings)
        )

    def ghost_cells(
        self,
        u: torch.Tensor,
        f: torch.Tensor | None = None,
        bc: BC | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Return the ``(low, high)`` ghost cells beyond each axis.

        Padding ``u = solver(f, bc)`` with these and applying the 3-point
        stencil gives back ``f``. ``f`` is not used by the standard ghost
        rule. Only available with ``grid="cell"``.
        """
        if self.grid != "cell":
            raise NotImplementedError('ghost_cells is only available for grid="cell"')
        self._check_field(u, "u")
        if f is not None:
            self._check_field(f, "f")
        bc = self._call_bc(bc)
        dims = self._spatial_dims(u)

        ghosts = []
        for i, (axis, dim) in enumerate(zip(self._axes, dims)):
            pair = []
            for side in (0, 1):
                if axis.periodic:  # the cell on the other side
                    pair.append(u.narrow(dim, axis.n - 1 if side == 0 else 0, 1))
                    continue
                edge = u.narrow(dim, 0 if side == 0 else axis.n - 1, 1)
                ghost = -edge if axis.kinds[side] == "dirichlet" else edge
                value = self._wall_value(i, side, u, bc, dims)
                pair.append(
                    ghost if value is None else ghost + axis.ghost_known(side, value)
                )
            ghosts.append(tuple(pair))
        return tuple(ghosts)

    # The steps of a solve.

    def _lifted_rhs(self, f: torch.Tensor, bc: Pairs, dims: list[int]) -> torch.Tensor:
        """Step 1: ``f`` on the unknown points, minus the wall values' terms.

        ``f`` is copied before the first subtraction, so it is never modified,
        and not copied at all when every wall value is zero.
        """
        rhs = self._unknowns_of(f, dims, self._wall_axes)
        copied = False
        for i in self._wall_axes:
            axis = self._axes[i]
            length = axis.unknowns[1]
            if length == 0:
                continue
            others = [j for j in self._wall_axes if j != i]
            for side in (0, 1):
                value = self._wall_value(i, side, f, bc, dims)
                if value is None:
                    continue
                value = self._unknowns_of(value, dims, others)
                if not copied:
                    rhs, copied = rhs.clone(), True
                row = 0 if side == 0 else length - 1
                rhs.narrow(dims[i], row, 1).sub_(axis.lift(side, value))
        return rhs

    def _apply_inverse(self, x: torch.Tensor, dims: list[int]) -> torch.Tensor:
        """Steps 2-4: transform, divide by the eigenvalues, transform back."""
        inv_eigenvalues = self._inv_eigenvalues_for(x)
        periodic_dims = [dims[i] for i in self._periodic_axes]
        for i in self._wall_axes:
            x = self._axes[i].transforms[0](x, dims[i])
        if periodic_dims:
            x = torch.fft.rfftn(x, dim=periodic_dims)

        x = x * inv_eigenvalues

        if periodic_dims:
            sizes = [self.shape[i] for i in self._periodic_axes]
            x = torch.fft.irfftn(x, s=sizes, dim=periodic_dims)
        for i in self._wall_axes:
            x = self._axes[i].transforms[1](x, dims[i])
        return x

    def _add_known_points(
        self, u: torch.Tensor, f: torch.Tensor, bc: Pairs, dims: list[int]
    ) -> torch.Tensor:
        """Step 5: put back the known (Dirichlet) wall nodes, one axis at a time.

        Nothing to do on the cell grid. Axes are padded in order, so a value is
        narrowed to the unknowns along the axes not padded yet, and a corner
        shared by two Dirichlet walls takes the later axis's value.
        """
        for i in self._wall_axes:
            dim = dims[i]
            later = [j for j in self._wall_axes if j > i]
            wall_shape = list(u.shape)
            wall_shape[dim] = 1
            for side in self._axes[i].known_sides():
                value = self._wall_value(i, side, f, bc, dims)
                if value is None:
                    wall = u.new_zeros(wall_shape)
                else:
                    wall = self._unknowns_of(value, dims, later).expand(wall_shape)
                u = torch.cat([wall, u] if side == 0 else [u, wall], dim=dim)
        return u

    # Setup and helpers.

    def _inverse_eigenvalues(self) -> torch.Tensor:
        """``1 / (Σ_i a_i λ_i + c)`` laid out like the transformed array.

        rfft halves the last periodic axis. With ``c = 0`` and no Dirichlet
        side, the constant mode has eigenvalue exactly 0; zeroing its inverse
        selects the zero-mean solution. Any other (near-)zero eigenvalue means
        the operator is singular.
        """
        ndim = len(self.shape)
        eigenvalues = torch.full((), self.shift, dtype=torch.float64)
        for i, axis in enumerate(self._axes):
            half = bool(self._periodic_axes) and i == self._periodic_axes[-1]
            view = [1] * ndim
            view[i] = -1
            eigenvalues = eigenvalues + axis.a * axis.eigenvalues(half).reshape(view)

        has_null_mode = self.shift == 0 and not any(
            "dirichlet" in kinds for kinds in self.boundary_types
        )
        if has_null_mode:
            null_mode = eigenvalues == 0
        else:
            null_mode = torch.zeros_like(eigenvalues, dtype=torch.bool)
        magnitude = eigenvalues.abs()
        # A node grid with two Dirichlet walls one cell apart has no unknowns.
        largest = magnitude.max() if magnitude.numel() else 0.0
        near_zero = magnitude <= 64 * torch.finfo(eigenvalues.dtype).eps * largest
        if (near_zero & ~null_mode).any():
            raise ValueError(
                f"operator is singular: shift {self.shift} matches an eigenvalue"
            )
        return torch.where(null_mode, 0.0, 1.0 / eigenvalues)

    def _inv_eigenvalues_for(self, x: torch.Tensor) -> torch.Tensor:
        key = (x.device, x.dtype)
        if key not in self._inv_eigenvalues_cache:
            self._inv_eigenvalues_cache[key] = self._inv_eigenvalues.to(
                device=x.device, dtype=x.dtype
            )
        return self._inv_eigenvalues_cache[key]

    def _unknowns_of(
        self, x: torch.Tensor, dims: list[int], axes: Sequence[int]
    ) -> torch.Tensor:
        """``x`` narrowed to the unknown points along ``axes``."""
        for i in axes:
            x = x.narrow(dims[i], *self._axes[i].unknowns)
        return x

    def _wall_value(
        self,
        axis: int,
        side: int,
        like: torch.Tensor,
        bc: Pairs,
        dims: list[int],
    ) -> torch.Tensor | None:
        """The wall value broadcast to the face, or ``None`` if it is the number 0."""
        value = bc[axis][side].value
        if isinstance(value, (int, float)) and value == 0:
            return None
        dim = dims[axis]
        face_shape = like.shape[:dim] + like.shape[dim + 1 :]
        value = torch.as_tensor(value, dtype=like.dtype, device=like.device)
        return torch.broadcast_to(value, face_shape).unsqueeze(dim)

    def _spatial_dims(self, x: torch.Tensor) -> list[int]:
        """Positions of the spatial axes in ``x``, after any batch dimensions."""
        ndim = len(self.shape)
        return [x.ndim - ndim + i for i in range(ndim)]

    def _check_field(self, x: torch.Tensor, name: str) -> None:
        if x.is_complex() or not x.is_floating_point():
            raise TypeError(
                f"{name} must be a real floating-point tensor, got {x.dtype}"
            )
        ndim = len(self.shape)
        if x.ndim < ndim or tuple(x.shape[-ndim:]) != self.field_shape:
            raise ValueError(
                f"expected {name} of shape (*batch, *{self.field_shape}), "
                f"got {tuple(x.shape)}"
            )

    def _call_bc(self, bc: BC | None) -> Pairs:
        if bc is None:
            return self.bc
        bc = parse_bc(bc, len(self.shape))
        types = tuple((low.kind, high.kind) for low, high in bc)
        if types != self.boundary_types:
            raise ValueError(
                f"boundary types {types} do not match the solver's {self.boundary_types}"
            )
        return bc
