from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Literal, Union

import torch

from .boundary import AxisBC, Boundary, parse_bc
from .eigenvalues import second_difference_eigenvalues
from .transforms.dct import (
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

_GHOST_RULES = ("standard",)
_GRIDS = ("cell", "node")


Transform = Callable[[torch.Tensor, int], torch.Tensor]

# (low, high) wall types -> (forward, inverse). The transform diagonalises the
# cell-centred 3-point Laplacian with ghost cells mirrored (Neumann) or
# anti-mirrored (Dirichlet) across the faces, with eigenvalues ordered as in
# second_difference_eigenvalues.
_CELL_CENTER_TRANSFORMS: dict[tuple[str, str], tuple[Transform, Transform]] = {
    ("neumann", "neumann"): (dct2, idct2),
    ("dirichlet", "dirichlet"): (dst2, idst2),
    ("neumann", "dirichlet"): (dct4, idct4),
    ("dirichlet", "neumann"): (dst4, idst4),
}


# The same for the node-centered Laplacian on the unknown nodes (Dirichlet wall
# nodes are known and left out; Neumann wall nodes mirror ``u_{-1} = u_1``).
_NODAL_TRANSFORMS: dict[tuple[str, str], tuple[Transform, Transform]] = {
    ("neumann", "neumann"): (dct1, idct1),
    ("dirichlet", "dirichlet"): (dst1, idst1),
    ("neumann", "dirichlet"): (dct3, idct3),
    ("dirichlet", "neumann"): (dst3, idst3),
}


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

        if isinstance(lengths, (int, float)):
            lengths = (lengths,) * ndim
        if len(lengths) != ndim:
            raise ValueError(f"expected {ndim} lengths, got {len(lengths)}")
        self.lengths = tuple(float(length) for length in lengths)
        if any(not 0 < length < float("inf") for length in self.lengths):
            raise ValueError(f"lengths must be positive and finite, got {self.lengths}")

        self.bc = parse_bc(bc, ndim)
        self.boundary_types = tuple((low.kind, high.kind) for low, high in self.bc)

        if isinstance(coefficients, (int, float)):
            coefficients = (coefficients,) * ndim
        if len(coefficients) != ndim:
            raise ValueError(f"expected {ndim} coefficients, got {len(coefficients)}")
        self.coefficients = tuple(float(a) for a in coefficients)
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
        self._periodic_axes = tuple(
            i for i, b in enumerate(self.boundary_types) if b[0] == "periodic"
        )
        self._wall_axes = tuple(
            i for i, b in enumerate(self.boundary_types) if b[0] != "periodic"
        )
        # Shape of u and f, and on nodes the (start, length) of the unknown
        # nodes along each axis (Dirichlet wall nodes are known).
        if grid == "cell":
            self.field_shape = self.shape
        else:
            self.field_shape = tuple(
                n if b[0] == "periodic" else n + 1
                for n, b in zip(self.shape, self.boundary_types)
            )
        self._unknowns = tuple(
            (
                int(b[0] == "dirichlet"),
                n - b.count("dirichlet") if grid == "node" else n,
            )
            for n, b in zip(self.field_shape, self.boundary_types)
        )

        # Eigenvalues of Σ a_i λ_i + c, laid out to match the transformed array
        # (rfft halves the last periodic axis).
        eigenvalues = torch.full((), self.shift, dtype=torch.float64)
        for i, (n, h, a) in enumerate(
            zip(self.shape, self.spacings, self.coefficients)
        ):
            half = bool(self._periodic_axes) and i == self._periodic_axes[-1]
            lam = second_difference_eigenvalues(
                n, h, self.boundary_types[i], half_spectrum=half, grid=grid
            )
            view = [1] * ndim
            view[i] = -1
            eigenvalues = eigenvalues + a * lam.reshape(view)

        # With c = 0 and no Dirichlet side, the constant mode has eigenvalue
        # exactly 0; zeroing its inverse selects the zero-mean solution. Any
        # other (near-)zero eigenvalue means the operator is singular.
        has_null_mode = self.shift == 0 and not any(
            "dirichlet" in b for b in self.boundary_types
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
        self._inv_eigenvalues = torch.where(null_mode, 0.0, 1.0 / eigenvalues)
        self._inv_eigenvalues_cache: dict[
            tuple[torch.device, torch.dtype], torch.Tensor
        ] = {}

    def coords(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Return the cell-centre (or node) coordinates along each axis as 1D tensors."""
        offset = 0.5 if self.grid == "cell" else 0.0
        return tuple(
            (torch.arange(n, device=device, dtype=dtype) + offset) * h
            for n, h in zip(self.field_shape, self.spacings)
        )

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
        ndim = len(self.shape)
        dims = [f.ndim - ndim + i for i in range(ndim)]
        if self.grid == "node":
            return self._solve_nodes(f, bc, dims)
        return self._solve_cells(f, bc, dims)

    def _solve_cells(
        self,
        f: torch.Tensor,
        bc: tuple[tuple[Boundary, Boundary], ...],
        dims: list[int],
    ) -> torch.Tensor:
        """Solve on cell centres, with wall values lifted through the ghost cells."""
        return self._spectral_solve(
            self._lift(f, bc, dims), dims, _CELL_CENTER_TRANSFORMS
        )

    def _spectral_solve(
        self,
        x: torch.Tensor,
        dims: list[int],
        transforms: dict[tuple[str, str], tuple[Transform, Transform]],
    ) -> torch.Tensor:
        """Apply the inverse operator to a lifted right-hand side on the unknowns."""
        inv_eigenvalues = self._inv_eigenvalues_for(x)
        periodic_dims = [dims[i] for i in self._periodic_axes]
        for i in self._wall_axes:
            x = transforms[self.boundary_types[i]][0](x, dims[i])
        if periodic_dims:
            x = torch.fft.rfftn(x, dim=periodic_dims)

        x = x * inv_eigenvalues

        if periodic_dims:
            sizes = [self.shape[i] for i in self._periodic_axes]
            x = torch.fft.irfftn(x, s=sizes, dim=periodic_dims)
        for i in self._wall_axes:
            x = transforms[self.boundary_types[i]][1](x, dims[i])
        return x

    def _solve_nodes(
        self,
        f: torch.Tensor,
        bc: tuple[tuple[Boundary, Boundary], ...],
        dims: list[int],
    ) -> torch.Tensor:
        """Solve on nodes: drop the known Dirichlet wall nodes, lift the wall
        values into the right-hand side, solve, then put the wall nodes back."""

        def restrict(x: torch.Tensor, skip: int | None = None) -> torch.Tensor:
            # Keep only the unknown nodes along wall axes other than ``skip``.
            for i in self._wall_axes:
                if i != skip:
                    x = x.narrow(dims[i], *self._unknowns[i])
            return x

        rhs = restrict(f)
        lifted = False
        for axis in self._wall_axes:
            dim, h, a = dims[axis], self.spacings[axis], self.coefficients[axis]
            start, length = self._unknowns[axis]
            if length == 0:
                continue
            for side in (0, 1):
                value = self._wall_value(axis, side, f, bc, dims)
                if value is None:
                    continue
                value = restrict(value, skip=axis)
                if self.boundary_types[axis][side] == "dirichlet":
                    # The wall node's stencil weight. With a single unknown that
                    # is a Neumann wall node, its ghost u_1 + 2 h q reflects onto
                    # this Dirichlet node too, so g enters twice.
                    other = self.boundary_types[axis][1 - side]
                    weight = 2 if length == 1 and other == "neumann" else 1
                    known = weight * a / h**2 * value
                else:
                    known = 2 * a / h * value  # from the ghost u_1 + 2 h q
                if not lifted:
                    rhs, lifted = rhs.clone(), True
                rhs.narrow(dim, 0 if side == 0 else length - 1, 1).sub_(known)

        if rhs.numel() == 0:
            u = rhs.clone()
        else:
            u = self._spectral_solve(rhs, dims, _NODAL_TRANSFORMS)

        # Put the Dirichlet wall nodes back, one axis at a time.
        for axis in self._wall_axes:
            pieces = []
            for side in (0, 1):
                if self.boundary_types[axis][side] != "dirichlet":
                    continue
                value = self._wall_value(axis, side, f, bc, dims)
                like = u.narrow(dims[axis], 0, 0)
                shape = list(like.shape)
                shape[dims[axis]] = 1
                if value is None:
                    wall = u.new_zeros(shape)
                else:
                    wall = value
                    for i in self._wall_axes:
                        if i > axis:
                            wall = wall.narrow(dims[i], *self._unknowns[i])
                    wall = wall.expand(shape)
                pieces.append((side, wall))
            for side, wall in pieces:
                u = torch.cat([wall, u] if side == 0 else [u, wall], dim=dims[axis])
        return u

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
        ndim = len(self.shape)
        dims = [u.ndim - ndim + i for i in range(ndim)]

        ghosts = []
        for axis, dim in enumerate(dims):
            n = self.shape[axis]
            pair = []
            for side in (0, 1):
                kind = self.boundary_types[axis][side]
                edge = u.narrow(dim, 0 if side == 0 else n - 1, 1)
                if kind == "periodic":
                    pair.append(u.narrow(dim, n - 1 if side == 0 else 0, 1))
                    continue
                known = self._ghost_known(axis, side, u, bc, dims)
                ghost = -edge if kind == "dirichlet" else edge
                pair.append(ghost if known is None else ghost + known)
            ghosts.append(tuple(pair))
        return tuple(ghosts)

    def _ghost_known(
        self,
        axis: int,
        side: int,
        like: torch.Tensor,
        bc: tuple[tuple[Boundary, Boundary], ...],
        dims: list[int],
    ) -> torch.Tensor | None:
        """The part of a wall's ghost cell not involving ``u``, or ``None`` if zero."""
        dirichlet = self.boundary_types[axis][side] == "dirichlet"
        h = self.spacings[axis]
        value = self._wall_value(axis, side, like, bc, dims)
        if value is None:
            return None
        return 2 * value if dirichlet else h * value

    def _wall_value(
        self,
        axis: int,
        side: int,
        like: torch.Tensor,
        bc: tuple[tuple[Boundary, Boundary], ...],
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

    def _lift(
        self,
        f: torch.Tensor,
        bc: tuple[tuple[Boundary, Boundary], ...],
        dims: list[int],
    ) -> torch.Tensor:
        """Move the known part of the ghost cells into the right-hand side."""
        rhs = f
        for axis in self._wall_axes:
            dim, h, a = dims[axis], self.spacings[axis], self.coefficients[axis]
            for side in (0, 1):
                known = self._ghost_known(axis, side, f, bc, dims)
                if known is None:
                    continue
                if rhs is f:
                    rhs = f.clone()
                index = 0 if side == 0 else self.shape[axis] - 1
                rhs.narrow(dim, index, 1).sub_(a / h**2 * known)
        return rhs

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

    def _call_bc(self, bc: BC | None) -> tuple[tuple[Boundary, Boundary], ...]:
        if bc is None:
            return self.bc
        bc = parse_bc(bc, len(self.shape))
        types = tuple((low.kind, high.kind) for low, high in bc)
        if types != self.boundary_types:
            raise ValueError(
                f"boundary types {types} do not match the solver's {self.boundary_types}"
            )
        return bc

    def _inv_eigenvalues_for(self, f: torch.Tensor) -> torch.Tensor:
        key = (f.device, f.dtype)
        if key not in self._inv_eigenvalues_cache:
            self._inv_eigenvalues_cache[key] = self._inv_eigenvalues.to(
                device=f.device, dtype=f.dtype
            )
        return self._inv_eigenvalues_cache[key]
