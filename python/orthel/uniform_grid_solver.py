from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, Union

import torch

from .boundary import AxisBC, Boundary, parse_bc
from .eigenvalues import second_difference_eigenvalues
from .transforms.dct import dct2, dct4, dst2, dst4, idct2, idct4, idst2, idst4

GhostRule = Literal["standard", "pde"]
BC = Union[AxisBC, Sequence[AxisBC]]

_GHOST_RULES = ("standard", "pde")


Transform = Callable[[torch.Tensor, int], torch.Tensor]

# (low, high) wall types -> (forward, inverse). The transform diagonalises the
# cell-centred 3-point Laplacian with ghost cells mirrored (Neumann) or
# anti-mirrored (Dirichlet) across the faces, with eigenvalues ordered as in
# second_difference_eigenvalues.
_WALL_TRANSFORMS: dict[tuple[str, str], tuple[Transform, Transform]] = {
    ("neumann", "neumann"): (dct2, idct2),
    ("dirichlet", "dirichlet"): (dst2, idst2),
    ("neumann", "dirichlet"): (dct4, idct4),
    ("dirichlet", "neumann"): (dst4, idst4),
}


def _second_difference(
    x: torch.Tensor, dim: int, h: float, periodic: bool
) -> torch.Tensor:
    """Second difference along ``dim``, one-sided at the ends unless periodic."""
    n = x.shape[dim]
    if periodic:
        return (x.roll(1, dim) - 2 * x + x.roll(-1, dim)) / h**2
    if n < 3:
        return torch.zeros_like(x)
    inner = (
        x.narrow(dim, 0, n - 2) - 2 * x.narrow(dim, 1, n - 2) + x.narrow(dim, 2, n - 2)
    )
    if n < 4:
        first, last = inner, inner
    else:
        flipped = x.flip(dim)
        first, last = (
            2 * y.narrow(dim, 0, 1)
            - 5 * y.narrow(dim, 1, 1)
            + 4 * y.narrow(dim, 2, 1)
            - y.narrow(dim, 3, 1)
            for y in (x, flipped)
        )
    return (
        torch.cat([first.narrow(dim, 0, 1), inner, last.narrow(dim, -1, 1)], dim) / h**2
    )


class UniformGridSolver:
    """Solve ``Σ_i a_i ∂²u/∂x_i² + c u = f`` on a uniform cell-centred grid.

    Args:
        shape: Number of cells along each axis.
        lengths: Domain length, one value for all axes or one per axis.
        bc: :class:`Periodic`, :class:`Dirichlet` or :class:`Neumann`
            boundaries, one for every side or one entry per axis (a boundary
            or a ``(low, high)`` pair).
        coefficients: Positive ``a_i``, one value for all axes or one per axis.
        shift: The constant ``c``.
        ghost_rule: ``"standard"`` fills ghost cells with ``2g - u_0``
            (Dirichlet) and ``u_0 + h q`` (Neumann). ``"pde"`` adds the
            missing ``(h²/4) ∂²u/∂n²`` and ``(h³/24) ∂³u/∂n³`` terms, taken
            from the equation, which makes the gradient of ``u`` second order
            up to the walls but assumes ``f`` is smooth there.
    """

    def __init__(
        self,
        shape: Sequence[int],
        lengths: float | Sequence[float],
        bc: BC,
        coefficients: float | Sequence[float] = 1.0,
        shift: float = 0.0,
        ghost_rule: GhostRule = "standard",
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
        if any(not a > 0 for a in self.coefficients):
            raise ValueError(f"coefficients must be positive, got {self.coefficients}")
        self.shift = float(shift)

        if ghost_rule not in _GHOST_RULES:
            raise ValueError(
                f"ghost_rule must be one of {_GHOST_RULES}, got {ghost_rule!r}"
            )
        self.ghost_rule = ghost_rule

        self.spacings = tuple(L / n for L, n in zip(self.lengths, self.shape))
        self._periodic_axes = tuple(
            i for i, b in enumerate(self.boundary_types) if b[0] == "periodic"
        )
        self._wall_axes = tuple(
            i for i, b in enumerate(self.boundary_types) if b[0] != "periodic"
        )

        # Eigenvalues of Σ a_i λ_i + c, laid out to match the transformed array
        # (rfft halves the last periodic axis).
        eigenvalues = torch.full((), self.shift, dtype=torch.float64)
        for i, (n, h, a) in enumerate(
            zip(self.shape, self.spacings, self.coefficients)
        ):
            half = bool(self._periodic_axes) and i == self._periodic_axes[-1]
            lam = second_difference_eigenvalues(
                n, h, self.boundary_types[i], half_spectrum=half
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
        near_zero = (
            magnitude <= 64 * torch.finfo(eigenvalues.dtype).eps * magnitude.max()
        )
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
        """Return the cell-centre coordinates along each axis as 1D tensors."""
        return tuple(
            (torch.arange(n, device=device, dtype=dtype) + 0.5) * h
            for n, h in zip(self.shape, self.spacings)
        )

    def __call__(self, f: torch.Tensor, bc: BC | None = None) -> torch.Tensor:
        """Solve for ``u`` given ``f`` of shape ``(*batch, *self.shape)``.

        Args:
            f: Right-hand side at cell centres.
            bc: Boundaries to use instead of the solver's, with the same
                types. Values must broadcast to the face, including batch
                dimensions.
        """
        self._check_field(f, "f")
        bc = self._call_bc(bc)
        ndim = len(self.shape)
        dims = [f.ndim - ndim + i for i in range(ndim)]
        periodic_dims = [dims[i] for i in self._periodic_axes]

        x = self._lift(f, bc, dims)
        for i in self._wall_axes:
            x = _WALL_TRANSFORMS[self.boundary_types[i]][0](x, dims[i])
        if periodic_dims:
            x = torch.fft.rfftn(x, dim=periodic_dims)

        x = x * self._inv_eigenvalues_for(f)

        if periodic_dims:
            sizes = [self.shape[i] for i in self._periodic_axes]
            x = torch.fft.irfftn(x, s=sizes, dim=periodic_dims)
        for i in self._wall_axes:
            x = _WALL_TRANSFORMS[self.boundary_types[i]][1](x, dims[i])
        return x

    def ghost_cells(
        self,
        u: torch.Tensor,
        f: torch.Tensor | None = None,
        bc: BC | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Return the ``(low, high)`` ghost cells beyond each axis.

        Padding ``u = solver(f, bc)`` with these and applying the 3-point
        stencil gives back ``f``. ``f`` is only needed with
        ``ghost_rule="pde"``.
        """
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
                known = self._ghost_known(axis, side, u, f, bc, dims)
                ghost = -edge if kind == "dirichlet" else edge
                pair.append(ghost if known is None else ghost + known)
            ghosts.append(tuple(pair))
        return tuple(ghosts)

    def _ghost_known(
        self,
        axis: int,
        side: int,
        like: torch.Tensor,
        f: torch.Tensor | None,
        bc: tuple[tuple[Boundary, Boundary], ...],
        dims: list[int],
    ) -> torch.Tensor | None:
        """The part of a wall's ghost cell not involving ``u``, or ``None`` if zero."""
        dirichlet = self.boundary_types[axis][side] == "dirichlet"
        h = self.spacings[axis]
        value = self._wall_value(axis, side, like, bc, dims)
        if value is None:
            known = None
        else:
            known = 2 * value if dirichlet else h * value
        if self.ghost_rule == "pde":
            if f is None:
                raise ValueError(
                    'f is required for wall ghost cells with ghost_rule="pde"'
                )
            derivative = self._normal_derivative(axis, side, f, value, dims)
            correction = (h**2 / 4 if dirichlet else h**3 / 24) * derivative
            known = correction if known is None else known + correction
        return known

    def _normal_derivative(
        self,
        axis: int,
        side: int,
        f: torch.Tensor,
        value: torch.Tensor | None,
        dims: list[int],
    ) -> torch.Tensor:
        """``∂²u/∂n²`` (Dirichlet) or ``∂³u/∂n³`` (Neumann) at the wall, from the equation.

        Dirichlet: ``a_n ∂²u/∂n² = f - c g - Σ_t a_t ∂²g/∂t²``.
        Neumann: ``a_n ∂³u/∂n³ = ∂f/∂n - c q - Σ_t a_t ∂²q/∂t²``.
        """
        dim, n, h = dims[axis], self.shape[axis], self.spacings[axis]
        near, far = (0, 1) if side == 0 else (n - 1, n - 2)
        f_near = f.narrow(dim, near, 1)
        f_far = f.narrow(dim, far, 1) if n >= 2 else f_near
        if self.boundary_types[axis][side] == "dirichlet":
            rhs = 1.5 * f_near - 0.5 * f_far
        else:
            rhs = (f_near - f_far) / h
        if value is not None:
            rhs = rhs - self.shift * value
            for t, (a, h) in enumerate(zip(self.coefficients, self.spacings)):
                if t != axis:
                    periodic = self.boundary_types[t][0] == "periodic"
                    rhs = rhs - a * _second_difference(value, dims[t], h, periodic)
        return rhs / self.coefficients[axis]

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
                known = self._ghost_known(axis, side, f, f, bc, dims)
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
        if x.ndim < ndim or tuple(x.shape[-ndim:]) != self.shape:
            raise ValueError(
                f"expected {name} of shape (*batch, *{self.shape}), got {tuple(x.shape)}"
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
