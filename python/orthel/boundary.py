from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Union

import torch

Value = Union[float, torch.Tensor]


@dataclass(frozen=True, eq=False)
class Periodic:
    """The axis wraps around; applies to both sides."""

    kind: ClassVar[str] = "periodic"


@dataclass(frozen=True, eq=False)
class Dirichlet:
    """``u = value`` on the face. ``value`` must broadcast to the face."""

    value: Value = 0.0
    kind: ClassVar[str] = "dirichlet"


@dataclass(frozen=True, eq=False)
class Neumann:
    """``∂u/∂n = value`` on the face, with ``n`` the outward normal."""

    value: Value = 0.0
    kind: ClassVar[str] = "neumann"


Boundary = Union[Periodic, Dirichlet, Neumann]
AxisBC = Union[Boundary, tuple[Boundary, Boundary]]

_BOUNDARIES = (Periodic, Dirichlet, Neumann)


def parse_bc(bc: AxisBC | Sequence[AxisBC], ndim: int) -> tuple[tuple[Boundary, Boundary], ...]:
    """convert AxisBC ``bc`` to a ``(low, high)`` pair of boundaries per axis.

    ``bc`` is one entry for every axis or a sequence of entries, with one entry per axis.
    An entry is a single boundary for both sides or a pair of boundaries ``(low, high)``.
    """
    if isinstance(bc, str):
        raise TypeError(f"expected a boundary object such as Periodic(), got {bc!r}")
    if isinstance(bc, _BOUNDARIES):
        bc = (bc,) * ndim
    if len(bc) != ndim:
        raise ValueError(f"expected {ndim} boundary entries, got {len(bc)}")
    pairs = []
    for axis, entry in enumerate(bc):
        low, high = entry if isinstance(entry, tuple) else (entry, entry)
        for side in (low, high):
            if not isinstance(side, _BOUNDARIES):
                raise TypeError(f"axis {axis}: expected a boundary object, got {side!r}")
        if isinstance(low, Periodic) != isinstance(high, Periodic):
            raise ValueError(f"axis {axis}: periodic must apply to both sides")
        pairs.append((low, high))
    return tuple(pairs)
