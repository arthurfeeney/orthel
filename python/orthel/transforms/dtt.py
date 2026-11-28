"""Discrete cosine and sine transforms, types I-IV, and their inverses.

:func:`dtt` applies a DCT or DST along each of several dimensions, and
:func:`dct`/:func:`dst` take the type as an argument. The numbered functions
(:func:`dct2`, :func:`idst1`, ...) act along one dimension.

The scaling matches ``scipy.fft.dct``/``dst``: unnormalised (``"backward"``)
by default, and the N-D functions take SciPy's ``norm`` (``"backward"``,
``"ortho"`` or ``"forward"``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal, get_args

import torch

from . import _cpp, _torch
from .backend import Backend, check_backend, get_backend

# The forward transforms :func:`dtt` takes, by name.
KINDS = tuple(f"{family}{type}" for family in ("dct", "dst") for type in range(1, 5))

# Scalings, as in ``scipy.fft``: "backward" leaves the forward transform
# unscaled, "forward" scales it by 1 / M and "ortho" makes it orthonormal.
Norm = Literal["backward", "ortho", "forward"]
_NORMS = get_args(Norm)

__all__ = [
    "dtt",
    "idtt",
    "dct",
    "dst",
    "idct",
    "idst",
    "dct1",
    "dct2",
    "dct3",
    "dct4",
    "dst1",
    "dst2",
    "dst3",
    "dst4",
    "idct1",
    "idct2",
    "idct3",
    "idct4",
    "idst1",
    "idst2",
    "idst3",
    "idst4",
]


def _apply(op: str, x: torch.Tensor, dim: int, backend: Backend | None):
    dim = _torch._normalize_dim(x, dim)
    if op in ("dct1", "idct1") and x.shape[dim] < 2:
        raise ValueError(
            f"dct1 needs at least 2 points along dim {dim}, got {x.shape[dim]}"
        )
    name = get_backend() if backend is None else check_backend(backend)
    if x.shape[dim] == 0:  # torch.fft rejects empty transforms
        return x.clone()
    if name == "auto":
        use_cpp = _cpp.AVAILABLE and _cpp.supports(x) and _cpp.is_fast(op, x, dim)
    elif name == "cpp":
        if not _cpp.supports(x):
            raise TypeError(
                "the cpp backend needs a float32 or float64 CPU tensor, got "
                f"{x.dtype} on {x.device}"
            )
        use_cpp = True
    else:
        use_cpp = False
    if use_cpp:
        return _cpp.r2r(x, dim, op, False)
    return getattr(_torch, op)(x, dim)


def dct1(
    x: torch.Tensor, dim: int = -1, *, backend: Backend | None = None
) -> torch.Tensor:
    """DCT-I along ``dim``."""
    return _apply("dct1", x, dim, backend)


def idct1(
    x: torch.Tensor, dim: int = -1, *, backend: Backend | None = None
) -> torch.Tensor:
    """Inverse DCT-I along ``dim``."""
    return _apply("idct1", x, dim, backend)


def dst1(
    x: torch.Tensor, dim: int = -1, *, backend: Backend | None = None
) -> torch.Tensor:
    """DST-I along ``dim``."""
    return _apply("dst1", x, dim, backend)


def idst1(
    x: torch.Tensor, dim: int = -1, *, backend: Backend | None = None
) -> torch.Tensor:
    """Inverse DST-I along ``dim``."""
    return _apply("idst1", x, dim, backend)


def dct2(
    x: torch.Tensor, dim: int = -1, *, backend: Backend | None = None
) -> torch.Tensor:
    """DCT-II along ``dim``."""
    return _apply("dct2", x, dim, backend)


def idct2(
    x: torch.Tensor, dim: int = -1, *, backend: Backend | None = None
) -> torch.Tensor:
    """Inverse DCT-II along ``dim``."""
    return _apply("idct2", x, dim, backend)


def dst2(
    x: torch.Tensor, dim: int = -1, *, backend: Backend | None = None
) -> torch.Tensor:
    """DST-II along ``dim``."""
    return _apply("dst2", x, dim, backend)


def idst2(
    x: torch.Tensor, dim: int = -1, *, backend: Backend | None = None
) -> torch.Tensor:
    """Inverse DST-II along ``dim``."""
    return _apply("idst2", x, dim, backend)


def dct3(
    x: torch.Tensor, dim: int = -1, *, backend: Backend | None = None
) -> torch.Tensor:
    """DCT-III along ``dim``."""
    return _apply("dct3", x, dim, backend)


def idct3(
    x: torch.Tensor, dim: int = -1, *, backend: Backend | None = None
) -> torch.Tensor:
    """Inverse DCT-III along ``dim``."""
    return _apply("idct3", x, dim, backend)


def dst3(
    x: torch.Tensor, dim: int = -1, *, backend: Backend | None = None
) -> torch.Tensor:
    """DST-III along ``dim``."""
    return _apply("dst3", x, dim, backend)


def idst3(
    x: torch.Tensor, dim: int = -1, *, backend: Backend | None = None
) -> torch.Tensor:
    """Inverse DST-III along ``dim``."""
    return _apply("idst3", x, dim, backend)


def dct4(
    x: torch.Tensor, dim: int = -1, *, backend: Backend | None = None
) -> torch.Tensor:
    """DCT-IV along ``dim``."""
    return _apply("dct4", x, dim, backend)


def idct4(
    x: torch.Tensor, dim: int = -1, *, backend: Backend | None = None
) -> torch.Tensor:
    """Inverse DCT-IV along ``dim``."""
    return _apply("idct4", x, dim, backend)


def dst4(
    x: torch.Tensor, dim: int = -1, *, backend: Backend | None = None
) -> torch.Tensor:
    """DST-IV along ``dim``."""
    return _apply("dst4", x, dim, backend)


def idst4(
    x: torch.Tensor, dim: int = -1, *, backend: Backend | None = None
) -> torch.Tensor:
    """Inverse DST-IV along ``dim``."""
    return _apply("idst4", x, dim, backend)


def _dims(x: torch.Tensor, dim: int | Sequence[int] | None) -> tuple[int, ...]:
    if dim is None:
        return tuple(range(x.ndim))
    dims = (dim,) if isinstance(dim, int) else tuple(dim)
    dims = tuple(_torch._normalize_dim(x, d) for d in dims)
    if len(set(dims)) != len(dims):
        raise ValueError(f"dim {dim} repeats a dimension")
    return dims


def _check_norm(norm: Norm | None) -> Norm:
    if norm is None:
        return "backward"
    if norm not in _NORMS:
        raise ValueError(f"norm must be one of {_NORMS}, got {norm!r}")
    return norm


def _period(kind: str, n: int) -> int:
    """``M``: the backward inverse of ``kind`` is its transpose-type over ``M``."""
    if kind == "dct1":
        return 2 * (n - 1)
    if kind == "dst1":
        return 2 * (n + 1)
    return 2 * n


# Orthonormal transforms weight their end points: (input, output) factors on
# (first, last) entries, applied on top of 1 / sqrt(M).
_SQRT2 = math.sqrt(2)
_ORTHO_EDGES = {
    "dct1": ((_SQRT2, _SQRT2), (1 / _SQRT2, 1 / _SQRT2)),
    "dct2": ((1, 1), (1 / _SQRT2, 1)),
    "dct3": ((_SQRT2, 1), (1, 1)),
    "dst2": ((1, 1), (1, 1 / _SQRT2)),
    "dst3": ((1, _SQRT2), (1, 1)),
}
# The orthonormal inverse is the transpose, which swaps types II and III.
_TRANSPOSE_TYPE = {"1": "1", "2": "3", "3": "2", "4": "4"}


def _weight(
    x: torch.Tensor, dim: int, scale: float, edges: tuple[float, float]
) -> torch.Tensor:
    """``x`` times ``scale``, with extra factors on its first and last entries."""
    if edges == (1, 1):
        return x * scale
    n = x.shape[dim]
    w = torch.full((n,), scale, dtype=x.dtype, device=x.device)
    w[0] *= edges[0]
    w[n - 1] *= edges[1]
    shape = [1] * x.ndim
    shape[dim] = n
    return x * w.reshape(shape)


def _apply_normed(
    op: str, x: torch.Tensor, dim: int, norm: Norm, backend: Backend | None
) -> torch.Tensor:
    """:func:`_apply` with ``norm``'s scaling; ``dim`` is non-negative."""
    n = x.shape[dim]
    if norm == "backward" or n == 0 or (op.endswith("ct1") and n < 2):
        return _apply(op, x, dim, backend)
    inverse = op.startswith("i")
    kind = op[1:] if inverse else op
    if norm == "forward":
        y = _apply(op, x, dim, backend)
        m = _period(kind, n)
        return y * m if inverse else y / m
    if inverse:
        kind = kind[:3] + _TRANSPOSE_TYPE[kind[3]]
    pre, post = _ORTHO_EDGES.get(kind, ((1, 1), (1, 1)))
    y = _apply(kind, _weight(x, dim, 1.0, pre), dim, backend)
    return _weight(y, dim, 1 / math.sqrt(_period(kind, n)), post)


def _apply_each(
    ops: str | Sequence[str],
    x: torch.Tensor,
    dim: int | Sequence[int] | None,
    norm: Norm | None,
    backend: Backend | None,
):
    """Apply ``ops`` (one name, or one per dimension) along each of ``dim``."""
    norm = _check_norm(norm)
    if backend is not None:
        check_backend(backend)
    dims = _dims(x, dim)
    if isinstance(ops, str):
        ops = (ops,) * len(dims)
    elif len(ops) != len(dims):
        raise ValueError(f"got {len(ops)} transforms for {len(dims)} dimensions {dims}")
    if not dims:
        return x.clone()
    # Transforms along different dimensions commute, so the order is free.
    for op, d in zip(ops, dims):
        x = _apply_normed(op, x, d, norm, backend)
    return x


def _apply_nd(
    family: str,
    type: int,
    x: torch.Tensor,
    dim: int | Sequence[int] | None,
    norm: Norm | None,
    backend: Backend | None,
):
    if isinstance(type, bool) or not isinstance(type, int) or not 1 <= type <= 4:
        raise ValueError(f"type must be 1, 2, 3 or 4, got {type!r}")
    return _apply_each(f"{family}{type}", x, dim, norm, backend)


def dct(
    x: torch.Tensor,
    type: int = 2,
    dim: int | Sequence[int] | None = -1,
    *,
    norm: Norm | None = "backward",
    backend: Backend | None = None,
) -> torch.Tensor:
    """DCT of type ``type`` along each dimension in ``dim``.

    Along one dimension this is :func:`dct1` to :func:`dct4`; along several
    it is those applied in turn, the separable N-D transform.

    Args:
        x: Input tensor.
        type: The DCT type, 1 to 4.
        dim: One dimension, a sequence of distinct dimensions, or ``None``
            for all of them.
        norm: The scaling, as in ``scipy.fft``: ``"backward"`` (or ``None``)
            leaves the forward transform unscaled, ``"forward"`` scales it by
            ``1 / M`` (``M = 2N``, ``2(N-1)`` for DCT-I, ``2(N+1)`` for DST-I)
            and ``"ortho"`` makes it orthonormal. An inverse uses the same
            ``norm`` as the transform it inverts.
        backend: The backend for this call. ``None`` uses the global one (see
            :func:`orthel.transforms.backend.get_backend`).

    Returns:
        The transform of ``x``, with its shape and dtype. With no dimensions to
        transform, a copy of ``x``.

    Raises:
        ValueError: If ``type`` is not 1, 2, 3 or 4, ``dim`` repeats a
            dimension, ``norm`` or ``backend`` is not valid, or a DCT-I
            dimension has fewer than 2 points.
        IndexError: If a dimension in ``dim`` is out of range.
        RuntimeError: If ``backend`` is ``"cpp"`` and the extension is not built.
        TypeError: If ``backend`` is ``"cpp"`` and ``x`` is not a float32 or
            float64 CPU tensor.
    """
    return _apply_nd("dct", type, x, dim, norm, backend)


def idct(
    x: torch.Tensor,
    type: int = 2,
    dim: int | Sequence[int] | None = -1,
    *,
    norm: Norm | None = "backward",
    backend: Backend | None = None,
) -> torch.Tensor:
    """Inverse of :func:`dct` with the same ``type`` and ``dim``.

    Along one dimension this is :func:`idct1` to :func:`idct4`.

    Args:
        x: Input tensor.
        type: The type of the DCT to invert, 1 to 4.
        dim: One dimension, a sequence of distinct dimensions, or ``None``
            for all of them.
        norm: The scaling, as in ``scipy.fft``: ``"backward"`` (or ``None``)
            leaves the forward transform unscaled, ``"forward"`` scales it by
            ``1 / M`` (``M = 2N``, ``2(N-1)`` for DCT-I, ``2(N+1)`` for DST-I)
            and ``"ortho"`` makes it orthonormal. An inverse uses the same
            ``norm`` as the transform it inverts.
        backend: The backend for this call. ``None`` uses the global one (see
            :func:`orthel.transforms.backend.get_backend`).

    Returns:
        The transform of ``x``, with its shape and dtype. With no dimensions to
        transform, a copy of ``x``.

    Raises:
        ValueError: If ``type`` is not 1, 2, 3 or 4, ``dim`` repeats a
            dimension, ``norm`` or ``backend`` is not valid, or a DCT-I
            dimension has fewer than 2 points.
        IndexError: If a dimension in ``dim`` is out of range.
        RuntimeError: If ``backend`` is ``"cpp"`` and the extension is not built.
        TypeError: If ``backend`` is ``"cpp"`` and ``x`` is not a float32 or
            float64 CPU tensor.
    """
    return _apply_nd("idct", type, x, dim, norm, backend)


def dst(
    x: torch.Tensor,
    type: int = 2,
    dim: int | Sequence[int] | None = -1,
    *,
    norm: Norm | None = "backward",
    backend: Backend | None = None,
) -> torch.Tensor:
    """DST of type ``type`` along each dimension in ``dim``.

    Along one dimension this is :func:`dst1` to :func:`dst4`; along several
    it is those applied in turn, the separable N-D transform.

    Args:
        x: Input tensor.
        type: The DST type, 1 to 4.
        dim: One dimension, a sequence of distinct dimensions, or ``None``
            for all of them.
        norm: The scaling, as in ``scipy.fft``: ``"backward"`` (or ``None``)
            leaves the forward transform unscaled, ``"forward"`` scales it by
            ``1 / M`` (``M = 2N``, ``2(N-1)`` for DCT-I, ``2(N+1)`` for DST-I)
            and ``"ortho"`` makes it orthonormal. An inverse uses the same
            ``norm`` as the transform it inverts.
        backend: The backend for this call. ``None`` uses the global one (see
            :func:`orthel.transforms.backend.get_backend`).

    Returns:
        The transform of ``x``, with its shape and dtype. With no dimensions to
        transform, a copy of ``x``.

    Raises:
        ValueError: If ``type`` is not 1, 2, 3 or 4, ``dim`` repeats a
            dimension, ``norm`` or ``backend`` is not valid, or a DCT-I
            dimension has fewer than 2 points.
        IndexError: If a dimension in ``dim`` is out of range.
        RuntimeError: If ``backend`` is ``"cpp"`` and the extension is not built.
        TypeError: If ``backend`` is ``"cpp"`` and ``x`` is not a float32 or
            float64 CPU tensor.
    """
    return _apply_nd("dst", type, x, dim, norm, backend)


def idst(
    x: torch.Tensor,
    type: int = 2,
    dim: int | Sequence[int] | None = -1,
    *,
    norm: Norm | None = "backward",
    backend: Backend | None = None,
) -> torch.Tensor:
    """Inverse of :func:`dst` with the same ``type`` and ``dim``.

    Along one dimension this is :func:`idst1` to :func:`idst4`.

    Args:
        x: Input tensor.
        type: The type of the DST to invert, 1 to 4.
        dim: One dimension, a sequence of distinct dimensions, or ``None``
            for all of them.
        norm: The scaling, as in ``scipy.fft``: ``"backward"`` (or ``None``)
            leaves the forward transform unscaled, ``"forward"`` scales it by
            ``1 / M`` (``M = 2N``, ``2(N-1)`` for DCT-I, ``2(N+1)`` for DST-I)
            and ``"ortho"`` makes it orthonormal. An inverse uses the same
            ``norm`` as the transform it inverts.
        backend: The backend for this call. ``None`` uses the global one (see
            :func:`orthel.transforms.backend.get_backend`).

    Returns:
        The transform of ``x``, with its shape and dtype. With no dimensions to
        transform, a copy of ``x``.

    Raises:
        ValueError: If ``type`` is not 1, 2, 3 or 4, ``dim`` repeats a
            dimension, ``norm`` or ``backend`` is not valid, or a DCT-I
            dimension has fewer than 2 points.
        IndexError: If a dimension in ``dim`` is out of range.
        RuntimeError: If ``backend`` is ``"cpp"`` and the extension is not built.
        TypeError: If ``backend`` is ``"cpp"`` and ``x`` is not a float32 or
            float64 CPU tensor.
    """
    return _apply_nd("idst", type, x, dim, norm, backend)


def _kinds(kind: str | Sequence[str]) -> str | tuple[str, ...]:
    kinds = (kind,) if isinstance(kind, str) else tuple(kind)
    for k in kinds:
        if k not in KINDS:
            raise ValueError(f"kind must be one of {', '.join(KINDS)}, got {k!r}")
    return kind if isinstance(kind, str) else kinds


def dtt(
    x: torch.Tensor,
    kind: str | Sequence[str],
    dim: int | Sequence[int] | None = -1,
    *,
    norm: Norm | None = "backward",
    backend: Backend | None = None,
) -> torch.Tensor:
    """Discrete trigonometric transform: a DCT or DST along each dimension.

    For example ``dtt(x, ("dct1", "dst2"), (0, 1))`` is
    ``dst2(dct1(x, 0), 1)``: the separable transform for a grid with a
    different boundary condition along each axis.

    Args:
        x: Input tensor.
        kind: A forward transform name from :data:`KINDS` (``"dct1"`` to
            ``"dct4"``, ``"dst1"`` to ``"dst4"``) for every dimension in
            ``dim``, or a sequence with one per dimension, in the same order.
        dim: One dimension, a sequence of distinct dimensions, or ``None``
            for all of them.
        norm: The scaling, as in ``scipy.fft``: ``"backward"`` (or ``None``)
            leaves the forward transform unscaled, ``"forward"`` scales it by
            ``1 / M`` (``M = 2N``, ``2(N-1)`` for DCT-I, ``2(N+1)`` for DST-I)
            and ``"ortho"`` makes it orthonormal. An inverse uses the same
            ``norm`` as the transform it inverts.
        backend: The backend for this call. ``None`` uses the global one (see
            :func:`orthel.transforms.backend.get_backend`).

    Returns:
        The transform of ``x``, with its shape and dtype. With no dimensions to
        transform, a copy of ``x``.

    Raises:
        ValueError: If a ``kind`` is not one of :data:`KINDS`, the number of
            kinds does not match the number of dimensions, ``dim`` repeats a
            dimension, ``norm`` or ``backend`` is not valid, or a DCT-I
            dimension has fewer than 2 points.
        IndexError: If a dimension in ``dim`` is out of range.
        RuntimeError: If ``backend`` is ``"cpp"`` and the extension is not built.
        TypeError: If ``backend`` is ``"cpp"`` and ``x`` is not a float32 or
            float64 CPU tensor.
    """
    return _apply_each(_kinds(kind), x, dim, norm, backend)


def idtt(
    x: torch.Tensor,
    kind: str | Sequence[str],
    dim: int | Sequence[int] | None = -1,
    *,
    norm: Norm | None = "backward",
    backend: Backend | None = None,
) -> torch.Tensor:
    """Inverse of :func:`dtt` with the same ``kind`` and ``dim``.

    ``idtt(dtt(x, kind, dim), kind, dim)`` recovers ``x``.

    Args:
        x: Input tensor.
        kind: The forward transforms being inverted, as for :func:`dtt`.
        dim: One dimension, a sequence of distinct dimensions, or ``None``
            for all of them.
        norm: The scaling, as in ``scipy.fft``: ``"backward"`` (or ``None``)
            leaves the forward transform unscaled, ``"forward"`` scales it by
            ``1 / M`` (``M = 2N``, ``2(N-1)`` for DCT-I, ``2(N+1)`` for DST-I)
            and ``"ortho"`` makes it orthonormal. An inverse uses the same
            ``norm`` as the transform it inverts.
        backend: The backend for this call. ``None`` uses the global one (see
            :func:`orthel.transforms.backend.get_backend`).

    Returns:
        The transform of ``x``, with its shape and dtype. With no dimensions to
        transform, a copy of ``x``.

    Raises:
        ValueError: If a ``kind`` is not one of :data:`KINDS`, the number of
            kinds does not match the number of dimensions, ``dim`` repeats a
            dimension, ``norm`` or ``backend`` is not valid, or a DCT-I
            dimension has fewer than 2 points.
        IndexError: If a dimension in ``dim`` is out of range.
        RuntimeError: If ``backend`` is ``"cpp"`` and the extension is not built.
        TypeError: If ``backend`` is ``"cpp"`` and ``x`` is not a float32 or
            float64 CPU tensor.
    """
    kinds = _kinds(kind)
    if isinstance(kinds, str):
        return _apply_each(f"i{kinds}", x, dim, norm, backend)
    return _apply_each([f"i{k}" for k in kinds], x, dim, norm, backend)
