"""Discrete cosine and sine transforms, types I-IV, and their inverses.

Every transform acts along ``dim`` and is unnormalised: half of SciPy's
``scipy.fft.dct``/``dst``, so each inverse is twice SciPy's. ``backend``
picks the implementation for one call (see :mod:`orthel.transforms.backend`).
"""

from __future__ import annotations

import torch

from . import _cpp, _torch
from .backend import Backend, check_backend, get_backend

__all__ = [
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
