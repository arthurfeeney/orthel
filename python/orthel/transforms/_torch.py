"""The ``"torch"`` backend: DCT/DST from ``torch.fft``, on any device and dtype."""

from __future__ import annotations

import math

import torch


def _normalize_dim(x: torch.Tensor, dim: int) -> int:
    if not -x.ndim <= dim < x.ndim:
        raise IndexError(
            f"dim {dim} is out of range for a tensor with {x.ndim} dimensions"
        )
    return dim % x.ndim


def _index(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Return ``0, 1, ..., N-1`` along ``dim``, shaped to broadcast against ``x``."""
    shape = [1] * x.ndim
    shape[dim] = -1
    return torch.arange(x.shape[dim], device=x.device).reshape(shape)


def _alternating_sign(x: torch.Tensor, dim: int) -> torch.Tensor:
    return 1.0 - 2.0 * (_index(x, dim) % 2).to(x.dtype)


def dct2(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """DCT-II along ``dim``: ``X_k = Σ_j x_j cos(πk(2j+1) / 2N)``."""
    dim = _normalize_dim(x, dim)
    n = x.shape[dim]
    k = _index(x, dim).to(x.dtype)
    v = torch.fft.rfft(x, n=2 * n, dim=dim).narrow(dim, 0, n)
    return (v * torch.exp(-1j * math.pi * k / (2 * n))).real


def idct2(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Inverse of :func:`dct2`: ``(2 / N) dct3(x)``."""
    dim = _normalize_dim(x, dim)
    n = x.shape[dim]
    k = _index(x, dim).to(x.dtype)
    weight = torch.where(k == 0, 0.5, 1.0)
    z = x * weight * torch.exp(1j * math.pi * k / (2 * n))
    return 4 * torch.fft.ifft(z, n=2 * n, dim=dim).narrow(dim, 0, n).real


def dst2(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """DST-II along ``dim``: ``X_k = Σ_j x_j sin(π(k+1)(2j+1) / 2N)``."""
    dim = _normalize_dim(x, dim)
    return dct2(x * _alternating_sign(x, dim), dim).flip(dim)


def idst2(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Inverse of :func:`dst2`: ``(2 / N) dst3(x)``."""
    dim = _normalize_dim(x, dim)
    return idct2(x.flip(dim), dim) * _alternating_sign(x, dim)


def dct3(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """DCT-III along ``dim``: ``X_k = x_0 / 2 + Σ_{j≥1} x_j cos(πj(2k+1) / 2N)``.

    The transpose of :func:`dct2`, computed as ``(N / 2) idct2(x)``.
    """
    dim = _normalize_dim(x, dim)
    return idct2(x, dim) * (x.shape[dim] / 2)


def idct3(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Inverse of :func:`dct3`: ``(2 / N) dct2(x)``."""
    dim = _normalize_dim(x, dim)
    return dct2(x, dim) * (2 / x.shape[dim])


def dst3(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """DST-III along ``dim``.

    ``X_k = (-1)^k x_{N-1} / 2 + Σ_{j≤N-2} x_j sin(π(2k+1)(j+1) / 2N)``, the
    transpose of :func:`dst2`, computed as ``(N / 2) idst2(x)``.
    """
    dim = _normalize_dim(x, dim)
    return idst2(x, dim) * (x.shape[dim] / 2)


def idst3(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Inverse of :func:`dst3`: ``(2 / N) dst2(x)``."""
    dim = _normalize_dim(x, dim)
    return dst2(x, dim) * (2 / x.shape[dim])


def dct4(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """DCT-IV along ``dim``: ``X_k = Σ_j x_j cos(π(2k+1)(2j+1) / 4N)``."""
    dim = _normalize_dim(x, dim)
    n = x.shape[dim]
    j = _index(x, dim).to(x.dtype)
    v = torch.fft.fft(x * torch.exp(-1j * math.pi * j / (2 * n)), n=2 * n, dim=dim)
    return (v.narrow(dim, 0, n) * torch.exp(-1j * math.pi * (2 * j + 1) / (4 * n))).real


def idct4(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Inverse of :func:`dct4`, which is its own inverse up to a factor of 2 / N."""
    dim = _normalize_dim(x, dim)
    return dct4(x, dim) * (2 / x.shape[dim])


def dst4(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """DST-IV along ``dim``: ``X_k = Σ_j x_j sin(π(2k+1)(2j+1) / 4N)``."""
    dim = _normalize_dim(x, dim)
    return dct4(x.flip(dim), dim) * _alternating_sign(x, dim)


def idst4(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Inverse of :func:`dst4`, which is its own inverse up to a factor of 2 / N."""
    dim = _normalize_dim(x, dim)
    return dst4(x, dim) * (2 / x.shape[dim])


def dst1(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """DST-I along ``dim``: ``X_k = Σ_j x_j sin(π(k+1)(j+1) / (N+1))``."""
    dim = _normalize_dim(x, dim)
    n = x.shape[dim]
    if n == 0:
        return x.clone()
    zero = torch.zeros_like(x.narrow(dim, 0, 1))
    odd = torch.cat([zero, x, zero, -x.flip(dim)], dim=dim)
    return -0.5 * torch.fft.rfft(odd, dim=dim).imag.narrow(dim, 1, n)


def idst1(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Inverse of :func:`dst1`, which is its own inverse up to a factor of 2 / (N+1)."""
    dim = _normalize_dim(x, dim)
    return dst1(x, dim) * (2 / (x.shape[dim] + 1))


def dct1(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """DCT-I along ``dim``, for ``N >= 2``.

    ``X_k = x_0 / 2 + (-1)^k x_{N-1} / 2 + Σ_{j=1}^{N-2} x_j cos(πkj / (N-1))``.
    """
    dim = _normalize_dim(x, dim)
    n = x.shape[dim]
    if n < 2:
        raise ValueError(f"dct1 needs at least 2 points along dim {dim}, got {n}")
    even = torch.cat([x, x.narrow(dim, 1, n - 2).flip(dim)], dim=dim)
    return 0.5 * torch.fft.rfft(even, dim=dim).real


def idct1(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Inverse of :func:`dct1`, which is its own inverse up to a factor of 2 / (N-1)."""
    dim = _normalize_dim(x, dim)
    return dct1(x, dim) * (2 / (x.shape[dim] - 1))
