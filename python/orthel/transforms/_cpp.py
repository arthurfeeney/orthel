"""The ``"cpp"`` backend: orthel's C++ kernels as a ``torch.library`` op.

The op ``orthel::r2r`` applies a named transform, or its transpose, along one
dimension. Autograd uses the transpose (every transform is linear), and the
fake implementation lets ``torch.compile`` trace through it.
"""

from __future__ import annotations

import torch

try:
    from . import _r2r
except ImportError:  # installed without the extension
    _r2r = None

AVAILABLE = _r2r is not None
DTYPES = (torch.float32, torch.float64)

# Odd lengths left after halving are summed directly in O(m^2). Above this
# size "auto" prefers the torch backend.
AUTO_MAX_DIRECT = 64

# The C++ kernel each transform runs on. dct2 and idct2 split alike, so the
# plan sizes only depend on these four families.
_KERNELS = {
    **dict.fromkeys(("dct1", "idct1"), "dct1"),
    **dict.fromkeys(("dst1", "idst1"), "dst1"),
    **dict.fromkeys(("dct2", "idct2", "dct3", "idct3"), "dct2"),
    **dict.fromkeys(("dst2", "idst2", "dst3", "idst3"), "dct2"),
    **dict.fromkeys(("dct4", "idct4", "dst4", "idst4"), "dct4"),
}


def max_direct_size(op: str, n: int) -> int:
    """Largest direct sum the C++ plan for ``op`` on ``n`` points uses.

    Mirrors the splits in ``cpp/cpu/r2r.hpp`` in plain Python, so that choosing a
    backend does not call into the extension (and ``torch.compile`` can trace
    it).
    """
    kernel = _KERNELS[op]
    if kernel in ("dct2", "dct4"):
        while n >= 2 and n % 2 == 0:
            n //= 2
        return n
    if n < 3 or n % 2 == 0:
        return n
    if kernel == "dct1":  # dct1(2m+1) -> dct1(m+1), idct2(m)
        m = (n - 1) // 2
        return max(max_direct_size("dct1", m + 1), max_direct_size("dct2", m))
    m = (n + 1) // 2  # dst1(2m-1) -> dst1(m-1), idct2(m)
    return max(max_direct_size("dst1", m - 1), max_direct_size("dct2", m))


def supports(x: torch.Tensor) -> bool:
    return x.device.type == "cpu" and x.dtype in DTYPES


def is_fast(op: str, x: torch.Tensor, dim: int) -> bool:
    return max_direct_size(op, x.shape[dim]) <= AUTO_MAX_DIRECT


if AVAILABLE:

    @torch.library.custom_op("orthel::r2r", mutates_args=())
    def r2r(x: torch.Tensor, dim: int, op: str, transpose: bool) -> torch.Tensor:
        """Transform ``op`` (or its transpose) of ``x`` along ``dim >= 0``."""
        x = x.contiguous()
        out = torch.empty_like(x)
        _r2r.r2r(x, out, dim, op, transpose, torch.get_num_threads())
        return out

    @r2r.register_fake
    def _(x: torch.Tensor, dim: int, op: str, transpose: bool) -> torch.Tensor:
        return torch.empty(x.shape, dtype=x.dtype, device=x.device)

    def _setup_context(ctx, inputs, output) -> None:
        _, ctx.dim, ctx.op, ctx.transpose = inputs

    def _backward(ctx, grad: torch.Tensor):
        return r2r(grad, ctx.dim, ctx.op, not ctx.transpose), None, None, None

    r2r.register_autograd(_backward, setup_context=_setup_context)
