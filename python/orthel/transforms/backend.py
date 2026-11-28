"""Choosing the implementation behind the DCT/DST functions.

``"torch"`` builds each transform from ``torch.fft`` and runs on any device
and dtype. ``"cpp"`` is orthel's own C++ kernels (radix-2 recursions, no
FFT) for float32 and float64 tensors on the CPU. ``"auto"``, the default,
uses ``"cpp"`` where it is available and fast and ``"torch"`` everywhere
else.

The backend can be set globally with :func:`set_backend`, for a block with
:func:`use_backend`, or for one call with the transforms' ``backend``
argument, which takes precedence. The global setting is shared by all
threads.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Literal, get_args

from . import _cpp

Backend = Literal["auto", "torch", "cpp"]

_BACKENDS = get_args(Backend)
_current: Backend = "auto"


def available_backends() -> tuple[str, ...]:
    """The backends that can run here: ``"torch"``, and ``"cpp"`` if built."""
    return ("torch", "cpp") if _cpp.AVAILABLE else ("torch",)


def get_backend() -> Backend:
    """The backend used when a transform is called without ``backend``."""
    return _current


def set_backend(name: Backend) -> None:
    """Use ``name`` for every transform called without ``backend``."""
    global _current
    _current = check_backend(name)


@contextlib.contextmanager
def use_backend(name: Backend) -> Iterator[None]:
    """Use ``name`` inside a ``with`` block, then restore the previous backend."""
    global _current
    previous = _current
    _current = check_backend(name)
    try:
        yield
    finally:
        _current = previous


def check_backend(name: str) -> Backend:
    if name not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {name!r}")
    if name == "cpp" and not _cpp.AVAILABLE:
        raise RuntimeError(
            "the cpp backend is not available: orthel was installed without "
            "its C++ extension"
        )
    return name
