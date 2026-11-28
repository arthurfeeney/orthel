"""Choosing a transform backend, and the C++ backend's internals.

How each backend's transforms behave is tested in test_transforms.py, which
runs once per backend (see conftest.py).
"""

import orthel.transforms.dtt as D
import pytest
import torch
from orthel.transforms import _cpp, _torch
from orthel.transforms.backend import (
    available_backends,
    get_backend,
    set_backend,
    use_backend,
)

needs_cpp = pytest.mark.skipif(not _cpp.AVAILABLE, reason="C++ backend not built")

# The numbered transforms, which each reach one backend call.
OPS = [op for op in D.__all__ if op[-1].isdigit()]


def random(shape, dtype=torch.float64, seed=0, **kwargs):
    return torch.randn(
        shape, dtype=dtype, generator=torch.Generator().manual_seed(seed), **kwargs
    )


def valid(op, n):
    return n >= 2 or not op.endswith("ct1")


@pytest.fixture
def backend():
    """Overrides conftest's: these tests choose backends themselves, starting
    from the default."""
    return None


@pytest.fixture
def calls(monkeypatch):
    """Record which backend each transform call reaches."""
    seen = []
    if _cpp.AVAILABLE:
        cpp = _cpp.r2r
        monkeypatch.setattr(
            _cpp, "r2r", lambda *a: seen.append("cpp") or cpp(*a), raising=True
        )
    for op in OPS:
        impl = getattr(_torch, op)
        monkeypatch.setattr(
            _torch, op, lambda *a, impl=impl: seen.append("torch") or impl(*a)
        )
    return seen


# Selecting a backend.


def test_default_backend_is_auto():
    assert get_backend() == "auto"


def test_torch_is_always_available():
    assert "torch" in available_backends()


def test_set_backend_changes_default():
    try:
        set_backend("torch")
        assert get_backend() == "torch"
    finally:
        set_backend("auto")
    assert get_backend() == "auto"


def test_use_backend_restores_previous_on_error():
    with pytest.raises(RuntimeError, match="boom"):
        with use_backend("torch"):
            assert get_backend() == "torch"
            raise RuntimeError("boom")
    assert get_backend() == "auto"


@pytest.mark.parametrize("name", ["fft", "CPP", ""])
def test_unknown_backend_raises(name):
    x = random((2, 8))
    with pytest.raises(ValueError, match="backend must be one of"):
        set_backend(name)
    with pytest.raises(ValueError, match="backend must be one of"):
        with use_backend(name):
            pass
    with pytest.raises(ValueError, match="backend must be one of"):
        D.dct2(x, backend=name)
    assert get_backend() == "auto"


def test_cpp_unavailable_raises(monkeypatch):
    monkeypatch.setattr(_cpp, "AVAILABLE", False)
    assert available_backends() == ("torch",)
    with pytest.raises(RuntimeError, match="not available"):
        set_backend("cpp")
    with pytest.raises(RuntimeError, match="not available"):
        D.dct2(random((2, 8)), backend="cpp")
    # auto quietly uses torch.
    assert torch.equal(D.dct2(random((2, 8))), _torch.dct2(random((2, 8)), -1))


def test_torch_backend_never_calls_cpp(calls):
    with use_backend("torch"):
        D.dct2(random((2, 8)))
    assert calls == ["torch"]


@needs_cpp
def test_per_call_backend_overrides_global(calls):
    x = random((2, 8))
    with use_backend("torch"):
        D.dct2(x, backend="cpp")
    with use_backend("cpp"):
        D.dct2(x, backend="torch")
    assert calls == ["cpp", "torch"]


def test_nd_transform_passes_backend_to_each_dim(calls):
    D.dct(random((4, 5, 6)), 2, (0, 2), backend="torch")
    assert calls == ["torch", "torch"]


def test_dtt_with_torch_backend_never_calls_cpp(calls):
    D.idtt(random((4, 5, 6)), ("dct1", "dst4"), (2, 0), backend="torch")
    # Some torch transforms are built on others, so count backends, not calls.
    assert set(calls) == {"torch"}


@needs_cpp
def test_dtt_passes_backend_to_each_dim(calls):
    D.idtt(random((4, 5, 6)), ("dct1", "dst4"), (2, 0), backend="cpp")
    assert calls == ["cpp", "cpp"]


def test_nd_transform_checks_backend_with_no_dims():
    with pytest.raises(ValueError, match="backend must be one of"):
        D.dct(random((4, 5)), 2, (), backend="fft")


@needs_cpp
@pytest.mark.parametrize(
    "shape, dtype, expected",
    [
        ((3, 64), torch.float64, "cpp"),
        ((3, 96), torch.float32, "cpp"),  # 96 = 32 * 3
        ((3, 1031), torch.float64, "torch"),  # prime: one 1031-point direct sum
        ((3, 8), torch.float16, "torch"),
    ],
)
def test_auto_picks_cpp_only_when_supported_and_fast(calls, shape, dtype, expected):
    x = random(shape).to(dtype)
    if dtype == torch.float16:
        # The torch backend may not support half on the CPU; only the choice
        # of backend matters here.
        try:
            D.dct2(x)
        except RuntimeError:
            pass
    else:
        D.dct2(x)
    assert calls == [expected]


@needs_cpp
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int64])
def test_cpp_rejects_unsupported_dtypes(dtype):
    x = torch.ones(2, 8, dtype=dtype)
    with pytest.raises(TypeError, match="float32 or float64 CPU tensor"):
        D.dct2(x, backend="cpp")


@needs_cpp
def test_cpp_rejects_complex():
    x = torch.ones(2, 8, dtype=torch.complex128)
    with pytest.raises(TypeError, match="float32 or float64 CPU tensor"):
        D.dct2(x, backend="cpp")


@needs_cpp
def test_dct1_single_point_raises_on_cpp():
    with pytest.raises(ValueError, match="at least 2"):
        D.dct1(torch.zeros(3, 1), backend="cpp")
    with pytest.raises(ValueError, match="at least 2"):
        D.idct1(torch.zeros(3, 1), backend="cpp")


def test_max_direct_size_is_odd_part_for_type_2_and_4():
    assert _cpp.max_direct_size("dct2", 96) == 3
    assert _cpp.max_direct_size("dct4", 1024) == 1
    assert _cpp.max_direct_size("idst2", 1031) == 1031
    # Node grids: n + 1 points for DCT-I and n - 1 for DST-I.
    assert _cpp.max_direct_size("dct1", 1025) == 2
    assert _cpp.max_direct_size("dst1", 1023) == 1


@needs_cpp
def test_max_direct_size_matches_cpp_plans():
    for op in OPS:
        for n in range(2 if op.endswith("ct1") else 1, 300):
            assert _cpp.max_direct_size(op, n) == _cpp._r2r.max_direct_size(op, n)


# The C++ backend against the torch backend, including slow odd sizes.


@needs_cpp
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=str)
@pytest.mark.parametrize(
    "n", [2, 5, 6, 12, 96, 100, 127, 129, 255, 1001, 1031, 2048, 4095, 4097]
)
@pytest.mark.parametrize("op", OPS)
def test_cpp_matches_torch(op, n, dtype):
    x = random((3, n))
    expected = getattr(D, op)(x, backend="torch")

    actual = getattr(D, op)(x.to(dtype), backend="cpp")

    assert actual.dtype == dtype
    scale = expected.abs().max().clamp_min(1.0)
    # Recursion steps add O(eps) each, but a direct sum over m points adds
    # O(sqrt(m) eps): allow for that on the large odd sizes "auto" avoids.
    direct = max(1.0, _cpp.max_direct_size(op, n) ** 0.5 / 8)
    tol = (1e-13 if dtype == torch.float64 else 2e-6) * direct
    assert (actual.double() - expected).abs().max() <= tol * scale


# The C++ op: transposes for autograd, and torch.library's checks.


@needs_cpp
@pytest.mark.parametrize("n", [1, 2, 3, 4, 7, 8, 16])
@pytest.mark.parametrize("op", OPS)
def test_cpp_transpose_is_matrix_transpose(op, n):
    if not valid(op, n):
        pytest.skip("DCT-I needs at least 2 points")
    eye = torch.eye(n, dtype=torch.float64)
    matrix = getattr(D, op)(eye, 0, backend="torch")
    assert torch.allclose(_cpp.r2r(eye, 0, op, True), matrix.T, rtol=0, atol=1e-13)


@needs_cpp
@pytest.mark.parametrize("op", ["dct1", "dst2", "idct3", "dst4"])
def test_cpp_op_passes_opcheck(op):
    x = random((3, 8), requires_grad=True)
    torch.library.opcheck(_cpp.r2r, (x, 1, op, False))
    torch.library.opcheck(_cpp.r2r, (x, 0, op, True))
