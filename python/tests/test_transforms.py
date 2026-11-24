import math

import pytest
import scipy.fft
import torch
from orthel.transforms.dct import (
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

# (forward, inverse, scipy forward, scipy inverse, scipy type). SciPy's unnormalised
# transforms carry an extra factor of 2 relative to the definitions here, so
# the inverses here are twice SciPy's inverses.
TRANSFORMS = {
    "dct1": (dct1, idct1, scipy.fft.dct, scipy.fft.idct, 1),
    "dst1": (dst1, idst1, scipy.fft.dst, scipy.fft.idst, 1),
    "dct2": (dct2, idct2, scipy.fft.dct, scipy.fft.idct, 2),
    "dst2": (dst2, idst2, scipy.fft.dst, scipy.fft.idst, 2),
    "dct3": (dct3, idct3, scipy.fft.dct, scipy.fft.idct, 3),
    "dst3": (dst3, idst3, scipy.fft.dst, scipy.fft.idst, 3),
    "dct4": (dct4, idct4, scipy.fft.dct, scipy.fft.idct, 4),
    "dst4": (dst4, idst4, scipy.fft.dst, scipy.fft.idst, 4),
}
SIZES = [1, 2, 3, 8, 17, 64, 257]
DTYPES = [torch.float64, torch.float32]


def _skip_undefined(name, n):
    if name == "dct1" and n < 2:
        pytest.skip("DCT-I needs at least 2 points")


def assert_close_to(actual, expected):
    """Assert agreement to a few ulps of ``max|expected|``, allowing O(log N) FFT error."""
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    eps = torch.finfo(actual.dtype).eps
    n = actual.shape[-1]
    tol = 20 * eps * math.log2(2 * n) * expected.abs().max().clamp_min(1.0)
    err = (actual - expected).abs().max()
    assert err <= tol, f"max error {err:.2e} exceeds {tol:.2e}"


def random(shape, dtype, seed=0, **kwargs):
    return torch.randn(
        shape, dtype=dtype, generator=torch.Generator().manual_seed(seed), **kwargs
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("name", TRANSFORMS)
def test_forward_matches_scipy(name, n, dtype):
    _skip_undefined(name, n)
    forward, _, scipy_fn, _, kind = TRANSFORMS[name]
    x = random((3, n), dtype)

    expected = torch.from_numpy(
        scipy_fn(x.double().numpy(), type=kind, axis=-1) / 2
    ).to(dtype)

    assert_close_to(forward(x), expected)


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("name", TRANSFORMS)
def test_inverse_matches_scipy(name, n, dtype):
    _skip_undefined(name, n)
    _, inverse, _, scipy_inverse, kind = TRANSFORMS[name]
    x = random((3, n), dtype)

    expected = torch.from_numpy(
        2 * scipy_inverse(x.double().numpy(), type=kind, axis=-1)
    ).to(dtype)

    assert_close_to(inverse(x), expected)


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("name", TRANSFORMS)
def test_inverse_round_trip(name, n, dtype):
    _skip_undefined(name, n)
    forward, inverse, *_ = TRANSFORMS[name]
    x = random((3, n), dtype)

    assert_close_to(inverse(forward(x)), x)
    assert_close_to(forward(inverse(x)), x)


@pytest.mark.parametrize("n", [1, 5, 16])
@pytest.mark.parametrize("name", TRANSFORMS)
def test_matches_definition(name, n):
    _skip_undefined(name, n)
    # Independent of SciPy: build the transform matrix from the docstring formula.
    forward, *_ = TRANSFORMS[name]
    k = torch.arange(n, dtype=torch.float64)[:, None]
    j = torch.arange(n, dtype=torch.float64)[None, :]
    matrices = {
        "dct1": torch.cos(math.pi * k * j / max(n - 1, 1))
        * torch.where((j == 0) | (j == n - 1), 0.5, 1.0),
        "dst1": torch.sin(math.pi * (k + 1) * (j + 1) / (n + 1)),
        "dct2": torch.cos(math.pi * k * (2 * j + 1) / (2 * n)),
        "dst2": torch.sin(math.pi * (k + 1) * (2 * j + 1) / (2 * n)),
        "dct3": torch.cos(math.pi * j * (2 * k + 1) / (2 * n))
        * torch.where(j == 0, 0.5, 1.0),
        "dst3": torch.sin(math.pi * (2 * k + 1) * (j + 1) / (2 * n))
        * torch.where(j == n - 1, 0.5, 1.0),
        "dct4": torch.cos(math.pi * (2 * k + 1) * (2 * j + 1) / (4 * n)),
        "dst4": torch.sin(math.pi * (2 * k + 1) * (2 * j + 1) / (4 * n)),
    }
    x = random((4, n), torch.float64)

    assert_close_to(forward(x), x @ matrices[name].T)


@pytest.mark.parametrize("name", TRANSFORMS)
def test_acts_on_last_dim_with_batch(name):
    forward, inverse, *_ = TRANSFORMS[name]
    x = random((2, 3, 4, 9), torch.float64)

    y = forward(x)

    assert y.shape == x.shape
    assert_close_to(y[1, 2], forward(x[1, 2]))
    assert_close_to(forward(x[0, 1, 3]), y[0, 1, 3])
    assert_close_to(inverse(y), x)


@pytest.mark.parametrize("name", TRANSFORMS)
def test_linear(name):
    forward, inverse, *_ = TRANSFORMS[name]
    x, y = (
        random((3, 12), torch.float64, seed=1),
        random((3, 12), torch.float64, seed=2),
    )

    for fn in (forward, inverse):
        assert_close_to(fn(2.5 * x - y), 2.5 * fn(x) - fn(y))


@pytest.mark.parametrize("name", TRANSFORMS)
def test_gradcheck(name):
    forward, inverse, *_ = TRANSFORMS[name]
    x = random((2, 7), torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(forward, (x,))
    assert torch.autograd.gradcheck(inverse, (x,))


@pytest.mark.parametrize("name", TRANSFORMS)
def test_does_not_modify_input(name):
    forward, inverse, *_ = TRANSFORMS[name]
    x = random((3, 10), torch.float64)
    original = x.clone()

    forward(x)
    inverse(x)

    assert torch.equal(x, original)


@pytest.mark.parametrize("dim", [0, 1, 2, -1, -2, -3])
@pytest.mark.parametrize("name", TRANSFORMS)
def test_dim_matches_scipy_axis(name, dim):
    forward, inverse, scipy_fn, scipy_inverse, kind = TRANSFORMS[name]
    x = random((5, 6, 7), torch.float64)

    expected = torch.from_numpy(scipy_fn(x.numpy(), type=kind, axis=dim) / 2)
    expected_inverse = torch.from_numpy(
        2 * scipy_inverse(x.numpy(), type=kind, axis=dim)
    )

    assert_close_to(forward(x, dim=dim).movedim(dim, -1), expected.movedim(dim, -1))
    assert_close_to(
        inverse(x, dim=dim).movedim(dim, -1), expected_inverse.movedim(dim, -1)
    )
    assert_close_to(inverse(forward(x, dim), dim), x)


@pytest.mark.parametrize("dim", [0, 1, -2])
@pytest.mark.parametrize("name", TRANSFORMS)
def test_dim_gradcheck(name, dim):
    forward, inverse, *_ = TRANSFORMS[name]
    x = random((4, 3, 5), torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(lambda t: forward(t, dim), (x,))
    assert torch.autograd.gradcheck(lambda t: inverse(t, dim), (x,))


@pytest.mark.parametrize("dim", [3, -4])
@pytest.mark.parametrize("name", TRANSFORMS)
def test_dim_out_of_range(name, dim):
    forward, inverse, *_ = TRANSFORMS[name]
    x = random((2, 3, 4), torch.float64)

    for fn in (forward, inverse):
        with pytest.raises(IndexError, match="out of range"):
            fn(x, dim)


def test_dct1_rejects_single_point():
    with pytest.raises(ValueError, match="at least 2"):
        dct1(torch.zeros(3, 1))


@pytest.mark.parametrize("n", [1, 4, 9])
@pytest.mark.parametrize("pair", [("dct2", "dct3"), ("dst2", "dst3")])
def test_type3_is_transpose_of_type2(pair, n):
    # DCT-III/DST-III differ from the transposed type-II matrices only in the
    # half weight on one input; check against the type-II transforms directly.
    type2, type3 = (TRANSFORMS[name][0] for name in pair)
    eye = torch.eye(n, dtype=torch.float64)
    weight = torch.ones(n, dtype=torch.float64)
    weight[0 if pair[0] == "dct2" else -1] = 0.5

    assert_close_to(type3(eye, dim=0), type2(eye, dim=0).T * weight)
