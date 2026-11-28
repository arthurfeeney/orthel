import math

import pytest
import scipy.fft
import torch
from orthel.transforms.dtt import (
    KINDS,
    dct,
    dct1,
    dct2,
    dct3,
    dct4,
    dst,
    dst1,
    dst2,
    dst3,
    dst4,
    dtt,
    idct,
    idct1,
    idct2,
    idct3,
    idct4,
    idst,
    idst1,
    idst2,
    idst3,
    idst4,
    idtt,
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
# Includes powers of two, node-grid sizes 2^k +- 1, composite and prime sizes.
SIZES = [1, 2, 3, 8, 17, 64, 96, 100, 257, 1023, 1024, 1025, 1031, 4096]
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


@pytest.mark.parametrize("name", TRANSFORMS)
def test_non_contiguous_input(name):
    forward, inverse, scipy_fn, scipy_inverse, kind = TRANSFORMS[name]
    base = random((12, 20), torch.float64)
    for x in (base.t(), base[:, ::2], base[1:, 3:].t()):
        expected = torch.from_numpy(scipy_fn(x.numpy(), type=kind, axis=0) / 2)
        assert_close_to(forward(x, 0).movedim(0, -1), expected.movedim(0, -1))
        assert_close_to(inverse(forward(x, 0), 0), x)


@pytest.mark.parametrize("name", TRANSFORMS)
def test_empty_batch(name):
    forward, inverse, *_ = TRANSFORMS[name]
    x = torch.zeros(0, 6, dtype=torch.float64)
    assert forward(x).shape == inverse(x).shape == (0, 6)


@pytest.mark.parametrize(
    "name, n",
    [(name, 65536) for name in ("dct2", "dst2", "dct3", "dct4", "dst4")]
    + [("dct1", 65537), ("dst1", 65535)],
)
def test_float32_error_stays_near_eps_at_large_sizes(name, n):
    forward, inverse, scipy_fn, scipy_inverse, kind = TRANSFORMS[name]
    x = random((2, n), torch.float64)
    expected = scipy_fn(x.numpy(), type=kind, axis=-1) / 2
    expected_inverse = 2 * scipy_inverse(x.numpy(), type=kind, axis=-1)

    for fn, reference in ((forward, expected), (inverse, expected_inverse)):
        reference = torch.from_numpy(reference)
        actual = fn(x.float()).double()
        relative = (actual - reference).norm() / reference.norm()
        assert relative < 4 * torch.finfo(torch.float32).eps


@pytest.mark.parametrize("name", ["dct2", "dst1", "dct4"])
def test_thread_count_does_not_change_results(name):
    forward, *_ = TRANSFORMS[name]
    x = random((64, 4095 if name == "dst1" else 4096), torch.float64)
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        one = forward(x)
        torch.set_num_threads(4)
        four = forward(x)
    finally:
        torch.set_num_threads(previous)
    assert torch.equal(one, four)


@pytest.mark.parametrize("name", TRANSFORMS)
def test_gradgradcheck(name):
    forward, inverse, *_ = TRANSFORMS[name]
    x = random((3, 6, 5), torch.float64, requires_grad=True)

    assert torch.autograd.gradgradcheck(lambda t: forward(t, 1), (x,))
    assert torch.autograd.gradgradcheck(lambda t: inverse(t, 1), (x,))


def test_compiles_without_graph_breaks():
    def fn(x):
        y = dct2(x, 0)
        return idst1(y.sin(), 1) * 2

    x = random((8, 9), torch.float64)
    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True, backend="eager")
    assert_close_to(compiled(x), fn(x))


# dct, idct, dst and idst: the type as an argument, along any dimensions.

FAMILIES = {
    "dct": (dct, idct, scipy.fft.dctn, scipy.fft.idctn),
    "dst": (dst, idst, scipy.fft.dstn, scipy.fft.idstn),
}
TYPES = [1, 2, 3, 4]


@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("type", TYPES)
@pytest.mark.parametrize("family", FAMILIES)
def test_typed_matches_numbered_on_one_dim(family, type, dim):
    forward, inverse, *_ = FAMILIES[family]
    numbered, numbered_inverse, *_ = TRANSFORMS[f"{family}{type}"]
    x = random((5, 6, 7), torch.float64)

    assert torch.equal(forward(x, type, dim), numbered(x, dim))
    assert torch.equal(inverse(x, type, dim), numbered_inverse(x, dim))


@pytest.mark.parametrize("type", TYPES)
@pytest.mark.parametrize("family", FAMILIES)
def test_typed_defaults_to_last_dim(family, type):
    forward, *_ = FAMILIES[family]
    numbered, *_ = TRANSFORMS[f"{family}{type}"]
    x = random((5, 6), torch.float64)

    assert torch.equal(forward(x, type), numbered(x, -1))


def test_typed_defaults_to_type_2():
    x = random((5, 6), torch.float64)
    assert torch.equal(dct(x), dct2(x))
    assert torch.equal(idst(x), idst2(x))


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("dim", [(0, 2), (-1, 1), (0, 1, 2), None])
@pytest.mark.parametrize("type", TYPES)
@pytest.mark.parametrize("family", FAMILIES)
def test_nd_matches_scipy(family, type, dim, dtype):
    forward, inverse, scipy_forward, scipy_inverse = FAMILIES[family]
    x = random((9, 16, 17), dtype)
    axes = None if dim is None else list(dim)
    ndims = x.ndim if dim is None else len(dim)
    ref = x.double().numpy()

    expected = scipy_forward(ref, type=type, axes=axes) / 2**ndims
    expected_inverse = scipy_inverse(ref, type=type, axes=axes) * 2**ndims

    assert_close_to(forward(x, type, dim), torch.from_numpy(expected).to(dtype))
    assert_close_to(inverse(x, type, dim), torch.from_numpy(expected_inverse).to(dtype))


@pytest.mark.parametrize("type", TYPES)
@pytest.mark.parametrize("family", FAMILIES)
def test_nd_inverse_round_trip(family, type):
    forward, inverse, *_ = FAMILIES[family]
    x = random((6, 8, 9), torch.float64)

    assert_close_to(inverse(forward(x, type, None), type, None), x)
    assert_close_to(forward(inverse(x, type, (0, 2)), type, (0, 2)), x)


@pytest.mark.parametrize("type", TYPES)
@pytest.mark.parametrize("family", FAMILIES)
def test_nd_does_not_depend_on_dim_order(family, type):
    forward, *_ = FAMILIES[family]
    x = random((6, 8, 9), torch.float64)

    assert_close_to(forward(x, type, (2, 0)), forward(x, type, (0, 2)))


def test_nd_equals_numbered_in_turn():
    x = random((6, 8, 9), torch.float64)
    assert torch.equal(dst(x, 1, (0, 2)), dst1(dst1(x, 0), 2))


@pytest.mark.parametrize("type", TYPES)
@pytest.mark.parametrize("family", FAMILIES)
def test_nd_gradcheck(family, type):
    forward, inverse, *_ = FAMILIES[family]
    x = random((3, 4, 5), torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(lambda t: forward(t, type, (0, 2)), (x,))
    assert torch.autograd.gradcheck(lambda t: inverse(t, type, None), (x,))


@pytest.mark.parametrize("dim", [(), []])
def test_nd_with_no_dims_returns_a_copy(dim):
    x = random((4, 5), torch.float64)
    y = dct(x, 2, dim)
    assert torch.equal(y, x) and y.data_ptr() != x.data_ptr()


def test_nd_on_zero_dim_tensor_with_all_dims_is_a_copy():
    x = torch.tensor(3.0, dtype=torch.float64)
    assert torch.equal(dct(x, 2, None), x)


@pytest.mark.parametrize("type", [0, 5, -1, 2.0, True, "2", None])
def test_invalid_type_raises(type):
    x = random((4, 5), torch.float64)
    for fn in (dct, idct, dst, idst):
        with pytest.raises(ValueError, match="type must be 1, 2, 3 or 4"):
            fn(x, type)


@pytest.mark.parametrize("dim", [(1, 1), (0, -3), (2, 0, 2)])
def test_repeated_dims_raise(dim):
    with pytest.raises(ValueError, match="repeats a dimension"):
        dct(random((4, 5, 6), torch.float64), 2, dim)


@pytest.mark.parametrize("dim", [3, (0, 3), (-4,)])
def test_nd_dim_out_of_range(dim):
    with pytest.raises(IndexError, match="out of range"):
        dst(random((4, 5, 6), torch.float64), 2, dim)


def test_nd_dct1_rejects_single_point_on_any_dim():
    with pytest.raises(ValueError, match="at least 2"):
        dct(torch.zeros(4, 1, 5, dtype=torch.float64), 1, None)
    # The single-point dimension is fine when it is not transformed.
    dct(torch.zeros(4, 1, 5, dtype=torch.float64), 1, (0, 2))


def test_nd_compiles_without_graph_breaks():
    def fn(x):
        return idst(dct(x, 2, (0, 1)).sin(), 1, None)

    x = random((8, 9, 5), torch.float64)
    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True, backend="eager")
    assert_close_to(compiled(x), fn(x))


# dtt and idtt: the discrete trigonometric transform, one kind per dimension.

SCIPY_BY_FAMILY = {
    "dct": (scipy.fft.dct, scipy.fft.idct),
    "dst": (scipy.fft.dst, scipy.fft.idst),
}


def scipy_dtt(x, kinds, dims, inverse=False):
    """SciPy reference in orthel's scaling: half of SciPy per forward axis."""
    out = x.double().numpy()
    for kind, d in zip(kinds, dims):
        forward, backward = SCIPY_BY_FAMILY[kind[:3]]
        if inverse:
            out = 2 * backward(out, type=int(kind[3]), axis=d)
        else:
            out = forward(out, type=int(kind[3]), axis=d) / 2
    return torch.from_numpy(out).to(x.dtype)


def test_kinds_are_the_forward_transforms():
    assert set(KINDS) == set(TRANSFORMS)


@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("kind", KINDS)
def test_dtt_matches_numbered_on_one_dim(kind, dim):
    forward, inverse, *_ = TRANSFORMS[kind]
    x = random((5, 6, 7), torch.float64)

    assert torch.equal(dtt(x, kind, dim), forward(x, dim))
    assert torch.equal(idtt(x, kind, dim), inverse(x, dim))
    assert torch.equal(dtt(x, [kind], dim), forward(x, dim))


@pytest.mark.parametrize("type", TYPES)
@pytest.mark.parametrize("family", FAMILIES)
def test_dtt_with_one_kind_matches_typed(family, type):
    forward, inverse, *_ = FAMILIES[family]
    x = random((5, 6, 7), torch.float64)

    assert torch.equal(dtt(x, f"{family}{type}", None), forward(x, type, None))
    assert torch.equal(idtt(x, f"{family}{type}", (0, 2)), inverse(x, type, (0, 2)))


MIXED = [
    (("dct1", "dst2"), (0, 1)),
    (("dst1", "dct2", "dst4"), (2, 0, 1)),
    (("dct3", "dst3"), (-1, 0)),
    (("dct4", "dst1", "dct2"), None),
]


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("kinds, dim", MIXED)
def test_mixed_dtt_matches_scipy(kinds, dim, dtype):
    x = random((9, 16, 17), dtype)
    dims = range(x.ndim) if dim is None else dim

    assert_close_to(dtt(x, kinds, dim), scipy_dtt(x, kinds, dims))
    assert_close_to(idtt(x, kinds, dim), scipy_dtt(x, kinds, dims, inverse=True))


@pytest.mark.parametrize("kinds, dim", MIXED)
def test_mixed_dtt_round_trip(kinds, dim):
    x = random((6, 8, 9), torch.float64)

    assert_close_to(idtt(dtt(x, kinds, dim), kinds, dim), x)
    assert_close_to(dtt(idtt(x, kinds, dim), kinds, dim), x)


def test_mixed_dtt_pairs_kinds_with_dims_in_order():
    x = random((6, 8, 9), torch.float64)
    assert torch.equal(dtt(x, ("dct1", "dst2"), (2, 0)), dst2(dct1(x, 2), 0))


@pytest.mark.parametrize("kinds, dim", MIXED)
def test_mixed_dtt_gradcheck(kinds, dim):
    x = random((3, 4, 5), torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(lambda t: dtt(t, kinds, dim), (x,))
    assert torch.autograd.gradcheck(lambda t: idtt(t, kinds, dim), (x,))


@pytest.mark.parametrize(
    "kind", ["dct5", "idct2", "DCT2", "dht", "", ("dct2", "fft"), 2, None]
)
def test_dtt_invalid_kind_raises(kind):
    x = random((4, 5), torch.float64)
    for fn in (dtt, idtt):
        with pytest.raises((ValueError, TypeError)):
            fn(x, kind, None if isinstance(kind, tuple) else -1)


def test_dtt_invalid_kind_message():
    with pytest.raises(ValueError, match="kind must be one of dct1, .*dst4"):
        dtt(random((4, 5), torch.float64), "idct2")


@pytest.mark.parametrize(
    "kinds, dim",
    [(("dct2", "dst2"), 0), (("dct2",), (0, 1)), (("dct2", "dst2", "dct1"), None)],
)
def test_dtt_kinds_must_match_dims(kinds, dim):
    x = random((4, 5), torch.float64)
    for fn in (dtt, idtt):
        with pytest.raises(ValueError, match="transforms for"):
            fn(x, kinds, dim)


def test_dtt_with_no_dims_returns_a_copy():
    x = random((4, 5), torch.float64)
    y = dtt(x, (), ())
    assert torch.equal(y, x) and y.data_ptr() != x.data_ptr()


def test_dtt_dct1_rejects_single_point_only_where_used():
    x = torch.zeros(4, 1, 5, dtype=torch.float64)
    with pytest.raises(ValueError, match="at least 2"):
        dtt(x, ("dst2", "dct1"), (0, 1))
    dtt(x, ("dct1", "dst2"), (0, 1))


def test_dtt_compiles_without_graph_breaks():
    def fn(x):
        return idtt(dtt(x, ("dct1", "dst2"), (0, 1)).sin(), ("dst1", "dct4"), (2, 0))

    x = random((8, 9, 5), torch.float64)
    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True, backend="eager")
    assert_close_to(compiled(x), fn(x))
