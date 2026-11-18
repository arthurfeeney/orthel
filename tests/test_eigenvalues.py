import pytest
import torch

from torch_poisson.eigenvalues import second_difference_eigenvalues
from torch_poisson.transforms import idct2, idct4, idst2, idst4

# (low, high) wall types -> inverse transform, whose columns applied to the
# identity are the eigenvectors in the order the eigenvalues are returned.
WALL_INVERSES = {
    ("neumann", "neumann"): idct2,
    ("dirichlet", "dirichlet"): idst2,
    ("neumann", "dirichlet"): idct4,
    ("dirichlet", "neumann"): idst4,
}
SIZES = [1, 2, 5, 8, 17]


def second_difference_matrix(n, h, bc):
    """Dense cell-centred 3-point second difference with ghost cells."""
    a = -2 * torch.eye(n, dtype=torch.float64)
    a += torch.diag(torch.ones(n - 1, dtype=torch.float64), 1)
    a += torch.diag(torch.ones(n - 1, dtype=torch.float64), -1)
    if bc == ("periodic", "periodic"):
        a[0, -1] += 1
        a[-1, 0] += 1
    else:
        a[0, 0] += 1 if bc[0] == "neumann" else -1
        a[-1, -1] += 1 if bc[1] == "neumann" else -1
    return a / h**2


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("bc", list(WALL_INVERSES), ids="/".join)
def test_wall_eigenpairs_match_transform(bc, n):
    h = 0.3
    lam = second_difference_eigenvalues(n, h, bc)
    vectors = WALL_INVERSES[bc](torch.eye(n, dtype=torch.float64), dim=0)

    torch.testing.assert_close(second_difference_matrix(n, h, bc) @ vectors, vectors * lam)


@pytest.mark.parametrize("n", SIZES)
def test_periodic_eigenpairs_match_fft(n):
    h = 0.3
    bc = ("periodic", "periodic")
    lam = second_difference_eigenvalues(n, h, bc)
    vectors = torch.fft.ifft(torch.eye(n, dtype=torch.complex128), dim=0)

    a = second_difference_matrix(n, h, bc).to(torch.complex128)
    torch.testing.assert_close(a @ vectors, vectors * lam)


@pytest.mark.parametrize("n", SIZES)
def test_half_spectrum_matches_rfft_layout(n):
    bc = ("periodic", "periodic")
    full = second_difference_eigenvalues(n, 0.3, bc)
    half = second_difference_eigenvalues(n, 0.3, bc, half_spectrum=True)

    assert half.shape == (n // 2 + 1,)
    torch.testing.assert_close(half, full[: n // 2 + 1])


@pytest.mark.parametrize("bc", [("periodic", "periodic"), *WALL_INVERSES], ids="/".join)
def test_eigenvalues_are_spectrum_of_matrix(bc):
    n, h = 9, 0.25
    lam = second_difference_eigenvalues(n, h, bc)
    expected = torch.linalg.eigvalsh(second_difference_matrix(n, h, bc))

    torch.testing.assert_close(lam.sort().values, expected.sort().values)


def test_dtype_and_device():
    lam = second_difference_eigenvalues(6, 0.5, ("neumann", "neumann"), dtype=torch.float32)

    assert lam.dtype == torch.float32 and lam.device.type == "cpu"


def test_half_spectrum_rejected_for_walls():
    with pytest.raises(ValueError, match="periodic"):
        second_difference_eigenvalues(6, 0.5, ("neumann", "dirichlet"), half_spectrum=True)


@pytest.mark.parametrize("bc", [("periodic", "neumann"), ("robin", "robin")])
def test_unsupported_boundary_types_rejected(bc):
    with pytest.raises(ValueError, match="unsupported"):
        second_difference_eigenvalues(6, 0.5, bc)
