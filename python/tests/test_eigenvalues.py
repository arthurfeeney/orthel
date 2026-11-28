import pytest
import torch
from orthel.eigenvalues import second_difference_eigenvalues
from orthel.transforms.dtt import idct1, idct2, idct3, idct4, idst1, idst2, idst3, idst4

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

    torch.testing.assert_close(
        second_difference_matrix(n, h, bc) @ vectors, vectors * lam
    )


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
    lam = second_difference_eigenvalues(
        6, 0.5, ("neumann", "neumann"), dtype=torch.float32
    )

    assert lam.dtype == torch.float32 and lam.device.type == "cpu"


def test_half_spectrum_rejected_for_walls():
    with pytest.raises(ValueError, match="periodic"):
        second_difference_eigenvalues(
            6, 0.5, ("neumann", "dirichlet"), half_spectrum=True
        )


@pytest.mark.parametrize("bc", [("periodic", "neumann"), ("robin", "robin")])
def test_unsupported_boundary_types_rejected(bc):
    with pytest.raises(ValueError, match="unsupported"):
        second_difference_eigenvalues(6, 0.5, bc)


# Node grids: inverse transforms whose columns are the eigenvectors on the
# unknown nodes (Dirichlet wall nodes are known and excluded).
NODE_INVERSES = {
    ("neumann", "neumann"): idct1,
    ("dirichlet", "dirichlet"): idst1,
    ("neumann", "dirichlet"): idct3,
    ("dirichlet", "neumann"): idst3,
}


def node_second_difference_matrix(n, h, bc):
    """Dense 3-point second difference on the unknown nodes of ``n`` cells.

    Neumann wall nodes use the mirrored ghost ``u_{-1} = u_1``.
    """
    m = n - 1 + bc.count("neumann")
    a = -2 * torch.eye(m, dtype=torch.float64)
    a += torch.diag(torch.ones(m - 1, dtype=torch.float64), 1)
    a += torch.diag(torch.ones(m - 1, dtype=torch.float64), -1)
    if bc[0] == "neumann" and m > 1:
        a[0, 1] = 2
    if bc[1] == "neumann" and m > 1:
        a[-1, -2] = 2
    return a / h**2


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("bc", list(NODE_INVERSES), ids="/".join)
def test_node_eigenpairs_match_transform(bc, n):
    h = 0.3
    m = n - 1 + bc.count("neumann")
    lam = second_difference_eigenvalues(n, h, bc, grid="node")
    assert lam.shape == (m,)
    if m == 0:  # n = 1 with two Dirichlet walls: both nodes are known
        return
    vectors = NODE_INVERSES[bc](torch.eye(m, dtype=torch.float64), dim=0)

    torch.testing.assert_close(
        node_second_difference_matrix(n, h, bc) @ vectors, vectors * lam
    )


@pytest.mark.parametrize("n", SIZES)
def test_periodic_node_eigenvalues_match_cells(n):
    bc = ("periodic", "periodic")
    torch.testing.assert_close(
        second_difference_eigenvalues(n, 0.3, bc, grid="node"),
        second_difference_eigenvalues(n, 0.3, bc),
    )


def test_invalid_grid_raises():
    with pytest.raises(ValueError, match="grid"):
        second_difference_eigenvalues(4, 0.3, ("neumann", "neumann"), grid="edge")
