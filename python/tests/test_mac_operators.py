import math

import pytest
import torch
from orthel.boundary import Dirichlet, Neumann, Periodic
from orthel.mac_operators import (
    mac_curl,
    mac_divergence,
    mac_gradient,
    mac_vel_from_streamfunction,
)
from orthel.uniform_grid_solver import UniformGridSolver


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_divergence_of_gradient_is_periodic_laplacian(dtype):
    bc = (Periodic(), Periodic())
    spacing = (0.5, 0.25)
    dy, dx = spacing
    grid = torch.randn(2, 4, 5, dtype=dtype)

    # div(grad(grid)) == laplacian(grid)
    facex, facey = mac_gradient(grid, spacing, bc)
    result = mac_divergence(facex, facey, spacing, bc)

    expected = (grid.roll(-1, -1) - 2 * grid + grid.roll(1, -1)) / dx**2 + (
        grid.roll(-1, -2) - 2 * grid + grid.roll(1, -2)
    ) / dy**2

    tolerance = 100 * torch.finfo(dtype).eps
    torch.testing.assert_close(
        result,
        expected,
        atol=tolerance,
        rtol=tolerance,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_curl_of_gradient_is_zero(dtype):
    bc = (Periodic(), Periodic())
    spacing = (0.5, 0.25)
    grid = torch.randn(2, 4, 5, dtype=dtype)

    facex, facey = mac_gradient(grid, spacing, bc)
    result = mac_curl(facex, facey, spacing, bc)

    tolerance = 100 * torch.finfo(dtype).eps
    torch.testing.assert_close(
        result,
        torch.zeros_like(result),
        atol=tolerance,
        rtol=tolerance,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_divergence_of_vel_from_streamfunction_is_zero(dtype):
    bc = (Periodic(), Periodic())
    spacing = (0.5, 0.25)

    streamfunction = torch.randn(2, 4, 5, dtype=dtype)
    facex, facey = mac_vel_from_streamfunction(streamfunction, spacing, bc)
    result = mac_divergence(facex, facey, spacing, bc)

    tolerance = 100 * torch.finfo(dtype).eps
    torch.testing.assert_close(
        result,
        torch.zeros_like(result),
        atol=tolerance,
        rtol=tolerance,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_neumann_gradient_has_zero_normal_wall_faces(dtype):
    bc = (Neumann(), Neumann())
    spacing = (0.5, 0.25)
    grid = torch.randn(2, 4, 5, dtype=dtype)

    facex, facey = mac_gradient(grid, spacing, bc)

    assert torch.equal(facex[..., :, 0], torch.zeros_like(facex[..., :, 0]))
    assert torch.equal(facex[..., :, -1], torch.zeros_like(facex[..., :, -1]))
    assert torch.equal(facey[..., 0, :], torch.zeros_like(facey[..., 0, :]))
    assert torch.equal(facey[..., -1, :], torch.zeros_like(facey[..., -1, :]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mixed_dirichlet_gradient_uses_ghost_faces(dtype):
    spacing = (0.5, 0.25)
    dy, dx = spacing
    bc = ((Neumann(), Dirichlet()), (Dirichlet(), Neumann()))
    grid = torch.full((2, 4, 5), 3.0, dtype=dtype)

    facex, facey = mac_gradient(grid, spacing, bc)

    tolerance = 100 * torch.finfo(dtype).eps
    torch.testing.assert_close(
        facex[..., :, 0],
        torch.full_like(facex[..., :, 0], 2 * 3.0 / dx),
        atol=tolerance,
        rtol=tolerance,
    )
    assert torch.equal(facex[..., :, -1], torch.zeros_like(facex[..., :, -1]))
    assert torch.equal(facey[..., 0, :], torch.zeros_like(facey[..., 0, :]))
    torch.testing.assert_close(
        facey[..., -1, :],
        torch.full_like(facey[..., -1, :], -2 * 3.0 / dy),
        atol=tolerance,
        rtol=tolerance,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_periodic_divergence_uses_neighboring_wraps(dtype):
    spacing = (0.5, 0.25)
    dy, dx = spacing
    bc = (Periodic(), Periodic())
    facex = torch.randn(2, 4, 5, dtype=dtype)
    facey = torch.randn(2, 4, 5, dtype=dtype)

    result = mac_divergence(facex, facey, spacing, bc)
    expected = (facex.roll(-1, -1) - facex) / dx + (facey.roll(-1, -2) - facey) / dy

    tolerance = 100 * torch.finfo(dtype).eps
    torch.testing.assert_close(
        result,
        expected,
        atol=tolerance,
        rtol=tolerance,
    )


# Boundary layouts covering every axis type, mixed axes, mixed sides and
# nonzero boundary values. Entries are (y_bc, x_bc).
BOUNDARIES = {
    "periodic": (Periodic(), Periodic()),
    "dirichlet": (Dirichlet(), Dirichlet()),
    "neumann": (Neumann(), Neumann()),
    "periodic_x": (Neumann(), Periodic()),
    "periodic_y": (Periodic(), Dirichlet()),
    "mixed_values": (
        (Neumann(0.5), Dirichlet(1.0)),
        (Dirichlet(-2.0), Neumann(0.25)),
    ),
}
DTYPES = [torch.float32, torch.float64]
SPACING = (0.3, 0.2)


def _periodic(bc):
    return tuple(not isinstance(b, tuple) and b.kind == "periodic" for b in bc)


def _node_shape(bc, ny, nx):
    py, px = _periodic(bc)
    return (ny if py else ny + 1, nx if px else nx + 1)


def _random(shape, dtype, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, dtype=torch.float64, generator=generator).to(dtype)


def _assert_zero(result, scale, dtype):
    tolerance = 100 * torch.finfo(dtype).eps * scale
    error = result.abs().max().item() if result.numel() else 0.0
    assert error <= tolerance, f"max |result| {error:.2e} > {tolerance:.2e}"


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("name", BOUNDARIES)
def test_curl_of_gradient_is_zero_for_all_boundaries(name, dtype):
    bc = BOUNDARIES[name]
    phi = _random((2, 6, 7), dtype)

    facex, facey = mac_gradient(phi, SPACING, bc)
    result = mac_curl(facex, facey, SPACING, bc)

    py, px = _periodic(bc)
    assert result.shape == (2, 6 if py else 5, 7 if px else 6)
    _assert_zero(result, 20 / min(SPACING) ** 2, dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("name", BOUNDARIES)
def test_divergence_of_vel_from_streamfunction_is_zero_for_all_boundaries(name, dtype):
    bc = BOUNDARIES[name]
    psi = _random((2, *_node_shape(bc, 6, 7)), dtype)

    facex, facey = mac_vel_from_streamfunction(psi, SPACING, bc)
    result = mac_divergence(facex, facey, SPACING, bc)

    assert result.shape == (2, 6, 7)
    _assert_zero(result, 20 / min(SPACING) ** 2, dtype)


@pytest.mark.parametrize("name", ["dirichlet", "periodic_x", "periodic_y"])
def test_streamfunction_constant_on_walls_has_no_wall_flux(name):
    bc = BOUNDARIES[name]
    py, px = _periodic(bc)
    psi = _random(_node_shape(bc, 6, 7), torch.float64)
    if not py:
        psi[0] = 1.5
        psi[-1] = -0.5
    if not px:
        psi[:, 0] = 1.5
        psi[:, -1] = 1.5 if py else -0.5
    if not py and not px:
        # One connected wall: psi takes a single value on all of it.
        psi[0] = psi[-1] = psi[:, 0] = psi[:, -1] = 1.5

    facex, facey = mac_vel_from_streamfunction(psi, SPACING, bc)

    if not px:
        assert torch.equal(facex[:, 0], torch.zeros_like(facex[:, 0]))
        assert torch.equal(facex[:, -1], torch.zeros_like(facex[:, -1]))
    if not py:
        assert torch.equal(facey[0], torch.zeros_like(facey[0]))
        assert torch.equal(facey[-1], torch.zeros_like(facey[-1]))


@pytest.mark.parametrize("name", BOUNDARIES)
def test_curl_of_vel_from_streamfunction_is_node_laplacian(name):
    # curl(curl psi) = -Δpsi with the 5-point node stencil, on interior nodes.
    bc = BOUNDARIES[name]
    py, px = _periodic(bc)
    dy, dx = SPACING
    psi = _random((3, *_node_shape(bc, 5, 6)), torch.float64)

    facex, facey = mac_vel_from_streamfunction(psi, SPACING, bc)
    result = mac_curl(facex, facey, SPACING, bc)

    padded = psi
    if py:
        padded = torch.cat([psi[..., -1:, :], psi, psi[..., :1, :]], dim=-2)
    if px:
        padded = torch.cat([padded[..., -1:], padded, padded[..., :1]], dim=-1)
    center = padded[..., 1:-1, 1:-1]
    laplacian = (
        padded[..., 2:, 1:-1] - 2 * center + padded[..., :-2, 1:-1]
    ) / dy**2 + (padded[..., 1:-1, 2:] - 2 * center + padded[..., 1:-1, :-2]) / dx**2
    torch.testing.assert_close(result, -laplacian, rtol=1e-12, atol=1e-12)


SOLVER_BOUNDARIES = {
    "dirichlet_values": (Dirichlet(1.5), Dirichlet(-0.5)),
    "neumann_dirichlet": (
        (Neumann(0.75), Dirichlet(2.0)),
        (Dirichlet(-1.0), Neumann(-0.5)),
    ),
    "periodic_neumann": (Periodic(), (Dirichlet(0.5), Neumann(1.25))),
}


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("name", SOLVER_BOUNDARIES)
def test_divergence_of_gradient_inverts_solver(name, dtype):
    # gradient must use the boundary values the solver lifted into f.
    bc = SOLVER_BOUNDARIES[name]
    solver = UniformGridSolver((6, 7), (1.8, 1.4), bc)
    f = _random((2, 6, 7), dtype)

    phi = solver(f)
    facex, facey = mac_gradient(phi, solver.spacings, bc)
    result = mac_divergence(facex, facey, solver.spacings, bc)

    # Round-off from the solve, amplified by the 1/h^2 stencil: up to a few
    # hundred eps on these small grids, depending on the transform backend.
    scale = 1e3
    tolerance = scale * torch.finfo(dtype).eps * f.abs().max().item()
    torch.testing.assert_close(result, f, rtol=0, atol=tolerance)


def test_gradient_uses_tensor_boundary_values():
    dy, dx = SPACING
    g_low = torch.linspace(0.0, 1.0, 4, dtype=torch.float64)  # along y
    q_high = torch.linspace(-1.0, 2.0, 5, dtype=torch.float64)  # along x
    bc = ((Neumann(), Neumann(q_high)), (Dirichlet(g_low), Neumann()))
    phi = _random((4, 5), torch.float64)

    facex, facey = mac_gradient(phi, SPACING, bc)

    torch.testing.assert_close(facex[:, 0], 2 * (phi[:, 0] - g_low) / dx)
    torch.testing.assert_close(facey[-1], q_high)
    assert torch.equal(facey[0], torch.zeros_like(facey[0]))
    assert torch.equal(facex[:, -1], torch.zeros_like(facex[:, -1]))


def _convergence_rates(errors):
    return [math.log2(a / b) for a, b in zip(errors, errors[1:])]


@pytest.mark.parametrize("name", ["dirichlet", "periodic_x", "periodic"])
def test_vel_from_streamfunction_and_curl_are_second_order(name):
    # Analytic psi on nodes against analytic face velocities and node vorticity.
    # A half-cell placement error would show up as first-order convergence.
    bc = BOUNDARIES[name]
    py, px = _periodic(bc)
    ly, lx = 1.0, 2.0
    ky, kx = 2 * math.pi / ly, 2 * math.pi / lx

    def psi(y, x):
        return torch.sin(ky * y + 0.3) * torch.cos(kx * x - 0.7)

    def velocity(y, x):  # (∂psi/∂y, -∂psi/∂x)
        return (
            ky * torch.cos(ky * y + 0.3) * torch.cos(kx * x - 0.7),
            kx * torch.sin(ky * y + 0.3) * torch.sin(kx * x - 0.7),
        )

    vel_errors, curl_errors = [], []
    for n in (16, 32, 64):
        ny, nx = n, 2 * n
        dy, dx = ly / ny, lx / nx
        y_nodes = torch.arange(ny + (not py), dtype=torch.float64) * dy
        x_nodes = torch.arange(nx + (not px), dtype=torch.float64) * dx
        y_cells = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
        x_cells = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx

        nodes = psi(y_nodes[:, None], x_nodes[None, :])
        facex, facey = mac_vel_from_streamfunction(nodes, (dy, dx), bc)
        exact_x = velocity(y_cells[:, None], x_nodes[None, :])[0]
        exact_y = velocity(y_nodes[:, None], x_cells[None, :])[1]
        vel_errors.append(
            max((facex - exact_x).abs().max(), (facey - exact_y).abs().max()).item()
        )

        exact_facex = velocity(y_cells[:, None], x_nodes[None, :])[0]
        exact_facey = velocity(y_nodes[:, None], x_cells[None, :])[1]
        vorticity = mac_curl(exact_facex, exact_facey, (dy, dx), bc)
        y_inner = y_nodes if py else y_nodes[1:-1]
        x_inner = x_nodes if px else x_nodes[1:-1]
        exact_vorticity = (ky**2 + kx**2) * psi(y_inner[:, None], x_inner[None, :])
        curl_errors.append((vorticity - exact_vorticity).abs().max().item())

    assert min(_convergence_rates(vel_errors)) > 1.9, vel_errors
    assert min(_convergence_rates(curl_errors)) > 1.9, curl_errors
    assert vel_errors[-1] / (ky + kx) < 1e-3
    assert curl_errors[-1] / (ky**2 + kx**2) < 1e-3


@pytest.mark.parametrize("name", ["dirichlet", "neumann", "periodic_y"])
def test_single_cell_wall_axis(name):
    bc = BOUNDARIES[name]
    py, px = _periodic(bc)
    psi = _random(_node_shape(bc, 4, 1), torch.float64)

    facex, facey = mac_vel_from_streamfunction(psi, SPACING, bc)
    assert facex.shape == (4, 2)
    assert facey.shape == (4 if py else 5, 1)
    assert mac_curl(facex, facey, SPACING, bc).shape == (4 if py else 3, 0)


def test_curl_rejects_mismatched_periodic_faces():
    bc = (Periodic(), Periodic())
    with pytest.raises(ValueError, match="spatial shape"):
        mac_curl(torch.randn(1, 4), torch.randn(2, 4), 1.0, bc)
    with pytest.raises(ValueError, match="facex"):
        mac_curl(torch.randn(4, 5), torch.randn(5, 4), 1.0, bc)
    with pytest.raises(ValueError, match="batch"):
        mac_curl(torch.randn(2, 4, 4), torch.randn(3, 4, 4), 1.0, bc)


def test_vel_from_streamfunction_rejects_too_few_wall_nodes():
    with pytest.raises(ValueError, match="at least 2 nodes"):
        mac_vel_from_streamfunction(torch.randn(1, 4), 1.0, (Dirichlet(), Dirichlet()))


@pytest.mark.parametrize("name", BOUNDARIES)
def test_operators_gradcheck(name):
    bc = BOUNDARIES[name]
    py, px = _periodic(bc)
    ny, nx = 3, 4
    phi = _random((2, ny, nx), torch.float64).requires_grad_()
    psi = _random((2, *_node_shape(bc, ny, nx)), torch.float64).requires_grad_()
    facex = _random((2, ny, nx if px else nx + 1), torch.float64).requires_grad_()
    facey = _random((2, ny if py else ny + 1, nx), torch.float64).requires_grad_()

    assert torch.autograd.gradcheck(lambda p: mac_gradient(p, SPACING, bc), (phi,))
    assert torch.autograd.gradcheck(
        lambda p: mac_vel_from_streamfunction(p, SPACING, bc), (psi,)
    )
    assert torch.autograd.gradcheck(
        lambda a, b: mac_divergence(a, b, SPACING, bc), (facex, facey)
    )
    assert torch.autograd.gradcheck(
        lambda a, b: mac_curl(a, b, SPACING, bc), (facex, facey)
    )
