import math

import pytest
import torch
from orthel.boundary import Dirichlet, Neumann, Periodic
from orthel.helmholtz import helmholtz
from orthel.operators import divergence, gradient, vel_from_streamfunction


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_periodic_gradient_pure_potential(dtype):
    bc = (Periodic(), Periodic())
    phi = torch.randn(3, 4, 5, dtype=dtype)
    facex, facey = gradient(phi, (0.375, 0.25), bc)

    psi, recovered_phi = helmholtz(
        facex,
        facey,
        lengths=(1.5, 1.25),
        bc=bc,
    )

    assert psi.shape == phi.shape
    assert recovered_phi.shape == phi.shape
    assert psi.dtype == dtype
    assert recovered_phi.dtype == dtype
    tolerance = 100 * torch.finfo(dtype).eps
    assert torch.allclose(
        psi,
        torch.zeros_like(psi),
        atol=tolerance,
        rtol=tolerance,
    )
    phi_mean = phi - phi.mean(dim=(-2, -1), keepdim=True)
    assert torch.allclose(
        recovered_phi,
        phi_mean,
        atol=100 * tolerance,
        rtol=100 * tolerance,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_wall_boundary_shapes(dtype):
    bc = ((Dirichlet(), Dirichlet()), (Dirichlet(), Dirichlet()))
    phi = torch.randn(3, 4, 5, dtype=dtype)
    facex, facey = gradient(phi, (0.5, 0.25), bc)

    psi, recovered_phi = helmholtz(
        facex,
        facey,
        lengths=(2.0, 1.25),
        bc=bc,
    )

    assert psi.shape == (3, 5, 6)
    assert recovered_phi.shape == phi.shape


BOUNDARIES = {
    "periodic": (Periodic(), Periodic()),
    "neumann": (Neumann(), Neumann()),
    "dirichlet_values": (Dirichlet(1.0), Dirichlet(-0.5)),
    "channel": (Neumann(), Periodic()),
    "mixed": ((Neumann(), Dirichlet(0.5)), (Dirichlet(1.0), Neumann())),
}
LENGTHS = (1.8, 1.4)


def _periodic(bc):
    return tuple(not isinstance(b, tuple) and b.kind == "periodic" for b in bc)


def _random(shape, dtype, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, dtype=torch.float64, generator=generator).to(dtype)


def _random_streamfunction(bc, ny, nx, dtype, seed=1):
    """Random node psi with psi = 0 on walls (zero mean if fully periodic)."""
    py, px = _periodic(bc)
    psi = _random((2, ny + (not py), nx + (not px)), dtype, seed)
    if not py:
        psi[..., 0, :] = 0
        psi[..., -1, :] = 0
    if not px:
        psi[..., :, 0] = 0
        psi[..., :, -1] = 0
    if py and px:
        psi = psi - psi.mean(dim=(-2, -1), keepdim=True)
    return psi


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
@pytest.mark.parametrize("name", BOUNDARIES)
def test_recovers_random_streamfunction_and_potential(name, dtype):
    # u = curl(psi) + grad(phi) has no harmonic part, so both are recovered
    # exactly (phi up to a constant when it has no Dirichlet side).
    bc = BOUNDARIES[name]
    ny, nx = 6, 7
    spacings = (LENGTHS[0] / ny, LENGTHS[1] / nx)
    psi = _random_streamfunction(bc, ny, nx, dtype)
    phi = _random((2, ny, nx), dtype)
    sx, sy = vel_from_streamfunction(psi, spacings, bc)
    gx, gy = gradient(phi, spacings, bc)

    recovered_psi, recovered_phi = helmholtz(sx + gx, sy + gy, lengths=LENGTHS, bc=bc)

    tolerance = 1e3 * torch.finfo(dtype).eps
    assert recovered_psi.shape == psi.shape
    torch.testing.assert_close(recovered_psi, psi, rtol=0, atol=tolerance)
    offset = recovered_phi - phi
    if name in ("periodic", "neumann", "channel"):
        offset = offset - offset.mean(dim=(-2, -1), keepdim=True)
    torch.testing.assert_close(offset, torch.zeros_like(offset), rtol=0, atol=tolerance)


@pytest.mark.parametrize("name", ["dirichlet_values", "mixed"])
def test_remainder_is_divergence_free_with_boundary_values(name):
    # Backward-error style check on a random field with no analytic answer.
    bc = BOUNDARIES[name]
    ny, nx = 32, 24
    spacings = (LENGTHS[0] / ny, LENGTHS[1] / nx)
    velx = _random((ny, nx + 1), torch.float64, 2)
    vely = _random((ny + 1, nx), torch.float64, 3)

    _, phi = helmholtz(velx, vely, lengths=LENGTHS, bc=bc)
    gx, gy = gradient(phi, spacings, bc)
    residual = divergence(velx - gx, vely - gy, spacings, bc)
    scale = divergence(velx, vely, spacings, bc).abs().max()

    assert residual.abs().max() <= 1e-12 * scale


@pytest.mark.parametrize("name", ["walls", "channel", "periodic"])
def test_analytic_decomposition_is_second_order(name):
    # psi vanishes on walls and phi has zero normal derivative there, so the
    # sampled u = curl(psi) + grad(phi) has no harmonic part.
    ly, lx = LENGTHS
    if name == "walls":
        bc = (Neumann(), Neumann())
        psi_k = (math.pi / ly, math.pi / lx)
        phi_k = (math.pi / ly, 2 * math.pi / lx)
    elif name == "channel":
        bc = (Neumann(), Periodic())
        psi_k = (math.pi / ly, 2 * math.pi / lx)
        phi_k = (2 * math.pi / ly, 2 * math.pi / lx)
    else:
        bc = (Periodic(), Periodic())
        psi_k = (2 * math.pi / ly, 2 * math.pi / lx)
        phi_k = (2 * math.pi / ly, 4 * math.pi / lx)
    py, px = _periodic(bc)
    (a, b), (c, d) = psi_k, phi_k

    def psi(y, x):
        return torch.sin(a * y) * torch.sin(b * x + (0.4 if px else 0.0))

    def phi(y, x):
        return torch.cos(c * y) * torch.cos(d * x)

    def velocity(y, x):
        shift = 0.4 if px else 0.0
        ux = a * torch.cos(a * y) * torch.sin(b * x + shift) - d * torch.cos(
            c * y
        ) * torch.sin(d * x)
        uy = -b * torch.sin(a * y) * torch.cos(b * x + shift) - c * torch.sin(
            c * y
        ) * torch.cos(d * x)
        return ux, uy

    psi_errors, phi_errors = [], []
    for n in (16, 32, 64):
        ny, nx = n, n
        dy, dx = ly / ny, lx / nx
        y_nodes = torch.arange(ny + (not py), dtype=torch.float64) * dy
        x_nodes = torch.arange(nx + (not px), dtype=torch.float64) * dx
        y_cells = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
        x_cells = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
        velx = velocity(y_cells[:, None], x_nodes[None, :])[0]
        vely = velocity(y_nodes[:, None], x_cells[None, :])[1]

        recovered_psi, recovered_phi = helmholtz(velx, vely, lengths=LENGTHS, bc=bc)

        exact_psi = psi(y_nodes[:, None], x_nodes[None, :])
        if py and px:
            exact_psi = exact_psi - exact_psi.mean()
        exact_phi = phi(y_cells[:, None], x_cells[None, :])
        phi_offset = recovered_phi - exact_phi
        psi_errors.append((recovered_psi - exact_psi).abs().max().item())
        phi_errors.append((phi_offset - phi_offset.mean()).abs().max().item())

    rates = [
        [math.log2(e0 / e1) for e0, e1 in zip(errors, errors[1:])]
        for errors in (psi_errors, phi_errors)
    ]
    assert min(rates[0]) > 1.8, psi_errors
    assert min(rates[1]) > 1.8, phi_errors
    assert psi_errors[-1] < 1e-2 and phi_errors[-1] < 1e-2


@pytest.mark.parametrize("name", BOUNDARIES)
def test_helmholtz_gradcheck(name):
    bc = BOUNDARIES[name]
    py, px = _periodic(bc)
    ny, nx = 3, 4
    velx = _random((2, ny, nx if px else nx + 1), torch.float64, 4)
    vely = _random((2, ny if py else ny + 1, nx), torch.float64, 5)

    assert torch.autograd.gradcheck(
        lambda u, v: helmholtz(u, v, lengths=LENGTHS, bc=bc),
        (velx.requires_grad_(), vely.requires_grad_()),
    )
