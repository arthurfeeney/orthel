import pytest
import torch
from orthel.boundary import Dirichlet, Periodic
from orthel.helmholtz import helmholtz
from orthel.operators import gradient


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_periodic_gradient_pure_potential(dtype):
    bc = (Periodic(), Periodic())
    phi = torch.randn(3, 4, 5, dtype=dtype)
    facex, facey = gradient(phi, (0.5, 0.25), bc)

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
    assert recovered_phi.abs().mean() >= 0


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

    assert psi.shape == phi.shape
    assert recovered_phi.shape == phi.shape
