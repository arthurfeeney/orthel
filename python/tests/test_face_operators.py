import pytest
import torch
from orthel.boundary import Neumann, Periodic
from orthel.operators import divergence, gradient


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_wall_gradient_and_divergence(dtype):
    grid = torch.arange(12, dtype=dtype).reshape(3, 4).requires_grad_()
    spacing = (0.5, 0.25)
    bc = (Neumann(), Neumann())

    facex, facey = gradient(grid, spacing, bc)

    assert facex.shape == (3, 5)
    assert facey.shape == (4, 4)
    assert torch.all(facex[:, 0] == 0)
    assert torch.all(facex[:, -1] == 0)
    assert torch.all(facey[0] == 0)
    assert torch.all(facey[-1] == 0)

    result = divergence(facex, facey, spacing, bc)
    assert result.shape == (3, 4)
    assert result.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_periodic_gradient_and_divergence(dtype):
    grid = torch.arange(12, dtype=dtype).reshape(3, 4)
    facex, facey = gradient(
        grid,
        (0.5, 0.25),
        (Periodic(), Periodic()),
    )

    assert facex.shape == (3, 4)
    assert facey.shape == (3, 4)

    result = divergence(
        facex,
        facey,
        (0.5, 0.25),
        (Periodic(), Periodic()),
    )
    assert result.shape == (3, 4)


def test_gradient_supports_batches():
    grid = torch.randn(2, 3, 4, dtype=torch.float64)
    facex, facey = gradient(grid, (0.5, 0.25), (Neumann(), Neumann()))

    assert facex.shape == (2, 3, 5)
    assert facey.shape == (2, 4, 4)
    assert divergence(facex, facey, (0.5, 0.25)).shape == (2, 3, 4)
