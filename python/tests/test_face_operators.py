import pytest
import torch
from orthel.boundary import Periodic
from orthel.operators import curl, divergence, gradient, vel_from_streamfunction


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_divergence_of_gradient_is_periodic_laplacian(dtype):
    bc = (Periodic(), Periodic())
    spacing = (0.5, 0.25)
    dy, dx = spacing
    grid = torch.randn(2, 4, 5, dtype=dtype)

    # div(grad(grid)) == laplacian(grid)
    facex, facey = gradient(grid, spacing, bc)
    result = divergence(facex, facey, spacing, bc)

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

    facex, facey = gradient(grid, spacing, bc)
    result = curl(facex, facey, spacing, bc)

    tolerance = 100 * torch.finfo(dtype).eps
    torch.testing.assert_close(
        result,
        torch.zeros_like(result),
        atol=tolerance,
        rtol=tolerance,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_divergence_of_curl_is_zero(dtype):
    bc = (Periodic(), Periodic())
    spacing = (0.5, 0.25)
    dy, dx = spacing

    streamfunction = torch.randn(2, 4, 5, dtype=dtype)
    facex, facey = vel_from_streamfunction(streamfunction, spacing, bc)
    result = divergence(facex, facey, spacing, bc)

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
    facex, facey = vel_from_streamfunction(streamfunction, spacing, bc)
    result = divergence(facex, facey, spacing, bc)

    tolerance = 100 * torch.finfo(dtype).eps
    torch.testing.assert_close(
        result,
        torch.zeros_like(result),
        atol=tolerance,
        rtol=tolerance,
    )
