from __future__ import annotations

import math

import torch

_WALL_TYPES = ("neumann", "dirichlet")


def second_difference_eigenvalues(
    n: int,
    spacing: float,
    bc: tuple[str, str],
    *,
    half_spectrum: bool = False,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Eigenvalues of the 1D cell-centred 3-point second difference.

    The operator is ``(u_{j-1} - 2 u_j + u_{j+1}) / h²`` on ``n`` cells, with
    ghost cells set by the ``(low, high)`` boundary types: wrapped for
    ``"periodic"``, mirrored for ``"neumann"`` and anti-mirrored for
    ``"dirichlet"``. Its eigenvalues are ``-(2/h)² sin²(θ_m / 2)``.

    They are ordered to match the transform that diagonalises the operator:
    ``torch.fft.fft`` on periodic axes (``torch.fft.rfft`` if
    ``half_spectrum``), and the matching DCT/DST in
    :mod:`orthel.transforms.dct` on wall axes.
    """
    if bc == ("periodic", "periodic"):
        freq = torch.fft.rfftfreq if half_spectrum else torch.fft.fftfreq
        theta = 2 * math.pi * freq(n, dtype=dtype, device=device)
    elif all(side in _WALL_TYPES for side in bc):
        if half_spectrum:
            raise ValueError("half_spectrum only applies to periodic axes")
        # Each Dirichlet end shifts the modes by a quarter-wave, so mode m has
        # angle π (m + offset) / N with offset 0 (DCT-II), 1/2 (DCT-IV and
        # DST-IV) or 1 (DST-II).
        offset = 0.5 * bc.count("dirichlet")
        theta = math.pi * (torch.arange(n, dtype=dtype, device=device) + offset) / n
    else:
        raise ValueError(f"unsupported boundary types {bc!r}")
    return -(((2 / spacing) * torch.sin(theta / 2)) ** 2)
