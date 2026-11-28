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
    grid: str = "cell",
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Eigenvalues of the 1D 3-point second difference on ``n`` cells.

    The operator is ``(u_{j-1} - 2 u_j + u_{j+1}) / h²``. Its eigenvalues are
    ``-(2/h)² sin²(θ_m / 2)``.

    With ``grid="cell"`` the unknowns are the ``n`` cell centres and ghost
    cells are set by the ``(low, high)`` boundary types: wrapped for
    ``"periodic"``, mirrored for ``"neumann"`` and anti-mirrored for
    ``"dirichlet"``.

    With ``grid="node"`` the unknowns are the cell corners: ``n`` of them on a
    periodic axis, otherwise the ``n + 1`` corners minus each Dirichlet wall
    node (which is known). Neumann wall nodes use the mirrored ghost
    ``u_{-1} = u_1``.

    They are ordered to match the transform that diagonalises the operator:
    ``torch.fft.fft`` on periodic axes (``torch.fft.rfft`` if
    ``half_spectrum``), and the matching DCT/DST in
    :mod:`orthel.transforms.dtt` on wall axes. On nodes those are DST-I
    (Dirichlet/Dirichlet), DCT-I (Neumann/Neumann), DCT-III
    (Neumann/Dirichlet) and DST-III (Dirichlet/Neumann).
    """
    if grid not in ("cell", "node"):
        raise ValueError(f'grid must be "cell" or "node", got {grid!r}')
    if bc == ("periodic", "periodic"):
        freq = torch.fft.rfftfreq if half_spectrum else torch.fft.fftfreq
        theta = 2 * math.pi * freq(n, dtype=dtype, device=device)
    elif all(side in _WALL_TYPES for side in bc):
        if half_spectrum:
            raise ValueError("half_spectrum only applies to periodic axes")
        # Each Dirichlet end shifts the modes by a quarter-wave, so mode m has
        # angle π (m + offset) / N with offset 0 (DCT-II), 1/2 (DCT-IV and
        # DST-IV) or 1 (DST-II).
        # On nodes the same angles apply; only the number of unknowns differs.
        offset = 0.5 * bc.count("dirichlet")
        modes = n if grid == "cell" else n - 1 + bc.count("neumann")
        theta = math.pi * (torch.arange(modes, dtype=dtype, device=device) + offset) / n
    else:
        raise ValueError(f"unsupported boundary types {bc!r}")
    return -(((2 / spacing) * torch.sin(theta / 2)) ** 2)
