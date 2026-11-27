"""Use orthel to compute the Helmholtz decomposition of a velocity field on a MAC grid
stored in a PyTorch tensor.
This example uses a velocity field from BubbleML, a dataset of two-phase boiling
simulations. The divergence is non-zero near the liquid-vapor interface so the
Helmholtz decompisition has an "interesting" dilational part.

Run from the project root with

    uv run --group examples python examples/helmholtz.py

Use ``--time-index 1`` to select a different time record, or ``--save`` to
write the plot to a file instead of showing it.
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from orthel.boundary import Dirichlet, Neumann
from orthel.helmholtz import helmholtz
from orthel.mac_operators import mac_divergence, mac_gradient

DEFAULT_DATA = Path("data/bubbleml-case-twall-70.hdf5")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--time-index",
        type=int,
        default=None,
        help="index of the time record to use (defaults to the middle record)",
    )
    parser.add_argument(
        "--save", type=Path, help="write the figure instead of showing it"
    )
    args = parser.parse_args()

    with h5py.File(args.data, "r") as data:
        times = data["time"][()]
        index = args.time_index if args.time_index is not None else len(times) // 2
        if not 0 <= index < len(times):
            raise IndexError(
                f"time index {args.time_index} out of range "
                f"[{-len(times)}, {len(times) - 1}]"
            )
        time = float(times[index])
        ux = torch.from_numpy(data["velfacex"][index].copy()).double()
        uy = torch.from_numpy(data["velfacey"][index].copy()).double()
        x_faces = data["x_faces"][()].copy()
        y_faces = data["y_faces"][()].copy()

    shape = (uy.shape[0] - 1, ux.shape[1] - 1)
    lengths = (
        float(y_faces[-1] - y_faces[0]),
        float(x_faces[-1] - x_faces[0]),
    )
    # pool boiling has walls on left, right, and bottom with an outflow on
    # the top. So the pressure solve "flips" the boundary type
    boundaries = (
        (Neumann(), Dirichlet()),  # y-axis
        (Neumann(), Neumann()),  # x-axis
    )

    psi, pressure = helmholtz(
        ux,
        uy,
        lengths=lengths,
        bc=boundaries,
    )

    spacings = (
        lengths[0] / shape[0],
        lengths[1] / shape[1],
    )
    gx, gy = mac_gradient(pressure, spacings, boundaries)
    projected_ux = ux - gx
    projected_uy = uy - gy

    dilational_ux = gx
    dilational_uy = gy
    solenoidal_ux = ux - gx
    solenoidal_uy = uy - gy

    original_divergence = mac_divergence(ux, uy, spacings, boundaries)
    projected_divergence = mac_divergence(
        projected_ux,
        projected_uy,
        spacings,
        boundaries,
    )

    solenoidal_ux_center = 0.5 * (
        solenoidal_ux[..., :, :-1] + solenoidal_ux[..., :, 1:]
    )
    solenoidal_uy_center = 0.5 * (
        solenoidal_uy[..., :-1, :] + solenoidal_uy[..., 1:, :]
    )
    dilational_ux_center = 0.5 * (
        dilational_ux[..., :, :-1] + dilational_ux[..., :, 1:]
    )
    dilational_uy_center = 0.5 * (
        dilational_uy[..., :-1, :] + dilational_uy[..., 1:, :]
    )

    solenoidal_magnitude = torch.sqrt(solenoidal_ux_center**2 + solenoidal_uy_center**2)
    dilational_magnitude = torch.sqrt(dilational_ux_center**2 + dilational_uy_center**2)

    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
    y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])
    x_grid, y_grid = np.meshgrid(x_centers, y_centers)

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(20, 7),
        constrained_layout=True,
    )

    extent = (x_centers[0], x_centers[-1], y_centers[0], y_centers[-1])
    psi_limit = float(psi.abs().max())
    phi_limit = float(pressure.abs().max())

    psi_image = axes[0].imshow(
        psi.numpy(),
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        # psi lives on cell corners, which span the face coordinates.
        extent=(x_faces[0], x_faces[-1], y_faces[0], y_faces[-1]),
        vmin=-psi_limit,
        vmax=psi_limit,
    )
    axes[0].set_title(f"Streamfunction psi, t={time:.1f}")
    fig.colorbar(psi_image, ax=axes[0], label="psi")

    phi_image = axes[1].imshow(
        pressure.numpy(),
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        extent=extent,
        vmin=-phi_limit,
        vmax=phi_limit,
    )
    axes[1].set_title("Potential phi")
    fig.colorbar(phi_image, ax=axes[1], label="phi")

    solenoidal_image = axes[2].imshow(
        solenoidal_magnitude.numpy(),
        origin="lower",
        aspect="auto",
        cmap="turbo",
        extent=extent,
        vmax=float(solenoidal_magnitude.max()),
    )
    axes[2].set_title("Solenoidal velocity")
    fig.colorbar(solenoidal_image, ax=axes[2], label="magnitude")

    dilational_image = axes[3].imshow(
        dilational_magnitude.numpy(),
        origin="lower",
        aspect="auto",
        cmap="turbo",
        extent=extent,
        vmax=float(dilational_magnitude.max()),
    )
    axes[3].set_title("Dilational velocity")
    fig.colorbar(dilational_image, ax=axes[3], label="magnitude")

    divergence_abs = original_divergence.abs()
    divergence_values = divergence_abs.numpy()
    divergence_floor = 1e-13
    divergence_ceiling = float(divergence_abs.max())
    divergence_values = np.where(
        divergence_values < divergence_floor,
        0.0,
        divergence_values,
    )
    divergence_cmap = plt.colormaps["magma"].copy()
    divergence_cmap.set_under("black")
    divergence_cmap.set_bad("black")
    divergence_image = axes[4].imshow(
        divergence_values,
        origin="lower",
        aspect="auto",
        cmap=divergence_cmap,
        extent=extent,
        norm=LogNorm(vmin=divergence_floor, vmax=divergence_ceiling),
    )
    axes[4].set_title("Original absolute divergence")
    fig.colorbar(divergence_image, ax=axes[4], label="absolute divergence")

    for axis in axes.flat:
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_aspect("equal")
        axis.set_xlim(extent[:2])
        axis.set_ylim(extent[2:])

    fig.suptitle(
        "BubbleML velocity Helmholtz/Leray projection\n"
        "Neumann x and bottom; Dirichlet pressure at top"
    )

    print(f"snapshot time: {time:.1f}")
    print(
        "maximum divergence:",
        f"{float(original_divergence.abs().max()):.3e}",
        "->",
        f"{float(projected_divergence.abs().max()):.3e}",
    )
    print(
        "pressure range:",
        f"{float(pressure.min()):.3e}",
        "to",
        f"{float(pressure.max()):.3e}",
    )

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=180)
        print(f"saved: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
