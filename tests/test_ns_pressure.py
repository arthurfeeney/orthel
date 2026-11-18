"""Navier-Stokes style pressure projection on a staggered (MAC) grid.

Pressure lives at cell centres and each velocity component on the faces normal
to its axis. With the face-to-cell divergence ``D`` and cell-to-face gradient
``G`` below, ``D G`` is the cell-centred Laplacian the solver inverts:

* periodic axes wrap, with ``n`` faces;
* a ``"neumann"`` side is a solid wall: its normal velocity is prescribed and
  the pressure gradient on that face is zero;
* a ``"dirichlet"`` side is an open boundary with ``p = 0`` on the face: its
  normal velocity is corrected with the gradient to the anti-mirrored ghost.

Projecting ``u*`` solves ``L p = D u*`` and sets ``u = u* - G p``, so
``D u = D u* - L p`` is zero up to the solve's residual.
"""

import math

import pytest
import torch

from torch_poisson.boundary import Dirichlet, Neumann, Periodic
from torch_poisson.eigenvalues import second_difference_eigenvalues
from torch_poisson.uniform_grid_solver import UniformGridSolver

CASES = [
    ((32, 24), (1.0, 1.5), (Periodic(), Periodic())),
    ((32, 24), (1.0, 1.5), (Neumann(), Neumann())),
    ((32, 24), (1.0, 1.5), (Periodic(), Neumann())),
    ((32, 24), (1.0, 1.5), ((Neumann(), Dirichlet()), Neumann())),
    ((32, 24), (1.0, 1.5), (Dirichlet(), Periodic())),
    ((12, 10, 8), (1.0, 1.5, 2.0), (Periodic(), Neumann(), Periodic())),
    ((12, 10, 8), (1.0, 1.5, 2.0), (Neumann(), Neumann(), Neumann())),
    ((12, 10, 8), (1.0, 1.5, 2.0), ((Neumann(), Dirichlet()), Neumann(), Neumann())),
]
# Without an open boundary the projection is orthogonal, so it is its own adjoint.
CLOSED_CASES = [
    case
    for case in CASES
    if not any(isinstance(s, Dirichlet) for b in case[2] for s in (b if isinstance(b, tuple) else (b,)))
]

DTYPES = [torch.float64, torch.float32]


def _case_id(case):
    shape, _, bc = case
    axes = ["/".join(s.kind for s in b) if isinstance(b, tuple) else b.kind for b in bc]
    return f"{len(shape)}d-" + "-".join(axes)


def face_shape(solver, axis):
    """Shape of the faces normal to ``axis``: ``n + 1`` along it on a wall axis."""
    shape = list(solver.shape)
    if solver.boundary_types[axis][0] != "periodic":
        shape[axis] += 1
    return tuple(shape)


def divergence(solver, faces):
    """Face-to-cell divergence ``Σ_i (u_i[k + 1] - u_i[k]) / h_i``."""
    out = 0
    for axis, (u, h) in enumerate(zip(faces, solver.spacings)):
        dim, n = u.ndim - len(solver.shape) + axis, solver.shape[axis]
        if solver.boundary_types[axis][0] == "periodic":
            out = out + (u.roll(-1, dim) - u) / h
        else:
            out = out + (u.narrow(dim, 1, n) - u.narrow(dim, 0, n)) / h
    return out


def gradient(solver, p):
    """Cell-to-face gradient with zero on wall faces and ``p = 0`` on open faces."""
    faces = []
    for axis, h in enumerate(solver.spacings):
        dim, n = p.ndim - len(solver.shape) + axis, solver.shape[axis]
        if solver.boundary_types[axis][0] == "periodic":
            faces.append((p - p.roll(1, dim)) / h)
            continue
        interior = (p.narrow(dim, 1, n - 1) - p.narrow(dim, 0, n - 1)) / h
        low, high = solver.boundary_types[axis]
        first, last = p.narrow(dim, 0, 1), p.narrow(dim, n - 1, 1)
        # The open face's ghost is -p, so the gradient across it is ±2p/h.
        low_face = 2 * first / h if low == "dirichlet" else torch.zeros_like(first)
        high_face = -2 * last / h if high == "dirichlet" else torch.zeros_like(last)
        faces.append(torch.cat([low_face, interior, high_face], dim=dim))
    return faces


def project(solver, faces):
    """Return the divergence-free part of ``faces`` and the pressure."""
    p = solver(divergence(solver, faces))
    return [u - g for u, g in zip(faces, gradient(solver, p))], p


def zero_wall_faces(solver, faces):
    """Copy of ``faces`` with the normal velocity on solid walls set to zero."""
    faces = [u.clone() for u in faces]
    for axis, bc in enumerate(solver.boundary_types):
        for side, kind in enumerate(bc):
            if kind == "neumann":
                faces[axis].narrow(axis, 0 if side == 0 else -1, 1).zero_()
    return faces


def random_velocity(solver, dtype, generator):
    """Random face velocities with zero normal velocity on solid walls."""
    faces = [
        torch.randn(face_shape(solver, axis), dtype=dtype, generator=generator)
        for axis in range(len(solver.shape))
    ]
    return zero_wall_faces(solver, faces)


def corner_difference(solver, psi, axis):
    """Difference of corner values ``(ψ[k + 1] - ψ[k]) / h`` along ``axis``, wrapping if periodic."""
    h, n = solver.spacings[axis], solver.shape[axis]
    if solver.boundary_types[axis][0] == "periodic":
        return (psi.roll(-1, axis) - psi) / h
    return (psi.narrow(axis, 1, n) - psi.narrow(axis, 0, n)) / h


def operator_norm(solver):
    """``‖L‖_∞`` of the cell-centred Laplacian."""
    return sum(4 / h**2 for h in solver.spacings)


def smallest_nonzero_eigenvalue(solver):
    total = torch.zeros((), dtype=torch.float64)
    ndim = len(solver.shape)
    for i, (n, h, bc) in enumerate(zip(solver.shape, solver.spacings, solver.boundary_types)):
        view = [1] * ndim
        view[i] = -1
        total = total + second_difference_eigenvalues(n, h, bc).reshape(view)
    magnitude = total.abs()
    return magnitude[magnitude > 0].min().item()


def velocity_tolerance(solver, dtype, scale):
    """Error bound for a projected velocity of size ``scale``.

    A residual ``r`` in ``L p = D u*`` moves the velocity by ``G L⁻¹ r``, and
    ``‖G L⁻¹‖ = 1 / sqrt(λ_min)`` because ``L = -Gᵀ G``. The residual is at
    most ``eps ‖L‖ ‖p‖ ~ eps sqrt(‖L‖) ‖u*‖``, so the velocity error is
    ``eps sqrt(‖L‖ / λ_min) ‖u*‖``.
    """
    ratio = operator_norm(solver) / smallest_nonzero_eigenvalue(solver)
    return 10 * torch.finfo(dtype).eps * math.sqrt(ratio) * scale


def max_abs(faces):
    return max(u.abs().max().item() for u in faces)


def max_diff(a, b):
    return max((x.double() - y.double()).abs().max().item() for x, y in zip(a, b))


def inner(a, b):
    return sum((x.double() * y.double()).sum() for x, y in zip(a, b))


def test_divergence_of_gradient_is_solver_laplacian():
    # D G must be the operator the solver inverts, so solving D G p = f recovers p.
    solver = UniformGridSolver((32, 24), (1.0, 1.5), ((Neumann(), Dirichlet()), Periodic()))
    p = torch.randn(32, 24, dtype=torch.float64, generator=torch.Generator().manual_seed(0))

    torch.testing.assert_close(solver(divergence(solver, gradient(solver, p))), p)


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_projection_is_divergence_free(case, dtype):
    shape, lengths, bc = case
    solver = UniformGridSolver(shape, lengths, bc)
    u_star = random_velocity(solver, dtype, torch.Generator().manual_seed(1))

    u, p = project(solver, u_star)

    # D u is the residual of L p = D u*, so it is bounded like a backward error.
    div_star = divergence(solver, u_star)
    scale = operator_norm(solver) * p.abs().max() + div_star.abs().max()
    tolerance = 10 * torch.finfo(dtype).eps * scale
    div = divergence(solver, u).abs().max()
    assert div <= tolerance, f"max divergence {div:.2e} exceeds {tolerance:.2e}"


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_projection_keeps_wall_normal_velocity(case, dtype):
    # The pressure gradient vanishes on solid walls, so no-penetration is kept.
    shape, lengths, bc = case
    solver = UniformGridSolver(shape, lengths, bc)
    u, _ = project(solver, random_velocity(solver, dtype, torch.Generator().manual_seed(2)))

    for axis, axis_bc in enumerate(solver.boundary_types):
        for side, kind in enumerate(axis_bc):
            if kind == "neumann":
                assert (u[axis].narrow(axis, 0 if side == 0 else -1, 1) == 0).all()


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_projection_is_idempotent(case, dtype):
    shape, lengths, bc = case
    solver = UniformGridSolver(shape, lengths, bc)
    u_star = random_velocity(solver, dtype, torch.Generator().manual_seed(3))
    u, _ = project(solver, u_star)

    u_again, p_again = project(solver, u)

    tolerance = velocity_tolerance(solver, dtype, max_abs(u_star))
    error = max_diff(u_again, u)
    assert error <= tolerance, f"reprojection changed u by {error:.2e} > {tolerance:.2e}"


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_gradient_field_is_removed(case, dtype):
    # A pure pressure gradient u* = G φ projects to zero, and p recovers φ.
    shape, lengths, bc = case
    solver = UniformGridSolver(shape, lengths, bc)
    phi = torch.randn(shape, dtype=dtype, generator=torch.Generator().manual_seed(4))
    u_star = gradient(solver, phi)

    u, _ = project(solver, u_star)

    tolerance = velocity_tolerance(solver, dtype, max_abs(u_star))
    error = max_abs(u)
    assert error <= tolerance, f"max velocity {error:.2e} exceeds {tolerance:.2e}"


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("case", [c for c in CASES if len(c[0]) == 2], ids=_case_id)
def test_streamfunction_velocity_is_unchanged(case, dtype):
    # u = (∂ψ/∂y, -∂ψ/∂x) with ψ at cell corners is discretely divergence-free,
    # so the projection must leave it alone. ψ = 0 on solid walls keeps them
    # impermeable.
    shape, lengths, bc = case
    solver = UniformGridSolver(shape, lengths, bc)
    corners = [face_shape(solver, axis)[axis] for axis in range(2)]
    psi = torch.randn(corners, dtype=dtype, generator=torch.Generator().manual_seed(5))
    for axis, axis_bc in enumerate(solver.boundary_types):
        for side, kind in enumerate(axis_bc):
            if kind == "neumann":
                psi.narrow(axis, 0 if side == 0 else -1, 1).zero_()
    # Differencing corners along one axis lands on the other component's faces.
    u_star = [corner_difference(solver, psi, 1), -corner_difference(solver, psi, 0)]
    assert divergence(solver, u_star).abs().max() <= 100 * torch.finfo(dtype).eps * max_abs(u_star)

    u, _ = project(solver, u_star)

    tolerance = velocity_tolerance(solver, dtype, max_abs(u_star))
    error = max_diff(u, u_star)
    assert error <= tolerance, f"projection changed u by {error:.2e} > {tolerance:.2e}"


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("case", CLOSED_CASES, ids=_case_id)
def test_projection_is_orthogonal(case, dtype):
    # With walls and periodic axes only, G = -Dᵀ, so u ⊥ G p and the kinetic
    # energy splits as ‖u*‖² = ‖u‖² + ‖G p‖².
    shape, lengths, bc = case
    solver = UniformGridSolver(shape, lengths, bc)
    u_star = random_velocity(solver, dtype, torch.Generator().manual_seed(6))

    u, p = project(solver, u_star)
    grad_p = gradient(solver, p)

    energy = inner(u_star, u_star)
    tolerance = 10 * torch.finfo(dtype).eps * energy * math.sqrt(sum(s for s in shape))
    assert inner(u, grad_p).abs() <= tolerance
    assert (inner(u, u) + inner(grad_p, grad_p) - energy).abs() <= tolerance


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("case", CLOSED_CASES, ids=_case_id)
def test_projection_gradient_is_projection(case, dtype):
    # The orthogonal projection is self-adjoint on the unknown (non-wall) faces,
    # so backpropagating g through it gives the projection of g there. Solid
    # wall velocities are boundary data: they pass through unchanged but still
    # drive the pressure, so their gradient is the sensitivity to inflow.
    shape, lengths, bc = case
    solver = UniformGridSolver(shape, lengths, bc)
    generator = torch.Generator().manual_seed(7)
    u_star = [u.requires_grad_() for u in random_velocity(solver, dtype, generator)]
    g = random_velocity(solver, dtype, generator)

    u, _ = project(solver, u_star)
    grads = zero_wall_faces(solver, torch.autograd.grad(u, u_star, g))

    expected, _ = project(solver, g)
    tolerance = velocity_tolerance(solver, dtype, max_abs(g))
    error = max_diff(grads, expected)
    assert error <= tolerance, f"gradient differs from P g by {error:.2e} > {tolerance:.2e}"


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_channel_inflow_outflow(dtype):
    # Flow enters through the low x wall and leaves through the high x wall with
    # the same profile; the walls keep their prescribed normal velocity and the
    # projected interior is divergence-free.
    shape, lengths = (32, 24), (2.0, 1.0)
    solver = UniformGridSolver(shape, lengths, (Neumann(), Neumann()))
    generator = torch.Generator().manual_seed(8)
    u_star = random_velocity(solver, dtype, generator)
    y = solver.coords(dtype=dtype)[1]
    profile = 4 * y * (lengths[1] - y)
    u_star[0][0] = profile
    u_star[0][-1] = profile

    u, p = project(solver, u_star)

    torch.testing.assert_close(u[0][0], profile, rtol=0, atol=0)
    torch.testing.assert_close(u[0][-1], profile, rtol=0, atol=0)
    div_star = divergence(solver, u_star)
    scale = operator_norm(solver) * p.abs().max() + div_star.abs().max()
    tolerance = 10 * torch.finfo(dtype).eps * scale
    assert divergence(solver, u).abs().max() <= tolerance
