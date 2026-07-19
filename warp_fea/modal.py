"""
M5 — modal analysis: lowest eigenfrequencies of K φ = λ M φ, on the GPU.

Fills the `eigenfrequencies` key the §1.2 contract reserves, so the Judge's
resonance check (Stage 3a `*FREQUENCY`) has a GPU path too.

Method: block inverse (subspace) iteration with Rayleigh-Ritz.
  Y <- K^-1 M X        (one Jacobi-preconditioned CG per column, on the GPU)
  Rayleigh-Ritz on the m-dimensional subspace -> new X
Converges to the m smallest eigenpairs. The only host work is the m x m dense
Ritz problem (m ~ 10), solved with numpy — this environment's scipy is broken and
a GPU FEA package should not depend on repairing a shared conda install.

Dirichlet handling: K and M are projected with the same projector, so a clamped
dof gets K=I, M=0 and can only satisfy K φ = λ M φ with φ = 0 — the constrained
modes drop out on their own instead of polluting the spectrum.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import warp as wp
import warp.fem as fem
import warp.sparse as wps
from warp.examples.fem import utils as fem_example_utils
from warp.fem.linalg import array_axpy

from bench.instrument import measure, MeasureResult

from .elasticity import elasticity_form, lame, mass_form
from .mesh_io import FEMesh, read_mesh
from .results import FEAResult, Measurement, SolverStatus
from .solver import (DEFAULT_DEVICE, _device_name, _dirichlet_projector,
                     _match_space_to_mesh)


@wp.kernel
def _axpy_vec3d(x: wp.array(dtype=wp.vec3d), y: wp.array(dtype=wp.vec3d),
                a: wp.float64):
    i = wp.tid()
    y[i] = y[i] + a * x[i]


@wp.kernel
def _scale_vec3d(x: wp.array(dtype=wp.vec3d), a: wp.float64):
    i = wp.tid()
    x[i] = a * x[i]


def _dot(a: wp.array, b: wp.array) -> float:
    return float(wp.utils.array_inner(a, b))


def solve_modal(
    mesh,
    load_case: dict,
    n_modes: int = 6,
    *,
    device: str = DEFAULT_DEVICE,
    tol: float = 1e-10,
    max_iters: int = 20000,
    degree: Optional[int] = None,
    subspace_iters: int = 60,
    eig_tol: float = 1e-9,
    verbose: bool = False,
) -> FEAResult:
    """
    Lowest `n_modes` natural frequencies [Hz]. Needs material.density.
    Loads in the load_case are ignored — only supports and material matter.
    """
    fe: FEMesh = mesh if isinstance(mesh, FEMesh) else read_mesh(mesh)
    mat = load_case.get("material") or {}
    rho = float(mat.get("density", 0.0))
    if rho <= 0:
        raise ValueError("modal analysis needs material.density [kg/m^3]")
    lam, mu = lame(float(mat["E"]), float(mat["nu"]))
    degree = degree if degree is not None else fe.order

    positions = wp.array(fe.vertex_points, dtype=wp.vec3d, device=device)
    tet_idx = wp.array(fe.corners_local.astype(np.int32), dtype=wp.int32, device=device)
    geo = fem.Tetmesh(tet_idx, positions)
    space = fem.make_polynomial_space(geo, degree=degree, dtype=wp.vec3d)
    n = space.node_count()

    domain = fem.Cells(geo)
    test = fem.make_test(space=space, domain=domain)
    trial = fem.make_trial(space=space, domain=domain)

    K = fem.integrate(elasticity_form, fields={"u": trial, "v": test},
                      values={"lam": wp.float64(lam), "mu": wp.float64(mu)},
                      output_dtype=wp.float64)
    M = fem.integrate(mass_form, fields={"u": trial, "v": test},
                      values={"rho": wp.float64(rho)}, output_dtype=wp.float64)

    g_of_s = _match_space_to_mesh(space.node_positions().numpy(), fe.points)
    s_of_g = {int(g): i for i, g in enumerate(g_of_s)}
    fixed: List[int] = []
    for sup in load_case.get("supports", []):
        fixed.extend(s_of_g[int(g)] for g in fe.region(sup["region"]).node_idx
                     if int(g) in s_of_g)
    fixed_nodes = np.asarray(sorted(set(fixed)), dtype=np.int64)
    P = _dirichlet_projector(fixed_nodes, n, device)

    # zero rhs: project_linear_system also normalises the projector for us
    dummy = wp.zeros(n, dtype=wp.vec3d, device=device)
    fem.project_linear_system(K, dummy, P, fixed_value=None, normalize_projector=True)
    fem.project_linear_system(M, wp.zeros(n, dtype=wp.vec3d, device=device), P,
                              fixed_value=None, normalize_projector=True)
    # project_linear_system leaves K=I on fixed dofs; do the same to M but with 0,
    # so fixed dofs contribute no mass and hence no finite eigenvalue.
    _zero_rows(M, fixed_nodes, device)

    m = max(n_modes + 4, n_modes * 2)      # guard vectors speed up convergence
    rng = np.random.default_rng(0)
    X = []
    for j in range(m):
        v = rng.standard_normal((n, 3))
        v[fixed_nodes] = 0.0
        X.append(wp.array(v, dtype=wp.vec3d, device=device))

    scratch = [wp.zeros(n, dtype=wp.vec3d, device=device) for _ in range(m)]
    theta = np.zeros(m)

    def matvec(A, x, out):
        wps.bsr_mv(A=A, x=x, y=out, alpha=1.0, beta=0.0)

    with measure(f"warp_modal:{os.path.basename(fe.source)}", interval=0.02,
                 verbose=verbose) as meas:
        prev = None
        converged = False
        for it in range(subspace_iters):
            # Y <- K^-1 (M X)
            Y = []
            for j in range(m):
                rhs = wp.zeros(n, dtype=wp.vec3d, device=device)
                matvec(M, X[j], rhs)
                y = wp.zeros(n, dtype=wp.vec3d, device=device)
                fem_example_utils.bsr_cg(K, b=rhs, x=y, tol=tol,
                                         max_iters=max_iters, quiet=True)
                Y.append(y)

            # Rayleigh-Ritz on span(Y)
            KY = [wp.zeros(n, dtype=wp.vec3d, device=device) for _ in range(m)]
            MY = [wp.zeros(n, dtype=wp.vec3d, device=device) for _ in range(m)]
            for j in range(m):
                matvec(K, Y[j], KY[j])
                matvec(M, Y[j], MY[j])
            Kr = np.empty((m, m)); Mr = np.empty((m, m))
            for i in range(m):
                for j in range(i, m):
                    Kr[i, j] = Kr[j, i] = _dot(Y[i], KY[j])
                    Mr[i, j] = Mr[j, i] = _dot(Y[i], MY[j])

            # small dense generalised problem  Kr q = theta Mr q  (Mr SPD)
            L = np.linalg.cholesky(Mr)
            Li = np.linalg.inv(L)
            A = Li @ Kr @ Li.T
            w, Q = np.linalg.eigh((A + A.T) / 2)
            q = Li.T @ Q
            theta = w

            # X <- Y q  (keep only what we need; all on device)
            for j in range(m):
                scratch[j].zero_()
                for i in range(m):
                    c = float(q[i, j])
                    if c != 0.0:
                        wp.launch(_axpy_vec3d, dim=n, inputs=[Y[i], scratch[j],
                                                              wp.float64(c)],
                                  device=device)
            for j in range(m):
                nrm = np.sqrt(_dot(scratch[j], scratch[j]))
                if nrm > 0:
                    wp.launch(_scale_vec3d, dim=n,
                              inputs=[scratch[j], wp.float64(1.0 / nrm)], device=device)
                wp.copy(X[j], scratch[j])

            lo = theta[:n_modes]
            if prev is not None:
                rel = np.max(np.abs(lo - prev) / np.maximum(np.abs(lo), 1e-30))
                if verbose:
                    print(f"    subspace it{it+1}: max rel change {rel:.3e}")
                if rel < eig_tol:
                    converged = True
                    break
            prev = lo.copy()
        wp.synchronize_device(device)
    res: MeasureResult = meas["result"]

    lam_eig = np.maximum(theta[:n_modes], 0.0)
    freqs = np.sqrt(lam_eig) / (2.0 * np.pi)      # rad/s -> Hz

    status = SolverStatus(
        regime="solid", backend="warp-gpu", meshed=True, converged=bool(converged),
        n_dof=3 * n, n_iters=it + 1, wall_time_s=res.wall_s, cpu_avg_pct=res.cpu_avg,
        gpu_util_avg_pct=res.gpu_util_avg, device=_device_name(device),
        element="C3D10" if degree == 2 else "C3D4", tol=eig_tol,
        message="" if converged else
                f"subspace iteration hit {subspace_iters} sweeps without reaching "
                f"eig_tol={eig_tol:g}",
    )
    out = FEAResult(solver_status=status)
    if not converged:
        return out.invalidate(status.message)
    # §1.2 reserves `eigenfrequencies` as a bare list — Hz, ascending.
    out.measured["eigenfrequencies"] = [float(f) for f in freqs]
    out.fields = {"modes": X[:n_modes]}
    return out


def _zero_rows(A: wps.BsrMatrix, rows: np.ndarray, device: str) -> None:
    """Zero whole block-rows of a BSR matrix (used to strip fixed dofs from M)."""
    if len(rows) == 0:
        return
    n = A.nrow
    keep = np.ones(n, dtype=np.float64)
    keep[rows] = 0.0
    D = wps.bsr_zeros(n, n, wp.mat33d, device=device)
    idx = wp.array(np.arange(n, dtype=np.int32), dtype=wp.int32, device=device)
    blocks = np.einsum("i,jk->ijk", keep, np.eye(3))
    wps.bsr_set_from_triplets(D, idx, idx, wp.array(blocks, dtype=wp.mat33d,
                                                    device=device))
    tmp = D @ A @ D
    wps.bsr_assign(dest=A, src=tmp)
