"""
Linear eigenvalue (Euler) buckling on the GPU — fills the contract's
`buckling_load_factor` key (PHASE3_SPEC §0, *BUCKLE).

Workflow:
  1. linear-static solve under the reference load        -> u0
  2. assemble the geometric stiffness  K_g  from σ(u0)
  3. solve the buckling pencil  K φ = λ (−K_g) φ          -> load factors λ

The smallest positive λ is the buckling load factor (BLF): the reference load
multiplied by BLF is the critical (buckling) load. Validated against ccx *BUCKLE.

Eigensolver: block inverse (subspace) iteration, same as modal.py, but the
Rayleigh-Ritz reduction factorises the SPD *stiffness* Kr (not the geometric
matrix, which is indefinite):
    Kr q = θ Br q ,  Br = Y^T(−K_g)Y      (Br symmetric, possibly indefinite)
  → C = Lr^{-1} Br Lr^{-T} ,  Kr = Lr Lr^T
  → eig(C) = 1/θ ;  θ = λ are the load factors.
Only the m×m Ritz problem touches the host (numpy); everything else is on cuda:0.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import warp as wp
import warp.fem as fem
import warp.sparse as wps
from warp.examples.fem import utils as fem_example_utils
from warp.fem.linalg import array_axpy

from bench.instrument import measure, MeasureResult

from .elasticity import (body_force_form, elasticity_form,
                         geometric_stiffness_form, lame, pressure_form,
                         traction_form)
from .mesh_io import FEMesh, read_mesh
from .modal import _axpy_vec3d, _dot, _scale_vec3d, _zero_rows
from .results import FEAResult, Measurement, SolverStatus
from .solver import (DEFAULT_DEVICE, _device_name, _dirichlet_projector,
                     _match_space_to_mesh, _surface_subdomain)


def _assemble_reference_load(fe: FEMesh, load_case: dict, space, geo, domain,
                             test, s_of_g: dict, n: int, device: str):
    """Reference load vector b (same load types as the static solver)."""
    b = wp.zeros(n, dtype=wp.vec3d, device=device)
    b_host = np.zeros((n, 3), dtype=np.float64)
    has_host = False
    for load in load_case.get("loads", []):
        lt = str(load.get("type", "")).lower()
        if lt == "force":
            nodes = np.array(sorted({s_of_g[int(g)]
                                     for g in fe.region(load["region"]).node_idx
                                     if int(g) in s_of_g}), dtype=np.int64)
            b_host[nodes] += np.asarray(load["vector"], float) / len(nodes)
            has_host = True
        elif lt in ("pressure", "traction"):
            surf = _surface_subdomain(geo, fe, load["region"], device)
            bt = fem.make_test(space=space, domain=surf)
            if lt == "pressure":
                rhs = fem.integrate(pressure_form, fields={"v": bt},
                                    values={"p": wp.float64(float(load["magnitude"]))},
                                    output_dtype=wp.vec3d)
            else:
                t = np.asarray(load["vector"], float)
                rhs = fem.integrate(traction_form, fields={"v": bt},
                                    values={"t": wp.vec3d(*t)}, output_dtype=wp.vec3d)
            array_axpy(x=rhs, y=b, alpha=1.0, beta=1.0)
        elif lt == "gravity":
            rho = float((load_case.get("material") or {}).get("density", 0.0))
            if rho <= 0:
                raise ValueError("gravity needs material.density")
            g = np.asarray(load["vector"], float)
            rhs = fem.integrate(body_force_form, fields={"v": test},
                                values={"f": wp.vec3d(*(rho * g))},
                                output_dtype=wp.vec3d)
            array_axpy(x=rhs, y=b, alpha=1.0, beta=1.0)
        else:
            raise ValueError(f"unsupported load type {lt!r}")
    if has_host:
        array_axpy(x=wp.array(b_host, dtype=wp.vec3d, device=device), y=b,
                   alpha=1.0, beta=1.0)
    return b


def solve_buckling(
    mesh,
    load_case: dict,
    n_modes: int = 4,
    *,
    device: str = DEFAULT_DEVICE,
    tol: float = 1e-10,
    max_iters: int = 20000,
    degree: Optional[int] = None,
    subspace_iters: int = 80,
    eig_tol: float = 1e-8,
    verbose: bool = False,
) -> FEAResult:
    """
    Lowest `n_modes` buckling load factors for (mesh, load_case).

    The load_case loads define the *reference* load; BLF × reference = critical.
    """
    fe: FEMesh = mesh if isinstance(mesh, FEMesh) else read_mesh(mesh)
    mat = load_case.get("material") or {}
    if "E" not in mat or "nu" not in mat:
        raise ValueError("load_case.material needs SI E [Pa] and nu")
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

    g_of_s = _match_space_to_mesh(space.node_positions().numpy(), fe.points)
    s_of_g = {int(g): i for i, g in enumerate(g_of_s)}
    fixed: List[int] = []
    for sup in load_case.get("supports", []):
        if str(sup.get("type", "fixed")).lower() not in ("fixed", "encastre", "clamped"):
            raise ValueError("buckling supports must be fully fixed")
        fixed.extend(s_of_g[int(g)] for g in fe.region(sup["region"]).node_idx
                     if int(g) in s_of_g)
    fixed_nodes = np.asarray(sorted(set(fixed)), dtype=np.int64)
    if len(fixed_nodes) == 0:
        raise ValueError("buckling needs supports (singular K)")

    b = _assemble_reference_load(fe, load_case, space, geo, domain, test,
                                 s_of_g, n, device)

    with measure(f"warp_buckle:{os.path.basename(fe.source)}", interval=0.02,
                 verbose=verbose) as meas:
        # ---- 1. pre-buckling static solve  K u0 = b ----
        Kstat = wps.bsr_copy(K)
        P = _dirichlet_projector(fixed_nodes, n, device)
        fem.project_linear_system(Kstat, b, P, fixed_value=None,
                                  normalize_projector=True)
        u0 = wp.zeros(n, dtype=wp.vec3d, device=device)
        err0, it0 = fem_example_utils.bsr_cg(Kstat, b=b, x=u0, tol=tol,
                                             max_iters=max_iters, quiet=True)
        b_norm = float(np.sqrt(wp.utils.array_inner(b, b)))
        static_ok = err0 <= max(tol * b_norm, tol) and it0 < max_iters

        # ---- 2. geometric stiffness  K_g(σ(u0)) ----
        u0_field = space.make_field()
        u0_field.dof_values = u0
        Kg = fem.integrate(geometric_stiffness_form,
                           fields={"u": trial, "v": test, "u0": u0_field},
                           values={"lam": wp.float64(lam), "mu": wp.float64(mu)},
                           output_dtype=wp.float64)

        # ---- 3. buckling pencil  K φ = λ (−K_g) φ ----
        # Project K (→ I on fixed dofs) and B = −K_g (→ 0 on fixed dofs), so
        # constrained dofs carry no geometric stiffness and drop out of the
        # spectrum — identical bookkeeping to the modal mass matrix.
        Keig = wps.bsr_copy(K)
        B = wps.bsr_scale(wps.bsr_copy(Kg), -1.0)      # B = −K_g (Kg untouched)
        z = wp.zeros(n, dtype=wp.vec3d, device=device)
        fem.project_linear_system(Keig, z, P, fixed_value=None, normalize_projector=True)
        fem.project_linear_system(B, wp.zeros(n, dtype=wp.vec3d, device=device), P,
                                  fixed_value=None, normalize_projector=True)
        _zero_rows(B, fixed_nodes, device)

        m = max(n_modes + 4, 2 * n_modes)
        rng = np.random.default_rng(0)
        X = []
        for _ in range(m):
            v = rng.standard_normal((n, 3))
            v[fixed_nodes] = 0.0
            X.append(wp.array(v, dtype=wp.vec3d, device=device))
        scratch = [wp.zeros(n, dtype=wp.vec3d, device=device) for _ in range(m)]

        def mv(A, x, out):
            wps.bsr_mv(A=A, x=x, y=out, alpha=1.0, beta=0.0)

        prev = None
        converged = False
        lams = np.zeros(n_modes)
        it = 0
        for it in range(subspace_iters):
            # Y <- K^-1 (B X)
            Y = []
            for j in range(m):
                rhs = wp.zeros(n, dtype=wp.vec3d, device=device)
                mv(B, X[j], rhs)
                y = wp.zeros(n, dtype=wp.vec3d, device=device)
                fem_example_utils.bsr_cg(Keig, b=rhs, x=y, tol=tol,
                                         max_iters=max_iters, quiet=True)
                Y.append(y)

            KY = [wp.zeros(n, dtype=wp.vec3d, device=device) for _ in range(m)]
            BY = [wp.zeros(n, dtype=wp.vec3d, device=device) for _ in range(m)]
            for j in range(m):
                mv(Keig, Y[j], KY[j])
                mv(B, Y[j], BY[j])
            Kr = np.empty((m, m)); Br = np.empty((m, m))
            for i in range(m):
                for j in range(i, m):
                    Kr[i, j] = Kr[j, i] = _dot(Y[i], KY[j])
                    Br[i, j] = Br[j, i] = _dot(Y[i], BY[j])

            # reduced  Kr q = θ Br q  via Cholesky of the SPD Kr
            Lr = np.linalg.cholesky(0.5 * (Kr + Kr.T))
            Li = np.linalg.inv(Lr)
            C = Li @ (0.5 * (Br + Br.T)) @ Li.T
            eigval, W = np.linalg.eigh(0.5 * (C + C.T))    # eigval = 1/θ = 1/λ
            q = Li.T @ W

            # order Ritz vectors by |1/λ| descending -> dominant (smallest |λ|) lead
            order = np.argsort(-np.abs(eigval))
            eigval = eigval[order]; q = q[:, order]

            for j in range(m):
                scratch[j].zero_()
                for i in range(m):
                    c = float(q[i, j])
                    if c != 0.0:
                        wp.launch(_axpy_vec3d, dim=n,
                                  inputs=[Y[i], scratch[j], wp.float64(c)],
                                  device=device)
            for j in range(m):
                nrm = np.sqrt(_dot(scratch[j], scratch[j]))
                if nrm > 0:
                    wp.launch(_scale_vec3d, dim=n,
                              inputs=[scratch[j], wp.float64(1.0 / nrm)], device=device)
                wp.copy(X[j], scratch[j])

            # smallest positive load factors λ = 1/eigval, eigval > 0
            pos = eigval[eigval > 1e-30]
            if len(pos) >= n_modes:
                lams = np.sort(1.0 / pos)[:n_modes]
                if prev is not None and len(prev) == len(lams):
                    rel = np.max(np.abs(lams - prev) / np.maximum(np.abs(lams), 1e-30))
                    if verbose:
                        print(f"    buckle it{it+1}: BLF[0]={lams[0]:.6f} rel {rel:.2e}")
                    if rel < eig_tol:
                        converged = True
                        break
                prev = lams.copy()
        wp.synchronize_device(device)
    res: MeasureResult = meas["result"]

    status = SolverStatus(
        regime="solid", backend="warp-gpu", meshed=True,
        converged=bool(converged and static_ok),
        n_dof=3 * n, n_iters=it + 1, wall_time_s=res.wall_s, cpu_avg_pct=res.cpu_avg,
        gpu_util_avg_pct=res.gpu_util_avg, device=_device_name(device),
        element="C3D10" if degree == 2 else "C3D4", tol=eig_tol,
        message="" if (converged and static_ok) else
                ("pre-buckling static solve did not converge" if not static_ok
                 else f"subspace iteration hit {subspace_iters} sweeps without "
                      f"reaching eig_tol={eig_tol:g}"),
    )
    out = FEAResult(solver_status=status)
    if not (converged and static_ok):
        return out.invalidate(status.message)

    out.measured["buckling_load_factor"] = Measurement(
        value=float(lams[0]), unit="dimensionless")
    out.measured["buckling_load_factors_all"] = [float(x) for x in lams]
    out.fields = {"static_displacement": u0}
    return out
