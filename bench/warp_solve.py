"""
Phase 2 — GPU linear-static elasticity solver on NVIDIA Warp (warp.fem).

Solves the *same* C3D8 cantilever problem that `gen_inp.py` writes and `ccx`
references, but entirely on the GPU:

  a(u, v) = ∫ [ 2μ ε(u):ε(v) + λ tr(ε(u)) tr(ε(v)) ] dΩ ,   ε = sym(∇u)

with
  * homogeneous Dirichlet u = 0 on the clamped face  (NFIX, x = 0)
  * concentrated nodal loads on the loaded face        (NLOAD, CLOAD in z)

Hard constraints (see PHASE2_SPEC.md §2), all verified numerically by the driver:
  1. GPU-only: every Warp array/kernel lives on cuda:0; no host<->device copy or
     .numpy() inside the measured solve region.  cpu_avg < 20% is asserted.
  2. Performance: Warp(GPU) wall-clock vs ccx CG wall-clock -> crossover curve.
  3. Accuracy: wp.float64 throughout; displacement rel-err < 1e-3 vs ccx .dat.
  4. Efficiency: bsr_cg captures the CG iteration in a CUDA graph (built in) with
     Jacobi (diagonal) preconditioning.

Public API:
    solve_inp(inp_path) -> (MeasureResult, disp_dict)
    solve_inp(inp_path, return_info=True) -> (MeasureResult, disp_dict, info)

`disp_dict` is keyed by the original .inp node id, so it lines up directly with
`run_ccx.parse_dat_displacements` for the parity comparison.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# Persistent, writable kernel cache -> no recompilation between runs (spec §5).
os.environ.setdefault(
    "WARP_CACHE_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".warp_cache"),
)

import warp as wp
import warp.fem as fem
import warp.sparse as wps
from warp.examples.fem import utils as fem_example_utils

from .instrument import measure, MeasureResult

wp.set_module_options({"enable_backward": False})

DEVICE = "cuda:0"


# --------------------------------------------------------------------------- #
# .inp parsing (targeted at gen_inp.py's C3D8 cantilever decks)
# --------------------------------------------------------------------------- #
@dataclass
class InpModel:
    node_ids: np.ndarray          # (N,) original .inp ids, sorted
    positions: np.ndarray         # (N,3) float64, row i <-> node_ids[i]
    hexes: np.ndarray             # (M,8) int32, 0-based into positions rows
    id2idx: Dict[int, int]
    E: float
    nu: float
    fixed_idx: np.ndarray         # rows into positions that are clamped
    load_idx: np.ndarray          # rows into positions that carry the point load
    load_vec: np.ndarray          # (3,) per-node force applied at each load node


def _num(tok: str) -> float:
    return float(tok.replace("D", "E").replace("d", "e"))


def parse_inp(path: str) -> InpModel:
    nodes: Dict[int, Tuple[float, float, float]] = {}
    hexes: List[List[int]] = []
    nsets: Dict[str, List[int]] = {}
    E = nu = None
    cload: List[Tuple[str, int, float]] = []   # (nset, dof, value)

    section = None
    cur_nset = None
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("**"):
                continue
            if line.startswith("*"):
                key = line.split(",")[0].strip().upper()
                up = line.upper()
                section = key
                cur_nset = None
                if key == "*NSET":
                    m = re.search(r"NSET\s*=\s*([A-Za-z0-9_]+)", up)
                    cur_nset = m.group(1)
                    nsets.setdefault(cur_nset, [])
                # keep material section context; ELASTIC data is on the next line
                continue

            if section == "*NODE":
                p = [t for t in line.split(",") if t.strip()]
                nid = int(p[0]); nodes[nid] = (float(p[1]), float(p[2]), float(p[3]))
            elif section == "*ELEMENT":
                p = [t for t in line.split(",") if t.strip()]
                hexes.append([int(x) for x in p[1:9]])
            elif section == "*NSET" and cur_nset is not None:
                nsets[cur_nset].extend(int(t) for t in line.split(",") if t.strip())
            elif section == "*ELASTIC":
                p = [t for t in line.split(",") if t.strip()]
                E, nu = _num(p[0]), _num(p[1])
            elif section == "*CLOAD":
                p = [t for t in line.split(",") if t.strip()]
                cload.append((p[0], int(p[1]), _num(p[2])))

    if E is None or nu is None:
        raise ValueError(f"{path}: could not parse *ELASTIC E, nu")

    node_ids = np.array(sorted(nodes), dtype=np.int64)
    id2idx = {int(nid): i for i, nid in enumerate(node_ids)}
    positions = np.array([nodes[int(nid)] for nid in node_ids], dtype=np.float64)
    hex_arr = np.array([[id2idx[n] for n in h] for h in hexes], dtype=np.int32)

    # Dirichlet: fixed nodes come from the NSET referenced by *BOUNDARY (NFIX).
    fixed_ids = nsets.get("NFIX")
    if fixed_ids is None:                      # fall back to the clamped face x = xmin
        xmin = positions[:, 0].min()
        fixed_idx = np.where(np.abs(positions[:, 0] - xmin) < 1e-9)[0]
    else:
        fixed_idx = np.array([id2idx[i] for i in fixed_ids], dtype=np.int64)

    # Load: concentrated force per node on the CLOAD set (matches ccx exactly).
    load_vec = np.zeros(3, dtype=np.float64)
    load_ids = None
    for nset, dof, val in cload:
        load_vec[dof - 1] += val
        load_ids = nsets.get(nset, [])
    if load_ids is None:
        raise ValueError(f"{path}: no *CLOAD found")
    load_idx = np.array([id2idx[i] for i in load_ids], dtype=np.int64)

    return InpModel(node_ids, positions, hex_arr, id2idx, E, nu,
                    fixed_idx, load_idx, load_vec)


# --------------------------------------------------------------------------- #
# Warp.fem elasticity forms
# --------------------------------------------------------------------------- #
@fem.integrand
def elasticity_form(s: fem.Sample, u: fem.Field, v: fem.Field,
                    lam: wp.float64, mu: wp.float64):
    """Displacement-based linear elasticity bilinear form (double precision)."""
    eps_u = fem.D(u, s)                    # sym(grad u), mat33d
    eps_v = fem.D(v, s)
    tr = wp.trace(eps_u)
    stress = (mu + mu) * eps_u + (lam * tr) * wp.identity(n=3, dtype=wp.float64)
    return wp.ddot(eps_v, stress)


@fem.integrand
def unit_projector_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(u(s), v(s))


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
def solve_inp(
    inp_path: str,
    tol: float = 1e-10,
    max_iters: int = 5000,
    return_info: bool = False,
    verbose: bool = True,
):
    model = parse_inp(inp_path)
    n_nodes = model.positions.shape[0]

    lam = model.E * model.nu / ((1.0 + model.nu) * (1.0 - 2.0 * model.nu))
    mu = model.E / (2.0 * (1.0 + model.nu))

    # ---- geometry & function space, all resident on cuda:0 ----
    positions = wp.array(model.positions, dtype=wp.vec3d, device=DEVICE)
    hex_idx = wp.array(model.hexes, dtype=wp.int32, device=DEVICE)
    geo = fem.Hexmesh(hex_idx, positions)

    space = fem.make_polynomial_space(geo, degree=1, dtype=wp.vec3d)
    assert space.node_count() == n_nodes, (space.node_count(), n_nodes)

    domain = fem.Cells(geo)
    test = fem.make_test(space=space, domain=domain)
    trial = fem.make_trial(space=space, domain=domain)

    # ---- stiffness matrix K (BSR, mat33d blocks) ----
    K = fem.integrate(
        elasticity_form,
        fields={"u": trial, "v": test},
        values={"lam": wp.float64(lam), "mu": wp.float64(mu)},
        output_dtype=wp.float64,
    )

    # ---- load vector b: concentrated nodal forces on the CLOAD set ----
    b_np = np.zeros((n_nodes, 3), dtype=np.float64)
    b_np[model.load_idx] += model.load_vec
    b = wp.array(b_np, dtype=wp.vec3d, device=DEVICE)

    # ---- Dirichlet projector P: identity 3x3 block on each clamped node ----
    nfix = len(model.fixed_idx)
    P = wps.bsr_zeros(n_nodes, n_nodes, wp.mat33d, device=DEVICE)
    rows = wp.array(model.fixed_idx.astype(np.int32), dtype=wp.int32, device=DEVICE)
    eye = np.tile(np.eye(3, dtype=np.float64), (nfix, 1, 1))
    vals = wp.array(eye, dtype=wp.mat33d, device=DEVICE)
    wps.bsr_set_from_triplets(P, rows, rows, vals)

    # Enforce u = 0 on fixed dofs (symmetric projection of K and b).
    fem.project_linear_system(K, b, P, fixed_value=None, normalize_projector=True)

    # ---- CG solve, captured in a CUDA graph, Jacobi-preconditioned ----
    x = wp.zeros_like(b)

    # Warm-up: JIT + graph capture excluded from the measured region (spec §5).
    fem_example_utils.bsr_cg(K, b=b, x=x, tol=tol, max_iters=max_iters, quiet=True)
    wp.synchronize_device(DEVICE)

    x.zero_()
    with measure(f"warp_solve:{os.path.basename(inp_path)}", interval=0.02,
                 verbose=verbose) as m:
        err, iters = fem_example_utils.bsr_cg(
            K, b=b, x=x, tol=tol, max_iters=max_iters, quiet=True
        )
        wp.synchronize_device(DEVICE)
    res: MeasureResult = m["result"]

    # ---- read displacements back (host copy is AFTER the measured region) ----
    u = x.numpy().reshape(n_nodes, 3)
    disp = {int(model.node_ids[i]): (float(u[i, 0]), float(u[i, 1]), float(u[i, 2]))
            for i in range(n_nodes)}

    info = {
        "ndof": 3 * n_nodes,
        "n_nodes": n_nodes,
        "n_elems": int(model.hexes.shape[0]),
        "cg_iters": int(iters),
        "cg_residual": float(err),
        "lam": lam, "mu": mu, "E": model.E, "nu": model.nu,
        "gpu_util_avg": res.gpu_util_avg,
        "gpu_mem_peak_mb": res.gpu_mem_peak_mb,
        "cpu_avg": res.cpu_avg,
    }
    if verbose:
        print(f"    ndof={info['ndof']}  cg_iters={iters}  residual={err:.2e}  "
              f"max|u|={np.linalg.norm(u, axis=1).max():.6e}")

    if return_info:
        return res, disp, info
    return res, disp


if __name__ == "__main__":
    import sys
    job = sys.argv[1] if len(sys.argv) > 1 else "cases/cant_20x4x4_iterative"
    inp = job if job.endswith(".inp") else job + ".inp"
    res, disp, info = solve_inp(inp, return_info=True)
    print(res.summary())
    print("info:", {k: info[k] for k in ("ndof", "cg_iters", "cg_residual",
                                         "gpu_util_avg", "cpu_avg")})
