"""
warp_fea.solver — GPU linear-static structural solver behind the Judge's Stage 3a.

Main entry point (spec §1.1):
    solve_structural(mesh, load_case) -> FEAResult      # what the Judge calls
    solve_inp(inp_path)               -> FEAResult      # validation-only helper

Everything is SI (Pa, m, N) and fp64. The solve region itself is GPU-resident:
no host<->device copy happens between `measure()` entry and exit.

load_case schema (Stage 2 "structural" case). Values come from here; only the
*place* they apply is resolved through gmsh physical groups (spec §4.4):

    {
      "material": {"E": 193e9, "nu": 0.29, "density": 7900,
                   "yield_strength": 215e6},          # yield unused by the solver
      "supports": [{"region": "fixed_face", "type": "fixed"}],
      "loads": [
        {"region": "load_face", "type": "pressure", "magnitude": 1e6},
        {"region": "load_face", "type": "traction", "vector": [0, 0, -1e6]},
        {"region": "tip_face",  "type": "force",    "vector": [0, 0, -1000]},
        {"type": "gravity", "vector": [0, 0, -9.81]}
      ]
    }

  * pressure  [Pa]  — positive pushes into the face (CalculiX *DLOAD convention)
  * traction  [Pa]  — surface stress vector
  * force     [N]   — TOTAL force on the region, split equally over its nodes
                      (mirrors a ccx *CLOAD on the same NSET, so the oracle and
                      this path solve the identical discrete problem)
  * gravity   [m/s^2] — needs material.density
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

os.environ.setdefault(
    "WARP_CACHE_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".warp_cache"),
)

import warp as wp
import warp.fem as fem
import warp.sparse as wps
from warp.examples.fem import utils as fem_example_utils
from warp.fem.linalg import array_axpy

from bench.instrument import measure, MeasureResult

from .elasticity import (
    body_force_form, disp_norm_kernel, elasticity_form, elasticity_form_ortho,
    lame, material_model, material_parts, position_at_qp, pressure_form,
    traction_form, von_mises_at_qp, von_mises_at_qp_ortho,
)
from .mesh_io import FEMesh, read_mesh, region_elements, region_faces
from .results import FEAResult, Location, Measurement, SolverStatus
from .validation import (peek_mesh_cell_types, post_solve_linearity_check,
                         rejected_result, validate_inputs)

wp.set_module_options({"enable_backward": False})

DEFAULT_DEVICE = "cuda:0"
_GEOM_TOL = 1e-9          # metres; node-matching tolerance


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _device_name(device: str) -> str:
    try:
        return str(wp.get_device(device).name)
    except Exception:
        return device


def _match_space_to_mesh(space_pos: np.ndarray, mesh_pts: np.ndarray) -> np.ndarray:
    """
    space node index -> mesh (gmsh) node index, by coordinate.

    warp.fem invents its own node numbering (degree-2 adds mid-edge nodes it
    positions itself), so the two numberings only agree geometrically. For
    straight-sided tets both codes put mid-side nodes at the exact edge midpoint,
    so matching on coordinates is well posed.

    Hash on a quantised key (fast path), then brute-force the handful of points
    that straddle a bucket boundary. No scipy: this env's scipy/numpy pairing is
    broken, and the package should not depend on fixing a shared conda install.
    """
    dec = 9                                              # 1e-9 m grid, >> fp noise
    table: Dict[tuple, int] = {}
    for i, p in enumerate(np.round(mesh_pts, dec)):
        table[(p[0], p[1], p[2])] = i

    idx = np.empty(len(space_pos), dtype=np.int64)
    misses: List[int] = []
    for i, p in enumerate(np.round(space_pos, dec)):
        j = table.get((p[0], p[1], p[2]), -1)
        idx[i] = j
        if j < 0:
            misses.append(i)
    for i in misses:                                     # rare: rounding straddle
        d = np.linalg.norm(mesh_pts - space_pos[i], axis=1)
        idx[i] = int(np.argmin(d))

    worst = float(np.max(np.linalg.norm(mesh_pts[idx] - space_pos, axis=1)))
    if worst > _GEOM_TOL:
        raise RuntimeError(
            f"a space node has no mesh node within {_GEOM_TOL} m "
            f"(worst mismatch {worst:.3e} m) — mesh order vs space degree mismatch?"
        )
    return idx


def _surface_subdomain(geo, fe: FEMesh, region: str, device: str):
    """Subdomain of the boundary holding exactly the named region's faces."""
    # geo indexes the *compacted* corner vertices; region faces index fe.points.
    g2v = {int(g): i for i, g in enumerate(fe.vertex_ids)}
    want = {frozenset(g2v[int(v)] for v in f) for f in region_faces(fe, region)}
    face_verts = geo.face_vertex_indices.numpy()
    bnd_sides = geo._boundary_face_indices.numpy()        # boundary elem -> side idx
    sel = [i for i, sid in enumerate(bnd_sides)
           if frozenset(int(v) for v in face_verts[sid]) in want]
    if not sel:
        raise ValueError(f"region {region!r}: none of its {len(want)} faces matched a "
                         f"boundary side of the mesh")
    if len(sel) != len(want):
        raise ValueError(f"region {region!r}: matched {len(sel)} of {len(want)} faces "
                         f"to boundary sides (is the region an interior surface?)")
    return fem.Subdomain(
        fem.BoundarySides(geo),
        element_indices=wp.array(np.asarray(sel, dtype=np.int32), dtype=wp.int32,
                                 device=device),
    )


def _dirichlet_projector(fixed_nodes: np.ndarray, n_nodes: int, device: str):
    """Block-diagonal identity projector on the clamped nodes."""
    P = wps.bsr_zeros(n_nodes, n_nodes, wp.mat33d, device=device)
    if len(fixed_nodes) == 0:
        return P
    rows = wp.array(np.asarray(fixed_nodes, dtype=np.int32), dtype=wp.int32, device=device)
    eye = np.tile(np.eye(3, dtype=np.float64), (len(fixed_nodes), 1, 1))
    wps.bsr_set_from_triplets(P, rows, rows, wp.array(eye, dtype=wp.mat33d, device=device))
    return P


# --------------------------------------------------------------------------- #
# core solve
# --------------------------------------------------------------------------- #
def _solve(
    fe: FEMesh,
    load_case: dict,
    device: str = DEFAULT_DEVICE,
    tol: float = 1e-8,
    max_iters: int = 0,
    degree: Optional[int] = None,
    want_fields: bool = True,
    verbose: bool = False,
) -> FEAResult:
    # An assembly is `materials: [{region, E, nu, ...}, ...]`; a single part keeps
    # using `material`. Both end up as a list of (region-or-None, kind, values) so
    # assembly and single-part share one assembly loop.
    parts = material_parts(load_case)

    degree = degree if degree is not None else fe.order
    if max_iters <= 0:
        max_iters = 20000

    # ---- geometry / space (cuda-resident) ----
    # Only the corner vertices go into the geometry (see FEMesh.corners_local):
    # feeding the full tetra10 point list would leave warp with isolated vertices
    # parked at the origin and empty stiffness rows.
    positions = wp.array(fe.vertex_points, dtype=wp.vec3d, device=device)
    tet_idx = wp.array(fe.corners_local.astype(np.int32), dtype=wp.int32, device=device)
    geo = fem.Tetmesh(tet_idx, positions)
    space = fem.make_polynomial_space(geo, degree=degree, dtype=wp.vec3d)
    n_nodes = space.node_count()

    domain = fem.Cells(geo)
    test = fem.make_test(space=space, domain=domain)
    trial = fem.make_trial(space=space, domain=domain)

    # ---- stiffness: one integration per material part, summed ----
    K = None
    part_domains = []          # (part, domain, test, trial) reused for stress/gravity
    for part in parts:
        if part.region is None:
            dom_p, test_p, trial_p = domain, test, trial
        else:
            elems = region_elements(fe, part.region)
            dom_p = fem.Subdomain(
                domain,
                element_indices=wp.array(elems.astype(np.int32), dtype=wp.int32,
                                         device=device),
            )
            test_p = fem.make_test(space=space, domain=dom_p)
            trial_p = fem.make_trial(space=space, domain=dom_p)
        part_domains.append((part, dom_p, test_p, trial_p))
        Ki = fem.integrate(
            part.stiffness_form,
            fields={"u": trial_p, "v": test_p},
            values=part.values,
            output_dtype=wp.float64,
        )
        K = Ki if K is None else K + Ki

    # ---- node correspondence (space <-> mesh) ----
    space_pos = space.node_positions().numpy()
    g_of_s = _match_space_to_mesh(space_pos, fe.points)
    s_of_g = {int(g): i for i, g in enumerate(g_of_s)}

    def space_nodes_of(region: str) -> np.ndarray:
        idx = [s_of_g[int(g)] for g in fe.region(region).node_idx if int(g) in s_of_g]
        if not idx:
            raise ValueError(f"region {region!r} has no nodes in the degree-{degree} space")
        return np.asarray(sorted(set(idx)), dtype=np.int64)

    # ---- loads ----
    b = wp.zeros(n_nodes, dtype=wp.vec3d, device=device)
    b_host = np.zeros((n_nodes, 3), dtype=np.float64)
    has_host_load = False

    for load in load_case.get("loads", []):
        ltype = str(load.get("type", "")).lower()
        if ltype == "force":
            nodes = space_nodes_of(load["region"])
            total = np.asarray(load["vector"], dtype=np.float64)
            b_host[nodes] += total / len(nodes)          # mirrors *CLOAD on the NSET
            has_host_load = True
        elif ltype in ("pressure", "traction"):
            surf = _surface_subdomain(geo, fe, load["region"], device)
            bd_test = fem.make_test(space=space, domain=surf)
            if ltype == "pressure":
                rhs = fem.integrate(pressure_form, fields={"v": bd_test},
                                    values={"p": wp.float64(float(load["magnitude"]))},
                                    output_dtype=wp.vec3d)
            else:
                t = np.asarray(load["vector"], dtype=np.float64)
                rhs = fem.integrate(traction_form, fields={"v": bd_test},
                                    values={"t": wp.vec3d(*t)}, output_dtype=wp.vec3d)
            array_axpy(x=rhs, y=b, alpha=1.0, beta=1.0)
        elif ltype == "gravity":
            g = np.asarray(load["vector"], dtype=np.float64)
            # body force is rho*g, and rho is per part: integrate each material
            # region with its own density (a single part covers the whole model)
            for part, dom_p, test_p, _ in part_domains:
                rho = part.density or 0.0
                if rho <= 0:
                    raise ValueError(
                        f"gravity load needs a positive density on material part "
                        f"{part.region or '<single>'}")
                rhs = fem.integrate(body_force_form, fields={"v": test_p},
                                    values={"f": wp.vec3d(*(rho * g))},
                                    output_dtype=wp.vec3d)
                array_axpy(x=rhs, y=b, alpha=1.0, beta=1.0)
        else:
            raise ValueError(f"unsupported load type {ltype!r}")

    if has_host_load:
        array_axpy(x=wp.array(b_host, dtype=wp.vec3d, device=device), y=b,
                   alpha=1.0, beta=1.0)

    # ---- supports ----
    fixed: List[int] = []
    for sup in load_case.get("supports", []):
        stype = str(sup.get("type", "fixed")).lower()
        if stype not in ("fixed", "encastre", "clamped"):
            raise ValueError(f"unsupported support type {stype!r} (only fully-fixed "
                             f"supports are implemented)")
        fixed.extend(int(i) for i in space_nodes_of(sup["region"]))
    fixed_nodes = np.asarray(sorted(set(fixed)), dtype=np.int64)
    if len(fixed_nodes) == 0:
        raise ValueError("load_case has no supports — the stiffness matrix is singular")

    P = _dirichlet_projector(fixed_nodes, n_nodes, device)
    fem.project_linear_system(K, b, P, fixed_value=None, normalize_projector=True)

    # ---- solve (GPU-only region; bsr_cg captures a CUDA graph + Jacobi precond) ----
    b_norm = float(np.sqrt(wp.utils.array_inner(b, b)))   # on-device reduction
    x = wp.zeros_like(b)
    fem_example_utils.bsr_cg(K, b=b, x=x, tol=tol, max_iters=max_iters, quiet=True)
    wp.synchronize_device(device)                        # warm-up: JIT + graph capture

    x.zero_()
    with measure(f"warp_fea:{os.path.basename(fe.source)}", interval=0.02,
                 verbose=verbose) as m:
        err, iters = fem_example_utils.bsr_cg(K, b=b, x=x, tol=tol,
                                              max_iters=max_iters, quiet=True)
        wp.synchronize_device(device)
    res: MeasureResult = m["result"]

    # warp's CG stops at err <= max(tol*||b||, tol); it returns err but not that
    # threshold, so recompute it here rather than guessing convergence.
    atol_eff = max(tol * b_norm, tol)
    converged = bool(err <= atol_eff) and int(iters) < max_iters

    status = SolverStatus(
        regime="solid", backend="warp-gpu", meshed=True, converged=converged,
        n_dof=3 * n_nodes, n_iters=int(iters), wall_time_s=res.wall_s,
        cpu_avg_pct=res.cpu_avg, gpu_util_avg_pct=res.gpu_util_avg,
        device=_device_name(device), residual=float(err), tol=float(tol),
        element="C3D10" if degree == 2 else "C3D4",
        message="" if converged else
                f"CG did not reach tol: residual {err:.3e} > atol {atol_eff:.3e} "
                f"after {iters} iters",
    )
    result = FEAResult(solver_status=status)

    # ---- von Mises at quadrature points (spec §4.2) ----
    u_field = space.make_field()
    u_field.dof_values = x
    # von Mises at quadrature points, per material part: the same strain gives a
    # different stress under a different material, so each region must be
    # evaluated with its own constitutive law and the results concatenated.
    vm_parts, xyz_parts = [], []
    for part, dom_p, _, _ in part_domains:
        q_p = fem.RegularQuadrature(dom_p, order=2)    # 4 pts/tet == ccx C3D10
        n_p = q_p.total_point_count()
        vm_p = wp.zeros(n_p, dtype=wp.float64, device=device)
        xyz_p = wp.zeros(n_p, dtype=wp.vec3d, device=device)
        fem.interpolate(part.von_mises_form, dest=vm_p, at=q_p,
                        fields={"u": u_field}, values=part.values)
        fem.interpolate(position_at_qp, dest=xyz_p, at=q_p)
        vm_parts.append(vm_p)
        xyz_parts.append(xyz_p)
    if len(vm_parts) == 1:
        vm, qp_xyz = vm_parts[0], xyz_parts[0]
    else:
        vm = wp.array(np.concatenate([a.numpy() for a in vm_parts]),
                      dtype=wp.float64, device=device)
        qp_xyz = wp.array(np.concatenate([a.numpy() for a in xyz_parts]),
                          dtype=wp.vec3d, device=device)

    unorm = wp.zeros(n_nodes, dtype=wp.float64, device=device)
    wp.launch(disp_norm_kernel, dim=n_nodes, inputs=[x, unorm], device=device)

    if want_fields:
        result.fields = {"displacement": x, "von_mises": vm,
                         "von_mises_points": qp_xyz, "displacement_norm": unorm}

    if not converged:
        return result.invalidate(status.message)

    # ---- measured (host reads happen only after the measured region) ----
    un = unorm.numpy()
    i_disp = int(np.argmax(un))
    g_disp = int(g_of_s[i_disp])
    result.measured["max_displacement"] = Measurement(
        value=float(un[i_disp]), unit="m",
        location=Location(coords=fe.points[g_disp].tolist(), node_id=g_disp + 1,
                          region=fe.region_of_node(g_disp)),
    )

    vm_np = vm.numpy()
    i_vm = int(np.argmax(vm_np))
    xyz = qp_xyz.numpy()[i_vm]
    # the peak sits at a quadrature point, not a node; report the nearest node so
    # Stage 5 can name the region it belongs to
    g_near = int(np.argmin(np.linalg.norm(fe.points - xyz, axis=1)))
    result.measured["max_von_mises_stress"] = Measurement(
        value=float(vm_np[i_vm]), unit="Pa",
        location=Location(coords=[float(c) for c in xyz], node_id=g_near + 1,
                          region=fe.region_of_node(g_near)),
    )
    return result


# --------------------------------------------------------------------------- #
# public API (spec §1.1)
# --------------------------------------------------------------------------- #
def solve_structural(
    mesh: Union[str, "object"],
    load_case: dict,
    *,
    device: str = DEFAULT_DEVICE,
    tol: float = 1e-8,
    max_iters: int = 0,
    degree: Optional[int] = None,
    want_fields: bool = True,
    verbose: bool = False,
    validate: bool = True,
) -> FEAResult:
    """
    ★ Judge Stage 3a entry point: gmsh .msh (path or meshio.Mesh) + load_case -> FEAResult.

    Regions named in `load_case` are resolved through the mesh's gmsh physical groups.

    With `validate=True` (default, PHASE4 §B) the input is gated first: out-of-scope
    problems (non-tet elements, non-isotropic/non-linear material, unsupported or
    unvalidated loads, contact, missing supports) are *refused* — a rejected
    FEAResult (`rejected=True`, `reject_codes=[...]`, empty `measured`) is returned
    instead of a plausible wrong number, and the reason is tallied for later
    demand analysis. The load_case is parsed through `load_case_adapter` so the
    solver never touches the raw external wire format.
    """
    if validate:
        cell_types = None if isinstance(mesh, FEMesh) else peek_mesh_cell_types(mesh)
        vr, clc = validate_inputs(cell_types, load_case)
        if not vr.ok:
            return rejected_result(vr, backend="warp-gpu")
        load_case = clc.to_solver_dict()        # canonical shape from the adapter
        warn_codes = [c for c, _ in vr.warnings]
        warn_msg = vr.warning_message()
    else:
        warn_codes, warn_msg = [], ""

    fe = mesh if isinstance(mesh, FEMesh) else read_mesh(mesh)
    result = _solve(fe, load_case, device=device, tol=tol, max_iters=max_iters,
                    degree=degree, want_fields=want_fields, verbose=verbose)

    # post-solve WARN: peak displacement large vs model size => linearity suspect
    if validate and result.solver_status.converged and "max_displacement" in result.measured:
        pts = fe.points
        bbox_diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        lin = post_solve_linearity_check(
            result.measured["max_displacement"].value, bbox_diag)
        if lin is not None:
            warn_codes = warn_codes + [lin[0]]
            warn_msg = "; ".join(m for m in (warn_msg, f"[{lin[0]}] {lin[1]}") if m)

    if validate:
        result.warnings = warn_codes
        if warn_msg:
            result.solver_status.message = "; ".join(
                m for m in (result.solver_status.message, "WARN: " + warn_msg) if m)
    return result


def solve_inp(
    inp_path: str,
    load_case: Optional[dict] = None,
    *,
    device: str = DEFAULT_DEVICE,
    tol: float = 1e-8,
    max_iters: int = 0,
    degree: Optional[int] = None,
    want_fields: bool = True,
    verbose: bool = False,
) -> FEAResult:
    """
    Validation-only helper: solve the same problem from an Abaqus/CalculiX .inp.

    The .inp supplies the mesh and its NSET/ELSET regions; `load_case` still
    supplies BC/load *values* (an .inp baked by `validate.bake_inp` carries the
    same load_case, so both paths solve the identical discrete problem).
    """
    fe = read_mesh(inp_path)
    if load_case is None:
        raise ValueError("solve_inp needs the load_case that produced the .inp")
    return _solve(fe, load_case, device=device, tol=tol, max_iters=max_iters,
                  degree=degree, want_fields=want_fields, verbose=verbose)
