"""
Validation harness (spec §3, §6): bake the *same* (mesh, load_case) into a
CalculiX deck, run it as the oracle, and compare measured dicts.

The oracle path is deliberately independent where it matters:
  * pressure goes in as a real *DLOAD on element faces (not as nodal forces the
    Warp side also computed), so the traction assembly is genuinely checked;
  * von Mises is compared at *integration points* on both sides (spec §4.2),
    because ccx extrapolates-and-averages to nodes and Warp does not — comparing
    nodal values would measure the post-processing, not the physics.

Usage:
    python -m warp_fea.validate                 # full case set -> measured_parity.csv
    python -m warp_fea.validate --case cantilever --order 2
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
from typing import Dict, List, Optional, Tuple

import numpy as np

from bench.instrument import measure
from bench.run_ccx import parse_dat_displacements

from .cases import ALL_CASES
from .mesh_io import FEMesh, read_mesh, region_faces, tet_face_lookup
from .results import FEAResult
from .solver import solve_structural

# spec §3-1 acceptance thresholds
TOL_DISP = 1e-3        # relative
TOL_VM = 0.03          # relative (3%)

_NUM = r"[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?"
_STRESS_ROW = re.compile(
    rf"^\s*(\d+)\s+(\d+)\s+({_NUM})\s+({_NUM})\s+({_NUM})\s+({_NUM})\s+({_NUM})\s+({_NUM})\s*$"
)


def pod_cpu_quota() -> int:
    """ccx thread count must follow the pod's cgroup quota, not os.cpu_count()."""
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()
        if q != "max":
            return max(1, int(float(q) / float(p)))
    except Exception:
        pass
    return os.cpu_count() or 1


# --------------------------------------------------------------------------- #
# .inp baking
# --------------------------------------------------------------------------- #
def bake_inp(fe: FEMesh, load_case: dict, path: str) -> str:
    """
    Write a CalculiX deck for exactly this (mesh, load_case).

    Node ids are 1-based meshio indices, so `node_id` in a FEAResult location and
    a node id in this deck refer to the same point.
    """
    etype = "C3D10" if fe.tet_type == "tetra10" else "C3D4"
    nn = 10 if etype == "C3D10" else 4
    mat = load_case["material"]

    L: List[str] = ["*HEADING", f" {load_case.get('name','case')} (SI: Pa, m, N)"]

    L.append("*NODE, NSET=NALL")
    for i, p in enumerate(fe.points):
        L.append(f"{i+1}, {p[0]:.12e}, {p[1]:.12e}, {p[2]:.12e}")

    L.append(f"*ELEMENT, TYPE={etype}, ELSET=EALL")
    for ei, t in enumerate(fe.tets):
        ids = ", ".join(str(int(v) + 1) for v in t[:nn])
        L.append(f"{ei+1}, {ids}")

    # region NSETs (named exactly like the gmsh physical groups)
    def nset(name: str, ids: np.ndarray) -> None:
        L.append(f"*NSET, NSET={name}")
        ids = np.asarray(sorted(int(i) + 1 for i in ids))
        for k in range(0, len(ids), 8):
            L.append(", ".join(str(v) for v in ids[k:k + 8]))

    used_regions = {s["region"] for s in load_case.get("supports", [])}
    used_regions |= {l["region"] for l in load_case.get("loads", []) if "region" in l}
    for rname in sorted(used_regions):
        nset(rname, fe.region(rname).node_idx)

    L += ["*MATERIAL, NAME=MAT", "*ELASTIC",
          f"{float(mat['E']):.10e}, {float(mat['nu']):.6f}"]
    if mat.get("density"):
        L += ["*DENSITY", f"{float(mat['density']):.6e}"]
    L.append("*SOLID SECTION, ELSET=EALL, MATERIAL=MAT")

    L += ["*STEP", "*STATIC"]

    for sup in load_case.get("supports", []):
        L += ["*BOUNDARY", f"{sup['region']}, 1, 3"]

    face_lut = None
    for load in load_case.get("loads", []):
        lt = str(load["type"]).lower()
        if lt == "force":
            nodes = fe.region(load["region"]).node_idx
            vec = np.asarray(load["vector"], dtype=np.float64) / len(nodes)
            L.append("*CLOAD")
            for d in range(3):
                if vec[d] != 0.0:
                    L.append(f"{load['region']}, {d+1}, {vec[d]:.12e}")
        elif lt == "pressure":
            if face_lut is None:
                face_lut = tet_face_lookup(fe)
            L.append("*DLOAD")
            for f in region_faces(fe, load["region"]):
                key = frozenset(int(v) for v in f)
                if key not in face_lut:
                    raise RuntimeError(f"face {sorted(key)} of region "
                                       f"{load['region']!r} is not a tet face")
                ei, fno = face_lut[key]
                L.append(f"{ei+1}, P{fno}, {float(load['magnitude']):.12e}")
        elif lt == "traction":
            # ccx has no vector-traction card for solids; apply the statically
            # equivalent consistent nodal forces computed from the same faces.
            f_nodal = _consistent_traction_forces(fe, load["region"],
                                                  np.asarray(load["vector"], float))
            L.append("*CLOAD")
            for nid, vec in sorted(f_nodal.items()):
                for d in range(3):
                    if abs(vec[d]) > 0.0:
                        L.append(f"{nid+1}, {d+1}, {vec[d]:.12e}")
        elif lt == "gravity":
            g = np.asarray(load["vector"], dtype=np.float64)
            gmag = float(np.linalg.norm(g))
            gdir = g / gmag
            L.append("*DLOAD")
            L.append(f"EALL, GRAV, {gmag:.10e}, {gdir[0]:.8f}, {gdir[1]:.8f}, {gdir[2]:.8f}")
        else:
            raise ValueError(f"cannot bake load type {lt!r}")

    L += ["*NODE PRINT, NSET=NALL", "U",
          "*EL PRINT, ELSET=EALL", "S",
          "*END STEP"]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


def bake_modal_inp(fe: FEMesh, load_case: dict, path: str, n_modes: int = 6) -> str:
    """A *FREQUENCY deck for the same (mesh, supports, material) — the M5 oracle."""
    etype = "C3D10" if fe.tet_type == "tetra10" else "C3D4"
    nn = 10 if etype == "C3D10" else 4
    mat = load_case["material"]
    if not mat.get("density"):
        raise ValueError("modal deck needs material.density")

    L: List[str] = ["*HEADING", f" {load_case.get('name','case')} modal (SI)"]
    L.append("*NODE, NSET=NALL")
    for i, p in enumerate(fe.points):
        L.append(f"{i+1}, {p[0]:.12e}, {p[1]:.12e}, {p[2]:.12e}")
    L.append(f"*ELEMENT, TYPE={etype}, ELSET=EALL")
    for ei, t in enumerate(fe.tets):
        L.append(f"{ei+1}, " + ", ".join(str(int(v) + 1) for v in t[:nn]))
    for sup in load_case.get("supports", []):
        ids = np.asarray(sorted(int(i) + 1 for i in fe.region(sup["region"]).node_idx))
        L.append(f"*NSET, NSET={sup['region']}")
        for k in range(0, len(ids), 8):
            L.append(", ".join(str(v) for v in ids[k:k + 8]))
    L += ["*MATERIAL, NAME=MAT", "*ELASTIC",
          f"{float(mat['E']):.10e}, {float(mat['nu']):.6f}",
          "*DENSITY", f"{float(mat['density']):.6e}",
          "*SOLID SECTION, ELSET=EALL, MATERIAL=MAT",
          "*STEP", "*FREQUENCY", f"{n_modes}"]
    for sup in load_case.get("supports", []):
        L += ["*BOUNDARY", f"{sup['region']}, 1, 3"]
    L += ["*END STEP"]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


_EIG_ROW = re.compile(rf"^\s*(\d+)\s+({_NUM})\s+({_NUM})\s+({_NUM})\s+({_NUM})\s*$")


def parse_dat_eigenfrequencies(dat_path: str) -> np.ndarray:
    """
    Hz from a ccx *FREQUENCY .dat.

    The block is
        MODE NO   EIGENVALUE   FREQUENCY(RAD/TIME)   (CYCLES/TIME)   IMAGINARY
    so column 4 is already the frequency in Hz.
    """
    out: List[float] = []
    if not os.path.exists(dat_path):
        return np.zeros(0)
    in_block = False
    with open(dat_path) as f:
        for line in f:
            low = line.lower()
            # ccx letter-spaces its banners ("E I G E N V A L U E   O U T P U T"),
            # so strip all whitespace before matching.
            squashed = "".join(low.split())
            if "eigenvalueoutput" in squashed:
                in_block = True
                continue
            if in_block and "participationfactors" in squashed:
                break                       # next banner: the block is done
            if in_block:
                m = _EIG_ROW.match(line)
                if m:
                    out.append(float(m.group(4).replace("D", "E")))
                elif line.strip() == "":
                    continue
                elif out:
                    in_block = False
    return np.array(out)


def bake_buckle_inp(fe: FEMesh, load_case: dict, path: str, n_modes: int = 4) -> str:
    """
    A *BUCKLE deck for the same (mesh, supports, reference load) — the M-buckling
    oracle. The reference load is applied as the identical consistent nodal forces
    the Warp traction/force path uses, so both codes buckle the same discrete load.
    """
    etype = "C3D10" if fe.tet_type == "tetra10" else "C3D4"
    nn = 10 if etype == "C3D10" else 4
    mat = load_case["material"]

    L: List[str] = ["*HEADING", f" {load_case.get('name','case')} buckling (SI)"]
    L.append("*NODE, NSET=NALL")
    for i, p in enumerate(fe.points):
        L.append(f"{i+1}, {p[0]:.12e}, {p[1]:.12e}, {p[2]:.12e}")
    L.append(f"*ELEMENT, TYPE={etype}, ELSET=EALL")
    for ei, t in enumerate(fe.tets):
        L.append(f"{ei+1}, " + ", ".join(str(int(v) + 1) for v in t[:nn]))
    for sup in load_case.get("supports", []):
        ids = np.asarray(sorted(int(i) + 1 for i in fe.region(sup["region"]).node_idx))
        L.append(f"*NSET, NSET={sup['region']}")
        for k in range(0, len(ids), 8):
            L.append(", ".join(str(v) for v in ids[k:k + 8]))
    L += ["*MATERIAL, NAME=MAT", "*ELASTIC",
          f"{float(mat['E']):.10e}, {float(mat['nu']):.6f}",
          "*SOLID SECTION, ELSET=EALL, MATERIAL=MAT",
          "*STEP", "*BUCKLE", f"{n_modes}"]
    for sup in load_case.get("supports", []):
        L += ["*BOUNDARY", f"{sup['region']}, 1, 3"]
    # reference load as consistent nodal forces (same as Warp's traction/force)
    cloads: Dict[int, np.ndarray] = {}
    for load in load_case.get("loads", []):
        lt = str(load["type"]).lower()
        if lt == "traction":
            f = _consistent_traction_forces(fe, load["region"],
                                            np.asarray(load["vector"], float))
        elif lt == "force":
            nodes = fe.region(load["region"]).node_idx
            per = np.asarray(load["vector"], float) / len(nodes)
            f = {int(nid): per for nid in nodes}
        else:
            raise ValueError(f"buckle bake supports traction/force, not {lt!r}")
        for nid, vec in f.items():
            cloads.setdefault(nid, np.zeros(3))
            cloads[nid] += vec
    L.append("*CLOAD")
    for nid, vec in sorted(cloads.items()):
        for d in range(3):
            if abs(vec[d]) > 0.0:
                L.append(f"{nid+1}, {d+1}, {vec[d]:.12e}")
    L += ["*END STEP"]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


def parse_dat_buckling_factors(dat_path: str) -> np.ndarray:
    """Buckling load factors from a ccx *BUCKLE .dat (letter-spaced banner)."""
    out: List[float] = []
    if not os.path.exists(dat_path):
        return np.zeros(0)
    in_block = False
    with open(dat_path) as f:
        for line in f:
            squashed = "".join(line.lower().split())
            if "bucklingfactoroutput" in squashed:
                in_block = True
                continue
            if in_block:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        out.append(float(parts[1].replace("D", "E")))
                    except ValueError:
                        if out:
                            break
                elif out:
                    break
    return np.array(out)


def _consistent_traction_forces(fe: FEMesh, region: str, t_vec: np.ndarray
                                ) -> Dict[int, np.ndarray]:
    """
    Consistent nodal forces for a uniform traction on a triangular surface region.
    Exact quadratic (tri6) / linear (tri3) integration of ∫ t·N_i dΓ.
    """
    r = fe.region(region)
    out: Dict[int, np.ndarray] = {}
    tri6 = r.cells.shape[1] >= 6
    for cell in r.cells:
        p = fe.points[cell[:3]]
        area = 0.5 * float(np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0])))
        if tri6:
            # exact ∫N dΓ for a straight-sided 6-node triangle:
            # corners 0, mid-side nodes area/3 each
            w = np.array([0.0, 0.0, 0.0, area / 3.0, area / 3.0, area / 3.0])
            nodes = cell[:6]
        else:
            w = np.full(3, area / 3.0)
            nodes = cell[:3]
        for n, wi in zip(nodes, w):
            if wi == 0.0:
                continue
            out.setdefault(int(n), np.zeros(3))
            out[int(n)] += wi * t_vec
    return out


# --------------------------------------------------------------------------- #
# ccx oracle
# --------------------------------------------------------------------------- #
def parse_dat_stresses(dat_path: str) -> np.ndarray:
    """Integration-point stresses (sxx,syy,szz,sxy,sxz,syz) from *EL PRINT, S."""
    rows: List[Tuple[float, ...]] = []
    if not os.path.exists(dat_path):
        return np.zeros((0, 6))
    in_block = False
    with open(dat_path) as f:
        for line in f:
            low = line.lower()
            if "stress" in low:
                in_block = True
                continue
            if in_block:
                m = _STRESS_ROW.match(line)
                if m:
                    rows.append(tuple(float(m.group(i).replace("D", "E"))
                                      for i in range(3, 9)))
                elif line.strip() == "":
                    continue
                elif rows:
                    in_block = False
    return np.array(rows) if rows else np.zeros((0, 6))


def von_mises_from_components(S: np.ndarray) -> np.ndarray:
    sxx, syy, szz, sxy, sxz, syz = S.T
    return np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                   + 3.0 * (sxy ** 2 + sxz ** 2 + syz ** 2))


def run_ccx_oracle_once(job_noext: str, n_threads: Optional[int] = None,
                        expect_nodes: Optional[int] = None,
                        expect_ip: Optional[int] = None) -> dict:
    """
    Run ccx once and extract the same measured scalars the Warp path reports.

    Prefer `run_ccx_oracle`, which runs this twice and demands agreement — see the
    note there. Stale outputs are wiped first and the parsed shape is checked
    against the mesh, so a leftover .dat can't silently poison the comparison.
    """
    if n_threads is None:
        n_threads = pod_cpu_quota()
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(n_threads)
    env["NUMBER_OF_CPUS"] = str(n_threads)
    workdir = os.path.dirname(os.path.abspath(job_noext)) or "."
    job = os.path.basename(job_noext)

    for ext in (".dat", ".frd", ".sta", ".cvg", ".12d"):
        try:
            os.remove(job_noext + ext)
        except FileNotFoundError:
            pass

    with measure(f"ccx[{n_threads}t]:{job}", interval=0.05, verbose=False) as m:
        proc = subprocess.run(["ccx", job], cwd=workdir, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    res = m["result"]
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        raise RuntimeError(f"ccx failed ({proc.returncode}) for {job}")
    for bad in ("*ERROR", "nonpositive jacobian", "singular"):
        if bad.lower() in proc.stdout.lower():
            raise RuntimeError(f"{job}: ccx reported {bad!r}; refusing to use it as "
                               f"an oracle\n{proc.stdout[-1500:]}")

    disp = parse_dat_displacements(job_noext + ".dat")
    if not disp:
        raise RuntimeError(f"{job}: no displacements parsed from .dat")
    if expect_nodes is not None and len(disp) != expect_nodes:
        raise RuntimeError(f"{job}: .dat has {len(disp)} displacement rows but the "
                           f"mesh has {expect_nodes} nodes — stale or mismatched .dat")
    d = np.array([v for v in disp.values()])
    ids = np.array(list(disp.keys()))
    mag = np.linalg.norm(d, axis=1)
    i = int(np.argmax(mag))

    S = parse_dat_stresses(job_noext + ".dat")
    if expect_ip is not None and len(S) != expect_ip:
        raise RuntimeError(f"{job}: .dat has {len(S)} integration-point stress rows, "
                           f"expected {expect_ip} — stale or mismatched .dat")
    vm = von_mises_from_components(S) if len(S) else np.zeros(1)

    return {
        "max_displacement": float(mag[i]),
        "max_disp_node": int(ids[i]),
        "max_von_mises": float(vm.max()),
        "n_ip": int(len(S)),
        "wall_s": res.wall_s,
        "cpu_avg": res.cpu_avg,
    }


class OracleUnstable(RuntimeError):
    """ccx returned two different answers for one deck — it cannot arbitrate."""


def run_ccx_oracle(job_noext: str, n_threads: Optional[int] = None,
                   expect_nodes: Optional[int] = None,
                   expect_ip: Optional[int] = None,
                   verify: bool = True) -> dict:
    """
    Get trustworthy measured values out of ccx — always **single-threaded**.

    **CalculiX 2.17's multi-threaded path is racy**, and not only for stress.
    ccx announces "Using up to N cpu(s) for the stress calculation", and a
    multi-threaded run intermittently writes corrupted output. Measured:

      * stress: peaks 15-80x too high (2.3e9 / 1.3e10 / 5.2e10 Pa vs a true
        2.9e8). On one deck: threads=28 -> 1/30 runs bad, threads=8 -> 4/30 bad.
      * displacement: also drifts — a 14-thread run of plate_with_hole gave
        5.4064e-6 where both 1-thread ccx and Warp give 5.4189e-6.
      * threads=1: 0/30, bit-identical every time.

    The corruption hides behind a normal-looking .dat (right row count, plausible
    magnitudes). For a Judge doing yield/stiffness checks that is exactly the
    input that fabricates a verdict. There is no reason to ever run the oracle
    multi-threaded — the values are what matter, so we take them from a single
    deterministic run. `verify` re-runs once more (still single-threaded) and
    requires agreement; single-thread ccx is deterministic, so this only ever
    fires if something new breaks.

    (`n_threads` is accepted for API compatibility but ignored for values.)
    """
    trusted = run_ccx_oracle_once(job_noext, 1, expect_nodes, expect_ip)
    if verify:
        again = run_ccx_oracle_once(job_noext, 1, expect_nodes, expect_ip)
        for key in ("max_displacement", "max_von_mises"):
            if _rel(again[key], trusted[key]) > 1e-9:
                raise OracleUnstable(
                    f"{os.path.basename(job_noext)}: single-threaded ccx gave "
                    f"{key}={trusted[key]:.8e} then {again[key]:.8e} on an identical "
                    f"deck; refusing to arbitrate with an unstable oracle"
                )
    trusted["value_threads"] = 1
    return trusted


# --------------------------------------------------------------------------- #
# parity
# --------------------------------------------------------------------------- #
def _rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b != 0 else float("nan")


def validate_case(name: str, order: int = 2, outdir: str = "fea_cases",
                  tol: float = 1e-10, verbose: bool = True) -> dict:
    msh, load_case = ALL_CASES[name](outdir=outdir, order=order)
    fe = read_mesh(msh)

    warp_res: FEAResult = solve_structural(msh, load_case, tol=tol)

    job = os.path.join(outdir, f"{name}_o{order}")
    bake_inp(fe, load_case, job + ".inp")
    # ccx integrates C3D10 at 4 points and C3D4 at 1
    orc = run_ccx_oracle(job, expect_nodes=fe.n_nodes,
                         expect_ip=fe.n_elems * (4 if fe.order == 2 else 1))

    st = warp_res.solver_status
    if not st.converged:
        row = {"case": name, "order": order, "n_dof": st.n_dof,
               "converged": False, "pass": False, "note": st.message}
        if verbose:
            print(f"  !! {name} o{order}: NOT CONVERGED — {st.message}")
        return row

    w_disp = warp_res.measured["max_displacement"].value
    w_vm = warp_res.measured["max_von_mises_stress"].value
    e_disp = _rel(w_disp, orc["max_displacement"])
    e_vm = _rel(w_vm, orc["max_von_mises"])
    loc = warp_res.measured["max_von_mises_stress"].location

    row = {
        "case": name, "order": order, "element": st.element,
        "n_dof": st.n_dof, "n_elems": fe.n_elems, "n_iters": st.n_iters,
        "converged": True,
        "warp_max_disp_m": w_disp, "ccx_max_disp_m": orc["max_displacement"],
        "rel_err_disp": e_disp,
        "warp_max_vm_Pa": w_vm, "ccx_max_vm_Pa": orc["max_von_mises"],
        "rel_err_vm": e_vm,
        "vm_region": loc.region, "vm_coords": [round(c, 6) for c in loc.coords],
        "warp_wall_s": st.wall_time_s, "ccx_wall_s": orc["wall_s"],
        "speedup": orc["wall_s"] / st.wall_time_s if st.wall_time_s > 0 else float("inf"),
        "warp_cpu_pct": st.cpu_avg_pct, "warp_gpu_pct": st.gpu_util_avg_pct,
        "disp_pass": bool(e_disp <= TOL_DISP),
        "vm_pass": bool(e_vm <= TOL_VM),
        "cpu_pass": bool(st.cpu_avg_pct < 20.0),
    }
    row["pass"] = bool(row["disp_pass"] and row["vm_pass"] and row["cpu_pass"])
    if verbose:
        print(f"  {name:18s} o{order} dof={st.n_dof:>7} | "
              f"disp {w_disp:.6e} vs {orc['max_displacement']:.6e} "
              f"({e_disp:.2e}) {'OK' if row['disp_pass'] else 'FAIL'} | "
              f"vM {w_vm:.4e} vs {orc['max_von_mises']:.4e} "
              f"({e_vm*100:.2f}%) {'OK' if row['vm_pass'] else 'FAIL'} | "
              f"cpu {st.cpu_avg_pct:.1f}% gpu {st.gpu_util_avg_pct} | "
              f"{row['speedup']:.1f}x")
    return row


def write_parity(rows: List[dict], path: str = "measured_parity.csv") -> None:
    cols = ["case", "order", "element", "n_dof", "n_elems", "n_iters", "converged",
            "warp_max_disp_m", "ccx_max_disp_m", "rel_err_disp",
            "warp_max_vm_Pa", "ccx_max_vm_Pa", "rel_err_vm", "vm_region",
            "warp_wall_s", "ccx_wall_s", "speedup", "warp_cpu_pct", "warp_gpu_pct",
            "disp_pass", "vm_pass", "cpu_pass", "pass"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])
    print(f"\nwrote {path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None, choices=list(ALL_CASES))
    ap.add_argument("--order", type=int, default=None, choices=(1, 2))
    ap.add_argument("--out", default="measured_parity.csv")
    args = ap.parse_args()

    names = [args.case] if args.case else list(ALL_CASES)
    orders = [args.order] if args.order else [1, 2]

    rows = []
    print(f"# ccx threads = {pod_cpu_quota()} (cgroup quota)\n")
    for n in names:
        for o in orders:
            rows.append(validate_case(n, order=o))
    write_parity(rows, args.out)

    ok = all(r.get("pass") for r in rows)
    print("\n" + "=" * 60)
    print(f"MEASURED PARITY vs ccx: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    print("=" * 60)
