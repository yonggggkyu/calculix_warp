"""
PHASE3_SPEC §3 acceptance criteria, run as one suite.

  1  measured parity vs ccx      -> warp_fea.validate (all cases)
  2  SI unit contract            -> here (scaling round-trip)
  3  GPU-only, cpu_avg < 20%     -> here (asserted per case, also in validate)
  4  honest non-convergence      -> here (starve the CG, demand converged=False)
  5  contract shape (§1.2)       -> here (schema/key/unit check)
  6  modal vs ccx *FREQUENCY     -> M5, reported as skipped unless implemented

Run:  python -m warp_fea.acceptance
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List

import numpy as np

from .cases import ALL_CASES, cantilever_beam
from .mesh_io import read_mesh
from .results import FEAResult
from .solver import solve_structural
from .validate import TOL_DISP, TOL_VM, validate_case, write_parity

CPU_LIMIT = 20.0


def _ok(flag: bool) -> str:
    return "PASS" if flag else "FAIL"


# --------------------------------------------------------------------------- #
# §3-2 SI unit contract
# --------------------------------------------------------------------------- #
def check_si_units() -> dict:
    """
    Linear elasticity is exactly scale-invariant in a way we can exploit:
    doubling E halves displacement and leaves stress unchanged; doubling the load
    doubles both. If the solver silently carried a Phase-2-style MPa/mm habit, or
    mixed unit systems anywhere, these identities break.
    """
    msh, lc = cantilever_beam(order=2)
    base = solve_structural(msh, lc, tol=1e-10)
    u0 = base.measured["max_displacement"].value
    s0 = base.measured["max_von_mises_stress"].value

    lc_2E = json.loads(json.dumps(lc))
    lc_2E["material"]["E"] *= 2.0
    r_2E = solve_structural(msh, lc_2E, tol=1e-10)

    lc_2F = json.loads(json.dumps(lc))
    lc_2F["loads"][0]["vector"] = [2 * v for v in lc["loads"][0]["vector"]]
    r_2F = solve_structural(msh, lc_2F, tol=1e-10)

    e_uE = abs(r_2E.measured["max_displacement"].value / (u0 / 2) - 1)
    e_sE = abs(r_2E.measured["max_von_mises_stress"].value / s0 - 1)
    e_uF = abs(r_2F.measured["max_displacement"].value / (2 * u0) - 1)
    e_sF = abs(r_2F.measured["max_von_mises_stress"].value / (2 * s0) - 1)

    units_ok = (base.measured["max_displacement"].unit == "m"
                and base.measured["max_von_mises_stress"].unit == "Pa")
    # sanity band: a 0.2 m steel beam under 2 kN must deflect ~mm and stress ~100s MPa
    magnitude_ok = (1e-5 < u0 < 1e-1) and (1e6 < s0 < 1e10)

    passed = bool(max(e_uE, e_sE, e_uF, e_sF) < 1e-6 and units_ok and magnitude_ok)
    print(f"  [§3-2] SI units: u={u0:.4e} m  sigma={s0:.4e} Pa")
    print(f"         2xE -> u/2 (err {e_uE:.2e}), sigma const (err {e_sE:.2e})")
    print(f"         2xF -> 2u   (err {e_uF:.2e}), 2 sigma      (err {e_sF:.2e})")
    print(f"         units m/Pa: {units_ok}   magnitude sane: {magnitude_ok}   -> {_ok(passed)}")
    return {"criterion": "si_units", "pass": passed,
            "err_scale_E_u": e_uE, "err_scale_E_s": e_sE,
            "err_scale_F_u": e_uF, "err_scale_F_s": e_sF,
            "u_m": u0, "sigma_Pa": s0}


# --------------------------------------------------------------------------- #
# §3-4 honest non-convergence
# --------------------------------------------------------------------------- #
def check_non_convergence() -> dict:
    """
    Starve the CG (absurd tol, 5 iterations) and require the solver to *say so*:
    converged=False and an empty measured dict. A Judge false-pass caused by a
    silently unconverged solve is the failure mode this whole contract exists to
    prevent, so this is a hard criterion, not a nicety.
    """
    msh, lc = cantilever_beam(order=2)
    r = solve_structural(msh, lc, tol=1e-16, max_iters=5)
    flagged = (r.solver_status.converged is False)
    withheld = (len(r.measured) == 0)
    explained = bool(r.solver_status.message)
    # and the healthy run must still report converged=True
    r_ok = solve_structural(msh, lc, tol=1e-10)
    healthy = r_ok.solver_status.converged and len(r_ok.measured) == 2

    passed = bool(flagged and withheld and explained and healthy)
    print(f"  [§3-4] starved solve (tol=1e-16, max_iters=5):")
    print(f"         converged={r.solver_status.converged}  measured_keys={list(r.measured)}")
    print(f"         message: {r.solver_status.message[:70]}")
    print(f"         healthy run still converges & reports: {healthy}   -> {_ok(passed)}")
    return {"criterion": "non_convergence", "pass": passed,
            "flagged": flagged, "measured_withheld": withheld,
            "healthy_ok": healthy}


# --------------------------------------------------------------------------- #
# §3-5 contract shape
# --------------------------------------------------------------------------- #
def check_contract_shape() -> dict:
    """The returned dict must match §1.2 exactly: keys, units, location, status."""
    msh, lc = ALL_CASES["hccx_bracket"](order=2)
    r = solve_structural(msh, lc, tol=1e-10)
    d = r.as_dict()
    problems: List[str] = []

    for k, unit in (("max_von_mises_stress", "Pa"), ("max_displacement", "m")):
        if k not in d["measured"]:
            problems.append(f"measured.{k} missing")
            continue
        m = d["measured"][k]
        if m.get("unit") != unit:
            problems.append(f"measured.{k}.unit={m.get('unit')!r} != {unit!r}")
        if not isinstance(m.get("value"), float):
            problems.append(f"measured.{k}.value not a float")
        loc = m.get("location")
        if not loc or len(loc.get("coords", [])) != 3:
            problems.append(f"measured.{k}.location.coords malformed")
        if loc and loc.get("node_id") is None:
            problems.append(f"measured.{k}.location.node_id missing")
    vm_region = d["measured"].get("max_von_mises_stress", {}).get("location", {}).get("region")
    if vm_region is None:
        problems.append("max_von_mises_stress.location.region is null (Stage 5 needs it)")

    st = d["solver_status"]
    for key, typ in (("regime", str), ("backend", str), ("meshed", bool),
                     ("converged", bool), ("n_dof", int), ("n_iters", int),
                     ("wall_time_s", float), ("cpu_avg_pct", float),
                     ("device", str)):
        if key not in st:
            problems.append(f"solver_status.{key} missing")
        elif not isinstance(st[key], typ):
            problems.append(f"solver_status.{key} is {type(st[key]).__name__}, want {typ.__name__}")
    if st.get("regime") != "solid" or st.get("backend") != "warp-gpu":
        problems.append("solver_status.regime/backend wrong")
    if "von_mises" not in r.fields or "displacement" not in r.fields:
        problems.append("fields.{von_mises,displacement} missing")
    if not json.dumps(d):                       # must be JSON-serialisable for the Judge
        problems.append("result is not JSON-serialisable")

    passed = not problems
    print(f"  [§3-5] contract shape on hccx_bracket:")
    print(f"         measured keys: {list(d['measured'])}")
    print(f"         vM location: region={vm_region} node_id="
          f"{d['measured']['max_von_mises_stress']['location']['node_id']}")
    print(f"         status: converged={st['converged']} n_dof={st['n_dof']} "
          f"device={st['device']!r} gpu={st.get('gpu_util_avg_pct')}")
    for p in problems:
        print(f"         !! {p}")
    print(f"         -> {_ok(passed)}")
    return {"criterion": "contract_shape", "pass": passed, "problems": problems,
            "example": d}


# --------------------------------------------------------------------------- #
# §3-6 modal (M5)
# --------------------------------------------------------------------------- #
def check_modal(n_modes: int = 10, order: int = 2) -> dict:
    """
    Lowest eigenfrequencies vs a ccx *FREQUENCY deck on the same mesh, <= 2%.

    Run on C3D10, the Judge's solid path (spec §0). On C3D4 the first torsional
    mode misses by ~2%: linear tets are far too stiff in torsion (they put it at
    ~4.4 kHz against a 3.55 kHz reference), and the two codes' versions of that
    same pathology don't coincide. That's an element-order artefact, not a solver
    discrepancy — every other C3D4 mode agrees to <0.7%, and on C3D10 the whole
    spectrum agrees to ~1e-5.
    """
    import subprocess
    from .modal import solve_modal
    from .validate import (bake_modal_inp, parse_dat_eigenfrequencies,
                           pod_cpu_quota)

    msh, lc = cantilever_beam(order=order)
    fe = read_mesh(msh)
    r = solve_modal(msh, lc, n_modes=n_modes)
    if not r.solver_status.converged:
        print(f"  [§3-6] modal did NOT converge: {r.solver_status.message}")
        return {"criterion": "modal", "pass": False,
                "note": r.solver_status.message}
    fw = np.array(r.measured["eigenfrequencies"])

    job = os.path.join("fea_cases", f"modal_o{order}")
    bake_modal_inp(fe, lc, job + ".inp", n_modes=n_modes)
    # single-threaded: CalculiX 2.17's parallel stress path is racy (see
    # validate.run_ccx_oracle), so the oracle never runs multi-threaded.
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["NUMBER_OF_CPUS"] = "1"
    for ext in (".dat", ".frd", ".sta", ".cvg"):
        try:
            os.remove(job + ext)
        except FileNotFoundError:
            pass
    subprocess.run(["ccx", os.path.basename(job)], cwd="fea_cases", env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    fc = parse_dat_eigenfrequencies(job + ".dat")
    n = min(n_modes, len(fc))
    if n == 0:
        return {"criterion": "modal", "pass": False,
                "note": "no eigenfrequencies parsed from the ccx oracle"}
    err = np.abs(fw[:n] - fc[:n]) / fc[:n]
    passed = bool(err.max() <= 0.02)
    st = r.solver_status
    print(f"  [§3-6] modal, {st.element}, n_dof={st.n_dof}, "
          f"{st.n_iters} subspace sweeps, {st.wall_time_s:.2f}s, "
          f"cpu={st.cpu_avg_pct:.1f}% gpu={st.gpu_util_avg_pct}")
    print(f"         warp Hz: {np.round(fw[:min(6, n)], 2).tolist()}")
    print(f"         ccx  Hz: {np.round(fc[:min(6, n)], 2).tolist()}")
    print(f"         max rel err over {n} modes = {err.max():.3e}  -> {_ok(passed)}")
    return {"criterion": "modal", "pass": passed, "order": order,
            "max_rel_err": float(err.max()),
            "warp_hz": [float(x) for x in fw[:n]],
            "ccx_hz": [float(x) for x in fc[:n]]}


if __name__ == "__main__":
    print("=" * 66)
    print("PHASE 3 ACCEPTANCE (PHASE3_SPEC §3)")
    print("=" * 66)

    print("\n[§3-1/3-3] measured parity vs ccx + GPU-only (all cases, both orders)")
    rows = []
    for name in ALL_CASES:
        for order in (1, 2):
            rows.append(validate_case(name, order=order))
    write_parity(rows, "measured_parity.csv")
    parity_pass = all(r.get("pass") for r in rows)
    cpu_pass = all(r.get("cpu_pass") for r in rows)

    print()
    si = check_si_units()
    print()
    nc = check_non_convergence()
    print()
    cs = check_contract_shape()
    print()
    md = check_modal()

    results = {
        "req1_measured_parity": parity_pass,
        "req2_si_units": si["pass"],
        "req3_gpu_only_cpu20": cpu_pass,
        "req4_non_convergence": nc["pass"],
        "req5_contract_shape": cs["pass"],
        "req6_modal_m5": md["pass"],
    }
    with open("acceptance.json", "w") as f:
        json.dump({"summary": results, "parity": rows,
                   "si": si, "non_convergence": nc, "modal": md,
                   "contract_example": cs["example"]}, f, indent=2, default=str)

    print("\n" + "=" * 66)
    print("ACCEPTANCE SUMMARY")
    print("=" * 66)
    labels = {
        "req1_measured_parity": "§3-1 measured == ccx (disp<=1e-3, vM<=3%)",
        "req2_si_units":        "§3-2 SI unit contract (Pa/m round-trip)",
        "req3_gpu_only_cpu20":  "§3-3 GPU-only (cpu_avg < 20%), fp64",
        "req4_non_convergence": "§3-4 honest converged=false",
        "req5_contract_shape":  "§3-5 §1.2 return schema",
        "req6_modal_m5":        "§3-6 modal vs *FREQUENCY (M5)",
    }
    for k, v in results.items():
        mark = "SKIP (M5)" if v is None else _ok(v)
        print(f"  {labels[k]:<46s} {mark}")
    core = [results[k] for k in ("req1_measured_parity", "req2_si_units",
                                 "req3_gpu_only_cpu20", "req4_non_convergence",
                                 "req5_contract_shape")]
    print("=" * 66)
    print(f"  M1-M4 CORE: {'ALL GREEN' if all(core) else 'FAILURES PRESENT'}")
    print(f"  M5 MODAL  : {'GREEN' if results['req6_modal_m5'] else 'not green'}")
    print("=" * 66)
    print("wrote acceptance.json, measured_parity.csv")
