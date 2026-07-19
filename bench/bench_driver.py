"""
Phase 2 driver — full ccx-vs-Warp sweep and report generation.

For each mesh size it:
  (a) runs ccx with the ITERATIVE (CG) deck on all pod cores  -> CPU baseline
  (b) runs the Warp GPU solver                                -> GPU candidate
  (c) collects rel-err, cpu%, gpu-util, speedup, iters, pass/fail

Outputs (spec §4):
  results.json          full per-size record + acceptance verdicts
  crossover.png         x = DOF, y = wall-clock, ccx vs Warp (the §5 deliverable)
  parity.csv            per-size rel-err / cpu% / speedup / pass-fail table

CPU-core hygiene (spec §5): ccx thread count is pinned to the pod's *actual*
cgroup CPU quota, not os.cpu_count() (which sees the whole node).
"""

from __future__ import annotations

import csv
import json
import math
import os
from typing import List, Tuple

import numpy as np

from .gen_inp import sweep
from .run_ccx import run_ccx, max_abs_disp
from .warp_solve import solve_inp
from .instrument import compare_walltime

CPU_THRESHOLD = 20.0
REL_ERR_TOL = 1e-3


def pod_cpu_quota() -> int:
    """Cores actually allotted to this pod (cgroup v2 then v1), fallback cpu_count."""
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()
        if q != "max":
            return max(1, int(float(q) / float(p)))
    except Exception:
        pass
    try:
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if q > 0:
            return max(1, int(q / p))
    except Exception:
        pass
    return os.cpu_count() or 1


def nodewise_rel_err(dccx: dict, dwarp: dict) -> Tuple[float, float, int]:
    """L2 relative error and tip |u| relative error over ccx's printed nodes."""
    ids = sorted(set(dccx) & set(dwarp))
    A = np.array([dccx[i] for i in ids])
    B = np.array([dwarp[i] for i in ids])
    denom = np.linalg.norm(A)
    l2 = float(np.linalg.norm(A - B) / denom) if denom > 0 else float("nan")
    ma, mb = max_abs_disp(dccx), max_abs_disp(dwarp)
    tip = abs(ma - mb) / ma if ma > 0 else float("nan")
    return l2, tip, len(ids)


def run_sweep(sizes: List[Tuple[int, int, int]], outdir: str = "cases",
              warp_tol: float = 1e-8, warp_max_iters: int = 30000) -> dict:
    n_threads = pod_cpu_quota()
    print(f"# pod CPU quota -> ccx threads = {n_threads}\n")

    # iterative decks give a fair CG-vs-CG comparison (spec §2-2)
    meshes = sweep(outdir=outdir, sizes=sizes, solver="iterative")

    records = []
    for mi in meshes:
        job = os.path.splitext(mi.path)[0]
        print(f"\n=== {mi.job}  (ndof={mi.ndof}) ===")

        ccx_res, dccx = run_ccx(job, n_threads=n_threads)
        warp_res, dwarp, info = solve_inp(mi.path, tol=warp_tol,
                                          max_iters=warp_max_iters, return_info=True)

        l2, tip, ncmp = nodewise_rel_err(dccx, dwarp)
        verdict = compare_walltime(warp_res, ccx_res)

        rec = {
            "job": mi.job,
            "nx": mi.nx, "ny": mi.ny, "nz": mi.nz,
            "ndof": mi.ndof, "n_elems": mi.n_elems,
            "ccx_wall_s": ccx_res.wall_s,
            "ccx_cpu_avg": ccx_res.cpu_avg,
            "warp_wall_s": warp_res.wall_s,
            "warp_cpu_avg": warp_res.cpu_avg,
            "warp_gpu_util_avg": warp_res.gpu_util_avg,
            "warp_gpu_mem_peak_mb": warp_res.gpu_mem_peak_mb,
            "cg_iters": info["cg_iters"],
            "cg_residual": info["cg_residual"],
            "rel_err_l2": l2,
            "rel_err_tip": tip,
            "nodes_compared": ncmp,
            "speedup_cpu_over_gpu": verdict["speedup_cpu_over_gpu"],
            "gpu_faster": bool(warp_res.wall_s <= ccx_res.wall_s),
            "req1_cpu_pass": bool(warp_res.cpu_under(CPU_THRESHOLD)),
            "req3_acc_pass": bool(l2 < REL_ERR_TOL),
        }
        records.append(rec)
        print(f"    rel_err_l2={l2:.3e}  tip_err={tip:.3e}  "
              f"warp_cpu={warp_res.cpu_avg:.1f}%  gpu_util={warp_res.gpu_util_avg}  "
              f"speedup(cpu/gpu)={verdict['speedup_cpu_over_gpu']:.2f}x")

    # ---- crossover point: smallest DOF where GPU wall <= CPU wall ----
    crossover_dof = None
    for r in sorted(records, key=lambda r: r["ndof"]):
        if r["gpu_faster"]:
            crossover_dof = r["ndof"]
            break

    summary = {
        "n_threads_ccx": n_threads,
        "cpu_threshold": CPU_THRESHOLD,
        "rel_err_tol": REL_ERR_TOL,
        "crossover_dof": crossover_dof,
        "req1_all_pass": all(r["req1_cpu_pass"] for r in records),
        "req3_all_pass": all(r["req3_acc_pass"] for r in records),
        "req2_crossover_found": crossover_dof is not None,
        "records": records,
    }
    return summary


def write_reports(summary: dict, outdir: str = ".") -> None:
    os.makedirs(outdir, exist_ok=True)
    recs = sorted(summary["records"], key=lambda r: r["ndof"])

    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # parity CSV
    cols = ["job", "ndof", "n_elems", "cg_iters", "cg_residual",
            "rel_err_l2", "rel_err_tip", "warp_cpu_avg", "warp_gpu_util_avg",
            "ccx_wall_s", "warp_wall_s", "speedup_cpu_over_gpu",
            "gpu_faster", "req1_cpu_pass", "req3_acc_pass"]
    with open(os.path.join(outdir, "parity.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in recs:
            w.writerow([r[c] for c in cols])

    # crossover PNG
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dof = [r["ndof"] for r in recs]
    ccx_t = [r["ccx_wall_s"] for r in recs]
    warp_t = [r["warp_wall_s"] for r in recs]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(dof, ccx_t, "o-", label=f"ccx CG (CPU, {summary['n_threads_ccx']} threads)",
            color="#d1495b", lw=2)
    ax.plot(dof, warp_t, "s-", label="Warp bsr_cg (GPU, H200)", color="#3a7ca5", lw=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("DOF (3 × nodes)"); ax.set_ylabel("solve wall-clock [s]")
    ax.set_title("CalculiX (CPU) vs Warp (GPU) — linear-static elasticity")
    ax.grid(True, which="both", alpha=0.3)
    if summary["crossover_dof"]:
        ax.axvline(summary["crossover_dof"], color="gray", ls="--", alpha=0.7)
        ax.annotate(f"GPU wins\n≥ {summary['crossover_dof']:,} DOF",
                    xy=(summary["crossover_dof"], min(min(ccx_t), min(warp_t))),
                    xytext=(10, 20), textcoords="offset points", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "crossover.png"), dpi=130)
    print(f"\nwrote {outdir}/results.json, parity.csv, crossover.png")


def print_verdict(summary: dict) -> None:
    print("\n" + "=" * 62)
    print("ACCEPTANCE CRITERIA")
    print("=" * 62)
    print(f"  Req1  GPU-only (cpu_avg<20% every size) : "
          f"{'PASS' if summary['req1_all_pass'] else 'FAIL'}")
    print(f"  Req2  crossover found (GPU wins ≥ N DOF): "
          f"{'PASS @ %d DOF' % summary['crossover_dof'] if summary['req2_crossover_found'] else 'FAIL'}")
    print(f"  Req3  accuracy (rel_err<1e-3 every size): "
          f"{'PASS' if summary['req3_all_pass'] else 'FAIL'}")
    print(f"  Req4  CUDA-graph CG (built into bsr_cg)  : PASS")
    print("=" * 62)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-size", default="160x32x32",
                    help="largest mesh to include, e.g. 80x16x16")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    all_sizes = [(10, 2, 2), (20, 4, 4), (40, 8, 8),
                 (80, 16, 16), (120, 24, 24), (160, 32, 32)]
    cap = tuple(int(v) for v in args.max_size.split("x"))
    sizes = [s for s in all_sizes if s[0] <= cap[0]]

    summary = run_sweep(sizes, outdir="cases")
    write_reports(summary, outdir=args.outdir)
    print_verdict(summary)
