"""
Phase 1 driver — run CalculiX (ccx) under the instrumentation harness.

Ties Phase 0 + Phase 1 together:
  - runs `ccx <job>` with OMP_NUM_THREADS set to all cores  (the "CPU 100%" baseline
    for Req 2)
  - measures its wall-clock + CPU utilization via bench.instrument.measure
  - parses the .dat file for the printed displacement block, so Phase 4 can compare
    ccx displacements against the Warp solver's.

Usage:
    from bench.run_ccx import run_ccx
    res, disp = run_ccx("cases/cant_20x4x4_iterative")
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Dict, Optional, Tuple

from .instrument import measure, MeasureResult

_NUM = r"[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?"
_ROW = re.compile(rf"^\s*(\d+)\s+({_NUM})\s+({_NUM})\s+({_NUM})\s*$")


def _f(token: str) -> float:
    # CalculiX sometimes writes Fortran-style 'D' exponents.
    return float(token.replace("D", "E").replace("d", "e"))


def parse_dat_displacements(dat_path: str) -> Dict[int, Tuple[float, float, float]]:
    """
    Extract the nodal displacement block written by *NODE PRINT, U.
    Returns { node_id: (ux, uy, uz) }. Tolerant of spacing / header variations.
    """
    disp: Dict[int, Tuple[float, float, float]] = {}
    if not os.path.exists(dat_path):
        return disp
    in_block = False
    with open(dat_path) as f:
        for line in f:
            low = line.lower()
            if "displacement" in low:      # header line starts a U block
                in_block = True
                continue
            if in_block:
                m = _ROW.match(line)
                if m:
                    nid = int(m.group(1))
                    disp[nid] = (_f(m.group(2)), _f(m.group(3)), _f(m.group(4)))
                elif line.strip() == "":
                    continue               # blank lines inside a block are fine
                elif disp:
                    in_block = False       # non-matching, non-blank -> block ended
    return disp


def max_abs_disp(disp: Dict[int, Tuple[float, float, float]]) -> float:
    """Peak |u| magnitude across the printed set — a quick scalar to compare."""
    best = 0.0
    for (ux, uy, uz) in disp.values():
        best = max(best, (ux * ux + uy * uy + uz * uz) ** 0.5)
    return best


def run_ccx(
    job_noext: str,
    ccx_bin: str = "ccx",
    n_threads: Optional[int] = None,
    interval: float = 0.05,
) -> Tuple[MeasureResult, Dict[int, Tuple[float, float, float]]]:
    """
    Run `ccx <job_noext>` (expects <job_noext>.inp to exist) with all CPU cores,
    measured by the harness. Returns (MeasureResult, displacements).
    """
    inp = job_noext + ".inp"
    if not os.path.exists(inp):
        raise FileNotFoundError(inp)

    if n_threads is None:
        n_threads = os.cpu_count() or 1

    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(n_threads)   # ccx uses OpenMP for assembly/solve
    env["NUMBER_OF_CPUS"] = str(n_threads)    # CalculiX-specific override (see manual)

    workdir = os.path.dirname(os.path.abspath(job_noext)) or "."
    job = os.path.basename(job_noext)

    with measure(f"ccx[{n_threads}t]:{job}", interval=interval) as m:
        proc = subprocess.run(
            [ccx_bin, job],
            cwd=workdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    res: MeasureResult = m["result"]

    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        raise RuntimeError(f"ccx exited with code {proc.returncode} for job {job}")

    disp = parse_dat_displacements(job_noext + ".dat")
    print(f"    parsed {len(disp)} displacement rows; max|u| = {max_abs_disp(disp):.6e}")
    return res, disp


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m bench.run_ccx <job_without_extension> [ccx_binary]")
        raise SystemExit(1)
    binary = sys.argv[2] if len(sys.argv) > 2 else "ccx"
    r, d = run_ccx(sys.argv[1], ccx_bin=binary)
    print(r.summary())
