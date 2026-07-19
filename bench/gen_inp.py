"""
Phase 1 — Parametric CalculiX (.inp) generator.

Builds a structured cantilever beam meshed with C3D8 (8-node linear hex) elements:
  - one end (x = 0) fully clamped
  - a transverse point load spread over the far-end (x = L) face, in -z

The whole point is the *mesh-size sweep*: `sweep()` emits a series of meshes of
growing DOF count. DOF is the x-axis of the crossover curve
(ccx-CPU wall-clock  vs  Warp-GPU wall-clock), so the same .inp files feed both
the CalculiX reference run (Phase 1) and the Warp solver (Phase 2), guaranteeing
an apples-to-apples comparison.

Node ordering follows the Abaqus/CalculiX C3D8 convention:
  bottom face (local z=0): n1..n4 CCW,  top face (local z=1): n5..n8 CCW.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

# Solver keyword to place on the *STATIC card. "default" leaves it off (SPOOLES).
# Use "iterative" for a fair CG-vs-CG comparison against the Warp bsr_cg solver.
_SOLVER_CARD = {
    "default": "*STATIC",
    "spooles": "*STATIC",
    "pardiso": "*STATIC, SOLVER=PARDISO",
    "pastix": "*STATIC, SOLVER=PASTIX",
    "iterative": "*STATIC, SOLVER=ITERATIVE CHOLESKY",
}


@dataclass
class MeshInfo:
    path: str
    job: str          # job name without extension (what you pass to `ccx`)
    nx: int
    ny: int
    nz: int
    n_nodes: int
    n_elems: int
    ndof: int


def _nid(i: int, j: int, k: int, nx: int, ny: int) -> int:
    """1-based node id for grid index (i,j,k)."""
    return i + j * (nx + 1) + k * (nx + 1) * (ny + 1) + 1


def generate_cantilever(
    nx: int,
    ny: int,
    nz: int,
    L: float = 10.0,
    W: float = 1.0,
    H: float = 1.0,
    E: float = 210000.0,
    nu: float = 0.3,
    total_force: float = -100.0,
    solver: str = "default",
) -> str:
    """Return the full .inp text for a clamped-cantilever C3D8 mesh."""
    if solver not in _SOLVER_CARD:
        raise ValueError(f"solver must be one of {list(_SOLVER_CARD)}")

    lines: List[str] = []
    lines.append("*HEADING")
    lines.append(f" Cantilever C3D8  nx={nx} ny={ny} nz={nz}  solver={solver}")

    # ---- nodes ----
    lines.append("*NODE, NSET=NALL")
    for k in range(nz + 1):
        z = H * k / nz
        for j in range(ny + 1):
            y = W * j / ny
            for i in range(nx + 1):
                x = L * i / nx
                nid = _nid(i, j, k, nx, ny)
                lines.append(f"{nid}, {x:.6f}, {y:.6f}, {z:.6f}")

    # ---- elements (C3D8) ----
    lines.append("*ELEMENT, TYPE=C3D8, ELSET=EALL")
    eid = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                eid += 1
                n1 = _nid(i,     j,     k,     nx, ny)
                n2 = _nid(i + 1, j,     k,     nx, ny)
                n3 = _nid(i + 1, j + 1, k,     nx, ny)
                n4 = _nid(i,     j + 1, k,     nx, ny)
                n5 = _nid(i,     j,     k + 1, nx, ny)
                n6 = _nid(i + 1, j,     k + 1, nx, ny)
                n7 = _nid(i + 1, j + 1, k + 1, nx, ny)
                n8 = _nid(i,     j + 1, k + 1, nx, ny)
                lines.append(
                    f"{eid}, {n1}, {n2}, {n3}, {n4}, {n5}, {n6}, {n7}, {n8}"
                )

    # ---- node sets: clamped end (i==0), loaded end (i==nx) ----
    fixed = [
        _nid(0, j, k, nx, ny)
        for k in range(nz + 1)
        for j in range(ny + 1)
    ]
    loaded = [
        _nid(nx, j, k, nx, ny)
        for k in range(nz + 1)
        for j in range(ny + 1)
    ]

    def _nset(name: str, ids: List[int]) -> None:
        lines.append(f"*NSET, NSET={name}")
        for chunk_start in range(0, len(ids), 8):
            lines.append(", ".join(str(n) for n in ids[chunk_start:chunk_start + 8]))

    _nset("NFIX", fixed)
    _nset("NLOAD", loaded)

    # ---- material / section ----
    lines.append("*MATERIAL, NAME=STEEL")
    lines.append("*ELASTIC")
    lines.append(f"{E:.6g}, {nu:.6g}")
    lines.append("*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL")

    # ---- step ----
    per_node = total_force / len(loaded)
    lines.append("*STEP")
    lines.append(_SOLVER_CARD[solver])
    lines.append("*BOUNDARY")
    lines.append("NFIX, 1, 3")                 # clamp all 3 translations
    lines.append("*CLOAD")
    lines.append(f"NLOAD, 3, {per_node:.8g}")  # transverse load in z
    lines.append("*NODE PRINT, NSET=NLOAD")    # -> writes U block to .dat
    lines.append("U")
    lines.append("*NODE FILE")                 # -> writes U to .frd for postprocessing
    lines.append("U")
    lines.append("*END STEP")

    return "\n".join(lines) + "\n"


def write_cantilever(outdir: str, job: str, nx: int, ny: int, nz: int,
                     solver: str = "default", **kw) -> MeshInfo:
    os.makedirs(outdir, exist_ok=True)
    text = generate_cantilever(nx, ny, nz, solver=solver, **kw)
    path = os.path.join(outdir, f"{job}.inp")
    with open(path, "w") as f:
        f.write(text)
    n_nodes = (nx + 1) * (ny + 1) * (nz + 1)
    n_elems = nx * ny * nz
    return MeshInfo(path, job, nx, ny, nz, n_nodes, n_elems, ndof=3 * n_nodes)


def sweep(outdir: str = "cases",
          sizes: List[Tuple[int, int, int]] = None,
          solver: str = "default",
          **kw) -> List[MeshInfo]:
    """
    Emit a family of meshes of growing DOF count. Returns a manifest you can
    iterate over in the benchmark driver (each entry has .job, .path, .ndof).
    """
    if sizes is None:
        # DOF grows ~ from a few hundred to a few hundred thousand.
        sizes = [
            (10, 2, 2),
            (20, 4, 4),
            (40, 8, 8),
            (80, 16, 16),
            (120, 24, 24),
            (160, 32, 32),
        ]
    manifest: List[MeshInfo] = []
    for (nx, ny, nz) in sizes:
        job = f"cant_{nx}x{ny}x{nz}_{solver}"
        info = write_cantilever(outdir, job, nx, ny, nz, solver=solver, **kw)
        manifest.append(info)
        print(f"  wrote {info.path}   nodes={info.n_nodes:>8}  ndof={info.ndof:>8}")
    return manifest


if __name__ == "__main__":
    print("Generating default mesh-size sweep (SPOOLES + iterative variants)...")
    sweep(outdir="cases", solver="default")
    sweep(outdir="cases", solver="iterative")
    print("Done. Feed cases/*.inp to both ccx (Phase 1) and the Warp solver (Phase 2).")
