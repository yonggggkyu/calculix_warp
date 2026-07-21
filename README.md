# calculix_warp

A GPU linear-static structural FEA solver built on [NVIDIA Warp](https://github.com/NVIDIA/warp)
(`warp.fem`), validated element-for-element against [CalculiX](http://www.calculix.de/) (`ccx`)
as an oracle.

It exists to replace/accelerate the **CalculiX runner** an LLM-Judge design-review
system uses: given a mesh and a load case, return `max_von_mises_stress`,
`max_displacement`, and the lowest natural frequencies — on the GPU, in SI units,
with an honest convergence flag.

Two phases, developed in order:

| Phase | Package | What it does |
|-------|---------|--------------|
| **2** | `bench/`     | Structured **hex (C3D8)** cantilever solver + a ccx-vs-Warp speed/accuracy sweep (crossover curve). Proves the GPU path is correct and fast. |
| **3** | `warp_fea/`  | Unstructured **quadratic tet (C3D10)** solver with von Mises recovery, pressure loads, region-based BCs, modal + buckling analysis, and the Judge-facing `solve_structural(mesh, load_case) → FEAResult` contract. |

The four quantities the Judge consumes — **max von Mises stress**, **max
displacement**, **eigenfrequencies**, **buckling load factor** — all have a GPU
path, each validated against the matching CalculiX analysis (`*STATIC`,
`*FREQUENCY`, `*BUCKLE`).

All results are **SI (Pa, m, N)** and **fp64**. The solve loop is GPU-resident:
`bsr_cg` (conjugate gradient) with Jacobi preconditioning, captured in a CUDA graph.

---

## Results at a glance

Reproduced numbers live in [`sample_results/`](sample_results/). Headlines:

**Phase 2 — hex cantilever, Warp (H200) vs ccx CG (28-core CPU):**

| DOF | ccx wall | Warp wall | speedup | displacement rel-err |
|----:|---------:|----------:|--------:|---------------------:|
| 9,963   | 0.36 s  | 0.015 s | 24× | 1.7e-07 |
| 70,227  | 3.8 s   | 0.035 s | 109× | 1.6e-07 |
| 525,987 | 54.2 s  | 0.25 s  | **216×** | 1.8e-07 |

GPU crosses over CPU at ~300 DOF; at 526k DOF the H200 sustains ~97% utilisation
while CPU stays < 20%.

**Phase 3 — tet solver, measured vs single-threaded ccx:**

| case | element | DOF | disp rel-err | von Mises rel-err |
|------|---------|----:|-------------:|------------------:|
| cantilever        | C3D10 | 13,452  | 3.3e-08 | 0.00% |
| pressure cylinder | C3D10 | 83,232  | 1.6e-07 | 0.00% |
| plate with hole   | C3D10 | 247,566 | 5.3e-08 | 0.00% |
| bolted bracket    | C3D10 | 22,119  | 4.3e-08 | 0.00% |
| self-weight (gravity)      | C3D10 | 13,452 | 7.6e-08 | 0.00% |
| orthotropic (CFRP-like)    | C3D10 | 13,452 | 1.4e-07 | 0.00% |
| two-material assembly      | C3D10 | 10,443 | 1.9e-07 | 0.00% |

Modal (lowest 10 frequencies, C3D10) agrees with ccx `*FREQUENCY` to **1.1e-05**.
Buckling (slender Euler column, lowest 4 factors, C3D10) agrees with ccx
`*BUCKLE` to **1.1e-07** (BLF 44.13 vs an analytical Euler estimate of ~44.1).

All seven acceptance criteria (PHASE3 §3) pass: `sample_results/acceptance.json`.

---

## Requirements

- **A CUDA GPU** (developed on an NVIDIA **H200**, `sm_90`; any recent CUDA GPU
  should work). Warp ships its own CUDA runtime, so you only need a recent driver
  (developed against driver 580 / CUDA 13.0; Warp used CUDA Toolkit 12.9 internally).
- **Python 3.10+** (developed on 3.11).
- **CalculiX `ccx` 2.17+** — only for *validation* (the oracle). The solver itself
  does not need it.

### Install

```bash
# 1. Python packages
pip install -r requirements.txt

# 2. CalculiX (validation oracle) — Debian/Ubuntu
sudo apt-get update && sudo apt-get install -y calculix-ccx

# 3. gmsh's Python wheel needs a few system libs for its shared object
sudo apt-get install -y libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1
```

Verify the GPU is visible to Warp:

```bash
python -c "import warp as wp; wp.init(); print(wp.get_devices())"
# -> [Device(cpu), Device(cuda:0), ...]
```

The Warp JIT kernel cache defaults to `<repo>/.warp_cache/` (gitignored) so kernels
are compiled once and reused. Override with `export WARP_CACHE_PATH=/some/writable/path`.

---

## Usage

Run everything **from the repo root** (both packages are top-level; `warp_fea`
imports the `bench` instrumentation harness).

### Phase 3 — the full acceptance suite (recommended entry point)

```bash
python -m warp_fea.acceptance
```

Generates the validation meshes, solves each on the GPU, runs ccx as an oracle,
and checks all six criteria. Writes `measured_parity.csv` and `acceptance.json`.

### Phase 3 — call the solver directly (the Judge contract)

```python
from warp_fea import solve_structural

load_case = {
    "material": {"E": 113.8e9, "nu": 0.342, "density": 4430.0},  # SI: Pa, kg/m^3
    "supports": [{"region": "mounting_holes", "type": "fixed"}], # region = gmsh physical group
    "loads":    [{"region": "payload_face", "type": "force", "vector": [0, 0, -3924.0]}],
}
result = solve_structural("bracket.msh", load_case)   # .msh path or a meshio.Mesh

result.measured["max_von_mises_stress"].value     # Pa
result.measured["max_von_mises_stress"].location  # {coords, node_id, region}
result.measured["max_displacement"].value         # m
result.solver_status.converged                    # never trust measured if this is False
```

Supported loads: `force` (total N over a node set, ≙ `*CLOAD`), `pressure`
(Pa on a face, ≙ `*DLOAD`), `traction` (Pa vector on a face), `gravity`/self-weight
(m/s², needs `density`, ≙ `*DLOAD ... GRAV`).
Supports: `fixed`. Regions are resolved through gmsh **physical groups**.

**Orthotropic materials** — give the nine engineering constants instead of `E`/`nu`
(≙ ccx `*ELASTIC, TYPE=ORTHO`):

```python
"material": {"type": "orthotropic",
             "E1": 135e9, "E2": 10e9, "E3": 10e9,
             "nu12": 0.30, "nu13": 0.30, "nu23": 0.45,
             "G12": 5e9, "G13": 5e9, "G23": 3.5e9, "density": 1600.0}
```

**Assemblies** — one material per volume region, nodes shared at the interfaces
(≙ several ccx `*SOLID SECTION, ELSET=...`). Use `materials` instead of `material`:

```python
"materials": [
    {"region": "part_steel", "E": 193e9, "nu": 0.29, "density": 7900.0},
    {"region": "part_alu",   "E": 70e9,  "nu": 0.33, "density": 2700.0},
]
```

Modal and buckling analysis:

```python
from warp_fea import solve_modal, solve_buckling

result = solve_modal("bracket.msh", load_case, n_modes=6)
result.measured["eigenfrequencies"]        # [Hz], ascending

result = solve_buckling("column.msh", load_case, n_modes=4)  # loads = reference load
result.measured["buckling_load_factor"].value   # BLF; critical load = BLF x reference
```

### Phase 2 — the CPU-vs-GPU crossover sweep

```bash
python -m bench.gen_inp                          # write the mesh-size sweep
python -m bench.bench_driver --max-size 160x32x32   # ccx vs Warp, all sizes
# -> results.json, parity.csv, crossover.png
```

---

## Scope: what it solves, and what it refuses

The solver guarantees **tetrahedral meshes · linear elasticity · linear static /
modal / buckling**. Anything outside that is *refused*, not approximated — a
rejected `FEAResult` carries `rejected=True` and stable `reject_codes`, and
`measured` is left empty. Silently returning a plausible-looking number for an
out-of-scope problem is the one failure mode a design Judge cannot tolerate.

In scope (each validated against a matching ccx analysis):

| feature | ccx equivalent |
|---|---|
| isotropic + **orthotropic** linear elasticity | `*ELASTIC`, `*ELASTIC, TYPE=ORTHO` |
| force / pressure / traction / **gravity (self-weight)** | `*CLOAD`, `*DLOAD`, `*DLOAD ... GRAV` |
| **assemblies**: material per region, shared interface nodes | several `*SOLID SECTION` |
| static, modal, buckling | `*STATIC`, `*FREQUENCY`, `*BUCKLE` |

Refused, with the reason code:

| input | code | why |
|---|---|---|
| **contact / interactions** | `contact:defined` | nonlinear; needs an active-set/penalty iteration this linear solver does not have — **Phase 5** |
| full anisotropy (21 constants), plasticity, hyperelasticity, creep | `material:anisotropic`, `material:plastic`, … | not implemented |
| hex / shell / beam elements | `element:hexahedron`, … | tet-only |
| tie constraints between non-conformal meshes | (assembly needs shared nodes) | not implemented |
| unconstrained model, malformed loads/materials | `constraint:none`, `load_malformed:*`, `material:*` | would produce garbage |

Every rejection is counted per reason in `rejection_counts.json`, so it is
answerable which unsupported feature real traffic actually needed.

## ⚠️ Known issue: CalculiX 2.17 multi-threaded stress recovery is racy

This bit us for hours, so it is worth stating plainly.

`ccx` prints `Using up to N cpu(s) for the stress calculation`, and **that parallel
path intermittently writes corrupted output** — mostly integration-point stresses,
but displacements too (a 14-thread run of the plate case gave a max displacement
0.2% off what both 1-thread ccx and Warp agree on). Measured on one deck, 30
repeats each:

| ccx threads | corrupted runs |
|------------:|---------------:|
| 28 | 1 / 30 |
| 8  | 4 / 30 |
| **1**  | **0 / 30** |

Only ~1% of integration points are hit, so the corrupted `.dat` looks completely
normal (right row count, right displacements) but reports a max von Mises silently
15–80× too high. For a yield check, that is exactly the input that fabricates a
false verdict.

**Mitigation in this repo:** `warp_fea.validate.run_ccx_oracle` always runs ccx
**single-threaded** (deterministic — 0/30 corrupt) and re-runs once to confirm the
result reproduces. If you use ccx for anything trustworthy, run it with
`OMP_NUM_THREADS=1`.

(Thread counts here are auto-detected from the pod's cgroup CPU quota, not
`os.cpu_count()`, which on a shared node reports the whole node — see
`pod_cpu_quota()`.)

---

## Supported scope & input validation (Phase 4 §B)

The solver guarantees exactly: **tetrahedral mesh · one isotropic linear-elastic
material · linear static / modal / buckling**. Feeding it something outside that
box and getting a plausible-looking number back is the worst Judge failure (a
false pass), so `solve_structural` **gates the input first** and *refuses* rather
than guesses:

| Refused (returns `rejected=True`, empty `measured`, reason code) |
|---|
| non-tet elements — `hexahedron`, `wedge`, `pyramid`, shell `quad`, beam `line` |
| non-isotropic / non-linear material — orthotropic, plastic, hyperelastic, … |
| more than one material, or any contact / interaction defined |
| unsupported **or unvalidated** load types — incl. `gravity` (implemented but never validated against ccx, so refused until a validation case promotes it) |
| missing supports (singular stiffness), non-fixed support types |

Soft issues **warn** instead of refusing (e.g. `nu ≥ 0.49` near-incompressible;
post-solve `max|u|` large vs model size ⇒ linearity suspect). Every refusal is
tallied by a stable **reason code**, persisted to `rejection_counts.json`, so you
can later see *which* unsupported feature real traffic actually needed and how
often:

```python
from warp_fea import counter_report
counter_report()   # {'load_type:gravity_unvalidated': 12, 'element:hexahedron': 3, ...}
```

The external `load_case` schema is currently **estimated** (Stage 2's real schema
is not yet in hand). All parsing is isolated in `load_case_adapter.py`; when the
real schema arrives, only that one file changes. `python -m tests.test_input_validation`
proves the gate (GPU-free).

## Layout

```
bench/          Phase 2: hex solver + CPU/GPU benchmark harness
  instrument.py     wall-clock + CPU%/GPU% measurement (measure() context)
  gen_inp.py        parametric C3D8 cantilever .inp generator (mesh sweep)
  run_ccx.py        run ccx under instrumentation, parse .dat displacements
  warp_solve.py     Warp GPU hex elasticity solver
  bench_driver.py   full sweep -> results.json, parity.csv, crossover.png
warp_fea/       Phase 3: tet solver + Judge contract
  solver.py            solve_structural(.msh, load_case) -> FEAResult  (main entry)
  mesh_io.py           gmsh/.inp -> Tetmesh + region (physical-group) resolution
  elasticity.py        bilinear form + von Mises recovery (at quadrature points)
  modal.py             eigenfrequencies via GPU subspace iteration
  buckling.py          linear buckling: geometric stiffness + subspace eigensolve
  results.py           FEAResult / measured / solver_status (§1.2 schema)
  validate.py          bake .inp, run ccx oracle, measured-parity comparison
  cases.py             SI validation geometries (gmsh)
  acceptance.py        the six PHASE3 §3 acceptance criteria, as one suite
  load_case_adapter.py Phase 4: the ONE place that knows the external load_case
                       wire format — the seam to re-map when Stage 2's real
                       schema arrives (§A). Nothing else parses the raw dict.
  validation.py        Phase 4 §B: pre-solve input gate — refuse out-of-scope
                       inputs with a reason, tally each refusal by reason code
                       (counter_report()); WARN (don't reject) on soft issues.
  stress_basis_study.py Phase 4 §D: integration-point vs nodal-.frd von Mises
                       quantified on plate_with_hole.
tests/          test_input_validation.py  — §B acceptance (GPU-free, 15 checks)
docs/           PHASE2_SPEC.md, PHASE3_SPEC.md (design specs)
sample_results/ reference outputs (CSV / JSON / crossover.png)
```

## Method notes

- **Elasticity:** displacement-based form `a(u,v)=∫[2μ ε(u):ε(v)+λ tr ε(u) tr ε(v)]dΩ`.
- **von Mises comparison** is done at **integration points** (not extrapolated to
  nodes): element stresses are discontinuous and ccx extrapolates-then-averages, so
  comparing at quadrature points measures physics, not post-processing. `warp.fem`'s
  `RegularQuadrature(order=2)` gives 4 points/tet, matching ccx's C3D10 rule.
- **Which von Mises the Judge consumes (Phase 4 §D).** `measured["max_von_mises_stress"]`
  is the **integration-point** max — the basis validated element-for-element against
  ccx (agrees to ~1e-7). Quantified on `plate_with_hole` (Kt≈3, run
  `python -m warp_fea.stress_basis_study`): the Kt peak sits on the hole *surface*,
  but quadrature points are element-*interior*, so the integration-point max is
  **~4% below** ccx's surface-aware nodal `.frd` value (15.32 vs 15.97 MPa). This is a
  **known, deliberate caveat**: for a yield check the integration-point basis is ~4%
  non-conservative at stress concentrations. It was kept because it is the basis with
  1e-7 ccx parity; ccx's nodal `.frd` value bounds the true surface peak from above if
  a fully conservative check is ever required.
- **Convergence is reported honestly:** `bsr_cg` stops at `err ≤ max(tol·‖b‖, tol)`;
  if it doesn't, `converged=False` and `measured` is emptied rather than returning a
  number the Judge might trust.
