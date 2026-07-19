"""
Input validation (PHASE4 §B) — refuse out-of-scope inputs *loudly* instead of
returning a plausible wrong number.

The solver guarantees exactly: tetrahedral mesh · single isotropic linear-elastic
material · linear static / modal / buckling. The dangerous Judge failure is a
false-pass: computing a real-looking stress for a problem physically outside that
box. So `solve_structural` gates on this module first.

Two severities:
  * REJECT  — do not solve; return a failed FEAResult with the reason(s).
  * WARN    — solve, but flag on solver_status so a reviewer sees the caveat.

Every REJECT increments a **per-reason counter** (`reasons` keyed by a stable
code), persisted to JSON. That answers the standing question "which unsupported
feature did real traffic actually need, and how often?" — see `counter_report()`.

Reason codes are stable strings (grep-able, aggregatable), e.g.:
    load_type:gravity_unvalidated   element:hexahedron   material:plastic
    material:multiple               contact:defined      constraint:insufficient
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .load_case_adapter import CanonicalLoadCase, parse_load_case
from .results import FEAResult, SolverStatus

# --------------------------------------------------------------------------- #
# scope policy (single source of truth)
# --------------------------------------------------------------------------- #
# Load types validated element-for-element against ccx and therefore trusted.
# gravity is IMPLEMENTED but was never validated against the oracle, so per the
# §B honesty rule it is rejected until a validation case promotes it here.
SUPPORTED_LOADS = frozenset({"force", "pressure", "traction"})
IMPLEMENTED_BUT_UNVALIDATED = {"gravity": "load_type:gravity_unvalidated"}

SUPPORTED_SUPPORTS = frozenset({"fixed", "encastre", "clamped"})

# meshio cell types
_TET_TYPES = frozenset({"tetra", "tetra10"})
_SURFACE_TYPES = frozenset({"triangle", "triangle6"})   # legit: pressure/traction faces
# volume/structural cell types that are explicitly out of scope
_NONTET_VOLUME = frozenset({
    "hexahedron", "hexahedron20", "hexahedron27",
    "wedge", "wedge15", "pyramid", "pyramid13",
})
_SHELL_BEAM = frozenset({"quad", "quad8", "quad9", "line", "line3"})  # S4/S3-ish, beams
# cell types that are harmless bookkeeping and ignored for scope decisions
_IGNORED_CELLS = frozenset({"vertex"})

# WARN thresholds
NU_NEARLY_INCOMPRESSIBLE = 0.49
LARGE_DISP_FRACTION = 0.10          # max|u| > 10% of model bbox diagonal => linearity suspect

_DEFAULT_COUNTER_PATH = os.environ.get(
    "WARP_FEA_REJECTION_LOG",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "rejection_counts.json"),
)


# --------------------------------------------------------------------------- #
# per-reason counter (persistent)
# --------------------------------------------------------------------------- #
class RejectionCounter:
    """
    Process-wide, thread-safe, JSON-persisted tally of rejection reason codes.

    Persistence lets counts accumulate across many Judge invocations/processes so
    the analyst can later see the real demand for each unsupported feature.
    """

    def __init__(self, path: str = _DEFAULT_COUNTER_PATH):
        self.path = path
        self._lock = threading.Lock()

    def _load(self) -> Dict[str, int]:
        try:
            with open(self.path) as f:
                d = json.load(f)
                return {str(k): int(v) for k, v in d.get("reasons", d).items()}
        except (FileNotFoundError, ValueError, OSError):
            return {}

    def bump(self, codes: List[str]) -> None:
        if not codes:
            return
        with self._lock:
            counts = self._load()
            for c in codes:
                counts[c] = counts.get(c, 0) + 1
            tmp = self.path + ".tmp"
            try:
                with open(tmp, "w") as f:
                    json.dump({"reasons": counts,
                               "total_rejections": sum(counts.values())},
                              f, indent=2, sort_keys=True)
                os.replace(tmp, self.path)
            except OSError:
                pass          # telemetry must never break a solve

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return dict(sorted(self._load().items(), key=lambda kv: -kv[1]))


_COUNTER = RejectionCounter()


def counter_report(path: Optional[str] = None) -> Dict[str, int]:
    """Current rejection tally, most-demanded unsupported feature first."""
    return (RejectionCounter(path) if path else _COUNTER).snapshot()


# --------------------------------------------------------------------------- #
# validation result
# --------------------------------------------------------------------------- #
@dataclass
class ValidationResult:
    rejects: List[Tuple[str, str]] = field(default_factory=list)   # (code, human msg)
    warnings: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.rejects

    @property
    def reject_codes(self) -> List[str]:
        return [c for c, _ in self.rejects]

    def reject_message(self) -> str:
        return "; ".join(f"[{c}] {m}" for c, m in self.rejects)

    def warning_message(self) -> str:
        return "; ".join(f"[{c}] {m}" for c, m in self.warnings)


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def _check_material(clc: CanonicalLoadCase, vr: ValidationResult) -> None:
    mat = clc.material
    for marker in mat.nonlinear_markers:
        vr.rejects.append((f"material:{marker}",
                           f"material carries non-linear/anisotropic key {marker!r}; "
                           f"solver is isotropic linear-elastic only"))
    if not mat.isotropic_scalar:
        vr.rejects.append(("material:non_isotropic",
                           "E/nu are array-like (orthotropic/anisotropic); "
                           "only scalar isotropic E, nu supported"))
    if clc.materials_count > 1:
        vr.rejects.append(("material:multiple",
                           f"{clc.materials_count} materials defined; single-material only"))
    if clc.has_contact:
        vr.rejects.append(("contact:defined",
                           "contact/interaction defined; not supported"))
    # malformed / missing elastic constants (only meaningful if not already rejected)
    if mat.isotropic_scalar and not mat.nonlinear_markers:
        if mat.E is None or mat.nu is None:
            vr.rejects.append(("material:missing_elastic_constants",
                               "material must provide scalar SI E [Pa] and nu"))
        elif mat.E <= 0:
            vr.rejects.append(("material:nonpositive_E", f"E={mat.E} must be > 0"))
        elif not (-1.0 < mat.nu < 0.5):
            vr.rejects.append(("material:nu_out_of_range",
                               f"nu={mat.nu} outside physical (-1, 0.5)"))
        elif mat.nu >= NU_NEARLY_INCOMPRESSIBLE:
            vr.warnings.append(("material:nearly_incompressible",
                                f"nu={mat.nu} >= {NU_NEARLY_INCOMPRESSIBLE}: "
                                f"near-incompressible, CG convergence may degrade"))
    if mat.extra_keys:
        vr.warnings.append(("material:unknown_keys",
                            f"ignored unrecognised material keys {mat.extra_keys}"))


def _check_loads_supports(clc: CanonicalLoadCase, vr: ValidationResult) -> None:
    if not clc.loads:
        vr.rejects.append(("load:none", "load_case has no loads"))
    for ld in clc.loads:
        t = ld.type
        if t in SUPPORTED_LOADS:
            if t in ("force", "traction") and not ld.vector:
                vr.rejects.append((f"load_malformed:{t}",
                                   f"{t} load needs a 'vector'"))
            if t == "pressure" and ld.magnitude is None:
                vr.rejects.append(("load_malformed:pressure",
                                   "pressure load needs a scalar 'magnitude'"))
            if t in ("force", "pressure", "traction") and not ld.region:
                vr.rejects.append((f"load_malformed:{t}",
                                   f"{t} load needs a 'region'"))
        elif t in IMPLEMENTED_BUT_UNVALIDATED:
            code = IMPLEMENTED_BUT_UNVALIDATED[t]
            vr.rejects.append((code,
                               f"load type {t!r} is implemented but not validated "
                               f"against ccx; refusing until a validation case exists"))
        else:
            vr.rejects.append((f"load_type:{t or '<empty>'}",
                               f"unsupported load type {t!r}"))

    if not clc.supports:
        vr.rejects.append(("constraint:none",
                           "no supports; static stiffness matrix is singular"))
    for sup in clc.supports:
        if sup.type not in SUPPORTED_SUPPORTS:
            vr.rejects.append((f"support_type:{sup.type or '<empty>'}",
                               f"unsupported support type {sup.type!r} "
                               f"(only fully-fixed supports implemented)"))
        if not sup.region:
            vr.rejects.append(("support_malformed:no_region",
                               "support needs a 'region'"))


def _check_mesh_cells(cell_types: Dict[str, int], vr: ValidationResult) -> None:
    """cell_types: {meshio cell type -> count}."""
    n_tet = sum(cell_types.get(t, 0) for t in _TET_TYPES)
    for t, n in cell_types.items():
        if t in _NONTET_VOLUME:
            vr.rejects.append((f"element:{t}",
                               f"mesh has {n} {t} (non-tet volume) elements; tet-only solver"))
        elif t in _SHELL_BEAM:
            vr.rejects.append((f"element:{t}",
                               f"mesh has {n} {t} (shell/beam) elements; solid tet-only solver"))
    if n_tet == 0 and not any(t in _NONTET_VOLUME or t in _SHELL_BEAM for t in cell_types):
        vr.rejects.append(("element:no_tet",
                           f"no tetrahedral volume elements found (have {sorted(cell_types)})"))
    # mixed tet + non-tet volume already rejected above via the non-tet branch


# --------------------------------------------------------------------------- #
# public entry
# --------------------------------------------------------------------------- #
def validate_inputs(mesh_cell_types: Optional[Dict[str, int]],
                    load_case: dict,
                    *, bump_counter: bool = True) -> Tuple[ValidationResult, CanonicalLoadCase]:
    """
    Full pre-solve gate. `mesh_cell_types` is a {meshio type -> count} map (or
    None to skip mesh-type checks, e.g. when the caller already holds a validated
    FEMesh). Returns (ValidationResult, parsed CanonicalLoadCase).

    Rejections are counted here (once), so callers must not double-count.
    """
    vr = ValidationResult()
    clc = parse_load_case(load_case)
    _check_material(clc, vr)
    _check_loads_supports(clc, vr)
    if mesh_cell_types is not None:
        _check_mesh_cells(mesh_cell_types, vr)
    if clc.unknown_top_level_keys:
        vr.warnings.append(("schema:unknown_top_level_keys",
                            f"ignored unrecognised top-level keys "
                            f"{clc.unknown_top_level_keys} (Stage 2 schema drift?)"))
    if bump_counter and vr.rejects:
        _COUNTER.bump(vr.reject_codes)
    return vr, clc


def rejected_result(vr: ValidationResult, backend: str = "warp-gpu") -> FEAResult:
    """A failed FEAResult carrying the rejection reasons (measured stays empty)."""
    st = SolverStatus(
        regime="solid", backend=backend, meshed=False, converged=False,
        message="input rejected (out of solver scope): " + vr.reject_message(),
    )
    res = FEAResult(solver_status=st)
    res.rejected = True                    # attribute the Judge can branch on
    res.reject_codes = vr.reject_codes
    res.warnings = [c for c, _ in vr.warnings]
    return res


def peek_mesh_cell_types(mesh) -> Optional[Dict[str, int]]:
    """
    {meshio cell type -> count} for a .msh/.inp path or meshio.Mesh, WITHOUT
    committing to read_mesh (which raises on non-tet meshes). Returns None if the
    mesh cannot be inspected cheaply (then mesh-type checks are skipped, not faked).
    """
    try:
        import meshio
    except ImportError:
        return None
    try:
        if isinstance(mesh, str):
            m = meshio.read(mesh)
        elif hasattr(mesh, "cells"):
            m = mesh
        else:
            return None
    except Exception:
        return None
    counts: Dict[str, int] = {}
    for c in getattr(m, "cells", []):
        if c.type in _IGNORED_CELLS:
            continue
        counts[c.type] = counts.get(c.type, 0) + len(c.data)
    return counts


def post_solve_linearity_check(max_disp_m: float, bbox_diag_m: float) -> Optional[Tuple[str, str]]:
    """WARN if the peak displacement is large vs the model size (linearity suspect)."""
    if bbox_diag_m > 0 and max_disp_m > LARGE_DISP_FRACTION * bbox_diag_m:
        frac = max_disp_m / bbox_diag_m
        return ("linearity:large_displacement",
                f"max|u|={max_disp_m:.3e} m is {frac*100:.1f}% of model diagonal "
                f"{bbox_diag_m:.3e} m; small-strain linear assumption suspect")
    return None
