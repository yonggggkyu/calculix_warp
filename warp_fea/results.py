"""
FEAResult / measured / solver_status dataclasses — the PHASE3_SPEC §1.2 contract.

Everything here is SI: stress in Pa, displacement in m, coordinates in m, force in N.
The `measured` keys are exactly the Stage 2 `check.quantity` strings the Judge
compares against thresholds:

    max_von_mises_stress, max_displacement
    eigenfrequencies, buckling_load_factor   (reserved; filled by M5)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Location:
    """Where a measured extremum lives. `region` is the named physical group."""
    coords: List[float]
    node_id: Optional[int] = None
    region: Optional[str] = None

    def as_dict(self) -> dict:
        return {"coords": [float(c) for c in self.coords],
                "node_id": None if self.node_id is None else int(self.node_id),
                "region": self.region}


@dataclass
class Measurement:
    value: float
    unit: str
    location: Optional[Location] = None

    def as_dict(self) -> dict:
        d: Dict[str, Any] = {"value": float(self.value), "unit": self.unit}
        if self.location is not None:
            d["location"] = self.location.as_dict()
        return d


@dataclass
class SolverStatus:
    regime: str = "solid"
    backend: str = "warp-gpu"
    meshed: bool = True
    converged: bool = False
    n_dof: int = 0
    n_iters: int = 0
    wall_time_s: float = 0.0
    cpu_avg_pct: float = 0.0
    gpu_util_avg_pct: Optional[float] = None
    device: str = ""
    # diagnostics beyond the spec's minimum (harmless additions, useful for triage)
    residual: Optional[float] = None
    tol: Optional[float] = None
    element: Optional[str] = None
    message: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class FEAResult:
    """
    §1.2 contract object.

    `measured` holds only quantities the solver actually trusts: when
    `solver_status.converged` is False the measured dict is emptied, because a
    Judge false-pass on an unconverged solve is the worst possible failure.
    `fields` keeps GPU arrays; call `field_numpy()` to pay for the host copy.

    A measured entry is a `Measurement` for scalars (`max_von_mises_stress`,
    `max_displacement`, `buckling_load_factor`) or a plain ascending list of
    floats for spectra (`eigenfrequencies`, in Hz) — matching §1.2 exactly.
    """
    measured: Dict[str, Any] = field(default_factory=dict)
    solver_status: SolverStatus = field(default_factory=SolverStatus)
    fields: Dict[str, Any] = field(default_factory=dict)
    # PHASE4 §B: input-validation outcome. `rejected` means the input was out of
    # solver scope and NOT solved; `reject_codes`/`warnings` carry the reason
    # codes so the Judge can branch (a rejected result is not a failed solve).
    rejected: bool = False
    reject_codes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def as_dict(self, include_fields: bool = True) -> dict:
        d: Dict[str, Any] = {
            "measured": {k: (v.as_dict() if isinstance(v, Measurement) else v)
                         for k, v in self.measured.items()},
            "solver_status": self.solver_status.as_dict(),
            "rejected": self.rejected,
            "reject_codes": list(self.reject_codes),
            "warnings": list(self.warnings),
        }
        if include_fields:
            d["fields"] = {k: "<gpu-array>" for k in self.fields}
        return d

    def field_numpy(self, name: str):
        """Explicit host copy — the only place a GPU->CPU transfer is allowed."""
        arr = self.fields[name]
        return arr.numpy() if hasattr(arr, "numpy") else arr

    def invalidate(self, message: str) -> "FEAResult":
        """Drop measured values that must not be trusted (spec §1 invariant)."""
        self.measured = {}
        self.solver_status.converged = False
        self.solver_status.message = message
        return self
