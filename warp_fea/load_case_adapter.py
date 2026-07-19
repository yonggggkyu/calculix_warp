"""
load_case_adapter — the ONE place that knows the external `load_case` wire format.

Why this file exists (PHASE4 §A + §4): the Judge's Stage 2 emits a `load_case`
JSON whose real schema we have not seen yet (A1 is *blocked* on obtaining it).
Everything downstream — the solver, the modal/buckling paths, the input
validator — consumes a *canonical* internal shape defined here, never the raw
external dict. So when the real Stage 2 schema arrives, the remap lives in the
`WIRE FORMAT` section below and **only this file changes**.

The current mapping reflects the *estimated* schema in solver.py's docstring. It
is deliberately close to 1:1 with the internal shape; treat the constants below
as the seam A1 will edit, not as a confirmed contract.

Design rules:
  * Parsing NEVER silently drops a field. Anything unrecognised is surfaced on
    the CanonicalLoadCase (unknown_* / raw markers) so the validator can reject
    or warn on it — silent-drop is exactly the false-pass failure §B guards.
  * The adapter does not judge scope (that's validation.py). It only normalises
    and *reports* what it saw. Keep policy out of here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# WIRE FORMAT  ── the seam A1 edits once the real Stage 2 schema is in hand.
# Map each external key to the canonical name the rest of the codebase uses.
# --------------------------------------------------------------------------- #
# Top-level sections
KEY_MATERIAL = "material"
KEY_SUPPORTS = "supports"
KEY_LOADS = "loads"
# Plural/assignment forms that would signal >1 material or contact (rejected in B)
KEY_MATERIALS_PLURAL = "materials"          # a list of material defs, if present
KEY_CONTACT = "contact"                     # contact/interaction section, if present
KEY_INTERACTIONS = "interactions"

# Material fields
MAT_E = "E"                                 # Young's modulus [Pa]
MAT_NU = "nu"                               # Poisson ratio
MAT_DENSITY = "density"                     # [kg/m^3]
MAT_YIELD = "yield_strength"                # [Pa] (not used by the solver; kept for D/Judge)
# Material fields that mark a *non*-isotropic-linear-elastic material -> reject in B.
# (Presence of ANY of these keys is treated as out-of-scope, regardless of value.)
MAT_NONLINEAR_MARKERS = (
    "plastic", "plasticity", "hardening", "stress_strain", "yield_curve",
    "hyperelastic", "ogden", "mooney_rivlin", "neo_hookean",
    "orthotropic", "anisotropic", "elastic_matrix", "D_matrix", "Cij",
    "creep", "viscoelastic", "damage",
)
# Material fields we understand and consume; everything else is "extra" (flagged).
MAT_KNOWN_KEYS = (MAT_E, MAT_NU, MAT_DENSITY, MAT_YIELD)

# Load/support fields
LD_TYPE = "type"
LD_REGION = "region"
LD_VECTOR = "vector"
LD_MAGNITUDE = "magnitude"
SUP_REGION = "region"
SUP_TYPE = "type"


# --------------------------------------------------------------------------- #
# Canonical internal representation (consumed everywhere downstream)
# --------------------------------------------------------------------------- #
@dataclass
class CanonicalMaterial:
    E: Optional[float]
    nu: Optional[float]
    density: Optional[float]
    yield_strength: Optional[float]
    nonlinear_markers: List[str]            # out-of-scope material feature keys seen
    extra_keys: List[str]                   # unrecognised material keys (warn)
    isotropic_scalar: bool                  # False if E/nu were array-like (orthotropic)


@dataclass
class CanonicalLoad:
    type: str                               # normalised, lower-case
    region: Optional[str]
    vector: Optional[List[float]]
    magnitude: Optional[float]
    raw: Dict[str, Any]


@dataclass
class CanonicalSupport:
    region: Optional[str]
    type: str                               # normalised, lower-case
    raw: Dict[str, Any]


@dataclass
class CanonicalLoadCase:
    material: CanonicalMaterial
    supports: List[CanonicalSupport]
    loads: List[CanonicalLoad]
    name: str
    materials_count: int                    # distinct material definitions detected
    has_contact: bool
    unknown_top_level_keys: List[str]
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_solver_dict(self) -> dict:
        """
        Render the shape `warp_fea.solver._solve` (and modal/buckling) already
        consume: {material, supports, loads}. This is the internal boundary — the
        solver never sees the raw external dict.
        """
        mat: Dict[str, Any] = {}
        if self.material.E is not None:
            mat["E"] = self.material.E
        if self.material.nu is not None:
            mat["nu"] = self.material.nu
        if self.material.density is not None:
            mat["density"] = self.material.density
        if self.material.yield_strength is not None:
            mat["yield_strength"] = self.material.yield_strength
        return {
            "name": self.name,
            "material": mat,
            "supports": [{"region": s.region, "type": s.type} for s in self.supports],
            "loads": [dict(l.raw) for l in self.loads],
        }


# --------------------------------------------------------------------------- #
# parse
# --------------------------------------------------------------------------- #
def _as_float(v: Any) -> Optional[float]:
    """Scalar float, or None if absent/array-like (array-like => not isotropic-scalar)."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_material(raw: dict) -> CanonicalMaterial:
    m = raw.get(KEY_MATERIAL) or {}
    if not isinstance(m, dict):
        # a non-dict material (e.g. a bare name/string referencing a library) is
        # out of scope; report it as an extra so validation rejects cleanly.
        return CanonicalMaterial(None, None, None, None,
                                 nonlinear_markers=[], extra_keys=["<material-not-a-dict>"],
                                 isotropic_scalar=False)
    markers = [k for k in MAT_NONLINEAR_MARKERS if k in m]
    extras = [k for k in m if k not in MAT_KNOWN_KEYS and k not in MAT_NONLINEAR_MARKERS]
    E_raw, nu_raw = m.get(MAT_E), m.get(MAT_NU)
    E, nu = _as_float(E_raw), _as_float(nu_raw)
    # isotropic-scalar iff both E and nu are present as scalars
    iso = not (isinstance(E_raw, (list, tuple)) or isinstance(nu_raw, (list, tuple)))
    return CanonicalMaterial(
        E=E, nu=nu,
        density=_as_float(m.get(MAT_DENSITY)),
        yield_strength=_as_float(m.get(MAT_YIELD)),
        nonlinear_markers=markers, extra_keys=extras, isotropic_scalar=iso,
    )


def _parse_loads(raw: dict) -> List[CanonicalLoad]:
    out: List[CanonicalLoad] = []
    for ld in raw.get(KEY_LOADS) or []:
        if not isinstance(ld, dict):
            out.append(CanonicalLoad(type="<malformed>", region=None, vector=None,
                                     magnitude=None, raw={"_malformed": ld}))
            continue
        t = str(ld.get(LD_TYPE, "")).strip().lower()
        vec = ld.get(LD_VECTOR)
        vec = [float(x) for x in vec] if isinstance(vec, (list, tuple)) else None
        out.append(CanonicalLoad(
            type=t, region=ld.get(LD_REGION), vector=vec,
            magnitude=_as_float(ld.get(LD_MAGNITUDE)), raw=dict(ld),
        ))
    return out


def _parse_supports(raw: dict) -> List[CanonicalSupport]:
    out: List[CanonicalSupport] = []
    for sup in raw.get(KEY_SUPPORTS) or []:
        if not isinstance(sup, dict):
            out.append(CanonicalSupport(region=None, type="<malformed>",
                                        raw={"_malformed": sup}))
            continue
        out.append(CanonicalSupport(
            region=sup.get(SUP_REGION),
            type=str(sup.get(SUP_TYPE, "fixed")).strip().lower(),
            raw=dict(sup),
        ))
    return out


# Top-level keys the current (estimated) schema is expected to carry; anything
# else present is reported so the validator/analyst can see it (A1 signal).
_KNOWN_TOP_LEVEL = {KEY_MATERIAL, KEY_SUPPORTS, KEY_LOADS, "name",
                    KEY_MATERIALS_PLURAL, KEY_CONTACT, KEY_INTERACTIONS}


def parse_load_case(raw: dict) -> CanonicalLoadCase:
    """
    External load_case dict -> CanonicalLoadCase. Pure structural normalisation;
    no scope policy (validation.py owns that). Never raises on unknown fields —
    it records them.
    """
    if not isinstance(raw, dict):
        raise TypeError(f"load_case must be a dict, got {type(raw).__name__}")

    material = _parse_material(raw)

    # material count: base material + any plural list of extra material defs
    plural = raw.get(KEY_MATERIALS_PLURAL)
    plural_n = len(plural) if isinstance(plural, (list, tuple)) else 0
    base_n = 1 if (raw.get(KEY_MATERIAL) is not None) else 0
    materials_count = max(base_n + plural_n, base_n, plural_n)

    has_contact = bool(raw.get(KEY_CONTACT) or raw.get(KEY_INTERACTIONS)) or any(
        (isinstance(l, dict) and str(l.get(LD_TYPE, "")).lower() == "contact")
        for l in (raw.get(KEY_LOADS) or [])
    )

    unknown_top = sorted(k for k in raw if k not in _KNOWN_TOP_LEVEL)

    return CanonicalLoadCase(
        material=material,
        supports=_parse_supports(raw),
        loads=_parse_loads(raw),
        name=str(raw.get("name", "case")),
        materials_count=materials_count,
        has_contact=has_contact,
        unknown_top_level_keys=unknown_top,
        raw=raw,
    )
