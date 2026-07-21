"""
Linear-elastic forms and stress recovery — all in fp64, all SI (Pa, m, N).

Reused from Phase 2 (bench/warp_solve.py): the displacement-based bilinear form
    a(u,v) = ∫ [ 2μ ε(u):ε(v) + λ tr(ε(u)) tr(ε(v)) ] dΩ ,  ε = sym(∇u)

New in Phase 3:
  * pressure (traction) linear form for *DLOAD-equivalent surface loads
  * von Mises recovery evaluated at *quadrature points* (spec §4.2) — element
    stresses are discontinuous and ccx extrapolates-then-averages to nodes, so
    comparing at integration points is the only apples-to-apples basis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import warp as wp
import warp.fem as fem

wp.set_module_options({"enable_backward": False})


def lame(E: float, nu: float) -> Tuple[float, float]:
    """(λ, μ) from Young's modulus and Poisson ratio. SI in -> SI out."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return lam, mu


@wp.func
def cauchy_stress(eps: wp.mat33d, lam: wp.float64, mu: wp.float64) -> wp.mat33d:
    """σ = 2μ ε + λ tr(ε) I"""
    return (mu + mu) * eps + (lam * wp.trace(eps)) * wp.identity(n=3, dtype=wp.float64)


def ortho_constants(m: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Orthotropic engineering constants -> ccx `*ELASTIC, TYPE=ORTHO` D components.

    Input (SI): E1,E2,E3 [Pa], nu12,nu13,nu23 [-], G12,G13,G23 [Pa].
    Returns (Dn, Gs):
      Dn : 3x3 symmetric normal-stiffness block
           [[D1111, D1122, D1133], [D1122, D2222, D2233], [D1133, D2233, D3333]]
      Gs : (D1212, D1313, D2323) = (G12, G13, G23)

    Dn is the inverse of the orthotropic compliance block; the symmetry
    nu21/E2 = nu12/E1 is assumed (standard orthotropic reciprocity), so only the
    three major Poisson ratios are needed.
    """
    E1, E2, E3 = (float(m[k]) for k in ("E1", "E2", "E3"))
    n12, n13, n23 = (float(m[k]) for k in ("nu12", "nu13", "nu23"))
    G12, G13, G23 = (float(m[k]) for k in ("G12", "G13", "G23"))
    S = np.array([
        [1.0 / E1,   -n12 / E1,  -n13 / E1],
        [-n12 / E1,   1.0 / E2,  -n23 / E2],
        [-n13 / E1,  -n23 / E2,   1.0 / E3],
    ], dtype=np.float64)
    Dn = np.linalg.inv(S)
    Dn = 0.5 * (Dn + Dn.T)                       # kill round-off asymmetry
    return Dn, np.array([G12, G13, G23], dtype=np.float64)


ORTHO_KEYS = ("E1", "E2", "E3", "nu12", "nu13", "nu23", "G12", "G13", "G23")


def is_orthotropic(m: dict) -> bool:
    """True if the material dict describes an orthotropic (not isotropic) solid."""
    if str(m.get("type", "")).lower() in ("orthotropic", "ortho"):
        return True
    return any(k in m for k in ORTHO_KEYS)


def material_model(m: dict) -> Tuple[str, dict]:
    """
    Resolve a material dict into ('iso'|'ortho', integrand values).

    The returned dict plugs straight into `fem.integrate(..., values=...)` for the
    matching *_ortho / isotropic form pair, so callers only branch on the kind.
    """
    if is_orthotropic(m):
        missing = [k for k in ORTHO_KEYS if k not in m]
        if missing:
            raise ValueError(f"orthotropic material missing {missing}; "
                             f"needs all of {list(ORTHO_KEYS)}")
        Dn, Gs = ortho_constants(m)
        return "ortho", {"Dn": wp.mat33d(*Dn.flatten().tolist()),
                         "Gs": wp.vec3d(*Gs.tolist())}
    if "E" not in m or "nu" not in m:
        raise ValueError("isotropic material needs SI 'E' [Pa] and 'nu'")
    lam, mu = lame(float(m["E"]), float(m["nu"]))
    return "iso", {"lam": wp.float64(lam), "mu": wp.float64(mu)}


@dataclass
class MaterialPart:
    """One material and the volume region it applies to (None = whole model)."""
    region: Optional[str]
    kind: str                       # 'iso' | 'ortho'
    values: dict                    # integrand values for the matching forms
    density: Optional[float]
    raw: dict

    @property
    def stiffness_form(self):
        return elasticity_form_ortho if self.kind == "ortho" else elasticity_form

    @property
    def von_mises_form(self):
        return von_mises_at_qp_ortho if self.kind == "ortho" else von_mises_at_qp

    @property
    def geometric_form(self):
        return (geometric_stiffness_form_ortho if self.kind == "ortho"
                else geometric_stiffness_form)


def material_parts(load_case: dict) -> List[MaterialPart]:
    """
    Resolve a load_case into its material parts.

    `materials: [{region, ...}, ...]` describes an assembly (shared nodes at the
    interfaces, one material per volume physical group). A plain `material` is the
    single-part case and yields one part with region=None, so callers can treat
    both identically.
    """
    plural = load_case.get("materials")
    if plural:
        parts: List[MaterialPart] = []
        for spec in plural:
            if not isinstance(spec, dict):
                raise ValueError("each entry of 'materials' must be a dict")
            region = spec.get("region")
            if not region:
                raise ValueError("each entry of 'materials' needs a volume 'region'")
            kind, values = material_model(spec)
            parts.append(MaterialPart(region, kind, values,
                                      _opt_float(spec.get("density")), spec))
        return parts
    m = load_case.get("material") or {}
    kind, values = material_model(m)
    return [MaterialPart(None, kind, values, _opt_float(m.get("density")), m)]


def _opt_float(v):
    return None if v is None else float(v)


@wp.func
def ortho_stress(eps: wp.mat33d, Dn: wp.mat33d, Gs: wp.vec3d) -> wp.mat33d:
    """
    Orthotropic Hooke's law, axes aligned with the global frame.

        σ_nn = Dn · ε_nn                (normal components)
        σ_12 = 2 D1212 ε_12,  etc.      (tensor strain, hence the factor 2)

    Reduces to the isotropic law when Dn = λ(11ᵀ) + 2μI and Gs = (μ, μ, μ).
    """
    e = wp.vec3d(eps[0, 0], eps[1, 1], eps[2, 2])
    sn = Dn * e
    s12 = (Gs[0] + Gs[0]) * eps[0, 1]
    s13 = (Gs[1] + Gs[1]) * eps[0, 2]
    s23 = (Gs[2] + Gs[2]) * eps[1, 2]
    return wp.mat33d(sn[0], s12, s13,
                     s12, sn[1], s23,
                     s13, s23, sn[2])


@wp.func
def von_mises(sig: wp.mat33d) -> wp.float64:
    """σ_vm = sqrt(3/2 s:s),  s = σ - tr(σ)/3 I"""
    s = sig - (wp.trace(sig) / wp.float64(3.0)) * wp.identity(n=3, dtype=wp.float64)
    return wp.sqrt(wp.float64(1.5) * wp.ddot(s, s))


# --------------------------------------------------------------------------- #
# forms
# --------------------------------------------------------------------------- #
@fem.integrand
def elasticity_form(s: fem.Sample, u: fem.Field, v: fem.Field,
                    lam: wp.float64, mu: wp.float64):
    """Stiffness bilinear form (identical physics to Phase 2, SI units)."""
    return wp.ddot(fem.D(v, s), cauchy_stress(fem.D(u, s), lam, mu))


@fem.integrand
def elasticity_form_ortho(s: fem.Sample, u: fem.Field, v: fem.Field,
                          Dn: wp.mat33d, Gs: wp.vec3d):
    """Stiffness bilinear form for an orthotropic material (global-axis aligned)."""
    return wp.ddot(fem.D(v, s), ortho_stress(fem.D(u, s), Dn, Gs))


@fem.integrand
def von_mises_at_qp_ortho(s: fem.Sample, u: fem.Field, Dn: wp.mat33d, Gs: wp.vec3d):
    """von Mises [Pa] at sample s for an orthotropic material."""
    return von_mises(ortho_stress(fem.D(u, s), Dn, Gs))


@fem.integrand
def geometric_stiffness_form_ortho(s: fem.Sample, u: fem.Field, v: fem.Field,
                                   u0: fem.Field, Dn: wp.mat33d, Gs: wp.vec3d):
    """Geometric stiffness built on an orthotropic pre-buckling stress."""
    sigma = ortho_stress(fem.D(u0, s), Dn, Gs)
    return wp.ddot(sigma, wp.transpose(fem.grad(v, s)) @ fem.grad(u, s))


@fem.integrand
def mass_form(s: fem.Sample, u: fem.Field, v: fem.Field, rho: wp.float64):
    """Consistent mass bilinear form — used by the M5 modal path."""
    return rho * wp.dot(u(s), v(s))


@fem.integrand
def pressure_form(s: fem.Sample, domain: fem.Domain, v: fem.Field, p: wp.float64):
    """
    Traction linear form for a uniform pressure `p` on a surface:
        ∫_Γ (-p n) · v dΓ
    Sign convention matches CalculiX *DLOAD: positive p pushes *into* the face
    (i.e. acts along -n, the inward normal).
    """
    n = fem.normal(domain, s)
    return -p * wp.dot(n, v(s))


@fem.integrand
def traction_form(s: fem.Sample, v: fem.Field, t: wp.vec3d):
    """Uniform surface traction vector t [Pa] on a surface: ∫_Γ t · v dΓ."""
    return wp.dot(t, v(s))


@fem.integrand
def body_force_form(s: fem.Sample, v: fem.Field, f: wp.vec3d):
    """Uniform body force density f [N/m^3] (e.g. rho*g): ∫_Ω f · v dΩ."""
    return wp.dot(f, v(s))


# --------------------------------------------------------------------------- #
# stress recovery (evaluated at quadrature points)
# --------------------------------------------------------------------------- #
@fem.integrand
def von_mises_at_qp(s: fem.Sample, u: fem.Field, lam: wp.float64, mu: wp.float64):
    """von Mises stress [Pa] of the solved displacement field, at sample s."""
    return von_mises(cauchy_stress(fem.D(u, s), lam, mu))


@fem.integrand
def position_at_qp(s: fem.Sample, domain: fem.Domain):
    """World position [m] of sample s — gives `location.coords` for the maximum."""
    return fem.position(domain, s)


# --------------------------------------------------------------------------- #
# geometric (initial-stress) stiffness — for linear eigenvalue buckling
# --------------------------------------------------------------------------- #
@fem.integrand
def geometric_stiffness_form(s: fem.Sample, u: fem.Field, v: fem.Field,
                             u0: fem.Field, lam: wp.float64, mu: wp.float64):
    """
    Geometric-stiffness bilinear form built on the pre-buckling stress σ(u0):

        k_g(u, v) = ∫_Ω σ_ij (∂u_k/∂x_i)(∂v_k/∂x_j) dΩ
                  = ∫_Ω σ : ( ∇v^T ∇u ) dΩ            (σ symmetric)

    where u0 is the displacement of the reference linear-static solve. The
    buckling pencil is then  K φ = λ (−K_g) φ,  smallest positive λ = load factor.
    """
    sigma = cauchy_stress(fem.D(u0, s), lam, mu)
    gu = fem.grad(u, s)                          # ∂u_i/∂x_j
    gv = fem.grad(v, s)
    return wp.ddot(sigma, wp.transpose(gv) @ gu)


# --------------------------------------------------------------------------- #
# small GPU reductions
# --------------------------------------------------------------------------- #
@wp.kernel
def disp_norm_kernel(u: wp.array(dtype=wp.vec3d), out: wp.array(dtype=wp.float64)):
    i = wp.tid()
    out[i] = wp.length(u[i])
