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

from typing import Tuple

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
# small GPU reductions
# --------------------------------------------------------------------------- #
@wp.kernel
def disp_norm_kernel(u: wp.array(dtype=wp.vec3d), out: wp.array(dtype=wp.float64)):
    i = wp.tid()
    out[i] = wp.length(u[i])
