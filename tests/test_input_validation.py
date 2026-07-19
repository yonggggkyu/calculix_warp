"""
PHASE4 §B acceptance: out-of-scope inputs are refused with a reason, and every
refusal is tallied by reason code.

Runs GPU-free — it exercises the validation gate directly (and solve_structural's
early reject path, which returns before any GPU work). Run either as:

    python -m tests.test_input_validation      # from repo root
    pytest tests/test_input_validation.py
"""

from __future__ import annotations

import os
import tempfile

from warp_fea.load_case_adapter import parse_load_case
from warp_fea.validation import (RejectionCounter, ValidationResult,
                                 counter_report, validate_inputs)
from warp_fea.solver import solve_structural

# a minimal in-scope load_case (single isotropic material, fixed support, force)
GOOD_LC = {
    "material": {"E": 193e9, "nu": 0.29, "density": 7900.0},
    "supports": [{"region": "fixed_face", "type": "fixed"}],
    "loads": [{"region": "tip_face", "type": "force", "vector": [0, 0, -1000.0]}],
}
TET_CELLS = {"tetra10": 2560, "triangle6": 400}


def _codes(load_case, cell_types=TET_CELLS):
    vr, _ = validate_inputs(cell_types, load_case, bump_counter=False)
    return set(vr.reject_codes), vr


# --------------------------------------------------------------------------- #
# in-scope input passes the gate
# --------------------------------------------------------------------------- #
def test_good_input_passes():
    codes, vr = _codes(GOOD_LC)
    assert vr.ok, f"in-scope input was rejected: {vr.reject_message()}"
    assert not codes


# --------------------------------------------------------------------------- #
# each out-of-scope input is refused with the expected reason code
# --------------------------------------------------------------------------- #
def test_gravity_rejected_as_unvalidated():
    lc = {**GOOD_LC, "loads": GOOD_LC["loads"] + [{"type": "gravity", "vector": [0, 0, -9.81]}]}
    codes, _ = _codes(lc)
    assert "load_type:gravity_unvalidated" in codes


def test_hex_mesh_rejected():
    codes, _ = _codes(GOOD_LC, {"hexahedron": 1000})
    assert "element:hexahedron" in codes


def test_shell_mesh_rejected():
    codes, _ = _codes(GOOD_LC, {"quad": 500})
    assert "element:quad" in codes


def test_no_tet_mesh_rejected():
    codes, _ = _codes(GOOD_LC, {"triangle6": 500})
    assert "element:no_tet" in codes


def test_mixed_tet_and_hex_rejected():
    codes, _ = _codes(GOOD_LC, {"tetra10": 2000, "hexahedron": 10})
    assert "element:hexahedron" in codes


def test_two_materials_rejected():
    lc = {**GOOD_LC, "materials": [{"E": 1e9, "nu": 0.3}, {"E": 2e9, "nu": 0.3}]}
    codes, _ = _codes(lc)
    assert "material:multiple" in codes


def test_plastic_material_rejected():
    lc = {**GOOD_LC, "material": {**GOOD_LC["material"], "plastic": [[215e6, 0.0]]}}
    codes, _ = _codes(lc)
    assert "material:plastic" in codes


def test_orthotropic_material_rejected():
    lc = {**GOOD_LC, "material": {"E": [200e9, 180e9, 180e9], "nu": 0.3}}
    codes, _ = _codes(lc)
    assert "material:non_isotropic" in codes


def test_contact_rejected():
    lc = {**GOOD_LC, "contact": [{"master": "a", "slave": "b"}]}
    codes, _ = _codes(lc)
    assert "contact:defined" in codes


def test_missing_supports_rejected():
    lc = {**GOOD_LC, "supports": []}
    codes, _ = _codes(lc)
    assert "constraint:none" in codes


def test_unsupported_support_type_rejected():
    lc = {**GOOD_LC, "supports": [{"region": "f", "type": "pinned"}]}
    codes, _ = _codes(lc)
    assert "support_type:pinned" in codes


# --------------------------------------------------------------------------- #
# WARN (not reject): near-incompressible still solves but is flagged
# --------------------------------------------------------------------------- #
def test_nearly_incompressible_warns_not_rejects():
    lc = {**GOOD_LC, "material": {"E": 1e6, "nu": 0.499, "density": 1000.0}}
    codes, vr = _codes(lc)
    assert vr.ok, "near-incompressible should WARN, not reject"
    assert "material:nearly_incompressible" in {c for c, _ in vr.warnings}


# --------------------------------------------------------------------------- #
# solve_structural returns a rejected result (no GPU work) for bad input
# --------------------------------------------------------------------------- #
def test_solve_structural_rejects_before_solving():
    bad = {**GOOD_LC, "material": {"E": 1e9, "nu": 0.3, "hyperelastic": {"C10": 1e5}}}
    res = solve_structural("nonexistent.msh", bad)     # must not touch the file/GPU
    assert res.rejected is True
    assert res.solver_status.converged is False
    assert res.measured == {}
    assert "material:hyperelastic" in res.reject_codes


# --------------------------------------------------------------------------- #
# per-reason counter accumulates and persists
# --------------------------------------------------------------------------- #
def test_counter_accumulates_and_persists():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rej.json")
        c = RejectionCounter(path)
        c.bump(["load_type:gravity_unvalidated", "element:hexahedron"])
        c.bump(["load_type:gravity_unvalidated"])
        snap = c.snapshot()
        assert snap["load_type:gravity_unvalidated"] == 2
        assert snap["element:hexahedron"] == 1
        # persisted: a fresh instance on the same path sees the same tally
        assert RejectionCounter(path).snapshot() == snap
        # most-demanded first
        assert list(snap)[0] == "load_type:gravity_unvalidated"


# --------------------------------------------------------------------------- #
def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} validation tests passed")


if __name__ == "__main__":
    _run_all()
