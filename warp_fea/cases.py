"""
Validation geometries (spec §6) — built with gmsh, labelled with physical groups,
dimensioned in SI metres.

Each builder returns (msh_path, load_case) so the Warp path and the ccx oracle
consume exactly the same mesh and the same load_case.

  cantilever_beam  — tip point load           (M1 displacement, M2 stress)
  pressure_cylinder— thick-walled, internal p (M3 pressure/*DLOAD; Lamé problem)
  plate_with_hole  — tension + stress raiser  (M2 location/region, Kt ~ 3)
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

# SI stainless-steel-ish material used across the validation set (spec §4.3)
STEEL_SI = {"E": 193e9, "nu": 0.29, "density": 7900.0, "yield_strength": 215e6}

# gmsh pads getBoundingBox by its geometric tolerance (~1e-7 m per side), so a
# bbox span never equals the nominal dimension exactly. Compare with a slack far
# below any feature size (mm) but well above that padding.
_BBOX_EPS = 1e-5


def _gmsh_begin(name: str):
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(name)
    return gmsh


def _gmsh_finish(gmsh, path: str, order: int, size: float) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size)
    gmsh.option.setNumber("Mesh.ElementOrder", order)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    if order == 2:
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)   # straight-sided tets:
        # keeps mid-side nodes at exact edge midpoints, which is what both the warp
        # affine Tetmesh and a ccx C3D10 deck assume.
    gmsh.model.mesh.generate(3)
    gmsh.write(path)
    gmsh.finalize()
    return path


def cantilever_beam(outdir: str = "fea_cases", order: int = 2,
                    size: float = 0.006, L: float = 0.2, W: float = 0.02,
                    H: float = 0.02, tip_force_N: float = -2000.0
                    ) -> Tuple[str, Dict]:
    """Clamped at x=0, total transverse force on the x=L face."""
    gmsh = _gmsh_begin("cantilever")
    gmsh.model.occ.addBox(0, 0, 0, L, W, H)
    gmsh.model.occ.synchronize()
    fixed, loaded = [], []
    for (d, t) in gmsh.model.getEntities(2):
        com = gmsh.model.occ.getCenterOfMass(d, t)
        if abs(com[0]) < 1e-9:
            fixed.append(t)
        elif abs(com[0] - L) < 1e-9:
            loaded.append(t)
    gmsh.model.addPhysicalGroup(2, fixed, name="fixed_face")
    gmsh.model.addPhysicalGroup(2, loaded, name="load_face")
    gmsh.model.addPhysicalGroup(3, [1], name="solid")
    path = _gmsh_finish(gmsh, os.path.join(outdir, f"beam_o{order}.msh"), order, size)

    load_case = {
        "name": "cantilever_tip_load",
        "material": dict(STEEL_SI),
        "supports": [{"region": "fixed_face", "type": "fixed"}],
        "loads": [{"region": "load_face", "type": "force",
                   "vector": [0.0, 0.0, tip_force_N]}],
    }
    return path, load_case


def pressure_cylinder(outdir: str = "fea_cases", order: int = 2,
                      size: float = 0.004, r_in: float = 0.02, r_out: float = 0.04,
                      length: float = 0.06, p_Pa: float = 1e6) -> Tuple[str, Dict]:
    """Thick-walled cylinder, internal pressure on the bore, one end clamped."""
    gmsh = _gmsh_begin("pcyl")
    outer = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, length, r_out)
    inner = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, length, r_in)
    gmsh.model.occ.cut([(3, outer)], [(3, inner)])
    gmsh.model.occ.synchronize()

    bore, fixed_end = [], []
    for (d, t) in gmsh.model.getEntities(2):
        com = gmsh.model.occ.getCenterOfMass(d, t)
        # bore: curved surface whose centroid sits on the axis at mid-height
        bbox = gmsh.model.occ.getBoundingBox(d, t)
        radial = max(abs(bbox[3]), abs(bbox[0]))
        on_axis = abs(com[0]) < 1e-9 and abs(com[1]) < 1e-9
        if on_axis and abs(com[2] - length / 2) < 1e-9 and radial < (r_in + r_out) / 2:
            bore.append(t)
        elif abs(com[2]) < 1e-9:
            fixed_end.append(t)
    gmsh.model.addPhysicalGroup(2, bore, name="bore")
    gmsh.model.addPhysicalGroup(2, fixed_end, name="fixed_end")
    gmsh.model.addPhysicalGroup(3, [v[1] for v in gmsh.model.getEntities(3)], name="solid")
    path = _gmsh_finish(gmsh, os.path.join(outdir, f"pcyl_o{order}.msh"), order, size)

    load_case = {
        "name": "internal_pressure_cylinder",
        "material": dict(STEEL_SI),
        "supports": [{"region": "fixed_end", "type": "fixed"}],
        "loads": [{"region": "bore", "type": "pressure", "magnitude": p_Pa}],
        "_analytic": {"r_in": r_in, "r_out": r_out, "p": p_Pa},
    }
    return path, load_case


def plate_with_hole(outdir: str = "fea_cases", order: int = 2, size: float = 0.0035,
                    L: float = 0.2, W: float = 0.1, t: float = 0.01,
                    r_hole: float = 0.01, traction_Pa: float = 5e6
                    ) -> Tuple[str, Dict]:
    """Plate with a central hole under end tension — Kt ≈ 3 stress concentration."""
    gmsh = _gmsh_begin("plate_hole")
    box = gmsh.model.occ.addBox(0, 0, 0, L, W, t)
    cyl = gmsh.model.occ.addCylinder(L / 2, W / 2, -t, 0, 0, 3 * t, r_hole)
    gmsh.model.occ.cut([(3, box)], [(3, cyl)])
    gmsh.model.occ.synchronize()

    fixed, pull, hole = [], [], []
    for (d, tag) in gmsh.model.getEntities(2):
        com = gmsh.model.occ.getCenterOfMass(d, tag)
        if abs(com[0]) < 1e-9:
            fixed.append(tag)
        elif abs(com[0] - L) < 1e-9:
            pull.append(tag)
        elif (abs(com[0] - L / 2) < 1e-6 and abs(com[1] - W / 2) < 1e-6
              and abs(com[2] - t / 2) < 1e-6):
            hole.append(tag)
    gmsh.model.addPhysicalGroup(2, fixed, name="fixed_face")
    gmsh.model.addPhysicalGroup(2, pull, name="pull_face")
    if hole:
        gmsh.model.addPhysicalGroup(2, hole, name="hole_surface")
    gmsh.model.addPhysicalGroup(3, [v[1] for v in gmsh.model.getEntities(3)], name="solid")
    # refine towards the hole so the Kt peak is resolved
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "SurfacesList", hole or [])
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", size / 2.5)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", size)
    gmsh.model.mesh.field.setNumber(2, "DistMin", r_hole * 0.5)
    gmsh.model.mesh.field.setNumber(2, "DistMax", r_hole * 3.0)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    path = _gmsh_finish(gmsh, os.path.join(outdir, f"plate_hole_o{order}.msh"),
                        order, size)

    load_case = {
        "name": "plate_with_hole_tension",
        "material": dict(STEEL_SI),
        "supports": [{"region": "fixed_face", "type": "fixed"}],
        "loads": [{"region": "pull_face", "type": "traction",
                   "vector": [traction_Pa, 0.0, 0.0]}],
    }
    return path, load_case


def hccx_bracket(outdir: str = "fea_cases", order: int = 2, size: float = 0.0045,
                 plate=(0.100, 0.060, 0.008), hole_d: float = 0.009,
                 pattern=(0.070, 0.040), payload_N: float = -3924.0
                 ) -> Tuple[str, Dict]:
    """
    M4 case, shaped after the H-CCX `034_s3_nasa_5020b_bolted_joint` bracket:
    a Ti-6Al-4V plate, 100 x 60 x 8 mm, with four 9 mm bolt holes on a 70 x 40 mm
    pattern. The bolt holes are the support ("mounting_holes" — the exact region
    name PHASE3_SPEC §4.4 uses as its example); the payload pulls on the top face.

    payload_N default = 40 kg x 9.81 m/s^2 = 392.4 N at 10 g = 3924 N.
    """
    L, W, T = plate
    gmsh = _gmsh_begin("hccx_bracket")
    box = gmsh.model.occ.addBox(0, 0, 0, L, W, T)
    px, py = pattern
    holes = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            cx, cy = L / 2 + sx * px / 2, W / 2 + sy * py / 2
            holes.append((3, gmsh.model.occ.addCylinder(cx, cy, -T, 0, 0, 3 * T,
                                                        hole_d / 2)))
    gmsh.model.occ.cut([(3, box)], holes)
    gmsh.model.occ.synchronize()

    bore_tags, top_tags = [], []
    for (d, t) in gmsh.model.getEntities(2):
        com = gmsh.model.occ.getCenterOfMass(d, t)
        bb = gmsh.model.occ.getBoundingBox(d, t)
        span_z = bb[5] - bb[2]
        is_bore = (abs(span_z - T) < _BBOX_EPS
                   and (bb[3] - bb[0]) < hole_d + _BBOX_EPS
                   and (bb[4] - bb[1]) < hole_d + _BBOX_EPS)
        if is_bore:
            bore_tags.append(t)
        elif abs(com[2] - T) < 1e-9:
            top_tags.append(t)
    if len(bore_tags) != 4:
        raise RuntimeError(f"expected 4 bolt-hole surfaces, tagged {len(bore_tags)}")
    gmsh.model.addPhysicalGroup(2, bore_tags, name="mounting_holes")
    gmsh.model.addPhysicalGroup(2, top_tags, name="payload_face")
    gmsh.model.addPhysicalGroup(3, [v[1] for v in gmsh.model.getEntities(3)],
                                name="solid")
    path = _gmsh_finish(gmsh, os.path.join(outdir, f"hccx_bracket_o{order}.msh"),
                        order, size)

    load_case = {
        "name": "hccx_bolted_bracket_payload",
        # Ti-6Al-4V Grade 5, SI (the H-CCX spec quotes E = 113.8 GPa, nu = 0.342)
        "material": {"E": 113.8e9, "nu": 0.342, "density": 4430.0,
                     "yield_strength": 860e6},
        "supports": [{"region": "mounting_holes", "type": "fixed"}],
        "loads": [{"region": "payload_face", "type": "force",
                   "vector": [0.0, 0.0, payload_N]}],
    }
    return path, load_case


ALL_CASES = {
    "cantilever": cantilever_beam,
    "pressure_cylinder": pressure_cylinder,
    "plate_with_hole": plate_with_hole,
    "hccx_bracket": hccx_bracket,
}


if __name__ == "__main__":
    for name, fn in ALL_CASES.items():
        for order in (1, 2):
            p, lc = fn(order=order)
            print(f"{name:18s} order={order}  -> {p}")
