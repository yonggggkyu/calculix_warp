"""
Mesh + region ingestion:  gmsh .msh (primary, Judge path) / Abaqus .inp (oracle path)
-> tet connectivity + named region sets.

Contract (spec §1, §4.4): BC/load *values* come from the load_case; the *place*
they apply comes from a named gmsh physical group, which meshio surfaces as
field_data + cell_data["gmsh:physical"] (and, for .inp, as NSET/ELSET). This
module turns a region name into concrete node / face sets.

All coordinates are taken as SI metres — no unit conversion happens anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np

try:
    import meshio
except ImportError as e:  # pragma: no cover
    raise ImportError("meshio is required: pip install meshio") from e

# meshio's tetra10 slot order is identical to the Abaqus C3D10 convention
# (verified empirically: slot4=mid(0,1), 5=mid(1,2), 6=mid(0,2), 7=mid(0,3),
#  8=mid(1,3), 9=mid(2,3)), so no permutation is needed when writing .inp decks.
TET_TYPES = ("tetra10", "tetra")
TRI_TYPES = ("triangle6", "triangle")

# Local corner ids of the four faces of a tet, in Abaqus face order
# (C3D4/C3D10: F1=1-2-3, F2=1-2-4, F3=2-3-4, F4=1-3-4 in 1-based corner ids).
TET_FACE_CORNERS = ((0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3))


@dataclass
class Region:
    name: str
    dim: int                 # 2 = surface (BC face / pressure face), 3 = volume
    cell_type: str
    cells: np.ndarray        # (K, n) node indices into FEMesh.points
    node_idx: np.ndarray     # unique node indices touched by this region

    def __repr__(self) -> str:
        return (f"Region({self.name!r}, dim={self.dim}, {self.cell_type}, "
                f"{len(self.cells)} cells, {len(self.node_idx)} nodes)")


@dataclass
class FEMesh:
    points: np.ndarray               # (N,3) float64, metres — every mesh node
    tets: np.ndarray                 # (M,10) or (M,4) node indices into points
    tet_type: str                    # "tetra10" | "tetra"
    corners: np.ndarray              # (M,4) corner connectivity, indices into points
    regions: Dict[str, Region]
    source: str = ""

    # --- compacted corner view, for building a warp Tetmesh ---
    # A tetra10 mesh's `points` also holds mid-side nodes, which no tet *corner*
    # references. Handing that full array to fem.Tetmesh would create isolated
    # vertices: warp allocates a node for each one, never writes its position
    # (leaving it at the origin) and gives it an empty stiffness row. Pass only
    # the corners, and the degree-2 space then regenerates the mid-side nodes
    # itself — landing exactly on the mesh's own mid-side nodes.
    vertex_ids: np.ndarray = None    # (V,) index into points for each Tetmesh vertex
    corners_local: np.ndarray = None # (M,4) indices into vertex_ids

    @property
    def n_nodes(self) -> int:
        return self.points.shape[0]

    @property
    def n_elems(self) -> int:
        return self.tets.shape[0]

    @property
    def order(self) -> int:
        return 2 if self.tet_type == "tetra10" else 1

    @property
    def vertex_points(self) -> np.ndarray:
        """Positions of the Tetmesh vertices (corner nodes only)."""
        return self.points[self.vertex_ids]

    def region(self, name: str) -> Region:
        if name not in self.regions:
            raise KeyError(
                f"region {name!r} not found in mesh {self.source!r}; "
                f"available physical groups: {sorted(self.regions)}"
            )
        return self.regions[name]

    def surface_nodes(self, name: str) -> np.ndarray:
        return self.region(name).node_idx

    def region_of_node(self, node: int) -> Optional[str]:
        """First surface region containing this node (for measured.location.region)."""
        for r in self.regions.values():
            if r.dim == 2 and node in r._node_set:      # type: ignore[attr-defined]
                return r.name
        return None


def _finalize_regions(regions: Dict[str, Region]) -> Dict[str, Region]:
    for r in regions.values():
        r._node_set = set(int(i) for i in r.node_idx)    # type: ignore[attr-defined]
    return regions


def read_mesh(mesh: Union[str, "meshio.Mesh"]) -> FEMesh:
    """
    Read a gmsh .msh / Abaqus .inp (or accept an in-memory meshio.Mesh) and
    resolve physical groups into named regions.
    """
    if isinstance(mesh, str):
        m = meshio.read(mesh)
        source = mesh
    else:
        m = mesh
        source = getattr(mesh, "_source", "<meshio.Mesh>")

    points = np.asarray(m.points, dtype=np.float64)
    if points.shape[1] == 2:                              # promote 2d meshes
        points = np.column_stack([points, np.zeros(len(points))])

    # ---- volume elements ----
    tets = None
    tet_type = None
    for ct in TET_TYPES:                                   # prefer quadratic
        blocks = [c.data for c in m.cells if c.type == ct]
        if blocks:
            tets = np.vstack(blocks).astype(np.int64)
            tet_type = ct
            break
    if tets is None:
        have = sorted({c.type for c in m.cells})
        raise ValueError(f"{source}: no tetra/tetra10 cells found (have {have})")
    corners = tets[:, :4].copy()
    vertex_ids, corners_local = np.unique(corners, return_inverse=True)
    corners_local = corners_local.reshape(corners.shape).astype(np.int64)

    # ---- regions from gmsh physical groups ----
    regions: Dict[str, Region] = {}
    phys = m.cell_data.get("gmsh:physical")
    if phys is not None and m.field_data:
        tag2name = {int(v[0]): (k, int(v[1])) for k, v in m.field_data.items()}
        for bi, block in enumerate(m.cells):
            tags = np.asarray(phys[bi])
            for tag in np.unique(tags):
                if int(tag) not in tag2name:
                    continue
                name, dim = tag2name[int(tag)]
                sel = block.data[tags == tag].astype(np.int64)
                if name in regions:                        # merge multi-block groups
                    sel = np.vstack([regions[name].cells, sel])
                regions[name] = Region(name, dim, block.type, sel,
                                       np.unique(sel))
    # ---- regions from meshio cell_sets / point_sets (.inp ELSET/NSET) ----
    if not regions:
        for name, blocks in (m.cell_sets or {}).items():
            if name.startswith("gmsh:"):
                continue
            sel_list = []
            ctype = ""
            for bi, idx in enumerate(blocks):
                if idx is None or len(idx) == 0:
                    continue
                blk = m.cells[bi]
                sel_list.append(blk.data[np.asarray(idx, dtype=np.int64)])
                ctype = blk.type
            if sel_list:
                sel = np.vstack(sel_list).astype(np.int64)
                dim = 3 if ctype in TET_TYPES else 2
                regions[name] = Region(name, dim, ctype, sel, np.unique(sel))
    for name, idx in (getattr(m, "point_sets", None) or {}).items():
        idx = np.asarray(idx, dtype=np.int64)
        regions[name] = Region(name, 0, "vertex", idx.reshape(-1, 1), np.unique(idx))

    return FEMesh(points, tets, tet_type, corners, _finalize_regions(regions), source,
                  vertex_ids=vertex_ids, corners_local=corners_local)


def region_elements(fe: FEMesh, name: str) -> np.ndarray:
    """
    Element indices (rows of `fe.tets`) belonging to a **volume** region.

    Assemblies assign a material per volume physical group, and warp.fem needs
    cell indices to build the Subdomain. Regions store node tuples, not element
    ids, so match on the corner-node set — that is independent of block ordering
    and of how meshio split the cells.
    """
    r = fe.region(name)
    if r.dim != 3:
        raise ValueError(f"region {name!r} is dim={r.dim}, expected a volume (dim=3)")
    lut = {frozenset(int(v) for v in tet): i for i, tet in enumerate(fe.corners)}
    out = []
    for cell in r.cells:
        key = frozenset(int(v) for v in cell[:4])
        idx = lut.get(key)
        if idx is None:
            raise RuntimeError(f"region {name!r} contains a cell that is not a mesh tet")
        out.append(idx)
    return np.asarray(sorted(set(out)), dtype=np.int64)


def region_faces(fe: FEMesh, name: str) -> np.ndarray:
    """Corner triples (K,3) of a surface region — the geometric face definition."""
    r = fe.region(name)
    if r.dim != 2:
        raise ValueError(f"region {name!r} is dim={r.dim}, expected a surface (dim=2)")
    return r.cells[:, :3].astype(np.int64)


def tet_face_lookup(fe: FEMesh) -> Dict[frozenset, tuple]:
    """
    Map a face's corner-node set -> (element index, local Abaqus face number 1..4).
    Used to bake *DLOAD pressure cards for the ccx oracle.
    """
    lut: Dict[frozenset, tuple] = {}
    for ei, tet in enumerate(fe.corners):
        for fi, loc in enumerate(TET_FACE_CORNERS):
            key = frozenset(int(tet[c]) for c in loc)
            lut.setdefault(key, (ei, fi + 1))
    return lut
