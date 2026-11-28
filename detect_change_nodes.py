"""Find where a local design edit changed the geometry, without node IDs.

Inputs are two meshes in one coordinate frame: the reference design (the one already
simulated) and the edited design. They need not share node numbers, node order,
element layout or mesh density: the edited design may be remeshed.

A geometric change is a move of the domain's boundary, so both designs are reduced
to their boundary first (mesh_boundary):
  shell (S3/S4 in 3D)        the elements themselves (a surface)
  2D solid (CPS4, DC2D4...)  element edges used by one element (closed curves)
  3D solid (C3D8, C3D4)      element faces used by one element (the skin)
Interior nodes of a solid carry no geometry: after a remesh they correspond to
nothing, so they are never reported. A morph that drags them along is not a change.

    on_reference, on_edited, ref_nodes, new_nodes = detect_mesh_edit(
        ref_xyz, ref_elements, new_xyz, new_elements)
    ref_nodes[on_reference.changed]       # reference mesh nodes on a moved boundary
    -on_reference.displacement            # where that boundary went (per boundary node)

Method: two-way point-to-boundary deviation with a discretization tolerance.
1. For every boundary node of one design, find the closest point on the other
   design's boundary (exact, VTK cell locator on triangles or line segments).
   Displacement = node - closest point; signed deviation = displacement along the
   closest cell's normal.
2. Noise floor sigma = 1.4826 * MAD of the signed deviations. An edit moves a few
   percent of the boundary, so the median and MAD see only meshing noise.
3. Chordal tolerance. Straight cells cut across curved boundary, so two meshes of one
   unchanged boundary differ there by about c * L^2 / 8 each (c the curvature, L the
   cell size). c comes from the reference only (the edit adds curvature, which must
   not raise the tolerance where the edit is): the angle between neighbouring cells /
   their size, over smooth joints. Sharper joints (> feature_deg) are corners or
   feature lines that a remesher keeps, so they add none.
   tolerance = c * (L_own^2 + L_other^2) / 8. When most reference boundary nodes
   coincide with edited ones the two share one discretization (a morph) and the
   tolerance is zero.
4. Hysteresis along each boundary's own edges: a node is changed when its distance
   exceeds tau_low = max(k_low * sigma, floor_mm) + kappa * tolerance and it is
   connected through such nodes to one above tau_high = max(k_high * sigma,
   2 * floor_mm) + 2 * kappa * tolerance, in a connected region of at least
   min_nodes nodes (10 on a surface, 3 on a curve: about three cells across).

Defaults were set on the hood's design 6 against gmsh remeshes of itself (no edit,
so no labels): kappa = 1, feature_deg = 45 flags <= 0.14% of the 18,182 nodes in
clusters of <= 7 nodes; without the tolerance 29-37%. See
verify_change_node_detection.py (hood) and verify_change_node_detection_benchmarks.py
(B1-B4).

Only the normal part of a change is observable. A node that slides along an
unchanged boundary (down a straight wall, say) leaves the shape as it was, and no
method without node correspondence can see it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

MAD_TO_SIGMA = 1.4826
SHARED_NODE_FRACTION = 0.5  # above this the meshes share a discretization (morph)
MIN_NODES = {1: 3, 2: 10}   # smallest changed region: about three cells across
# Corner order of the faces of first-order solid elements (Abaqus numbering).
SOLID_FACES = {
    8: ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)),  # C3D8
    4: ((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)),                                         # C3D4
}


def triangulate(faces) -> np.ndarray:
    """(F, 3) int64 triangles from triangles and quads (an array, or a list of tuples)."""
    triangles = []
    for face in faces:
        face = [int(n) for n in face]
        if len(face) == 3:
            triangles.append(face)
        elif len(face) == 4:
            triangles.extend((face[:3], [face[0], face[2], face[3]]))
        else:
            raise ValueError(f"Faces must have 3 or 4 nodes; got {len(face)}")
    return np.asarray(triangles, dtype=np.int64).reshape(-1, 3)


def mesh_edges(faces) -> np.ndarray:
    """(E, 2) unique undirected cell edges (for quads, the four sides; a segment is its own edge)."""
    pairs = []
    for face in faces:
        face = [int(n) for n in face]
        pairs.extend(zip(face, face[1:] + face[:1]) if len(face) > 2 else [tuple(face)])
    pairs = np.sort(np.asarray(pairs, dtype=np.int64).reshape(-1, 2), axis=1)
    return np.unique(pairs, axis=0)


class Surface:
    """A boundary as cells that answer closest-point queries: triangles/quads in 3D
    (dim 2), or line segments in the z = 0 plane (dim 1). 2D points get z = 0."""

    def __init__(self, points, faces):
        import pyvista as pv

        points = np.asarray(points, dtype=np.float64)
        if points.ndim == 2 and points.shape[1] == 2:
            points = np.column_stack((points, np.zeros(len(points))))
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("points must be a finite (N, 2) or (N, 3) array")
        self.points = points
        self.faces = [tuple(int(n) for n in f) for f in faces]
        lengths = {len(f) for f in self.faces}
        if lengths == {2}:
            self.dim = 1
            if np.ptp(points[:, 2]) > 1e-9 * max(np.ptp(points), 1.0):
                raise ValueError("Line-segment boundaries must lie in the z = 0 plane")
            self.cells = np.asarray(self.faces, dtype=np.int64).reshape(-1, 2)
            self.polydata = pv.PolyData(points, lines=np.hstack((np.full((len(self.cells), 1), 2),
                                                                 self.cells)).ravel())
            tangent = points[self.cells[:, 1]] - points[self.cells[:, 0]]
            normals = np.column_stack((-tangent[:, 1], tangent[:, 0], np.zeros(len(tangent))))
        elif lengths <= {3, 4}:
            self.dim = 2
            self.cells = triangulate(self.faces)
            self.polydata = pv.PolyData(points, np.hstack((np.full((len(self.cells), 1), 3), self.cells)).ravel())
            corners = points[self.cells]
            normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        else:
            raise ValueError(f"Cells must be all segments, or triangles/quads; got sizes {sorted(lengths)}")
        if self.cells.size and (self.cells.min() < 0 or self.cells.max() >= len(points)):
            raise ValueError("faces refer to nodes outside points")
        length = np.linalg.norm(normals, axis=1, keepdims=True)
        self.normals = np.divide(normals, length, out=np.zeros_like(normals), where=length > 0)
        self._edges = None
        self._curvature = {}

    @property
    def triangles(self) -> np.ndarray:
        if self.dim != 2:
            raise AttributeError("A curve boundary has segments, not triangles")
        return self.cells

    def closest(self, query) -> tuple[np.ndarray, np.ndarray]:
        """(closest points (Q, 3), cell index (Q,)) on this boundary."""
        query = np.asarray(query, dtype=np.float64)
        if query.shape[1] == 2:
            query = np.column_stack((query, np.zeros(len(query))))
        cell, point = self.polydata.find_closest_cell(query, return_closest_point=True)
        return np.asarray(point, dtype=np.float64).reshape(-1, 3), np.asarray(cell, dtype=np.int64).reshape(-1)

    def edges(self) -> np.ndarray:
        if self._edges is None:
            self._edges = mesh_edges(self.faces)
        return self._edges

    def element_size(self) -> tuple[np.ndarray, np.ndarray]:
        """Mean edge length at each node, and the largest of those over each cell's corners."""
        edges = self.edges()
        length = np.linalg.norm(self.points[edges[:, 0]] - self.points[edges[:, 1]], axis=1)
        count = len(self.points)
        total = np.bincount(edges[:, 0], length, count) + np.bincount(edges[:, 1], length, count)
        degree = np.bincount(edges[:, 0], None, count) + np.bincount(edges[:, 1], None, count)
        node = total / np.maximum(degree, 1)
        return node, node[self.cells].max(axis=1)

    def curvature(self, feature_deg: float) -> tuple[np.ndarray, np.ndarray]:
        """Largest turning angle / size (rad/mm) over the smooth joints (angle <= feature_deg)
        at each (node, cell). A joint is the edge between two triangles, or the node
        between two segments."""
        if feature_deg not in self._curvature:
            if self.dim == 1:
                self._curvature[feature_deg] = self._curve_curvature(feature_deg)
            else:
                self._curvature[feature_deg] = self._surface_curvature(feature_deg)
        return self._curvature[feature_deg]

    def _surface_curvature(self, feature_deg):
        count = len(self.cells)
        edges = np.concatenate((self.cells[:, [0, 1]], self.cells[:, [1, 2]], self.cells[:, [2, 0]]))
        owner = np.tile(np.arange(count), 3)
        edges = np.sort(edges, axis=1)
        order = np.lexsort((edges[:, 1], edges[:, 0]))
        edges, owner = edges[order], owner[order]
        shared = np.flatnonzero((edges[1:] == edges[:-1]).all(axis=1))
        first, second, pair = owner[shared], owner[shared + 1], edges[shared]
        cosine = np.abs(np.einsum("ij,ij->i", self.normals[first], self.normals[second]))
        theta = np.arccos(np.clip(cosine, 0.0, 1.0))
        length = np.linalg.norm(self.points[pair[:, 0]] - self.points[pair[:, 1]], axis=1)
        value = np.where(theta <= math.radians(feature_deg), theta / np.maximum(length, 1e-12), 0.0)
        node = np.zeros(len(self.points))
        np.maximum.at(node, pair[:, 0], value)
        np.maximum.at(node, pair[:, 1], value)
        cell = np.zeros(count)
        np.maximum.at(cell, first, value)
        np.maximum.at(cell, second, value)
        return node, cell

    def _curve_curvature(self, feature_deg):
        count = len(self.points)
        ends = self.cells.ravel()
        owner = np.repeat(np.arange(len(self.cells)), 2)
        degree = np.bincount(ends, minlength=count)
        order = np.argsort(ends, kind="stable")
        ends, owner = ends[order], owner[order]
        pair = np.flatnonzero((ends[1:] == ends[:-1]) & (degree[ends[:-1]] == 2))
        node_id, first, second = ends[pair], owner[pair], owner[pair + 1]
        cosine = np.abs(np.einsum("ij,ij->i", self.normals[first], self.normals[second]))
        theta = np.arccos(np.clip(cosine, 0.0, 1.0))
        size = 0.5 * (np.linalg.norm(np.diff(self.points[self.cells[first]], axis=1)[:, 0], axis=1)
                      + np.linalg.norm(np.diff(self.points[self.cells[second]], axis=1)[:, 0], axis=1))
        value = np.where(theta <= math.radians(feature_deg), theta / np.maximum(size, 1e-12), 0.0)
        node = np.zeros(count)
        np.maximum.at(node, node_id, value)
        return node, node[self.cells].max(axis=1)


def mesh_boundary(points, elements, kind: str = "auto") -> tuple[Surface, np.ndarray]:
    """(boundary Surface, mesh node index of each boundary node) of a first-order mesh.

    kind: "shell" (the elements are the surface), "solid" (2D: quads/triangles in a
    plane; 3D: C3D8 hexes or C3D4 tets), or "auto": 2D points or a planar mesh with
    3/4-node elements is a 2D solid; 8-node elements are hexes; 3D 4-node elements
    are tets when they have volume, else shell quads; 3D triangles are a shell.
    """
    points = np.asarray(points, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int64)
    if elements.ndim != 2:
        raise ValueError("elements must be an (E, k) array of 0-based node indices (one element type)")
    k = elements.shape[1]
    planar = points.shape[1] == 2 or np.ptp(points[:, 2]) <= 1e-9 * max(np.ptp(points), 1.0)
    if kind == "auto":
        if planar and k in (3, 4):
            kind = "solid"
        elif k == 8:
            kind = "solid"
        elif k == 4:
            corners = points[elements[: min(len(elements), 200)]]
            volume = np.abs(np.einsum("ij,ij->i", np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
                                      corners[:, 3] - corners[:, 0])) / 6.0
            edge = np.linalg.norm(corners[:, 1] - corners[:, 0], axis=1)
            kind = "solid" if np.median(volume / np.maximum(edge, 1e-12) ** 3) > 1e-3 else "shell"
        elif k == 3:
            kind = "shell"
        else:
            raise ValueError(f"Cannot tell the mesh kind for {k}-node elements; first-order elements only")
    if kind == "shell":
        cells = elements
    elif kind == "solid" and planar:
        sides = np.concatenate([elements[:, [i, (i + 1) % k]] for i in range(k)])
        cells = _unshared(sides)
    elif kind == "solid":
        if k not in SOLID_FACES:
            raise ValueError(f"3D solids must be C3D8 or C3D4; got {k}-node elements")
        faces = np.concatenate([elements[:, list(f)] for f in SOLID_FACES[k]])
        cells = _unshared(faces)
    else:
        raise ValueError(f"kind must be auto, shell or solid; got {kind!r}")
    node_index = np.unique(cells)
    local = np.full(len(points), -1, dtype=np.int64)
    local[node_index] = np.arange(len(node_index))
    return Surface(points[node_index], [tuple(c) for c in local[cells]]), node_index


def _unshared(faces):
    """Faces (rows of node indices) that occur once, ignoring orientation."""
    key = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return faces[counts[inverse.ravel()] == 1]


@dataclass
class ChangeDetection:
    """Per-node result for the boundary nodes of one design against the other's boundary."""

    changed: np.ndarray       # (N,) bool
    distance: np.ndarray      # (N,) mm, node to the other boundary
    signed: np.ndarray        # (N,) mm, along the closest cell's normal
    displacement: np.ndarray  # (N, 3) mm, node - closest point
    closest: np.ndarray       # (N, 3) mm, closest point on the other boundary
    tolerance: np.ndarray     # (N,) mm, chordal tolerance before kappa
    sigma: float              # robust noise floor, mm
    tau_low: np.ndarray       # (N,) mm
    tau_high: np.ndarray      # (N,) mm
    shared_fraction: float    # reference nodes that coincide with an edited node


def noise_sigma(signed) -> float:
    signed = np.asarray(signed, dtype=np.float64)
    return float(MAD_TO_SIGMA * np.median(np.abs(signed - np.median(signed))))


def hysteresis(low, high, edges, count, min_nodes: int = 1) -> np.ndarray:
    """Nodes in ``low`` connected through ``low`` nodes to a node in ``high``, in
    connected regions of at least ``min_nodes`` nodes."""
    low = np.asarray(low, dtype=bool)
    high = np.asarray(high, dtype=bool) & low
    keep = low[edges[:, 0]] & low[edges[:, 1]]
    pairs = edges[keep]
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(count, count))
    _, label = connected_components(graph, directed=False)
    seeded = np.zeros(label.max() + 1, dtype=bool)
    seeded[label[high]] = True
    seeded &= np.bincount(label[low], minlength=len(seeded)) >= min_nodes
    return low & seeded[label]


def shared_node_fraction(reference: Surface, edited: Surface) -> float:
    """Fraction of reference nodes with an edited node at the same position (1e-6 of the size)."""
    scale = float(np.linalg.norm(reference.points.max(0) - reference.points.min(0)))
    distance, _ = cKDTree(edited.points).query(reference.points)
    return float((distance <= 1e-6 * scale).mean())


def _detect(nodes: Surface, other: Surface, tolerance_of, k_low, k_high, floor_mm, kappa, min_nodes, shared):
    closest, cell = other.closest(nodes.points)
    displacement = nodes.points - closest
    distance = np.linalg.norm(displacement, axis=1)
    signed = np.einsum("ij,ij->i", displacement, other.normals[cell])
    tolerance = tolerance_of(cell)
    # Noise and chordal tolerance add. On a uniformly curved part (the B2 pipe) sigma
    # also absorbs some sag, which costs sensitivity there; every alternative tried
    # (sigma beyond the tolerance, sigma on near-flat nodes, max instead of sum, any
    # kappa) flagged nodes on no-edit pairs. See verify_change_node_detection_benchmarks.py.
    sigma = noise_sigma(signed)
    tau_low = max(k_low * sigma, floor_mm) + kappa * tolerance
    tau_high = max(k_high * sigma, 2.0 * floor_mm) + 2.0 * kappa * tolerance
    changed = hysteresis(distance > tau_low, distance > tau_high, nodes.edges(), len(nodes.points), min_nodes)
    return ChangeDetection(changed, distance, signed, displacement, closest, tolerance, sigma, tau_low, tau_high,
                           shared)


def detect_edit(reference: Surface, edited: Surface, k_low: float = 3.0, k_high: float = 6.0,
                floor_mm: float = 0.01, kappa: float = 1.0, feature_deg: float = 45.0,
                min_nodes: int | None = None) -> tuple[ChangeDetection, ChangeDetection]:
    """(changes at the reference boundary nodes, changes at the edited boundary nodes).

    floor_mm: smallest deviation reported on a noise-free pair (the hood edits are
    binned to 0.01 mm). kappa: multiple of the chordal tolerance (0 = none).
    feature_deg: joints sharper than this are kept corners / feature lines.
    min_nodes: smallest changed region kept (default 10 on a surface, 3 on a curve).
    """
    if reference.dim != edited.dim:
        raise ValueError("Both boundaries must be curves, or both surfaces")
    min_nodes = MIN_NODES[reference.dim] if min_nodes is None else min_nodes
    if not 0 < k_low <= k_high or floor_mm <= 0 or kappa < 0 or min_nodes < 1:
        raise ValueError("Need 0 < k_low <= k_high, floor_mm > 0, kappa >= 0 and min_nodes >= 1")
    shared = shared_node_fraction(reference, edited)
    if shared >= SHARED_NODE_FRACTION:
        kappa = 0.0  # one discretization: no chordal error between the two
    curvature_node, curvature_cell = reference.curvature(feature_deg)
    size_reference_node, size_reference_cell = reference.element_size()
    size_edited_node, size_edited_cell = edited.element_size()
    options = (k_low, k_high, floor_mm, kappa, min_nodes, shared)
    on_reference = _detect(
        reference, edited,
        lambda cell: curvature_node * (size_reference_node ** 2 + size_edited_cell[cell] ** 2) / 8.0,
        *options)
    on_edited = _detect(
        edited, reference,
        lambda cell: curvature_cell[cell] * (size_edited_node ** 2 + size_reference_cell[cell] ** 2) / 8.0,
        *options)
    return on_reference, on_edited


def detect_mesh_edit(reference_points, reference_elements, edited_points, edited_elements,
                     kind: str = "auto", **options):
    """detect_edit on the two meshes' boundaries.

    Returns (on_reference, on_edited, reference_nodes, edited_nodes): the detections
    are per boundary node, and *_nodes give each boundary node's index in its mesh.
    """
    reference, reference_nodes = mesh_boundary(reference_points, reference_elements, kind)
    edited, edited_nodes = mesh_boundary(edited_points, edited_elements, kind)
    on_reference, on_edited = detect_edit(reference, edited, **options)
    return on_reference, on_edited, reference_nodes, edited_nodes
