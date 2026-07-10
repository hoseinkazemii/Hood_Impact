"""
inp_geom.py
===========
Shared parsing + geometry utilities for the HoodImpact decks, used by
    * euroncap_grid.py          (define the 50 Euro-NCAP-style impact locations)
    * generate_impact_decks.py  (write the 600 repositioned-headform decks)

Deck facts these utilities rely on (verified on HoodImpact_1.inp / _31.inp):
    * flat .inp (no parts/instances), units mm / tonne / s
    * ONE *NODE block containing hood + impactor nodes, lines "id, x, y, z"
    * shell elements in *ELEMENT, TYPE=S3R and TYPE=S4 blocks
    * hood outer skin      = ELSET "Hood_Outer-1-2"
    * impactor (headform)  = ELSET "Rigid Body1_2-1-3" (rigid, ~286 shells)
    * rigid ref node       = *RIGID BODY, REF NODE=<id>, ELSET="Rigid Body1_2-1-3"
      (id differs between mesh families: 44166 for runs 1-30, 45368 for 31-60)
    * NSET "Initial Velocity1" = the ref node only (velocity DOF 3 = -11100 mm/s)
    * ref node sits exactly at the delivered impact coordinate (X1, X2, X3)
      and the headform is a 165 mm diameter sphere: mesh zmax = X3 + 82.5
"""
import re
from collections import defaultdict

import numpy as np

HOOD_OUTER_ELSET = "Hood_Outer-1-2"
IMPACTOR_ELSET = "Rigid Body1_2-1-3"
VELOCITY_NSET = "Initial Velocity1"
HEADFORM_RADIUS = 82.5  # mm (165 mm adult pedestrian headform)


# ---------------------------------------------------------------------------
# deck parsing
# ---------------------------------------------------------------------------
class Deck(object):
    """Parsed .inp: nodes, shell connectivity, elsets/nsets, rigid ref node.

    If keep_lines=True, also stores the raw file lines and the line index of
    every node line so the deck can be rewritten with translated nodes while
    keeping every other byte identical.
    """

    def __init__(self, path, keep_lines=False):
        self.path = path
        self.nodes = {}            # id -> (x, y, z)
        self.elems = {}            # eid -> tuple(node ids)   (S3R + S4 only)
        self.elsets = {}           # name -> [eids]
        self.nsets = {}            # name -> [nids]
        self.rigid_ref = None      # ref node id of the impactor rigid body
        self.lines = [] if keep_lines else None
        self.node_line_idx = {} if keep_lines else None  # node id -> line index
        self._parse(keep_lines)

    def _parse(self, keep_lines):
        mode = None          # 'node' | 'elem' | 'elset' | 'nset' | None
        set_name = None
        set_generate = False

        rigid_re = re.compile(
            r'^\*RIGID BODY.*REF NODE=(\d+).*ELSET="%s"' % re.escape(IMPACTOR_ELSET),
            re.IGNORECASE)

        # newline="" keeps the original line terminators (the delivered decks
        # are CRLF); generated decks then match byte-for-byte outside the
        # translated node lines regardless of the OS this runs on.
        with open(self.path, "r", newline="") as f:
            for i, line in enumerate(f):
                if keep_lines:
                    self.lines.append(line)
                s = line.strip()
                if not s or s.startswith("**"):
                    continue

                if s.startswith("*"):
                    u = s.upper()
                    m = rigid_re.match(s)
                    if m:
                        self.rigid_ref = int(m.group(1))
                        mode = None
                    elif u == "*NODE":
                        mode = "node"
                    elif u.startswith("*ELEMENT") and ("TYPE=S3R" in u or "TYPE=S4" in u.replace(" ", "")):
                        mode = "elem"
                    elif u.startswith("*ELSET"):
                        m2 = re.search(r'ELSET="([^"]+)"', s)
                        set_name = m2.group(1) if m2 else None
                        set_generate = "GENERATE" in u
                        if set_name is not None:
                            self.elsets.setdefault(set_name, [])
                        mode = "elset" if set_name else None
                    elif u.startswith("*NSET"):
                        m2 = re.search(r'NSET="([^"]+)"', s)
                        set_name = m2.group(1) if m2 else None
                        set_generate = "GENERATE" in u
                        if set_name is not None:
                            self.nsets.setdefault(set_name, [])
                        mode = "nset" if set_name else None
                    else:
                        mode = None
                    continue

                if mode == "node":
                    p = s.split(",")
                    if len(p) >= 4:
                        nid = int(p[0])
                        self.nodes[nid] = (float(p[1]), float(p[2]), float(p[3]))
                        if keep_lines:
                            self.node_line_idx[nid] = i
                elif mode == "elem":
                    p = s.split(",")
                    if len(p) >= 4:
                        self.elems[int(p[0])] = tuple(int(a) for a in p[1:] if a.strip())
                elif mode in ("elset", "nset"):
                    target = self.elsets if mode == "elset" else self.nsets
                    vals = [int(a) for a in s.split(",") if a.strip()]
                    if set_generate and len(vals) >= 2:
                        inc = vals[2] if len(vals) > 2 else 1
                        target[set_name].extend(range(vals[0], vals[1] + 1, inc))
                    else:
                        target[set_name].extend(vals)

    # -- convenience -------------------------------------------------------
    def elset_nodes(self, elset_name):
        """Union of node ids used by the elements of an elset."""
        out = set()
        for eid in self.elsets[elset_name]:
            out.update(self.elems[eid])
        return out

    def impactor_node_ids(self):
        """All node ids that must move with the headform."""
        ids = self.elset_nodes(IMPACTOR_ELSET)
        if self.rigid_ref is not None:
            ids.add(self.rigid_ref)
        ids.update(self.nsets.get(VELOCITY_NSET, []))
        return ids

    def coords(self, ids):
        ids = sorted(ids)
        return ids, np.array([self.nodes[i] for i in ids], float)


# ---------------------------------------------------------------------------
# hood outer surface: z(x, y) interpolation + boundary outline
# ---------------------------------------------------------------------------
class OuterSurface(object):
    """Linear interpolation of the hood outer skin height z(x, y) plus the
    plan-view boundary polygon of the outer panel."""

    def __init__(self, deck):
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        node_ids = sorted(deck.elset_nodes(HOOD_OUTER_ELSET))
        pts = np.array([deck.nodes[i] for i in node_ids], float)
        self.xy = pts[:, :2]
        self.z = pts[:, 2]
        self._lin = LinearNDInterpolator(self.xy, self.z)
        self._near = NearestNDInterpolator(self.xy, self.z)
        self.outline = _boundary_polygon(deck, node_ids)

    def surf_z(self, x, y):
        """z of the outer skin at (x, y); nearest-neighbour fallback at edges."""
        x = np.atleast_1d(np.asarray(x, float))
        y = np.atleast_1d(np.asarray(y, float))
        z = self._lin(x, y)
        bad = np.isnan(z)
        if bad.any():
            z[bad] = self._near(x[bad], y[bad])
        return z if z.size > 1 else float(z[0])


def _boundary_polygon(deck, outer_node_ids):
    """Ordered XY polygon (closed, (M,2)) of the outer panel's free boundary."""
    outer_ids = set(outer_node_ids)
    edge_count = defaultdict(int)
    for eid in deck.elsets[HOOD_OUTER_ELSET]:
        conn = deck.elems[eid]
        n = len(conn)
        for k in range(n):
            a, b = conn[k], conn[(k + 1) % n]
            edge_count[(min(a, b), max(a, b))] += 1
    boundary = [e for e, c in edge_count.items() if c == 1]

    adj = defaultdict(list)
    for a, b in boundary:
        adj[a].append(b)
        adj[b].append(a)

    unused = set(boundary)
    loops = []
    while unused:
        a, b = next(iter(unused))
        unused.discard((a, b))
        loop = [a, b]
        while True:
            cur, prev = loop[-1], loop[-2]
            nxt = None
            for cand in adj[cur]:
                e = (min(cur, cand), max(cur, cand))
                if cand != prev and e in unused:
                    nxt = cand
                    unused.discard(e)
                    break
            if nxt is None:
                break
            loop.append(nxt)
            if nxt == loop[0]:
                break
        loops.append(loop)

    def perimeter(loop):
        p = np.array([deck.nodes[i][:2] for i in loop])
        return np.linalg.norm(np.diff(p, axis=0), axis=1).sum()

    best = max(loops, key=perimeter)
    poly = np.array([deck.nodes[i][:2] for i in best], float)
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[:1]])
    return poly


# ---------------------------------------------------------------------------
# 2-D polygon helpers (no shapely dependency)
# ---------------------------------------------------------------------------
def points_inside(pts, poly):
    """Boolean mask: which pts (N,2) are inside the closed polygon (M,2)."""
    from matplotlib.path import Path
    return Path(poly).contains_points(np.atleast_2d(pts))


def dist_to_polyline(pts, poly):
    """Min distance from each pt (N,2) to the closed polyline poly (M,2)."""
    pts = np.atleast_2d(np.asarray(pts, float))
    a = poly[:-1]                       # (M-1, 2) segment starts
    ab = poly[1:] - a                   # (M-1, 2) segment vectors
    ab2 = (ab ** 2).sum(1)              # (M-1,)
    ab2[ab2 == 0] = 1e-30
    out = np.empty(len(pts))
    for i, p in enumerate(pts):         # N is small (grid candidates)
        t = np.clip(((p - a) * ab).sum(1) / ab2, 0.0, 1.0)
        proj = a + t[:, None] * ab
        out[i] = np.sqrt(((p - proj) ** 2).sum(1).min())
    return out


def polygon_x_span(poly, y):
    """(x_min, x_max) where the horizontal line at `y` crosses the polygon,
    or None if it does not cross."""
    xs = []
    for (x1, y1), (x2, y2) in zip(poly[:-1], poly[1:]):
        if (y1 - y) * (y2 - y) <= 0 and y1 != y2:
            t = (y - y1) / (y2 - y1)
            if 0.0 <= t <= 1.0:
                xs.append(x1 + t * (x2 - x1))
    if len(xs) < 2:
        return None
    return min(xs), max(xs)
