"""
local_edit_doe.py
=================
Design of experiments for learning LOCAL design changes: 51 beam-notch edits of
hood design 6 (HoodImpact_31.inp, cluster C) and the impact locations each one
is solved at. Pure numpy/scipy, no Abaqus needed.

Designs
-------
Every edit changes the depth of an inner-panel hat-section beam over a short
stretch ("notch") and is mirror-symmetric about the hood centreline y =
MIRROR_Y, like the delivered designs. The notch math is that of
cae_local_edit_designs.py (same result to 1e-4 mm on N1-N7):

    along, across = node position in the beam frame at (x0, y0, axis)
    w_len = 0.5 * (1 + cos(pi * along / half_length))   |along| < half_length
    w_gap = clip((gap - G0) / (g1 - G0), 0, 1)          gap = skin z - node z
    dz    = DZ * w_len * w_gap                           |across| <= HALF_WIDTH

DZ > 0 raises the channel floor (shallower, softer beam), DZ < 0 lowers it.
12 sites x 4 variants + 3 combinations:

    S  standard  largest raise keeping every edge >= 5.25 mm, L = 60 mm
    H  half      half of S (at diagonal x = -130, where S is only 7 mm, the
                 slot holds a -10 mm deeper beam instead)
    D  deeper    -20 mm (even site index) or -10 mm (odd), L = 60 mm
    L  length    largest raise at L = 35 mm, or L = 90 mm where the longer box
                 stays clear of beam junctions
    C  two standard notches at once (additivity checks)

N1-N7 keep the names and parameters of the first batch; N8-N51 are new. The
base design is "D6".

Runs
----
Every edited design is solved at the same number of locations (RUNS_PER_DESIGN
= 20), chosen from the 142 Euro-NCAP grid locations by plan distance to the
nearest node moved by more than 0.5 mm:

    near  16  the 16 nearest locations on the +y half and the centreline
              (col_k >= 0); the 16th lies 115-278 mm away (median 182)
    twin   2  the mirror images (col_k -> -col_k) of the two nearest
              off-centre near locations: checks the symmetry assumption and
              measures the mesh-asymmetry noise where the signal is largest
    far    2  controls 300-450 mm from the edit (expected: no change), spread
              apart and used evenly across designs

The base design D6 is solved at all 142 locations, so every edited run has its
reference under identical settings, and design 6 has a complete clean map.

Solver settings (applied by make_local_edit_decks.py, identical for every run):
* *DYNAMIC, EXPLICIT, DIRECT USER CONTROL at 4.5e-7 s. The deck's variable mass
  scaling keeps every element's stable increment >= 5e-7 s, so this is stable
  for every design, and all designs share the same increments (no time-step
  artifact between designs).
* gap start: the headform is translated so its contact surface starts 1.0 mm
  clear of the hood outer contact surface (no initial overclosure). The outer
  skin is the same in every design, so the placement depends on the location
  only.

Run numbering: an 11-run pilot first (N1 and N2 at their two nearest locations
and first twin, N1 at its first far control, D6 at the same locations), then
D6 at its other locations, then N1..N51 with their 20 runs each (near by
distance, twins, far).

Usage (from the Code directory):
    python abaqus_scripts/local_edit_doe.py            # plan + write the csvs
    python abaqus_scripts/local_edit_doe.py --check    # recompute, compare with the csvs

Writes Data/HoodImpact_LocalEdits/:
    local_edit_doe.csv        one row per notch (design, site, x0, y0, axis, g1, dz, half_length, mirror)
    local_edit_designs.csv    one row per design (variant, nodes moved, checks)
    manifest_local_edits.csv  one row per run (design, location, role, distance, headform shift)
"""
import argparse
import csv
import math
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from inp_geom import Deck, OuterSurface, HOOD_OUTER_ELSET, HEADFORM_RADIUS  # noqa: E402

BASE_INP = os.path.join(CODE_ROOT, "Data", "HoodImpact_60_IndustryLike", "inp_files", "HoodImpact_31.inp")
LOCS_CSV = os.path.join(CODE_ROOT, "Data", "HoodImpact_1704_EuroNCAP", "impact_locations_142.csv")
OUT_DIR = os.path.join(CODE_ROOT, "Data", "HoodImpact_LocalEdits")
DOE_CSV, DESIGNS_CSV, MANIFEST_CSV = "local_edit_doe.csv", "local_edit_designs.csv", "manifest_local_edits.csv"

INNER_ELSET = "Hood_Inner-1-2"
BASE_DESIGN = "D6"
MIRROR_Y = 155.3432          # hood centreline (impact grid symmetry line)
HALF_WIDTH = 45.0            # mm across the beam
G0 = 3.0                     # mm, flange gap below the skin
MAX_GAP = 45.0               # deeper nodes belong to the perimeter frame
BIN_MM = 0.01                # offsets binned like the CAE script
MIN_EDGE_MM = 5.25           # deck minimum is 5.083 mm
FASTENER_ZONE_MM = 10.0
MIN_SKIN_CLEARANCE_MM = 2.0
SIGNIFICANT_MM = 0.5         # a node counts as moved for distances above this
N_NEAR, N_TWIN, N_FAR = 16, 2, 2
RUNS_PER_DESIGN = N_NEAR + N_TWIN + N_FAR
FAR_MM, FAR_MAX_MM = 300.0, 450.0
FIXED_DT = 4.5e-7            # s
CLEARANCE_MM = 1.0           # clear gap between contact surfaces at t = 0
CONTACT_OFFSET_MM = 1.25     # hood outer 1.5 mm / 2 + headform layer 1.0 mm / 2
PILOT_DESIGNS = ("N1", "N2")
VARIANT_NAMES = {"S": "standard", "H": "half raise", "D": "deeper beam", "L": "length", "C": "combination"}


# --------------------------------------------------------------------- hood
class Hood(object):
    """Design 6: inner-panel nodes, their gap below the skin, edit checks."""

    def __init__(self, path=BASE_INP, keep_lines=False):
        self.deck = Deck(path, keep_lines=keep_lines)
        d = self.deck
        self.inner_eids = list(d.elsets[INNER_ELSET])
        self.labels = np.array(sorted(d.elset_nodes(INNER_ELSET)))
        self.xyz = np.array([d.nodes[i] for i in self.labels], float)
        outer_ids = sorted(d.elset_nodes(HOOD_OUTER_ELSET))
        self.outer = np.array([d.nodes[i] for i in outer_ids], float)
        self._otree = cKDTree(self.outer[:, :2])
        self.gap = np.array([self.skin_z(x, y) - z for x, y, z in self.xyz])
        self.node_el = {}
        for e in self.inner_eids:
            for n in d.elems[e]:
                self.node_el.setdefault(n, []).append(e)

    def skin_z(self, x, y):
        """Skin height as in the CAE script: LSQ plane through skin nodes within 12 mm."""
        near = self.outer[self._otree.query_ball_point((x, y), 12.0)]
        if len(near) < 3:
            return float(self.outer[self._otree.query((x, y))[1], 2])
        a = np.c_[np.ones(len(near)), near[:, 0] - x, near[:, 1] - y]
        return float(np.linalg.lstsq(a, near[:, 2], rcond=None)[0][0])

    def min_edge(self, dz):
        """Smallest edge of the inner-panel elements touching a moved node."""
        best = np.inf
        for e in {e for n in dz for e in self.node_el[n]}:
            c = np.array([self.deck.nodes[n] for n in self.deck.elems[e]], float)
            c[:, 2] += [dz.get(n, 0.0) for n in self.deck.elems[e]]
            best = min(best, np.linalg.norm(c - np.roll(c, 1, 0), axis=1).min())
        return best


def site_frame(hood, site):
    a = math.radians(site["axis"])
    rx, ry = hood.xyz[:, 0] - site["x0"], hood.xyz[:, 1] - site["y0"]
    return rx * math.cos(a) + ry * math.sin(a), -rx * math.sin(a) + ry * math.cos(a)


def mirror_site(site):
    return dict(site, y0=2.0 * MIRROR_Y - site["y0"], axis=-site["axis"])


def notch_offsets(hood, site, dz_max, half_length):
    """({label: raw dz}, deepest base gap) of one notch."""
    along, across = site_frame(hood, site)
    out, deepest = {}, 0.0
    for k in np.where((np.abs(along) < half_length) & (np.abs(across) <= HALF_WIDTH))[0]:
        w = 0.5 * (1.0 + math.cos(math.pi * abs(along[k]) / half_length))
        w *= min(max((hood.gap[k] - G0) / (site["g1"] - G0), 0.0), 1.0)
        if w > 0.0:
            out[int(hood.labels[k])] = dz_max * w
            deepest = max(deepest, hood.gap[k])
    return out, deepest


def design_offsets(hood, notches):
    """notches = [(site, dz, half_length)], each site expanded to its mirror twin
    when site['mirror']. Returns ({label: binned dz}, [per-notch info])."""
    raw, info = {}, []
    for site, dz, hl in notches:
        for s in [site] + ([mirror_site(site)] if site["mirror"] else []):
            off, deepest = notch_offsets(hood, s, dz, hl)
            if not off:
                raise ValueError("notch at (%.1f, %.1f) moves no node" % (s["x0"], s["y0"]))
            info.append(dict(x0=s["x0"], y0=s["y0"], axis=s["axis"], deepest=deepest,
                             n=sum(1 for v in off.values() if abs(v) >= 0.5 * BIN_MM)))
            for lab, v in off.items():
                raw[lab] = raw.get(lab, 0.0) + v
    binned = {}
    for lab, v in raw.items():
        key = int(round(v / BIN_MM))
        if key:
            binned[lab] = key * BIN_MM
    return binned, info


def junction_nodes(hood, site, half_length):
    """Nodes inside the notch box that belong to another channel (> 31 mm across, gap > 8 mm)."""
    count = 0
    for s in [site] + ([mirror_site(site)] if site["mirror"] else []):
        along, across = site_frame(hood, s)
        count += int(((np.abs(along) < half_length) & (np.abs(across) <= HALF_WIDTH)
                      & (np.abs(across) > 31) & (hood.gap > 8)).sum())
    return count


class EditChecks(object):
    """The checks of apply_node_edits.py, evaluated on a dz map."""

    def __init__(self, hood):
        from apply_node_edits import attachment_node_sets, connector_node_ids
        d = hood.deck if hood.deck.lines is not None else Deck(BASE_INP, keep_lines=True)
        self.hood = hood
        self.conn = cKDTree(np.array([d.nodes[i] for i in sorted(connector_node_ids(d))], float))
        self.attach = set().union(*attachment_node_sets(d).values())
        self.impactor = d.impactor_node_ids()
        self.skin = OuterSurface(d)

    def __call__(self, dz, notches_info=()):
        labs = sorted(dz)
        P = np.array([self.hood.deck.nodes[n] for n in labs], float)
        v = np.array([dz[n] for n in labs])
        Q = P.copy()
        Q[:, 2] += v
        gap_old = self.skin.surf_z(P[:, 0], P[:, 1]) - P[:, 2]
        gap_new = self.skin.surf_z(Q[:, 0], Q[:, 1]) - Q[:, 2]
        out = dict(n_nodes=len(labs), min_edge=self.hood.min_edge(dz),
                   fastener_mm=float(min(self.conn.query(P)[0].min(), self.conn.query(Q)[0].min())),
                   skin_min_gap=float(gap_new.min()), dz_min=float(v.min()), dz_max=float(v.max()))
        problems = []
        if out["min_edge"] < MIN_EDGE_MM:
            problems.append("min edge %.3f mm" % out["min_edge"])
        if out["fastener_mm"] < FASTENER_ZONE_MM:
            problems.append("fastener %.1f mm" % out["fastener_mm"])
        if set(labs) & self.attach:
            problems.append("%d BC/coupling nodes" % len(set(labs) & self.attach))
        if set(labs) & self.impactor:
            problems.append("headform nodes")
        if ((gap_new < MIN_SKIN_CLEARANCE_MM) & (gap_new < gap_old - 1e-9)).any():
            problems.append("skin clearance %.2f mm" % gap_new.min())
        deepest = max([i["deepest"] for i in notches_info] or [0.0])
        if deepest > MAX_GAP:
            problems.append("frame node (gap %.1f mm)" % deepest)
        out["problems"] = problems
        return out


# --------------------------------------------------------------------- designs
def _diag(x):
    a, b = (0.3573, 412.59) if x < 290 else (0.3726, 408.55)
    return dict(beam="diag", x0=float(x), y0=round(a * x + b, 2), axis=round(math.degrees(math.atan(a)), 2), mirror=True)


def _horiz(x):
    return dict(beam="horiz", x0=float(x), y0=155.34, axis=0.0, mirror=False)


# Beam channel-floor axes fitted to design 6 (upper half; lower = mirror):
#   upper diagonal y = 0.3573 x + 412.59 (x < 290), 0.3726 x + 408.55 (x > 290)
#   horizontal y = 155.34; vertical x = -0.0206 y + 281.60
# Usable stretches (clear of junction flares, the frame and fasteners):
#   diagonal x -140..100 and 390..530, horizontal -140..120 and 420..470,
#   vertical y ~ 320-350. g1 = channel-floor gap; the first batch's values are kept.
SITES = [
    ("diag_xm130", _diag(-130.0)), ("diag_xm60", dict(_diag(-60.0), y0=391.15, g1=28.5)),
    ("diag_x30", dict(_diag(30.0), y0=423.31, g1=31.0)), ("diag_x100", _diag(100.0)),
    ("diag_x400", _diag(400.0)), ("diag_x470", dict(_diag(470.0), y0=583.67, g1=39.0)),
    ("horiz_xm120", _horiz(-120.0)), ("horiz_xm40", _horiz(-40.0)),
    ("horiz_x30", dict(_horiz(30.0), g1=29.5)), ("horiz_x100", _horiz(100.0)),
    ("horiz_x450", _horiz(450.0)),
    ("vert_y330", dict(beam="vert", x0=274.80, y0=330.0, axis=91.18, mirror=True, g1=34.5)),
]
FIRST_BATCH = {  # name -> (site, variant, dz, half_length); N4 is a combination
    "N1": ("diag_x30", "S", 12.5, 60.0), "N2": ("diag_x30", "D", -20.0, 60.0),
    "N3": ("horiz_x30", "S", 12.5, 60.0), "N5": ("diag_xm60", "S", 10.0, 60.0),
    "N6": ("vert_y330", "S", 14.0, 60.0), "N7": ("diag_x470", "S", 15.0, 60.0),
}
COMBINATIONS = [("N4", ("diag_x30", "horiz_x30")), (None, ("diag_x100", "horiz_x100")),
                (None, ("diag_xm130", "horiz_xm120"))]


def floor_gap(hood, site):
    along, across = site_frame(hood, site)
    m = (np.abs(along) < 10) & (np.abs(across) < 15) & (hood.gap > 15)
    return round(float(np.median(hood.gap[m])) * 2) / 2


def max_raise(hood, site, half_length):
    best = None
    for dz in np.arange(4.0, 24.01, 0.5):
        off, _ = design_offsets(hood, [(site, float(dz), half_length)])
        if hood.min_edge(off) < MIN_EDGE_MM:
            break
        best = float(dz)
    return best


def plan_designs(hood):
    """[{name, kind, variant, notches: [(site_key, dz, half_length)]}] and the sites."""
    sites = {}
    for key, s in SITES:
        s = dict(s, key=key)
        s.setdefault("g1", floor_gap(hood, s))
        sites[key] = s
    designs = []
    for name, (key, var, dz, hl) in FIRST_BATCH.items():
        designs.append(dict(name=name, kind="single", variant=var, notches=[(key, dz, hl)]))
    standard = {}
    n = 8
    for i, (key, _) in enumerate(SITES):
        s = sites[key]
        have = {d["variant"]: d for d in designs if d["kind"] == "single" and d["notches"][0][0] == key}
        std = have["S"]["notches"][0][1] if "S" in have else max_raise(hood, s, 60.0)
        standard[key] = std
        half = round(std / 2 * 2) / 2
        deep = -20.0 if i % 2 == 0 else -10.0
        alt = 90.0 if (i % 2 and junction_nodes(hood, s, 90.0) == 0) else 35.0
        plan = [("S", std, 60.0), ("H", half if half >= 5.0 else (-10.0 if deep == -20.0 else -20.0), 60.0),
                ("D", deep, 60.0), ("L", None, alt)]
        for var, dz, hl in plan:
            if var in have:
                continue
            designs.append(dict(name="N%d" % n, kind="single", variant=var,
                                notches=[(key, dz if dz is not None else max_raise(hood, s, hl), hl)]))
            n += 1
    for name, keys in COMBINATIONS:
        if name is None:
            name = "N%d" % n
            n += 1
        designs.append(dict(name=name, kind="combo", variant="C", notches=[(k, standard[k], 60.0) for k in keys]))
    designs.sort(key=lambda d: int(d["name"][1:]))
    return designs, sites


def describe(sites, notches):
    beam = {"diag": "diagonal beams", "horiz": "horizontal centre beam", "vert": "vertical cross beam"}
    parts = []
    for key, dz, hl in notches:
        s = sites[key]
        where = ("y = %g" % s["y0"]) if s["beam"] == "vert" else ("x = %g" % s["x0"])
        verb = ("%g mm shallower" % dz) if dz > 0 else ("%g mm deeper" % -dz)
        parts.append("%s at %s: %s, +-%g mm" % (beam[s["beam"]], where, verb, hl))
    return "; ".join(parts)


# --------------------------------------------------------------------- locations
def load_locations(path=LOCS_CSV):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [dict(loc=int(r["loc"]), x=float(r["X1"]), y=float(r["X2"]), col_k=int(r["col_k"])) for r in rows]


def location_distances(hood, dz, locations):
    """Plan distance from each location to the nearest node moved > SIGNIFICANT_MM."""
    P = np.array([hood.deck.nodes[n][:2] for n, v in dz.items() if abs(v) > SIGNIFICANT_MM])
    tree = cKDTree(P)
    return np.array([tree.query((l["x"], l["y"]))[0] for l in locations])


def select_locations(designs, distances, locations):
    """Per design: near / twin / far location ids, the same count for every design."""
    idx = {l["loc"]: i for i, l in enumerate(locations)}

    def mirror_of(l):
        """The location at -col_k with the closest x (the grid pairs agree to 0.36 mm)."""
        c = min((m for m in locations if m["col_k"] == -l["col_k"]), key=lambda m: abs(m["x"] - l["x"]))
        if abs(c["x"] - l["x"]) > 1.0 or abs((c["y"] - MIRROR_Y) + (l["y"] - MIRROR_Y)) > 1.0:
            raise ValueError("no mirror twin for location %d" % l["loc"])
        return c["loc"]

    use = {}
    plan = {}
    for d in designs:
        dist = distances[d["name"]]
        half = [i for i in np.argsort(dist, kind="stable") if locations[i]["col_k"] >= 0]
        near = [locations[i]["loc"] for i in half[:N_NEAR]]
        twins = []
        for loc in near:
            l = locations[idx[loc]]
            if l["col_k"] > 0 and len(twins) < N_TWIN:
                twins.append(mirror_of(l))
        # controls in the FAR_MM..FAR_MAX_MM band: far from the edit, but still
        # over the same part of the hood rather than at a distant corner
        cand = [l["loc"] for i, l in enumerate(locations) if l["col_k"] >= 0 and FAR_MM <= dist[i] <= FAR_MAX_MM]
        if len(cand) < N_FAR:
            cand = [l["loc"] for i, l in enumerate(locations) if l["col_k"] >= 0 and dist[i] >= FAR_MM]
        far = []
        for _ in range(N_FAR):
            def score(loc):
                if not far:   # least used, then closest to the middle of the band
                    return (use.get(loc, 0), abs(dist[idx[loc]] - 0.5 * (FAR_MM + FAR_MAX_MM)), loc)
                spread = min(np.hypot(locations[idx[loc]]["x"] - locations[idx[f]]["x"],
                                      locations[idx[loc]]["y"] - locations[idx[f]]["y"]) for f in far)
                return (use.get(loc, 0), -spread, loc)
            pick = min((c for c in cand if c not in far), key=score)
            far.append(pick)
            use[pick] = use.get(pick, 0) + 1
        if len(near) != N_NEAR or len(twins) != N_TWIN or len(far) != N_FAR:
            raise ValueError("%s: %d near, %d twins, %d far" % (d["name"], len(near), len(twins), len(far)))
        plan[d["name"]] = dict(near=near, twin=twins, far=far)
    return plan


def headform_shifts(locations, path=BASE_INP):
    """Per location: rigid headform shift (dx, dy, dz) from its position in the base
    deck to the location, with the contact surfaces CLEARANCE_MM apart."""
    deck = Deck(path)
    ref = np.array(deck.nodes[deck.rigid_ref], float)
    outer = np.array([deck.nodes[i] for i in sorted(deck.elset_nodes(HOOD_OUTER_ELSET))], float)
    skin = OuterSurface(deck)
    target = -(CONTACT_OFFSET_MM + CLEARANCE_MM)
    out = {}
    for l in locations:
        z0 = ref[2] + float(skin.surf_z(l["x"], l["y"])) - float(skin.surf_z(ref[0], ref[1]))
        centre = np.array([l["x"], l["y"], z0])
        near = outer[np.linalg.norm(outer[:, :2] - centre[:2], axis=1) < HEADFORM_RADIUS + 30.0]
        lift = 0.0
        for _ in range(60):
            pen = HEADFORM_RADIUS - float(np.linalg.norm(near - (centre + [0, 0, lift]), axis=1).min())
            if abs(pen - target) <= 1e-4:
                break
            lift += pen - target
        else:
            raise RuntimeError("headform placement did not converge at location %d" % l["loc"])
        out[l["loc"]] = (l["x"] - ref[0], l["y"] - ref[1], z0 + lift - ref[2], pen)
    return out


# --------------------------------------------------------------------- csv i/o
def read_doe(path):
    """{design: [(site dict, dz, half_length)]} from local_edit_doe.csv."""
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            site = dict(key=r["site"], x0=float(r["x0"]), y0=float(r["y0"]), axis=float(r["axis_deg"]),
                        g1=float(r["g1_mm"]), mirror=r["mirror"] == "1")
            out.setdefault(r["design"], []).append((site, float(r["dz_mm"]), float(r["half_length_mm"])))
    return out


def build_plan(hood):
    designs, sites = plan_designs(hood)
    checks = EditChecks(hood)
    locations = load_locations()
    distances, results = {}, {}
    for d in designs:
        notches = [(sites[k], dz, hl) for k, dz, hl in d["notches"]]
        dz, info = design_offsets(hood, notches)
        res = checks(dz, info)
        junction = sum(junction_nodes(hood, sites[k], hl) for k, _, hl in d["notches"])
        if junction:
            res["problems"].append("%d junction nodes" % junction)
        results[d["name"]] = res
        distances[d["name"]] = location_distances(hood, dz, locations)
    plan = select_locations(designs, distances, locations)
    return designs, sites, locations, distances, results, plan


def run_rows(designs, locations, distances, plan, shifts):
    """Manifest rows in run order: pilot, base, then the designs."""
    idx = {l["loc"]: i for i, l in enumerate(locations)}
    # Pilot (11 runs, one round of 12 lanes): N1 (standard raise) and N2 (20 mm
    # deeper, same site) at their two nearest locations and first twin, N1 also
    # at its first far control, and D6 at the same locations.
    pilot = {name: [(plan[name]["near"][0], "near"), (plan[name]["near"][1], "near"),
                    (plan[name]["twin"][0], "twin")] for name in PILOT_DESIGNS}
    pilot[PILOT_DESIGNS[0]].append((plan[PILOT_DESIGNS[0]]["far"][0], "far"))
    pilot_base = sorted({loc for runs in pilot.values() for loc, _ in runs})
    entries = [(BASE_DESIGN, loc, "base", True) for loc in pilot_base]
    for name in PILOT_DESIGNS:
        entries += [(name, loc, role, True) for loc, role in pilot[name]]
    done = {(e[0], e[1]) for e in entries}
    for l in locations:
        if (BASE_DESIGN, l["loc"]) not in done:
            entries.append((BASE_DESIGN, l["loc"], "base", False))
    for d in designs:
        for role in ("near", "twin", "far"):
            for loc in plan[d["name"]][role]:
                if (d["name"], loc) not in done:
                    entries.append((d["name"], loc, role, False))
    rows = []
    for run, (name, loc, role, pilot) in enumerate(entries, start=1):
        l = locations[idx[loc]]
        dist = "" if name == BASE_DESIGN else "%.1f" % distances[name][idx[loc]]
        dx, dy, dz, pen = shifts[loc]
        rows.append(dict(run=run, design=name, location=loc, role=role, pilot=int(pilot), col_k=l["col_k"],
                         x_mm="%.4f" % l["x"], y_mm="%.4f" % l["y"], dist_to_edit_mm=dist,
                         headform_dx_mm="%.6f" % dx, headform_dy_mm="%.6f" % dy, headform_dz_mm="%.6f" % dz,
                         sphere_pen_mm="%.4f" % pen, contact_clearance_mm="%.4f" % (-pen - CONTACT_OFFSET_MM),
                         time_increment_s="%r" % FIXED_DT))
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--check", action="store_true", help="recompute and compare with the existing csvs")
    args = ap.parse_args()

    hood = Hood(keep_lines=True)
    designs, sites, locations, distances, results, plan = build_plan(hood)
    bad = {n: r["problems"] for n, r in results.items() if r["problems"]}
    if bad:
        raise SystemExit("designs failing the edit checks: %s" % bad)
    shifts = headform_shifts(locations)
    rows = run_rows(designs, locations, distances, plan, shifts)

    doe_rows, design_rows = [], []
    for d in designs:
        for k, dz, hl in d["notches"]:
            s = sites[k]
            doe_rows.append(dict(design=d["name"], site=k, beam=s["beam"], x0="%.2f" % s["x0"], y0="%.2f" % s["y0"],
                                 axis_deg="%.2f" % s["axis"], g1_mm="%.1f" % s["g1"], dz_mm="%g" % dz,
                                 half_length_mm="%g" % hl, half_width_mm="%g" % HALF_WIDTH, g0_mm="%g" % G0,
                                 mirror=int(s["mirror"])))
        r = results[d["name"]]
        design_rows.append(dict(design=d["name"], kind=d["kind"], variant=d["variant"],
                                variant_name=VARIANT_NAMES[d["variant"]], description=describe(sites, d["notches"]),
                                n_nodes_moved=r["n_nodes"], dz_min_mm="%.2f" % r["dz_min"], dz_max_mm="%.2f" % r["dz_max"],
                                min_edge_mm="%.3f" % r["min_edge"], fastener_mm="%.1f" % r["fastener_mm"],
                                skin_min_gap_mm="%.2f" % r["skin_min_gap"],
                                near=" ".join(map(str, plan[d["name"]]["near"])),
                                twin=" ".join(map(str, plan[d["name"]]["twin"])),
                                far=" ".join(map(str, plan[d["name"]]["far"]))))
    outputs = {DOE_CSV: doe_rows, DESIGNS_CSV: design_rows, MANIFEST_CSV: rows}
    if args.check:
        same = True
        for name, new in outputs.items():
            with open(os.path.join(args.out_dir, name)) as f:
                old = list(csv.DictReader(f))
            ok = old == [{k: str(v) for k, v in r.items()} for r in new]
            same &= ok
            print("%-26s %s" % (name, "identical" if ok else "DIFFERS"))
        raise SystemExit(0 if same else 1)
    os.makedirs(args.out_dir, exist_ok=True)
    for name, data in outputs.items():
        write_csv(os.path.join(args.out_dir, name), data)

    n_base = sum(r["design"] == BASE_DESIGN for r in rows)
    print("%d designs x %d runs + %s at %d locations = %d runs (pilot: runs 1-%d)"
          % (len(designs), RUNS_PER_DESIGN, BASE_DESIGN, n_base, len(rows), sum(r["pilot"] for r in rows)))
    pen = np.array([s[3] for s in shifts.values()])
    print("headform start: sphere penetration %.3f..%.3f mm (target %.2f) -> %.2f mm clear"
          % (pen.min(), pen.max(), -(CONTACT_OFFSET_MM + CLEARANCE_MM), CLEARANCE_MM))
    for d in designs:
        p = plan[d["name"]]
        dist = distances[d["name"]]
        idx = {l["loc"]: i for i, l in enumerate(locations)}
        print("%-4s %s  near %4.0f-%4.0f mm  twins %s  far %s"
              % (d["name"], d["variant"], dist[idx[p["near"][0]]], dist[idx[p["near"][-1]]], p["twin"], p["far"]))
    print("wrote %s" % ", ".join(os.path.join(args.out_dir, n) for n in outputs))


if __name__ == "__main__":
    main()
