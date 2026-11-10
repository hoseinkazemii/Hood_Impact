"""
cae_local_edit_designs.py  --  runs INSIDE Abaqus/CAE (File > Run Script...)
===========================================================================
Builds local-edit hood designs from design 6 (cluster C, HoodImpact_31.inp)
for viewing in CAE. The designs (N1-N51) and their notch parameters are read
from Data/HoodImpact_LocalEdits/local_edit_doe.csv, written by
local_edit_doe.py; set ONLY to build a few of them (each design is a full
model copy). The Delta decks are written by make_local_edit_decks.py with the
same notch math (agreement 1e-4 mm), not from CAE. Every design changes the
DEPTH of inner-panel hat-section beams over a short stretch ("notch"); the
outer skin, headform, hinges and fasteners are not touched.

Every edit is mirror-symmetric about the hood centreline y = MIRROR_Y, like the
delivered designs: a notch on one diagonal (or on the upper half of the
vertical beam) always comes with the same notch at (x0, 2*MIRROR_Y - y0) with
the axis mirrored; the horizontal beam lies on the centreline itself. The mesh
is only approximately symmetric (mirrored nodes sit ~1 mm, at most 8.6 mm, from
a real node), so the two halves move slightly different node sets.

Why beams: most impact points sit over open holes in the inner panel, and
their response is set by the stiffness of the beams around the hole. The
delivered designs 7 and 11 (three notches each, channel floor raised 15-20 mm)
moved HIC by 26-42% at locations 30-110 mm away; nothing else in the data
changes the response as much.

Notch on one beam (computed from the BASE coordinates):
    along, across = node position relative to (X0, Y0) in the beam frame
    w_len  = 0.5 * (1 + cos(pi * along / L))   for |along| < L (half-length, per notch)
    w_gap  = clip((gap - G0) / (G1 - G0), 0, 1),  gap = outer-skin z - node z
    dz     = DZ * w_len * w_gap                    for |across| <= HALF_WIDTH
The flanges (gap 3 mm) stay under the skin, the channel floor (gap >= G1)
moves by the full DZ and the walls between are compressed or stretched
evenly. DZ > 0 raises the floor (shallower, softer beam), DZ < 0 lowers it
(deeper, stiffer beam). Each raise is the largest that keeps every new edge
>= 5.25 mm on BOTH halves, above the deck's smallest edge (5.083 mm), so the
stable time increment stays that of design 6 (design 7's 20 mm notches left
3 mm edges and took 20% more increments).

For every design the script
  * copies the base model to a new model named after the design,
  * moves the inner-panel nodes,
  * adds part sets EDIT_NODES / EDIT_ELEMENTS (for viewing),
  * writes <work dir>/<design>_edited_nodes.csv (label,x,y,z of every node)
and finally <work dir>/local_edit_designs.csv with the parameters.
CAE drops the deck's 62 fastener couplings on import, so do NOT write a deck
from CAE: patch each CSV into the original deck with apply_node_edits.py.

Headless:
    abq2025 cae noGUI=cae_local_edit_designs.py -- <HoodImpact_31.inp> <out_dir>
(from PowerShell quote the separator: '--')
"""
import builtins
import csv
import math
import os
import sys

import numpy as np
from abaqus import mdb

# the CAE kernel namespace can shadow builtins (sum() for one); pin the ones used
min, max, abs, round, sorted = (builtins.min, builtins.max, builtins.abs,
                                builtins.round, builtins.sorted)

# ----------------------------------------------------------------- settings
BASE_MODEL = "HoodImpact_31"
INP = r"D:\Papers\Hood_Impact\Code\Data\HoodImpact_60_IndustryLike\inp_files\HoodImpact_31.inp"
OUT_DIR = os.getcwd()
PART = "PART-1"
INNER_SET_KEY = "HOOD_INNER"     # matched case-insensitively against set names
OUTER_SET_KEY = "HOOD_OUTER"
HALF_WIDTH = 45.0                # mm across the beam (flanges reach ~60)
G0 = 3.0                         # mm, flange gap below the skin
MAX_GAP = 45.0                   # refuse to move anything deeper (perimeter frame)
BIN_MM = 0.01                    # offsets are applied in 0.01 mm bins (fast)
MIRROR_Y = 155.3432              # hood centreline (impact grid symmetry line)
DOE_CSV = r"D:\Papers\Hood_Impact\Code\Data\HoodImpact_LocalEdits\local_edit_doe.csv"
ONLY = []                        # e.g. ["N8", "N17"]; empty = every design in DOE_CSV

# One row per notch: design, site, x0, y0 (centre on the channel-floor axis,
# upper half), axis_deg, g1_mm (floor gap there), dz_mm, half_length_mm,
# mirror (1 = add the twin notch on the lower half; the horizontal beam lies
# on the centreline itself).
DESIGNS = []                     # [(name, description, [(site_name, site, dz, half_length)])]
with open(DOE_CSV) as _f:
    for _r in csv.DictReader(_f):
        if ONLY and _r["design"] not in ONLY:
            continue
        if not DESIGNS or DESIGNS[-1][0] != _r["design"]:
            DESIGNS.append((_r["design"], "", []))
        _site = dict(x0=float(_r["x0"]), y0=float(_r["y0"]), axis=float(_r["axis_deg"]),
                     g1=float(_r["g1_mm"]), mirror=_r["mirror"] == "1")
        DESIGNS[-1][2].append((_r["site"], _site, float(_r["dz_mm"]), float(_r["half_length_mm"])))
DESIGNS = [(n, "; ".join("%s %+g mm, +-%g mm" % (s, dz, hl) for s, _, dz, hl in notches), notches)
           for n, _, notches in DESIGNS]


def expand(site_name, s):
    """[(name, site)] for a site and, if it has one, its mirror twin."""
    out = [(site_name, s)]
    if s["mirror"]:
        out.append((site_name + "_mirror",
                    dict(s, y0=2.0 * MIRROR_Y - s["y0"], axis=-s["axis"])))
    return out
# ---------------------------------------------------------------------------

# Abaqus drops the "--" and appends the script arguments after its own flags,
# so locate them by the deck path.
_inp = [i for i, a in enumerate(sys.argv) if a.lower().endswith(".inp")]
if _inp:
    INP = sys.argv[_inp[-1]]
    if len(sys.argv) > _inp[-1] + 1:
        OUT_DIR = sys.argv[_inp[-1] + 1]
    BASE_MODEL = os.path.splitext(os.path.basename(INP))[0]
if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)

LOG = []


def say(msg):
    """print() is not visible in noGUI runs, so also keep a log file."""
    print(msg)
    LOG.append(msg)
    with open(os.path.join(OUT_DIR, "local_edit_designs.log"), "w") as f:
        f.write("\n".join(LOG) + "\n")


def find_set(part, key):
    hits = [k for k in part.sets.keys() if key in k.upper()]
    if len(hits) != 1:
        raise ValueError("expected one part set matching %r, found %r" % (key, hits))
    return part.sets[hits[0]]


def set_nodes(part, key):
    """label -> (x, y, z) of the nodes of an element set."""
    out = {}
    for el in find_set(part, key).elements:
        for nd in el.getNodes():
            out[nd.label] = nd.coordinates
    return out


def skin_z(outer, x, y):
    """Outer-skin z above (x, y): least-squares plane through the skin nodes
    within 12 mm in plan (skin mesh ~10 mm), nearest node if fewer than 3."""
    d = np.hypot(outer[:, 0] - x, outer[:, 1] - y)
    near = outer[d < 12.0]
    if len(near) < 3:
        return float(outer[d.argmin(), 2])
    a = np.c_[np.ones(len(near)), near[:, 0] - x, near[:, 1] - y]
    return float(np.linalg.lstsq(a, near[:, 2], rcond=None)[0][0])


def notch_offsets(labels, xyz, outer, site, dz_max, half_length):
    """{label: dz} of one notch, from the base coordinates."""
    ux, uy = math.cos(math.radians(site["axis"])), math.sin(math.radians(site["axis"]))
    rx, ry = xyz[:, 0] - site["x0"], xyz[:, 1] - site["y0"]
    along, across = rx * ux + ry * uy, -rx * uy + ry * ux
    out, deepest = {}, 0.0
    for k in np.where((np.abs(along) < half_length) & (np.abs(across) <= HALF_WIDTH))[0]:
        x, y, z = xyz[k]
        gap = skin_z(outer, x, y) - z
        w = 0.5 * (1.0 + math.cos(math.pi * abs(along[k]) / half_length))
        w *= min(max((gap - G0) / (site["g1"] - G0), 0.0), 1.0)
        if w > 0.0:
            out[labels[k]] = dz_max * w
            deepest = max(deepest, gap)
    if deepest > MAX_GAP:
        raise ValueError("notch at (%.1f, %.1f) reaches a node %.1f mm below the skin "
                         "(frame?) -- move the site" % (site["x0"], site["y0"], deepest))
    return out


if BASE_MODEL in mdb.models.keys():
    base = mdb.models[BASE_MODEL]
else:
    say("importing %s ..." % INP)
    base = mdb.ModelFromInputFile(name=BASE_MODEL, inputFileName=INP)
bpart = base.parts[PART]
inner = set_nodes(bpart, INNER_SET_KEY)
labels = np.array(sorted(inner))
xyz = np.array([inner[i] for i in labels], float)
outer = np.array(list(set_nodes(bpart, OUTER_SET_KEY).values()), float)
say("base %s: %d nodes, %d inner-panel nodes, %d outer-skin nodes"
    % (BASE_MODEL, len(bpart.nodes), len(labels), len(outer)))

rows = []
for name, desc, notches in DESIGNS:
    dz, per_site = {}, []
    for site_name, site0, dz_max, half_length in notches:
        for sname, site in expand(site_name, site0):
            off = notch_offsets(labels, xyz, outer, site, dz_max, half_length)
            if not off:
                raise ValueError("%s: notch %s moved no node" % (name, sname))
            n_real = len([v for v in off.values() if abs(v) >= 0.5 * BIN_MM])
            per_site.append((sname, site, dz_max, half_length, n_real))
            for lab, v in off.items():
                dz[lab] = dz.get(lab, 0.0) + v
    bins = {}
    for lab, v in dz.items():
        key = int(round(v / BIN_MM))
        if key != 0:
            bins.setdefault(key, []).append(int(lab))
    moved = sorted(l for labs in bins.values() for l in labs)

    if name in mdb.models.keys():
        del mdb.models[name]
    model = mdb.Model(name=name, objectToCopy=base)
    part = model.parts[PART]
    for key, labs in sorted(bins.items()):
        part.editNode(nodes=part.nodes.sequenceFromLabels(sorted(labs)), offset3=key * BIN_MM)
    part.Set(name="EDIT_NODES", nodes=part.nodes.sequenceFromLabels(moved))
    elems = set()
    for nd in part.nodes.sequenceFromLabels(moved):
        for el in nd.getElements():
            elems.add(el.label)
    part.Set(name="EDIT_ELEMENTS", elements=part.elements.sequenceFromLabels(sorted(elems)))
    model.rootAssembly.regenerate()

    csv_path = os.path.join(OUT_DIR, "%s_edited_nodes.csv" % name)
    with open(csv_path, "w") as f:
        f.write("label,x,y,z\n")
        for nd in part.nodes:
            x, y, z = nd.coordinates
            f.write("%d,%r,%r,%r\n" % (nd.label, float(x), float(y), float(z)))
    vals = [dz[l] for l in dz]
    say("%s  %-60s %4d nodes moved (%s), dz [%.2f, %.2f] mm -> %s"
        % (name, desc, len(moved), " + ".join(str(p[4]) for p in per_site),
           min(vals), max(vals), csv_path))
    for sname, s, dz_max, half_length, n_site in per_site:
        rows.append((name, desc, sname, s["x0"], s["y0"], s["axis"], dz_max,
                     half_length, HALF_WIDTH, G0, s["g1"], n_site, len(moved)))

with open(os.path.join(OUT_DIR, "local_edit_designs.csv"), "w") as f:
    f.write("design,description,site,x0,y0,axis_deg,dz_mm,half_length_mm,"
            "half_width_mm,g0_mm,g1_mm,n_nodes_site,n_nodes_moved\n")
    for r in rows:
        f.write("%s,\"%s\",%s,%.2f,%.2f,%.2f,%.1f,%.1f,%.1f,%.1f,%.1f,%d,%d\n" % r)
say("done: models %s; patch each CSV into %s with apply_node_edits.py"
    % (", ".join(d[0] for d in DESIGNS), INP))
