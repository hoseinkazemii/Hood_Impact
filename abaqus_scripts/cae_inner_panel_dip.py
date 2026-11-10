"""
cae_inner_panel_dip.py  --  runs INSIDE Abaqus/CAE (File > Run Script...)
=========================================================================
Test design change for the HoodImpact decks: lower (or raise) the hood INNER
panel locally around a plan point, with a smooth cosine falloff so no element
gets a sharp kink. The outer skin, the headform, hinges and fasteners are not
touched. Then export the part's node coordinates so the edit can be patched
back into the ORIGINAL flat deck with apply_node_edits.py.

Why export instead of Job > Write Input: CAE's .inp reader does not carry
over the 62 fastener *COUPLINGs of these decks (REF NODE = multi-node set +
INFLUENCE RADIUS), and it writes a part/assembly deck our tooling cannot
parse. The original deck with patched node lines keeps all physics intact.

    dz(r) = DZ * 0.5 * (1 + cos(pi * r / RADIUS))   for r < RADIUS, else 0
    r     = plan distance (x, y) from (X0, Y0); the impact direction is -Z

Beam notch (local change of a hat-section beam's depth):
    AXIS_DEG   = beam direction in plan; r becomes the distance ALONG the
                 beam axis through (X0, Y0), so the whole cross-section moves
                 together, and nodes farther than HALF_WIDTH across the axis
                 are left alone.
    GAP_RAMP   = (G0, G1): dz is also scaled by clip((gap - G0)/(G1 - G0), 0, 1),
                 gap = outer-skin z minus node z. The flanges (gap ~3 mm) stay
                 put, the channel floor (gap >= G1) moves by the full dz and
                 the walls in between are compressed/stretched uniformly.
    DZ > 0 with the ramp raises the channel floor = a shallower beam (notch);
    DZ < 0 deepens it.

Set DZ = 0 to skip the morph and only export (e.g. after editing nodes by
hand with Mesh > Edit).

Headless use (imports the deck when the model is not open):
    abq2025 cae noGUI=cae_inner_panel_dip.py -- <deck.inp> <out.csv> [X0 Y0 RADIUS DZ]
(from PowerShell quote the separator: '--')
"""
import math
import os
import sys

from abaqus import mdb

# ----------------------------------------------------------------- settings
# Design N1 = design 6 (HoodImpact_31.inp) + one notch in the upper-left
# diagonal beam at mid-span: channel floor raised 13 mm (depth ~28 -> ~15 mm)
# over +-60 mm along the beam. 13 mm keeps the shortest wall edge at 5.31 mm,
# above the deck's smallest edge (5.08 mm; 15 mm would give 4.81), so the
# stable time increment should not change (design 7 raised 20 mm, left 3 mm
# edges and took 20% more increments).
MODEL = "HoodImpact_31"
PART = "PART-1"
INNER_SET_KEY = "HOOD_INNER"     # matched case-insensitively against set names
OUTER_SET_KEY = "HOOD_OUTER"     # outer skin, used for the gap ramp
X0, Y0 = 30.0, 423.31            # plan centre of the edit (mm): beam axis, x = 30
RADIUS = 60.0                    # mm, cosine half-length (along the axis if AXIS_DEG)
DZ = 13.0                        # mm; < 0 moves the inner panel away from the skin
AXIS_DEG = 19.66                 # beam direction in plan (deg from +X), None = circle
HALF_WIDTH = 45.0                # mm across the axis (beam flanges reach ~60)
GAP_RAMP = (3.0, 31.0)           # mm; None moves every node in the region rigidly
OUT_CSV = os.path.join(os.getcwd(), "edited_nodes.csv")
INP = None                       # only used if MODEL is not open
BIN_MM = 0.01                    # offsets are applied in 0.01 mm bins (fast)
# ---------------------------------------------------------------------------

# Abaqus drops the "--" and appends the script arguments after its own flags,
# so locate them by the deck path.
_inp = [i for i, a in enumerate(sys.argv) if a.lower().endswith(".inp")]
argv = sys.argv[_inp[-1]:] if _inp else []
if argv:
    INP = argv[0]
    OUT_CSV = argv[1]
    if len(argv) >= 6:
        X0, Y0, RADIUS, DZ = (float(v) for v in argv[2:6])
    MODEL = os.path.splitext(os.path.basename(INP))[0]


LOG = []


def say(msg):
    """print() is not visible in noGUI runs, so also keep a log file."""
    print(msg)
    LOG.append(msg)
    with open(os.path.splitext(OUT_CSV)[0] + ".log", "w") as f:
        f.write("\n".join(LOG) + "\n")


def get_model():
    if MODEL in mdb.models.keys():
        return mdb.models[MODEL]
    if not INP:
        raise ValueError("model %r is not open and no deck path was given" % MODEL)
    say("importing %s ..." % INP)
    return mdb.ModelFromInputFile(name=MODEL, inputFileName=INP)


def find_set(part, key):
    hits = [k for k in part.sets.keys() if key in k.upper()]
    if len(hits) != 1:
        raise ValueError("expected one part set matching %r, found %r" % (key, hits))
    return part.sets[hits[0]]


def skin_height(part, reach):
    """Function (x, y) -> z of the outer skin: least-squares plane through the
    skin nodes within 12 mm in plan (skin mesh ~10 mm), near the edit only."""
    import numpy
    pts = {}
    for el in find_set(part, OUTER_SET_KEY).elements:
        for nd in el.getNodes():
            x, y, z = nd.coordinates
            if abs(x - X0) < reach and abs(y - Y0) < reach:
                pts[nd.label] = (x, y, z)
    pts = numpy.array(list(pts.values()), float)
    if not len(pts):
        raise ValueError("no outer-skin nodes near (%.1f, %.1f)" % (X0, Y0))

    def skin_z(x, y):
        d = numpy.hypot(pts[:, 0] - x, pts[:, 1] - y)
        near = pts[d < 12.0]
        if len(near) < 3:
            return float(pts[d.argmin(), 2])
        a = numpy.c_[numpy.ones(len(near)), near[:, 0] - x, near[:, 1] - y]
        return float(numpy.linalg.lstsq(a, near[:, 2], rcond=None)[0][0])
    return skin_z


def weight(x, y, z, skin_z):
    """Morph weight in [0, 1] of a node (0 = not moved)."""
    if AXIS_DEG is None:
        r = math.hypot(x - X0, y - Y0)
    else:
        ux, uy = math.cos(math.radians(AXIS_DEG)), math.sin(math.radians(AXIS_DEG))
        if abs(-(x - X0) * uy + (y - Y0) * ux) > HALF_WIDTH:
            return 0.0
        r = abs((x - X0) * ux + (y - Y0) * uy)
    if r >= RADIUS:
        return 0.0
    w = 0.5 * (1.0 + math.cos(math.pi * r / RADIUS))
    if GAP_RAMP:
        g0, g1 = GAP_RAMP
        w *= min(max((skin_z(x, y) - z - g0) / (g1 - g0), 0.0), 1.0)
    return w


model = get_model()
part = model.parts[PART]
say("model %s: %d nodes, %d elements, %d constraints in CAE"
      % (MODEL, len(part.nodes), len(part.elements), len(model.constraints)))
say("constraints: %s" % sorted(model.constraints.keys())[:20])
say("connector section assignments in the assembly: %d"
    % len(model.rootAssembly.sectionAssignments))

if DZ != 0.0:
    inner = find_set(part, INNER_SET_KEY)
    labels = set()
    for el in inner.elements:
        for nd in el.getNodes():
            labels.add(nd.label)
    skin_z = skin_height(part, RADIUS + HALF_WIDTH + 30.0) if GAP_RAMP else None
    bins = {}
    for lab in labels:
        x, y, z = part.nodes.getFromLabel(lab).coordinates
        w = weight(x, y, z, skin_z)
        if w <= 0.0:
            continue
        dz = DZ * w
        key = int(round(dz / BIN_MM))
        if key != 0:
            bins.setdefault(key, []).append(lab)
    n_moved = 0                  # the CAE kernel namespace shadows builtin sum()
    for v in bins.values():
        n_moved += len(v)
    say("morph: centre (%.1f, %.1f), R %.0f mm, DZ %.1f mm, axis %s, gap ramp %s"
        " -> %d inner nodes, %d offset bins"
        % (X0, Y0, RADIUS, DZ, AXIS_DEG, GAP_RAMP, n_moved, len(bins)))
    for key, labs in sorted(bins.items()):
        part.editNode(nodes=part.nodes.sequenceFromLabels(sorted(labs)),
                      offset3=key * BIN_MM)
    model.rootAssembly.regenerate()

with open(OUT_CSV, "w") as f:
    f.write("label,x,y,z\n")
    for nd in part.nodes:
        x, y, z = nd.coordinates
        f.write("%d,%r,%r,%r\n" % (nd.label, float(x), float(y), float(z)))
say("wrote %d node coordinates -> %s" % (len(part.nodes), OUT_CSV))
