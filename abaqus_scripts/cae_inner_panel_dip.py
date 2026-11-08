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
MODEL = "HoodImpact_1"
PART = "PART-1"
INNER_SET_KEY = "HOOD_INNER"     # matched case-insensitively against set names
X0, Y0 = -100.275, 155.3432      # plan centre of the edit (mm); run 1 headform
RADIUS = 100.0                   # mm, influence radius in plan
DZ = -15.0                       # mm; < 0 moves the inner panel away from the skin
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
    bins = {}
    for lab in labels:
        x, y, _ = part.nodes.getFromLabel(lab).coordinates
        r = math.hypot(x - X0, y - Y0)
        if r >= RADIUS:
            continue
        dz = DZ * 0.5 * (1.0 + math.cos(math.pi * r / RADIUS))
        key = int(round(dz / BIN_MM))
        if key != 0:
            bins.setdefault(key, []).append(lab)
    n_moved = 0                  # the CAE kernel namespace shadows builtin sum()
    for v in bins.values():
        n_moved += len(v)
    say("morph: centre (%.1f, %.1f), R %.0f mm, DZ %.1f mm -> %d inner nodes, %d offset bins"
          % (X0, Y0, RADIUS, DZ, n_moved, len(bins)))
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
