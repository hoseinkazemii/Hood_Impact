# -*- coding: utf-8 -*-
"""
extract_acc_odb.py
==================
Extract the impactor (headform) reference-node ACCELERATION history from a
Hood-impact Abaqus/Explicit .odb and write it as raw CSV: Time,A1,A2,A3.

This is the *raw* solver output (every increment, FREQUENCY=1). The SAE-1000
filtering + resampling to 1000 points is done afterwards by postprocess_acc.py
(plain Python, needs scipy) so that this script only needs the Abaqus Python
interpreter.

Run it INSIDE Abaqus:
    abq2025 python extract_acc_odb.py -- <odb_path> <out_csv> [node_label]

Why we search by output name, not node id
-----------------------------------------
The acceleration that HIC is computed from is the acceleration of the rigid
headform reference node, which the decks expose through the history request
    *OUTPUT, HISTORY, FREQUENCY=1
    *NODE OUTPUT, NSET="Nodal Probe12"
    A
That node is the ONLY node in the model with an A (acceleration) history
request, but its *label is not constant* across designs (e.g. 44166 for
designs 1-30, 45368 for designs 31-60). So we locate the history region that
owns A1/A2/A3 instead of trusting a fixed node number.
"""
from __future__ import print_function

import sys
import io
import csv

from odbAccess import openOdb


def parse_args(argv):
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    if len(argv) < 2:
        raise SystemExit(
            "usage: abq2025 python extract_acc_odb.py -- <odb_path> <out_csv> [node_label]"
        )
    odb_path = argv[0]
    out_csv = argv[1]
    node_label = argv[2] if len(argv) > 2 else None
    return odb_path, out_csv, node_label


def find_acc_region(step, node_label=None):
    """Return (region_name, region) of the history region that owns A1/A2/A3."""
    for rname in step.historyRegions.keys():
        region = step.historyRegions[rname]
        outs = region.historyOutputs.keys()
        if ("A1" in outs) and ("A2" in outs) and ("A3" in outs):
            if (node_label is None) or (str(node_label) in rname):
                return rname, region
    return None, None


def main():
    odb_path, out_csv, node_label = parse_args(sys.argv)
    print("Opening %s" % odb_path)
    odb = openOdb(odb_path, readOnly=True)

    times, A1, A2, A3 = [], [], [], []
    region_name = None

    # History "total time" is continuous across steps; there is only one step
    # in these decks, but loop generically just in case.
    for sName in odb.steps.keys():
        step = odb.steps[sName]
        rname, region = find_acc_region(step, node_label)
        if region is None:
            continue
        region_name = rname
        d1 = region.historyOutputs["A1"].data
        d2 = region.historyOutputs["A2"].data
        d3 = region.historyOutputs["A3"].data
        for i in range(len(d1)):
            times.append(d1[i][0])
            A1.append(d1[i][1])
            A2.append(d2[i][1])
            A3.append(d3[i][1])

    odb.close()

    if not times:
        raise SystemExit(
            "ERROR: no history region with A1/A2/A3 found in %s.\n"
            "       Check that the job ran with the 'Nodal Probe12' A output." % odb_path
        )

    # newline='' keeps Windows from doubling line endings (Abaqus 2025 = Py3).
    with io.open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Time", "A1", "A2", "A3"])
        for i in range(len(times)):
            # repr() preserves full float precision.
            w.writerow([repr(times[i]), repr(A1[i]), repr(A2[i]), repr(A3[i])])

    print("Extracted %d raw points from region '%s'" % (len(times), region_name))
    print("Wrote %s" % out_csv)


if __name__ == "__main__":
    main()
