"""ODB -> <name>_results.npz: every history output and nodal field histories (run inside Abaqus).

    abq2025 python extract_odb.py -- <job.odb> <out.npz> [solver status]

Fields (every saved frame of the last step):
  nodal variables (U, NT11)        -> field_<VAR>, shape (frames, nodes, components)
  element variables (S, LE, PEEQ)  -> extrapolated to the nodes (ELEMENT_NODAL) and averaged
                                      over the elements sharing each node, same shape
History: every output of every history region, grouped by variable:
  hist_<VAR> (times, regions), hist_<VAR>_time, hist_<VAR>_regions
Rows follow node_labels (the deck's node labels; the generators number them 1..N).
"""
import sys

import numpy as np
from abaqusConstants import ELEMENT_NODAL, NODAL
from odbAccess import openOdb

NODAL_VARIABLES = ("U", "NT11")
ELEMENT_VARIABLES = ("S", "LE", "PEEQ")


def nodal_average(field, lookup, n_nodes):
    subset = field.getSubset(position=ELEMENT_NODAL)
    total, count = None, np.zeros(n_nodes)
    for block in subset.bulkDataBlocks:
        rows = lookup[np.asarray(block.nodeLabels, dtype=np.int64)]
        data = np.asarray(block.data, dtype=np.float64).reshape(len(rows), -1)
        if total is None:
            total = np.zeros((n_nodes, data.shape[1]))
        np.add.at(total, rows, data)
        np.add.at(count, rows, 1)
    return total / np.maximum(count, 1)[:, None]


def nodal_values(field, lookup, n_nodes):
    out = None
    for block in field.bulkDataBlocks:
        rows = lookup[np.asarray(block.nodeLabels, dtype=np.int64)]
        data = np.asarray(block.data, dtype=np.float64).reshape(len(rows), -1)
        if out is None:
            out = np.zeros((n_nodes, data.shape[1]))
        out[rows] = data
    return out


def main(argv):
    argv = argv[argv.index("--") + 1:] if "--" in argv else argv[1:]
    odb_path, out_path = argv[0], argv[1]
    status = argv[2] if len(argv) > 2 else "unknown"
    odb = openOdb(odb_path, readOnly=True)
    instance = odb.rootAssembly.instances[list(odb.rootAssembly.instances.keys())[0]]
    labels = np.array([node.label for node in instance.nodes], dtype=np.int64)
    lookup = np.full(labels.max() + 1, -1, dtype=np.int64)
    lookup[labels] = np.arange(len(labels))
    step = odb.steps[list(odb.steps.keys())[-1]]
    out = {"node_labels": labels, "solver_status": np.array(status)}

    frames = step.frames
    out["field_time"] = np.array([frame.frameValue for frame in frames])
    last = frames[-1].fieldOutputs
    for name in NODAL_VARIABLES + ELEMENT_VARIABLES:
        if name not in last.keys():
            continue
        stack = []
        for frame in frames:
            field = frame.fieldOutputs[name]
            if name in NODAL_VARIABLES:
                stack.append(nodal_values(field, lookup, len(labels)))
            else:
                stack.append(nodal_average(field, lookup, len(labels)))
        out["field_" + name] = np.array(stack, dtype=np.float32)
        out["field_%s_components" % name] = np.array(list(last[name].componentLabels) or [name])

    grouped = {}
    for region_name, region in step.historyRegions.items():
        for variable, history in region.historyOutputs.items():
            grouped.setdefault(variable, []).append((region_name, np.array(history.data, dtype=np.float64)))
    for variable, items in grouped.items():
        length = min(len(data) for _, data in items)
        out["hist_%s_time" % variable] = items[0][1][:length, 0]
        out["hist_" + variable] = np.column_stack([data[:length, 1] for _, data in items])
        out["hist_%s_regions" % variable] = np.array([name for name, _ in items])
    odb.close()
    np.savez_compressed(out_path, **out)
    print("wrote %s: %d frames, %d nodes, history %s" % (out_path, len(frames), len(labels), sorted(grouped)))


if __name__ == "__main__":
    main(sys.argv)
