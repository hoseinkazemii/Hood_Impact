"""Write the pilot decks for the local-edit benchmarks and draw each family's meshes.

    python generate.py B1 B2 B3 B4 [--out D:\\Papers\\Hood_Impact\\LocalEditBenchmarks\\pilot]

Every design gets its own folder under <out>/<benchmark>/: <name>.inp (solve
it there), <name>.json (edit record + hand-model estimate) and <name>_mesh.npz
(mesh + change nodes). <out>/<benchmark>/meshes.png shows the base and the
edits with the change nodes in red; designs.json collects the records.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b1_plate  # noqa: E402
import b2_elbow  # noqa: E402
import b3_heatsink  # noqa: E402
import b4_cantilever  # noqa: E402
from common import DATA_ROOT  # noqa: E402

MODULES = {"B1": b1_plate, "B2": b2_elbow, "B3": b3_heatsink, "B4": b4_cantilever}
WINDOWS = {"B1": ((55, 145), (0, 60), 2), "B3": ((0, 100), (0, 50), 2), "B4": ((0, 130), (0, 20), 1)}


def proxy_text(record):
    change = record["proxy"].get("change_vs_base_percent")
    if change is None or record["design"] == "base":
        return ""
    if isinstance(change, list):
        return "hand model: " + " / ".join(f"{c:+.0f}%" for c in change)
    return f"hand model: {change:+.0f}%"


def title(record):
    return (f"{record['design']}: {record['edit']}\n{record['change_nodes']} change nodes "
            f"({record['change_nodes_over_0.1mm']} moved > 0.1 mm)   {proxy_text(record)}")


def draw_2d(key, folder, records):
    (x0, x1), (y0, y1), cols = WINDOWS[key]
    rows = int(np.ceil(len(records) / cols))
    width = 15 if cols == 2 else 14
    fig, axes = plt.subplots(rows, cols, figsize=(width, rows * (width / cols) * (y1 - y0) / (x1 - x0) + rows * 0.6),
                             squeeze=False)
    for ax, record in zip(axes.flat, records):
        mesh = np.load(folder / record["design"] / f"{record['design']}_mesh.npz")
        xyz, quads = mesh["xyz"], mesh["elements"] - 1
        ax.add_collection(PolyCollection(xyz[quads], facecolors="none", edgecolors="0.55", linewidths=0.15))
        moved = mesh["change_node_ids"] - 1
        ax.scatter(xyz[moved, 0], xyz[moved, 1], s=1.5, c="#c0392b", lw=0)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_title(title(record), fontsize=8.5, loc="left")
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(records):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(folder / "meshes.png", dpi=170)
    plt.close(fig)


def draw_elbow(folder, records):
    module = b2_elbow
    s, theta = module.stations(), np.arange(module.N_CIRC) * 360.0 / module.N_CIRC
    ss, tt = np.meshgrid(s, theta, indexing="ij")
    fig, axes = plt.subplots(3, 2, figsize=(15, 13), squeeze=False, gridspec_kw={"hspace": 0.6, "wspace": 0.25})
    for ax, record in zip(axes.flat, records):
        loss = np.zeros_like(ss)
        for defect in record["params"]["defects"]:
            loss = np.maximum(loss, module.defect_depth(ss, tt, defect))
        image = ax.pcolormesh(ss, tt, loss / module.T * 100, cmap="magma_r", vmin=0, vmax=65, shading="nearest")
        for edge in (module.TANGENT, module.TANGENT + module.ARC):
            ax.axvline(edge, color="0.4", lw=0.8, ls="--")
        ax.text(module.TANGENT + module.ARC / 2, 352, "bend", ha="center", va="top", fontsize=7.5, color="0.3")
        ax.set_yticks([0, 90, 180, 270, 360], ["intrados 0", "crown 90", "extrados 180", "crown 270", "360"],
                      fontsize=7)
        ax.set_xlabel("centreline position from anchored end 1 (mm)", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.set_title(title(record), fontsize=8, loc="left")
    fig.colorbar(image, ax=axes, shrink=0.6, label="metal loss, % of wall")
    fig.savefig(folder / "meshes.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("benchmarks", nargs="+", choices=sorted(MODULES))
    parser.add_argument("--out", type=Path, default=DATA_ROOT / "pilot")
    args = parser.parse_args()
    for key in args.benchmarks:
        module = MODULES[key]
        records = module.generate(args.out)
        folder = args.out / module.BENCHMARK
        (folder / "designs.json").write_text(json.dumps(records, indent=2))
        if key == "B2":
            draw_elbow(folder, records)
        else:
            draw_2d(key, folder, records)
        print(f"{module.BENCHMARK}: {len(records)} designs -> {folder}")
        for r in records:
            print(f"  {r['design']:20s} nodes {r['nodes']:6d}  change nodes {r['change_nodes']:5d}  "
                  f"max move {r['max_node_move_mm']:6.2f} mm  min Jacobian {r['min_scaled_jacobian']:.2f}  "
                  f"{proxy_text(r)}")


if __name__ == "__main__":
    main()
