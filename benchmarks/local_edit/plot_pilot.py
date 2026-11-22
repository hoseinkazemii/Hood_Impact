"""Pilot results: main curves, field maps and the effect-size table for each benchmark.

    python plot_pilot.py B1 B2 B3 B4 [--root D:\\Papers\\Hood_Impact\\LocalEditBenchmarks\\pilot]

Writes into each benchmark folder:
  curves.png    the main output of the base and every edit
  fields.png    the field each design ends with, and its difference from the base
                (same node IDs, so the difference is node by node)
  effects.csv   per design: the hand-model change, the FE change of the headline
                number, and the mean |curve difference| relative to the base curve
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b2_elbow  # noqa: E402
import b3_heatsink  # noqa: E402
from run_pilot import DEFAULT_ROOT, FOLDERS  # noqa: E402

GRID = 101  # points of the common curve grid
CURVE_LABELS = {
    "B1": ("grip elongation (mm)", "grip force (kN)", "peak force"),
    "B2": ("water pumped in (10^6 mm^3)", "pressure (MPa)", "burst pressure"),
    "B3": ("time (s)", "chip temperature (C)", "chip rise at 600 s"),
    "B4": ("tip deflection (mm)", "tip force (kN)", "peak force"),
}


def load(folder):
    name = folder.name
    results = folder / f"{name}_results.npz"
    if not results.exists():
        return None
    data = dict(np.load(results, allow_pickle=False))
    data.update({f"mesh_{k}": v for k, v in np.load(folder / f"{name}_mesh.npz", allow_pickle=False).items()})
    data["record"] = json.loads((folder / f"{name}.json").read_text())
    return data


def curves(key, d):
    """[(label, x, y)] main-output curves of one design."""
    if key == "B1":
        return [("force", d["hist_U1"].mean(1), d["hist_RF1"].sum(1) / 1e3)]
    if key == "B4":
        return [("force", -d["hist_U2"].mean(1), -d["hist_RF2"].sum(1) / 1e3)]
    if key == "B2":
        return [("pressure", d["hist_PCAV_time"] * b2_elbow.INJECTED_VOLUME / 1e6, d["hist_PCAV"][:, 0])]
    xyz = d["mesh_xyz"]
    out = []
    for c, (xc, _) in enumerate(b3_heatsink.CHIPS, 1):
        rows = np.where((xyz[:, 1] < 1e-9) & (np.abs(xyz[:, 0] - xc) <= b3_heatsink.CHIP_WIDTH / 2 + 1e-6))[0]
        out.append((f"chip {c}", d["field_time"], d["field_NT11"][:, rows, 0].mean(1)))
    return out


def headline(key, curve):
    _, x, y = curve
    if key == "B3":
        return float(np.interp(b3_heatsink.END_TIME, x, y) - b3_heatsink.AMBIENT)
    return float(y.max())


def resample(curve, x_max):
    _, x, y = curve
    grid = np.linspace(0, x_max, GRID)
    inside = grid <= x.max() + 1e-9
    return grid[inside], np.interp(grid[inside], x, y)


def effects(key, designs):
    base = designs["base"]
    base_curves = curves(key, base)
    rows = []
    for name, d in designs.items():
        record = d["record"]
        proxy = record["proxy"].get("change_vs_base_percent")
        for b, c in zip(base_curves, curves(key, d)):
            x_max = min(b[1].max(), c[1].max())
            _, yb = resample(b, x_max)
            _, yc = resample(c, x_max)
            rise = (yb - (b3_heatsink.AMBIENT if key == "B3" else 0.0))
            h_base, h = headline(key, b), headline(key, c)
            rows.append({
                "design": name, "output": c[0], "edit": record["edit"],
                "hand_model_change_percent": proxy if not isinstance(proxy, list) else proxy[int(c[0][-1]) - 1],
                "fe_headline": round(h, 3), "fe_change_percent": round(100 * (h / h_base - 1), 1),
                "curve_mean_abs_diff_percent": round(100 * np.abs(yc - yb).mean() / np.abs(rise).mean(), 1),
                "change_nodes": record["change_nodes"], "solver_status": str(d["solver_status"]),
            })
    return pd.DataFrame(rows)


def draw_curves(key, designs, out):
    xlabel, ylabel, _ = CURVE_LABELS[key]
    n_out = len(curves(key, designs["base"]))
    fig, axes = plt.subplots(1, n_out, figsize=(6.2 * n_out, 4.6), squeeze=False)
    colors = plt.cm.tab10(np.arange(10))
    for i, (name, d) in enumerate(designs.items()):
        for ax, curve in zip(axes.flat, curves(key, d)):
            ax.plot(curve[1], curve[2], color="k" if name == "base" else colors[i], lw=2.2 if name == "base" else 1.4,
                    label=name)
            ax.set_title(curve[0], fontsize=10)
    for ax in axes.flat:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def field_of(key, d, frame):
    if key == "B3":
        return d["field_NT11"][frame, :, 0], "temperature (C)"
    return d["field_PEEQ"][frame, :, 0], "equivalent plastic strain"


def draw_fields(key, designs, out):
    base = designs["base"]
    names = list(designs)
    if key == "B2":
        return draw_elbow_fields(designs, out)
    window = {"B1": ((55, 145), (0, 60)), "B3": ((0, 100), (0, 50)), "B4": ((0, 130), (0, 20))}[key]
    aspect = (window[1][1] - window[1][0]) / (window[0][1] - window[0][0])
    fig, axes = plt.subplots(len(names), 2, figsize=(14, len(names) * 7 * aspect + 1.2 * len(names)), squeeze=False)
    frame = -1
    base_field, label = field_of(key, base, frame)
    vmax = max(np.percentile(field_of(key, d, frame)[0], 99.5) for d in designs.values())
    vmin = b3_heatsink.AMBIENT if key == "B3" else 0.0
    diffs = {n: field_of(key, d, frame)[0] - base_field for n, d in designs.items()}
    dmax = max(np.percentile(np.abs(v), 99.5) for v in diffs.values()) or 1.0
    for row, name in enumerate(names):
        d = designs[name]
        xyz, quads = d["mesh_xyz"], d["mesh_elements"] - 1
        tri = mtri.Triangulation(xyz[:, 0], xyz[:, 1], np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]]))
        field, _ = field_of(key, d, frame)
        a = axes[row, 0].tripcolor(tri, field, shading="gouraud", cmap="inferno", vmin=vmin, vmax=vmax)
        b = axes[row, 1].tripcolor(tri, diffs[name], shading="gouraud", cmap="RdBu_r", vmin=-dmax, vmax=dmax)
        axes[row, 0].set_title(f"{name}: {label} at the end", fontsize=9, loc="left")
        axes[row, 1].set_title(f"{name} minus base (same node IDs)", fontsize=9, loc="left")
        for ax in axes[row]:
            ax.set_xlim(*window[0])
            ax.set_ylim(*window[1])
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
    fig.colorbar(a, ax=axes[:, 0], shrink=0.5, label=label)
    fig.colorbar(b, ax=axes[:, 1], shrink=0.5, label="difference")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def draw_elbow_fields(designs, out):
    m = b2_elbow
    s, theta = m.stations(), np.arange(m.N_CIRC) * 360.0 / m.N_CIRC
    outer = (np.arange(len(s))[:, None] * m.N_CIRC + np.arange(m.N_CIRC)[None, :]) * (m.N_WALL + 1) + m.N_WALL
    base = designs["base"]
    base_peak = int(np.argmax(np.interp(base["field_time"], base["hist_PCAV_time"], base["hist_PCAV"][:, 0])))
    fig, axes = plt.subplots(len(designs), 2, figsize=(15, 3.1 * len(designs)), squeeze=False,
                             gridspec_kw={"hspace": 0.55})
    vmax = 0.0
    maps = {}
    for name, d in designs.items():
        peak = int(np.argmax(np.interp(d["field_time"], d["hist_PCAV_time"], d["hist_PCAV"][:, 0])))
        frame = min(base_peak, len(d["field_time"]) - 1)
        maps[name] = (d["field_PEEQ"][peak, :, 0][outer], d["field_PEEQ"][frame, :, 0][outer]
                      - base["field_PEEQ"][frame, :, 0][outer], d["field_time"][peak])
        vmax = max(vmax, np.percentile(maps[name][0], 99.5))
    dmax = max(np.percentile(np.abs(v[1]), 99.5) for v in maps.values()) or 1.0
    for row, (name, (peak_map, diff_map, t)) in enumerate(maps.items()):
        a = axes[row, 0].pcolormesh(s, theta, peak_map.T, cmap="inferno", vmin=0, vmax=vmax, shading="nearest")
        b = axes[row, 1].pcolormesh(s, theta, diff_map.T, cmap="RdBu_r", vmin=-dmax, vmax=dmax, shading="nearest")
        axes[row, 0].set_title(f"{name}: PEEQ on the outer wall at its pressure peak "
                               f"({t * m.INJECTED_VOLUME / 1e6:.2f}e6 mm^3)", fontsize=8.5, loc="left")
        axes[row, 1].set_title(f"{name} minus base at the base's peak frame", fontsize=8.5, loc="left")
        for ax in axes[row]:
            for edge in (m.TANGENT, m.TANGENT + m.ARC):
                ax.axvline(edge, color="0.6", lw=0.7, ls="--")
            ax.set_yticks([0, 90, 180, 270, 360], ["intrados", "90", "extrados", "270", "360"], fontsize=7)
            ax.tick_params(labelsize=7)
    fig.colorbar(a, ax=axes[:, 0], shrink=0.5, label="equivalent plastic strain")
    fig.colorbar(b, ax=axes[:, 1], shrink=0.5, label="difference")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("benchmarks", nargs="+", choices=sorted(FOLDERS))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    for key in args.benchmarks:
        folder = args.root / FOLDERS[key]
        designs = {}
        for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
            d = load(sub)
            if d is not None:
                designs[sub.name] = d
        if "base" not in designs:
            print(f"{FOLDERS[key]}: no base results yet")
            continue
        designs = {"base": designs.pop("base"), **designs}
        table = effects(key, designs)
        table.to_csv(folder / "effects.csv", index=False)
        draw_curves(key, designs, folder / "curves.png")
        draw_fields(key, designs, folder / "fields.png")
        with pd.option_context("display.width", 200, "display.max_colwidth", 40):
            print(f"\n{FOLDERS[key]} ({CURVE_LABELS[key][2]})")
            print(table.drop(columns=["edit"]).to_string(index=False))


if __name__ == "__main__":
    main()
