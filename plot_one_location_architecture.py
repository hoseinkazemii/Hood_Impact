"""Slide figure of the geometry-only, single-location network.

Draws the MeshImpactHistoryNet configuration that run_one_impact_location.sh
trains (``--impact-conditioning none``, inner frame only) as one left-to-right
diagram: the geometry path on top, the time path below, joined in the decoder.
Parameter counts are rebuilt from the run's config.json, the input panel is
the run's design-0 inner frame and the output panel its predicted history.

    python plot_one_location_architecture.py [run_dir]

Writes figures/one_impact_location_architecture.png/svg/pdf. The SVG keeps
text as text, so PowerPoint can edit it after "Convert to Shape".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np
import pandas as pd

from abaqus_scripts.inp_geom import Deck
from mesh_impact_history import MeshImpactHistoryNet, saved_model_kwargs

DEFAULT_RUN = Path("runs/one_impact_location/20261024_182919_loc137_test7")
OUTPUT_STEM = Path("figures/one_impact_location_architecture")
SAMPLES_PER_DESIGN, DESIGN = 142, 0
REFERENCE_PARAMETERS = 1_592_457  # run 20261004_050747, rebuilt from its config

W, H = 133, 62  # canvas units: 10 per inch
INK, INK_2, ARROW, REMOVED = "#1d1d1b", "#5f5e5a", "#4a4a48", "#b3261e"
GEOMETRY = ("#e8f0fb", "#2a78d6")  # fill, edge
TIME = ("#fcf0e0", "#c47a1c")
DECODER = ("#efebf9", "#6b52b5")
HEAD = ("#f1f0ec", "#6d6c67")
PREDICTION = "#2a78d6"

plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "svg.fonttype": "none",
                     "pdf.fonttype": 42})


def parameter_groups(config):
    """Trainable parameters per drawn block, from the saved architecture."""
    model = MeshImpactHistoryNet(**saved_model_kwargs(config["architecture"]["kwargs"]))
    blocks = {
        "embedding": ("node_embedding",),
        "neighborhood": ("neighborhood_blocks",),
        "pooling": ("latent_queries", "query_norm", "mesh_query", "mesh_key_value", "mesh_output"),
        "mixing": ("latent_blocks", "memory_norm"),
        "time": ("time_embedding",),
        "decoder": ("decoder_blocks",),
        "head": ("acceleration_head",),
    }
    owner = {module: block for block, modules in blocks.items() for module in modules}
    counts = dict.fromkeys(blocks, 0)
    for name, tensor in model.named_parameters():
        counts[owner[name.split(".")[0]]] += tensor.numel()
    return counts, sum(counts.values())


def short(count):
    return f"{count / 1000:.1f}k" if count < 10_000 else f"{count / 1000:.0f}k"


def block(ax, x, y, w, h, colors, title, lines, tag):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor=colors[0], edgecolor=colors[1], linewidth=1.4))
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=INK)
    for index, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 5.9 - 2.4 * index, line, ha="center", va="center",
                fontsize=9, color=INK)
    ax.text(x + w / 2, y + 1.9, tag, ha="center", va="center", fontsize=8.5, color=INK_2)


def arrow(ax, *points, dashed=False):
    """Polyline whose last segment ends in an arrowhead."""
    style = (0, (3, 2)) if dashed else "solid"
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color=ARROW, lw=1.4, linestyle=style, solid_capstyle="butt")
    ax.annotate("", xy=points[-1], xytext=points[-2],
                arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.4, linestyle=style,
                                shrinkA=0, shrinkB=0, mutation_scale=13))


def inset(fig, x, y, w, h):
    return fig.add_axes((x / W, y / H, w / W, h / H))


def draw_input(fig, ax, deck_path, nodes):
    deck = Deck(deck_path)
    frame = np.array([deck.nodes[n] for n in deck.elset_nodes("Hood_Inner-1-2")])
    head = np.array([deck.nodes[n] for n in deck.impactor_node_ids()])[:, :2]
    centre = head.mean(axis=0)
    radius = np.linalg.norm(head - centre, axis=1).max()
    mesh_ax = inset(fig, 1, 32, 19, 18)
    mesh_ax.scatter(frame[:, 0], frame[:, 1], s=0.15, color="#4f6b8a", linewidths=0,
                    rasterized=True)
    mesh_ax.add_patch(Circle(centre, radius, fill=False, edgecolor=REMOVED, lw=1.2,
                             linestyle=(0, (2.5, 1.5))))
    # Label below the frame, clear of the mesh lines.
    low, high = frame[:, 1].min(), frame[:, 1].max()
    mesh_ax.annotate("head position (not read)", centre - (0, radius),
                     xytext=(frame[:, 0].mean(), low - 0.13 * (high - low)), fontsize=7.5,
                     color=INK, ha="center", va="center",
                     arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8, shrinkB=4))
    mesh_ax.set_ylim(low - 0.2 * (high - low), high + 0.02 * (high - low))
    mesh_ax.set_aspect("equal")
    mesh_ax.axis("off")
    ax.text(10.5, 55.2, "Input: inner frame only", ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=INK)
    ax.text(10.5, 52.9, f"design {DESIGN}, top view, N = {nodes:,} nodes", ha="center",
            va="center", fontsize=8.5, color=INK_2)


def draw_output(fig, run_dir, run_number):
    rows = pd.read_csv(run_dir / "train_acceleration_histories.csv")
    rows = rows[rows.run_number == run_number].sort_values("time")
    curve_ax = inset(fig, 106, 11.5, 25, 12)
    curve_ax.plot(rows.time * 1e3, rows.acceleration_pred_g, color=PREDICTION, lw=1.6)
    curve_ax.set_xlim(0, 25)
    curve_ax.set_ylim(0, None)
    curve_ax.set_xlabel("time (ms)", fontsize=8, color=INK_2, labelpad=1)
    curve_ax.tick_params(labelsize=7.5, colors=INK_2, length=2.5)
    curve_ax.grid(alpha=0.25)
    for side in ("top", "right"):
        curve_ax.spines[side].set_visible(False)
    curve_ax.set_title(f"Output: acceleration (g)\n{len(rows)} points, design {DESIGN} prediction",
                       fontsize=9, color=INK, pad=4)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, nargs="?", default=DEFAULT_RUN)
    args = parser.parse_args(argv)
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    kwargs = config["architecture"]["kwargs"]
    if kwargs["impact_conditioning"] != "none" or config["preprocessing"].get("mesh_parts") != "inner":
        raise ValueError("This diagram describes a geometry-only, inner-frame run")
    counts, total = parameter_groups(config)
    width, tokens, k = kwargs["width"], kwargs["num_latents"], kwargs["neighborhood_k"]
    location = config["data"]["locations"][0]
    run_number = SAMPLES_PER_DESIGN * DESIGN + location
    times = np.load(args.run_dir / "prediction_times.npy")
    steps = len(times)
    deck_path = Path(config["data"]["inp_dir"]) / f"HoodImpact_{run_number}.inp"
    nodes = len(Deck(deck_path).elset_nodes("Hood_Inner-1-2"))

    fig = plt.figure(figsize=(W / 10, H / 10))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # Geometry path.
    draw_input(fig, ax, deck_path, nodes)
    block(ax, 23, 32, 17, 18, GEOMETRY, "Node embedding",
          [f"MLP 3 → {width} → {width}", "+ LayerNorm", "input: XYZ only"],
          f"N × {width} · {short(counts['embedding'])} params")
    block(ax, 44, 32, 23, 18, GEOMETRY, f"kNN attention ×{kwargs['neighborhood_layers']}",
          ["each frame node attends to", f"its {k} nearest frame nodes",
           f"{kwargs['num_heads']} heads; learned bias and",
           f"messages from offsets ({kwargs['neighborhood_scale_mm']:g} mm)"],
          f"N × {width} · {short(counts['neighborhood'])} params")
    block(ax, 71, 32, 23, 18, GEOMETRY, "Latent pooling",
          [f"{tokens} learned query tokens", "attend to all N frame nodes",
           f"{kwargs['num_heads']} heads; all tokens global", "(no FiLM, no radial prior)"],
          f"{tokens} × {width} · {short(counts['pooling'])} params")
    block(ax, 98, 32, 18, 18, GEOMETRY, f"Latent mixing ×{kwargs['latent_layers']}",
          ["self-attention among", f"the {tokens} tokens", "+ feed-forward"],
          f"{tokens} × {width} · {short(counts['mixing'])} params")
    for start, end in ((20.5, 23), (40, 44), (67, 71), (94, 98)):
        arrow(ax, (start, 41), (end, 41))

    # The impact-location input of the reference model, no longer connected.
    ax.add_patch(FancyBboxPatch((72, 54.5), 21, 4.6, boxstyle="round,pad=0,rounding_size=1",
                                facecolor="white", edgecolor=INK_2, linewidth=1.1,
                                linestyle=(0, (3, 2))))
    ax.text(82.5, 56.8, "Impact location (x, y)", ha="center", va="center", fontsize=9.5,
            color=INK_2)
    arrow(ax, (82.5, 54.5), (82.5, 50.2), dashed=True)
    ax.plot(82.5, 52.35, marker="x", markersize=10, markeredgewidth=2.4, color=REMOVED)
    ax.text(84.3, 52.35, "removed", ha="left", va="center", fontsize=9, fontweight="bold",
            color=INK)

    # Time path and decoder.
    block(ax, 1, 9, 19, 17, TIME, "Output times",
          [f"{steps} points, 0–{times.max() * 1e3:.0f} ms",
           f"(every {config['preprocessing']['time_subsample_stride']}th of 1,000",
           "solver samples)"],
          f"{steps} × 1")
    block(ax, 23, 9, 17, 17, TIME, "Time embedding",
          [f"{kwargs['fourier_num_frequencies']} Fourier frequencies",
           f"{kwargs['fourier_min_frequency_hz']:g}–{kwargs['fourier_max_frequency_hz']:g} Hz"
           f" → {2 * kwargs['fourier_num_frequencies']} features",
           f"MLP {2 * kwargs['fourier_num_frequencies']} → {width} → {width}"],
          f"{steps} × {width} · {short(counts['time'])} params")
    block(ax, 44, 9, 35, 17, DECODER, f"Decoder ×{kwargs['temporal_layers']}", [],
          f"{steps} × {width} · {short(counts['decoder'])} params")
    for index, text in enumerate((f"1   Cross-attention to the {tokens} memory tokens",
                                  f"2   Self-attention across the {steps} times",
                                  "3   Feed-forward")):
        y = 19.0 - 3.4 * index
        ax.add_patch(FancyBboxPatch((46.5, y), 30, 2.9, boxstyle="round,pad=0,rounding_size=0.6",
                                    facecolor="white", edgecolor=DECODER[1], linewidth=0.8))
        ax.text(48, y + 1.45, text, ha="left", va="center", fontsize=9, color=INK)
    block(ax, 83, 9, 16, 17, HEAD, "Output head", [f"LayerNorm → {width}", "→ GELU → 1"],
          f"{steps} × 1 · {short(counts['head'])} params")
    for start, end in ((20, 23), (40, 44), (79, 83)):
        arrow(ax, (start, 17.5), (end, 17.5))
    arrow(ax, (99, 17.5), (102.5, 17.5))
    arrow(ax, (107, 32), (107, 29), (61.5, 29), (61.5, 26))
    ax.text(84, 30.3, f"mesh memory: {tokens} tokens × {width}", ha="center", va="center",
            fontsize=9, color=INK)
    draw_output(fig, args.run_dir, run_number)

    # What differs from the reference model.
    ax.text(1, 4.6, "Removed:", ha="left", va="center", fontsize=9, fontweight="bold", color=INK)
    ax.text(10, 4.6, "impact-location input (embedding + FiLM on the pooling queries), radial prior "
            "around the impact, outer skin and hinges; headform rows are loaded but masked",
            ha="left", va="center", fontsize=9, color=INK)
    ax.text(1, 1.9, "Smaller:", ha="left", va="center", fontsize=9, fontweight="bold", color=INK)
    ax.text(10, 1.9, f"width 128 → {width}, tokens 256 → {tokens}, kNN / mixing / decoder layers "
            f"2 / 3 / 2 → {kwargs['neighborhood_layers']} / {kwargs['latent_layers']} / "
            f"{kwargs['temporal_layers']}, dropout 0.1 → {kwargs['dropout']:g}, "
            f"parameters {REFERENCE_PARAMETERS / 1e6:.2f} M → {total / 1e6:.2f} M",
            ha="left", va="center", fontsize=9, color=INK)

    OUTPUT_STEM.parent.mkdir(exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(OUTPUT_STEM.with_suffix(f".{extension}"), dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved {OUTPUT_STEM}.png/svg/pdf ({total:,} parameters)")


if __name__ == "__main__":
    main()
