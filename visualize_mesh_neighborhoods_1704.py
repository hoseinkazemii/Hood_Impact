"""Inspect cached kNN neighborhoods at scattered centers on an actual hood mesh."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import numpy as np
from PIL import Image
import torch

from abaqus_scripts.inp_geom import Deck
from mesh_neighborhood import NeighborGraphCache


ROOT = Path(__file__).resolve().parent
PART_NAMES = ("Outer panel", "Inner panel", "Other structure", "Shared outer/inner")
PART_COLORS = ("#237fba", "#e99622", "#8854ad", "#138a7c")
CENTER_COLOR = "#d61c4e"


def scattered_centers(points, count):
    """Deterministic farthest-point sampling in physical 3D, starting centrally."""
    points = np.asarray(points, dtype=np.float64)
    if not 1 <= count <= len(points):
        raise ValueError("Sample count must be between one and the number of structural nodes")
    chosen = [int(np.argmin(np.square(points - points.mean(0)).sum(1)))]
    distance = np.square(points - points[chosen[0]]).sum(1)
    distance[chosen[0]] = -1
    while len(chosen) < count:
        center = int(np.argmax(distance))
        chosen.append(center)
        distance = np.minimum(distance, np.square(points - points[center]).sum(1))
        distance[chosen] = -1
    return np.asarray(chosen, dtype=np.int64)


def load_geometry(path, impactor_nodes):
    deck = Deck(path)
    all_ids = np.asarray(list(deck.nodes), dtype=np.int64)
    all_points = np.asarray(list(deck.nodes.values()), dtype=np.float32)
    if impactor_nodes and set(all_ids[:impactor_nodes]) != deck.impactor_node_ids():
        raise ValueError("The held-out prefix does not match the headform node set")
    ids, points = all_ids[impactor_nodes:], all_points[impactor_nodes:]
    lookup = {int(node): index for index, node in enumerate(ids)}
    outer_elements = set(deck.elsets["Hood_Outer-1-2"])
    inner_elements = set(deck.elsets["Hood_Inner-1-2"])
    outer_ids = deck.elset_nodes("Hood_Outer-1-2")
    inner_ids = deck.elset_nodes("Hood_Inner-1-2")
    outer = np.asarray([int(node) in outer_ids for node in ids])
    inner = np.asarray([int(node) in inner_ids for node in ids])
    parts = np.where(outer & inner, 3, np.where(outer, 0, np.where(inner, 1, 2)))
    edge_groups = [[] for _ in range(3)]
    for element_id, connection in deck.elems.items():
        if not all(node in lookup for node in connection):
            continue
        group = 0 if element_id in outer_elements else 1 if element_id in inner_elements else 2
        rows = [lookup[node] for node in connection]
        edge_groups[group].extend((min(a, b), max(a, b)) for a, b in zip(rows, rows[1:] + rows[:1]))
    edges = [np.unique(np.asarray(group, dtype=np.int32).reshape(-1, 2), axis=0) for group in edge_groups]
    return ids, points, parts, edges


def sample_metrics(points, ids, parts, centers, graph, impactor_nodes):
    metrics = []
    for sample, center in enumerate(centers, 1):
        selected = graph[center]
        offsets = points[selected].astype(np.float64) - points[center].astype(np.float64)
        distances = np.linalg.norm(offsets, axis=1)
        counts = [int(np.sum(parts[selected] == part)) for part in range(4)]
        row = dict(sample_id=sample, center_node_id=int(ids[center]), center_structural_index=int(center),
                   center_full_mesh_row=int(center + impactor_nodes), center_panel=PART_NAMES[parts[center]],
                   x_mm=float(points[center, 0]), y_mm=float(points[center, 1]), z_mm=float(points[center, 2]),
                   selected_count=len(selected), includes_self=bool(center in selected),
                   radius_mm=float(distances.max()), median_distance_mm=float(np.median(distances)),
                   span_x_mm=float(np.ptp(points[selected, 0])), span_y_mm=float(np.ptp(points[selected, 1])),
                   span_z_mm=float(np.ptp(points[selected, 2])), outer_count=counts[0], inner_count=counts[1],
                   other_count=counts[2], shared_count=counts[3],
                   mixes_outer_and_inner=bool((counts[0] + counts[3]) and (counts[1] + counts[3])),
                   other_panel_fraction=float(np.mean(parts[selected] != parts[center])),
                   image=f"per_node/sample_{sample:03d}_node_{int(ids[center])}.png")
        metrics.append(row)
    return metrics


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plan_background(points, edges, bounds):
    fig, ax = plt.subplots(figsize=(7, 7), dpi=125)
    combined = np.concatenate(edges)
    ax.add_collection(LineCollection(points[combined, :2], colors="#c8cdd3", linewidths=.26))
    ax.set(xlim=bounds[:2], ylim=bounds[2:])
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    stream = io.BytesIO()
    fig.savefig(stream, format="png", transparent=False)
    plt.close(fig)
    stream.seek(0)
    return np.asarray(Image.open(stream).convert("RGB"))


def context(ax, background, bounds):
    ax.imshow(background, extent=bounds, origin="upper", interpolation="bilinear")
    ax.set(xlim=bounds[:2], ylim=bounds[2:])
    ax.set_aspect("equal")


def highlighted(ax, points, parts, selected, center, dimensions=(0, 1), size=11, center_size=95):
    selected = selected[selected != center]
    for part, color in enumerate(PART_COLORS):
        rows = selected[parts[selected] == part]
        if len(rows):
            args = [points[rows, dimension] for dimension in dimensions]
            options = dict(s=size, color=color, linewidths=0, zorder=5)
            if len(dimensions) == 3:
                options["depthshade"] = False
            ax.scatter(*args, **options)
    options = dict(s=center_size, c=CENTER_COLOR, marker="*", edgecolors="white", linewidths=.7, zorder=10)
    if len(dimensions) == 3:
        options["depthshade"] = False
    ax.scatter(*[points[center, dimension] for dimension in dimensions], **options)


def legend_handles():
    handles = [Line2D([], [], marker="*", color="none", markerfacecolor=CENTER_COLOR,
                      markeredgecolor="white", markersize=13, label="Center node")]
    handles += [Line2D([], [], marker="o", color="none", markerfacecolor=color, markersize=7, label=name)
                for name, color in zip(PART_NAMES, PART_COLORS)]
    return handles


def static_exports(output, points, parts, centers, graph, metrics, background, bounds, design):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    radii = np.asarray([row["radius_mm"] for row in metrics])
    k = graph.shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(15, 9))
    for ax in axes:
        context(ax, background, bounds)
        ax.set(xlabel="X (mm)", ylabel="Y (mm)")
    axes[0].scatter(points[centers, 0], points[centers, 1], s=18, c=CENTER_COLOR, zorder=4)
    for sample, center in enumerate(centers, 1):
        axes[0].annotate(str(sample), points[center, :2], xytext=(3, 3), textcoords="offset points", fontsize=6)
    axes[0].set_title(f"{len(centers)} scattered centers | sample labels match the gallery")
    scatter = axes[1].scatter(points[centers, 0], points[centers, 1], c=radii, s=65,
                              cmap="viridis", edgecolors="white", linewidths=.5)
    axes[1].set_title("Distance from center to farthest selected neighbor")
    fig.colorbar(scatter, ax=axes[1], shrink=.7, label=f"k{k} neighborhood radius (mm)")
    fig.suptitle(f"Design {design} | physical XYZ neighbors | headform excluded", fontsize=17)
    fig.tight_layout()
    fig.savefig(output / "sample_centers_and_radii.png", dpi=180)
    plt.close(fig)

    columns = 10
    rows = int(np.ceil(len(centers) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(25, 2.65 * rows), squeeze=False)
    for index, ax in enumerate(axes.flat):
        if index >= len(centers):
            ax.set_axis_off()
            continue
        center = centers[index]
        context(ax, background, bounds)
        highlighted(ax, points, parts, graph[center], center, size=1.8, center_size=22)
        ax.set_title(f"#{index+1:03d} | Node {metrics[index]['center_node_id']}\nR = {radii[index]:.1f} mm", fontsize=7)
        ax.set_axis_off()
    fig.suptitle(f"Design {design}: {len(centers)} k{k} neighborhoods | all panels at the same physical scale", fontsize=20)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=5, frameon=False, fontsize=13)
    fig.subplots_adjust(top=.95, bottom=.035, left=.01, right=.99, hspace=.2, wspace=.05)
    fig.savefig(output / f"all_{len(centers)}_neighborhoods.png", dpi=180)
    fig.savefig(output / f"all_{len(centers)}_neighborhoods.pdf")
    plt.close(fig)

    # A fixed local scale makes the physical extents comparable across samples.
    half_width = float(radii.max() * 1.25)
    (output / "per_node").mkdir(exist_ok=True)
    with PdfPages(output / f"neighborhood_details_{len(centers)}_pages.pdf") as pdf:
        pdf.infodict()["Title"] = f"Design {design}: cached neighborhoods at {len(centers)} structural nodes"
        for index, center in enumerate(centers):
            row, selected = metrics[index], graph[center]
            cx, cy, cz = points[center]
            nearby = np.where(np.all(np.abs(points - points[center]) <= half_width, axis=1))[0]
            fig = plt.figure(figsize=(13.5, 10))
            whole = fig.add_subplot(221)
            local_plan = fig.add_subplot(222)
            local_3d = fig.add_subplot(223, projection="3d", computed_zorder=False)
            side = fig.add_subplot(224)
            context(whole, background, bounds)
            highlighted(whole, points, parts, selected, center, size=6, center_size=100)
            whole.add_patch(Rectangle((cx-half_width, cy-half_width), 2*half_width, 2*half_width,
                                      fill=False, edgecolor="#526375", linewidth=.9))
            whole.set(title="Whole structural mesh | box marks the local views", xlabel="X (mm)", ylabel="Y (mm)")
            for ax, dimensions in ((local_plan, (0, 1)), (side, (0, 2))):
                ax.scatter(points[nearby, dimensions[0]], points[nearby, dimensions[1]],
                           s=3, color="#ccd1d7", linewidths=0, zorder=1)
                highlighted(ax, points, parts, selected, center, dimensions, size=12)
                location = (points[center, dimensions[0]], points[center, dimensions[1]])
                ax.add_patch(Circle(location, row["radius_mm"], fill=False, color="#536475", linewidth=.9, linestyle="--"))
                ax.add_patch(Circle(location, 20, fill=False, color="#a8afb7", linewidth=.7))
                ax.set(xlim=(location[0]-half_width, location[0]+half_width),
                       ylim=(location[1]-half_width, location[1]+half_width),
                       xlabel=f"{'XYZ'[dimensions[0]]} (mm)", ylabel=f"{'XYZ'[dimensions[1]]} (mm)")
                ax.set_aspect("equal")
                ax.grid(alpha=.12)
            local_plan.set_title("Local plan view | same scale for all 100 samples")
            side.set_title("Local X-Z view | shows separation between panels")
            local_3d.scatter(*points[nearby].T, s=3, color="#ccd1d7", alpha=.35, linewidths=0, depthshade=False)
            highlighted(local_3d, points, parts, selected, center, (0, 1, 2), size=14)
            local_3d.set(xlim=(cx-half_width, cx+half_width), ylim=(cy-half_width, cy+half_width),
                         zlim=(cz-half_width, cz+half_width), xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)",
                         title="Local 3D view | equal XYZ scale, no vertical exaggeration")
            local_3d.set_box_aspect((1, 1, 1))
            local_3d.view_init(elev=28, azim=-60)
            local_3d.tick_params(labelsize=7)
            fig.suptitle(f"Sample #{index+1:03d} | Abaqus node {row['center_node_id']} | center: {row['center_panel']}\n"
                         f"{len(selected)} entries including self | farthest distance {row['radius_mm']:.1f} mm | "
                         f"outer {row['outer_count']}, inner {row['inner_count']}, other {row['other_count']}, shared {row['shared_count']}", fontsize=15)
            fig.legend(handles=legend_handles(), loc="lower center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(.5, .038))
            fig.text(.5, .012, "Dashed circle: projected bounding sphere of selected nodes. Solid small circle: 20 mm reference, not a cutoff.",
                     ha="center", fontsize=9, color="#596473")
            fig.subplots_adjust(left=.07, right=.96, top=.85, bottom=.11, hspace=.3, wspace=.3)
            fig.savefig(output / row["image"], dpi=135)
            pdf.savefig(fig, dpi=100)
            plt.close(fig)
            if (index+1) % 10 == 0 or index+1 == len(centers):
                print(f"Rendered {index+1}/{len(centers)} detailed neighborhood views", flush=True)
    return half_width


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "Data/HoodImpact_1704_EuroNCAP")
    parser.add_argument("--design", type=int, default=0)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--impactor-nodes", type=int, default=286)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if not 0 <= args.design < 12 or args.impactor_nodes < 0 or args.k < 1:
        parser.error("Require design in 0..11, impactor-nodes >= 0 and k > 0")
    output = args.output_dir or ROOT / "figures/mesh_neighborhoods" / f"design_{args.design:02d}_k{args.k}"
    output.mkdir(parents=True, exist_ok=True)
    source = args.data_root / "inp_files" / f"HoodImpact_{args.design*142+1}.inp"
    ids, points, parts, edges = load_geometry(source, args.impactor_nodes)
    cache = NeighborGraphCache(directory=args.cache_dir or args.data_root / "neighbor_graphs")
    graph = cache.neighbors(torch.from_numpy(points), args.k).cpu().numpy()
    centers = scattered_centers(points, args.samples)
    metrics = sample_metrics(points, ids, parts, centers, graph, args.impactor_nodes)
    write_csv(output / "neighborhood_metrics.csv", metrics)
    extracted = []
    for row, center in zip(metrics, centers):
        for rank, neighbor in enumerate(graph[center], 1):
            extracted.append(dict(sample_id=row["sample_id"], center_node_id=int(ids[center]),
                neighbor_rank=rank, neighbor_node_id=int(ids[neighbor]), neighbor_structural_index=int(neighbor),
                neighbor_full_mesh_row=int(neighbor+args.impactor_nodes), panel=PART_NAMES[parts[neighbor]],
                x_mm=float(points[neighbor, 0]), y_mm=float(points[neighbor, 1]), z_mm=float(points[neighbor, 2]),
                distance_mm=float(np.linalg.norm(points[neighbor].astype(float)-points[center].astype(float))),
                is_center=bool(neighbor == center)))
    write_csv(output / "selected_neighbors.csv", extracted)
    radii = np.asarray([row["radius_mm"] for row in metrics])
    metadata = dict(design_id=args.design, source=str(source.resolve()), structural_nodes=len(points),
        impactor_nodes_excluded=args.impactor_nodes, requested_k=args.k, sample_count=len(centers),
        sampling="Deterministic farthest-point sampling in physical XYZ; first point nearest the structural centroid",
        units="mm", coordinate_dtype="float32, matching the model input geometry",
        cache_file=str(cache.path_for(points, args.k).resolve()), cache_builds=cache.builds, cache_disk_loads=cache.disk_hits,
        selected_entries=len(extracted), radius_min_mm=float(radii.min()), radius_median_mm=float(np.median(radii)),
        radius_max_mm=float(radii.max()), mixed_outer_inner_neighborhoods=sum(row["mixes_outer_and_inner"] for row in metrics),
        sample_center_panel_counts={name:int(np.sum(parts[centers] == part)) for part, name in enumerate(PART_NAMES)},
        interpretation="Candidate kNN indices, with center included. Colors identify panel membership; learned attention weights are not plotted.")
    print(json.dumps(metadata, indent=2), flush=True)
    xy_min, xy_max = points[:, :2].min(0), points[:, :2].max(0)
    midpoint = (xy_min + xy_max) / 2
    half = float((xy_max - xy_min).max() * .53)
    bounds = (midpoint[0]-half, midpoint[0]+half, midpoint[1]-half, midpoint[1]+half)
    background = plan_background(points, edges, bounds)
    metadata["local_plot_half_width_mm"] = static_exports(output, points, parts, centers, graph, metrics, background, bounds, args.design)
    payload = dict(points=points.round(5).reshape(-1).tolist(), ids=ids.tolist(), parts=parts.tolist(),
                   edges=[group.reshape(-1).tolist() for group in edges], centers=centers.tolist(),
                   neighbors=graph[centers].tolist(), metrics=metrics, meta=metadata,
                   partNames=PART_NAMES, partColors=PART_COLORS, centerColor=CENTER_COLOR)
    template = (ROOT / "mesh_neighborhood_viewer.html").read_text(encoding="utf-8")
    (output / "neighborhood_viewer.html").write_text(template.replace("__NEIGHBORHOOD_DATA__", json.dumps(payload, separators=(",", ":"))), encoding="utf-8")
    (output / "visualization_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved viewer, {len(centers)} detailed images, PDF atlas and CSVs to {output.resolve()}", flush=True)
    return metadata


if __name__ == "__main__":
    main()
