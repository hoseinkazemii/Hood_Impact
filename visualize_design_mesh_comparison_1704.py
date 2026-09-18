"""Render actual hood shell meshes on a common camera and physical scale."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pyvista as pv

from abaqus_scripts.inp_geom import Deck

ROOT = Path(__file__).resolve().parent


def read_clusters():
    for node in ast.parse((ROOT / "mesh_design_sensitivity.py").read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "GEOMETRY_CLUSTERS" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("Geometry cluster mapping missing")


def shell_mesh(deck, element_ids):
    elements = [deck.elems[e] for e in element_ids]
    ids = sorted({n for conn in elements for n in conn})
    lookup = {node: i for i, node in enumerate(ids)}
    points = np.array([deck.nodes[n] for n in ids])
    cells = np.concatenate([np.array([len(conn), *(lookup[n] for n in conn)]) for conn in elements])
    result = pv.PolyData(points, cells)
    result.point_data["node_id"] = np.asarray(ids)
    return result


def load_design(root, design):
    deck = Deck(root / "inp_files" / f"HoodImpact_{142*design+1}.inp")
    impactor = deck.impactor_node_ids()
    structural = [eid for eid, conn in deck.elems.items() if not (set(conn) & impactor)]
    result = {"assembled": shell_mesh(deck, structural)}
    for name, element_set in (("outer", "Hood_Outer-1-2"), ("inner", "Hood_Inner-1-2")):
        result[name] = shell_mesh(deck, deck.elsets[element_set])
        if set(result[name].point_data["node_id"]) & impactor:
            raise ValueError("Headform nodes included in hood panel")
    return result


def configure_view(plotter, center, view="under"):
    directions = {"under": np.array([.8, -1.3, -2.7]),
                  "above": np.array([.8, -1.3, 2.7]),
                  "top": np.array([0., 0., 3.])}
    up = (0, 1, 0) if view == "top" else (0, 0, 1)
    plotter.camera_position = [center + 1200 * directions[view], center, up]
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = 1030 if view != "top" else 920
    plotter.set_background("white")


def draw(plotter, mesh, color="#a8c2d2", scalars=None):
    kwargs = dict(show_edges=True, edge_color="#405769", edge_opacity=.20,
                  line_width=.45, smooth_shading=False, ambient=.35, diffuse=.65,
                  specular=.15, reset_camera=False)
    if scalars is None:
        plotter.add_mesh(mesh, color=color, **kwargs)
    else:
        plotter.add_mesh(mesh, scalars=scalars, cmap="inferno", clim=(0, 40),
                         show_scalar_bar=False, **kwargs)


def labels(plotter, title, subtitle=None):
    plotter.add_text(title, position="upper_left", font_size=14, color="#1f2937", font="arial")
    if subtitle:
        plotter.add_text(subtitle, position="lower_left", font_size=10, color="#4b5563", font="arial")


def save(plotter, path):
    plotter.screenshot(str(path))
    plotter.close()
    print(f"Saved {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "Data/HoodImpact_1704_EuroNCAP")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "figures/design_mesh_geometries_1704")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    clusters = read_clusters()
    cluster_of = {d: name for name, members in clusters.items() for d in members}
    designs = [load_design(args.data_dir, d) for d in range(12)]
    all_points = np.concatenate([d["assembled"].points for d in designs])
    center = (all_points.min(0) + all_points.max(0)) / 2
    print("Parsed all 12 designs; removed headform using its actual node set", flush=True)

    # Actual assembled meshes viewed from below, exposing the reinforcement.
    p = pv.Plotter(shape=(3, 4), off_screen=True, window_size=(3000, 2250), border=False)
    for d, geometry in enumerate(designs):
        p.subplot(d // 4, d % 4)
        draw(p, geometry["assembled"])
        configure_view(p, center)
        labels(p, f"Design {d}  |  Cluster {cluster_of[d]}", "Assembled hood: underside / same scale")
    save(p, args.out_dir / "all_12_hood_meshes.png")

    # The same 12 inner panels, with the outer skin removed, in plan view.
    p = pv.Plotter(shape=(3, 4), off_screen=True, window_size=(3000, 2250), border=False)
    for d, geometry in enumerate(designs):
        p.subplot(d // 4, d % 4)
        draw(p, geometry["inner"])
        configure_view(p, center, "top")
        labels(p, f"Design {d}  |  Cluster {cluster_of[d]}", "Inner hood panel / outer skin hidden")
    save(p, args.out_dir / "all_12_inner_meshes.png")

    representatives = [m[0] for m in clusters.values()]
    p = pv.Plotter(shape=(2, 4), off_screen=True, window_size=(3200, 1650), border=False)
    for col, d in enumerate(representatives):
        for row, component in enumerate(("outer", "inner")):
            p.subplot(row, col)
            draw(p, designs[d][component])
            configure_view(p, center, "above")
            members = ", ".join(map(str, clusters[cluster_of[d]]))
            labels(p, f"{cluster_of[d]}: design {d}  |  {component} panel", f"Cluster designs: {members} / same camera and scale")
    save(p, args.out_dir / "four_clusters_outer_and_inner.png")

    # True closest-surface distance, computed separately for each component.
    # This highlights geometry differences without inventing node correspondence.
    p = pv.Plotter(shape=(1, 4), off_screen=True, window_size=(3200, 900), border=False)
    reference = designs[0]["inner"].triangulate()
    distances = []
    for col, d in enumerate(representatives):
        p.subplot(0, col)
        inner = designs[d]["inner"].copy()
        distance = np.abs(inner.compute_implicit_distance(reference)["implicit_distance"])
        if d == 0:
            distance[:] = 0
        inner["surface_distance_mm"] = distance
        draw(p, inner, scalars="surface_distance_mm")
        configure_view(p, center, "top")
        p.camera.parallel_scale = 1150
        labels(p, f"Cluster {cluster_of[d]}  |  Design {d}", "Distance to A inner surface / 0-40 mm color scale")
        p.add_scalar_bar(title=f"Surface distance to A (mm){' ' * col}", vertical=False, width=.65,
                         height=.065, position_x=.18, position_y=.065, color="#1f2937",
                         title_font_size=16, label_font_size=14, n_labels=5)
        distances.append(dict(design=d, max_surface_distance_mm=float(distance.max()),
                              mean_surface_distance_mm=float(distance.mean())))
    save(p, args.out_dir / "inner_geometry_differences.png")
    metadata = dict(clusters=clusters, source="First impact deck per design in the 1704 dataset",
                    projection="Orthographic, shared camera/scale; XYZ in mm; no vertical exaggeration",
                    geometry="Original S3R/S4 shell connectivity; headform excluded by node set",
                    difference="Absolute closest-surface distance to design 0 inner panel, sampled at each design's nodes; values above 40 mm saturate",
                    distance_summary=distances)
    (args.out_dir / "geometry_views.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
