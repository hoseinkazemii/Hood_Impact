"""
plot_local_edit_designs.py
==========================
Figures of the local-edit designs (N1-N7) read from the PATCHED decks in
Data/HoodImpact_LocalEdits/inp_templates, compared with the base deck
(design 6, HoodImpact_31.inp):

    overview.png   plan view of design 6's inner panel, every notch, the
                   impact locations
    N<k>.png       one row per notch: plan map of dz, 3-D view of the edited
                   panel, cross-section across the beam, profile along it

Also checks that each deck differs from the base only in its node lines.

    python abaqus_scripts/plot_local_edit_designs.py
"""
import argparse
import csv
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from inp_geom import Deck, OuterSurface  # noqa: E402

INNER = "Hood_Inner-1-2"
OUTER = "Hood_Outer-1-2"
MIRROR_Y = 155.3432


def save(fig, path, dpi):
    """Write next to the target, then rename into place, retrying for a few
    seconds: on Windows a viewer or virus scanner can briefly hold the old
    PNG, and truncating it in place then fails with EINVAL."""
    import time
    tmp = path + ".tmp.png"
    fig.savefig(tmp, dpi=dpi)
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except OSError:
            time.sleep(0.5)
    raise OSError("could not replace %s (open in another program?); new figure left at %s"
                  % (path, tmp))


def read_designs(path):
    """design -> (description, [site dict]) from local_edit_designs.csv."""
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            site = {k: float(r[k]) for k in ("x0", "y0", "axis_deg", "dz_mm",
                                             "half_length_mm", "half_width_mm")}
            site["name"] = r["site"]
            out.setdefault(r["design"], (r["description"], []))[1].append(site)
    return out


def frame(xy, site):
    """(along, across) of plan points in a notch's beam frame."""
    a = math.radians(site["axis_deg"])
    ux, uy = math.cos(a), math.sin(a)
    rx, ry = xy[:, 0] - site["x0"], xy[:, 1] - site["y0"]
    return rx * ux + ry * uy, -rx * uy + ry * ux


def node_lines_differ(base, deck):
    """Number of differing lines, and whether they are all node lines."""
    diff = [i for i, (a, b) in enumerate(zip(base.lines, deck.lines)) if a != b]
    node_idx = set(base.node_line_idx.values())
    return len(diff), len(base.lines) == len(deck.lines) and all(i in node_idx for i in diff)


def section(ax, ids, P0, P1, gap0, site, which):
    """Depth below the skin across (which='across') or along the beam."""
    al, cr = frame(P0[:, :2], site)
    if which == "across":
        m = (np.abs(al) < 6.0) & (np.abs(cr) < 60.0)
        x = cr
        ax.set_xlabel("across the beam (mm)")
        ax.set_title("cross-section at the notch centre", fontsize=9)
    else:
        m = (np.abs(cr) < 12.0) & (np.abs(al) < 1.6 * site["half_length_mm"])
        x = al
        ax.set_xlabel("along the beam (mm)")
        ax.set_title("channel floor along the beam", fontsize=9)
    o = np.argsort(x[m])
    d0 = -gap0[m][o]
    d1 = d0 + (P1[m, 2] - P0[m, 2])[o]
    if which == "across":
        # average the few nodes at each cross position into one profile
        xs = np.round(x[m][o] / 1.5) * 1.5
        ux = np.unique(xs)
        d0 = np.array([d0[xs == u].mean() for u in ux])
        d1 = np.array([d1[xs == u].mean() for u in ux])
        xx = ux
    else:
        xs = np.round(x[m][o] / 5.0) * 5.0
        ux = np.unique(xs)
        d0 = np.array([d0[xs == u].min() for u in ux])
        d1 = np.array([d1[xs == u].min() for u in ux])
        xx = ux
    ax.axhline(0, color="0.3", lw=2, label="outer skin")
    ax.plot(xx, d0, "-o", ms=2.5, color="0.55", label="design 6")
    ax.plot(xx, d1, "-o", ms=2.5, color="C3" if site["dz_mm"] > 0 else "C0", label="edited")
    ax.set_ylabel("height relative to skin (mm)")
    ax.set_ylim(min(d0.min(), d1.min()) - 5, 5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")


def plan_dz(ax, P0, dz, site, locs, vmax):
    r = site["half_length_mm"] + 90.0
    box = (np.abs(P0[:, 0] - site["x0"]) < r) & (np.abs(P0[:, 1] - site["y0"]) < r)
    ax.scatter(P0[box, 0], P0[box, 1], s=3, c="0.82", zorder=1)
    mv = box & (np.abs(dz) > 1e-6)
    sc = ax.scatter(P0[mv, 0], P0[mv, 1], s=9, c=dz[mv], cmap="RdBu_r", vmin=-vmax, vmax=vmax, zorder=2)
    near = [(l, x, y) for l, x, y in locs if abs(x - site["x0"]) < r and abs(y - site["y0"]) < r]
    for l, x, y in near:
        ax.scatter([x], [y], s=40, facecolor="none", edgecolor="k", zorder=3)
        ax.annotate(str(l), (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.set_xlim(site["x0"] - r, site["x0"] + r)
    ax.set_ylim(site["y0"] - r, site["y0"] + r)
    ax.set_aspect("equal")
    ax.set_title("plan view: node dz (impact locations circled)", fontsize=9)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    plt.colorbar(sc, ax=ax, shrink=0.75, label="dz (mm)")


def view3d(ax, deck, base, eids, site, vmax, z_scale=2.0):
    """Inner-panel elements around the notch in the beam frame (along,
    across, z), edited surface coloured by dz, design 6 as a grey wireframe."""
    r = site["half_length_mm"] + 30.0
    w = 65.0
    new, old, cols = [], [], []
    cmap = plt.get_cmap("RdBu_r")
    for e in eids:
        con = deck.elems[e]
        c1 = np.array([deck.nodes[n] for n in con])
        c0 = np.array([base.nodes[n] for n in con])
        al, cr = frame(c1[:, :2], site)
        if abs(al.mean()) > r or abs(cr.mean()) > w:
            continue
        a0, r0 = frame(c0[:, :2], site)
        new.append(np.c_[al, cr, c1[:, 2]])
        old.append(np.c_[a0, r0, c0[:, 2]])
        cols.append(cmap(0.5 + 0.5 * (c1[:, 2] - c0[:, 2]).mean() / vmax))
    ax.add_collection3d(Poly3DCollection(new, facecolors=cols, edgecolors="0.3", linewidths=0.2))
    ax.add_collection3d(Poly3DCollection(old, facecolors=(0, 0, 0, 0),
                                         edgecolors=(0.35, 0.35, 0.35, 0.45), linewidths=0.4))
    z = np.vstack(new + old)[:, 2]
    ax.set_xlim(-r, r)
    ax.set_ylim(-w, w)
    ax.set_zlim(z.min() - 2, z.max() + 2)
    ax.set_box_aspect((2 * r, 2 * w, z_scale * (z.max() - z.min() + 4)))
    ax.view_init(elev=24, azim=-62)
    ax.set_xlabel("along (mm)", fontsize=7, labelpad=-2)
    ax.set_ylabel("across (mm)", fontsize=7, labelpad=-2)
    ax.set_zlabel("z (mm)", fontsize=7, labelpad=-2)
    ax.set_title("3-D, beam frame (colour = dz; grey wire = design 6; z x%g)" % z_scale, fontsize=9)
    ax.tick_params(labelsize=6, pad=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=os.path.join(
        ROOT, "Data/HoodImpact_60_IndustryLike/inp_files/HoodImpact_31.inp"))
    ap.add_argument("--templates", default=os.path.join(
        ROOT, "Data/HoodImpact_LocalEdits/inp_templates"))
    ap.add_argument("--locations", default=os.path.join(
        ROOT, "Data/HoodImpact_1704_EuroNCAP/impact_locations_142.csv"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "figures/local_edit_designs"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    designs = read_designs(os.path.join(args.templates, "local_edit_designs.csv"))
    base = Deck(args.base, keep_lines=True)
    ids = sorted(base.elset_nodes(INNER))
    P0 = np.array([base.nodes[i] for i in ids])
    skin = OuterSurface(base)
    gap0 = skin.surf_z(P0[:, 0], P0[:, 1]) - P0[:, 2]
    with open(args.locations, newline="") as f:
        locs = [(int(r["loc"]), float(r["X1"]), float(r["X2"])) for r in csv.DictReader(f)]
    outer = np.array([base.nodes[i] for i in sorted(base.elset_nodes(OUTER))])
    inner_eids = base.elsets[INNER]

    # ---------------------------------------------------------- overview
    fig, ax = plt.subplots(figsize=(13, 11))
    ax.scatter(outer[::5, 0], outer[::5, 1], s=1, c="0.9", zorder=0)
    sc = ax.scatter(P0[:, 0], P0[:, 1], s=2, c=np.clip(gap0, 0, 60), cmap="Greys", vmin=-20, vmax=70, zorder=1)
    plt.colorbar(sc, ax=ax, shrink=0.5, label="inner panel depth below skin (mm), design 6")
    lx = np.array([[x, y] for _, x, y in locs])
    ax.scatter(lx[:, 0], lx[:, 1], s=14, c="k", zorder=3)
    for l, x, y in locs:
        ax.annotate(str(l), (x, y), xytext=(3, 2), textcoords="offset points", fontsize=6, color="0.25")
    ax.axhline(MIRROR_Y, color="0.5", lw=0.8, ls="--")
    ax.text(-470, MIRROR_Y + 8, "mirror line", fontsize=7, color="0.4")
    seen = {}
    for name, (desc, sites) in designs.items():
        for s in sites:
            seen.setdefault(s["name"], []).append((name, s["dz_mm"]))
    pairs = []                   # a notch and its mirror twin share one colour
    for sname in seen:
        if sname.replace("_mirror", "") not in pairs:
            pairs.append(sname.replace("_mirror", ""))
    for sname, users in seen.items():
        k = pairs.index(sname.replace("_mirror", ""))
        s = next(st for _, (_, ss) in designs.items() for st in ss if st["name"] == sname)
        al, cr = frame(P0[:, :2], s)
        m = (np.abs(al) < s["half_length_mm"]) & (np.abs(cr) <= s["half_width_mm"])
        ax.scatter(P0[m, 0], P0[m, 1], s=6, color="C%d" % k, zorder=2)
        label = ", ".join("%s %+g mm" % (n, dz) for n, dz in users)
        ax.annotate(label, (s["x0"], s["y0"]), xytext=(10, -26), textcoords="offset points",
                    fontsize=9, color="C%d" % k, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="C%d" % k, alpha=0.9), zorder=5)
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Local-edit designs on design 6: notch regions (+ = channel floor raised = shallower beam,"
                 " - = deeper)", fontsize=10)
    fig.tight_layout()
    save(fig, os.path.join(args.out_dir, "overview.png"), dpi=110)
    plt.close(fig)

    # ---------------------------------------------------------- per design
    for name, (desc, sites) in designs.items():
        deck = Deck(os.path.join(args.templates, "HoodImpact_%s.inp" % name), keep_lines=True)
        n_diff, only_nodes = node_lines_differ(base, deck)
        P1 = np.array([deck.nodes[i] for i in ids])
        dz = P1[:, 2] - P0[:, 2]
        dxy = np.abs(P1[:, :2] - P0[:, :2]).max()
        vmax = max(abs(s["dz_mm"]) for s in sites)
        fig = plt.figure(figsize=(20, 4.8 * len(sites)))
        for row, s in enumerate(sites):
            c = 4 * row
            plan_dz(fig.add_subplot(len(sites), 4, c + 1), P0, dz, s, locs, vmax)
            view3d(fig.add_subplot(len(sites), 4, c + 2, projection="3d"), deck, base, inner_eids, s, vmax)
            section(fig.add_subplot(len(sites), 4, c + 3), ids, P0, P1, gap0, s, "across")
            section(fig.add_subplot(len(sites), 4, c + 4), ids, P0, P1, gap0, s, "along")
        fig.suptitle("%s: %s   |   %d nodes moved, dz [%.1f, %.1f] mm, max in-plane move %.0e mm, "
                     "deck differs from design 6 in %d lines (%s)"
                     % (name, desc, int((np.abs(dz) > 1e-6).sum()), dz.min(), dz.max(), dxy, n_diff,
                        "node lines only" if only_nodes else "NOT ONLY NODE LINES"), fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save(fig, os.path.join(args.out_dir, "%s.png" % name), dpi=100)
        plt.close(fig)
        print("%s: %d node lines changed, only node lines: %s" % (name, n_diff, only_nodes))
    print("figures -> %s" % args.out_dir)


if __name__ == "__main__":
    main()
