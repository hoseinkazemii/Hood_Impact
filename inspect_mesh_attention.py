"""Plot one exported head/token without needing the dataset or PyTorch."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path, help="Downloaded attention_export/run_NNNN folder")
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--token", type=int, default=0)
    parser.add_argument("--kind", choices=["pooling", "tokens", "temporal"], default="pooling")
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args()
    if args.kind != "pooling":
        weights = np.load(args.case_dir / f"{args.kind}_self_attention.npy", mmap_mode="r")
        if not 0 <= args.layer < weights.shape[0] or not 0 <= args.head < weights.shape[1]:
            parser.error(f"Layer/head out of range for {weights.shape}")
        matrix = weights[args.layer, args.head]
        labels = (np.load(args.case_dir / "prediction_times_seconds.npy") if args.kind == "temporal"
                  else np.arange(matrix.shape[0]))
        stem = args.case_dir / f"{args.kind}_layer_{args.layer}_head_{args.head}"
        pd.DataFrame(matrix, index=labels, columns=labels).to_csv(stem.with_suffix(".csv"),
                    index_label="query_time_seconds" if args.kind == "temporal" else "query_token")
        fig, ax = plt.subplots(figsize=(8, 7))
        artist = ax.imshow(matrix, cmap="viridis", vmin=0, interpolation="nearest", aspect="auto")
        ax.set(title=f"{args.kind} attention · layer {args.layer}, head {args.head}",
               xlabel="Attended-to token" if args.kind == "tokens" else "Attended-to time (ms)",
               ylabel="Query token" if args.kind == "tokens" else "Query time (ms)")
        if args.kind == "temporal":
            ticks = np.unique(np.linspace(0, len(labels) - 1, min(6, len(labels)), dtype=int))
            tick_labels = [f"{labels[i] * 1000:.2f}" for i in ticks]
            ax.set_xticks(ticks, tick_labels, rotation=45)
            ax.set_yticks(ticks, tick_labels)
        fig.colorbar(artist, ax=ax, label="Attention weight")
        fig.tight_layout()
        fig.savefig(stem.with_suffix(".png"), dpi=200)
        plt.close(fig)
        print(f"Saved {stem}.png and .csv")
        return
    weights = np.load(args.case_dir / "pooling_weights.npy", mmap_mode="r")
    if not 0 <= args.head < weights.shape[0] or not 0 <= args.token < weights.shape[1]:
        parser.error(f"Head/token out of range for {weights.shape}")
    nodes = pd.read_csv(args.case_dir / "node_attention.csv")
    import json
    meta = json.loads((args.case_dir / "metadata.json").read_text())
    probabilities = weights[args.head, args.token]
    nodes["selected_weight"] = probabilities
    stem = args.case_dir / f"head_{args.head}_token_{args.token}"
    nodes.sort_values("selected_weight", ascending=False).to_csv(stem.with_suffix(".csv"), index=False)
    structural = ~nodes.is_headform.to_numpy(bool)
    fig, ax = plt.subplots(figsize=(9, 7))
    points = ax.scatter(nodes.loc[structural, "X2"], nodes.loc[structural, "X1"],
                        c=probabilities[structural], s=3, cmap="viridis", vmin=0, linewidths=0)
    impact = meta["impact_xy_mm"]
    ax.scatter(impact[1], impact[0], c="yellow", edgecolors="black", marker="*", s=170)
    bank = "global" if args.token <= meta["global_token_indices"][1] else "local"
    ax.set(title=f"{args.case_dir.name} · head {args.head}, token {args.token} ({bank})\n"
                 "Structural nodes; headform omitted from view only", xlabel="X2 (mm)", ylabel="X1 (mm)", aspect="equal")
    fig.colorbar(points, ax=ax, label="Attention weight")
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"Saved {stem}.png and .csv | attention sum={probabilities.sum():.8f}")


if __name__ == "__main__":
    main()
