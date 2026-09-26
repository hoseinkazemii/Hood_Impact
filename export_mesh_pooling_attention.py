"""Export direct pooling, token self-attention and temporal self-attention weights.

Default: every run in the saved test split, full float32 [head, token, node]
NPY arrays, CSV summaries, and PNG maps. Attention is a learned pooling weight,
not a causal importance score or an attribution for a particular output time.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_mesh_impact_history import CHECKPOINT_NAME, HistoryPredictor, write_json


def read_nodes(path):
    ids, xyz, recording = [], [], False
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if not recording:
                recording = line.upper() == "*NODE"
                continue
            if line.startswith("*"):
                break
            parts = line.split(",")
            if len(parts) >= 4:
                ids.append(int(parts[0]))
                xyz.append([float(x) for x in parts[1:4]])
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"Missing or duplicate node IDs: {path}")
    xyz = np.asarray(xyz, dtype=np.float32)
    if not np.isfinite(xyz).all():
        raise ValueError(f"Nonfinite node coordinates: {path}")
    return np.asarray(ids, dtype=np.int64), xyz


def plot_nodes(frame, impact, columns, output, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    structural = ~frame.is_headform.to_numpy(bool)
    fig, axes = plt.subplots(1, len(columns), figsize=(6 * len(columns), 5.8), squeeze=False)
    for ax, column in zip(axes[0], columns):
        color = frame[column].to_numpy()
        artist = ax.scatter(frame.loc[structural, "X2"], frame.loc[structural, "X1"],
                            c=color[structural], s=2, cmap="viridis", vmin=0,
                            rasterized=True, linewidths=0)
        ax.scatter(impact[1], impact[0], marker="*", s=150, c="yellow", edgecolors="black", label="Impact")
        ax.set(title=column, xlabel="X2 (mm)", ylabel="X1 (mm)", aspect="equal")
        ax.legend(loc="upper right", fontsize=8)
        fig.colorbar(artist, ax=ax, label="Attention weight (mean)", shrink=.75)
    fig.suptitle(title + "\nStructural nodes shown; headform weights retained in files", fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def capture_self_attention(predictor, xyz, impact):
    """Ask the actual forward-pass MHA modules for unaveraged head weights.

    Hooks are temporary, export-only, and removed even on failure. The model's
    blocks consume the attention output normally; weights are copied to CPU.
    """
    captured, handles = {}, []
    def request_weights(module, inputs, kwargs):
        return inputs, {**kwargs, "need_weights": True, "average_attn_weights": False}
    def save_weights(key):
        def hook(module, inputs, output):
            weights = output[1].detach().float().cpu().numpy()
            if weights.shape[0] != 1:
                raise ValueError("Export expects one case per forward pass")
            captured[key] = weights[0]
        return hook
    modules = [("tokens", i, block.attention) for i, block in enumerate(predictor.model.latent_blocks)]
    modules += [("temporal", i, block.time_attention) for i, block in enumerate(predictor.model.decoder_blocks)
                if hasattr(block, "time_attention")]
    try:
        for kind, layer, module in modules:
            handles.append(module.register_forward_pre_hook(request_weights, with_kwargs=True))
            handles.append(module.register_forward_hook(save_weights((kind, layer))))
        prediction = predictor.predict(xyz, impact)
    finally:
        for handle in handles:
            handle.remove()
    result = {}
    for kind in ("tokens", "temporal"):
        layers = [captured[key] for key in sorted(captured) if key[0] == kind]
        if layers:
            weights = np.stack(layers)
            if not np.isfinite(weights).all() or (weights < 0).any() or not np.allclose(weights.sum(-1), 1, atol=1e-5):
                raise ValueError(f"Invalid {kind} attention weights")
            result[kind] = weights
    return result, prediction


def plot_self_attention(weights, kind, times, output):
    import matplotlib.pyplot as plt
    layers, heads = weights.shape[:2]
    fig, axes = plt.subplots(layers, heads, figsize=(3.7 * heads, 3.3 * layers), squeeze=False)
    maximum = float(weights.max())
    for layer in range(layers):
        for head in range(heads):
            ax = axes[layer, head]
            artist = ax.imshow(weights[layer, head], cmap="viridis", vmin=0, vmax=maximum,
                               interpolation="nearest", aspect="auto")
            ax.set_title(f"Layer {layer}, head {head}", fontsize=10)
            ax.set_xlabel("Attended-to token" if kind == "tokens" else "Attended-to time (ms)")
            ax.set_ylabel("Query token" if kind == "tokens" else "Query time (ms)")
            if kind == "temporal":
                ticks = np.unique(np.linspace(0, len(times) - 1, min(5, len(times)), dtype=int))
                labels = [f"{times[i] * 1000:.2f}" for i in ticks]
                ax.set_xticks(ticks, labels, rotation=45)
                ax.set_yticks(ticks, labels)
            fig.colorbar(artist, ax=ax, label="Attention weight", shrink=.8)
    fig.suptitle(f"{kind.capitalize()} self-attention · direct model weights", fontsize=13)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def export_case(predictor, node_ids, xyz, impact, directory, *, full=True, chunk_size=16):
    directory.mkdir(parents=True, exist_ok=False)
    model = predictor.model
    heads, tokens, nodes = model.num_heads, model.num_latents, len(xyz)
    global_count = model.num_global_latents
    device = predictor.device
    mesh = torch.as_tensor(predictor.preprocessor.transform_mesh(xyz), dtype=torch.float32, device=device)
    point = torch.as_tensor(predictor.preprocessor.transform_indentor(impact), dtype=torch.float32, device=device)
    weights_path = directory / "pooling_weights.npy"
    weights_file = np.lib.format.open_memmap(weights_path, mode="w+", dtype=np.float32,
                                            shape=(heads, tokens, nodes)) if full else None
    sums = np.zeros((2, heads, nodes), dtype=np.float64)
    max_row_error = 0.
    for start, weights in model.iter_pooling_attention(mesh, point, chunk_size):
        array = weights.cpu().numpy()
        if not np.isfinite(array).all() or (array < 0).any():
            raise ValueError("Invalid attention probabilities")
        max_row_error = max(max_row_error, float(np.max(np.abs(array.sum(-1, dtype=np.float64) - 1))))
        if weights_file is not None:
            weights_file[:, start:start + array.shape[1]] = array
        for j in range(array.shape[1]):
            sums[int(start + j >= global_count)] += array[:, j]
    if max_row_error > 1e-4:
        raise ValueError(f"Attention rows do not sum to one: {max_row_error}")
    if weights_file is not None:
        weights_file.flush()
        del weights_file
    headform = np.arange(nodes) < model.impactor_nodes
    frame = pd.DataFrame({"node_index": np.arange(nodes), "node_id": node_ids,
                          "X1": xyz[:, 0], "X2": xyz[:, 1], "X3": xyz[:, 2], "is_headform": headform,
                          "mean_all": sums.sum(axis=(0, 1)) / (heads * tokens),
                          "mean_global": sums[0].sum(0) / (heads * global_count),
                          "mean_local": sums[1].sum(0) / (heads * (tokens - global_count))})
    for head in range(heads):
        frame[f"mean_head_{head}"] = sums[:, head].sum(0) / tokens
    frame.to_csv(directory / "node_attention.csv", index=False)
    frame.loc[~frame.is_headform].nlargest(100, "mean_all").to_csv(directory / "top100_structural_nodes.csv", index=False)
    np.save(directory / "node_ids.npy", node_ids, allow_pickle=False)
    np.save(directory / "node_xyz_mm.npy", xyz, allow_pickle=False)
    plot_nodes(frame, impact, ["mean_all", "mean_global", "mean_local"], directory / "hood_attention.png",
               directory.name + " · best-checkpoint pooling attention")
    self_attention, prediction = capture_self_attention(predictor, xyz, impact)
    np.save(directory / "prediction_times_seconds.npy", predictor.time_points, allow_pickle=False)
    np.save(directory / "predicted_acceleration_g.npy", prediction, allow_pickle=False)
    for kind, weights in self_attention.items():
        np.save(directory / f"{kind}_self_attention.npy", weights, allow_pickle=False)
        plot_self_attention(weights, kind, predictor.time_points, directory / f"{kind}_self_attention.png")
    return {"shape_head_token_node": [heads, tokens, nodes], "dtype": "float32",
            "self_attention_shapes_layer_head_query_key": {key: list(value.shape) for key, value in self_attention.items()},
            "attention_values": "Direct post-softmax model weights; no logarithm or additional rescaling",
            "temporal_attention_available": "temporal" in self_attention,
            "full_weights": full, "global_token_indices": [0, global_count - 1],
            "local_token_indices": [global_count, tokens - 1], "max_row_sum_error": max_row_error,
            "headform_mean_attention_mass": float(frame.loc[frame.is_headform, "mean_all"].sum()),
            "impact_xy_mm": np.asarray(impact).tolist(), "headform_nodes": model.impactor_nodes}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--data-root", type=Path, help="Override saved dataset paths after moving files")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="test")
    parser.add_argument("--runs", type=int, nargs="+", help="Original run numbers, restricted to the selected split")
    parser.add_argument("--summary-only", action="store_true", help="Omit full matrices; still compute exact summaries")
    parser.add_argument("--token-chunk-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.token_chunk_size < 1:
        parser.error("--token-chunk-size must be positive")
    saved = json.loads((args.run_dir / "config.json").read_text())
    if saved["architecture"]["name"] != "MeshImpactHistoryNet" or saved["data"]["data_format"] != "euroncap1704":
        parser.error("Export requires a MeshImpactHistoryNet euroncap1704 run")
    splits = json.loads((args.run_dir / "splits.json").read_text())
    selected = sorted({run for key in (splits if args.split == "all" else [args.split]) for run in splits[key]["run_numbers"]})
    if args.runs:
        if not set(args.runs) <= set(selected):
            parser.error("Requested runs must belong to the saved selected split")
        selected = sorted(set(args.runs))
    root = args.output_dir or args.run_dir / "attention_export"
    if root.exists():
        parser.error(f"Output already exists: {root}; choose a new --output-dir")
    inp_dir = args.data_root / "inp_files" if args.data_root else Path(saved["data"]["inp_dir"])
    impact_path = args.data_root / "ImpactCoords_1704.csv" if args.data_root else Path(saved["data"]["impact_coords_path"])
    impacts = pd.read_csv(impact_path)
    predictor = HistoryPredictor.from_run(args.run_dir, device=args.device)
    root.mkdir(parents=True)
    with (args.run_dir / CHECKPOINT_NAME).open("rb") as handle:
        checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    manifest = {"status": "in_progress", "checkpoint": str((args.run_dir / CHECKPOINT_NAME).resolve()),
                "checkpoint_sha256": checkpoint_hash, "split": args.split, "run_numbers": selected,
                "stages": ["mesh_to_token_pooling", "token_self_attention", "temporal_self_attention"],
                "dropout": "off (eval mode)",
                "normalization": "Model softmax over keys; no additional transformation",
                "interpretation": "Pooling probabilities, not causal importance or per-time output attribution",
                "cases": []}
    write_json(root / "manifest.json", manifest)
    for run in selected:
        ids, xyz = read_nodes(inp_dir / f"HoodImpact_{run}.inp")
        impact = impacts.iloc[run - 1][["X1", "X2"]].to_numpy(np.float32)
        metadata = export_case(predictor, ids, xyz, impact, root / f"run_{run:04d}",
                               full=not args.summary_only, chunk_size=args.token_chunk_size)
        metadata.update(run_number=run, design_id=(run - 1) // saved["data"]["samples_per_design"],
                        location_id=(run - 1) % saved["data"]["samples_per_design"] + 1)
        write_json(root / f"run_{run:04d}" / "metadata.json", metadata)
        manifest["cases"].append(metadata)
        write_json(root / "manifest.json", manifest)
        print(f"Exported run {run}: {metadata['shape_head_token_node']}", flush=True)
    manifest["status"] = "complete"
    write_json(root / "manifest.json", manifest)
    print(f"Attention export complete: {root.resolve()}")


if __name__ == "__main__":
    main()
