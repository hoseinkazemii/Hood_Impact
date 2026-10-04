"""Frozen-model geometry probes: activations, pooling interventions, and matched curves.

These are descriptive post-training diagnostics, including held-out designs.
No weights, graph settings, scalers, or training data selections are fitted.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
import types

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from export_mesh_pooling_attention import read_nodes
from train_mesh_impact_history import HistoryPredictor


def rms(x):
    return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64)))))


@torch.inference_mode()
def decode_pooled(model, pooled, normalized_times):
    memory = pooled[None]
    for block in model.latent_blocks:
        memory = block(memory)
    memory = model.memory_norm(memory)
    t = normalized_times
    features = model.time_features(t)
    queries = model.time_embedding(features)[None]
    mask = torch.zeros((1, len(t)), dtype=torch.bool, device=t.device)
    for block in model.decoder_blocks:
        queries = block(queries, memory, mask)
    return model.acceleration_head(queries)[0, :, 0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("Data/HoodImpact_1704_EuroNCAP"))
    parser.add_argument("--locations", type=int, nargs="+", default=[9, 115, 137, 142])
    parser.add_argument("--designs", type=int, nargs="+", default=[0, 1, 4, 5, 6, 7, 10, 11])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    out = args.output_dir or args.run_dir / "design_sensitivity_audit"
    out.mkdir(parents=True, exist_ok=True)
    predictor = HistoryPredictor.from_run(args.run_dir, device=args.device)
    model = predictor.model
    if model.impact_conditioning != "film":
        raise ValueError("This probe implements the FiLM time-only decoder")
    for block in model.neighborhood_blocks:
        block.chunk_size = min(block.chunk_size, 128)
    config = json.loads((args.run_dir / "config.json").read_text())
    splits = json.loads((args.run_dir / "splits.json").read_text())
    count = config["data"]["samples_per_design"]
    xy = pd.read_csv(args.data_root / "ImpactCoords_1704.csv")[["X1", "X2"]].to_numpy(np.float32)
    saved = pd.read_csv(args.run_dir / "test_acceleration_histories.csv")
    scale = float(predictor.preprocessor.accel_scaler.scale_[0])
    mean = float(predictor.preprocessor.accel_scaler.mean_[0])
    normalized_times = torch.tensor(predictor.preprocessor.transform_time(predictor.time_points),
                                    dtype=torch.float32, device=args.device)
    cache, pooling = {}, {}
    def hook(name):
        def capture(module, inputs, result):
            result = result[0] if isinstance(result, tuple) else result
            cache[name] = result.detach().float().cpu().numpy().copy()
        return capture
    modules = {"node_embedding": model.node_embedding, "pooling_update": model.mesh_output,
               "final_memory": model.memory_norm, "film_tokens": model.impact_film}
    modules.update({f"neighborhood_{i+1}": b for i, b in enumerate(model.neighborhood_blocks)})
    modules.update({f"latent_{i+1}": b for i, b in enumerate(model.latent_blocks)})
    modules.update({f"decoder_{i+1}": b for i, b in enumerate(model.decoder_blocks)})
    handles = [module.register_forward_hook(hook(name)) for name, module in modules.items()]
    original = model._mesh_pooling_inputs
    def record_pooling(self, *inputs):
        result = original(*inputs)
        pooling["args"] = tuple(v.detach() for v in result)
        return result
    model._mesh_pooling_inputs = types.MethodType(record_pooling, model)
    curves, traces, interventions, masses, residuals, alternate_curves = [], [], [], [], [], []
    replay_errors = []
    decoder_replay_errors = []
    try:
        for location in args.locations:
            previous = {}
            for design in args.designs:
                start = time.monotonic()
                run = design * count + location
                ids, xyz = read_nodes(args.data_root / "inp_files" / f"HoodImpact_{run}.inp")
                pred = predictor.predict(xyz, xy[run-1])
                # Capture before intervention forwards overwrite hooks.
                captured = dict(cache)
                tokens, query, key, value, bias = pooling["args"]
                history = pd.read_csv(args.data_root / "output_history_acc" / f"HoodImpact_{run}_SAE1000_interp1000.csv")
                if config["preprocessing"].get("max_train_time") is not None:
                    history = history[history.Time <= config["preprocessing"]["max_train_time"]]
                history = history.iloc[::config["preprocessing"]["time_subsample_stride"]]
                np.testing.assert_allclose(history.Time, predictor.time_points, atol=1e-9, rtol=1e-6)
                truth = history["A(in g)"].to_numpy()
                split = next(name for name, data in splits.items() if design in data["design_ids"])
                for t, y, p in zip(predictor.time_points, truth, pred):
                    curves.append(dict(run_number=run, design_id=design, location_id=location, split=split,
                                       time=float(t), acceleration_true_g=float(y), acceleration_pred_g=float(p)))
                if run in set(saved.run_number):
                    replay_errors.append(float(np.max(np.abs(pred-saved.loc[saved.run_number == run, "acceleration_pred_g"].to_numpy()))))
                structural_reference = previous.get(design-1)
                if structural_reference is not None and design in (1, 5, 7, 11):
                    old_ids, old_xyz, old = structural_reference
                    np.testing.assert_array_equal(ids, old_ids)
                    for name, feature in captured.items():
                        a, b = old[name], feature
                        if name == "node_embedding":
                            a, b = a[model.impactor_nodes:], b[model.impactor_nodes:]
                        delta = rms(b-a)
                        traces.append(dict(location_id=location, pair=f"{design-1}-{design}", stage=name,
                                           difference_rms=delta, relative_difference=delta/max(rms(a), 1e-30),
                                           reference_rms=rms(a)))
                    changed = np.linalg.norm(xyz-old_xyz, axis=1) > .5
                    changed[:model.impactor_nodes] = False
                else:
                    changed = None
                if design in (0, 4, 6, 10):
                    previous[design] = (ids, xyz, captured)
                with torch.inference_mode():
                    # Report direct attention mass, separately for global/local banks.
                    headform_mass, changed_mass = [], []
                    for first in range(0, model.num_latents, 16):
                        weights = torch.softmax(query[:, first:first+16] @ key.transpose(-1,-2)
                            / (model.width // model.num_heads)**.5 + bias[None,first:first+16], dim=-1)
                        headform_mass.append(weights[:,:,:model.impactor_nodes].sum(-1).mean(0).cpu().numpy())
                        if changed is not None:
                            mask = torch.as_tensor(changed, device=query.device)
                            changed_mass.append(weights[:,:,mask].sum(-1).mean(0).cpu().numpy())
                    hf = np.concatenate(headform_mass)
                    cm = np.concatenate(changed_mass) if changed_mass else None
                    for bank, sl in (("global",slice(0,model.num_global_latents)),("local",slice(model.num_global_latents,None))):
                        masses.append(dict(design_id=design,location_id=location,bank=bank,
                            headform_mass=float(hf[sl].mean()),headform_uniform_fraction=model.impactor_nodes/len(xyz),
                            changed_node_mass=float(cm[sl].mean()) if cm is not None else None,
                            changed_node_uniform_fraction=float(changed.mean()) if changed is not None else None))
                    update = captured["pooling_update"]
                    residuals.append(dict(design_id=design,location_id=location,
                        film_token_rms=rms(captured["film_tokens"]),mesh_update_rms=rms(update)))
                    reconstructed = (decode_pooled(model, tokens + torch.as_tensor(update, device=args.device),
                                                    normalized_times)*scale+mean).cpu().numpy()
                    decoder_replay_errors.append(float(np.max(np.abs(reconstructed-pred))))
                    # Frozen interventions are diagnostics, not retrained ablations.
                    for name in ("zero_mesh_update", "remove_pooling_residual", "exclude_headform"):
                        if name == "zero_mesh_update":
                            pooled = tokens
                        elif name == "remove_pooling_residual":
                            pooled = torch.as_tensor(update,device=args.device)
                        else:
                            changed_bias = bias.clone()
                            changed_bias[:, :model.impactor_nodes] = -torch.inf
                            u = F.scaled_dot_product_attention(query[None],key[None],value[None],
                                attn_mask=changed_bias[None,None],dropout_p=0)
                            u = u[0].transpose(0,1).reshape(model.num_latents,model.width)
                            pooled = tokens + model.mesh_output(u)
                        alternate = (decode_pooled(model,pooled,normalized_times)*scale+mean).cpu().numpy()
                        for t, y, p in zip(predictor.time_points, truth, alternate):
                            alternate_curves.append(dict(run_number=run, design_id=design, location_id=location,
                                split=split, intervention=name, time=float(t), acceleration_true_g=float(y),
                                acceleration_pred_g=float(p)))
                        interventions.append(dict(design_id=design,location_id=location,split=split,intervention=name,
                            prediction_change_rms_g=rms(alternate-pred),original_rmse_g=rms(pred-truth),
                            intervention_rmse_g=rms(alternate-truth)))
                print(f"location={location} design={design} ({split}) RMSE={rms(pred-truth):.3f}g {time.monotonic()-start:.1f}s",flush=True)
            for filename, rows in (("probe_acceleration_histories.csv",curves),("activation_differences.csv",traces),
                                   ("pooling_interventions.csv",interventions),("pooling_attention_mass.csv",masses),
                                   ("pooling_residual_sizes.csv",residuals),
                                   ("intervention_acceleration_histories.csv",alternate_curves)):
                pd.DataFrame(rows).to_csv(out/filename,index=False)
        metadata = dict(run_dir=str(args.run_dir.resolve()),locations=args.locations,designs=args.designs,
            checkpoint_sha256=hashlib.sha256((args.run_dir/"hood_impact_best_model.pt").read_bytes()).hexdigest(),
            max_saved_prediction_replay_error_g=max(replay_errors) if replay_errors else None,
            max_decoder_reconstruction_error_g=max(decoder_replay_errors) if decoder_replay_errors else None,
            chunk_size=128,trained_or_modified_checkpoint=False,
            caveats=["Selected locations/designs are descriptive probes, not the whole split.",
                     "Frozen interventions can create representations outside the training distribution.",
                     "Attention on changed nodes omits messages already propagated to other nodes."])
        (out/"probe_metadata.json").write_text(json.dumps(metadata,indent=2))
        if replay_errors and max(replay_errors) > .05:
            raise RuntimeError(f"Saved prediction replay mismatch: {max(replay_errors)} g")
        if decoder_replay_errors and max(decoder_replay_errors) > .05:
            raise RuntimeError(f"Intervention decoder reconstruction mismatch: {max(decoder_replay_errors)} g")
    finally:
        model._mesh_pooling_inputs = original
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    main()
