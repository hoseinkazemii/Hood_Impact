"""Linear probe: is the near-clone difference still decodable inside a trained surrogate?

The earlier forgetting probe measured how far apart near-clone latents are
(0.54% after pooling). Small is not the same as absent: a readout could still
decode the difference from a small but consistent latent shift. This script
answers decodability directly.

For a frozen MeshImpactHistoryNet run it records, for every design and impact
location, three internal states with forward hooks:

    pooled   mesh -> token pooling output (num_latents x width), before latent mixing
    memory   after the latent blocks and memory_norm: what the decoder reads
    decoder  the final per-time decoder states, input of the acceleration head

Then, per stage, a ridge probe maps the latent difference z_b - z_a of a
near-clone pair (PCA-compressed, fitted on training pairs only) to the true
acceleration difference a_b - a_a. There is no intercept: the probe is
antisymmetric in (a, b) by construction. Scored with the approved difference R2
under the same protocols as oracle_delta_features.py (in-sample, unseen edit by
leaving one training design out, and the validation/test pairs), next to the
model's own prediction difference.

Reading it:
    probe R2 >> model R2 at "memory"   the decoder ignores information it has:
                                       an objective/readout problem
    probe R2 ~ 0 at "pooled"           pooling already destroyed it: a
                                       representation problem
    probe R2 ~ 0 everywhere, even      the latent shift does not encode the
    in-sample                          difference at all

Usage:
    python probe_frozen_latents.py                            # default no-shortcut run
    python probe_frozen_latents.py --run-dir runs/mesh_impact_history/<run> --reuse
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import design_difference_data as dd

DEFAULT_RUN = dd.CODE_ROOT / "runs" / "mesh_impact_history" / "20261016_151923_3394500_1704"
DEFAULT_OUT = dd.CODE_ROOT / "runs" / "design_difference" / "latent_probe"
STAGES = ("pooled", "memory", "decoder")
HEADFORM_NODES = 286


def extract_latents(run_dir, data_dir, out_dir, device, log):
    """Run the frozen model on every (design, location); save states and predictions."""
    from infer_designs_mesh_impact_history import build_source_preprocessor
    from train_mesh_impact_history import HistoryPredictor

    predictor = HistoryPredictor.from_run(run_dir, device=device, neighbor_cache_dir=Path(data_dir) / "neighbor_graphs")
    model = predictor.model
    if model.geometry_film != "none" or model.pooling_query_residual:
        raise ValueError("the hooks assume geometry_film='none' and no pooling query residual")
    source, _ = build_source_preprocessor(run_dir, data_dir, out_dir)
    xy = source.load_impact_coords()[["X1", "X2"]].to_numpy(np.float32)
    captured = {}
    hooks = [
        # With the query residual off, the pooled memory is exactly mesh_dropout's output (identity in eval).
        model.mesh_dropout.register_forward_hook(lambda m, i, o: captured.__setitem__("pooled", o.detach())),
        model.memory_norm.register_forward_hook(lambda m, i, o: captured.__setitem__("memory", o.detach()[0])),
        model.acceleration_head.register_forward_pre_hook(
            lambda m, i: captured.__setitem__("decoder", i[0].detach()[0])),
    ]
    states = {stage: None for stage in STAGES}
    prediction = None
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for design in range(dd.NUM_DESIGNS):
                structure = None
                for location in range(1, dd.SAMPLES_PER_DESIGN + 1):
                    number = dd.run_number(design, location)
                    mesh = source.load_mesh_geometry(number)
                    if structure is None:
                        structure = mesh[HEADFORM_NODES:]
                    curve = predictor.predict(np.concatenate((mesh[:HEADFORM_NODES], structure)), xy[number - 1])
                    if prediction is None:
                        prediction = np.zeros((dd.NUM_DESIGNS, dd.SAMPLES_PER_DESIGN, len(curve)), np.float32)
                        for stage in STAGES:
                            shape = tuple(captured[stage].shape)
                            states[stage] = np.zeros((dd.NUM_DESIGNS, dd.SAMPLES_PER_DESIGN) + shape, np.float16)
                    prediction[design, location - 1] = curve
                    for stage in STAGES:
                        states[stage][design, location - 1] = captured[stage].float().cpu().numpy()
                log.info("design %s done (%.0f s)", design, time.perf_counter() - started)
    finally:
        for hook in hooks:
            hook.remove()
    np.save(out_dir / "prediction.npy", prediction)
    for stage in STAGES:
        np.save(out_dir / f"latent_{stage}.npy", states[stage])
    np.save(out_dir / "prediction_times.npy", predictor.time_points)
    return prediction, states, predictor.time_points


def probe_stage(latent, truth, components, log, stage):
    """Ridge probe from PCA(z_b - z_a) to a_b - a_a; returns rows of difference R2."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge

    flat = latent.reshape(dd.NUM_DESIGNS, dd.SAMPLES_PER_DESIGN, -1).astype(np.float32)

    def stack(pairs, sign=1):
        x = np.concatenate([sign * (flat[b] - flat[a]) for a, b in pairs])
        y = np.concatenate([sign * (truth[b] - truth[a]) for a, b in pairs])
        return x, y

    def fit(pairs, alpha):
        x_forward, y_forward = stack(pairs)
        x = np.concatenate((x_forward, -x_forward))
        y = np.concatenate((y_forward, -y_forward))
        pca = PCA(min(components, len(x_forward) - 1), random_state=0).fit(x)
        scores = pca.transform(x)
        scale = scores.std(axis=0) + 1e-12
        ridge = Ridge(alpha=alpha, fit_intercept=False).fit(scores / scale, y)
        return lambda x_new: ridge.predict(pca.transform(x_new) / scale)

    def unseen_edit(alpha):
        predicted, actual = [], []
        for held in dd.TRAIN_DESIGNS:
            predictor = fit([p for p in dd.TRAIN_PAIRS if held not in p], alpha)
            x, y = stack([p for p in dd.TRAIN_PAIRS if held in p])
            predicted.append(predictor(x))
            actual.append(y)
        return dd.difference_r2(np.concatenate(predicted), np.concatenate(actual))

    alphas = np.logspace(-2, 6, 9)
    cv = [unseen_edit(alpha) for alpha in alphas]
    alpha = float(alphas[int(np.argmax(cv))])
    rows = [{"stage": stage, "protocol": "unseen_edit", "alpha": alpha, "difference_r2": max(cv)}]
    predictor = fit(dd.TRAIN_PAIRS, alpha)
    for protocol, pairs in (("in_sample", dd.TRAIN_PAIRS), ("validation", dd.VALIDATION_PAIRS),
                            ("test", dd.TEST_PAIRS)):
        x, y = stack(pairs)
        rows.append({"stage": stage, "protocol": protocol, "alpha": alpha,
                     "difference_r2": dd.difference_r2(predictor(x), y)})
    log.info("probe %-8s alpha %-8g unseen-edit %.3f | in-sample %.3f | val %.3f | test %.3f", stage, alpha,
             *[r["difference_r2"] for r in rows])
    return rows


def relative_shift(latent):
    """Median ||z_b - z_a|| / ||z_a|| over locations, for near-clone and cross-family pairs."""
    flat = latent.reshape(dd.NUM_DESIGNS, dd.SAMPLES_PER_DESIGN, -1).astype(np.float32)
    norms = np.linalg.norm(flat, axis=2)

    def median(pairs):
        return float(np.median(np.concatenate(
            [np.linalg.norm(flat[b] - flat[a], axis=1) / norms[a] for a, b in pairs])))
    cross = [(a, c) for a in dd.FAMILIES["A"] for c in dd.FAMILIES["C"]]
    return {"near_clone_train": median(dd.TRAIN_PAIRS), "near_clone_test": median(dd.TEST_PAIRS),
            "cross_family_A_C": median(cross)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--data-dir", type=Path, default=dd.DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--components", type=int, default=256, help="PCA components of the latent difference")
    parser.add_argument("--reuse", action="store_true", help="reuse latents saved in --out-dir")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("latent_probe")
    log.setLevel(logging.INFO)
    log.propagate = False
    for handler in (logging.FileHandler(args.out_dir / "latent_probe.log", mode="a"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        log.addHandler(handler)
    log.info("Run %s | device %s", args.run_dir, args.device)

    if args.reuse and all((args.out_dir / f"latent_{s}.npy").is_file() for s in STAGES):
        prediction = np.load(args.out_dir / "prediction.npy")
        states = {s: np.load(args.out_dir / f"latent_{s}.npy") for s in STAGES}
        prediction_times = np.load(args.out_dir / "prediction_times.npy")
    else:
        prediction, states, prediction_times = extract_latents(args.run_dir, args.data_dir, args.out_dir,
                                                               args.device, log)
    times, acceleration = dd.load_histories(args.data_dir)
    times, truth = dd.model_grid(times, acceleration)
    if not np.allclose(times, prediction_times, rtol=0, atol=1e-8):
        raise ValueError("the run's prediction grid differs from the 63-point model grid")

    rows = []
    for protocol, pairs in (("in_sample", dd.TRAIN_PAIRS), ("validation", dd.VALIDATION_PAIRS),
                            ("test", dd.TEST_PAIRS)):
        rows.append({"stage": "model_output", "protocol": protocol, "alpha": np.nan,
                     "difference_r2": dd.difference_r2(dd.pair_differences(prediction, pairs),
                                                       dd.pair_differences(truth, pairs))})
    log.info("model's own prediction difference R2: %s", {r["protocol"]: round(r["difference_r2"], 4) for r in rows})
    shifts = {}
    for stage in STAGES:
        shifts[stage] = relative_shift(states[stage])
        log.info("relative latent shift %-8s %s", stage, {k: round(v, 4) for k, v in shifts[stage].items()})
        rows += probe_stage(states[stage], truth, args.components, log, stage)
    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "latent_probe.csv", index=False)
    pivot = table.pivot_table(index="stage", columns="protocol", values="difference_r2", sort=False)
    log.info("Difference R2 (probe on z_b - z_a vs the model's own output):\n%s", pivot.round(3).to_string())
    dd.write_json(args.out_dir / "latent_probe.json", {"run": str(args.run_dir), "difference_r2": pivot.to_dict("index"),
                                                       "relative_shift": shifts})
    return table


if __name__ == "__main__":
    main()
