"""Run the real MSE training pipeline on a clearly labelled reduced-data demo.

Uses one HIC-qualified location across 12 real designs and 512 structural nodes
per mesh plus the 286 headform nodes. The production model architecture is kept.
Only the data loader and shape instrumentation are adapted inside this process.
This is an execution demonstration, not an accuracy experiment or Slurm job.
"""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import train_mesh_impact_history as training
from hic_location_filter import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("Data/HoodImpact_1704_EuroNCAP"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--threshold", type=float, default=10)
    args = parser.parse_args()
    output = args.output_dir or Path("runs/pipeline_demo") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    table, selection = analyze(args.data_root / "output_history_acc", args.threshold)
    if not selection["selected_locations"]:
        raise ValueError("No qualifying location for this demonstration")
    location = selection["selected_locations"][0]
    table.to_csv(output / "full_dataset_hic_variation.csv", index=False)
    training.write_json(output / "full_dataset_selection.json", selection)
    print(f"Full-data filter: {len(selection['selected_locations'])}/142 locations qualify.", flush=True)
    print(f"DEMO ONLY: location {location}, 12 designs, 512 structural + 286 headform nodes.", flush=True)

    original_preprocessor = training.DataPreprocessor
    original_model = training.MeshImpactHistoryNet
    shapes, loaded = {}, {}

    class DemoPreprocessor(original_preprocessor):
        def load_all_data(self):
            data = {key: [] for key in ("run_numbers", "mesh_geometries", "indentor_positions",
                                       "time_arrays", "accelerations")}
            impacts = self.load_impact_coords()
            for design in range(12):
                run = design * 142 + location
                full_mesh = self.load_mesh_geometry(run)
                indices = np.concatenate((np.arange(286),
                                          np.linspace(286, len(full_mesh) - 1, 512, dtype=int)))
                mesh = full_mesh[indices]
                times, acceleration = self.load_acceleration_history(run)
                data["run_numbers"].append(run)
                data["mesh_geometries"].append(mesh)
                data["indentor_positions"].append(impacts.iloc[run - 1][["X1", "X2"]].to_numpy(float))
                data["time_arrays"].append(times)
                data["accelerations"].append(acceleration)
                loaded[str(run)] = {"original_nodes": len(full_mesh), "demo_nodes": len(mesh),
                                    "sampled_times": len(times)}
            data["indentor_positions"] = np.asarray(data["indentor_positions"], dtype=np.float32)
            return data

    class TracedModel(original_model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            def hook(name):
                def record(module, inputs, result):
                    if name not in shapes:
                        shapes[name] = {"input": list(inputs[0].shape), "output": list(result.shape)}
                        print(f"SHAPE {name}: {shapes[name]}", flush=True)
                return record
            for name, module in self.named_modules():
                if name in ("node_embedding", "impact_embedding", "memory_norm", "time_embedding",
                            "acceleration_head") or name in {
                                "neighborhood_blocks.0", "neighborhood_blocks.1",
                                "latent_blocks.0", "latent_blocks.1", "latent_blocks.2",
                                "decoder_blocks.0", "decoder_blocks.1"}:
                    module.register_forward_hook(hook(name))

    training.DataPreprocessor, training.MeshImpactHistoryNet = DemoPreprocessor, TracedModel
    try:
        metrics = training.main([
            "--data-format", "euroncap1704", "--num-samples", "12", "--samples-per-design", "142",
            "--inp-dir", str(args.data_root / "inp_files"),
            "--impact-coords-path", str(args.data_root / "ImpactCoords_1704.csv"),
            "--acceleration-dir", str(args.data_root / "output_history_acc"),
            "--neighborhood-layers", "2", "--neighborhood-k", "256", "--impactor-nodes", "286",
            "--epochs", "2", "--batch-size", "3", "--device", "cpu", "--wandb-mode", "disabled",
            "--run-name", "DEMO_reduced_mesh_one_location_MSE", "--output-dir", str(output / "training")])
    finally:
        training.DataPreprocessor, training.MeshImpactHistoryNet = original_preprocessor, original_model

    # Verify the exported predictions with the public inference API and saved scalers.
    predictor = training.HistoryPredictor.from_run(output / "training", device="cpu")
    reader = original_preprocessor(training.Config(data_format="euroncap1704",
                                   inp_dir=str(args.data_root / "inp_files"),
                                   impact_coords_path=str(args.data_root / "ImpactCoords_1704.csv")))
    run = 4 * 142 + location
    mesh = reader.load_mesh_geometry(run)
    indices = np.concatenate((np.arange(286), np.linspace(286, len(mesh) - 1, 512, dtype=int)))
    impact = reader.load_impact_coords().iloc[run - 1][["X1", "X2"]].to_numpy(float)
    prediction = predictor.predict(mesh[indices], impact)
    import pandas as pd
    exported = pd.read_csv(output / "training/test_acceleration_histories.csv")
    expected = exported.loc[exported.run_number == run, "acceleration_pred_g"].to_numpy()
    np.testing.assert_allclose(prediction, expected, rtol=1e-5, atol=1e-4)
    report = {"demonstration_only": True, "location": location, "source_runs": loaded,
              "shapes_first_forward": shapes, "model_parameters": sum(p.numel() for p in predictor.model.parameters()),
              "training_split_sizes": {"train": 9, "validation": 1, "test": 2},
              "epochs": 2, "batch_size": 3, "optimizer_steps": 6,
              "inference_reload_max_abs_error_g": float(np.max(np.abs(prediction - expected))),
              "test_metrics_not_an_accuracy_result": metrics}
    training.write_json(output / "demonstration.json", report)
    print(f"DEMO complete; strict checkpoint reload reproduces exported predictions. Artifacts: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
