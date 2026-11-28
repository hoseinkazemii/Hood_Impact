"""Data loading, batching, training and evaluation for the mesh-change model.

``DataPreprocessor`` reads the local-edit dataset (local_edits_data.py) and
holds the train-only scalers. ``collate_fn`` flattens variable-size meshes and
histories with per-row sample indices. ``Trainer`` optimizes the normalized
acceleration MSE with AdamW and cosine annealing, keeping the best-validation
and the latest checkpoint; ``Evaluator`` scores a loader in g.
"""
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import wandb

CHECKPOINT_NAME = "hood_impact_best_model.pt"
LAST_CHECKPOINT_NAME = "hood_impact_last_model.pt"
# The hood inner panel's element set: the only nodes the model reads (no headform,
# outer skin, hinges or fasteners).
INNER_PANEL_ELSET = "Hood_Inner-1-2"


def read_inp_nodes(path):
    """IDs and coordinates of the first ``*NODE`` block of an Abaqus deck, in file order.

    Later ``*NODE OUTPUT`` blocks are ignored because parsing stops at the next
    keyword line.
    """
    ids, coords = [], []
    recording = False
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if not recording:
                recording = s.upper() == "*NODE"
                continue
            if s.startswith("*"):
                break
            parts = s.split(",")
            if len(parts) >= 4:
                ids.append(int(parts[0]))
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(ids, dtype=np.int64), np.array(coords, dtype=np.float32)


def inner_panel_node_ids(path):
    """IDs of the nodes used by the hood inner panel's elements."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "abaqus_scripts"))
    from inp_geom import Deck

    return np.array(sorted(Deck(str(path)).elset_nodes(INNER_PANEL_ELSET)), dtype=np.int64)


@dataclass
class Config:
    """Data paths, preprocessing and optimization settings of one training run."""

    # The local-edit dataset (local_edits_data.py): one design-6 deck, the run
    # manifest and the curves.
    inp_dir: str = "./Data/HoodImpact_LocalEdits/inp_files/"
    impact_coords_path: str = "./Data/HoodImpact_LocalEdits/manifest_local_edits.csv"
    acceleration_dir: str = "./Data/HoodImpact_LocalEdits/output_history_acc/"
    output_dir: str = field(default_factory=lambda: f"./runs/{datetime.now():%Y%m%d_%H%M%S}/")
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    seed: int = 42
    max_train_time: Optional[float] = None  # seconds; None keeps the whole history
    time_subsample_stride: int = 16  # keep every Nth time point
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


class DataPreprocessor:
    """Curve loading with the configured cutoff and stride, and the train-only scalers."""

    def __init__(self, config):
        self.config = config
        self.mesh_scaler = StandardScaler()
        self.indentor_scaler = StandardScaler()
        self.time_scaler = StandardScaler()
        self.accel_scaler = StandardScaler()

    def load_acceleration_history(self, run_number: int) -> Tuple[np.ndarray, np.ndarray]:
        """(time, acceleration in g) of one run, cut at max_train_time and subsampled."""
        path = os.path.join(self.config.acceleration_dir, f"HoodImpact_{run_number}_SAE1000_interp1000.csv")
        frame = pd.read_csv(path)
        time = frame["Time"].values.astype(np.float32)
        acceleration = frame["A(in g)"].values.astype(np.float32)
        if self.config.max_train_time is not None:
            keep = time <= self.config.max_train_time
            # Keep at least two points.
            time, acceleration = (time[keep], acceleration[keep]) if keep.sum() >= 2 else (time[:2], acceleration[:2])
        stride = self.config.time_subsample_stride
        if stride > 1:
            time, acceleration = time[::stride], acceleration[::stride]
        return time, acceleration

    def load_all_data(self) -> Dict:
        """Every edited run of the local-edit dataset (design 6 is left out)."""
        from local_edits_data import load_local_edit_data
        return load_local_edit_data(self)

    def fit_scalers(self, mesh_geometries, indentor_positions, time_arrays, accelerations):
        self.mesh_scaler.fit(np.concatenate(mesh_geometries, axis=0))
        self.indentor_scaler.fit(indentor_positions)
        self.time_scaler.fit(np.concatenate(time_arrays).reshape(-1, 1))
        self.accel_scaler.fit(np.concatenate(accelerations).reshape(-1, 1))

    def save_scalers(self, path: str):
        joblib.dump({"mesh": self.mesh_scaler, "indentor": self.indentor_scaler,
                     "time": self.time_scaler, "accel": self.accel_scaler}, path)

    def load_scalers(self, path: str):
        # Runs before 2026-11-27 also saved an unused "hic" scaler.
        scalers = joblib.load(path)
        self.mesh_scaler = scalers["mesh"]
        self.indentor_scaler = scalers["indentor"]
        self.time_scaler = scalers["time"]
        self.accel_scaler = scalers["accel"]

    def transform_mesh(self, mesh: np.ndarray) -> np.ndarray:
        return self.mesh_scaler.transform(mesh)

    def transform_indentor(self, indentor: np.ndarray) -> np.ndarray:
        return self.indentor_scaler.transform(indentor.reshape(1, -1)).squeeze(0)

    def transform_time(self, time: np.ndarray) -> np.ndarray:
        return self.time_scaler.transform(time.reshape(-1, 1)).squeeze(1)

    def transform_acceleration(self, accel: np.ndarray) -> np.ndarray:
        return self.accel_scaler.transform(accel.reshape(-1, 1)).squeeze(1)

    def inverse_transform_acceleration(self, accel: np.ndarray) -> np.ndarray:
        return self.accel_scaler.inverse_transform(accel.reshape(-1, 1)).squeeze(1)


class HoodImpactDataset(Dataset):
    """Raw runs, normalized on access with the preprocessor's scalers."""

    def __init__(self, mesh_geometries, indentor_positions, time_arrays, accelerations, run_numbers, preprocessor):
        self.mesh_geometries = mesh_geometries
        self.indentor_positions = indentor_positions
        self.time_arrays = time_arrays
        self.accelerations = accelerations
        self.run_numbers = run_numbers
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.mesh_geometries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        p = self.preprocessor
        return {
            "mesh": torch.from_numpy(p.transform_mesh(self.mesh_geometries[idx])),
            "indentor": torch.from_numpy(p.transform_indentor(self.indentor_positions[idx])),
            "time": torch.from_numpy(p.transform_time(self.time_arrays[idx])),
            "acceleration": torch.from_numpy(p.transform_acceleration(self.accelerations[idx])),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Flatten variable-size meshes and histories, with the sample index of every row."""
    def membership(key):
        return torch.cat([torch.full((item[key].shape[0],), i, dtype=torch.long) for i, item in enumerate(batch)])

    return {
        "mesh": torch.cat([item["mesh"] for item in batch]),                  # (total_nodes, 3)
        "mesh_batch": membership("mesh"),                                     # (total_nodes,)
        "indentor": torch.stack([item["indentor"] for item in batch]),        # (batch_size, 2)
        "time": torch.cat([item["time"] for item in batch]),                  # (total_times,)
        "time_batch": membership("time"),                                     # (total_times,)
        "acceleration": torch.cat([item["acceleration"] for item in batch]),  # (total_times,)
        "batch_size": len(batch),
    }


def _forward(model, batch, device):
    """(prediction, target) of one collated batch, both normalized."""
    prediction = model(batch["mesh"].to(device), batch["mesh_batch"].to(device), batch["indentor"].to(device),
                       batch["time"].to(device), batch["time_batch"].to(device), batch_size=batch["batch_size"])
    return prediction, batch["acceleration"].to(device)


class Trainer:
    """AdamW + cosine annealing on the normalized acceleration MSE; best and last checkpoints."""

    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, config: Config,
                 preprocessor: DataPreprocessor):
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.preprocessor = preprocessor
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                           weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs, eta_min=config.learning_rate * 0.01)
        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")

    def train_epoch(self) -> float:
        self.model.train()
        total_loss, num_batches = 0.0, 0
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            prediction, target = _forward(self.model, batch, self.config.device)
            loss = self.criterion(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validation loss (normalized MSE, mean over batches) and MAE in g."""
        self.model.eval()
        total_loss, total_mae, total_points, num_batches = 0.0, 0.0, 0, 0
        for batch in self.val_loader:
            prediction, target = _forward(self.model, batch, self.config.device)
            loss = self.criterion(prediction, target)
            prediction_g = self.preprocessor.inverse_transform_acceleration(prediction.cpu().numpy())
            target_g = self.preprocessor.inverse_transform_acceleration(target.cpu().numpy())
            total_loss += loss.item()
            total_mae += np.sum(np.abs(prediction_g - target_g))
            total_points += len(target_g)
            num_batches += 1
        return total_loss / max(num_batches, 1), total_mae / max(total_points, 1)

    def train(self) -> Dict:
        """Train up to config.num_epochs, continuing after any completed (resumed) epochs."""
        print(f"\nTraining on {self.config.device}")
        print("=" * 60)
        for epoch in range(len(self.train_losses), self.config.num_epochs):
            train_loss = self.train_epoch()
            val_loss, val_mae = self.validate()
            self.scheduler.step()
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch + 1:3d}/{self.config.num_epochs} | Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | Val MAE: {val_mae:.4f} g | LR: {current_lr:.2e}")
            wandb.log({"epoch": epoch + 1, "train/loss": train_loss, "val/loss": val_loss,
                       "val/mae_g": val_mae, "lr": current_lr})
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(os.path.join(self.config.output_dir, CHECKPOINT_NAME))
                print(f"Best model saved at epoch {epoch + 1} with val Loss {val_loss:.4f}")
                wandb.run.summary["best_val_loss"] = self.best_val_loss
            self.save_checkpoint(os.path.join(self.config.output_dir, LAST_CHECKPOINT_NAME))
        return {"train_losses": self.train_losses, "val_losses": self.val_losses,
                "best_val_loss": self.best_val_loss}

    def checkpoint_state(self):
        numpy_state = np.random.get_state()
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "completed_epochs": len(self.train_losses),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy_rng_state": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            "python_rng_state": random.getstate(),
        }

    def save_checkpoint(self, filepath: str):
        # Replace only after a complete write, preserving the previous epoch
        # if Slurm terminates the process while serializing the next one.
        temporary = str(filepath) + ".tmp"
        try:
            torch.save(self.checkpoint_state(), temporary)
            os.replace(temporary, filepath)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)

    def load_checkpoint(self, filepath: str):
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=True)
        completed = len(checkpoint["train_losses"])
        if len(checkpoint["val_losses"]) != completed or checkpoint.get("completed_epochs", completed) != completed:
            raise ValueError("Checkpoint epoch count and loss histories disagree")
        if checkpoint["scheduler_state_dict"]["T_max"] != self.config.num_epochs:
            raise ValueError("Resume must retain the original total epochs and cosine schedule")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.best_val_loss = checkpoint["best_val_loss"]
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
            if torch.cuda.is_available() and checkpoint.get("cuda_rng_states"):
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
            numpy_state = checkpoint["numpy_rng_state"]
            np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
            random.setstate(checkpoint["python_rng_state"])
        return checkpoint


class Evaluator:
    """Predictions of a trained model over a loader, in g."""

    def __init__(self, model: nn.Module, config: Config, preprocessor: DataPreprocessor):
        self.model = model.to(config.device)
        self.config = config
        self.preprocessor = preprocessor

    @torch.no_grad()
    def evaluate(self, loader: DataLoader):
        """(metrics, predictions, targets): MSE/RMSE/MAE/R2 in g over all points, flat arrays in loader order."""
        self.model.eval()
        predictions, targets = [], []
        for batch in loader:
            prediction, target = _forward(self.model, batch, self.config.device)
            predictions.append(self.preprocessor.inverse_transform_acceleration(prediction.cpu().numpy()))
            targets.append(self.preprocessor.inverse_transform_acceleration(target.cpu().numpy()))
        predictions, targets = np.concatenate(predictions), np.concatenate(targets)
        mse = np.mean((predictions - targets) ** 2)
        r2 = 1.0 - np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2)
        metrics = {"mse": float(mse), "rmse": float(np.sqrt(mse)),
                   "mae": float(np.mean(np.abs(predictions - targets))), "r2": float(r2)}
        print(f"\nTest Set Metrics:\n  MSE:  {metrics['mse']:.6f}\n  RMSE: {metrics['rmse']:.6f}\n"
              f"  MAE:  {metrics['mae']:.6f}\n  R²:   {metrics['r2']:.4f}")
        return metrics, predictions, targets


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def create_data_loaders(data: Dict, preprocessor: DataPreprocessor, config: Config, splits: Dict):
    """Train (shuffled), validation and test loaders, plus the test dataset, from ``splits``.

    ``splits`` maps "train", "validation" and "test" to {"run_numbers": [...]}
    (train_mesh_impact_history.resolve_splits).
    """
    index_of = {run: i for i, run in enumerate(data["run_numbers"])}

    def dataset(name):
        rows = [index_of[run] for run in splits[name]["run_numbers"]]
        return HoodImpactDataset(
            mesh_geometries=[data["mesh_geometries"][i] for i in rows],
            indentor_positions=data["indentor_positions"][rows],
            time_arrays=[data["time_arrays"][i] for i in rows],
            accelerations=[data["accelerations"][i] for i in rows],
            run_numbers=[data["run_numbers"][i] for i in rows],
            preprocessor=preprocessor,
        )

    train_dataset, val_dataset, test_dataset = dataset("train"), dataset("validation"), dataset("test")
    print(f"\nDesign-wise split: train {len(train_dataset)} | validation {len(val_dataset)} | "
          f"test {len(test_dataset)} runs")
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn,
                              pin_memory=(config.device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader, test_dataset
