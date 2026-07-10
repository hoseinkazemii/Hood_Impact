import os
import wandb
import sys
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')
import pickle
from dataclasses import dataclass
from pathlib import Path

# ============================================================================
# CONFIGURATION for acceleration prediction
# ============================================================================

class Config:
    # Dataset selection: "legacy" (1000 samples, 20 designs x 50)
    #                    "industrylike" (60 designs x 1, new HoodImpact_60_IndustryLike)
    data_format = "industrylike"
    samples_per_design = 50  # overridden to 1 for industrylike in __init__

    # Data paths (legacy dataset)
    mesh_geometry_dir = "./Data/mesh_geometry/"
    doe_path = "./Data/DOE.csv"
    acceleration_dir = "./Data/acceleration_history/"
    hic_path = "./Data/HIC_SAE1000.csv"

    # Data paths (industrylike dataset) -- resolved in __init__
    inp_dir = "./Data/HoodImpact_60_IndustryLike/inp_files/"
    impact_coords_path = "./Data/HoodImpact_60_IndustryLike/ImpactCoords_60.csv"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./runs/{stamp}/"

    prediction_target = "acceleration"  # options: "acceleration", "hic"
    num_samples = 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42
    max_train_time = None   # seconds (None = use full history)
    time_subsample_stride = 16  # Subsample time series by this factor
    local_radius = 0.8  # Radius for local neighborhood in PointNet++

    # Model architecture
    pointnet_input_dim = 3  # Input dimension for PointNet (x,y,z, rel_x, rel_y)
    pointnet_hidden_dims = [64, 128, 128, 128, 256]  # PointNet MLP dimensions
    pointnet_output_dim = 128  # PointNet output feature dimension
    trunk_hidden_dims = [64, 128, 128]  # Trunk network for time encoding
    trunk_output_dim = 128  # Trunk output dimension
    film_condition_dim = 2  # FiLM condition dimension (indentor position)
    film_hidden_dim = 64  # FiLM generator hidden dimension
    film_feature_dim = 128  # FiLM feature dimension (must match trunk_output_dim)
    operator_head_feature_dim = 384  # After combining branch and trunk
    operator_head_hidden_dims = [256, 256]  # After combining branch and trunk
    num_fourier_frequencies = 8  # Number of Fourier features for time encoding
    num_tcn_layers = 5  # Number of TCN layers in trunk network
    geometry_encoder_hidden_dim = 128 # Geometry encoder hidden dimension
    geometry_encoder_output_dim = 128  # Global feature dimension (branch output)
    local_pointnet_input_dim = 5  # (x,y,z, dx,dy)
    local_pointnet_hidden_dims = [64, 128]
    local_pointnet_output_dim = 256
    num_heads = 16
    head_dim = 64
    num_tokens = 128
    num_attn_blocks = 2

    # Training parameters
    batch_size = 8   # lowered for the 45-sample industrylike set (~39k nodes/mesh)
    learning_rate = 3e-4
    weight_decay = 1e-5
    num_epochs = 100

    def __init__(self, **kwargs):
        # Resolve dataset-format defaults first, so explicit kwargs can override them.
        fmt = kwargs.get("data_format", self.data_format)
        if fmt == "industrylike":
            self.num_samples = 60
            self.samples_per_design = 1
            self.acceleration_dir = "./Data/HoodImpact_60_IndustryLike/output_history_acc/"
            self.hic_path = "./Data/HoodImpact_60_IndustryLike/output_scalar_HIC.csv"

        for key, value in kwargs.items():
            setattr(self, key, value)

        os.makedirs(self.output_dir, exist_ok=True)


# ============================================================================
# CONFIGURATION for hic prediction
# ============================================================================

# class Config:
#     # Data paths
#     mesh_geometry_dir = "./Data/mesh_geometry/"
#     doe_path = "./Data/DOE_ball_based.csv"
#     acceleration_dir = "./Data/acceleration_history/"
#     hic_path = "./Data/HIC_SAE1000.csv"
#     stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_dir = f"./runs/{stamp}/"
    
#     prediction_target = "hic"  # options: "acceleration", "hic"
#     num_samples = 1000
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     seed = 42
#     max_train_time = None   # seconds (None = use full history)
#     time_subsample_stride = 16  # Subsample time series by this factor
#     local_radius = 0.8  # Radius for local neighborhood in PointNet++

#     # Model architecture
#     pointnet_input_dim = 3  # Input dimension for PointNet (x,y,z, rel_x, rel_y)
#     pointnet_hidden_dims = [64, 128]  # PointNet MLP dimensions
#     pointnet_output_dim = 256  # PointNet output feature dimension
#     trunk_hidden_dims = [64, 128, 128]  # Trunk network for time encoding
#     trunk_output_dim = 256  # Trunk output dimension
#     film_condition_dim = 2  # FiLM condition dimension (indentor position)
#     film_hidden_dim = 64  # FiLM generator hidden dimension
#     film_feature_dim = 128  # FiLM feature dimension (must match trunk_output_dim)
#     operator_head_feature_dim = 384  # After combining branch and trunk
#     operator_head_hidden_dims = [256]  # After combining branch and trunk
#     num_fourier_frequencies = 8  # Number of Fourier features for time encoding
#     num_tcn_layers = 5  # Number of TCN layers in trunk network
#     geometry_encoder_hidden_dim = 128 # Geometry encoder hidden dimension
#     geometry_encoder_output_dim = 128  # Global feature dimension (branch output)
#     local_pointnet_input_dim = 5  # (x,y,z, dx,dy)
#     local_pointnet_hidden_dims = [64, 128]
#     local_pointnet_output_dim = 256
#     num_heads = 16
#     head_dim = 64
#     num_tokens = 128
#     num_attn_blocks = 2

#     # Training parameters
#     batch_size = 16
#     learning_rate = 3e-4
#     weight_decay = 1e-5
#     num_epochs = 200

#     def __init__(self, **kwargs):
#         for key, value in kwargs.items():
#             setattr(self, key, value)
        
#         os.makedirs(self.output_dir, exist_ok=True)

def log_config(config: Config, logger):
    logger.info("Configuration:")
    logger.info("-" * 60)

    keys = set(vars(config).keys())
    keys |= {
        k for k in dir(config)
        if not k.startswith("_")
        and not callable(getattr(config, k))
    }

    for k in sorted(keys):
        logger.info(f"{k:30s}: {getattr(config, k)}")

    logger.info("-" * 60)

def log_model_architecture(model: nn.Module, logger):
    logger.info("Model Architecture:")
    logger.info("=" * 60)
    for line in str(model).splitlines():
        logger.info(line)
    logger.info("=" * 60)

def setup_logging(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(sys.stdout),

        ],
    )

    logger = logging.getLogger()

    class StreamToLogger:
        def __init__(self, level):
            self.level = level
            self._buffer = ""

        def write(self, message):
            if message.strip():
                self.level(message.rstrip())

        def flush(self):
            return False

        def isatty(self):
            return False

    sys.stdout = StreamToLogger(logger.info)
    sys.stderr = StreamToLogger(logger.error)

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    sys.excepthook = handle_exception

    return logger

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

class DataPreprocessor:  
    def __init__(self, config: Config):
        self.config = config
        self.mesh_scaler = StandardScaler()
        self.indentor_scaler = StandardScaler()
        self.time_scaler = StandardScaler()
        self.accel_scaler = StandardScaler()
        self.hic_scaler = StandardScaler()
        self._fitted = False

    def load_mesh_geometry(self, run_number: int) -> np.ndarray:
        """Load mesh geometry for a given run number.

        Args:
            run_number: Experiment run number (1-indexed)

        Returns:
            coords: (N, 3) array of node coordinates
        """
        if self.config.data_format == "industrylike":
            return self._load_mesh_from_inp(run_number)

        filename = f"HoodImpactor_{run_number}_COORD.csv"
        filepath = os.path.join(self.config.mesh_geometry_dir, filename)

        df = pd.read_csv(filepath)
        coords = df[['X1', 'X2', 'X3']].values.astype(np.float32)

        return coords

    def _load_mesh_from_inp(self, run_number: int) -> np.ndarray:
        """Parse the *NODE block of an Abaqus .inp file into an (N, 3) point cloud.

        The first occurrence of a line equal to ``*NODE`` marks the main mesh
        node block (later ``*NODE OUTPUT`` blocks are ignored because parsing
        stops at the next keyword line beginning with ``*``).
        """
        filepath = os.path.join(self.config.inp_dir, f"HoodImpact_{run_number}.inp")

        coords = []
        recording = False
        with open(filepath, "r") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if not recording:
                    if s.upper() == "*NODE":
                        recording = True
                    continue
                # recording == True
                if s.startswith("*"):
                    break
                parts = s.split(",")
                if len(parts) >= 4:
                    coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

        return np.array(coords, dtype=np.float32)

    def load_impact_coords(self) -> pd.DataFrame:
        """Load impactor positions for the industrylike dataset.

        Returns a DataFrame with columns X1, X2, X3 (one row per design,
        0-indexed by row -> 1-indexed run number = row + 1).
        """
        df = pd.read_csv(self.config.impact_coords_path)
        df.columns = [c.strip() for c in df.columns]
        return df

    def load_doe(self) -> pd.DataFrame:
        """Load Design of Experiments data.
        
        Returns:
            doe_df: DataFrame with Run_Number, Indentor X Position, Indentor Y Position
        """
        doe_df = pd.read_csv(self.config.doe_path)
        return doe_df

    def load_hic(self) -> pd.DataFrame:
        """
        Load HIC values.

        Returns:
            hic_df with columns: Job ID, HIC Value
        """
        df = pd.read_csv(self.config.hic_path)
        # industrylike HIC csv has leading spaces in headers (" HIC Value")
        df.columns = [c.strip() for c in df.columns]
        return df

    def subsample_time_series(
        self,
        time: np.ndarray,
        accel: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Subsample time and acceleration using a fixed stride.

        Args:
            time: (T,) array
            accel: (T,) array

        Returns:
            time_sub: (T',) array
            accel_sub: (T',) array
        """
        stride = self.config.time_subsample_stride

        if stride <= 1:
            return time, accel

        return time[::stride], accel[::stride]

    def truncate_by_time(
        self,
        time: np.ndarray,
        accel: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Truncate time and acceleration arrays up to config.max_train_time.
        """
        if self.config.max_train_time is None:
            return time, accel

        mask = time <= self.config.max_train_time

        # Safety: ensure at least 2 points survive
        if np.sum(mask) < 2:
            return time[:2], accel[:2]

        return time[mask], accel[mask]

    def load_acceleration_history(self, run_number: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load acceleration history for a given run number.
        
        Args:
            run_number: Experiment run number (1-indexed)
            
        Returns:
            time: (T,) array of time points
            acceleration: (T,) array of acceleration values
        """
        if self.config.data_format == "industrylike":
            filename = f"HoodImpact_{run_number}_SAE1000_interp1000.csv"
        else:
            filename = f"HoodImpactor_{run_number}_SAE1000.csv"
        filepath = os.path.join(self.config.acceleration_dir, filename)

        df = pd.read_csv(filepath)
        time = df['Time'].values.astype(np.float32)
        acceleration = df['A(in g)'].values.astype(np.float32)
        time, acceleration = self.truncate_by_time(time, acceleration)
        time, acceleration = self.subsample_time_series(time, acceleration)

        return time, acceleration
    
    def load_all_data(self) -> Dict:
        if self.config.data_format == "industrylike":
            doe_df = None
            impact_df = self.load_impact_coords()
        else:
            doe_df = self.load_doe()
            impact_df = None
        hic_df = self.load_hic() if self.config.prediction_target == "hic" else None

        mesh_geometries = []
        indentor_positions = []
        time_arrays = []
        accelerations = []
        hic_values = []
        valid_run_numbers = []

        for run_number in range(1, self.config.num_samples + 1):
            try:
                coords = self.load_mesh_geometry(run_number)

                if self.config.data_format == "industrylike":
                    # impact coords are 0-indexed by row; drop X3 (drop/height axis),
                    # keep in-plane (X1, X2) to match the 2-D indentor convention.
                    pos_row = impact_df.iloc[run_number - 1]
                    indentor = [pos_row['X1'], pos_row['X2']]
                else:
                    doe_row = doe_df[doe_df['Run_Number'] == run_number].iloc[0]
                    indentor = [doe_row['Indentor X Position'],
                                doe_row['Indentor Y Position']]

                if self.config.prediction_target == "acceleration":
                    time, accel = self.load_acceleration_history(run_number)
                    time_arrays.append(time)
                    accelerations.append(accel)

                elif self.config.prediction_target == "hic":  # HIC regression
                    hic_row = hic_df[hic_df['Job ID'] == run_number].iloc[0]
                    hic_values.append(hic_row['HIC Value'])

                mesh_geometries.append(coords)
                indentor_positions.append(indentor)
                valid_run_numbers.append(run_number)

            except Exception as e:
                print(f"[WARN] Skipping run {run_number}: {e}")
                continue

        data_dict = {
            "mesh_geometries": mesh_geometries,
            "indentor_positions": np.array(indentor_positions, dtype=np.float32),
            "run_numbers": valid_run_numbers,
        }

        if self.config.prediction_target == "acceleration":
            data_dict.update({
                "time_arrays": time_arrays,
                "accelerations": accelerations,
            })
        elif self.config.prediction_target == "hic":
            data_dict["hic_values"] = np.array(hic_values, dtype=np.float32)

        return data_dict

    def fit_scalers(
        self,
        mesh_geometries,
        indentor_positions,
        time_arrays=None,
        accelerations=None,
        hic_values=None,
    ):
        all_mesh = np.concatenate(mesh_geometries, axis=0)  # (sum N_i, 3)
        self.mesh_scaler.fit(all_mesh)
        self.indentor_scaler.fit(indentor_positions)  # (N_train, 2)

        if time_arrays is not None and accelerations is not None:
            all_times = np.concatenate(time_arrays).reshape(-1, 1)
            self.time_scaler.fit(all_times)
            all_accels = np.concatenate(accelerations).reshape(-1, 1)
            self.accel_scaler.fit(all_accels)
        if hic_values is not None:
            all_hic = np.asarray(hic_values, dtype=np.float32).reshape(-1, 1)
            self.hic_scaler.fit(all_hic)
        self._fitted = True

    def save_scalers(self, path: str):
        joblib.dump(
            {
                "mesh": self.mesh_scaler,
                "indentor": self.indentor_scaler,
                "time": self.time_scaler,
                "accel": self.accel_scaler,
                "hic": self.hic_scaler,
            },
            path,
        )

    def load_scalers(self, path: str):
        d = joblib.load(path)
        self.mesh_scaler = d["mesh"]
        self.indentor_scaler = d["indentor"]
        self.time_scaler = d["time"]
        self.accel_scaler = d["accel"]
        self.hic_scaler = d["hic"]
        self._fitted = True

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
    
    def transform_hic(self, hic: np.ndarray) -> np.ndarray:
        return self.hic_scaler.transform(hic.reshape(-1, 1)).squeeze(1)

    def inverse_transform_hic(self, hic: np.ndarray) -> np.ndarray:
        return self.hic_scaler.inverse_transform(hic.reshape(-1, 1)).squeeze(1)

# ============================================================================
# DATASET
# ============================================================================
class HoodImpactDataset(Dataset):
    def __init__(
        self,
        mesh_geometries,
        indentor_positions,
        preprocessor,
        time_arrays=None,
        accelerations=None,
        hic_values=None,
        run_numbers=None,
    ):
        """
        Args:
            mesh_geometries: List of (N_i, 3) arrays (variable size)
            indentor_positions: (num_samples, 2) array
            time_arrays: List of (T_i,) arrays (variable size)
            accelerations: List of (T_i,) arrays (variable size)
            preprocessor: DataPreprocessor with normalization stats
            hic_values: List of (T_i,) arrays (variable size)
            run_numbers: Original 1-indexed simulation run numbers
        """
        self.mesh_geometries = mesh_geometries
        self.indentor_positions = indentor_positions
        self.time_arrays = time_arrays
        self.accelerations = accelerations
        self.hic_values = hic_values
        self.run_numbers = run_numbers
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.mesh_geometries)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        mesh = self.mesh_geometries[idx]
        indentor = self.indentor_positions[idx]
        hic = self.hic_values[idx] if self.hic_values is not None else None

        mesh_norm = self.preprocessor.transform_mesh(mesh)
        indentor_norm = self.preprocessor.transform_indentor(indentor)
        hic_norm = self.preprocessor.transform_hic(np.array([hic]))[0] if hic is not None else None
        if self.preprocessor.config.prediction_target == "hic":
            return {
                "mesh": torch.from_numpy(mesh_norm),          # (N, 3)
                "indentor": torch.from_numpy(indentor_norm),  # (2,)
                "hic": torch.tensor(hic_norm, dtype=torch.float32),  # scalar
            }

        # acceleration mode
        time = self.time_arrays[idx]
        accel = self.accelerations[idx]

        time_norm = self.preprocessor.transform_time(time)
        accel_norm = self.preprocessor.transform_acceleration(accel)

        return {
            "mesh": torch.from_numpy(mesh_norm),
            "indentor": torch.from_numpy(indentor_norm),
            "time": torch.from_numpy(time_norm),
            "acceleration": torch.from_numpy(accel_norm),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for variable-size meshes and time series.
    
    Uses flattened representation with batch indices (PyG-style).
    """
    batch_size = len(batch)
    
    # Flatten meshes with batch indices
    mesh_list = []
    mesh_batch_indices = []
    for i, item in enumerate(batch):
        mesh_list.append(item['mesh'])
        mesh_batch_indices.append(torch.full((item['mesh'].shape[0],), i, dtype=torch.long))
    
    meshes = torch.cat(mesh_list, dim=0)  # (total_nodes, 3)
    mesh_batch = torch.cat(mesh_batch_indices, dim=0)  # (total_nodes,)
    
    # Flatten time points with batch indices
    time_list = []
    time_batch_indices = []
    for i, item in enumerate(batch):
        time_list.append(item['time'])
        time_batch_indices.append(torch.full((item['time'].shape[0],), i, dtype=torch.long))
    
    times = torch.cat(time_list, dim=0)  # (total_time_points,)
    time_batch = torch.cat(time_batch_indices, dim=0)  # (total_time_points,)
    
    # Flatten accelerations (targets)
    accel_list = [item['acceleration'] for item in batch]
    accelerations = torch.cat(accel_list, dim=0)  # (total_time_points,)
    
    # Stack indentor positions (fixed size)
    indentors = torch.stack([item['indentor'] for item in batch])  # (batch_size, 2)
    
    return {
        'mesh': meshes,  # (total_nodes, 3)
        'mesh_batch': mesh_batch,  # (total_nodes,)
        'indentor': indentors,  # (batch_size, 2)
        'time': times,  # (total_time_points,)
        'time_batch': time_batch,  # (total_time_points,)
        'acceleration': accelerations,  # (total_time_points,)
        'batch_size': batch_size,
    }

def collate_fn_hic(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)

    meshes, mesh_batch = [], []
    indentors, hics = [], []

    for i, item in enumerate(batch):
        meshes.append(item["mesh"])
        mesh_batch.append(torch.full((item["mesh"].shape[0],), i, dtype=torch.long))
        indentors.append(item["indentor"])
        hics.append(item["hic"])

    return {
        "mesh": torch.cat(meshes, dim=0),          # (sum N, 3)
        "mesh_batch": torch.cat(mesh_batch, dim=0),
        "indentor": torch.stack(indentors),         # (B, 2)
        "hic": torch.stack(hics),                   # (B,)
        "batch_size": batch_size,
    }

# ============================================================================
# Utility functions for batched processing
# ============================================================================

def _split_by_batch(x: torch.Tensor, batch: torch.Tensor, batch_size: int):
    """Split a flattened (total_nodes, D) tensor into a list of per-sample tensors."""
    xs = []
    lengths = []
    for b in range(batch_size):
        xb = x[batch == b]
        xs.append(xb)
        lengths.append(xb.shape[0])
    return xs, lengths


def _pad_to_max(x_list, max_len: int, device, dtype):
    """Pads a list of (N_b, D) to a single (B, max_len, D) and a mask (B, max_len)."""
    B = len(x_list)
    D = x_list[0].shape[-1] if B > 0 else 0
    x_pad = torch.zeros(B, max_len, D, device=device, dtype=dtype)
    mask = torch.zeros(B, max_len, device=device, dtype=torch.bool)
    for b, xb in enumerate(x_list):
        n = xb.shape[0]
        x_pad[b, :n] = xb
        mask[b, :n] = True
    return x_pad, mask


# ============================================================================
# TRAINING
# ============================================================================
class Trainer:    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config,
        preprocessor: DataPreprocessor
    ):
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.preprocessor = preprocessor
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=config.learning_rate * 0.01
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            self.optimizer.zero_grad()

            mesh = batch["mesh"].to(self.config.device)
            mesh_batch = batch["mesh_batch"].to(self.config.device)
            indentor = batch["indentor"].to(self.config.device)
            batch_size = batch["batch_size"]

            if self.config.prediction_target == "hic":
                # HIC: scalar regression
                target = batch["hic"].to(self.config.device)  # (B,)

                pred = self.model(
                    mesh,
                    mesh_batch,
                    indentor,
                    batch_size=batch_size,
                )  # (B,)

            else:
                # Acceleration: time series
                time = batch["time"].to(self.config.device)
                time_batch = batch["time_batch"].to(self.config.device)
                target = batch["acceleration"].to(self.config.device)  # (T_total,)

                pred = self.model(
                    mesh,
                    mesh_batch,
                    indentor,
                    time,
                    time_batch,
                    batch_size=batch_size,
                )  # (T_total,)

            loss = self.criterion(pred, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        self.model.eval()

        total_loss = 0.0
        total_mae = 0.0
        total_points = 0
        num_batches = 0

        for batch in self.val_loader:
            mesh = batch["mesh"].to(self.config.device)
            mesh_batch = batch["mesh_batch"].to(self.config.device)
            indentor = batch["indentor"].to(self.config.device)
            batch_size = batch["batch_size"]

            if self.config.prediction_target == "hic":
                # HIC
                target = batch["hic"].to(self.config.device)  # (B,)

                pred = self.model(
                    mesh,
                    mesh_batch,
                    indentor,
                    batch_size=batch_size,
                )  # (B,)

                loss = self.criterion(pred, target)

                pred_denorm = self.preprocessor.inverse_transform_hic(pred.cpu().numpy())
                target_denorm = self.preprocessor.inverse_transform_hic(target.cpu().numpy())

            else:
                # Acceleration
                time = batch["time"].to(self.config.device)
                time_batch = batch["time_batch"].to(self.config.device)
                target = batch["acceleration"].to(self.config.device)  # (T_total,)

                pred = self.model(
                    mesh,
                    mesh_batch,
                    indentor,
                    time,
                    time_batch,
                    batch_size=batch_size,
                )  # (T_total,)

                loss = self.criterion(pred, target)

                pred_denorm = self.preprocessor.inverse_transform_acceleration(pred.cpu().numpy())
                target_denorm = self.preprocessor.inverse_transform_acceleration(target.cpu().numpy())

            mae = np.sum(np.abs(pred_denorm - target_denorm))

            total_loss += loss.item()
            total_mae += mae
            total_points += len(target_denorm)
            num_batches += 1

        return (
            total_loss / max(num_batches, 1),
            total_mae / max(total_points, 1),
        )

    def train(self) -> Dict:
        """Full training loop with early stopping."""
        print(f"\nTraining on {self.config.device}")
        print(f"{'='*60}")
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_mae = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Track losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Print progress
            unit = "g" if self.config.prediction_target == "acceleration" else ""
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{self.config.num_epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Val MAE: {val_mae:.4f} {unit} | "
                  f"LR: {current_lr:.2e}")
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/mae_g": val_mae,
                "lr": current_lr,
            })

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(os.path.join(self.config.output_dir, 'hood_impact_best_model.pt'))
                print(f"Best model saved at epoch {epoch+1} with val Loss {val_loss:.4f} g")
                wandb.run.summary["best_val_loss"] = self.best_val_loss
                
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
        }

    def save_checkpoint(self, filepath: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
        }, filepath)
    
    def load_checkpoint(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.config.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']


# ============================================================================
# EVALUATION
# ============================================================================
@dataclass
class AttentionData:
    """Container for attention data from a single sample."""
    coords: np.ndarray                  # (N, 3) node coordinates
    slice_assignments: np.ndarray       # (H, N, K) soft assignments
    token_attention: np.ndarray         # (H, K, K) token-to-token attention
    indentor: np.ndarray               # (2,) indentor position
    sample_id: Optional[str] = None    # Optional identifier
    metadata: Optional[Dict] = None    # Optional additional info
    
    def save(self, path: str):
        """Save attention data to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'coords': self.coords,
                'slice_assignments': self.slice_assignments,
                'token_attention': self.token_attention,
                'indentor': self.indentor,
                'sample_id': self.sample_id,
                'metadata': self.metadata
            }, f)
        print(f"Saved attention data to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'AttentionData':
        """Load attention data from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return cls(**data)
    
    @property
    def num_heads(self) -> int:
        return self.slice_assignments.shape[0]
    
    @property
    def num_nodes(self) -> int:
        return self.slice_assignments.shape[1]
    
    @property
    def num_tokens(self) -> int:
        return self.slice_assignments.shape[2]


class AttentionDataCollection:
    """Collection of attention data from multiple samples."""
    
    def __init__(self):
        self.samples: List[AttentionData] = []
    
    def add(self, data: AttentionData):
        self.samples.append(data)
    
    def save(self, path: str):
        """Save all samples to a single file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump([{
                'coords': s.coords,
                'slice_assignments': s.slice_assignments,
                'token_attention': s.token_attention,
                'indentor': s.indentor,
                'sample_id': s.sample_id,
                'metadata': s.metadata
            } for s in self.samples], f)
        print(f"Saved {len(self.samples)} samples to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'AttentionDataCollection':
        """Load collection from file."""
        collection = cls()
        with open(path, 'rb') as f:
            data_list = pickle.load(f)
        for data in data_list:
            collection.add(AttentionData(**data))
        print(f"Loaded {len(collection.samples)} samples from {path}")
        return collection
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> AttentionData:
        return self.samples[idx]
    
class Evaluator:
    """Evaluation utilities for the trained model."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        preprocessor: DataPreprocessor
    ):
        self.model = model.to(config.device)
        self.config = config
        self.preprocessor = preprocessor
        
    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Dict:
        self.model.eval()

        all_preds = []
        all_targets = []

        for batch in test_loader:
            mesh = batch["mesh"].to(self.config.device)
            mesh_batch = batch["mesh_batch"].to(self.config.device)
            indentor = batch["indentor"].to(self.config.device)
            batch_size = batch["batch_size"]

            if self.config.prediction_target == "hic":
                # HIC
                target = batch["hic"].to(self.config.device)

                pred = self.model(
                    mesh,
                    mesh_batch,
                    indentor,
                    batch_size=batch_size,
                )

                pred_denorm = self.preprocessor.inverse_transform_hic(pred.cpu().numpy())
                target_denorm = self.preprocessor.inverse_transform_hic(target.cpu().numpy())

            else:
                # Acceleration
                time = batch["time"].to(self.config.device)
                time_batch = batch["time_batch"].to(self.config.device)
                target = batch["acceleration"].to(self.config.device)

                pred = self.model(
                    mesh,
                    mesh_batch,
                    indentor,
                    time,
                    time_batch,
                    batch_size=batch_size,
                )

                pred_denorm = self.preprocessor.inverse_transform_acceleration(pred.cpu().numpy())
                target_denorm = self.preprocessor.inverse_transform_acceleration(target.cpu().numpy())

            all_preds.append(pred_denorm)
            all_targets.append(target_denorm)

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        mse = np.mean((all_preds - all_targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(all_preds - all_targets))
        r2 = 1.0 - np.sum((all_targets - all_preds) ** 2) / np.sum(
            (all_targets - np.mean(all_targets)) ** 2
        )

        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
        }

        print("\nTest Set Metrics:")
        print(f"  MSE:  {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE:  {mae:.6f}")
        print(f"  R²:   {r2:.4f}")

        return metrics, all_preds, all_targets

    @torch.no_grad()
    def predict_single(
        self,
        mesh_coords: np.ndarray,
        indentor_x: float,
        indentor_y: float,
        time_points: np.ndarray
    ) -> np.ndarray:
        """Make prediction for a single sample.
        
        Args:
            mesh_coords: (N, 3) array of mesh node coordinates
            indentor_x: Indentor X position
            indentor_y: Indentor Y position
            time_points: (T,) array of time points to predict at
            
        Returns:
            acceleration: (T,) predicted acceleration at each time point
        """
        self.model.eval()
        
        # Normalize inputs
        mesh_norm = self.preprocessor.transform_mesh(mesh_coords)
        indentor = np.array([indentor_x, indentor_y], dtype=np.float32)
        indentor_norm = self.preprocessor.transform_indentor(indentor)
        time_norm = self.preprocessor.transform_time(time_points)
        
        # Convert to tensors
        mesh_tensor = torch.from_numpy(mesh_norm).to(self.config.device)
        mesh_batch = torch.zeros(len(mesh_norm), dtype=torch.long, device=self.config.device)
        indentor_tensor = torch.from_numpy(indentor_norm).unsqueeze(0).to(self.config.device)
        time_tensor = torch.from_numpy(time_norm).to(self.config.device)
        time_batch = torch.zeros(len(time_norm), dtype=torch.long, device=self.config.device)
        
        # Forward pass
        pred = self.model(mesh_tensor, mesh_batch, indentor_tensor, time_tensor, time_batch, batch_size=1)
        
        # Denormalize output
        acceleration = self.preprocessor.inverse_transform_acceleration(pred.cpu().numpy())
        
        return acceleration
    
    def evaluate_per_sample(self, test_dataset: HoodImpactDataset) -> List[Dict]:
        """Evaluate each test sample individually."""
        self.model.eval()
        
        results = []
        for idx in range(len(test_dataset)):          
            # Get original (unnormalized) data for this sample
            mesh = test_dataset.mesh_geometries[idx]
            indentor = test_dataset.indentor_positions[idx]
            time = test_dataset.time_arrays[idx]
            target = test_dataset.accelerations[idx]
            
            # Predict
            pred = self.predict_single(mesh, indentor[0], indentor[1], time)
            
            # Compute per-sample metrics
            mse = np.mean((pred - target) ** 2)
            mae = np.mean(np.abs(pred - target))
            
            results.append({
                'time': time,
                'target': target,
                'prediction': pred,
                'mse': mse,
                'mae': mae,
                'run_number': (
                    test_dataset.run_numbers[idx]
                    if test_dataset.run_numbers is not None
                    else None
                ),
            })
        
        return results

    def plot_predictions(
        self,
        results: List[Dict],
        output_dir: str
    ):
        os.makedirs(output_dir, exist_ok=True)

        for i, result in enumerate(results):
            time = result["time"]
            target = result["target"]
            pred = result["prediction"]

            df = pd.DataFrame({
                "time": time,
                "ground_truth": target,
                "prediction": pred,
                "error": pred - target
            })

            run_number = result.get("run_number")
            if run_number is not None:
                df.insert(0, "design_id", run_to_design_id(run_number))
                df.insert(0, "run_number", run_number)

            csv_path = os.path.join(output_dir, f"test_sample_{i:02d}.csv")
            df.to_csv(csv_path, index=False)

            plt.figure(figsize=(10, 4))
            plt.plot(time, target, label="Ground Truth", linewidth=2)
            plt.plot(time, pred, "--", label="Prediction", linewidth=2)
            plt.xlabel("Time")
            plt.ylabel("Acceleration (g)")
            plt.title(f"Test Sample {i:02d} | MAE = {result['mae']:.4f} g")
            plt.legend()
            plt.grid(alpha=0.3)

            fig_path = os.path.join(output_dir, f"test_sample_{i:02d}.png")
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close()

        print(f"Exported {len(results)} test-sample plots + CSVs to {output_dir}")

    def plot_training_history(
        self, 
        train_losses: List[float], 
        val_losses: List[float],
        save_path: Optional[str] = None
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('Training History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history plot to {save_path}")

def export_hic_test_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    data_dict: Dict,
    config,
    test_design_ids: List[int],
    samples_per_design: int,
    filename: str = "hic_test_predictions.csv",
):
    assert config.prediction_target == "hic", \
        "export_hic_test_predictions should only be used in HIC mode"
    run_numbers = data_dict["run_numbers"]
    test_run_numbers = [
        run for run in run_numbers
        if run_to_design_id(run, samples_per_design) in test_design_ids
    ]
    assert len(test_run_numbers) == len(predictions) == len(targets), (
        f"Mismatch: {len(test_run_numbers)=}, "
        f"{len(predictions)=}, {len(targets)=}"
    )
    df = pd.DataFrame({
        "run_number": test_run_numbers,
        "hic_ground_truth": targets,
        "hic_prediction": predictions,
    })
    df["error"] = df["hic_prediction"] - df["hic_ground_truth"]
    df["abs_error"] = df["error"].abs()
    output_path = os.path.join(config.output_dir, filename)
    df.to_csv(output_path, index=False)
    print(f"[INFO] HIC test predictions exported to: {output_path}")

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def run_to_design_id(run_number: int, samples_per_design: int = 50) -> int:
    """
    Maps a 1-indexed run number to a 0-indexed design ID.
    Example:
        runs 1-50   -> design 0
        runs 51-100 -> design 1
        ...
    """
    return (run_number - 1) // samples_per_design

def create_data_loaders(
    data_dict: Dict,
    preprocessor: DataPreprocessor,
    config: Config,
    test_design_ids: List[int] = [2],   # e.g. runs 101-150
    val_design_id: int = 10,   # e.g. runs 501-550
    samples_per_design: int = 50,
) -> Tuple[DataLoader, DataLoader, DataLoader, HoodImpactDataset]:
    """
    Create train/val/test loaders using DESIGN-WISE splitting.
    """

    mesh_geometries = data_dict['mesh_geometries']
    indentor_positions = data_dict['indentor_positions']
    time_arrays = data_dict.get('time_arrays', None)
    accelerations = data_dict.get('accelerations', None)
    run_numbers = data_dict['run_numbers']

    # number of runs per design comes from the config (1 for industrylike)
    samples_per_design = config.samples_per_design

    # val_design_id may be a single int or an iterable of design ids
    val_design_ids = {val_design_id} if isinstance(val_design_id, int) else set(val_design_id)

    train_idx, val_idx, test_idx = [], [], []

    for i, run_number in enumerate(run_numbers):
        design_id = run_to_design_id(run_number, samples_per_design)

        if design_id in test_design_ids:
            test_idx.append(i)
        elif design_id in val_design_ids:
            val_idx.append(i)
        else:
            train_idx.append(i)

    print("\nDesign-wise data split")
    print("=" * 40)
    print(f"Test design IDs: {test_design_ids}  ({len(test_idx)} samples)")
    print(f"Val  design IDs: {sorted(val_design_ids)}   ({len(val_idx)} samples)")
    print(f"Train designs: remaining           ({len(train_idx)} samples)")

    collate = collate_fn_hic if config.prediction_target == "hic" else collate_fn

    if config.prediction_target == "hic":
        train_dataset = HoodImpactDataset(
            mesh_geometries=[mesh_geometries[i] for i in train_idx],
            indentor_positions=indentor_positions[train_idx],
            hic_values=[data_dict["hic_values"][i] for i in train_idx],
            run_numbers=[run_numbers[i] for i in train_idx],
            preprocessor=preprocessor,
        )

        val_dataset = HoodImpactDataset(
            mesh_geometries=[mesh_geometries[i] for i in val_idx],
            indentor_positions=indentor_positions[val_idx],
            hic_values=[data_dict["hic_values"][i] for i in val_idx],
            run_numbers=[run_numbers[i] for i in val_idx],
            preprocessor=preprocessor,
        )

        test_dataset = HoodImpactDataset(
            mesh_geometries=[mesh_geometries[i] for i in test_idx],
            indentor_positions=indentor_positions[test_idx],
            hic_values=[data_dict["hic_values"][i] for i in test_idx],
            run_numbers=[run_numbers[i] for i in test_idx],
            preprocessor=preprocessor,
        )

    elif config.prediction_target == "acceleration":
        train_dataset = HoodImpactDataset(
            mesh_geometries=[mesh_geometries[i] for i in train_idx],
            indentor_positions=indentor_positions[train_idx],
            time_arrays=[time_arrays[i] for i in train_idx],
            accelerations=[accelerations[i] for i in train_idx],
            run_numbers=[run_numbers[i] for i in train_idx],
            preprocessor=preprocessor,
        )        

        val_dataset = HoodImpactDataset(
            mesh_geometries=[mesh_geometries[i] for i in val_idx],
            indentor_positions=indentor_positions[val_idx],
            time_arrays=[time_arrays[i] for i in val_idx],
            accelerations=[accelerations[i] for i in val_idx],
            run_numbers=[run_numbers[i] for i in val_idx],
            preprocessor=preprocessor,
        )

        test_dataset = HoodImpactDataset(
            mesh_geometries=[mesh_geometries[i] for i in test_idx],
            indentor_positions=indentor_positions[test_idx],
            time_arrays=[time_arrays[i] for i in test_idx],
            accelerations=[accelerations[i] for i in test_idx],
            run_numbers=[run_numbers[i] for i in test_idx],
            preprocessor=preprocessor,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
        pin_memory=(config.device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    return train_loader, val_loader, test_loader, test_dataset



# ============================================================================
# INFERENCE UTILITIES
# ============================================================================

class Predictor:
    """Utility class for inference with trained model."""
    
    def __init__(
        self,
        model_path: str,
        stats_path: str,
        config: Optional[Config] = None
    ):
        from temporal_deeponet import HoodImpactNeuralOperator

        self.config = config or Config()
        
        # Load model
        self.model = HoodImpactNeuralOperator(self.config)
        checkpoint = torch.load(model_path, map_location=self.config.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.config.device)
        self.model.eval()
        
        # Load preprocessor stats
        self.preprocessor = DataPreprocessor(self.config)
        self.preprocessor.load_scalers(stats_path)

    @torch.no_grad()
    def predict(
        self,
        mesh_coords: np.ndarray,
        indentor_x: float,
        indentor_y: float,
        time_points: np.ndarray
    ) -> np.ndarray:
        """Make prediction for a single sample.
        
        Args:
            mesh_coords: (N, 3) array of mesh node coordinates
            indentor_x: Indentor X position
            indentor_y: Indentor Y position
            time_points: (T,) array of exact time points to predict at
            
        Returns:
            acceleration: (T,) predicted acceleration at each time point
        """
        # Normalize inputs
        mesh_norm = self.preprocessor.transform_mesh(mesh_coords.astype(np.float32))
        indentor = np.array([indentor_x, indentor_y], dtype=np.float32)
        indentor_norm = self.preprocessor.transform_indentor(indentor)
        time_norm = self.preprocessor.transform_time(time_points.astype(np.float32))
        
        # Convert to tensors
        mesh_tensor = torch.from_numpy(mesh_norm).to(self.config.device)
        mesh_batch = torch.zeros(len(mesh_norm), dtype=torch.long, device=self.config.device)
        indentor_tensor = torch.from_numpy(indentor_norm).unsqueeze(0).to(self.config.device)
        time_tensor = torch.from_numpy(time_norm).to(self.config.device)
        time_batch = torch.zeros(len(time_norm), dtype=torch.long, device=self.config.device)
        
        # Forward pass
        pred = self.model(mesh_tensor, mesh_batch, indentor_tensor, time_tensor, time_batch, batch_size=1)
        
        # Denormalize output
        acceleration = self.preprocessor.inverse_transform_acceleration(pred.cpu().numpy())

        return acceleration
    
