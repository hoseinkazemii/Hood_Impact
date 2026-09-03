import os
import argparse
import wandb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import warnings
warnings.filterwarnings('ignore')
from utils.utils import log_config, set_seed, setup_logging, Config, DataPreprocessor, create_data_loaders, Trainer, Evaluator, log_model_architecture, export_hic_test_predictions, HIC_DIRECT_BEST_ARCH


# MODEL ARCHITECTURE
def scatter_max(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Scatter max operation (like PyG's scatter_max but simplified).
    
    Args:
        src: (N, D) source tensor
        index: (N,) batch indices
        dim_size: Number of unique indices (batch size)
        
    Returns:
        out: (dim_size, D) max-pooled tensor
    """
    D = src.shape[1]
    out = torch.full((dim_size, D), float('-inf'), dtype=src.dtype, device=src.device)
    index_expanded = index.unsqueeze(1).expand(-1, D)
    out = out.scatter_reduce(0, index_expanded, src, reduce='amax', include_self=False)
    # Replace -inf with 0 for empty batches (shouldn't happen in practice)
    out = torch.where(torch.isinf(out), torch.zeros_like(out), out)
    return out


class FourierFeatures(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        num_frequencies: int = 8,
        scale: float = 0.1,
        learnable: bool = False,
        debug: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.output_dim = input_dim + 2 * input_dim * num_frequencies
        self.debug = debug
        self._printed = False  # print only once

        if learnable:
            self.frequencies = nn.Parameter(torch.randn(num_frequencies, input_dim) * scale)
        else:
            freqs = 2.0 ** torch.arange(0, num_frequencies)
            freqs = freqs.unsqueeze(1) * scale  # (num_frequencies, 1)
            self.register_buffer("frequencies", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., input_dim)
        phase = 2 * math.pi * torch.matmul(x, self.frequencies.T)

        sin_feat = torch.sin(phase)
        cos_feat = torch.cos(phase)

        if self.debug and not self._printed:
            self._printed = True

            with torch.no_grad():
                print("\n[FourierFeatures DEBUG]")
                print(f"  x.shape: {tuple(x.shape)}")
                print(f"  x.min / max: {x.min().item():.4e} / {x.max().item():.4e}")
                print(f"  x.mean / std: {x.mean().item():.4e} / {x.std().item():.4e}")

                print(f"  frequencies.min / max: "
                      f"{self.frequencies.min().item():.4e} / "
                      f"{self.frequencies.max().item():.4e}")

                print(f"  phase.min / max: "
                      f"{phase.min().item():.4e} / {phase.max().item():.4e}")
                print(f"  phase.mean / std: "
                      f"{phase.mean().item():.4e} / {phase.std().item():.4e}")

                print(f"  sin.mean / std: "
                      f"{sin_feat.mean().item():.4e} / {sin_feat.std().item():.4e}")
                print(f"  cos.mean / std: "
                      f"{cos_feat.mean().item():.4e} / {cos_feat.std().item():.4e}")

                print("  First 5 frequencies:", self.frequencies[:5, 0].tolist())
                print("  Last 5 frequencies:", self.frequencies[-5:, 0].tolist())
                print("[End FourierFeatures DEBUG]\n")

        return torch.cat([x, sin_feat, cos_feat], dim=-1)


class TemporalConvBlock(nn.Module):
    """Single temporal convolution block with dilation and residual connection.""" 
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Causal padding: pad only on the left
        self.padding = ((kernel_size - 1) * dilation) // 2
        
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding
        )
        self.norm = nn.GroupNorm(1, out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time) 
        Returns:
            out: (batch, channels, time)
        """
        # Remove future padding (causal)
        out = self.conv(x)
        out = self.norm(out)
        out = F.gelu(out)
        out = self.dropout(out)
        
        return out + self.residual(x)


class TemporalConvStack(nn.Module):
    """Stack of dilated convolutions with exponentially increasing dilation."""
    
    def __init__(
        self,
        channels: int,
        num_tcn_layers: int,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = []
        for i in range(num_tcn_layers):
            dilation = 2 ** i
            layers.append(TemporalConvBlock(
                channels, channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout
            ))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(
        self,
        x: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        Args:
            x: (total_time_points, channels)
            batch_indices: (total_time_points,)
            batch_size: int
            
        Returns:
            out: (total_time_points, channels)
        """
        # Separate by batch, pad to same length, apply convs, then unpad
        max_len = 0
        sequences = []
        lengths = []
        
        for b in range(batch_size):
            mask = batch_indices == b
            seq = x[mask]  # (T_b, channels)
            sequences.append(seq)
            lengths.append(seq.shape[0])
            max_len = max(max_len, seq.shape[0])
        
        # Pad and stack: (batch, channels, max_len)
        padded = torch.zeros(batch_size, x.shape[1], max_len, device=x.device, dtype=x.dtype)
        for b, seq in enumerate(sequences):
            padded[b, :, :lengths[b]] = seq.T
        
        # Apply conv layers
        out = padded
        for layer in self.layers:
            out = layer(out)
        
        # Unpad and flatten back
        result = torch.zeros_like(x)
        for b in range(batch_size):
            mask = batch_indices == b
            result[mask] = out[b, :, :lengths[b]].T
        
        return result


class TrunkNetwork(nn.Module):
    def __init__(
        self,
        trunk_output_dim: int,
        num_frequencies: int,
        trunk_hidden_dims: List[int],
        num_tcn_layers: int,
        dropout: float = 0.0
    ):
        super().__init__()
                
        # Fourier feature encoding
        self.fourier = FourierFeatures(
            input_dim=1,
            num_frequencies=num_frequencies,
            scale=0.1,
            learnable=False
        )
        
        # Initial MLP
        fourier_dim = self.fourier.output_dim
        dims = [fourier_dim] + trunk_hidden_dims
        
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        
        self.initial_mlp = nn.Sequential(*layers)
        
        embed_dim = trunk_hidden_dims[-1]
        
        # Temporal convolutions
        self.temporal_conv = TemporalConvStack(
            channels=embed_dim,
            num_tcn_layers=num_tcn_layers,
            dropout=dropout
        )
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, trunk_output_dim),
            nn.LayerNorm(trunk_output_dim)
        )
        
        self.output_dim = trunk_output_dim
    
    def forward(
        self,
        time: torch.Tensor,
        time_batch: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        Args:
            time: (total_time_points,) - normalized time values
            time_batch: (total_time_points,) - batch indices
            batch_size: int
            
        Returns:
            features: (total_time_points, trunk_output_dim)
        """
        # Fourier encoding
        t = time.unsqueeze(-1)  # (total_time_points, 1)
        t_fourier = self.fourier(t)  # (total_time_points, fourier_dim)
        
        # Initial MLP
        x = self.initial_mlp(t_fourier)  # (total_time_points, embed_dim)
        
        # Temporal convolutions (local patterns)
        x = self.temporal_conv(x, time_batch, batch_size)
        
        # Final projection
        return self.output_proj(x)


class LocalPointNetEncoder(nn.Module):
    """
    Encodes a local patch of nodes near the indentor.
    """

    def __init__(
        self,
        local_pointnet_input_dim: int,
        local_pointnet_hidden_dims: List[int],
        local_pointnet_output_dim: int
    ):
        super().__init__()

        dims = [local_pointnet_input_dim] + local_pointnet_hidden_dims + [local_pointnet_output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i+1]))
                layers.append(nn.ReLU(inplace=True))

        self.mlp = nn.Sequential(*layers)
        self.output_dim = local_pointnet_output_dim
        
    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        x: (N_local, input_dim)
        batch: (N_local,)
        """
        x = self.mlp(x)
        return scatter_max(x, batch, batch_size)


class PointNetEncoder(nn.Module):
    """PointNet encoder for variable-size point clouds.
    Uses shared MLPs and global max pooling for permutation invariance.
    No padding - uses batch indices for pooling.
    """
    
    def __init__(
        self, 
        pointnet_input_dim: int,
        pointnet_hidden_dims: List[int],
        pointnet_output_dim: int
    ):
        super().__init__()
        
        # Build shared MLP layers
        dims = [pointnet_input_dim] + pointnet_hidden_dims + [pointnet_output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation after last layer before pooling
                layers.append(nn.LayerNorm(dims[i+1]))
                layers.append(nn.ReLU(inplace=True))
        
        self.shared_mlp = nn.Sequential(*layers)
        self.output_dim = pointnet_output_dim
        
    def forward(self, x: torch.Tensor, batch: torch.Tensor, indentor: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Args:
            x: (total_nodes, 3) - All point coordinates concatenated
            batch: (total_nodes,) - Batch index for each point
            batch_size: Number of samples in batch
            
        Returns:
            global_features: (batch_size, pointnet_output_dim)
        """
        # Expand indentor to nodes
        # indentor_nodes = indentor[batch]           # (N,2)
        # rel_xy = x[:, :2] - indentor_nodes    # (N,2)
        # x = torch.cat([x, rel_xy], dim=1)  # (N,5)
        # Apply shared MLP to all points
        x = self.shared_mlp(x)  # (total_nodes, pointnet_output_dim)
        
        # Global max pooling per sample using scatter
        global_features = scatter_max(x, batch, batch_size)  # (batch_size, pointnet_output_dim)
        
        return global_features


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    Applies a learned, condition-dependent affine transformation:
        y = gamma(condition) * x + beta(condition)
    where gamma and beta are produced by a small MLP from the conditioning input.
    """

    def __init__(
        self,
        film_feature_dim: int,
        film_condition_dim: int,
        film_hidden_dim: int
    ):
        super().__init__()

        # MLP to process conditioning input
        self.mlp = nn.Sequential(
            nn.Linear(film_condition_dim, film_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(film_hidden_dim, film_hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Heads for FiLM parameters
        self.gamma = nn.Linear(film_hidden_dim, film_feature_dim)
        self.beta = nn.Linear(film_hidden_dim, film_feature_dim)

        # Initialize FiLM parameters:
        # gamma ≈ 1, beta ≈ 0  → identity modulation at initialization
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:         (batch_size, film_feature_dim)       - features to be modulated
            condition: (batch_size, film_condition_dim)     - conditioning input
        Returns:
            modulated_x: (batch_size, film_feature_dim)
        """
        h = self.mlp(condition)
        gamma = self.gamma(h)
        beta = self.beta(h)
        return gamma * x + beta


class OutputNetwork(nn.Module):
    """
    Operator head that maps combined branch-trunk features to scalar output.
    """

    def __init__(
        self,
        operator_head_feature_dim: int,
        operator_head_hidden_dims: List[int],
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dims = operator_head_hidden_dims

        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Linear(operator_head_feature_dim, operator_head_hidden_dims[0]),
            nn.LayerNorm(operator_head_hidden_dims[0]),
            nn.GELU(),
        )

        # Residual MLP blocks
        blocks = []
        for dim in operator_head_hidden_dims:
            blocks.append(
                nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.LayerNorm(dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.blocks = nn.ModuleList(blocks)

        # Final output layer
        self.output_layer = nn.Linear(operator_head_hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (total_time_points, operator_head_feature_dim)

        Returns:
            output: (total_time_points,)
        """

        x = self.input_proj(x)

        for block in self.blocks:
            x = x + block(x)

        return self.output_layer(x).squeeze(-1)


class HoodImpactNeuralOperator(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # ============================================================
        # 1. GLOBAL GEOMETRY ENCODER
        # ============================================================
        self.mesh_encoder = PointNetEncoder(
            pointnet_input_dim=config.pointnet_input_dim,
            pointnet_hidden_dims=config.pointnet_hidden_dims,
            pointnet_output_dim=config.pointnet_output_dim,
        )

        # ============================================================
        # 2. LOCAL (INDENTOR-CENTERED) GEOMETRY ENCODER
        # ============================================================
        self.local_encoder = LocalPointNetEncoder(
            local_pointnet_input_dim=config.local_pointnet_input_dim,  # (x,y,z, dx,dy)
            local_pointnet_hidden_dims=config.local_pointnet_hidden_dims,
            local_pointnet_output_dim=config.local_pointnet_output_dim,
        )

        # ============================================================
        # 3. FiLM CONDITIONING (GLOBAL BRANCH ONLY)
        # ============================================================
        self.film_layer1 = FiLMLayer(
            film_feature_dim=config.pointnet_output_dim,
            film_condition_dim=config.film_condition_dim,
            film_hidden_dim=config.film_hidden_dim,
        )

        self.film_layer2 = FiLMLayer(
            film_feature_dim=config.pointnet_output_dim,
            film_condition_dim=config.film_condition_dim,
            film_hidden_dim=config.film_hidden_dim,
        )

        # ============================================================
        # 4. BRANCH FEATURE TRANSFORM
        # ============================================================
        branch_dim = config.pointnet_output_dim + config.local_pointnet_output_dim  # global + local

        self.branch_transform = nn.Sequential(
            nn.Linear(branch_dim, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.GELU(),
        )

        # ============================================================
        # 5. TRUNK (TIME) ENCODER
        # ============================================================
        self.trunk = TrunkNetwork(
            trunk_output_dim=config.trunk_output_dim,
            num_frequencies=config.num_fourier_frequencies,
            trunk_hidden_dims=config.trunk_hidden_dims,
            num_tcn_layers=config.num_tcn_layers,
            dropout=0.1,
        )

        # ============================================================
        # 6. OUTPUT HEAD
        # ============================================================
        self.output_net = OutputNetwork(
            operator_head_feature_dim=(
                branch_dim if config.prediction_target == "hic"
                else config.trunk_output_dim
            ),
            operator_head_hidden_dims=config.operator_head_hidden_dims,
            dropout=0.1,
        )

    def forward(
        self,
        mesh: torch.Tensor,          # (total_nodes, 3)
        mesh_batch: torch.Tensor,    # (total_nodes,)
        indentor: torch.Tensor,      # (batch_size, 2 or 3)
        time: torch.Tensor | None = None,
        time_batch: torch.Tensor | None = None,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """
        Returns:
            hic: (batch_size,)                     if prediction_target == "hic"
            acceleration: (total_time_points,)     if prediction_target == "acceleration"
        """

        # ============================================================
        # 1. GLOBAL GEOMETRY ENCODING
        # ============================================================
        global_branch = self.mesh_encoder(
            mesh,
            mesh_batch,
            indentor,
            batch_size,
        )  # (B, pointnet_output_dim)

        global_branch = self.film_layer1(global_branch, indentor)
        global_branch = self.film_layer2(global_branch, indentor)

        # ============================================================
        # 2. LOCAL GEOMETRY ENCODING
        # ============================================================
        indentor_per_node = indentor[mesh_batch][:, :2]
        mesh_xy = mesh[:, :2]
        rel_xy = mesh_xy - indentor_per_node

        dist_xy = torch.norm(rel_xy, dim=-1)
        local_mask = dist_xy < self.config.local_radius

        if not local_mask.any():
            local_mask[:] = True

        local_input = torch.cat([mesh, rel_xy], dim=-1)

        local_branch = self.local_encoder(
            local_input[local_mask],
            mesh_batch[local_mask],
            batch_size,
        )  # (B, local_feature_dim)

        # ---------------- DEBUG: local neighborhood size ----------------
        if not hasattr(self, "_debug_local_printed"):
            self._debug_local_printed = 0

        if self._debug_local_printed < 5:  # print only first few times
            with torch.no_grad():
                counts = []
                for b in range(batch_size):
                    counts.append((mesh_batch[local_mask] == b).sum().item())

                counts_tensor = torch.tensor(counts, device=mesh.device)

                print(
                    f"[LocalRadius DEBUG] radius={self.config.local_radius:.4f} | "
                    f"local nodes per sample: "
                    f"min={counts_tensor.min().item()}, "
                    f"max={counts_tensor.max().item()}, "
                    f"mean={counts_tensor.float().mean().item():.1f}"
                )

            self._debug_local_printed += 1
        # ---------------------------------------------------------------

        # ============================================================
        # 3. COMBINE GLOBAL + LOCAL
        # ============================================================
        branch = torch.cat([global_branch, local_branch], dim=-1)
        branch = self.branch_transform(branch)

        # ============================================================
        # 4. HIC MODE → BRANCH ONLY
        # ============================================================
        if self.config.prediction_target == "hic":
            out = self.output_net(branch)      # (B, 1)
            return out.squeeze(-1)             # (B,)

        # ============================================================
        # 5. ACCELERATION MODE → BRANCH × TRUNK
        # ============================================================
        trunk = self.trunk(time, time_batch, batch_size)
        branch_expanded = branch[time_batch]

        combined = torch.cat(
            [branch_expanded, trunk],
            dim=-1
        )  # (total_time_points, branch_dim + trunk_output_dim)

        # ============================================================
        # 6. OUTPUT HEAD
        # ============================================================
        out = self.output_net(combined).squeeze(-1)

        return out


def parse_args(argv=None):
    """Command line overrides. With no flags the historical defaults are used."""
    p = argparse.ArgumentParser(
        description="Train the temporal DeepONet + PointNet++ hood-impact operator.",
    )
    # dataset
    p.add_argument("--data-format", default=None,
                   choices=["legacy", "industrylike", "euroncap1704"])
    p.add_argument("--prediction-target", default=None, choices=["hic", "acceleration"])
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--samples-per-design", type=int, default=None)
    # design-wise split (0-indexed design ids)
    p.add_argument("--test-designs", type=int, nargs="+", default=[2])
    p.add_argument("--val-designs", type=int, nargs="+", default=[10])
    # architecture
    p.add_argument("--arch", default=None, choices=["hic_direct_best"],
                   help="Architecture preset applied before any other override; "
                        "hic_direct_best = runs/hic_target_value/20260123_113525_best.")
    p.add_argument("--local-radius", type=float, default=None)
    # optimisation
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    # bookkeeping
    p.add_argument("--output-dir", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--wandb-project", default="hood-impact")
    return p.parse_args(argv)


def build_config(args) -> Config:
    """Config() with the preset applied first, then the explicit CLI overrides."""
    overrides = dict(HIC_DIRECT_BEST_ARCH) if args.arch == "hic_direct_best" else {}
    for key, value in (
        ("data_format", args.data_format),
        ("prediction_target", args.prediction_target),
        ("num_samples", args.num_samples),
        ("samples_per_design", args.samples_per_design),
        ("local_radius", args.local_radius),
        ("num_epochs", args.epochs),
        ("batch_size", args.batch_size),
        ("learning_rate", args.lr),
        ("weight_decay", args.weight_decay),
        ("seed", args.seed),
        ("output_dir", args.output_dir),
    ):
        if value is not None:
            overrides[key] = value
    return Config(**overrides)


def main(argv=None):
    args = parse_args(argv)
    config = build_config(args)
    set_seed(config.seed)
    log_path = os.path.join(config.output_dir, "log.log")
    logger = setup_logging(log_path)
    logger.info(f"Output directory: {config.output_dir}")
    log_config(config, logger)
    run_name = args.run_name or f"run_{config.stamp}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(config),
        dir=config.output_dir,
    )
    for k in sorted(vars(config)):
        logger.info(f"{k:30s}: {getattr(config, k)}")
    logger.info("-" * 60)
    logger.info(f"Device: {config.device}")
    logger.info(f"Prediction target: {config.prediction_target}")
    logger.info(f"Number of samples: {config.num_samples}")
    logger.info("\n" + "=" * 60)
    logger.info("Loading and Preprocessing Data")
    logger.info("=" * 60)
    TEST_DESIGNS = sorted(set(args.test_designs))
    VAL_DESIGN = sorted(set(args.val_designs))
    overlap = set(TEST_DESIGNS) & set(VAL_DESIGN)
    if overlap:
        raise ValueError(f"test and validation designs overlap: {sorted(overlap)}")
    num_designs = -(-config.num_samples // config.samples_per_design)
    out_of_range = [d for d in TEST_DESIGNS + VAL_DESIGN if not 0 <= d < num_designs]
    if out_of_range:
        raise ValueError(
            f"design ids {out_of_range} are outside 0..{num_designs - 1} "
            f"({config.num_samples} samples / {config.samples_per_design} per design)"
        )
    train_designs = [d for d in range(num_designs)
                     if d not in TEST_DESIGNS and d not in VAL_DESIGN]
    logger.info(
        f"Design split -> train {train_designs} | val {VAL_DESIGN} | test {TEST_DESIGNS}"
    )

    preprocessor = DataPreprocessor(config)
    data_dict = preprocessor.load_all_data()
    num_loaded = len(data_dict["run_numbers"])
    if num_loaded != config.num_samples:
        logger.warning(
            f"loaded {num_loaded} of {config.num_samples} runs -- some runs were "
            "skipped, see the [WARN] lines above"
        )
    train_loader, val_loader, test_loader, test_dataset = create_data_loaders(
        data_dict,
        preprocessor,
        config,
        test_design_ids=TEST_DESIGNS,
        val_design_id=VAL_DESIGN,
    )
    train_dataset = train_loader.dataset
    if config.prediction_target == "hic":
        preprocessor.fit_scalers(
            mesh_geometries=train_dataset.mesh_geometries,
            indentor_positions=train_dataset.indentor_positions,
            hic_values=train_dataset.hic_values,
        )
    else:
        preprocessor.fit_scalers(
            mesh_geometries=train_dataset.mesh_geometries,
            indentor_positions=train_dataset.indentor_positions,
            time_arrays=train_dataset.time_arrays,
            accelerations=train_dataset.accelerations,
        )
    scaler_path = os.path.join(config.output_dir, "scalers.joblib")
    preprocessor.save_scalers(scaler_path)
    logger.info(f"Saved scalers to {scaler_path}")
    logger.info("\n" + "=" * 60)
    logger.info("Initializing Model")
    logger.info("=" * 60)
    model = HoodImpactNeuralOperator(config)
    log_model_architecture(model, logger)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {num_params:,}")
    logger.info(f"Trainable parameters: {num_trainable:,}")
    logger.info("\n" + "=" * 60)
    logger.info("Training Model")
    logger.info("=" * 60)
    trainer = Trainer(model, train_loader, val_loader, config, preprocessor)
    training_history = trainer.train()
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating Model")
    logger.info("=" * 60)
    ckpt_path = os.path.join(config.output_dir, "hood_impact_best_model.pt")
    trainer.load_checkpoint(ckpt_path)
    evaluator = Evaluator(model, config, preprocessor)
    metrics, predictions, targets = evaluator.evaluate(test_loader)
    wandb.log({f"test/{k}": v for k, v in metrics.items()})
    logger.info("Test metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")
    evaluator.plot_training_history(
        training_history["train_losses"],
        training_history["val_losses"],
        save_path=os.path.join(config.output_dir, "training_history.png"),
    )
    if config.prediction_target == "acceleration":
        per_sample_results = evaluator.evaluate_per_sample(test_dataset)

        test_export_dir = os.path.join(config.output_dir, "test_predictions")
        evaluator.plot_predictions(
            per_sample_results,
            output_dir=test_export_dir,
        )
    elif config.prediction_target == "hic":
        export_hic_test_predictions(
            predictions=predictions,
            targets=targets,
            data_dict=data_dict,
            config=config,
            test_design_ids=TEST_DESIGNS,
            samples_per_design=config.samples_per_design,
        )
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Complete!")
    logger.info("=" * 60)
    logger.info(f"Outputs saved to: {config.output_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()