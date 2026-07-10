import os
import wandb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')
from utils.utils import log_config, set_seed, setup_logging, Config, DataPreprocessor, create_data_loaders, Trainer, Evaluator, log_model_architecture, export_hic_test_predictions


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


# ============================================================
# POINTNET++ CORE OPERATIONS
# ============================================================

def farthest_point_sampling(
    pos: torch.Tensor,
    batch: torch.Tensor,
    npoint: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Farthest Point Sampling per batch element.

    Args:
        pos: (N, 3) point coordinates
        batch: (N,) batch indices
        npoint: number of points to sample per batch element

    Returns:
        sampled_idx: (total_sampled,) global indices of sampled points
        sampled_batch: (total_sampled,) batch indices for sampled points
    """
    device = pos.device
    batch_size = batch.max().item() + 1

    all_indices = []
    all_batch_ids = []

    for b in range(batch_size):
        mask = batch == b
        local_indices = torch.where(mask)[0]
        n = local_indices.shape[0]

        if n <= npoint:
            all_indices.append(local_indices)
            all_batch_ids.append(torch.full((n,), b, dtype=torch.long, device=device))
            continue

        local_pos = pos[local_indices]  # (n, 3)

        selected = torch.zeros(npoint, dtype=torch.long, device=device)
        distances = torch.full((n,), float('inf'), device=device)
        selected[0] = 0

        for i in range(1, npoint):
            centroid = local_pos[selected[i - 1]].unsqueeze(0)
            dist = torch.sum((local_pos - centroid) ** 2, dim=-1)
            distances = torch.min(distances, dist)
            selected[i] = torch.argmax(distances)

        all_indices.append(local_indices[selected])
        all_batch_ids.append(torch.full((npoint,), b, dtype=torch.long, device=device))

    return torch.cat(all_indices), torch.cat(all_batch_ids)


def ball_query(
    query_pos: torch.Tensor,
    support_pos: torch.Tensor,
    query_batch: torch.Tensor,
    support_batch: torch.Tensor,
    radius: float,
    nsample: int,
) -> torch.Tensor:
    """
    Ball query: for each query point, find up to nsample support points
    within the given radius. Points beyond radius are replaced with the
    nearest neighbor.

    Args:
        query_pos: (M, 3) query positions (centroids)
        support_pos: (N, 3) support positions
        query_batch: (M,) batch indices for queries
        support_batch: (N,) batch indices for supports
        radius: search radius
        nsample: max number of neighbors

    Returns:
        neighbor_idx: (M, nsample) indices into support_pos
    """
    device = query_pos.device
    M = query_pos.shape[0]
    batch_size = max(query_batch.max().item(), support_batch.max().item()) + 1

    neighbor_idx = torch.zeros(M, nsample, dtype=torch.long, device=device)

    for b in range(batch_size):
        q_mask = query_batch == b
        s_mask = support_batch == b

        q_idx = torch.where(q_mask)[0]
        s_idx = torch.where(s_mask)[0]

        if q_idx.shape[0] == 0 or s_idx.shape[0] == 0:
            continue

        # Pairwise distances: (Mq, Ns)
        dists = torch.cdist(query_pos[q_idx], support_pos[s_idx])

        K = min(nsample, s_idx.shape[0])
        sorted_dists, sorted_local = dists.sort(dim=1)
        top_local = sorted_local[:, :K]       # (Mq, K)
        top_dists = sorted_dists[:, :K]        # (Mq, K)

        # Convert to global indices
        top_global = s_idx[top_local]           # (Mq, K)

        # Replace neighbors beyond radius with nearest neighbor
        beyond = top_dists >= radius
        first_col = top_global[:, 0:1].expand_as(top_global)
        top_global = torch.where(beyond, first_col, top_global)

        # Pad to nsample if K < nsample
        if K < nsample:
            pad = top_global[:, 0:1].expand(-1, nsample - K)
            top_global = torch.cat([top_global, pad], dim=1)

        neighbor_idx[q_idx] = top_global

    return neighbor_idx


class SetAbstraction(nn.Module):
    """
    PointNet++ Set Abstraction (SA) module.
    FPS -> Ball Query -> Grouping -> Shared MLP -> Max Pool
    """

    def __init__(
        self,
        npoint: int,
        radius: float,
        nsample: int,
        in_channels: int,
        mlp_dims: List[int],
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        # MLP input: relative_xyz (3) + features (in_channels)
        input_dim = 3 + in_channels
        dims = [input_dim] + mlp_dims

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*layers)

        self.output_dim = mlp_dims[-1]

    def forward(
        self,
        pos: torch.Tensor,
        features: Optional[torch.Tensor],
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pos: (N, 3) point positions
            features: (N, C) point features, or None
            batch: (N,) batch indices

        Returns:
            new_pos: (M, 3) sampled centroid positions
            new_features: (M, D) aggregated features
            new_batch: (M,) batch indices for centroids
        """
        # 1. Farthest Point Sampling
        fps_idx, new_batch = farthest_point_sampling(pos, batch, self.npoint)
        new_pos = pos[fps_idx]  # (M, 3)

        # 2. Ball Query
        neighbor_idx = ball_query(
            new_pos, pos, new_batch, batch,
            self.radius, self.nsample,
        )  # (M, nsample)

        # 3. Grouping: compute relative positions and gather features
        M = new_pos.shape[0]
        neighbor_pos = pos[neighbor_idx]  # (M, nsample, 3)
        relative_pos = neighbor_pos - new_pos.unsqueeze(1)  # (M, nsample, 3)

        if features is not None:
            neighbor_feat = features[neighbor_idx]  # (M, nsample, C)
            grouped = torch.cat([relative_pos, neighbor_feat], dim=-1)
        else:
            grouped = relative_pos  # (M, nsample, 3)

        # 4. Shared MLP (point-wise)
        grouped_flat = grouped.reshape(-1, grouped.shape[-1])  # (M*nsample, in)
        out_flat = self.mlp(grouped_flat)                      # (M*nsample, D)
        out = out_flat.reshape(M, self.nsample, -1)            # (M, nsample, D)

        # 5. Max Pool over neighbors
        new_features = out.max(dim=1)[0]  # (M, D)

        return new_pos, new_features, new_batch


class GlobalSetAbstraction(nn.Module):
    """
    Global Set Abstraction: aggregates all remaining points to one
    feature vector per batch via shared MLP + max pooling.
    """

    def __init__(self, in_channels: int, mlp_dims: List[int]):
        super().__init__()

        input_dim = 3 + in_channels
        dims = [input_dim] + mlp_dims

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*layers)

        self.output_dim = mlp_dims[-1]

    def forward(
        self,
        pos: torch.Tensor,
        features: torch.Tensor,
        batch: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Args:
            pos: (N, 3)
            features: (N, C)
            batch: (N,)
            batch_size: int

        Returns:
            global_features: (batch_size, D)
        """
        x = torch.cat([pos, features], dim=-1)  # (N, 3+C)
        x = self.mlp(x)                         # (N, D)
        return scatter_max(x, batch, batch_size) # (batch_size, D)


class PointNetPPEncoder(nn.Module):
    """
    PointNet++ encoder for variable-size point clouds.
    Hierarchical feature learning via Set Abstraction layers.
    """

    def __init__(self, config):
        super().__init__()

        # SA layer hyper-parameters (configurable with defaults)
        sa1_npoint  = getattr(config, 'sa1_npoint', 256)
        sa1_radius  = getattr(config, 'sa1_radius', 0.2)
        sa1_nsample = getattr(config, 'sa1_nsample', 32)
        sa1_mlp     = getattr(config, 'sa1_mlp', [64, 64, 128])

        sa2_npoint  = getattr(config, 'sa2_npoint', 64)
        sa2_radius  = getattr(config, 'sa2_radius', 0.4)
        sa2_nsample = getattr(config, 'sa2_nsample', 64)
        sa2_mlp     = getattr(config, 'sa2_mlp', [128, 128, 256])

        global_sa_mlp = getattr(
            config, 'global_sa_mlp',
            [256, 256, config.pointnet_output_dim],
        )

        # SA1: raw xyz -> local features
        self.sa1 = SetAbstraction(
            npoint=sa1_npoint, radius=sa1_radius,
            nsample=sa1_nsample, in_channels=0, mlp_dims=sa1_mlp,
        )

        # SA2: local features -> higher-level features
        self.sa2 = SetAbstraction(
            npoint=sa2_npoint, radius=sa2_radius,
            nsample=sa2_nsample, in_channels=self.sa1.output_dim,
            mlp_dims=sa2_mlp,
        )

        # Global SA: aggregate to per-batch feature
        self.global_sa = GlobalSetAbstraction(
            in_channels=self.sa2.output_dim,
            mlp_dims=global_sa_mlp,
        )

        self.output_dim = self.global_sa.output_dim

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        indentor: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Args:
            x: (total_nodes, 3) - All point coordinates concatenated
            batch: (total_nodes,) - Batch index for each point
            indentor: (batch_size, 2 or 3) - Indentor positions (unused)
            batch_size: Number of samples in batch

        Returns:
            global_features: (batch_size, pointnet_output_dim)
        """
        pos = x
        features = None

        # Hierarchical set abstraction
        pos, features, batch_idx = self.sa1(pos, features, batch)
        pos, features, batch_idx = self.sa2(pos, features, batch_idx)

        # Global aggregation
        global_features = self.global_sa(pos, features, batch_idx, batch_size)

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
        self.mesh_encoder = PointNetPPEncoder(config)

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
                else branch_dim + config.trunk_output_dim
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


def main():
    config = Config()
    set_seed(config.seed)
    log_path = os.path.join(config.output_dir, "log.log")
    logger = setup_logging(log_path)
    logger.info(f"Output directory: {config.output_dir}")
    log_config(config, logger)
    wandb.init(
        project="hood-impact",
        name=f"run_{config.stamp}",
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
    preprocessor = DataPreprocessor(config)
    data_dict = preprocessor.load_all_data()
    TEST_DESIGNS = [1, 2, 6, 7, 8, 9, 11, 15, 17, 19]
    VAL_DESIGN = 12
    # TEST_DESIGNS = [2]
    # VAL_DESIGN = 10
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
            samples_per_design=50
        )
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Complete!")
    logger.info("=" * 60)
    logger.info(f"Outputs saved to: {config.output_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()