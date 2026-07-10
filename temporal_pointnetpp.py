import os
import wandb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import warnings
warnings.filterwarnings('ignore')

from utils.utils import (
    log_config, set_seed, setup_logging, Config, DataPreprocessor,
    create_data_loaders, Trainer, Evaluator, log_model_architecture,
    export_hic_test_predictions
)


# ============================================================
# BASIC UTILS
# ============================================================

def scatter_max(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    Scatter max operation.

    Args:
        src: (N, D)
        index: (N,)
        dim_size: number of groups

    Returns:
        (dim_size, D)
    """
    D = src.shape[1]
    out = torch.full((dim_size, D), float('-inf'), dtype=src.dtype, device=src.device)
    index_expanded = index.unsqueeze(1).expand(-1, D)
    out = out.scatter_reduce(0, index_expanded, src, reduce='amax', include_self=False)
    out = torch.where(torch.isinf(out), torch.zeros_like(out), out)
    return out


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    Scatter sum.

    Args:
        src: (N, D)
        index: (N,)
        dim_size: number of groups

    Returns:
        (dim_size, D)
    """
    D = src.shape[1]
    out = torch.zeros((dim_size, D), dtype=src.dtype, device=src.device)
    index_expanded = index.unsqueeze(1).expand(-1, D)
    out = out.scatter_add(0, index_expanded, src)
    return out


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    Scatter mean.

    Args:
        src: (N, D)
        index: (N,)
        dim_size: number of groups
    """
    summed = scatter_sum(src, index, dim_size)
    counts = torch.bincount(index, minlength=dim_size).clamp(min=1).to(src.device).unsqueeze(1)
    return summed / counts


def expand_indentor_to_3d(indentor: torch.Tensor) -> torch.Tensor:
    """
    Ensures indentor is 3D.
    If input is (B,2), pads a zero z-coordinate.
    If input is already (B,3), returns as is.
    """
    if indentor.shape[1] == 3:
        return indentor
    elif indentor.shape[1] == 2:
        z = torch.zeros(indentor.shape[0], 1, dtype=indentor.dtype, device=indentor.device)
        return torch.cat([indentor, z], dim=-1)
    else:
        raise ValueError(f"Indentor must have shape (B,2) or (B,3), got {tuple(indentor.shape)}")


def build_node_features(mesh: torch.Tensor, mesh_batch: torch.Tensor, indentor: torch.Tensor) -> torch.Tensor:
    """
    Builds per-node input features.

    Features:
      - xyz
      - relative xyz to indentor
      - Euclidean distance to indentor

    Args:
        mesh: (N, 3)
        mesh_batch: (N,)
        indentor: (B, 2) or (B, 3)

    Returns:
        node_features: (N, 7) = [x,y,z, dx,dy,dz, dist]
    """
    indentor3d = expand_indentor_to_3d(indentor)               # (B,3)
    indentor_per_node = indentor3d[mesh_batch]                 # (N,3)
    rel = mesh - indentor_per_node                             # (N,3)
    dist = torch.norm(rel, dim=-1, keepdim=True)               # (N,1)
    return torch.cat([mesh, rel, dist], dim=-1)                # (N,7)


def split_by_batch(x: torch.Tensor, batch: torch.Tensor, batch_size: int) -> List[torch.Tensor]:
    """
    Splits a concatenated tensor by batch index.

    Args:
        x: (N, C)
        batch: (N,)
        batch_size: B

    Returns:
        list of B tensors of shape (N_b, C)
    """
    return [x[batch == b] for b in range(batch_size)]


def farthest_point_sampling(xyz: torch.Tensor, num_centroids: int) -> torch.Tensor:
    """
    Naive farthest point sampling for one sample.

    Args:
        xyz: (N, 3)
        num_centroids: M

    Returns:
        centroid_indices: (M,)
    """
    device = xyz.device
    N = xyz.shape[0]

    if N == 0:
        return torch.zeros(0, dtype=torch.long, device=device)

    M = min(num_centroids, N)

    centroids = torch.zeros(M, dtype=torch.long, device=device)
    distances = torch.full((N,), float('inf'), device=device)

    # Start with a deterministic point for reproducibility
    farthest = 0
    for i in range(M):
        centroids[i] = farthest
        centroid_xyz = xyz[farthest].unsqueeze(0)                  # (1,3)
        dist = torch.sum((xyz - centroid_xyz) ** 2, dim=-1)        # (N,)
        distances = torch.minimum(distances, dist)
        farthest = torch.argmax(distances).item()

    return centroids


def knn_group(xyz: torch.Tensor, query_xyz: torch.Tensor, k: int) -> torch.Tensor:
    """
    Finds kNN indices from xyz for each query point.

    Args:
        xyz: (N, 3)
        query_xyz: (M, 3)
        k: int

    Returns:
        idx: (M, k)
    """
    N = xyz.shape[0]
    M = query_xyz.shape[0]

    if N == 0 or M == 0:
        return torch.zeros((M, 0), dtype=torch.long, device=xyz.device)

    k = min(k, N)
    dists = torch.cdist(query_xyz, xyz)       # (M, N)
    idx = torch.topk(dists, k=k, dim=-1, largest=False).indices
    return idx                                # (M, k)


class SharedMLP(nn.Module):
    """
    Pointwise MLP applied to the last dimension.
    """

    def __init__(self, dims: List[int], dropout: float = 0.0, final_norm: bool = False):
        super().__init__()

        layers = []
        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))

            is_last = (i == len(dims) - 2)
            if (not is_last) or final_norm:
                layers.append(nn.LayerNorm(out_dim))
            if not is_last:
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)
        self.output_dim = dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# FOURIER + TCN TRUNK (UNCHANGED IN SPIRIT)
# ============================================================

class FourierFeatures(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        num_frequencies: int = 8,
        scale: float = 0.1,
        learnable: bool = False,
        debug: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.output_dim = input_dim + 2 * input_dim * num_frequencies
        self.debug = debug
        self._printed = False

        if learnable:
            self.frequencies = nn.Parameter(torch.randn(num_frequencies, input_dim) * scale)
        else:
            freqs = 2.0 ** torch.arange(0, num_frequencies)
            freqs = freqs.unsqueeze(1) * scale
            self.register_buffer("frequencies", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phase = 2 * math.pi * torch.matmul(x, self.frequencies.T)
        sin_feat = torch.sin(phase)
        cos_feat = torch.cos(phase)

        if self.debug and not self._printed:
            self._printed = True
            with torch.no_grad():
                print("\n[FourierFeatures DEBUG]")
                print(f"x.shape: {tuple(x.shape)}")
                print(f"x.min/max: {x.min().item():.4e} / {x.max().item():.4e}")
                print(f"phase.min/max: {phase.min().item():.4e} / {phase.max().item():.4e}")
                print("[End FourierFeatures DEBUG]\n")

        return torch.cat([x, sin_feat, cos_feat], dim=-1)


class TemporalConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()

        self.padding = ((kernel_size - 1) * dilation) // 2

        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding
        )
        self.norm = nn.GroupNorm(1, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.norm(out)
        out = F.gelu(out)
        out = self.dropout(out)
        return out + self.residual(x)


class TemporalConvStack(nn.Module):
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
            layers.append(
                TemporalConvBlock(
                    channels, channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        x: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        max_len = 0
        sequences = []
        lengths = []

        for b in range(batch_size):
            mask = batch_indices == b
            seq = x[mask]
            sequences.append(seq)
            lengths.append(seq.shape[0])
            max_len = max(max_len, seq.shape[0])

        padded = torch.zeros(batch_size, x.shape[1], max_len, device=x.device, dtype=x.dtype)
        for b, seq in enumerate(sequences):
            if seq.numel() > 0:
                padded[b, :, :lengths[b]] = seq.T

        out = padded
        for layer in self.layers:
            out = layer(out)

        result = torch.zeros_like(x)
        for b in range(batch_size):
            mask = batch_indices == b
            if lengths[b] > 0:
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

        self.fourier = FourierFeatures(
            input_dim=1,
            num_frequencies=num_frequencies,
            scale=0.1,
            learnable=False,
            debug=False
        )

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

        self.temporal_conv = TemporalConvStack(
            channels=embed_dim,
            num_tcn_layers=num_tcn_layers,
            dropout=dropout
        )

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
        t = time.unsqueeze(-1)
        t_fourier = self.fourier(t)
        x = self.initial_mlp(t_fourier)
        x = self.temporal_conv(x, time_batch, batch_size)
        return self.output_proj(x)


# ============================================================
# POINTNET++-STYLE HIERARCHICAL ENCODER
# ============================================================

class SetAbstractionLayer(nn.Module):
    """
    Simplified PointNet++ set abstraction layer:
      1. sample centroids by FPS
      2. gather kNN neighborhoods
      3. encode local neighborhoods with shared MLP
      4. max-pool over neighbors

    This version does not need connectivity.
    """

    def __init__(
        self,
        in_feature_dim: int,
        out_feature_dim: int,
        num_centroids: int,
        k_neighbors: int,
        hidden_dims: List[int],
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_centroids = num_centroids
        self.k_neighbors = k_neighbors

        # Input to local MLP:
        # [relative_xyz(3), center_feat, neighbor_feat]
        local_input_dim = 3 + in_feature_dim + in_feature_dim

        dims = [local_input_dim] + hidden_dims + [out_feature_dim]
        self.local_mlp = SharedMLP(dims, dropout=dropout, final_norm=False)
        self.out_feature_dim = out_feature_dim

    def forward(
        self,
        xyz_list: List[torch.Tensor],
        feat_list: List[torch.Tensor]
    ):
        """
        Args:
            xyz_list: list of B tensors, each (N_b, 3)
            feat_list: list of B tensors, each (N_b, C)

        Returns:
            new_xyz_list: list of B tensors, each (M_b, 3)
            new_feat_list: list of B tensors, each (M_b, out_feature_dim)
        """
        new_xyz_list = []
        new_feat_list = []

        for xyz, feat in zip(xyz_list, feat_list):
            N = xyz.shape[0]
            if N == 0:
                new_xyz_list.append(xyz.new_zeros((0, 3)))
                new_feat_list.append(feat.new_zeros((0, self.out_feature_dim)))
                continue

            fps_idx = farthest_point_sampling(xyz, self.num_centroids)      # (M,)
            centers_xyz = xyz[fps_idx]                                      # (M,3)
            centers_feat = feat[fps_idx]                                    # (M,C)

            knn_idx = knn_group(xyz, centers_xyz, self.k_neighbors)         # (M,K)
            nbr_xyz = xyz[knn_idx]                                          # (M,K,3)
            nbr_feat = feat[knn_idx]                                        # (M,K,C)

            rel_xyz = nbr_xyz - centers_xyz.unsqueeze(1)                    # (M,K,3)
            center_feat_exp = centers_feat.unsqueeze(1).expand_as(nbr_feat) # (M,K,C)

            local_input = torch.cat([rel_xyz, center_feat_exp, nbr_feat], dim=-1)  # (M,K,3+2C)
            local_out = self.local_mlp(local_input)                         # (M,K,F)
            pooled = local_out.max(dim=1).values                            # (M,F)

            new_xyz_list.append(centers_xyz)
            new_feat_list.append(pooled)

        return new_xyz_list, new_feat_list


class SoftIndentorPooling(nn.Module):
    """
    Soft attention pooling of point features conditioned on indentor position.
    This replaces the brittle hard-radius local branch.
    """

    def __init__(self, feature_dim: int, indentor_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.Linear(feature_dim + 4 + indentor_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        xyz_list: List[torch.Tensor],
        feat_list: List[torch.Tensor],
        indentor: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            xyz_list: list of B tensors, each (N_b, 3)
            feat_list: list of B tensors, each (N_b, C)
            indentor: (B, 2) or (B, 3)

        Returns:
            pooled: (B, C)
        """
        B = len(xyz_list)
        pooled = []

        indentor3d = expand_indentor_to_3d(indentor)

        for b in range(B):
            xyz = xyz_list[b]                      # (N,3)
            feat = feat_list[b]                    # (N,C)
            ind_raw = indentor[b].unsqueeze(0)     # (1,2 or 3)
            ind3 = indentor3d[b].unsqueeze(0)      # (1,3)

            if xyz.shape[0] == 0:
                pooled.append(torch.zeros(feat.shape[-1], device=indentor.device, dtype=indentor.dtype))
                continue

            rel = xyz - ind3                       # (N,3)
            dist = torch.norm(rel, dim=-1, keepdim=True)  # (N,1)
            ind_expanded = ind_raw.expand(xyz.shape[0], -1)

            score_input = torch.cat([feat, rel, dist, ind_expanded], dim=-1)
            scores = self.score_mlp(score_input).squeeze(-1)       # (N,)
            weights = torch.softmax(scores, dim=0).unsqueeze(-1)   # (N,1)

            pooled_feat = torch.sum(weights * feat, dim=0)         # (C,)
            pooled.append(pooled_feat)

        return torch.stack(pooled, dim=0)                          # (B,C)


class PointNetPPEncoder(nn.Module):
    """
    Hierarchical point-cloud encoder for irregular meshes without connectivity.

    Pipeline:
      - initial per-node stem MLP
      - set abstraction L1
      - set abstraction L2
      - global max from L2
      - soft indentor pooling from L1 and L2
      - concatenate and project
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Defaults so you do not have to immediately modify Config
        stem_hidden_dim = getattr(config, "pnpp_stem_hidden_dim", 64)
        sa1_out_dim = getattr(config, "pnpp_sa1_out_dim", 128)
        sa2_out_dim = getattr(config, "pnpp_sa2_out_dim", 256)

        sa1_num_centroids = getattr(config, "pnpp_sa1_num_centroids", 256)
        sa2_num_centroids = getattr(config, "pnpp_sa2_num_centroids", 64)

        sa1_k = getattr(config, "pnpp_sa1_k", 32)
        sa2_k = getattr(config, "pnpp_sa2_k", 32)

        stem_dropout = getattr(config, "pnpp_stem_dropout", 0.0)
        sa_dropout = getattr(config, "pnpp_sa_dropout", 0.0)

        self.node_input_dim = 7  # [xyz, rel_xyz, dist]

        self.stem = SharedMLP(
            [self.node_input_dim, stem_hidden_dim, stem_hidden_dim],
            dropout=stem_dropout,
            final_norm=False
        )

        self.sa1 = SetAbstractionLayer(
            in_feature_dim=stem_hidden_dim,
            out_feature_dim=sa1_out_dim,
            num_centroids=sa1_num_centroids,
            k_neighbors=sa1_k,
            hidden_dims=[sa1_out_dim],
            dropout=sa_dropout
        )

        self.sa2 = SetAbstractionLayer(
            in_feature_dim=sa1_out_dim,
            out_feature_dim=sa2_out_dim,
            num_centroids=sa2_num_centroids,
            k_neighbors=sa2_k,
            hidden_dims=[sa2_out_dim],
            dropout=sa_dropout
        )

        indentor_dim = getattr(config, "film_condition_dim", 2)
        self.local_pool_l1 = SoftIndentorPooling(sa1_out_dim, indentor_dim=indentor_dim, hidden_dim=128)
        self.local_pool_l2 = SoftIndentorPooling(sa2_out_dim, indentor_dim=indentor_dim, hidden_dim=128)

        branch_dim_raw = sa1_out_dim + sa2_out_dim + sa2_out_dim  # local_l1 + local_l2 + global_l2
        branch_output_dim = getattr(config, "branch_output_dim", sa2_out_dim)

        self.branch_proj = nn.Sequential(
            nn.Linear(branch_dim_raw, branch_output_dim),
            nn.LayerNorm(branch_output_dim),
            nn.GELU(),
            nn.Dropout(getattr(config, "branch_dropout", 0.1)),
            nn.Linear(branch_output_dim, branch_output_dim),
            nn.LayerNorm(branch_output_dim),
            nn.GELU()
        )

        self.output_dim = branch_output_dim

    def forward(
        self,
        mesh: torch.Tensor,
        mesh_batch: torch.Tensor,
        indentor: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        Args:
            mesh: (N,3)
            mesh_batch: (N,)
            indentor: (B,2) or (B,3)
            batch_size: B

        Returns:
            branch: (B, output_dim)
        """
        # Per-node features
        node_input = build_node_features(mesh, mesh_batch, indentor)      # (N,7)
        node_feat = self.stem(node_input)                                 # (N,C0)

        # Split by batch
        xyz_list = split_by_batch(mesh, mesh_batch, batch_size)
        feat0_list = split_by_batch(node_feat, mesh_batch, batch_size)

        # Hierarchical abstraction
        l1_xyz_list, l1_feat_list = self.sa1(xyz_list, feat0_list)
        l2_xyz_list, l2_feat_list = self.sa2(l1_xyz_list, l1_feat_list)

        # Global max pool at deepest level
        global_l2 = []
        for feat in l2_feat_list:
            if feat.shape[0] == 0:
                global_l2.append(torch.zeros(self.sa2.out_feature_dim, device=mesh.device, dtype=mesh.dtype))
            else:
                global_l2.append(feat.max(dim=0).values)
        global_l2 = torch.stack(global_l2, dim=0)                        # (B, C2)

        # Soft indentor-conditioned local pools
        local_l1 = self.local_pool_l1(l1_xyz_list, l1_feat_list, indentor)  # (B, C1)
        local_l2 = self.local_pool_l2(l2_xyz_list, l2_feat_list, indentor)  # (B, C2)

        # Combine
        branch = torch.cat([local_l1, local_l2, global_l2], dim=-1)
        branch = self.branch_proj(branch)

        return branch


# ============================================================
# OUTPUT HEAD
# ============================================================

class OutputNetwork(nn.Module):
    """
    Operator head that maps combined features to scalar output.
    """

    def __init__(
        self,
        operator_head_feature_dim: int,
        operator_head_hidden_dims: List[int],
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(operator_head_feature_dim, operator_head_hidden_dims[0]),
            nn.LayerNorm(operator_head_hidden_dims[0]),
            nn.GELU(),
        )

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

        self.output_layer = nn.Linear(operator_head_hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = x + block(x)
        return self.output_layer(x).squeeze(-1)


# ============================================================
# MAIN MODEL
# ============================================================

class HoodImpactNeuralOperator(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # 1. Hierarchical geometry encoder
        self.mesh_encoder = PointNetPPEncoder(config)
        branch_dim = self.mesh_encoder.output_dim

        # 2. Trunk (time encoder)
        self.trunk = TrunkNetwork(
            trunk_output_dim=config.trunk_output_dim,
            num_frequencies=config.num_fourier_frequencies,
            trunk_hidden_dims=config.trunk_hidden_dims,
            num_tcn_layers=config.num_tcn_layers,
            dropout=0.1,
        )

        # 3. Output head
        # IMPORTANT FIX:
        # - HIC mode: branch only
        # - acceleration mode: branch + trunk
        operator_input_dim = (
            branch_dim
            if config.prediction_target == "hic"
            else branch_dim + config.trunk_output_dim
        )

        self.output_net = OutputNetwork(
            operator_head_feature_dim=operator_input_dim,
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
            hic: (batch_size,)                  if prediction_target == "hic"
            acceleration: (total_time_points,) if prediction_target == "acceleration"
        """

        # Geometry branch
        branch = self.mesh_encoder(
            mesh=mesh,
            mesh_batch=mesh_batch,
            indentor=indentor,
            batch_size=batch_size,
        )  # (B, branch_dim)

        # HIC mode
        if self.config.prediction_target == "hic":
            out = self.output_net(branch)
            return out

        # Acceleration mode
        if time is None or time_batch is None:
            raise ValueError("time and time_batch must be provided when prediction_target == 'acceleration'")

        trunk = self.trunk(time, time_batch, batch_size)    # (total_time_points, trunk_dim)
        branch_expanded = branch[time_batch]                # (total_time_points, branch_dim)

        combined = torch.cat([branch_expanded, trunk], dim=-1)
        out = self.output_net(combined)
        return out


# ============================================================
# MAIN PIPELINE
# ============================================================

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

    # TEST_DESIGNS = [1, 2, 6, 7, 8, 9, 11, 15, 17, 19]
    # VAL_DESIGN = 12
    TEST_DESIGNS = [2]
    VAL_DESIGN = 10

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