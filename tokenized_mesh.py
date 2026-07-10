import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
import warnings
import wandb
warnings.filterwarnings('ignore')
from utils.utils import set_seed, setup_logging, Config, DataPreprocessor, create_data_loaders, Trainer, Evaluator, AttentionData


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
    # Replace -inf with 0 for empty batches
    out = torch.where(torch.isinf(out), torch.zeros_like(out), out)
    return out


class TransolverMeshEncoderCrossAttn(nn.Module):
    """
    Transolver-style mesh encoder (irregular meshes) with CROSS-attention conditioning on impact position.

    Input:
        coords: (N_total, 3)  [x,y,z]
        batch : (N_total,)
        impact: (B, 2)        [impact x,y]
        batch_size: B

    Per block:
        nodes -> slice weights -> geometry tokens
        geometry tokens cross-attend to impact-context tokens (no token self-attn)
        deslice back to nodes + residual + LN

    Output:
        global_features: (B, output_dim)
    """

    def __init__(
        self,
        input_dim: int = 3,
        impact_dim: int = 2,
        heads: int = 8,
        dim_head: int = 32,
        slices: int = 32,
        num_attn_blocks: int = 2,
        dropout: float = 0.0,
        output_dim: int = 512,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.slices = slices
        self.num_attn_blocks = num_attn_blocks
        self.scale = dim_head ** -0.5

        # Node embedding (xyz only)
        self.node_embed = nn.Sequential(
            nn.Linear(input_dim, self.inner_dim),
            nn.LayerNorm(self.inner_dim),
        )

        # Temperature for slicing (per head)
        self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)

        # Projections to head space for slicing & token formation
        self.in_project_x  = nn.Linear(self.inner_dim, self.inner_dim)
        self.in_project_fx = nn.Linear(self.inner_dim, self.inner_dim)

        # Slice logits from per-head features (D -> K)
        self.in_project_slice = nn.Linear(dim_head, slices, bias=True)
        nn.init.orthogonal_(self.in_project_slice.weight)

        self.impact_proj = nn.Linear(impact_dim, dim_head)

        # Cross-attn projections:
        # Q from geometry tokens; K,V from impact context tokens
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        # Map back to node feature space
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, self.inner_dim),
            nn.Dropout(dropout),
        )

        # Norm after each block
        self.norms = nn.ModuleList([nn.LayerNorm(self.inner_dim) for _ in range(num_attn_blocks)])

        # Output head after pooling
        self.out_proj = nn.Sequential(
            nn.Linear(self.inner_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )
        self.output_dim = output_dim

    def forward(
        self,
        coords: torch.Tensor,      # (N_total, 3)
        batch: torch.Tensor,       # (N_total,)
        impact: torch.Tensor,      # (B, 2)
        batch_size: int,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:

        x = self.node_embed(coords)  # (N_total, inner_dim)

        debug = {} if return_attention else None

        for blk in range(self.num_attn_blocks):
            outputs = []
            blk_debug = [] if return_attention else None

            for b in range(batch_size):
                mask = (batch == b)
                xb = x[mask]                      # (N_b, inner_dim)
                N_b = xb.shape[0]
                D = self.dim_head

                # ---- Slice (geometry-only) ----
                fx_mid = self.in_project_fx(xb).view(N_b, self.heads, self.dim_head).permute(1, 0, 2)  # (H, N, D)
                x_mid  = self.in_project_x(xb).view(N_b, self.heads, self.dim_head).permute(1, 0, 2)   # (H, N, D)

                slice_logits = self.in_project_slice(x_mid)  # (H, N, K)

                # IMPORTANT: avoid broadcasting to 4D; use (H,1,1) temperature
                temp = self.temperature[0, :, 0, 0].view(self.heads, 1, 1).clamp(0.1, 10.0)  # (H,1,1)
                slice_weights = F.softmax(slice_logits / temp, dim=-1)  # (H, N, K)

                slice_norm = slice_weights.sum(dim=1).clamp_min(1e-6)  # (H, K)

                # Geometry tokens: (H, K, D)
                geom_tokens = torch.einsum("hnd,hnk->hkd", fx_mid, slice_weights)
                geom_tokens = geom_tokens / slice_norm.unsqueeze(-1)

                impact_ctx = self.impact_proj(impact[b])      # (D,)
                impact_ctx = impact_ctx.view(1, 1, D)          # (1,1,D)

                # ---- Cross-attention: geometry tokens attend to impact tokens ----
                K = self.to_k(geom_tokens) + impact_ctx        # (H,K,D)
                V = self.to_v(geom_tokens)                     # (H,K,D)
                Q = self.to_q(geom_tokens)

                attn = torch.einsum("hkd,hqd->hkq", Q, K) * self.scale  # (H, K, K)
                attn = F.softmax(attn, dim=-1)

                tokens_out = torch.einsum("hkq,hqd->hkd", attn, V)  # (H, K, D)

                # ---- De-slice back to nodes ----
                out_x = torch.einsum("hkd,hnk->hnd", tokens_out, slice_weights)  # (H, N, D)
                out_x = out_x.permute(1, 0, 2).contiguous().view(N_b, self.inner_dim)  # (N, inner_dim)
                out_x = self.to_out(out_x)

                outputs.append(out_x)

                if return_attention:
                    blk_debug.append({
                        "slice_weights": slice_weights.detach().cpu(),  # (H,N,K)
                        "cross_attn": attn.detach().cpu(),              # (H,K,K)
                    })

            y = torch.cat(outputs, dim=0)           # (N_total, inner_dim)
            x = self.norms[blk](x + y)              # residual + LN

            if return_attention:
                debug[f"block_{blk}"] = blk_debug

        pooled = scatter_max(x, batch, batch_size)      # (B, inner_dim)
        global_features = self.out_proj(pooled)         # (B, output_dim)
        return global_features, debug

    
class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.

    Applies a learned, condition-dependent affine transformation:
        y = gamma(condition) * x + beta(condition)

    where gamma and beta are produced by a small MLP from the conditioning input.
    """

    def __init__(
        self,
        feature_dim: int,
        condition_dim: int,
        hidden_dim: int = 128
    ):
        super().__init__()

        # MLP to process conditioning input
        self.mlp = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Heads for FiLM parameters
        self.gamma = nn.Linear(hidden_dim, feature_dim)
        self.beta = nn.Linear(hidden_dim, feature_dim)

        # Initialize FiLM parameters:
        # gamma ≈ 1, beta ≈ 0  → identity modulation at initialization
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:         (batch_size, feature_dim)       - features to be modulated
            condition: (batch_size, condition_dim)     - conditioning input

        Returns:
            modulated_x: (batch_size, feature_dim)
        """
        h = self.mlp(condition)
        gamma = self.gamma(h)
        beta = self.beta(h)
        return gamma * x + beta


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
            self.frequencies = nn.Parameter(
                torch.randn(num_frequencies, input_dim) * scale
            )
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


class TrunkNetwork(nn.Module):
    """Trunk network for encoding time queries (DeepONet-style).
    
    Maps scalar time values to high-dimensional features that can
    interact with the branch (mesh encoder) output.
    """
    
    def __init__(
        self,
        input_dim: int = 1,  # Time is 1D
        hidden_dims: List[int] = [128, 128, 256],
        output_dim: int = 512
    ):
        super().__init__()
        self.fourier_features = FourierFeatures(
            input_dim=input_dim,
            num_frequencies=8,
            scale=0.1,
            learnable=False,
            debug=True
        )
        dims = [self.fourier_features.output_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i+1]))
                layers.append(nn.GELU())
        
        self.mlp = nn.Sequential(*layers)
        self.output_dim = output_dim
        
    def forward(self, t: torch.Tensor, time_batch_indices: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Args:
            t: (total_time_points,) - Flattened time values
            time_batch_indices: (total_time_points,) - Batch indices for each time point
            batch_size: int - Number of batches
        Returns:
            time_features: (total_time_points, output_dim)
        """
        t_ff = self.fourier_features(t.unsqueeze(-1))

        if hasattr(self, "_printed") is False:
            self._printed = True
            with torch.no_grad():
                print("[Trunk DEBUG]")
                print(f"  Fourier output shape: {t_ff.shape}")
                print(f"  Fourier mean / std: "
                      f"{t_ff.mean().item():.4e} / {t_ff.std().item():.4e}")

        t_ff = self.mlp(t_ff)

        return t_ff


class OutputNetwork(nn.Module):
    """Combines branch and trunk features to produce scalar output."""
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dims: List[int] = [256, 128]
    ):
        super().__init__()
        
        dims = [feature_dim] + hidden_dims + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i+1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(0.1))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (total_time_points, feature_dim)
            
        Returns:
            output: (total_time_points,)
        """
        return self.mlp(x).squeeze(-1)


class HoodImpactNeuralOperator(nn.Module):
    """
    Neural operator for hood impact acceleration prediction.

    Design:
        1. Branch: Transolver-style mesh encoder with CROSS-attention
           (geometry tokens attend to impact context)
        2. Trunk: Time encoder (Fourier + MLP)
        3. Operator head: combine branch (impact-conditioned geometry)
           with trunk (time query) via elementwise product
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # ---- Branch: impact-conditioned mesh encoder ----
        self.mesh_encoder = TransolverMeshEncoderCrossAttn(
            input_dim=3,                    # (x, y, z)
            impact_dim=2,                   # (impact_x, impact_y)
            heads=config.num_heads,
            dim_head=config.head_dim,
            slices=config.num_tokens,
            num_attn_blocks=config.num_attn_blocks,
            dropout=0.0,
            output_dim=config.pointnet_output_dim
        )

        # Optional light post-branch refinement (kept minimal)
        self.branch_transform = nn.Sequential(
            nn.Linear(config.pointnet_output_dim, config.pointnet_output_dim),
            nn.LayerNorm(config.pointnet_output_dim),
            nn.GELU(),
        )

        # ---- Trunk: time query encoder ----
        self.trunk = TrunkNetwork(
            input_dim=1,
            hidden_dims=config.trunk_hidden_dims,
            output_dim=config.trunk_output_dim
        )

        # ---- Operator head ----
        self.output_net = OutputNetwork(
            feature_dim=config.operator_head_feature_dim,
            hidden_dims=config.operator_head_hidden_dims
        )

    def forward(
        self,
        mesh: torch.Tensor,         # (N_total, 3)
        mesh_batch: torch.Tensor,   # (N_total,)
        impact: torch.Tensor,       # (B, 2)
        time: torch.Tensor,         # (T_total,)
        time_batch: torch.Tensor,   # (T_total,)
        batch_size: int
    ) -> torch.Tensor:
        """
        Returns:
            acceleration: (T_total,)
        """

        # ===== Branch: geometry conditioned on impact via cross-attention =====
        branch_features, _ = self.mesh_encoder(
            coords=mesh,
            batch=mesh_batch,
            impact=impact,
            batch_size=batch_size,
            return_attention=False
        )  # (B, F)

        # Light refinement (residual)
        branch_features = self.branch_transform(branch_features) + branch_features

        # ===== Trunk: encode time queries =====
        trunk_features = self.trunk(
            t=time,
            time_batch_indices=time_batch,
            batch_size=batch_size
        )  # (T_total, F)

        # ===== Operator: time queries the branch =====
        branch_expanded = branch_features[time_batch]   # (T_total, F)
        combined = branch_expanded * trunk_features     # (T_total, F)

        # ===== Output =====
        acceleration = self.output_net(combined)        # (T_total,)
        return acceleration


def main():
    config = Config()
    set_seed(config.seed)
    wandb.init(project="hood-impact", name=f"run_{config.stamp}", config=vars(config), dir=config.output_dir)
    log_path = os.path.join(config.output_dir, "log.log")
    logger = setup_logging(log_path)
    logger.info(f"Output directory: {config.output_dir}")
    logger.info("Configuration:")
    for k, v in vars(config).items():
        logger.info(f"  {k}: {v}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Number of samples: {config.num_samples}")
    logger.info("\n" + "="*60)
    logger.info("Loading and Preprocessing Data")
    logger.info("="*60)
    preprocessor = DataPreprocessor(config)
    data_dict = preprocessor.load_all_data()   
    train_loader, val_loader, test_loader, test_dataset = create_data_loaders(data_dict, preprocessor, config, test_design_id=2, val_design_id=10)
    train_dataset = train_loader.dataset
    preprocessor.fit_scalers(train_dataset.mesh_geometries, train_dataset.indentor_positions, train_dataset.time_arrays, train_dataset.accelerations)
    preprocessor.save_scalers(os.path.join(config.output_dir, "scalers.joblib"))
    logger.info("\n" + "="*60)
    logger.info("Initializing Model")
    logger.info("="*60)
    model = HoodImpactNeuralOperator(config)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {num_params:,}")
    logger.info(f"Trainable parameters: {num_trainable:,}")
    logger.info("\n" + "="*60)
    logger.info("Training Model")
    logger.info("="*60)
    trainer = Trainer(model, train_loader, val_loader, config, preprocessor)
    training_history = trainer.train()  
    logger.info("\n" + "="*60)
    logger.info("Evaluating Model")
    logger.info("="*60)
    trainer.load_checkpoint(os.path.join(config.output_dir, 'hood_impact_best_model.pt'))
    evaluator = Evaluator(model, config, preprocessor)
    metrics, predictions, targets = evaluator.evaluate(test_loader)    
    per_sample_results = evaluator.evaluate_per_sample(test_dataset)
    wandb.log({
        "test/mse": metrics["mse"],
        "test/rmse": metrics["rmse"],
        "test/mae": metrics["mae"],
        "test/relative_error": metrics["relative_error"],
        "test/r2": metrics["r2"],
    })
    logger.info("Test metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")
    evaluator.plot_training_history(
        training_history['train_losses'],
        training_history['val_losses'],
        save_path=os.path.join(config.output_dir, 'training_history.png')
    )
    test_export_dir = os.path.join(config.output_dir, "test_predictions")
    evaluator.plot_predictions(
        per_sample_results,
        output_dir=test_export_dir
    )
    logger.info("\n" + "="*60)
    logger.info("Pipeline Complete!")
    logger.info("="*60)
    logger.info(f"\nOutputs saved to: {config.output_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()