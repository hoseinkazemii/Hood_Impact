import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import warnings
import wandb
warnings.filterwarnings('ignore')
from utils.utils import set_seed, setup_logging, Config, DataPreprocessor, create_data_loaders, Trainer, Evaluator


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


class PointNetEncoder(nn.Module):
    """PointNet encoder for variable-size point clouds.
    
    Uses shared MLPs and global max pooling for permutation invariance.
    No padding - uses batch indices for pooling.
    """
    
    def __init__(
        self, 
        pointnet_input_dim: int = 3,
        pointnet_hidden_dims: List[int] = [64, 128, 256],
        pointnet_output_dim: int = 512
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
        
    def forward(self, x: torch.Tensor, batch: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Args:
            x: (total_nodes, 3) - All point coordinates concatenated
            batch: (total_nodes,) - Batch index for each point
            batch_size: Number of samples in batch
            
        Returns:
            global_features: (batch_size, output_dim)
        """
        # Apply shared MLP to all points
        x = self.shared_mlp(x)  # (total_nodes, output_dim)
        
        # Global max pooling per sample using scatter
        global_features = scatter_max(x, batch, batch_size)  # (batch_size, output_dim)
        
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
        film_hidden_dim: int = 128
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


class TrunkNetwork(nn.Module):
    """Trunk network for encoding time queries (DeepONet-style).
    
    Maps scalar time values to high-dimensional features that can
    interact with the branch (mesh encoder) output.
    """
    
    def __init__(
        self,
        trunk_input_dim: int,
        trunk_hidden_dims: List[int],
        trunk_output_dim: int
    ):
        super().__init__()
        
        dims = [trunk_input_dim] + trunk_hidden_dims + [trunk_output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i+1]))
                layers.append(nn.GELU())
        
        self.mlp = nn.Sequential(*layers)
        self.output_dim = trunk_output_dim
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (total_time_points,) - Flattened time values
            
        Returns:
            time_features: (total_time_points, output_dim)
        """
        t = t.unsqueeze(-1)  # (total_time_points, 1)
        return self.mlp(t)


class OutputNetwork(nn.Module):
    """Combines branch and trunk features to produce scalar output."""
    
    def __init__(
        self,
        operator_head_feature_dim: int,
        operator_head_hidden_dims: List[int]
    ):
        super().__init__()
        
        dims = [operator_head_feature_dim] + operator_head_hidden_dims + [1]
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
            x: (total_time_points, operator_head_feature_dim)
            
        Returns:
            output: (total_time_points,)
        """
        return self.mlp(x).squeeze(-1)


class HoodImpactNeuralOperator(nn.Module):
    """Neural operator for hood impact acceleration prediction.
    
    DeepONet-style architecture:
        1. Branch: PointNet encodes mesh geometry -> global features
        2. FiLM layers modulate branch features based on indentor position
        3. Trunk: MLP encodes query time points
        4. Combine branch and trunk features to predict acceleration at each time point
    
    This allows predicting at arbitrary/exact time points (no interpolation needed).
    """
    
    def __init__(self, config: Config):
        super().__init__()
        
        self.config = config
        
        # Branch: PointNet encoder for mesh geometry
        self.mesh_encoder = PointNetEncoder(
            input_dim=config.pointnet_input_dim,
            hidden_dims=config.pointnet_hidden_dims,
            output_dim=config.pointnet_output_dim
        )
        
        # FiLM layers for indentor position conditioning
        self.film_layer1 = FiLMLayer(
            film_feature_dim=config.pointnet_output_dim,
            film_condition_dim=config.film_condition_dim,  # (x, y) position
            film_hidden_dim=config.film_hidden_dim
        )
        
        self.film_layer2 = FiLMLayer(
            film_feature_dim=config.pointnet_output_dim,
            film_condition_dim=config.film_condition_dim,
            film_hidden_dim=config.film_hidden_dim
        )
        
        # Additional branch processing
        self.branch_transform = nn.Sequential(
            nn.Linear(config.pointnet_output_dim, config.pointnet_output_dim),
            nn.LayerNorm(config.pointnet_output_dim),
            nn.GELU(),
        )
        
        # Trunk: Time encoder
        self.trunk = TrunkNetwork(
            trunk_input_dim=config.trunk_input_dim,
            trunk_hidden_dims=config.trunk_hidden_dims,
            trunk_output_dim=config.trunk_output_dim
        )
        
        # Output network: combines branch and trunk features
        # Using element-wise product + MLP (modified DeepONet)
        self.output_net = OutputNetwork(
            operator_head_feature_dim=config.pointnet_output_dim,  # After element-wise product
            operator_head_hidden_dims=config.operator_head_hidden_dims
        )
        
    def forward(
        self, 
        mesh: torch.Tensor,
        mesh_batch: torch.Tensor,
        indentor: torch.Tensor,
        time: torch.Tensor,
        time_batch: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        Args:
            mesh: (total_nodes, 3) - All mesh coordinates concatenated
            mesh_batch: (total_nodes,) - Batch index for each node
            indentor: (batch_size, 2) - Indentor (x, y) positions
            time: (total_time_points,) - All time points concatenated
            time_batch: (total_time_points,) - Batch index for each time point
            batch_size: Number of samples in batch
            
        Returns:
            acceleration: (total_time_points,) - Predicted acceleration at each time point
        """
        # ===== Branch: Encode mesh + condition on indentor =====
        # Encode mesh geometry
        branch_features = self.mesh_encoder(mesh, mesh_batch, batch_size)  # (batch_size, feature_dim)
        
        # Apply FiLM conditioning with indentor position
        branch_features = self.film_layer1(branch_features, indentor)
        branch_features = F.gelu(branch_features)
        branch_features = self.film_layer2(branch_features, indentor)
        
        # Additional transformation with residual
        branch_features = self.branch_transform(branch_features) + branch_features  # (batch_size, feature_dim)
        
        # ===== Trunk: Encode time queries =====
        trunk_features = self.trunk(time)  # (total_time_points, feature_dim)
        
        # ===== Combine branch and trunk =====
        # Expand branch features to match time points using time_batch indices
        branch_expanded = branch_features[time_batch]  # (total_time_points, feature_dim)
        
        # Element-wise product (DeepONet-style combination)
        combined = branch_expanded * trunk_features  # (total_time_points, feature_dim)
        
        # Output network
        acceleration = self.output_net(combined)  # (total_time_points,)
        
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