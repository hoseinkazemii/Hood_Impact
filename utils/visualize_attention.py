import os
from typing import Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict


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




# ============================================================================
# Visualization Functions
# ============================================================================

def get_head_assignments(
    data: AttentionData,
    head: Union[int, str] = 'mean'
) -> np.ndarray:
    """
    Get slice assignments for a specific head or averaged across heads.
    
    Args:
        data: AttentionData object
        head: int for specific head, 'mean' for average, 'max' for max across heads
        
    Returns:
        assignments: (N, K) array
    """
    S = data.slice_assignments  # (H, N, K)
    
    if isinstance(head, int):
        return S[head]  # (N, K)
    elif head == 'mean':
        return S.mean(axis=0)  # (N, K)
    elif head == 'max':
        return S.max(axis=0)  # (N, K)
    else:
        raise ValueError(f"head must be int, 'mean', or 'max', got {head}")


def get_head_attention(
    data: AttentionData,
    head: Union[int, str] = 'mean'
) -> np.ndarray:
    """
    Get token attention for a specific head or averaged across heads.
    
    Args:
        data: AttentionData object
        head: int for specific head, 'mean' for average
        
    Returns:
        attention: (K, K) array
    """
    attn = data.token_attention  # (H, K, K)
    
    if isinstance(head, int):
        return attn[head]  # (K, K)
    elif head == 'mean':
        return attn.mean(axis=0)  # (K, K)
    else:
        raise ValueError(f"head must be int or 'mean', got {head}")


def visualize_token_assignments_3d(
    data: AttentionData,
    head: Union[int, str] = 'mean',
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 6),
    point_size: float = 1.0,
    elev: float = 30,
    azim: float = 45,
    save_path: Optional[str] = None,
    alpha: float = 0.7,
    cmap: str = 'tab20'
):
    """
    Visualize token assignments on 3D mesh.
    
    Args:
        data: AttentionData object
        head: int for specific head, 'mean' for average across heads
        title: Plot title (auto-generated if None)
        figsize: Figure size
        point_size: Scatter point size
        elev, azim: 3D view angles
        save_path: Path to save figure
        alpha: Point transparency
        cmap: Colormap for token coloring
    """
    coords = data.coords
    S = get_head_assignments(data, head)  # (N, K)
    
    # Hard assignment
    token_assignments = np.argmax(S, axis=-1)  # (N,)
    
    # Entropy
    entropy = -np.sum(S * np.log(S + 1e-10), axis=-1)  # (N,)
    
    # Max assignment weight (confidence)
    max_weight = S.max(axis=-1)  # (N,)
    
    head_str = f"Head {head}" if isinstance(head, int) else f"{head.capitalize()} of Heads"
    title = title or f"Token Assignments - {head_str}"
    
    fig = plt.figure(figsize=(figsize[0], figsize[1]))
    
    # Plot 1: Hard token assignments
    ax1 = fig.add_subplot(131, projection='3d')
    scatter1 = ax1.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=token_assignments, cmap=cmap, s=point_size, alpha=alpha
    )
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'{title}\nHard Assignment (argmax)')
    ax1.view_init(elev=elev, azim=azim)
    plt.colorbar(scatter1, ax=ax1, shrink=0.5, label='Token ID')
    
    # Plot 2: Entropy
    ax2 = fig.add_subplot(132, projection='3d')
    scatter2 = ax2.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=entropy, cmap='viridis', s=point_size, alpha=alpha
    )
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title(f'{title}\nEntropy (Uncertainty)')
    ax2.view_init(elev=elev, azim=azim)
    plt.colorbar(scatter2, ax=ax2, shrink=0.5, label='Entropy')
    
    # Plot 3: Max weight (confidence)
    ax3 = fig.add_subplot(133, projection='3d')
    scatter3 = ax3.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=max_weight, cmap='plasma', s=point_size, alpha=alpha
    )
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title(f'{title}\nMax Weight (Confidence)')
    ax3.view_init(elev=elev, azim=azim)
    plt.colorbar(scatter3, ax=ax3, shrink=0.5, label='Max Weight')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    # plt.show()
    
    return {'token_assignments': token_assignments, 'entropy': entropy, 'max_weight': max_weight}


def visualize_multi_head_assignments(
    data: AttentionData,
    heads: Optional[List[int]] = None,
    include_mean: bool = True,
    figsize_per_plot: Tuple[float, float] = (5, 4),
    point_size: float = 1.0,
    elev: float = 30,
    azim: float = 45,
    save_path: Optional[str] = None,
    alpha: float = 0.7
):
    """
    Visualize token assignments for multiple heads side by side.
    
    Args:
        data: AttentionData object
        heads: List of head indices (default: all heads)
        include_mean: Whether to include mean of heads
        figsize_per_plot: Size per subplot
        point_size: Scatter point size
        elev, azim: 3D view angles
        save_path: Path to save figure
        alpha: Point transparency
    """
    if heads is None:
        heads = list(range(data.num_heads))
    
    plots = heads.copy()
    if include_mean:
        plots.append('mean')
    
    n_plots = len(plots)
    n_cols = min(4, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows))
    
    coords = data.coords
    
    for idx, head in enumerate(plots):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
        
        S = get_head_assignments(data, head)
        token_assignments = np.argmax(S, axis=-1)
        
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2],
            c=token_assignments, cmap='tab20', s=point_size, alpha=alpha
        )
        
        head_str = f"Head {head}" if isinstance(head, int) else "Mean"
        ax.set_title(head_str)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=elev, azim=azim)
    
    plt.suptitle('Token Assignments Across Heads', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    # plt.show()


def visualize_per_token_assignments(
    data: AttentionData,
    head: Union[int, str] = 'mean',
    tokens: Optional[List[int]] = None,
    figsize_per_plot: Tuple[float, float] = (4, 3.5),
    point_size: float = 2.0,
    elev: float = 30,
    azim: float = 45,
    save_path: Optional[str] = None,
    alpha: float = 0.7
):
    """
    Visualize soft assignment weights for individual tokens.
    
    Args:
        data: AttentionData object
        head: int for specific head, 'mean' for average
        tokens: List of token indices (default: first 12)
        figsize_per_plot: Size per subplot
        point_size: Scatter point size
        elev, azim: 3D view angles
        save_path: Path to save figure
        alpha: Point transparency
    """
    coords = data.coords
    S = get_head_assignments(data, head)  # (N, K)
    K = S.shape[-1]
    
    if tokens is None:
        tokens = list(range(min(12, K)))
    
    n_tokens = len(tokens)
    n_cols = min(4, n_tokens)
    n_rows = (n_tokens + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows))
    
    head_str = f"Head {head}" if isinstance(head, int) else f"{head.capitalize()}"
    
    for idx, token_id in enumerate(tokens):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
        
        weights = S[:, token_id]
        
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2],
            c=weights, cmap='hot', s=point_size, alpha=alpha,
            vmin=0, vmax=max(0.01, weights.max())
        )
        ax.set_title(f'Token {token_id}\nmax={weights.max():.3f}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=elev, azim=azim)
        plt.colorbar(scatter, ax=ax, shrink=0.5)
    
    plt.suptitle(f'Per-Token Soft Assignments - {head_str}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    # plt.show()


def visualize_token_attention(
    data: AttentionData,
    head: Union[int, str, None] = None,
    figsize: Tuple[int, int] = (14, 12),
    save_path: Optional[str] = None,
    annotate: bool = False
):
    """
    Visualize token-to-token attention.
    
    Args:
        data: AttentionData object
        head: int for specific head, 'mean' for average, None for all heads grid
        figsize: Figure size
        save_path: Path to save figure
        annotate: Whether to show values in cells (only for small K)
    """
    attn = data.token_attention  # (H, K, K)
    H, K, _ = attn.shape
    
    if head is None:
        # Show all heads in grid
        n_cols = min(4, H)
        n_rows = (H + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if H == 1:
            axes = np.array([[axes]])
        axes = axes.flatten()
        
        for h in range(H):
            ax = axes[h]
            im = ax.imshow(attn[h], cmap='Blues', aspect='equal', vmin=0, vmax=1)
            ax.set_title(f'Head {h}')
            ax.set_xlabel('Key Token')
            ax.set_ylabel('Query Token')
            plt.colorbar(im, ax=ax, shrink=0.6)
        
        for h in range(H, len(axes)):
            axes[h].axis('off')
        
        plt.suptitle('Token-to-Token Attention (All Heads)', fontsize=14)
    
    else:
        # Single head or mean
        fig, ax = plt.subplots(figsize=(figsize[0] // 2, figsize[1] // 2))
        
        attn_head = get_head_attention(data, head)
        head_str = f"Head {head}" if isinstance(head, int) else f"{head.capitalize()}"
        
        im = ax.imshow(attn_head, cmap='Blues', aspect='equal', vmin=0, vmax=1)
        ax.set_xlabel('Key Token')
        ax.set_ylabel('Query Token')
        ax.set_title(f'Token-to-Token Attention - {head_str}')
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        if annotate and K <= 16:
            for i in range(K):
                for j in range(K):
                    color = 'white' if attn_head[i, j] > 0.5 else 'black'
                    ax.text(j, i, f'{attn_head[i, j]:.2f}', ha='center', va='center',
                            fontsize=6, color=color)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    # plt.show()


def visualize_attention_statistics(
    data: AttentionData,
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None
):
    """
    Comprehensive attention statistics visualization.
    """
    attn = data.token_attention  # (H, K, K)
    S = data.slice_assignments   # (H, N, K)
    H, K, _ = attn.shape
    N = S.shape[1]
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # 1. Mean attention across heads
    mean_attn = attn.mean(axis=0)
    ax1 = axes[0, 0]
    im1 = ax1.imshow(mean_attn, cmap='Blues', aspect='equal')
    ax1.set_title('Mean Attention Across Heads')
    ax1.set_xlabel('Key Token')
    ax1.set_ylabel('Query Token')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # 2. Attention entropy per head
    entropy_per_head = -np.sum(attn * np.log(attn + 1e-10), axis=-1).mean(axis=-1)
    ax2 = axes[0, 1]
    bars = ax2.bar(range(H), entropy_per_head, color='steelblue')
    ax2.axhline(y=np.log(K), color='r', linestyle='--', label=f'Max (log K)')
    ax2.set_xlabel('Head')
    ax2.set_ylabel('Mean Entropy')
    ax2.set_title('Attention Entropy per Head')
    ax2.legend()
    
    # 3. Self vs cross attention
    diagonal = np.array([np.diag(attn[h]).mean() for h in range(H)])
    off_diag = np.array([(attn[h].sum() - np.diag(attn[h]).sum()) / (K*K - K) for h in range(H)])
    
    ax3 = axes[0, 2]
    x = np.arange(H)
    width = 0.35
    ax3.bar(x - width/2, diagonal, width, label='Self (diagonal)', color='steelblue')
    ax3.bar(x + width/2, off_diag, width, label='Cross (off-diag)', color='coral')
    ax3.set_xlabel('Head')
    ax3.set_ylabel('Mean Weight')
    ax3.set_title('Self vs Cross Attention')
    ax3.legend()
    
    # 4. Token importance (attention received)
    token_importance = attn.sum(axis=(0, 1)) / H  # Average over heads
    ax4 = axes[1, 0]
    ax4.bar(range(K), token_importance, color='steelblue')
    ax4.axhline(y=1.0, color='r', linestyle='--', label='Uniform')
    ax4.set_xlabel('Token ID')
    ax4.set_ylabel('Importance')
    ax4.set_title('Token Importance (Attention Received)')
    ax4.legend()
    
    # 5. Nodes per token (based on hard assignment, mean over heads)
    nodes_per_token = np.zeros((H, K))
    for h in range(H):
        hard_assign = np.argmax(S[h], axis=-1)
        for k in range(K):
            nodes_per_token[h, k] = (hard_assign == k).sum()
    
    ax5 = axes[1, 1]
    mean_nodes = nodes_per_token.mean(axis=0)
    std_nodes = nodes_per_token.std(axis=0)
    ax5.bar(range(K), mean_nodes, yerr=std_nodes, color='steelblue', capsize=2)
    ax5.axhline(y=N/K, color='r', linestyle='--', label=f'Uniform ({N/K:.0f})')
    ax5.set_xlabel('Token ID')
    ax5.set_ylabel('Number of Nodes')
    ax5.set_title('Nodes per Token (Hard Assignment)')
    ax5.legend()
    
    # 6. Slice assignment entropy per head
    slice_entropy = -np.sum(S * np.log(S + 1e-10), axis=-1).mean(axis=-1)  # (H,)
    ax6 = axes[1, 2]
    ax6.bar(range(H), slice_entropy, color='steelblue')
    ax6.axhline(y=np.log(K), color='r', linestyle='--', label=f'Max (log K)')
    ax6.set_xlabel('Head')
    ax6.set_ylabel('Mean Entropy')
    ax6.set_title('Slice Assignment Entropy per Head')
    ax6.legend()
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    # plt.show()


def visualize_token_centers(
    data: AttentionData,
    head: Union[int, str] = 'mean',
    figsize: Tuple[int, int] = (12, 10),
    point_size: float = 1.0,
    center_size: float = 200,
    elev: float = 30,
    azim: float = 45,
    save_path: Optional[str] = None,
    show_labels: bool = True
):
    """
    Visualize token weighted centroids on mesh.
    """
    coords = data.coords
    S = get_head_assignments(data, head)  # (N, K)
    K = S.shape[-1]
    
    # Compute centroids
    centroids = []
    for k in range(K):
        weights = S[:, k:k+1]
        weights_norm = weights / (weights.sum() + 1e-10)
        centroid = (coords * weights_norm).sum(axis=0)
        centroids.append(centroid)
    centroids = np.stack(centroids)
    
    # Hard assignments for coloring
    token_assignments = np.argmax(S, axis=-1)
    
    head_str = f"Head {head}" if isinstance(head, int) else f"{head.capitalize()}"
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Mesh nodes
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=token_assignments, cmap='tab20', s=point_size, alpha=0.3
    )
    
    # Token centroids
    for k in range(K):
        ax.scatter(
            centroids[k, 0], centroids[k, 1], centroids[k, 2],
            c=[plt.cm.tab20(k / K)], s=center_size, marker='*',
            edgecolors='black', linewidths=1
        )
        if show_labels:
            ax.text(centroids[k, 0], centroids[k, 1], centroids[k, 2],
                    f'{k}', fontsize=8, ha='center')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Token Centroids - {head_str}')
    ax.view_init(elev=elev, azim=azim)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    # plt.show()
    
    return centroids


def visualize_head_agreement(
    data: AttentionData,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
):
    """
    Visualize how much heads agree on token assignments.
    """
    S = data.slice_assignments  # (H, N, K)
    H, N, K = S.shape
    
    # Hard assignments per head
    hard_assign = np.argmax(S, axis=-1)  # (H, N)
    
    # Agreement: for each node, how many heads agree with the majority assignment
    from scipy import stats
    modes, counts = stats.mode(hard_assign, axis=0, keepdims=False)
    agreement_ratio = counts / H  # (N,)
    
    coords = data.coords
    
    fig = plt.figure(figsize=figsize)
    
    # Plot 1: Agreement ratio on mesh
    ax1 = fig.add_subplot(121, projection='3d')
    scatter = ax1.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=agreement_ratio, cmap='RdYlGn', s=1, alpha=0.7,
        vmin=0, vmax=1
    )
    ax1.set_title('Head Agreement Ratio\n(Green=All Agree, Red=Disagree)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    plt.colorbar(scatter, ax=ax1, shrink=0.5)
    
    # Plot 2: Histogram of agreement
    ax2 = fig.add_subplot(122)
    ax2.hist(agreement_ratio, bins=20, color='steelblue', edgecolor='black')
    ax2.axvline(x=agreement_ratio.mean(), color='r', linestyle='--',
                label=f'Mean: {agreement_ratio.mean():.2f}')
    ax2.set_xlabel('Agreement Ratio')
    ax2.set_ylabel('Number of Nodes')
    ax2.set_title('Distribution of Head Agreement')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    # plt.show()
    
    return agreement_ratio


# ============================================================================
# Convenience Functions
# ============================================================================

def run_all_visualizations(
    data: AttentionData,
    save_dir: str,
    heads_to_show: Optional[List[int]] = None
):
    """
    Run all visualizations and save to directory.
    
    Args:
        data: AttentionData object
        save_dir: Directory to save figures
        heads_to_show: Specific heads to visualize (default: [0, 1, 2, 3] + mean)
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    if heads_to_show is None:
        heads_to_show = list(range(min(4, data.num_heads)))
    
    print(f"Running visualizations for sample: {data.sample_id}")
    print(f"  Nodes: {data.num_nodes}, Heads: {data.num_heads}, Tokens: {data.num_tokens}")
    print(f"  Saving to: {save_dir}")
    
    # 1. Multi-head token assignments
    print("\n1. Multi-head token assignments...")
    visualize_multi_head_assignments(
        data, heads=heads_to_show, include_mean=True,
        save_path=os.path.join(save_dir, "multi_head_assignments.png")
    )
    
    # 2. Mean assignments detailed
    print("2. Mean assignments (detailed)...")
    visualize_token_assignments_3d(
        data, head='mean',
        save_path=os.path.join(save_dir, "mean_assignments_detailed.png")
    )
    
    # 3. Per-token assignments (mean)
    print("3. Per-token assignments (mean)...")
    visualize_per_token_assignments(
        data, head='mean',
        save_path=os.path.join(save_dir, "per_token_mean.png")
    )
    
    # 4. Token attention (all heads)
    print("4. Token attention (all heads)...")
    visualize_token_attention(
        data, head=None,
        save_path=os.path.join(save_dir, "attention_all_heads.png")
    )
    
    # 5. Token attention (mean)
    print("5. Token attention (mean)...")
    visualize_token_attention(
        data, head='mean',
        save_path=os.path.join(save_dir, "attention_mean.png")
    )
    
    # 6. Attention statistics
    print("6. Attention statistics...")
    visualize_attention_statistics(
        data,
        save_path=os.path.join(save_dir, "attention_statistics.png")
    )
    
    # 7. Token centers (mean)
    print("7. Token centers (mean)...")
    visualize_token_centers(
        data, head='mean',
        save_path=os.path.join(save_dir, "token_centers_mean.png")
    )
    
    # 8. Head agreement
    print("8. Head agreement...")
    visualize_head_agreement(
        data,
        save_path=os.path.join(save_dir, "head_agreement.png")
    )
    
    # 9. Per-head detailed (for first few heads)
    for h in heads_to_show[:3]:
        print(f"9.{h}. Head {h} detailed...")
        visualize_token_assignments_3d(
            data, head=h,
            save_path=os.path.join(save_dir, f"assignments_head{h}.png")
        )
        visualize_per_token_assignments(
            data, head=h,
            save_path=os.path.join(save_dir, f"per_token_head{h}.png")
        )
    
    print(f"\nDone! All visualizations saved to {save_dir}")



collection = AttentionDataCollection.load('attention_data.pkl')
data = collection[0]  # Visualize first sample
run_all_visualizations(data, save_dir='visualizations/sample_0')