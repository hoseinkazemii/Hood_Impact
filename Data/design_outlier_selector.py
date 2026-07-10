import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def load_unique_designs(doe_path: str, samples_per_design: int = 50):
    df = pd.read_csv(doe_path)

    design_params = [
        "Stiffener Rib Angle",
        "Stiffener Rib Height",
        "Stiffener Rib Position",
        "Stiffener Rib Width",
    ]

    designs = []
    design_ids = []

    num_designs = len(df) // samples_per_design

    for d in range(num_designs):
        start = d * samples_per_design
        row = df.iloc[start]
        designs.append(row[design_params].values.astype(float))
        design_ids.append(d)

    return np.array(designs), design_ids


def select_most_different_designs(designs, design_ids, num_test=10):

    scaler = StandardScaler()
    X = scaler.fit_transform(designs)

    D = pairwise_distances(X, metric="euclidean")
    mean_dist = D.mean(axis=1)

    sorted_indices = np.argsort(-mean_dist)

    test_indices = sorted_indices[:num_test]
    train_indices = sorted_indices[num_test:]

    test_design_ids = [design_ids[i] for i in test_indices]
    train_design_ids = [design_ids[i] for i in train_indices]

    print("\n=== DESIGN DISTANCE ANALYSIS ===")
    for i in sorted_indices:
        print(f"Design {design_ids[i]:2d} | mean distance = {mean_dist[i]:.3f}")

    print("\nSelected TEST designs:")
    print(sorted(test_design_ids))

    print("\nSelected TRAIN designs:")
    print(sorted(train_design_ids))

    visualize_design_space(X, design_ids, test_indices, train_indices)

    return train_design_ids, test_design_ids


def visualize_design_space(X, design_ids, test_indices, train_indices):

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))

    # Plot train
    plt.scatter(
        X_pca[train_indices, 0],
        X_pca[train_indices, 1],
        marker="o",
        label="Train Designs",
    )

    # Plot test
    plt.scatter(
        X_pca[test_indices, 0],
        X_pca[test_indices, 1],
        marker="s",
        label="Test Designs (Outliers)",
    )

    # Label each point
    for i, design_id in enumerate(design_ids):
        plt.text(
            X_pca[i, 0] + 0.02,
            X_pca[i, 1] + 0.02,
            str(design_id),
            fontsize=9,
        )

    # Plot centroid
    centroid = X_pca.mean(axis=0)
    plt.scatter(
        centroid[0],
        centroid[1],
        marker="X",
        s=200,
        label="Design Space Centroid",
    )

    explained = pca.explained_variance_ratio_.sum() * 100

    plt.title(f"PCA Projection of Design Space (Explained Variance: {explained:.1f}%)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig("design_space_split.png", dpi=300)
    plt.show()


if __name__ == "__main__":

    DOE_PATH = "./DOE_ball_based.csv"

    designs, design_ids = load_unique_designs(DOE_PATH)

    train_designs, test_designs = select_most_different_designs(
        designs,
        design_ids,
        num_test=10
    )