import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Load initial geometry
# -------------------------
coord_file = "HoodImpactor_624_COORD.csv"
coord_df = pd.read_csv(coord_file)

coord_df = coord_df.set_index("Node")

X0 = coord_df[["X1", "X2", "X3"]].values


# -------------------------
# Load displacement data
# -------------------------
disp_file = "HoodImpactor_624_U.csv"
disp_df = pd.read_csv(disp_file)


# Ensure correct indexing
disp_df = disp_df.set_index("Node")


# Pick a timestep to visualize
timestep = disp_df["Time"].unique()[10]  # change index as desired

disp_t = disp_df[disp_df["Time"] == timestep]

# Ensure node alignment
disp_t = disp_t.loc[coord_df.index]

U = disp_t[["U1", "U2", "U3"]].values

# Deformed coordinates
X_def = X0 + U


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

# Color by displacement magnitude
u_mag = np.linalg.norm(U, axis=1)

sc = ax.scatter(
    X_def[:, 0],
    X_def[:, 1],
    X_def[:, 2],
    c=u_mag,
    cmap="viridis",
    s=2
)

ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("X3")
ax.set_title(f"Displaced Mesh at t = {timestep:.6f} s")

# Underside view (stiffeners visible)
ax.view_init(elev=-90, azim=0)

# Preserve geometry proportions
ax.set_box_aspect([
    np.ptp(X_def[:, 0]),
    np.ptp(X_def[:, 1]),
    np.ptp(X_def[:, 2]),
])

fig.colorbar(sc, ax=ax, label="Displacement magnitude")

plt.tight_layout()
plt.savefig(f"displaced_mesh_t_{timestep:.6f}.png")
plt.show()


times = np.sort(disp_df["Time"].unique())

for t in times:
    disp_t = disp_df[disp_df["Time"] == t].loc[coord_df.index]
    U = disp_t[["U1", "U2", "U3"]].values
    X_def = X0 + U
    u_mag = np.linalg.norm(U, axis=1)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        X_def[:, 0],
        X_def[:, 1],
        X_def[:, 2],
        c=u_mag,
        cmap="viridis",
        s=1
    )

    ax.view_init(elev=-90, azim=0)
    ax.set_box_aspect([
        np.ptp(X_def[:, 0]),
        np.ptp(X_def[:, 1]),
        np.ptp(X_def[:, 2]),
    ])

    ax.set_title(f"t = {t:.6f} s")
    plt.tight_layout()
    plt.savefig(f"displacement_t_{t:.6f}.png")
    # plt.show()
