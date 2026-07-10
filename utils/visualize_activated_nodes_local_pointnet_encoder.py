import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# USER PARAMETERS (CHANGE THESE)
# ============================================================

COORD_FILE = "HoodImpactor_624_COORD.csv"
DISP_FILE  = "HoodImpactor_624_U.csv"

# Pick timestep index (early impact recommended)
TIMESTEP_INDEX = 10

# Radius used in local PointNet encoder
LOCAL_RADIUS = 0.5   # <-- CHANGE THIS (0.2, 0.4, 0.6, 1.0, ...)

# Indentor position in XY (MUST MATCH TRAINING COORDS)
INDENTOR_XY = np.array([0.0, 0.0])  # <-- SET THIS CORRECTLY

# Visualization options
SHOW_ONLY_LOCAL = False   # True = only red points
POINT_SIZE_BASE = 2
POINT_SIZE_LOCAL = 8

# ============================================================
# LOAD INITIAL GEOMETRY
# ============================================================

coord_df = pd.read_csv(COORD_FILE).set_index("Node")
X0 = coord_df[["X1", "X2", "X3"]].values

# ============================================================
# LOAD DISPLACEMENT DATA
# ============================================================

disp_df = pd.read_csv(DISP_FILE).set_index("Node")
times = np.sort(disp_df["Time"].unique())

timestep = times[TIMESTEP_INDEX]
disp_t = disp_df[disp_df["Time"] == timestep].loc[coord_df.index]

U = disp_t[["U1", "U2", "U3"]].values
X_def = X0 + U
u_mag = np.linalg.norm(U, axis=1)

# ============================================================
# COMPUTE LOCAL MASK (MATCHES MODEL LOGIC)
# ============================================================

XY = X0[:, :2]                        # use UNDEFORMED geometry
rel_xy = XY - INDENTOR_XY[None, :]
dist_xy = np.linalg.norm(rel_xy, axis=1)
local_mask = dist_xy < LOCAL_RADIUS

num_local = local_mask.sum()
num_total = len(local_mask)

print(
    f"[LOCAL REGION DEBUG]\n"
    f"  Radius              : {LOCAL_RADIUS}\n"
    f"  Local nodes          : {num_local} / {num_total}\n"
    f"  Percentage of mesh   : {100 * num_local / num_total:.2f}%"
)

# ============================================================
# VISUALIZATION
# ============================================================

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

# Base mesh
if not SHOW_ONLY_LOCAL:
    sc = ax.scatter(
        X_def[:, 0],
        X_def[:, 1],
        X_def[:, 2],
        c=u_mag,
        cmap="viridis",
        s=POINT_SIZE_BASE,
        alpha=0.25,
        linewidths=0
    )

# Local neighborhood overlay
ax.scatter(
    X_def[local_mask, 0],
    X_def[local_mask, 1],
    X_def[local_mask, 2],
    c="red",
    s=POINT_SIZE_LOCAL,
    label="Local neighborhood",
    linewidths=0
)

# Indentor marker (projected above mesh)
ax.scatter(
    INDENTOR_XY[0],
    INDENTOR_XY[1],
    X_def[:, 2].max(),
    c="black",
    s=120,
    marker="x",
    label="Indentor"
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_title(
    f"Local neighborhood (r = {LOCAL_RADIUS})\n"
    f"t = {timestep:.6f} s | {num_local} nodes ({100*num_local/num_total:.2f}%)"
)

# Underside view (stiffeners visible)
ax.view_init(elev=-90, azim=0)

ax.set_box_aspect([
    np.ptp(X_def[:, 0]),
    np.ptp(X_def[:, 1]),
    np.ptp(X_def[:, 2]),
])

if not SHOW_ONLY_LOCAL:
    cbar = fig.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label("Displacement magnitude")

ax.legend()
plt.tight_layout()
plt.savefig(
    f"local_region_radius_{LOCAL_RADIUS:.2f}_t_{timestep:.6f}.png",
    dpi=300
)
plt.show()
