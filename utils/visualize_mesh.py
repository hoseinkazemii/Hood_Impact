import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -------------------------
# Load coordinate file
# -------------------------
fname = "HoodImpactor_910_COORD.csv"
df = pd.read_csv(fname)

# Extract coordinates
x = df["X1"].values
y = df["X2"].values
z = df["X3"].values

# -------------------------
# 3D scatter plot
# -------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(x, y, z, s=2)

ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("X3")
ax.set_title(f"3D Node Geometry: {fname}")
ax.view_init(elev=-90, azim=0)
# Equal aspect ratio for correct shape perception
ax.set_box_aspect([
    x.max() - x.min(),
    y.max() - y.min(),
    z.max() - z.min()
])

plt.tight_layout()
plt.savefig(f"hood_shape_3d_{fname.replace('.csv', '')}.png")
plt.show()


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(
    x, y, z,
    c=z, cmap="viridis",
    s=2
)

ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("X3")
ax.set_title(f"3D Node Geometry (colored by X3): {fname}")
ax.view_init(elev=-90, azim=0)

ax.set_box_aspect([
    x.max() - x.min(),
    y.max() - y.min(),
    z.max() - z.min()
])

fig.colorbar(sc, ax=ax, label="X3")
plt.tight_layout()
plt.savefig(f"hood_shape_3d_{fname.replace('.csv', '')}_colored.png")
plt.show()
