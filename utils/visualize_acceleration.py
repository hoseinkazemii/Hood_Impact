import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
experiment_number = 13
csv_path = f"HoodImpactor_{experiment_number}_SAE1000.csv"
df = pd.read_csv(csv_path)

# Extract columns
time = df["Time"]
A1 = df["A1"]
A2 = df["A2"]
A3 = df["A3"]
A_mm_s2 = df["A(mm/s2)"]
A_g = df["A(in g)"]

# ---- Plot 1: Acceleration magnitude ----
plt.figure()
plt.plot(time, A_mm_s2, label="Acceleration (mm/s²)")
plt.xlabel("Time")
plt.ylabel("Acceleration")
plt.title("Resultant Acceleration vs Time")
plt.legend()
plt.grid(True)
plt.savefig(f"acceleration_magnitude_{experiment_number}.png")
# plt.show()

# ---- Plot 2: Component-wise acceleration ----
plt.figure()
plt.plot(time, A1, label="A1")
plt.plot(time, A2, label="A2")
plt.plot(time, A3, label="A3")
plt.xlabel("Time")
plt.ylabel("Acceleration (mm/s²)")
plt.title("Acceleration Components vs Time")
plt.legend()
plt.grid(True)
plt.savefig(f"acceleration_components_{experiment_number}.png")
# plt.show()
