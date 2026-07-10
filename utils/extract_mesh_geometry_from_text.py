import os
import csv

# -----------------------------
# Configuration
# -----------------------------
INPUT_DIR = "."        # directory with mesh_HoodImpactor_*.txt
OUTPUT_DIR = "."       # where COORD CSVs will be written
START_ID = 1
END_ID = 500

HOOD_NODE_ID_MAX = 41000   # <-- critical filter

# -----------------------------
# Helper: parse NODE section
# -----------------------------
def extract_hood_nodes_from_mesh(mesh_file):
    nodes = []

    with open(mesh_file, "r") as f:
        in_node_section = False

        for line in f:
            line = line.strip()

            # Start of NODE section
            if line.upper().startswith("*NODE"):
                in_node_section = True
                continue

            # End of NODE section
            if in_node_section and line.startswith("*"):
                break

            if in_node_section and line:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    node_id = int(parts[0])

                    # -----------------------------
                    # FILTER: keep only hood nodes
                    # -----------------------------
                    if node_id >= HOOD_NODE_ID_MAX:
                        continue

                    x1 = float(parts[1])
                    x2 = float(parts[2])
                    x3 = float(parts[3])

                    nodes.append((node_id, x1, x2, x3))

    return nodes


# -----------------------------
# Main loop
# -----------------------------
for exp_id in range(START_ID, END_ID + 1):
    mesh_fname = f"mesh_HoodImpactor_{exp_id}.txt"
    mesh_path = os.path.join(INPUT_DIR, mesh_fname)

    if not os.path.exists(mesh_path):
        print(f"[WARNING] Missing file: {mesh_fname}")
        continue

    print(f"Processing {mesh_fname}...")

    nodes = extract_hood_nodes_from_mesh(mesh_path)

    if not nodes:
        print(f"[ERROR] No hood nodes found in {mesh_fname}")
        continue

    # Sort by Node ID (ascending)
    nodes.sort(key=lambda x: x[0])

    out_fname = f"HoodImpactor_{exp_id}_COORD.csv"
    out_path = os.path.join(OUTPUT_DIR, out_fname)

    with open(out_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Node", "X1", "X2", "X3"])
        for row in nodes:
            writer.writerow(row)

    print(f"  → wrote {out_fname} ({len(nodes)} hood nodes)")

print("Done.")
