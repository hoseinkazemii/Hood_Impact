"""A toy local-edit campaign on disk, in the layout of Data/HoodImpact_LocalEdits.

Design 6 ("D6") is solved at locations 1-3 and the edits N1-N4 at locations 1-2.
The deck's inner panel is nodes 20-23; the rest is headform and outer skin.
With TRAINING_ARGS, N4 is the test design, N3 validation and N1-N2 training.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DECK = """*HEADING
*NODE
1, 0.0, 0.0, 90.0
2, 1.0, 0.0, 90.0
20, 0.0, 0.0, 0.0
10, 0.0, 0.0, 5.0
21, 1.0, 0.0, 0.0
3, 0.0, 1.0, 90.0
11, 1.0, 0.0, 5.0
22, 1.0, 1.0, 0.0
12, 1.0, 1.0, 5.0
23, 0.0, 1.0, 0.0
*ELEMENT, TYPE=S3R
100, 1, 2, 3
*ELEMENT, TYPE=S3R
200, 10, 11, 12
*ELEMENT, TYPE=S4
300, 20, 21, 22, 23
*ELSET, ELSET="Rigid Body1_2-1-3"
100
*ELSET, ELSET="Hood_Outer-1-2"
200
*ELSET, ELSET="Hood_Inner-1-2"
300
"""
INNER_PANEL = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
RAW_TIMES = np.linspace(0.0, 0.007, 8)
# label, x, y, z_design6, dz
CHANGES = {"N1": [(21, 1.0, 0.0, 0.0, 2.0)], "N2": [(22, 1.0, 1.0, 0.0, 3.0), (23, 0.0, 1.0, 0.0, 1.0)],
           "N3": [(20, 0.0, 0.0, 0.0, -1.5)], "N4": [(21, 1.0, 0.0, 0.0, -2.5)]}
TRAINING_ARGS = ["--test-designs", "4", "--val-designs", "3", "--batch-size", "2", "--time-subsample-stride", "1",
                 "--width", "16", "--num-heads", "2", "--num-latents", "4", "--latent-layers", "1",
                 "--temporal-layers", "1", "--neighborhood-k", "3", "--fourier-max-frequency-hz", "400",
                 "--device", "cpu", "--wandb-mode", "disabled"]


def curve(design, location):
    shape = np.sin(np.pi * RAW_TIMES / RAW_TIMES[-1])
    return 50.0 * shape * (1.0 + 0.1 * location) + 4.0 * design * shape ** 2


def impact(location):
    return [100.0 * location, 0.5]


def mesh(design):
    """Inner-panel mesh of edit Nn (design 0 = D6)."""
    result = INNER_PANEL.copy()
    for label, *_, dz in CHANGES.get(f"N{design}", ()):
        result[label - 20, 2] += dz
    return result


def write_dataset(root, base_gain=1.0):
    """Write the campaign under ``root``; design 6's curves are scaled by ``base_gain``."""
    root = Path(root)
    for name in ("inp_files", "output_history_acc", "design_changes"):
        (root / name).mkdir(parents=True)
    (root / "inp_files" / "HoodImpact_1.inp").write_text(DECK)
    rows, run = [], 0
    for design, locations in [("D6", (1, 2, 3)), *((name, (1, 2)) for name in CHANGES)]:
        for location in locations:
            run += 1
            x, y = impact(location)
            rows.append({"run": run, "design": design, "location": location, "role": "near",
                         "x_mm": x, "y_mm": y, "dist_to_edit_mm": 10.0 * location})
            number = 0 if design == "D6" else int(design[1:])
            accel = curve(number, location) * (base_gain if design == "D6" else 1.0)
            pd.DataFrame({"Time": RAW_TIMES, "A(in g)": accel}).to_csv(
                root / "output_history_acc" / f"HoodImpact_{run}_SAE1000_interp1000.csv", index=False)
    pd.DataFrame(rows).to_csv(root / "manifest_local_edits.csv", index=False)
    for design, changes in CHANGES.items():
        pd.DataFrame(changes, columns=["label", "x", "y", "z_design6", "dz"]).to_csv(
            root / "design_changes" / f"{design}_changed_nodes.csv", index=False)
    pd.DataFrame({"design": list(CHANGES), "kind": "single", "variant": "S",
                  "dz_max_mm": [2.0, 3.0, 0.0, 0.0], "dz_min_mm": [0.0, 0.0, -1.5, -2.5]}).to_csv(
        root / "local_edit_designs.csv", index=False)
    pd.DataFrame({"design": list(CHANGES), "site": ["s1", "s1", "s2", "s1"]}).to_csv(
        root / "local_edit_doe.csv", index=False)
    return root


def training_args(root, run_dir, *extra):
    """train_mesh_impact_history arguments for the toy campaign; ``extra`` overrides earlier options."""
    root = Path(root)
    return ["--inp-dir", str(root / "inp_files"), "--impact-coords-path", str(root / "manifest_local_edits.csv"),
            "--acceleration-dir", str(root / "output_history_acc"), *TRAINING_ARGS,
            "--output-dir", str(run_dir), *extra]
