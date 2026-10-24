#!/usr/bin/env bash
# Submit the three design-change sweep variants at once, one GPU job each.
# With the running reference they form a 2 x 2 design, so each change can be
# judged on its own and together:
#
#                          change path: full (0.55 M)   change path: small (0.15 M)
#   dropout 0.1, wd 1e-5   reference (..._nodes.sh)     ..._small.sh
#   dropout 0.2, wd 0.05   ..._regularized.sh           ..._small_regularized.sh
#
# Extra arguments are forwarded to every sbatch call.
#   bash submit_mesh_impact_history_1704_design_change_sweep.sh
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
for variant in small regularized small_regularized; do
    bash "${PROJECT_DIR}/submit_mesh_impact_history_1704_design_change_${variant}.sh" "$@"
done
