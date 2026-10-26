#!/usr/bin/env bash
# Single-location, geometry-only test (README_one_impact_location.md).
#
# Learns the 12 designs at ONE impact location from the inner frame alone, with
# no impact-location input:
#   --locations 137             the 12 runs at the HIC-variation rank-1 location
#   --mesh-parts inner          headform rows (kept out of pooling) + Hood_Inner;
#                               the outer skin and hinges are dropped
#   --impact-conditioning none  no impact embedding, FiLM or radial prior
#   test design 7 and no validation design: the other 11 designs train (4 and 5
#   included) and, evaluated in eval mode, select the checkpoint
#   small network for 11 samples: width 64, 32 tokens, one latent, decoder and
#   neighborhood layer (0.18 M parameters vs 1.59 M), no dropout
#   250 time points (stride 4), full batch of 11, 3000 epochs (= 3000 steps),
#   lr 1e-3 cosine-annealed to 1e-5
#
# Runs locally on one GPU (about 2 s per epoch on an RTX 5050 laptop GPU).
# Settings come from the environment; extra arguments go to the trainer and win:
#   HOOD_ONE_LOCATION=9 HOOD_ONE_TEST_DESIGNS="2" bash run_one_impact_location.sh
#   bash run_one_impact_location.sh --epochs 10 --wandb-mode disabled
# This Windows Python has protobuf 7, which breaks wandb 0.22; point
# HOOD_PROTOBUF_COMPAT at a folder made with
#   python -m pip install --target <folder> "protobuf>=5,<6"
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${PROJECT_DIR}"

LOCATION=${HOOD_ONE_LOCATION:-137}
read -r -a TEST_DESIGNS <<< "${HOOD_ONE_TEST_DESIGNS:-7}"
EPOCHS=${HOOD_ONE_EPOCHS:-3000}
SEED=${HOOD_ONE_SEED:-42}
TAG="loc${LOCATION}_test$(IFS=-; echo "${TEST_DESIGNS[*]}")"
OUTPUT_DIR=${HOOD_ONE_OUTPUT_DIR:-${PROJECT_DIR}/runs/one_impact_location/$(date +%Y%m%d_%H%M%S)_${TAG}}
if [[ -n "${HOOD_PROTOBUF_COMPAT:-}" ]]; then
    export PYTHONPATH="${HOOD_PROTOBUF_COMPAT}"
fi
mkdir -p "$(dirname "${OUTPUT_DIR}")"

printf 'Location %s | test designs %s | no validation design | %s epochs | seed %s\n' \
    "${LOCATION}" "${TEST_DESIGNS[*]}" "${EPOCHS}" "${SEED}"
printf 'Output: %s\n' "${OUTPUT_DIR}"
exec "${PYTHON:-python}" -u train_mesh_impact_history.py \
    --data-format euroncap1704 --locations "${LOCATION}" --mesh-parts inner \
    --test-designs "${TEST_DESIGNS[@]}" --val-designs \
    --impact-conditioning none --impactor-nodes 286 --pool-impactor-nodes false \
    --width 64 --num-heads 4 --num-latents 32 --latent-layers 1 --temporal-layers 1 \
    --neighborhood-layers 1 --neighborhood-k 256 --dropout 0 \
    --time-subsample-stride 4 --batch-size 11 --epochs "${EPOCHS}" --lr 0.001 \
    --weight-decay 0.00001 --seed "${SEED}" --device cuda \
    --wandb-mode "${WANDB_MODE:-online}" --run-name "one_location_${TAG}" \
    --output-dir "${OUTPUT_DIR}" "$@"
