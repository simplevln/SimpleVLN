#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export VLN_DATA_ROOT="${VLN_DATA_ROOT:-./data/vln_data}"

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet

# Force single-process mode (unset distributed env vars from cluster)
unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT

PYTHON=${PYTHON:-python}

# ──────────────── Dataset paths ────────────────
# R2R
R2R_DATA_PATH=${R2R_DATA_PATH:-${VLN_DATA_ROOT}/eval_data/r2r/train/train.json.gz}
R2R_GT_ANNOTATIONS_PATH=${R2R_GT_ANNOTATIONS_PATH:-${VLN_DATA_ROOT}/eval_data/r2r/train/train_gt.json.gz}

# RxR
RXR_DATA_PATH=${RXR_DATA_PATH:-${VLN_DATA_ROOT}/eval_data/rxr/train/train_guide_en.json.gz}
RXR_GT_ANNOTATIONS_PATH=${RXR_GT_ANNOTATIONS_PATH:-${VLN_DATA_ROOT}/eval_data/rxr/train/train_guide_gt.json.gz}

# ──────────────── DAgger config ────────────────
DAGGER_UPDATE_SIZE=${DAGGER_UPDATE_SIZE:-1000000}
DAGGER_P=${DAGGER_P:-0}             # 0 = always use model; >0 = mix expert with probability p^data_it
DAGGER_DATA_IT=${DAGGER_DATA_IT:-3}
DAGGER_BO_N=${DAGGER_BO_N:-4}           # pass@N: each trajectory runs up to N times, save on first success
PARALLEL_ENVS=${PARALLEL_ENVS:-64}

# ──────────────── vLLM server ────────────────
MAX_ACTION_HISTORY=${MAX_ACTION_HISTORY:-200}
VLLM_HOST=${VLLM_HOST:-localhost}
VLLM_PORT_START=${VLLM_PORT_START:-22001}
VLLM_NUM_SERVERS=${VLLM_NUM_SERVERS:-4}
VLLM_MODEL_NAME=${VLLM_MODEL_NAME:-qwen3vl}

# ──────────────── Output ────────────────
OUTPUT_ROOT=${OUTPUT_ROOT:-./data/dagger}
DAGGER_DATASET=${DAGGER_DATASET:-R2R}
LOG_FILE="${OUTPUT_ROOT}/dagger_${DAGGER_DATASET}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${OUTPUT_ROOT}"

run_dagger_collect() {
    local dagger_dataset="$1"
    local dagger_data_path="$2"
    local dagger_gt_annotations_path="$3"
    local dagger_output_path="${OUTPUT_ROOT}/${dagger_dataset}"

    mkdir -p "${dagger_output_path}"

    "${PYTHON}" vln/dagger.py \
        --vllm_host "${VLLM_HOST}" \
        --vllm_port_start "${VLLM_PORT_START}" \
        --vllm_num_servers "${VLLM_NUM_SERVERS}" \
        --vllm_model_name "${VLLM_MODEL_NAME}" \
        --parallel_envs "${PARALLEL_ENVS}" \
        --max-action-history "${MAX_ACTION_HISTORY}" \
        --dagger_dataset "${dagger_dataset}" \
        --dagger_data_path "${dagger_data_path}" \
        --dagger_update_size "${DAGGER_UPDATE_SIZE}" \
        --dagger_p "${DAGGER_P}" \
        --dagger_data_it "${DAGGER_DATA_IT}" \
        --dagger_bo_n "${DAGGER_BO_N}" \
        --dagger_output_path "${dagger_output_path}" \
        --dagger_gt_annotations_path "${dagger_gt_annotations_path}" \
        --dagger-save-model-trace
}

case "$DAGGER_DATASET" in
    R2R) run_dagger_collect R2R "$R2R_DATA_PATH" "$R2R_GT_ANNOTATIONS_PATH" ;;
    RxR) run_dagger_collect RxR "$RXR_DATA_PATH" "$RXR_GT_ANNOTATIONS_PATH" ;;
    *) echo "DAGGER_DATASET must be R2R or RxR" >&2; exit 1 ;;
esac 2>&1 | tee -a "$LOG_FILE"
