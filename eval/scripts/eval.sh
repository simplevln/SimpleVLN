#!/bin/bash
# pass@4 eval on R2R and RxR val_unseen.
# Requires vLLM servers running on ports 22001-22008 (see scripts/serve_vllm.sh).
#
# Usage:
#   EXP_NAME=my-run bash scripts/eval.sh
#     → results/<EXP_NAME>/r2r_pass4/  and  results/<EXP_NAME>/rxr_pass4/

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="${PYTHON:-python}"
EXP_NAME="${EXP_NAME:?set EXP_NAME to a tag for the output dir}"
RESULT_ROOT="${RESULT_ROOT:-./results}"

VLLM_HOST="${VLLM_HOST:-http://localhost}"
VLLM_PORT_START="${VLLM_PORT_START:-22001}"
VLLM_NUM_SERVERS="${VLLM_NUM_SERVERS:-8}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-qwen3vl}"

SPLIT_NUM="${SPLIT_NUM:-64}"
PASS_K="${PASS_K:-4}"
FORWARD_DISTANCE="${FORWARD_DISTANCE:-25}"
TURN_ANGLE="${TURN_ANGLE:-15}"
MAX_ACTION_HISTORY="${MAX_ACTION_HISTORY:-200}"

run_one() {
    local cfg="$1"
    local out="$2"
    local server_args=(--vllm_host "${VLLM_HOST}" --vllm_port_start "${VLLM_PORT_START}" --vllm_num_servers "${VLLM_NUM_SERVERS}")
    if [[ -n "${VLLM_BASE_URL:-}" ]]; then
        server_args=(--vllm_base_url "${VLLM_BASE_URL}")
    fi
    "${PYTHON}" vln/eval.py \
        --exp-config "${cfg}" \
        "${server_args[@]}" \
        --vllm_model_name "${VLLM_MODEL_NAME}" \
        --split-num "${SPLIT_NUM}" \
        --pass-k "${PASS_K}" \
        --result-path "${out}" \
        --forward-distance "${FORWARD_DISTANCE}" \
        --turn-angle "${TURN_ANGLE}" \
        --max-action-history "${MAX_ACTION_HISTORY}"
}

run_one config/vln_r2r.yaml "${RESULT_ROOT}/${EXP_NAME}/r2r_pass${PASS_K}"
run_one config/vln_rxr.yaml "${RESULT_ROOT}/${EXP_NAME}/rxr_pass${PASS_K}"
