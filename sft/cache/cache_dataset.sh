#!/bin/bash
set -euo pipefail

# Export already-converted training JSONs as ms-swift Arrow caches.
#
# Inputs : ${JSON_DIR}/<dataset>.json   (produced by DataProcess/tool/convert_vln.py
#          or convert_dagger.py; see README section 3)
# Outputs: ${CACHE_DIR}/<dataset>/train/
#
# Usage:
#   bash cache_dataset.sh                    # cache all five datasets
#   bash cache_dataset.sh SFT/R2R SFT/RxR    # cache selected datasets

MODEL="/path/to/models/Qwen3-VL-4B-Instruct"
CACHE_DIR="/path/to/cache_data"
JSON_DIR="${JSON_DIR:-${CACHE_DIR}/annotations}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

ALL_DATASETS=(SFT/R2R SFT/RxR SFT/ScaleVLN DAgger/R2R DAgger/RxR)
if [ $# -eq 0 ]; then
    DATASETS=("${ALL_DATASETS[@]}")
else
    DATASETS=("$@")
fi

for DATASET in "${DATASETS[@]}"; do
    case "${DATASET}" in
        SFT/R2R|SFT/RxR|SFT/ScaleVLN|DAgger/R2R|DAgger/RxR)
            ;;
        *)
            echo "Unknown dataset: ${DATASET}" >&2
            echo "Available: ${ALL_DATASETS[*]}" >&2
            exit 1
            ;;
    esac

    JSON="${JSON_DIR}/${DATASET}.json"
    if [ ! -f "${JSON}" ]; then
        echo "Missing converted JSON: ${JSON}" >&2
        echo "Run the corresponding converter first (see README section 3)." >&2
        exit 1
    fi

    echo "=========================================="
    echo "Caching dataset: ${DATASET}"
    echo "Input : ${JSON}"
    echo "Output: ${CACHE_DIR}/${DATASET}"
    echo "=========================================="

    MAX_PIXELS=1003520 \
    python "${SCRIPT_DIR}/fast_cache_dataset.py" \
        --model "${MODEL}" \
        --dataset "${JSON}" \
        --dataset_num_proc 64 \
        --to_cached_dataset true \
        --output_dir "${CACHE_DIR}/${DATASET}"

    echo "Done: ${DATASET}"
    echo ""
done

echo "All done! Cached datasets are in: ${CACHE_DIR}"
