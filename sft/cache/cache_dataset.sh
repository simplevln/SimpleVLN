#!/bin/bash
set -euo pipefail

# Usage: bash cache_dataset.sh <dataset_name1> [dataset_name2] ...
# Example: bash cache_dataset.sh SFT/R2R SFT/RxR
# Example: bash cache_dataset.sh DAgger/R2R DAgger/RxR

MODEL="/path/to/models/Qwen3-VL-4B-Instruct"
DATA_ROOT="/path/to/SimpleVLN_SFT_Data"
CACHE_DIR="/path/to/cache_data"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONVERT_DIR="${SCRIPT_DIR}/../../DataProcess/tool"

if [ $# -eq 0 ]; then
    echo "Usage: bash cache_dataset.sh <dataset_name1> [dataset_name2] ..."
    echo "Available datasets: SFT/R2R SFT/RxR SFT/ScaleVLN DAgger/R2R DAgger/RxR"
    exit 1
fi

for DATASET in "$@"; do
    case "${DATASET}" in
        SFT/R2R|SFT/RxR|SFT/ScaleVLN)
            CONVERTER="convert_vln.py"
            CONVERT_ARGS=(--task_type vln)
            ;;
        DAgger/R2R|DAgger/RxR)
            CONVERTER="convert_dagger.py"
            CONVERT_ARGS=(--mode atomic)
            ;;
        *)
            echo "Unknown dataset: ${DATASET}" >&2
            exit 1
            ;;
    esac

    ANNOTATION_PATH="${DATA_ROOT}/${DATASET}/annotations.json"
    TRAIN_ANNOTATION="${CACHE_DIR}/annotations/${DATASET}.json"
    if [ ! -f "${ANNOTATION_PATH}" ]; then
        echo "Missing ${ANNOTATION_PATH}; extract the annotation archive first." >&2
        exit 1
    fi

    echo "=========================================="
    echo "Caching dataset: ${DATASET}"
    echo "Output: ${CACHE_DIR}/${DATASET}"
    echo "=========================================="

    # Published annotations contain trajectories; convert them to training samples.
    # An absolute DATA_ROOT writes local image paths into the generated samples.
    python "${CONVERT_DIR}/${CONVERTER}" \
        --annotation_path "${ANNOTATION_PATH}" \
        --image_root "${DATA_ROOT}/${DATASET}" \
        --output_path "${TRAIN_ANNOTATION}" \
        --max_history 8 \
        --seed 0 \
        "${CONVERT_ARGS[@]}"

    MAX_PIXELS=1003520 \
    python "${SCRIPT_DIR}/fast_cache_dataset.py" \
        --model "${MODEL}" \
        --dataset "${TRAIN_ANNOTATION}" \
        --dataset_num_proc 64 \
        --to_cached_dataset true \
        --output_dir "${CACHE_DIR}/${DATASET}"

    echo "Done: ${DATASET}"
    echo ""
done

echo "All done! Cached datasets are in: ${CACHE_DIR}"
