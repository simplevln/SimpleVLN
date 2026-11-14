#!/bin/bash
# Convert and cache the five datasets in SimpleVLN_SFT_Data.
# Extract image shards into SFT/<dataset>/ or DAgger/<dataset>/ first.
# Extract annotation archives into SFT/ or DAgger/ (they include <dataset>/).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

DATASETS=(
    SFT/R2R
    SFT/RxR
    SFT/ScaleVLN
    DAgger/R2R
    DAgger/RxR
)

echo "Will cache ${#DATASETS[@]} datasets"
echo ""

bash "${SCRIPT_DIR}/cache_dataset.sh" "${DATASETS[@]}"
