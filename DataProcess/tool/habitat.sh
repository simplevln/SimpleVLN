#!/bin/bash
# Install into an existing Python 3.10 conda environment.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
eval "$(conda shell.bash hook)"
CONDA_ENV="${CONDA_ENV:-vln-data}"
conda activate "$CONDA_ENV"
python -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use Python 3.10 for Habitat 0.3.3"'
conda install -y habitat-sim=0.3.3 headless -c conda-forge -c aihabitat
python -m pip install -r "${SCRIPT_DIR}/../requirements.txt"
echo "Habitat 0.3.3 and DataProcess dependencies installed."
