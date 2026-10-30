#!/usr/bin/env bash
# SimpleVLN DAPO preset: qwen35_9b. Override NNODES and NGPUS for your cluster.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_dapo.sh" qwen35_9b "$@"
