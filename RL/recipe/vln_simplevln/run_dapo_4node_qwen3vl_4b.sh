#!/usr/bin/env bash
# SimpleVLN DAPO preset: qwen3vl_4b. Override NNODES and NGPUS for your cluster.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_dapo.sh" qwen3vl_4b "$@"
