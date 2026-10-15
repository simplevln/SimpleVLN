#!/bin/bash
# Launch N vLLM servers on sequential ports, one per GPU (or per TP group).
# Pairs with eval.sh, which uses --vllm_port_start 22001
# --vllm_num_servers <N>.
#
# Usage:
#   MODEL_PATH=/path/to/hf-ckpt bash serve_vllm.sh [NUM_GPUS] [TP_SIZE]
#   NUM_GPUS default 8, TP_SIZE default 1  →  8 servers on 22001..22008
#   NUM_GPUS=8 TP_SIZE=2                    →  4 servers, each on 2 GPUs
#
# Server PIDs are written to the run log directory.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to a HuggingFace-format checkpoint dir}"
MODEL_NAME="${MODEL_NAME:-qwen3vl}"
BASE_PORT="${BASE_PORT:-22001}"
LOG_DIR="${LOG_DIR:-$(dirname "$0")/../logs/vllm}"

PYTHON="${PYTHON:-python}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
GPU_IDS=()
if [[ ${CUDA_VISIBLE_DEVICES+x} ]]; then
    if [[ -z "$CUDA_VISIBLE_DEVICES" || "$CUDA_VISIBLE_DEVICES" == ,* ||
          "$CUDA_VISIBLE_DEVICES" == *, || "$CUDA_VISIBLE_DEVICES" == *,,* ]]; then
        echo "Error: CUDA_VISIBLE_DEVICES is empty or contains an empty device"
        exit 1
    fi
    IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
    NUM_GPUS=${1:-${NUM_GPUS:-${#GPU_IDS[@]}}}
else
    NUM_GPUS=${1:-${NUM_GPUS:-8}}
fi
TP_SIZE=${2:-${TP_SIZE:-1}}

if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ && "$TP_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: NUM_GPUS and TP_SIZE must be positive integers"
    exit 1
fi

if [ $((NUM_GPUS % TP_SIZE)) -ne 0 ]; then
    echo "Error: NUM_GPUS($NUM_GPUS) must be divisible by TP_SIZE($TP_SIZE)"
    exit 1
fi

NUM_SERVERS=$((NUM_GPUS / TP_SIZE))
if ! [[ "$BASE_PORT" =~ ^[1-9][0-9]*$ && "$STARTUP_TIMEOUT" =~ ^[1-9][0-9]*$ ]] ||
   (( BASE_PORT + NUM_SERVERS - 1 > 65535 )); then
    echo "Error: invalid port range or STARTUP_TIMEOUT"
    exit 1
fi
if (( ${#GPU_IDS[@]} )); then
    if (( NUM_GPUS > ${#GPU_IDS[@]} )); then
        echo "Error: NUM_GPUS exceeds the devices in CUDA_VISIBLE_DEVICES"
        exit 1
    fi
    for device in "${GPU_IDS[@]}"; do
        if [[ -z "$device" || "$device" == "-1" ]]; then
            echo "Error: CUDA_VISIBLE_DEVICES contains an unavailable device"
            exit 1
        fi
    done
else
    mapfile -t GPU_IDS < <(seq 0 $((NUM_GPUS - 1)))
fi
# Check the whole port range before allocating GPU memory.
"${PYTHON}" - "$BASE_PORT" "$NUM_SERVERS" <<'PYTHON_CHECK'
import socket, sys
for port in range(int(sys.argv[1]), int(sys.argv[1]) + int(sys.argv[2])):
    with socket.socket() as sock:
        try:
            sock.bind(("0.0.0.0", port))
        except OSError as exc:
            raise SystemExit(f"Cannot start vLLM on port {port}: {exc}")
PYTHON_CHECK

mkdir -p "$LOG_DIR"
LOG_DIR=$(mktemp -d "${LOG_DIR}/run.XXXXXXXX")

for i in $(seq 0 $((NUM_SERVERS - 1))); do
    GPU_START=$((i * TP_SIZE))
    GPU_END=$((GPU_START + TP_SIZE - 1))
    GROUP_IDS=("${GPU_IDS[@]:GPU_START:TP_SIZE}")
    DEVICES=$(IFS=','; echo "${GROUP_IDS[*]}")
    PORT=$((BASE_PORT + i))
    LOG_FILE="${LOG_DIR}/server_gpu${GPU_START}-${GPU_END}_tp${TP_SIZE}_port${PORT}.log"

    echo "Starting server $((i+1))/${NUM_SERVERS}: GPUs=[${DEVICES}], tp=${TP_SIZE}, port=${PORT}"

    CUDA_VISIBLE_DEVICES=${DEVICES} \
    XDG_CACHE_HOME="${LOG_DIR}/cache_${i}" \
    VLLM_USE_MODELSCOPE=false \
    nohup vllm serve "$MODEL_PATH" \
        --served-model-name "$MODEL_NAME" \
        --tensor-parallel-size "$TP_SIZE" \
        --gpu-memory-utilization 0.8 \
        --mm-encoder-tp-mode data \
        --async-scheduling \
        --enable-prefix-caching \
        --host 0.0.0.0 \
        --port "$PORT" \
        > "$LOG_FILE" 2>&1 &

    PID=$!
    echo "$PID" >> "${LOG_DIR}/server.pids"
    echo "  PID: $PID"

    echo "  Waiting for server to be ready..."
    DEADLINE=$((SECONDS + STARTUP_TIMEOUT))
    while true; do
        if ! kill -0 $PID 2>/dev/null; then
            echo "  ERROR: server process exited unexpectedly, check $LOG_FILE"
            exit 1
        fi
        if "${PYTHON}" - "$PORT" <<'PYTHON_HEALTH'
import sys, urllib.request
try:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=2) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception:
    sys.exit(1)
PYTHON_HEALTH
        then
            echo "  Server $((i+1)) is ready!"
            break
        fi
        if (( SECONDS >= DEADLINE )); then
            echo "  ERROR: server readiness timed out after ${STARTUP_TIMEOUT}s; check $LOG_FILE"
            echo "  Started process IDs: ${LOG_DIR}/server.pids"
            exit 1
        fi
        sleep 2
    done
done

echo ""
echo "Launched ${NUM_SERVERS} servers (tp=${TP_SIZE}). Ports: ${BASE_PORT}-$((BASE_PORT + NUM_SERVERS - 1))"
echo "Logs: ${LOG_DIR}"
echo ""
echo "To check status:  ps aux | grep vllm"
echo "Server PID file: ${LOG_DIR}/server.pids"
