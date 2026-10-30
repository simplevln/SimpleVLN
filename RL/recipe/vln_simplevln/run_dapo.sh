#!/usr/bin/env bash
# Shared SimpleVLN DAPO launcher. Model presets set memory-related defaults only.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RL_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PRESET=${1:?Usage: run_dapo.sh MODEL_PRESET [Hydra overrides...]}
shift
case "${PRESET}" in
  qwen3vl_4b|qwen3vl_8b) default_micro=16; default_logprob=32; default_padding=True ;;
  qwen35_4b) default_micro=4; default_logprob=8; default_padding=False ;;
  qwen35_9b) default_micro=2; default_logprob=4; default_padding=False ;;
  *) echo "Unknown model preset: ${PRESET}" >&2; exit 2 ;;
esac

PYTHON=${PYTHON:-python3}
: "${MODEL_PATH:?Set MODEL_PATH to a local Hugging Face checkpoint directory}"
VLN_RL_DATA_ROOT=${VLN_RL_DATA_ROOT:-${RL_ROOT}/data/release_rl_data}
TRAIN_FILE=${TRAIN_FILE:-${VLN_RL_DATA_ROOT}/manifests/vln_train.parquet}
# Validation is disabled by default. Set VAL_FILE and trainer validation options
# together when using a separate validation dataset served by the environment.
VAL_FILE=${VAL_FILE:-${TRAIN_FILE}}
AGENT_CFG=${AGENT_CFG:-${SCRIPT_DIR}/config/agent_loop.yaml}
PROJECT=${PROJECT:-simplevln-dapo}
EXPERIMENT=${EXPERIMENT:-${PRESET}}
CKPT_DIR=${CKPT_DIR:-${RL_ROOT}/checkpoints/${PROJECT}/${EXPERIMENT}}
LOG_DIR=${LOG_DIR:-${RL_ROOT}/logs}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/${EXPERIMENT}.log}
LOGGER=${LOGGER:-console}

# Run against an existing Ray cluster. All nodes need the same code and model.
export RAY_ADDRESS=${RAY_ADDRESS:-auto}
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray}
export PYTHONPATH="${RL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_MM_INPUT_CACHE_GIB=${VLLM_MM_INPUT_CACHE_GIB:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export VERL_USE_EXTERNAL_MODULES=recipe.vln_simplevln.patches,recipe.vln_simplevln.dapo_patch

NNODES=${NNODES:-4}
NGPUS=${NGPUS:-8}
ROLLOUT_TP=${ROLLOUT_TP:-2}
FSDP_SIZE=${FSDP_SIZE:-8}
CUDA_MAX_CONN=${CUDA_MAX_CONN:-16}
ROLLOUT_N=${ROLLOUT_N:-8}
TEMPERATURE=${TEMPERATURE:-1.0}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.7}
NUM_WORKERS=${NUM_WORKERS:-48}
ROLLOUT_WINDOW=${ROLLOUT_WINDOW:-${VLN_ROLLOUT_WINDOW:-48}}
ROLLOUT_SCHEDULER=${ROLLOUT_SCHEDULER:-${VLN_ROLLOUT_SCHEDULER:-sliding}}
export VLN_ROLLOUT_WINDOW="${ROLLOUT_WINDOW}"
export VLN_ROLLOUT_SCHEDULER="${ROLLOUT_SCHEDULER}"

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-192}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-256}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
FILTER_GROUPS_ENABLE=${FILTER_GROUPS_ENABLE:-True}
FILTER_MAX_GEN=${FILTER_MAX_GEN:-0}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-${default_micro}}
LOG_PROB_MICRO=${LOG_PROB_MICRO:-${default_logprob}}
PPO_EPOCHS=${PPO_EPOCHS:-1}
ACTOR_LR=${ACTOR_LR:-5e-7}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
CLIP_GRAD=${CLIP_GRAD:-1.0}
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.28}
CLIP_RATIO_C=${CLIP_RATIO_C:-10.0}
LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
PARAM_OFFLOAD=${PARAM_OFFLOAD:-True}
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-True}
ENABLE_GRAD_CKPT=${ENABLE_GRAD_CKPT:-True}
# Qwen3.5 defaults to padded attention; packed attention requires compatible FLA kernels.
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-${default_padding}}
TOTAL_STEPS=${TOTAL_STEPS:-99999}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-20}
SAVE_FREQ=${SAVE_FREQ:-5}
TEST_FREQ=${TEST_FREQ:--1}

for name in NNODES NGPUS ROLLOUT_TP FSDP_SIZE ROLLOUT_N NUM_WORKERS ROLLOUT_WINDOW \
            TRAIN_BATCH_SIZE GEN_BATCH_SIZE PPO_MINI_BATCH_SIZE PPO_MICRO_BATCH_SIZE \
            LOG_PROB_MICRO MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH MAX_MODEL_LEN; do
  if ! [[ ${!name} =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer" >&2; exit 2
  fi
done
if (( NGPUS % ROLLOUT_TP != 0 || (NNODES * NGPUS) % FSDP_SIZE != 0 )); then
  echo "ROLLOUT_TP must divide NGPUS; FSDP_SIZE must divide the total GPU count" >&2; exit 2
fi
if (( MAX_MODEL_LEN < MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH )); then
  echo "MAX_MODEL_LEN must cover MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH" >&2; exit 2
fi
case "${ROLLOUT_SCHEDULER}" in
  sliding|window) ;;
  *) echo "ROLLOUT_SCHEDULER must be sliding or window" >&2; exit 2 ;;
esac
case "${FILTER_GROUPS_ENABLE,,}" in
  true) FILTER_GROUPS_ENABLE=True ;;
  false) FILTER_GROUPS_ENABLE=False ;;
  *) echo "FILTER_GROUPS_ENABLE must be True or False" >&2; exit 2 ;;
esac

DS_ARGS=()
if [[ "${FILTER_GROUPS_ENABLE}" == True ]]; then
  DS_ARGS+=( "+data.gen_batch_size=${GEN_BATCH_SIZE}" )
fi
DS_ARGS+=( "++algorithm.filter_groups={enable:${FILTER_GROUPS_ENABLE},metric:trajectory_reward,max_num_gen_batches:${FILTER_MAX_GEN}}" )
# Shell quoting protects argv; Hydra also needs quoting for literal strings.
hydra_string() {
  local value=$1 quoted='' backslashes='' char i
  for ((i = 0; i < ${#value}; i++)); do
    char=${value:i:1}
    case "${char}" in
      \\) backslashes+='\' ;;
      \") quoted+="${backslashes}${backslashes}\\\""; backslashes='' ;;
      *) quoted+="${backslashes}${char}"; backslashes='' ;;
    esac
  done
  # Hydra doubles backslashes only before a quote (including the closing quote).
  printf '"%s%s%s"' "${quoted}" "${backslashes}" "${backslashes}"
}

# Explicitly propagate recipe settings to Ray actors on every node.
ENV_ARGS=()
for name in VLN_ENV_SERVER_URL VLN_PROGRESS_COEF VLN_ADV_TRAJ_LEVEL VLN_TRAJ_LOSS_WEIGHT; do
  if [[ -v ${name} ]]; then
    ENV_ARGS+=( "++ray_kwargs.ray_init.runtime_env.env_vars.${name}=$(hydra_string "${!name}")" )
  fi
done

command=( "${PYTHON}" -m verl.trainer.main_ppo \
  "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES=$(hydra_string "${VERL_USE_EXTERNAL_MODULES}")" \
  ++ray_kwargs.ray_init.runtime_env.env_vars.VLN_ROLLOUT_SCHEDULER="${ROLLOUT_SCHEDULER}" \
  "++ray_kwargs.ray_init.runtime_env.env_vars.VLN_ROLLOUT_WINDOW=$(hydra_string "${ROLLOUT_WINDOW}")" \
  "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_MM_INPUT_CACHE_GIB=$(hydra_string "${VLLM_MM_INPUT_CACHE_GIB}")" \
  "++ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_CUDA_ALLOC_CONF=$(hydra_string "${PYTORCH_CUDA_ALLOC_CONF}")" \
  "++ray_kwargs.ray_init.runtime_env.env_vars.CUDA_DEVICE_MAX_CONNECTIONS=$(hydra_string "${CUDA_MAX_CONN}")" \
  "${DS_ARGS[@]}" "${ENV_ARGS[@]}" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  "data.train_files=[$(hydra_string "${TRAIN_FILE}")]" \
  "data.val_files=$(hydra_string "${VAL_FILE}")" \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.max_prompt_length=${MAX_PROMPT_LENGTH} \
  data.max_response_length=${MAX_RESPONSE_LENGTH} \
  data.filter_overlong_prompts=False \
  data.truncation=error \
  data.return_raw_chat=True \
  "actor_rollout_ref.model.path=$(hydra_string "${MODEL_PATH}")" \
  actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING} \
  actor_rollout_ref.model.enable_gradient_checkpointing=${ENABLE_GRAD_CKPT} \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
  actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
  actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enable_prefix_caching=False \
  actor_rollout_ref.rollout.n=${ROLLOUT_N} \
  actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
  actor_rollout_ref.rollout.max_num_seqs=32 \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO} \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_MODEL_LEN} \
  "actor_rollout_ref.rollout.agent.agent_loop_config_path=$(hydra_string "${AGENT_CFG}")" \
  actor_rollout_ref.rollout.agent.default_agent_loop=vln_full_episode_agent \
  actor_rollout_ref.rollout.agent.num_workers=${NUM_WORKERS} \
  ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=recipe.vln_simplevln.online_rollout_manager.VLNOnlineRolloutManager \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model] \
  actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
  actor_rollout_ref.actor.optim.clip_grad=${CLIP_GRAD} \
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS} \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
  actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
  actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
  actor_rollout_ref.actor.clip_ratio_c=${CLIP_RATIO_C} \
  actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE} \
  actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=${FSDP_SIZE} \
  actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD} \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD} \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO} \
  actor_rollout_ref.ref.fsdp_config.param_offload=${PARAM_OFFLOAD} \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${MAX_MODEL_LEN} \
  "trainer.logger=[${LOGGER}]" \
  "trainer.project_name=$(hydra_string "${PROJECT}")" \
  "trainer.experiment_name=$(hydra_string "${EXPERIMENT}")" \
  trainer.n_gpus_per_node=${NGPUS} \
  trainer.nnodes=${NNODES} \
  trainer.save_freq=${SAVE_FREQ} \
  "trainer.default_local_dir=$(hydra_string "${CKPT_DIR}")" \
  trainer.test_freq=${TEST_FREQ} \
  trainer.val_before_train=False \
  trainer.total_epochs=${TOTAL_EPOCHS} \
  trainer.total_training_steps=${TOTAL_STEPS} \
  trainer.balance_batch=False \
  "$@" )

if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
[[ -d "${MODEL_PATH}" ]] || { echo "Model directory does not exist: ${MODEL_PATH}" >&2; exit 2; }
for path in "${TRAIN_FILE}" "${VAL_FILE}" "${AGENT_CFG}"; do
  [[ -f "${path}" ]] || { echo "Input file does not exist: ${path}" >&2; exit 2; }
done
mkdir -p -- "$(dirname -- "${LOG_FILE}")" "${RAY_TMPDIR}"
"${command[@]}" 2>&1 | tee -- "${LOG_FILE}"
