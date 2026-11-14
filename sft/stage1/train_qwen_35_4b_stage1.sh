# 4 nodes x 8 GPUs. Set NODE_RANK=0..3 and MASTER_ADDR to node 0.
EXPERIMENT_NAME="Qwen_35_4B_stage1"
OUTPUT_DIR="/path/to/outputs/1epoch/${EXPERIMENT_NAME}"
CACHE_DIR="/path/to/cache_data"
echo ${OUTPUT_DIR}

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NNODES=4 \
NODE_RANK="${NODE_RANK:?Set NODE_RANK to 0, 1, 2, or 3}" \
MASTER_ADDR="${MASTER_ADDR:?Set MASTER_ADDR to the address of node 0}" \
MASTER_PORT="${MASTER_PORT:-29500}" \
NPROC_PER_NODE=8 \
MAX_PIXELS=1003520 \
megatron sft \
    --model /path/to/models/Qwen3.5-4B \
    --save_safetensors true \
    --cached_dataset "${CACHE_DIR}/SFT/R2R/train" "${CACHE_DIR}/SFT/RxR/train" \
    --tensor_model_parallel_size 2 \
    --sequence_parallel true \
    --packing true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --micro_batch_size 2 \
    --global_batch_size 256 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --num_train_epochs 1 \
    --output_dir ${OUTPUT_DIR} \
    --save_steps 50 \
    --max_length 40960 \
    --dataloader_num_workers 16 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 32
