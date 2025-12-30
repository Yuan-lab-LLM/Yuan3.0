#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

ulimit -c 0
ulimit -n 65536

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6032"}
NUM_NODES=${NNODES:-"1"}
NODE_RANK=${NODE_RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CASE_CHECKPOINT_PATH=<Specify path>
CHECKPOINT_PATH=$CASE_CHECKPOINT_PATH/ckpt
CHECKPOINT_PATH_LOAD=<Specify path>
LOG_PATH=$CASE_CHECKPOINT_PATH/log/log-${NUM_NODES}n-$DATETIME
DATA_PATH=<Specify path and file prefix>_text_document
TOKENIZER_MODEL=<Specify path to file>


mkdir -p $LOG_PATH
mkdir -p $CHECKPOINT_PATH

YUANVL_FULL_NPY_PATH=$CASE_CHECKPOINT_PATH/datanpy
mkdir -p $YUANVL_FULL_NPY_PATH
CLIP_MODEL_NAME=InternViT-448
CLIP_DOWNLOAD_PATH='OpenGVLab/InternViT-300M-448px'
CLIP_VISUAL_SIZE=1024
CLIP_HIDDEN_SIZE=1024

IMAGE_SEGMENT_METHOD=dynamic

MODEL_VL_ARGS=(
    --use-yuanvl
    --yuanvl-full-npy-path $YUANVL_FULL_NPY_PATH
    --max-split-tile-num-single-image 9
    --max-split-tile-num-multi-image 4
    --image-segment-method $IMAGE_SEGMENT_METHOD
    --clip-download-path $CLIP_DOWNLOAD_PATH
    --clip-model-name $CLIP_MODEL_NAME
    --downsample-ratio 0.5
    --clip-visual-size $CLIP_VISUAL_SIZE
    --clip-hidden-size $CLIP_HIDDEN_SIZE
    --yuanvl-use-te-imagemlp
    --eod-mask-loss
    --reset-position-ids
    --data-cut-length 33792
)


DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT

)

MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 32768
    --max-position-embeddings 32768
    --num-layers 24
    --hidden-size 2048
    --ffn-hidden-size 8192
    --num-attention-heads 16
    --kv-channels 256
    --init-method-std 0.01
    --attention-dropout 0.1
    --hidden-dropout 0.0
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 1000000
    --rotary-percent 0.5
    --use-lf-gate
    --lf-conv2d-group 1
    --lf-conv2d-num-pad 1
    --lf-conv2d-add-bias
    --no-rope-fusion
    --use-flash-attn
    --no-bias-dropout-fusion
    --ckpt-format torch
    --finetune
)


MOE_ARGS=(
    --num-experts 32
    --moe-router-topk 2
    --moe-router-load-balancing-type none
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --use-attention-router
)

DATA_ARGS=(
    --tokenizer-type YuanTokenizer
    --tokenizer-model-path ${TOKENIZER_MODEL}
    --data-path $DATA_PATH
    --split 10,0,0
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 1440
    --num-workers 16
    --lr 8.787e-6
    --train-iters 13125
    --lr-decay-iters 13125
    --lr-decay-style constant
    --min-lr 8.787e-6
    --weight-decay 0.1
    --lr-warmup-iters 0
    --clip-grad 1.0
    --bf16
    --adam-beta1 0.9
    --adam-beta2 0.95
    --use-loss-chunk
    --recompute-method uniform
    --recompute-granularity full
    --recompute-num-layers 1
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 8
    --expert-model-parallel-size 1
    --use-distributed-optimizer
    --sequence-parallel
)

LOGGING_ARGS=(
    --log-interval 1
    --timing-log-level 2
    --save-interval 525
    --eval-interval 10000000
    --eval-iters 10
    --log-progress
    --log-throughput
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH_LOAD
    --tensorboard-dir "${CASE_CHECKPOINT_PATH}/tensorboard"
)


torchrun ${DISTRIBUTED_ARGS[@]} pretrain_yuanvl.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${MODEL_VL_ARGS[@]} 2>&1 | tee $LOG_PATH/$NODE_RANK.log


