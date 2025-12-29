torchrun --nproc_per_node 8  -m verl.model_merger merge \
    --backend megatron \
    --trust-remote-code \
    --local_dir $CKPT_DIR \
    --target_dir $TARGET_DIR \
    --vit_dir $VIT_DIR

