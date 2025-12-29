python tools/convert_yuanvl.py \
    --orig-ckpt-path /mnt/beegfs/wangshenling/vl_yuan/train/log/40b/40B_stage2_pcase12_12pp_fuse/1tp8pp/1tp8pp_new \
    --vit-ckpt-path /icfsikc/zhaoxudong/ckpt/models--InternViT-300M-448px \
    --output-ckpt-path /mnt/beegfs3/zhaoxudong/ckpt/40B_stage2_pcase12_12pp_verl \
    --pipeline-parallel-size 12 \
    --num-layer 24 \
    --hidden-size 2048 \
    --num-attention-heads 16 \
    --kv-channels 256 \
    --ffn-hidden-size 8192 \
    --pipeline-model-parallel-blocks 2,2,2,2,2,2,2,2,2,2,2,2 \
    --per-layer-experts-blocks 32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32 \
    --tokenizer-path /mnt/beegfs3/wangshenling/vl_yuan/train/megatron-lm-20240812-pipe2-v5-stage2-base-vllm/hf_tokenizer \
    #--orig-ckpt-path /mnt/beegfs3/wangshenling/vl_yuan/train/log/40B_stage2_case1_12pp/1tp2pp/ \
    #--orig-ckpt-path /mnt/beegfs3/wangshenling/vl_yuan/train/log/210b/210B_stage2_pcase4_24pp_2ep/8pp1tp/ckpt_5 \
    # --per-layer-experts-blocks 32,32,20,24,24,28,28,28,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,28,28,24,20 \
