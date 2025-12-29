python tools/convert_yuanvl2.py \
    --orig-ckpt-path /mnt/beegfs/wangshenling/vl_yuan/train/log/40B-train/ns_moe_case1_multimodel-sft-20250819-newcode/ckpt \
    --vit-ckpt-path /icfsikc/zhaoxudong/ckpt/models--InternViT-300M-448px \
    --output-ckpt-path /mnt/beegfs/wangshenling/vl_yuan/train/log/40B-train/ns_moe_case1_multimodel-sft-20250819-newcode/yuanvl_hf_1980_iter_8pp_debug \
    --pipeline-parallel-size 12 \
    --num-layer 24 \
    --hidden-size 2048 \
    --num-attention-heads 16 \
    --kv-channels 256 \
    --ffn-hidden-size 8192 \
    --pipeline-model-parallel-blocks 2,2,2,2,2,2,2,2,2,2,2,2 \
    --per-layer-experts-blocks 32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32 \
    --tokenizer-path /icfsikc/wangshenling/vl_yuan/train/megatron-lm-20240812-pipe2-v5-stage2-base-6bvit-stage0-v2/hf_tokenizer \
    #--orig-ckpt-path /mnt/beegfs3/wangshenling/vl_yuan/train/log/40B_stage2_case1_12pp/1tp2pp/ \
    #--orig-ckpt-path /mnt/beegfs3/wangshenling/vl_yuan/train/log/210b/210B_stage2_pcase4_24pp_2ep/8pp1tp/ckpt_5 \
    # --per-layer-experts-blocks 32,32,20,24,24,28,28,28,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,28,28,24,20 \
