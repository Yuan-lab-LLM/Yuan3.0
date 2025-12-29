CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server --model=$CKPTPATH --served-model-name Qwen-72B --tensor-parallel-size=4 --trust-remote-code --disable-custom-all-reduce --max-num-seqs=256  --port=7211 --dtype bfloat16  &
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server --model=$CKPTPATH --served-model-name Qwen-72B --tensor-parallel-size=4 --trust-remote-code --disable-custom-all-reduce --max-num-seqs=256  --port=7210 --dtype bfloat16 

