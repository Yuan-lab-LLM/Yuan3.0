https://verl.readthedocs.io/en/latest/advance/checkpoint.html#convert-fsdp-and-megatron-checkpoints-to-huggingface-format-model
  
训练过程中，Megatron格式的ckpt应该是如下格式保存：
```
checkpoints/${trainer.project_name}/${trainer.experiment_name}
├── global_steps_${i}
│   ├── actor
│   │   ├── huggingface     # default save config and tokenizer, save huggingface model if include ``hf_mode`` in checkpoint.contents
│   │   └── dist_ckpt       # save sharded model/optimizer/rng_states, naming the same as Megatron
│   └── critic
│   │   ├── huggingface
│   │   └── dist_ckpt
└── latest_checkpointed_iteration.txt
```
使用说明
```
usage: python -m verl.model_merger merge [-h] --backend {fsdp,megatron} [--local_dir LOCAL_DIR] [--tie-word-embedding] [--is-value-model] [--use_cpu_initialization] [--target_dir TARGET_DIR]
                     [--hf_upload_path HF_UPLOAD_PATH] [--private]

options:
-h, --help            show this help message and exit
--backend {fsdp,megatron}
                        The backend of the model
--local_dir LOCAL_DIR
                        Path to the saved model checkpoints
--tie-word-embedding  Whether to tie word embedding weights (currently only Megatron supported)
--is-value-model      Whether the model is a value model (currently only Megatron supported)
--use_cpu_initialization
                        Whether to use CPU initialization for the model. This is useful for large models that cannot fit into GPU memory during initialization.
--target_dir TARGET_DIR
                        Directory to save the merged huggingface model
--vit_dir
                        Directory to load vit model
--hf_upload_path HF_UPLOAD_PATH
                        Hugging Face repository ID to upload the model
--private             Whether to upload the model to a private Hugging Face repository
```

使用示例
```
torchrun --nproc_per_node 8  -m verl.model_merger merge \
    --backend megatron \
    --trust-remote-code \
    --local_dir $CKPT_PATH \
    --target_dir $TARGET_PATH \
    --vit_dir $VIT_PATH
```
注意，目前暂不支持world_size=1
