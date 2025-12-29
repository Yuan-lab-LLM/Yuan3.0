for iteration in 5 15 25
do
  echo "Running with iteration=$iteration"
  ITERATION=$iteration
  LOAD_PATH=$CASE_PATH/global_step_${ITERATION}/actor
  SAVE_PARH=$CASE_PATH/yuanvl_hf_${ITERATION}_iter_8pp

  cp -r $BASE_CKPT_PATH/*.py $LOAD_PATH/huggingface/
  cp -r $BASE_CKPT_PATH/tokenizer* $LOAD_PATH/huggingface/
  torchrun --nproc_per_node 8  -m verl.model_merger merge \
    --backend megatron \
    --trust-remote-code \
    --local_dir $LOAD_PATH \
    --target_dir $SAVE_PARH \
    --vit_dir $VIT_PATH

  cp -r $BASE_CKPT_PATH/*.py $SAVE_PARH/
  cp -r $BASE_CKPT_PATH/tokenizer* $SAVE_PARH/
  cp -r $BASE_CKPT_PATH/preprocessor_config.json $SAVE_PARH/
  cp -r $BASE_CKPT_PATH/config.json $SAVE_PARH/
  cp -r $BASE_CKPT_PATH/device_map.json $SAVE_PARH/

  rm -rf $SAVE_PARH/chat_template.jinja
  rm -rf $SAVE_PARH/tokenizer.json
  rm -rf $SAVE_PARH/added_tokens.json
done
