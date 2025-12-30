#!/usr/bin/env bash

set -xeuo pipefail
HYDRA_FULL_ERROR=1
ulimit -n 65535

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

project_name=<Specify project name>
exp_name=<Specify exp_name>

adv_estimator=grpo
norm_adv_by_std_in_grpo=True
loss_mode=eighty-twenty-overlong-filter

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 8))
max_response_length=$((1024 * 16))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 1))
overlong_penalty_factor=1.0
enable_overlong_prompts_filter=True

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=seq_final_reward

hard_to_easy=False
easy_to_hard=True
center_to_sides=False
max_num_gen_batches=10
train_prompt_bsz=512
gen_prompt_bsz=$((train_prompt_bsz * 1))
n_resp_per_prompt=8
train_prompt_mini_bsz=64

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-32}
# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}

MODEL_PATH=<Specify model path>
HF_PROCESSOR_PATH=<Specify processor path>

CKPTS_DIR=<Specify path>/${project_name}/${exp_name}
TRAIN_16K_FILE=<Specify 16k_files path>
TRAIN_4K_FILE=<Specify 4k_files path>
TEST_FILE=<Specify test_files path>

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=True
rollout_mode="async"
gen_pp=1
gen_tp=4
train_tp=1
train_pp=8

log_path=$CKPTS_DIR/logs
mkdir -p $log_path

step_rewards_path=$CKPTS_DIR/steps_reward_data
mkdir -p $step_rewards_path

ray job submit --address="http://127.0.0.1:8265" \
--runtime-env=verl/trainer/runtime_env.yaml \
--log-style record \
-- \
python3 -m recipe.dapo.main_dapo \
    --config-path=config \
    --config-name='dapo_trainer_megatron.yaml' \
    data.train_16k_files="${TRAIN_16K_FILE}" \
    data.train_4k_files="${TRAIN_4K_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    data.dataloader_num_workers=8 \
    data.filter_overlong_prompts=${enable_overlong_prompts_filter} \
    data.filter_overlong_prompts_workers=100 \
    data.image_key=images \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.nccl_timeout=6000000 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.hard_to_easy=${hard_to_easy} \
    algorithm.filter_groups.easy_to_hard=${easy_to_hard} \
    algorithm.filter_groups.center_to_sides=${center_to_sides} \
    algorithm.norm_adv_by_std_in_grpo=${norm_adv_by_std_in_grpo} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.hf_processor_path="${HF_PROCESSOR_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.adam_beta1=0.9 \
    actor_rollout_ref.actor.optim.adam_beta2=0.95 \
    actor_rollout_ref.actor.optim.adam_eps=1e-15 \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.main_params_dtype=fp32 \
    actor_rollout_ref.actor.optim.exp_avg_dtype=fp32 \
    actor_rollout_ref.actor.optim.exp_avg_sq_dtype=fp32 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.optim.main_grads_dtype=fp32 \
    actor_rollout_ref.actor.optim.optimizer_cpu_offload=False \
    actor_rollout_ref.actor.optim.use_precision_aware_optimizer=False \
    actor_rollout_ref.actor.optim.use_offload_precision_aware_optimizer=False \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.use_distributed_optimizer=True \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method="uniform" \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity="full" \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.actor.megatron.override_transformer_config.use_loss_chunk=True \
    actor_rollout_ref.actor.megatron.override_transformer_config.attention_dropout=0 \
    actor_rollout_ref.actor.megatron.override_transformer_config.compute_probs_in_model=True \
    actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend='flash' \
    actor_rollout_ref.rollout.lm_head_dtype=float32 \
    actor_rollout_ref.actor.megatron.override_transformer_config.compute_lm_head_fp32=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.adjustable_response_length_by_sample=True \
    actor_rollout_ref.rollout.pipeline_model_parallel_size=${gen_pp} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    reward_model.reward_manager=yuan_dapo_mp \
    reward_model.reward_kwargs.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.reward_kwargs.overlong_buffer.len=${overlong_buffer_len} \
    trainer.val_before_train=False \
    trainer.logger='["console", "tensorboard"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    trainer.test_freq=5000000000 \
    trainer.save_freq=10 \
    trainer.total_epochs=2 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto | tee ${log_path}/log_${DATETIME}.log

