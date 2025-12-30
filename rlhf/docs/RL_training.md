# Yuan3.0 Flash Reinforcement Learning

## 1. Introduction

This document provides instructions for Decoupled Advantage Policy Optimization (DAPO) reinforcement learning for the Yuan3.0 Flash model.

The reinforcement learning training of Yuan3.0 Flash adopts the Decoupled Advantage Policy Optimization (DAPO) framework at its core. It simultaneously supports multiple preference optimization strategies such as GSPO (General-Sum Preference Optimization) and SAPO (Strategy-Aware Preference Optimization). It can flexibly adapt to training data of varying length scales (4K/16K) and provides a robust mechanism for handling ultra-long inputs.

## 2. Usage

### Step 1: Start the Ray Cluster
```bash
# Navigate to the verl module directory
# start head node
RAY_USE_IP_ADDRESS=True ray start --head --num-cpus=64 --num-gpus=8 --port=6400 --memory=873741824000 --dashboard-host 0.0.0.0  --node-ip-address=${your_head_node_ip}
# start worker node
RAY_USE_IP_ADDRESS=True ray start --num-cpus=64 --num-gpus=8 --memory=873741824000 --dashboard-host 0.0.0.0 --address ${your_head_node_ip}:6400 --node-ip-address=${your_worker_node_ip}
```

### Step 2: Start DAPO Training
```bash
# Execute the DAPO training script for the 40B-scale YuanVL model
cd Yuan3.0/rlhf/verl
bash recipe/dapo/run_dapo_yuanvl_megatron_40B.sh
```

## 3. Parameter Configuration

### 3.1 Variable Configuration
Before training, configure the following variables according to the actual paths to specify the data, model, and checkpoint storage locations:

| Variable | Type | Description |
|----------------------|------|-------------|
| `MODEL_PATH` | String | Path to the directory containing the pre-trained model files. |
| `TRAIN_16K_FILE` | String | File path for the 16K-length training dataset. |
| `TRAIN_4K_FILE` | String | File path for the 4K-length training dataset. |
| `CKPTS_DIR` | String | Directory for saving checkpoint files during training. |

### 3.2 Training Parameters
The following parameters control input/output length and long-text handling strategies. Adjust them in the training script as needed:

| Parameter | Type | Description |
|-----------|------|-------------|
| `max_prompt_length` | Integer | Maximum length of input prompt text that the model can accept. |
| `max_response_length` | Integer | Maximum text length limit for the model's generated responses. |
| `enable_overlong_buffer` | Boolean | Whether to enable the overlong buffer mechanism to handle inputs exceeding `max_prompt_length`. |
| `overlong_buffer_len` | Integer | Buffer size for the overlong buffer mechanism. |
| `overlong_penalty_factor` | Float | Penalty factor for overlong inputs (used for loss function weight adjustment). |
| `enable_overlong_prompts_filter` | Boolean | Whether to automatically filter training samples exceeding `max_prompt_length`. |

## 4. Optimization Strategy Switching

Yuan3.0 Flash supports three preference optimization strategies. Switch between them by setting the following variables. The parameter configurations for each strategy are as follows:

### 4.1 GSPO (General-Sum Preference Optimization)
```bash
# GSPO Strategy Configuration
loss_mode=gspo
clip_ratio_low=0.0003
clip_ratio_high=0.0004
loss_agg_mode="seq-mean-token-mean"
```

### 4.2 DAPO (Distribution-Aware Preference Optimization)
```bash
# DAPO Strategy Configuration (Default Recommended)
loss_mode=eighty-twenty
clip_ratio_low=0.2
clip_ratio_high=0.28
loss_agg_mode="token-mean"
```

### 4.3 SAPO (Strategy-Aware Preference Optimization)
```bash
# SAPO Strategy Configuration
loss_mode=sapo
clip_ratio_low=1.05  # tau_neg (tau value for negative samples)
clip_ratio_high=1.00 # tau_pos (tau value for positive samples)
loss_agg_mode="token-mean"
```
