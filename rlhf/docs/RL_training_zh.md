# Yuan3.0 Flash 强化学习

## 一、简介

本文档提供了 Yuan3.0 Flash 模型的 Decoupled Advantage Policy Optimization (DAPO) 强化学习说明。

Yuan3.0 Flash 强化学习训练核心采用 Decoupled Advantage Policy Optimization (DAPO) 框架，同时支持 GSPO（General-Sum Preference Optimization）、SAPO（Strategy-Aware Preference Optimization）等多种偏好优化策略，可灵活适配不同长度规模的训练数据（4K/16K），并提供完善的超长输入处理机制。

## 二、使用方法

### 步骤1：启动 Ray 集群
```bash
# 进入verl模块目录
# 启动ray服务
# start head node
RAY_USE_IP_ADDRESS=True ray start --head --num-cpus=64 --num-gpus=8 --port=6400 --memory=873741824000 --dashboard-host 0.0.0.0  --node-ip-address=${your_head_node_ip}
# start worker node
RAY_USE_IP_ADDRESS=True ray start --num-cpus=64 --num-gpus=8 --memory=873741824000 --dashboard-host 0.0.0.0 --address $2:6400 --node-ip-address=${your_worker_node_ip}
bash config.sh host
```

### 步骤2：启动 DAPO 训练
```bash
# 执行40B规模YuanVL模型的DAPO训练脚本
cd Yuan3.0/rlhf/verl
bash recipe/dapo/run_dapo_yuanvl_megatron_40B.sh
```

## 参数配置

### 3.1 环境变量配置
训练前需根据实际路径配置以下环境变量，用于指定数据、模型和 checkpoint 存储位置：

| 环境变量 | 类型 | 说明 |
|---------|------|------|
| `MODEL_PATH` | 字符串 | 预训练模型文件的存放路径 |
| `TRAIN_16K_FILE` | 字符串 | 16K长度训练数据集的文件路径 |
| `TRAIN_4K_FILE` | 字符串 | 4K长度训练数据集的文件路径 |
| `CKPTS_DIR` | 字符串 | 训练过程中checkpoints文件的保存目录 |

### 3.2 核心训练参数
以下参数用于控制输入输出长度、超长文本处理策略，需在训练脚本中按需调整：

| 参数 | 类型 | 说明 |
|------|------|------|
| `max_prompt_length` | 整数 | 模型可接受的最大输入提示文本长度 |
| `max_response_length` | 整数 | 模型生成回复的最大文本长度限制 |
| `enable_overlong_buffer` | 布尔值 | 是否启用超长缓冲机制，处理超过`max_prompt_length`的输入 |
| `overlong_buffer_len` | 整数 | 超长缓冲机制的缓冲区大小 |
| `overlong_penalty_factor` | 浮点数 | 超长输入惩罚因子（用于损失函数权重调整） |
| `enable_overlong_prompts_filter` | 布尔值 | 是否自动过滤超过`max_prompt_length`的训练样本 |

## 四、优化策略切换
Yuan3.0 Flash 支持三种偏好优化策略，通过设置以下环境变量即可切换，各策略的参数配置如下：

### 4.1 GSPO（General-Sum Preference Optimization）
```bash
# GSPO 策略配置
loss_mode=gspo
clip_ratio_low=0.0003     
clip_ratio_high=0.0004    
loss_agg_mode="seq-mean-token-mean"
```

### 4.2 DAPO（Distribution-Aware Preference Optimization）
```bash
# DAPO 策略配置（默认推荐）
loss_mode=eighty-twenty
clip_ratio_low=0.2
clip_ratio_high=0.28
loss_agg_mode="token-mean"
```

### 4.3 SAPO（Strategy-Aware Preference Optimization）
```bash
# SAPO 策略配置
loss_mode=sapo
clip_ratio_low=1.05  # tau_neg（负样本tau值）
clip_ratio_high=1.00 # tau_pos（正样本tau值）
loss_agg_mode="token-mean"
```
