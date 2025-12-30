# Yuan3.0 Flash Supervised Fine-Tuning

## Introduction

This document provides guidance and instructions for performing Supervised Fine-Tuning (SFT) on Yuan3.0 Flash.

## Usage

To run supervised fine-tuning for Yuan3.0 Flash 40B model, use the following example script:

```shell
# cd <Specific_Path>/Megatron-LM
cd Yuan3.0/rlhf/megatron-lm
bash examples/pretrain_yuan3.0_40B_sft.sh
```

### Parameter Settings

Relevant parameters should be correctly set before running the script.

First, make the necessary modifications, including setting the environment variables `CASE_CHECKPOINT_PATH`, `DATA_PATH`, `TOKENIZER_MODEL`, `CLIP_DOWNLOAD_PATH`, and `CHECKPOINT_PATH_LOAD`.

Use `--finetune` to load the model for fine-tuning. This option does not load the optimizer or random number generator states from the checkpoint and sets the iteration count to 0. Use this option when loading a released checkpoint.

If the dataset path is:
```
/path/dataset.bin
```
You can set `DATA_PATH` as:
```shell
DATA_PATH='1 /path/dataset'
```

For more descriptions of command-line arguments, please refer to the source file `arguments.py` and [REAMME.md](https://github.com/NVIDIA/Megatron-LM/blob/main/README.md).
