# Data Processing Module

## Introduction

This module provides specialized utilities for converting training datasets into parquet format with flexible configuration.

### Configuration Variables

| Variable Name | Description |
|---------------|-------------|
| `--input_path` | Each line specifies: number of rows to select, file path, data category, and thinking mode status |
| `--output_path` | Storage path for the converted parquet data |
| `--split_type` | Data split type: 'train' or 'test' |
| `--flag_image` | Multimodal flag: 1 for image-text data, 0 for text-only data |

## Dataset Structure

The training data must comply with the following format:

```json
{
    "reward_method": "llm_math",
    "language": "en",
    "data_source": "llm_math",
    "prompt": "[{'content': 'Two pipes A and B can fill a tank in 10 hours and 15 hours respectively, while a third pipe C can drain the tank in 20 hours. If all three pipes are opened simultaneously, how much time will be taken to fill the tank completely?', 'role': 'user'}]",
    "ability": "llm_math",
    "reward_model": "{'ground_truth': '8.57', 'style': 'rule'}",
    "extra_info": "{'answer': '8.57', 'enable_thinking_flag': False, 'expect_len': 529.0, 'index': 0, 'question': 'Two pipes A and B can fill a tank in 10 hours and 15 hours respectively, while a third pipe C can drain the tank in 20 hours. If all three pipes are opened simultaneously, how much time will be taken to fill the tank completely?', 'split': 'train'}"
}
```

Each piece of data is supposed to contain the four mandatory fields, namely `reward_method`, `language`, `enable_thinking_flag` and `expect_len`. Among them, `reward_method` serves to identify the data category, `language` is used to specify the language type of the input data, `enable_thinking_flag` indicates whether the thinking model is enabled for the data during training, and `expect_len` defines the expected length of the model output.

## Usage

Run the following command to start data conversion:

```bash
cd Yuan3.0/rlhf/verl
python examples/data_preprocess/data_preprocess_select_except_len.py --input_path '<Specify input information>' --output_path '<Specify path>' --split_type '<train/test>' --flag_image '<0/1>'
```
