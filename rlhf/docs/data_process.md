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

Below is an example of a training data entry:

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

## Usage

Run the following command to start data conversion:

```bash
cd Yuan3.0/rlhf/verl
python examples/data_preprocess/data_preprocess_select_except_len.py --input_path '<Specify input information>' --output_path '<Specify path>' --split_type '<train/test>' --flag_image '<0/1>'
```

## Script Overview

| Script | Purpose | Special Features |
|-----------|---------|------------------|
| `data_process_select_except_len.py` | General data processing | Standard text data conversion |
| `data_process_select_mllm_math_enable_except_len.py` | Multimodal math data | Image-text integration for mathematical reasoning |
| `data_process_select_tool_enable_except_len.py` | Tool-specific data | Specialized processing for tool interaction data |


