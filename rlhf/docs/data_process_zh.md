# data_process
本模块提供工具，支持通过灵活配置将训练数据集转换为 Parquet 格式。

## 配置变量

代码中的主要变量设置如下：

| 变量名称      | 描述 |
| ------------------ | -----------------------------------------------------------------------------------------------------------------------------------------------|
| `--input_path`     | 每一行存储待转训练数据的选取行数、文件路径、数据类别、是否启用思考模式  |
| `--output_path`    | 转换完成的parquet数据存储路|
| `--split_type`     | 数据划分类型：取值为 `train`（训练集）或 `test`（测试集）|
| `--flag_image`     | 多模态标记：取值 1 表示图文数据，取值 0 表示纯文本数据|

## Dataset

下面是一条训练数据样本示例：

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


## 使用方法

执行如下命令启动数据转换。

```bash
cd Yuan3.0/rlhf/verl
python ./examples/data_preprocess/data_preprocess_select_except_len.py --input_path '<Specify input informations>' --output_path '<Specify path>' --split_type '<train/test>' --flag_image '<0/1>'
```

## 脚本文件概览
| 脚本文件 | 用途 | 特色功能 |
|-----------|---------|------------------|
| `data_process_select_except_len.py` | 通用数据处理 | 标准文本数据格式转换 |
| `data_process_select_mllm_math_enable_except_len.py` | 多模态数学数据处理 | 实现图文融合，支撑数学推理任务 |
| `data_process_select_tool_enable_except_len.py` | 工具专属数据处理 | 针对工具交互类数据的专业化处理 |
