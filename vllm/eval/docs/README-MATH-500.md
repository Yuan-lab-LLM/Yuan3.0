# MATH-500 Evaluation Operation Guide
## 一、Environment Configuration
### 1. The vllm module needs to be installed
### 2. Requires CUDA environment and two GPU cards

## 二、Input Data Processing
### 1.If the input data file is a single file, this step can be ignored; if there are multiple files, use the `cat` command to combine all input files into one file, which can be placed in the same directory. Note that the file names must not be duplicate。
### cat command demonstration：
```bash
cat result/MATH-500/HuggingFaceH4_MATH-500_standard* > result/MATH-500_all/HuggingFaceH4_MATH-500_standard_all.txt
```

## 三、Evaluate Script Execution
### 1. Establish an evaluation result path, for example:eval/eval_output/MATH-500。
### 2. Modify the scoring model path in the judge_with_vllm_model_math.py file, changing the path to the desired scoring model path.
### 3.Switch the current path to the directory path of the judge_with_vllm_model_math.py file, which is eval/scripts/MATH-500/, and execute it using the following command:
```bash
GPU_NUMS=2 MAX_TOKENS=4096 BATCH_SIZE=480 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0,1 python judge_with_vllm_model_math.py --input_path result/MATH-500_all/ --output_path eval/eval_output/MATH-500 --origin_path eval/datasets/MATH-500/HuggingFaceH4_MATH-500_standard_001.txt
```
```bash
Parameter interpretation：
GPU_NUMS=2: This environment variable specifies that the number of GPUs used is 2.
MAX_TOKENS=4096: This parameter specifies the maximum number of tokens for model processing, which is 4096.
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python: This environment variable specifies the use of the Python implementation of Protocol Buffers.
VLLM_WORKER_MULTIPROC_METHOD=spawn: This environment variable sets the multiprocessing method for VLLM to spawn.
CUDA_VISIBLE_DEVICES=0,1: In this environment variable, GPUs with IDs 0 and 1 will be used, which can be customized and adjusted.
python judge_with_vllm_model_math.py: This is the name of the Python script to be executed.
--input_path: Directory path of the input file.
--output_path: Directory path for output files.
--origin_path: The absolute path of the original data.
```
## 四、View Evaluation Results
### Look at the parameter "accuracy" in the screen print result. For example, if "accuracy" is 96.6667%, then the typing result is 96.6667%.
