# livecodebench

GitHub Repository: https://github.com/LiveCodeBench/LiveCodeBench

Modify the data path in `eval/scripts/livecodebench/lcb_runner/benchmarks/code_generation.py` to your locally saved path.
Modify the tokenizer path in `eval/scripts/livecodebench/lcb_runner/runner/vllm_runner.py` to your locally saved path.
In `eval/scripts/livecodebench/lcb_runner/runner/main_txt_2.py`, adjust the format and method of data reading, and parse the results accordingly.



## Environment
Use the same environment as the official version:


## Data Processing
Process the data into JSONL format. Each line of data must comply with the following format:
```
question<sep>answer<|end_of_sentence|><eod>
```
Write the absolute path of the processed data file into `eval/scripts/livecodebench/files_eval.txt`. Each line in the file shall contain one file path only.


## Execution

Navigate to the path `eval/scripts/livecodebench` (the target scoring file `files_eval.txt` is stored in this directory). Modify the `OUTPUT_PATH` configuration item in the script file `eval_livecodebench_v6.sh` under this path to specify your custom output directory. Then run the following command to start scoring:
```
bash eval_livecodebench_v6.sh
```

The scoring results will be output to the specified path, which is a multi-level directory structure. The name of the folder at the final level is consistent with the corresponding file name. Four files are generated in the output path. Among them, the log file records the complete scoring process, and the numerical value in the last line of the file is the final scoring result.
