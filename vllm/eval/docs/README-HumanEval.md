# HumanEval

GitHub Repository: https://github.com/openai/human-eval


## Environment
Use the same environment as the official version:
```
pip install termcolor
pip install fire

cd eval/scripts/HumanEval
python setup.py install
```

## Data Processing
Process the data into JSONL format. Each line of data must comply with the following format:
```
question[SEP]answer<|end_of_sentence|><eod>
```
Write the absolute path of the processed data file into `eval/scripts/HumanEval/files_eval.txt`. Each line in the file shall contain one file path only.

## Execution

Navigate to the path `eval/scripts/HumanEval` (the target scoring file `files_eval.txt` is stored in this directory). Modify the `OUTPUT_PATH` configuration item in the script file `score_humaneval_ly.sh` under this path to specify your custom output directory. Then run the following command to start scoring:
```
bash score_humaneval_ly.sh
```


The scoring results will be output to the specified path, which is a multi-level directory structure. The name of the folder at the final level is consistent with the corresponding file name. Three files are generated in the output path. Among them, `result.txt` records the complete scoring process, and the numerical value in the last line of the file is the final scoring result.

