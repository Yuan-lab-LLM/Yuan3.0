# AIME2024 Evaluation Operation Guide
## 一、Environment Configuration
### Standard environmental configuration is sufficient, with no special requirements

## 二、Input Data Processing
### 1.If the input data file is a single file, this step can be ignored; if there are multiple files, use the `cat` command to combine all input files into one file, which can be placed in the same directory. Note that the file names must not be duplicate。
### cat command demonstration：
```bash
cat result/AIME2024/2024_II_cle* > result/AIME2024/2024_II_cle_all.txt
```
### 2.Create files_eval_your.txt and files_origin_your.txt in the same directory as the eval_easy_lzy.sh file, located in eval/scripts/AIME2024/. Place the absolute path of the input data in the files_eval_your.txt file, with one absolute path per line for each input data file. Meanwhile, place the absolute path of the original data file, eval/datasets/AIME2024/2024_II_cle_001.txt, in the files_origin_your.txt file。

## 三、Evaluate Script Execution
### 1.Establish an evaluation result path, for example:eval/eval_output/AIME2024。
### 2.Modify the OUTPUT_PATH parameter in the eval_easy_lzy.sh file, changing the path to the directory where you want to save the evaluation results.
### 3.Switch the current path to the directory path of the eval_easy_lzy.sh file, which is eval/scripts/AIME2024/, and execute it using the following command:
```bash
bash eval_easy_lzy.sh
```

## 四、View Evaluation Results
### Switch to the directory eval/eval_output/AIME2024. If the input file name is 2024_II_cle_all.txt, check the value of the data parameter "accuracy" in the last line of the result_2024_II_cle_all_judge/result.txt file. That is, if the input file name is xxx.txt, check the value of the data parameter "accuracy" in the last line of the result_xxx_judge/result.txt file. For example, if "accuracy: 96.6667%", then the result is 96.6667%.