MMTab Evaluation Operation Guide

1、Environment Configuration

```
pip install -r requirements.txt
```

2、Input Data Processing

​	If the data files `question_md.jsonl` and `question_html.jsonl` are split into multiple parts, first merge them using `cat` and save the result to the `result/` directory.

 	The textual results need to be converted into a result file compatible with the evaluation code. The processing scripts are `convert2evaldata_md.py` and `convert2evaldata_html.py`. For image results, this step is omitted.Use the `--path` argument to specify the result file, which will be saved separately as `{path}_eval_md.jsonl` and `{path}_eval_html.jsonl`.

```
python3 convert2evaldata_html.py --path result/question_html.jsonl

python3 convert2evaldata_md.py --path result/question_md.jsonl
```

3.Scoring

Scoring code: `score.py`
`--path` Result file (for text categories, use the converted file)
`--excel-file` Consistent with the format of ChatQA content
`--write-row` Consistent with the format of ChatQA content

```
python3 score.py --path result/question_md.jsonl_eval_md.jsonl --excel-file result/test.xlsx:mmtab --write-row 4

python3 score.py --path result/question_html.jsonl_eval_html.jsonl --excel-file result1test.xlsx:mmtab --write-row 5

python3 score.py --path result/MMTab-eval_test_data_49K.jsonl --excel-file result1test.xlsx:mmtab --write-row 6
```

4.View Evaluation Results

​	The results are saved in the `mmtab` sheet of the file `result/test.xlsx`.`--write-row` is consistent with the format of ChatQA content. A value of `-1` indicates a scoring error for that task, which may be due to zero results for the task.The metrics corresponding to each column are shown below:

```
- TABMWP: Acc.
- WTQ: Acc.
- HiTab: Acc.
- TAT-QA: Acc.
- FeTaQA: BLEU
- AIT-QA: Acc.
- TabMCQ: Acc.
- TabFact: Acc.
- infoTab: Acc.
- bHealthTiTab_T2: Acc.
- Rotowire: BLEU
- WikiBIO: BLEU
- TSD: Row Acc., Col. Acc.
- TCE: Acc.
- TCL: Acc.
- MCD: F1
- RCE: Row F1, Col. F1
```

5.One-click execution script
`bash table.sh`

​	Use the `table.sh` script to execute all the above steps in one click.
