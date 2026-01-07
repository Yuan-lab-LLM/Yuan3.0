python3 convert2evaldata_html.py --path result1/questions_html.jsonl
python3 convert2evaldata_md.py --path result1/questions_md.jsonl
python3 score.py --path result1/questions_md.jsonl_eval_md.jsonl --excel-file /result1/test.xlsx:mmtab --write-row 1
python3 score.py --path result1/questions_html.jsonl_eval_html.jsonl --excel-file /result1/test.xlsx:mmtab --write-row 2
python3 score.py --path result1/MMTab-eval_test_data_49K.jsonl --excel-file /result1/test.xlsx:mmtab --write-row 3

