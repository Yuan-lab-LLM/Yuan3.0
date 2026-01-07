import sys
sys.path.append("MMTab-eval_evalution")

import json
import random
import re
import tqdm
from collections import defaultdict
from sacrebleu.metrics import BLEU
from metric import TEDS
from eval_task import *
from write2xlsx import write_number_to_excel
import argparse

def convert_table_to_html_str(table_row_list=[]):
    """
    Given a list of table rows, build the corresponding html string, which is used to compute the TEDS score.
    We use the official code of PubTabNet to compute TEDS score, it does not consider '<th>' label.
    We also remove unneccessary spaces within a table cell and extra '\n' as they will influence the TEDS score.
    """
    html_table_str = "<html><body><table>" + '\n'
    for data_row in table_row_list:
        html_table_str += "<tr>"
        for cell_str in data_row:
            html_table_str += f"<td>{cell_str}</td>"
        html_table_str += "</tr>"
        html_table_str += '\n'
    html_table_str += "</table></body></html>"
    html_table_str = html_table_str.replace('\n','')
    return html_table_str

def convert_markdown_table_to_html(markdown_table):
    """
    Converts a markdown table to the corresponding html string for TEDS computation.
    """
    # remove extra code block tokens like '```markdown' and '```
    markdown_table = markdown_table.strip('```markdown').strip('```').strip() 
    row_str_list = markdown_table.split('\n')
    # extra the first header row and other data rows
    valid_row_str_list = [row_str_list[0]]+row_str_list[2:]
    table_rows = []
    for row_str in valid_row_str_list:
        one_row = []
        for cell in row_str.strip().split('|')[1:-1]:
            if set(cell) != set(' '):
                one_row.append(cell.strip())
            else:
                one_row.append(' ')
        table_rows.append(one_row)
    # build html string based on table rows
    html_str = convert_table_to_html_str(table_rows)
    return html_str

def convert_latex_table_to_html(latex_table):
    """
    Converts a markdown table to html string for TEDS computation.
    In the MMTab-eval, we only consider latex tables with similar structures of markdown tables.
    For other latex tables with compicated structures like merged cells, you need to rewrite this function to convert them.
    """
    # remove extra code block tokens like '```latex' and '```
    latex_table = latex_table.strip('```latex').strip('```').strip() 
    latex_table = latex_table.replace('\n', ' ')
    row_str_list = [row_str.strip('\n').strip('\\') for row_str in latex_table.split('\\hline')[1:-1]]
    table_rows = []
    for row_str in row_str_list:
        one_row = []
        for c in row_str.split('&'):
            if set(c) != set(' '):
                one_row.append(c.strip())
            else:
                one_row.append(' ')
        table_rows.append(one_row)
    html_str = convert_table_to_html_str(table_rows)
    return html_str

def wrap_html_table(html_table):
    """
    The TEDS computation from PubTabNet code requires that the input html table should have <html>, <body>, and <table> tags.
    Add them if they are missing.
    """
    html_table = html_table.replace('\n','')
    # add missing <table> tag if missing
    if "<table" in html_table and "</table>" not in html_table:
        html_table = html_table + "</table>"
    elif "<table" not in html_table and "</table>" in html_table:
        html_table = "<table>" + html_table
    elif "<table" not in html_table and "</table>" not in html_table:
        html_table = "<table>" + html_table + "</table>"
    else:
        pass
    # add <body> and <html> tags if missing
    if '<body>' not in html_table:
        html_table = '<body>' + html_table + '</body>'
    if '<html>' not in html_table:
        html_table = '<html>' + html_table + '</html>'
    return html_table

# Read inference results of LLaVA model (merged.jsonl)
def read_llava_prediction_file(file_path):
    """
    Read LLaVA's inference results (e.g., merge.jsonl) and extract data of different benchmarks based on 'category' field.
    """
    predict_results = []
    # with open(file_path, 'r', encoding='utf-8') as f:
    #     examples = json.load(f)
    # for item in examples:
    #     item = convert2json(item)
    #     predict_results.append(item)
    with open(file_path, encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm.tqdm(lines):
            item = json.loads(line.strip())
            item = convert2json(item)
            predict_results.append(item)
    print("Predicted Sample Number:",len(predict_results))
    benchmark_name_to_predicted_item_list = defaultdict(list)
    for item in predict_results:
        item_id = item['question_id']
        category = item['category'] # {dataset_name}_for_{task_name}, e.g., TabFact_for_TFV
        dataset_name = category.split('_for_')[0] # e.g., TabFact
        task_name = category.split('_for_')[1] # e.g., TFV
        # for table structure understanding tasks, benchmark name is the task name
        if task_name not in ['TSD','TCL','RCE','MCD','TCE','TR','OOD_TSD','OOD_TCL','OOD_RCE','OOD_TCE']:
            benchmark_name = dataset_name
        else:
            benchmark_name = task_name
        benchmark_name_to_predicted_item_list[benchmark_name].append(item)
    for benchmark_name,  predicted_item_list in benchmark_name_to_predicted_item_list.items():
        item_num = len(predicted_item_list)
        print(f'benchmark name: {benchmark_name}, test data num: {item_num}')
    return benchmark_name_to_predicted_item_list

def convert2json(item):
    new_item = {}
    new_item['question_id'] = item['item_id']
    new_item['image'] = f"{item['image_id']}.jpg"
    new_item['input'] = item['input']
    # new_item['text'] = item['solution'].replace("<|end_of_sentence|>", "").replace("<eod>", "") if item['solution'] else item['solution']
    new_item['text'] = item['solution'].split('</think>')[-1] if item['solution'] else item['solution']
    new_item['category'] = f"{item['dataset_name']}_for_{item['task_type']}"
    new_item['original_query_type'] = item['original_query_type']
    # new_item['llm_response'] = item['solution']
    return new_item

parser = argparse.ArgumentParser(description="QwQ-32B")
parser.add_argument("--path", type=str, default="/home/docker/epai/wangxinjing/yuan_eval_rag/results/kimi_result/md/questions_md.jsonl_eval_md.jsonl")
parser.add_argument("--excel-file", type=str, default="/home/docker/epai/wangxinjing/yuan_eval_rag/results/kimi_result/md/test.xlsx:mmtabmd")
parser.add_argument("--write-row", type=int, default=1)
args = parser.parse_args()

benchmark_name_to_predicted_item_list = read_llava_prediction_file(args.path)

# benchmark_name_list = ['TSD','TCL','RCE','MCD','TCE','OOD_TSD','OOD_TCE','OOD_TCL','OOD_RCE','TABMWP','WTQ','HiTab','TAT-QA','TabFact','InfoTabs','AIT-QA','PubHealthTab','TabMCQ','FeTaQA','HiTab_t2t','Rotowire','WikiBIO']
benchmark_name_list = ["TABMWP", "WTQ", "HiTab", "TAT-QA", "FeTaQA", "AIT-QA", "TabMCQ", "TabFact", "InfoTabs", "PubHealthTab", "HiTab_t2t", "Rotowire", "WikiBIO", "TSD", "TCE", "TCL", "MCD", "RCE"]

excel_file = args.excel_file.split(":")[0]
table_name = args.excel_file.split(":")[-1]
write_row = args.write_row
# datasets = ["quac"]
idx = 2
# benchmark_name_list = ["HiTab_t2t", "MCD", "TABMWP"]
# benchmark_name_list = ["FeTaQA"]
# benchmark_name_list = ["WTQ", "HiTab", "FeTaQA", "TabFact", "InfoTabs"]
for benchmark_name in benchmark_name_list:
    try:
        predicted_item_list = benchmark_name_to_predicted_item_list[benchmark_name]
        if benchmark_name in ['TSD','OOD_TSD']:
            scores = evaluate_tsd_questions(benchmark_name,predicted_item_list)
        elif benchmark_name in ['TCE','OOD_TCE']:
            scores = evaluate_tce_questions(benchmark_name,predicted_item_list)
        elif benchmark_name in ['TCL','OOD_TCL']:
            scores = evaluate_tcl_questions(benchmark_name,predicted_item_list)
        elif benchmark_name in ['RCE','OOD_RCE']:
            scores = evaluate_rce_questions(benchmark_name,predicted_item_list)
        elif benchmark_name == 'MCD':
            scores = evaluate_mcd_questions(benchmark_name,predicted_item_list)
        elif benchmark_name == 'TabMCQ':
            scores = evaluate_tabmcq_questions(benchmark_name,predicted_item_list)
        elif benchmark_name in ["FeTaQA", 'HiTab_t2t','Rotowire','WikiBIO']:
            scores = evaluate_text_generation_questions(benchmark_name,predicted_item_list)
        # elif benchmark_name == "FeTaQA":
        #     evaluate_FeTaQA(benchmark_name,predicted_item_list)
        else:
            print(benchmark_name)
            scores = evaluate_tqa_questions(benchmark_name,predicted_item_list)
    except Exception as e:
        #scores = [-1]
        if benchmark_name in ['TSD', 'OOD_TSD', 'RCE','OOD_RCE']:
            scores = [-1, -1]
        else:
            scores = [-1]
        print(benchmark_name)
        print(e)
    for score in scores:
        write_number_to_excel(file_path=excel_file, value=score, row=write_row, column=idx, sheet_name=table_name)
        idx += 1
