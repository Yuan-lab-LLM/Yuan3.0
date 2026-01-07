#python3 MMTab-eval_evaluation/convert2evaldata_html.py --path /home/docker/epai/suntianxing/yuan_eval_rag/MMTab-eval_evaluation/result1/rag-case3_mmtab_yuanvl_1118_base6825/yuanvl_hf_35_iter_8pp/table_language/split/question_html.jsonl
import json
from tabulate import tabulate
from tqdm import tqdm
import argparse

def list_to_html_table_string(data, headers=None):
    """
    将 Python 列表转换为 HTML 表格字符串。

    Args:
        data: 包含表格数据的列表，列表的每个元素是一个字典或列表。
        headers: 可选，表头列表。如果为 None，则使用 data 中第一个元素作为表头。

    Returns:
        一个包含 HTML 表格的字符串。
    """
    html = "<table>"
    if headers:
        html += "<tr>"
        for header in headers:
            html += f"<th>{header}</th>"
        html += "</tr>"
    elif data:
        if isinstance(data[0], dict):
            headers = list(data[0].keys())
            html += "<tr>"
            for header in headers:
                html += f"<th>{header}</th>"
            html += "</tr>"
        elif isinstance(data[0], list):
           headers = data[0]
        #    headers = [f"Column {i+1}" for i in range(len(data[0]))]
           html += "<tr>"
           for header in headers:
                html += f"<th>{header}</th>"
           html += "</tr>"
           data = data[1:]


    for row in data:
        html += "<tr>"
        if isinstance(row, dict):
           for header in headers:
                html += f"<td>{row.get(header, '')}</td>"
        elif isinstance(row, list):
           for item in row:
                html += f"<td>{item}</td>"
        html += "</tr>"
    html += "</table>"
    return html

with open("MMTab-eval_test_data_49K.json", "r", encoding="utf-8") as f:
    MMTab_test_data = json.load(f)

with open("MMTab-eval_test_tables_23K.json", "r", encoding="utf-8") as f:
    examples = json.load(f)

MMtab_test_tables = {}
for item in tqdm(examples):
    MMtab_test_tables[item['image_id']] = list_to_html_table_string(item['table_rows'])

question_to_data_dict = {}
for item in MMTab_test_data:
    question_idx = item['input'].rfind("Question:")
    if question_idx == -1:
        question = item['input'].strip()
    else:
        question = item['input'][question_idx+len("Question:"):].strip()
    if question not in question_to_data_dict:
        question_to_data_dict[question] = [item]
    else:
        question_to_data_dict[question].append(item)


parser = argparse.ArgumentParser(description="QwQ-32B")
parser.add_argument("--path", type=str, default="questions_html.jsonl")
args = parser.parse_args()

prediction_path = args.path

with open(prediction_path, "r", encoding="utf-8") as f:
    examples = [json.loads(line) for line in f]

new_data = []

for item in examples:
    # item['solution'] = "test"
    # ori_item = json.loads(item['question'])
    # item['solution'] = item['model_result']
    ori_item = item
    item['prompt'] = ori_item['prompt']
    item['ground_truth'] = ori_item['ground_truth']
    question_idx = item['prompt'].rfind("Question:")
    assert question_idx != -1
    question = item['prompt'][question_idx + len("Question:"):].strip()
    # assert question in question_to_data_dict
    if question not in question_to_data_dict:
        continue
    question_data = question_to_data_dict[question]
    if len(question_data) == 1:
        new_item = question_data[0]
        new_item['solution'] = item['answer']
    else:
        table_content_idx = item['prompt'].find("Table Content:")
        table_content = item['prompt'][table_content_idx+len("Table Content:"):question_idx].strip()
        new_items = []
        for _item in question_data:
            if _item['output'] == item['ground_truth']:
                new_items.append(_item)
        # assert len(new_items) == 1
        if len(new_items) != 1:
            continue
        new_item = new_items[0]
        new_item['solution'] = item['answer']
    new_data.append(new_item)
print(f"Predicted Sample Number:",len(new_data))
# with open(r"D:\RAG\Table Understanding\yuan-outputs\20251015_res\ckpt_iter_000{}0_nothink_new\questions_html\eval_yuan_mtab_1015-html.jsonl".format(ckpt), "w", encoding="utf-8") as f:
with open(f"{prediction_path}_eval_html.jsonl", "w", encoding="utf-8") as f:
    for item in new_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
