import json
from tabulate import tabulate
from tqdm import tqdm
import argparse

with open("MMTab-eval_test_data_49K.json", "r", encoding="utf-8") as f:
    MMTab_test_data = json.load(f)

with open("MMTab-eval_test_tables_23K.json", "r", encoding="utf-8") as f:
    examples = json.load(f)
MMtab_test_tables = {}
for item in tqdm(examples):
    MMtab_test_tables[item['image_id']] = tabulate(item['table_rows'], tablefmt="pipe", headers='firstrow')

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
parser.add_argument("--path", type=str, default="/home/docker/epai/wangxinjing/yuan_eval_rag/results/kimi_result/md/questions_md.jsonl")
args = parser.parse_args()

prediction_path = args.path
with open(prediction_path, "r", encoding="utf-8") as f:
    examples = [json.loads(line) for line in f]

new_data = []

for item in examples:
    # ori_item = json.loads(item['question'])
    ori_item = item
    # item['solution'] = item['model_result']
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
            if _item['output'] == item['ground_truth'] and MMtab_test_tables[_item['image_id']] == table_content:
                new_items.append(_item)
        # assert len(new_items) == 1
        if len(new_items) != 1:
            continue
        new_item = new_items[0]
        new_item['solution'] = item['answer']
    new_data.append(new_item)
print(f"Predicted Sample Number:",len(new_data))
# with open(r"D:\RAG\yuan-results\yuan-mmtab-250926\ckpt_iter_000{}00\MMTab-questions\questions_md\eval_yuan_mtab_0928-md.jsonl".format(ckpt), "w", encoding="utf-8") as f:
with open(f"{prediction_path}_eval_md.jsonl", "w", encoding="utf-8") as f:
    for item in new_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


#python3 MMTab-eval_evaluation/convert2evaldata_md.py --path /home/docker/epai/suntianxing/yuan_eval_rag/MMTab-eval_evaluation/result1/rag-case3_mmtab_yuanvl_1118_base6825/yuanvl_hf_35_iter_8pp/table_language/split/question_md.jsonl
