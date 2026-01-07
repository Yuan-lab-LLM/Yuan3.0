from bert_score import score
import json
import os
import argparse
import pdb

parser = argparse.ArgumentParser(description="QwQ-32B")
parser.add_argument("--path", type=str, default="../inputs/251104/Summary/")
parser.add_argument("--output", type=str, default="outputs")
args = parser.parse_args()
path = args.path
output = args.output

data_names = ["big_patent_10_percent", "billsum", "cnn_dailymail", "pubmed-summarization", "samsum", "wikihow", "xsum"]
sum_keys = ["abstract", "summary", "highlights", "abstract", "summary", "summary", "summary"]

for data_name, summ_key in zip(data_names, sum_keys):
    #with open(f"/home/docker/epai/wangxinjing/162_wangxinjing/SummEval/Summary/{data_name}", "r", encoding="utf-8") as f:
    #with open(f"/home/docker/wangcarol/new_40b_0913/iter_000{ckpt}_nothink/Summary/final_res/{data_name}", "r", encoding="utf-8") as f:
    #    data = [json.loads(line) for line in f]
    '''
    data = []
    for file in os.listdir(f"{path}/{data_name}"):
        file_path = f"{path}/{data_name}/{file}"
        with open(file_path, "r", encoding="utf-8") as f:
            examples = [json.loads(line) for line in f]
        for item in examples:
            data.append(item)
    '''
    with open(f"{path}/{data_name}.jsonl", "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    cands = [item['solution'].replace("<|end_of_sentence|>", "").replace("<eod>", "").replace("</think>", "") for item in data]
    refs = [item[summ_key] for item in data]
    P, R, F1 = score(cands, refs, lang="en")
    print(f"{data_name}: {F1.mean():.4f}")
    with open(f"{output}/bert.txt", "a", encoding="utf-8") as f:
        f.write(f"{data_name}: {F1.mean():.4f}\n")

