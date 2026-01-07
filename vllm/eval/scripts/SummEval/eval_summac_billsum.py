import argparse
import json
from summac.model_summac import SummaCConv
import numpy as np
import pdb
import os

key_dict = {
    "cnn_dailymail": "article",
    "pubmed-summarization": "article",
    "wikihow": "text",
    "samsum": "dialogue",
    "xsum": "dialogue",
    "big_patent_10_percent": "description",
    "billsum": "text"
}

# key: the source context
def get_args():
    parser = argparse.ArgumentParser(description="generate summary")
    parser.add_argument("--path", type=str, default="../inputs/251104/Summary")
    parser.add_argument("--output", type=str, default="outputs")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    data_names = ["billsum"]
    # data_names = ["billsum", "big_patent_10_percent", "cnn_dailymail", "pubmed-summarization"]
    keys = [key_dict[name] for name in data_names]
    args = get_args()
    # ckpt = args.ckpt
    
    path = args.path
    output = args.output

    model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda", start_file="default", agg="mean")

    for data_name, key in zip(data_names, keys):
        '''
        data = []
        for file in os.listdir(f"{path}/{data_name}"):
            file_path = f"{path}/{data_name}/{file}"
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line))
        '''
        with open(f"{path}/{data_name}.jsonl", "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        summary = [item['solution'].replace("<|end_of_sentence|>", "").replace("<eod>", "").replace("</think>", "") for item in data]
        source = [item[key] for item in data]
        summary = [str(s) for s in summary]
        source = [str(s) for s in source]
        score_conv1 = model_conv.score(source, summary)
        if not os.path.exists(f"{output}/npy"):
            os.makedirs(f"{output}/npy", exist_ok=True)
        np.save(f"{output}/npy/{data_name}_conv1_1031.npy", score_conv1['scores'])
        print(f"{data_name}: {sum(score_conv1['scores'])/len(score_conv1['scores']):.4f}")
        with open(f"{output}/summac.txt", "a", encoding="utf-8") as f:
            f.write(f"{data_name}: {sum(score_conv1['scores'])/len(score_conv1['scores']):.4f}\n")
