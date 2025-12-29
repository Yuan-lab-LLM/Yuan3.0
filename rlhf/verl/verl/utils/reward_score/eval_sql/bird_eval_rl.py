import sys
import os
import re
import json
import argparse
import traceback

def format_data(one_json):
    answer = one_json['answer']
    db_id = one_json['db_id']
    gold = f'{answer}\t----- bird -----\t{db_id}'
    solution = one_json['yuan_solution']
    result = one_json['yuan_solution'].split('</think>')[-1]
    pattern = r'```sql.*?\n(.*?)```'
    code_match = re.search(pattern, result, re.DOTALL)
    if  code_match:
        predict = code_match.group(1).strip()
    else:
        predict = one_json['yuan_solution'].split('</think>')[-1].replace('<|end_of_sentence|>','').replace('<eod>','').replace("<｜end▁of▁sentence｜>","").strip()

    key = one_json['question_id']
    one_json['gold'] = {key: gold}
    if predict:
        value = predict.replace("\n", " ") + "\t----- bird -----\t" + gold.split("\t----- bird -----\t")[1]
        key = one_json['question_id']
        one_json['predict'] = {key: value}

    return one_json


def bird_eval_one_json(one_json):
    one_json["yuan_solution"] = one_json["response"]
    one_json = format_data(one_json)
    sys.path.append('/home/bird_src')
    from bird_evaluation_hhb import bird_eval
    db_root_path = '/home/data/train_databases/'
    num_cpus = 1
    meta_time_out = 30.0

    try:
        predict = one_json['predict']
        gold = one_json['gold']
        acc = bird_eval(one_json['predict'], one_json['gold'], db_root_path, num_cpus, meta_time_out)
    except Exception as e:
        print("Error:",e)
        print("Traceback:",traceback.format_exc())
        acc=0.0

    return acc
    
