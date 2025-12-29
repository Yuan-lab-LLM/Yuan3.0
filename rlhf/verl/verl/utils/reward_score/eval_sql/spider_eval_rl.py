import sys
import os
import re
import json
import argparse



def format_data(one_json):
    answer = one_json['answer']
    db_id = one_json['db_id']
    gold = answer + '\t' + db_id
    solution = one_json['yuan_solution']
    result = one_json['yuan_solution'].split('</think>')[-1]
    pattern = r'```sql.*?\n(.*?)```'
    code_match = re.search(pattern, result, re.DOTALL)
    if  code_match:
        predict = code_match.group(1).strip()
    else:
        predict = one_json['yuan_solution'].split('</think>')[-1].replace('<|end_of_sentence|>','').replace('<eod>','').replace("<｜end▁of▁sentence｜>","").strip()

    one_json['gold'] = gold
    if predict:
        one_json['predict'] = predict.replace("\n", " ").replace(";", "")
    else:
        one_json['predict'] = ''

    return one_json


def spider_eval_one_json(one_json):
    one_json["yuan_solution"]=one_json["response"]
    one_json = format_data(one_json)

    sys.path.append('/home/spider-master')
    from spider_evaluation_hhb import spider_eval


    db_dir="/home/spider_data/database/"
    table_file="/home/spider_data/tables.json"

    tmp_glist = [one_json['gold']]
    tmp_plist = [one_json['predict']]

    predict = one_json['predict']
    gold = one_json['gold']
    acc = spider_eval(tmp_glist, tmp_plist, db_dir, table_file)
    return acc
