from evaluation_utils import unanswerable_keyphrases
import json
from metrics import F1Metric
import copy
import re
import sys
import os
from write_to_excel import write_number_to_excel

# 绝对路径 > 相对路径
data_folder = "ChatQA/chatqa-data"

path_dict = {
    "doc2dial": os.path.join(data_folder, "doc2dial/test.json"),
    "convfinqa": os.path.join(data_folder, "convfinqa/dev.json"),
    "quac": os.path.join(data_folder, "quac/test.json"),
    "qrecc": os.path.join(data_folder, "qrecc/test.json"),
    "doqa_cooking": os.path.join(data_folder, "doqa/test_cooking.json"),
    "doqa_travel": os.path.join(data_folder, "doqa/test_travel.json"),
    "doqa_movies": os.path.join(data_folder, "doqa/test_movies.json"),
    "coqa": os.path.join(data_folder, "coqa/dev.json"),
    "hybridial": os.path.join(data_folder, "hybridial/test.json"),
    "sqa": os.path.join(data_folder, "sqa/test.json"),
    "topiocqa": os.path.join(data_folder, "topiocqa/dev.json"),
    "inscit": os.path.join(data_folder, "inscit/dev.json"),
}


def compute_f1_score(predicted_answers, groundtruth_answer, exp_name="default"):
    """Evaluating F1 Score"""
    print(len(predicted_answers), len(groundtruth_answer))
    if len(predicted_answers) != len(groundtruth_answer):
        groundtruth_answer = groundtruth_answer[:len(predicted_answers)]

    guess_list = []
    for guess in predicted_answers:
        guess = guess.strip()
        if "</s>" in guess:
            guess = guess.replace("</s>", "")
        guess_list.append(guess)

    answer_list = []
    for answer in groundtruth_answer:
        answer_list.append(answer)

    assert len(guess_list) == len(answer_list), \
        "lengths of guess and answer are different!"
    
    precision, recall, f1 = F1Metric.compute_all_pairs(guess_list, answer_list)
    print('Method: %s; Precision: %.4f; recall: %.4f; f1: %.4f' % (\
        exp_name, precision, recall, f1))
    return f1
    
def load_groundtruth_file(data_name, prompt_path):
    bos_token = "<|begin_of_text|>"
    data_file = path_dict[data_name]
    print(data_file)
    with open(data_file, "r") as f:
        examples = json.load(f)

    data = []
    for instance in examples:
        if "answers" in instance:
            answers = instance["answers"]
        elif "answer" in instance:
            if type(instance["answer"]) is str:
                answers = [instance["answer"]]
            elif type(instance["answer"]) is list:
                answers = instance["answer"]
            else:
                answers = [str(instance["answer"])]
        else:
            raise ValueError("need to have answer or answers")
        data.append(answers)
    
    prompt_list = []
    with open(prompt_path, 'rb') as f:
        for line in f.readlines():
            line = line.decode('utf-8')
            answer = line.strip()
            #answer = bos_token + answer
            prompt_list.append(answer)
    print("包含重复问题个数: ", len(data))
    assert len(prompt_list) == len(data), "Error, prompt not equal data"
    ground_truth_answers = {}
    for i in range(len(prompt_list)):
        if prompt_list[i] not in ground_truth_answers:
            ground_truth_answers[prompt_list[i].strip()] = data[i]
    return ground_truth_answers

def remove_special_char1(text):
    txt_list = text.split("</eog>")
    if len(txt_list) != 2:
        return '<the answer is true>'
    else:
        return txt_list[-1].strip()

def read_jsonl_files_to_dict(data_name,  folder_path):
    result_dict = {}
    path = os.path.join(folder_path, f"{data_name}.txt")
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                line_list = line.strip().split("[SEP]")
                if len(line_list) != 2:
                    continue
                #result_dict[line_list[0].strip()] = line_list[1].strip()
                #clean_answer = remove_special_char1(line_list[1])
                if line_list[0].strip() not in result_dict:
                    # result_dict[line_list[0].strip()] = line_list[1].strip()
                    ans_idx = line_list[1].strip().find("<|end_of_sentence|>")
                    result_dict[line_list[0].strip()] = line_list[1][:ans_idx]
            except json.JSONDecodeError as e:
                print(f"在文件 {path} 中解析JSON时出错: {e}")
    return result_dict
def read_jsonl_files_to_dict_1(data_name, folder_path):
    result_dict = {}
    # folder_path = os.path.join(folder_path, data_name)
    folders = []
    for entry in os.scandir(folder_path):
        if entry.is_dir() and entry.name.startswith(data_name):
            folders.append(entry.path)
    for folder in folders:
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            with open(file_path, 'r', encoding='utf-8') as f:
    # for file in os.listdir(folder_path):
    #     if file.endswith('.txt'):
    #         file_path = os.path.join(folder_path, file)
    #         with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        line_list = line.strip().split("<sep>")
                        if len(line_list) != 2:
                            continue
                        #result_dict[line_list[0].strip()] = line_list[1].strip()
                        #clean_answer = remove_special_char1(line_list[1])
                        if line_list[0].strip() not in result_dict:
                            # result_dict[line_list[0].strip()] = line_list[1].strip()
                            ans_idx = line_list[1].strip().find("<|end_of_sentence|>")
                            result_dict[line_list[0].strip()] = line_list[1][:ans_idx]
                    except json.JSONDecodeError as e:
                        print(f"在文件 {file} 中解析JSON时出错: {e}")
    return result_dict

def read_jsonl_files_to_dict1(data_name, base_path):
    result_dict = {}
    # base_path = r"D:\RAG\result\chatqa\chatqa-result-0324\res_rag_2\questions_2_new_split"
    print(base_path)
    for files in os.listdir(base_path):
        if files.startswith(data_name):
            file_path = os.path.join(base_path, files)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        line_list = json.loads(line)
                        answer_idx = line_list['solution'].rfind("</think>")
                        if answer_idx != -1:
                            answer = line_list['solution'][answer_idx+len("</think>"):].replace("<n>", "\n").strip()
                        else:
                            answer = line_list['solution'].replace("<n>", "\n").strip()
                        if line_list['question'].strip() not in result_dict:
                            result_dict[line_list['question'].strip()] = answer
                            # ans_idx = line_list[1].strip().find("<|end_of_sentence|>")
                            # result_dict[line_list[0].strip()] = line_list[1][:ans_idx]
                    except json.JSONDecodeError as e:
                        print(f"在文件 {files} 中解析JSON时出错: {e}")
            # folder_path = os.path.join(base_path, files)
            # for file in os.listdir(folder_path):
            #     file_path = os.path.join(folder_path, file)
            #     with open(file_path, 'r', encoding='utf-8') as f:
            #         for line in f:
            #             try:
            #                 line_list = line.strip().split("<sep>")
            #                 if len(line_list) != 2:
            #                     continue
            #                 #result_dict[line_list[0].strip()] = line_list[1].strip()
            #                 #clean_answer = remove_special_char1(line_list[1])
            #                 if line_list[0].strip() not in result_dict:
            #                     result_dict[line_list[0].strip()] = line_list[1].strip().replace("<|end_of_sentence|><eod>", "")
            #                     # ans_idx = line_list[1].strip().find("<|end_of_sentence|>")
            #                     # result_dict[line_list[0].strip()] = line_list[1][:ans_idx]
            #             except json.JSONDecodeError as e:
            #                 print(f"在文件 {file} 中解析JSON时出错: {e}")
    return result_dict 
def get_real_truth_answer(ground_truth_file, prediction_file):
    print("ground_truth: ", len(ground_truth_file))
    print("prediction: ", len(prediction_file))
    prediction = []
    ground = []
    for key in prediction_file:
        if key in ground_truth_file:
            #if prediction_file[key] == '<the answer is true>':
            #    prediction.append(ground_truth_file[key][0])
            #    ground.append(ground_truth_file[key])
            #    continue
            #else:
            prediction.append(prediction_file[key])
            ground.append(ground_truth_file[key])
    assert len(ground) == len(prediction), "error, ground is not equal prediction"
    print("最终测试数据个数: ", len(ground))
    return ground, prediction


def evaluate_f1(ground_truth_file, prediction_file):
    groundtruth_answers, predicted_answers = ground_truth_file, prediction_file
    # groundtruth_answers = load_groundtruth_file(ground_truth_file)
    # groundtruth_answers, predicted_answers = get_real_truth_answer(ground_truth_file, prediction_file)
    if "inscit" in ground_truth_file:
        count = 0
        groundtruth_answers_update = []
        for answers in groundtruth_answers:
            answers_update = []
            for ans in answers:
                ## this answer is additionally added to the answer_list for inscit dataset, needs to remove
                if ans != "Sorry. I cannot find the answer based on the context.":
                    answers_update.append(ans)
                else:
                    count += 1
            assert len(answers_update) > 0
            groundtruth_answers_update.append(copy.deepcopy(answers_update))
        groundtruth_answers = groundtruth_answers_update
        print(f"cannot answer {count}")
    # predicted_answers = load_prediction(prediction_file)
    if "quac" in prediction_file or "doqa" in prediction_file:
        predicted_answers_new = []
        for pred in predicted_answers:
            pred = pred.lower()
            for keyphrase in unanswerable_keyphrases:
                if keyphrase in pred:
                    pred = "Sorry. I cannot find the answer based on the context."
                    break
            predicted_answers_new.append(pred)
        predicted_answers = predicted_answers_new

    res = compute_f1_score(predicted_answers, groundtruth_answers)
    return res
def evaluate_convfinqa1(ground_truth_file, prediction_file):
    """
    Since the model will give a long answer output, while the gold answer for ConvFinQA are either
    a arithmetic formula or a final executed number.
    We consider the output containing either the executed number or the arithmetic formula as correct.
    This script is to measure the proportion of the outputs containing these elements.
    """

    def _is_float(string):
        try:
            float(string)
            return True
        except ValueError:
            return False
    with open(ground_truth_file, "r") as f:
        gold_list = json.load(f)

    bos_token = "<|begin_of_text|>"
    groundtruth_answers_1 = [item['exe_answer'] for item in gold_list]
    groundtruth_answers_formula_1 = [item['answers'][0] for item in gold_list]
    ## last turn question_list
    question_list_1 = [item['messages'][-1]['content'] for item in gold_list]

    # predicted_answers = load_prediction(prediction_file)

    prompt_list = []
    prompt_path = r"D:\RAG\code\evaluation_new\questions_2\convfinqa.txt"
    with open(prompt_path, 'rb') as f:
        for line in f.readlines():
            line = line.decode('utf-8')
            answer = line.strip()
            #answer = bos_token + answer
            prompt_list.append(answer)
    print("包含重复问题个数: ", len(groundtruth_answers_1))
    assert len(prompt_list) == len(groundtruth_answers_1), "Error, prompt not equal data"
    ground_truth_answers = {}
    for i in range(len(prompt_list)):
        if prompt_list[i] not in ground_truth_answers:
            ground_truth_answers[prompt_list[i].strip()] = [groundtruth_answers_1[i], groundtruth_answers_formula_1[i], question_list_1[i]]

    predicted_answers_1 = read_jsonl_files_to_dict(prediction_file)

    print("ground_truth: ", len(ground_truth_answers))
    print("prediction: ", len(predicted_answers_1))
    predicted_answers = []
    groundtruth_answers, groundtruth_answers_formula, question_list = [], [], []
    for key in predicted_answers_1:
        if key in ground_truth_answers:
            groundtruth_answers.append(ground_truth_answers[key][0])
            groundtruth_answers_formula.append(ground_truth_answers[key][1])
            question_list.append(ground_truth_answers[key][2])
            predicted_answers.append(predicted_answers_1[key])

    print(len(predicted_answers), len(groundtruth_answers))
    if len(predicted_answers) != len(groundtruth_answers):
        groundtruth_answers = groundtruth_answers[:len(predicted_answers)]

    count_exact_match = 0
    for question, pred, gold, gold_formula in zip(question_list, predicted_answers, groundtruth_answers, groundtruth_answers_formula):
        original_pred = pred
        ## convert 1,000,000 into 1000000
        original_pred = original_pred.replace(",", "")

        ## convert $10 million + $20 million into 10 + 20
        original_pred = original_pred.replace("$", "").replace("million", "").replace("billion", "")

        ## convert 10 (2017) + 20 (2018) into 10 + 20
        pattern = r'\((\b\w+\b)\)'
        original_pred = re.sub(pattern, '', original_pred)

        ## make sure it each token only has one space in between
        original_pred = " ".join(original_pred.split())

        if str(gold) in original_pred:
            count_exact_match += 1

        elif str(gold_formula) in original_pred:
            count_exact_match += 1

        elif _is_float(gold) and (str(round(float(gold), 3)) in original_pred or str(round(float(gold), 2)) in original_pred):
            count_exact_match += 1

        elif "percent" in question and (str(float(gold)*100) in original_pred or str(round(float(gold)*100, 1)) in original_pred or str(round(float(gold)*100, 2)) in original_pred):
            count_exact_match += 1

        elif str(gold).endswith(".0") and str(int(gold)) in original_pred:
            ## gold is a integer like 80.0 then convert it into 80
            count_exact_match += 1

        elif "decrease" in original_pred and _is_float(gold) and gold < 0 and (str(-1 * gold) in original_pred):
            ## for the case where model generates something like a decrese of 10 million, while gold is -10.
            count_exact_match += 1

    print("accuracy of exact match: %.4f" % (count_exact_match/len(predicted_answers)))

def evaluate_convfinqa(ground_truth_file, prediction_file):
    """
    Since the model will give a long answer output, while the gold answer for ConvFinQA are either
    a arithmetic formula or a final executed number.
    We consider the output containing either the executed number or the arithmetic formula as correct.
    This script is to measure the proportion of the outputs containing these elements.
    """

    def _is_float(string):
        try:
            float(string)
            return True
        except ValueError:
            return False
    with open(ground_truth_file, "r") as f:
        gold_list = json.load(f)

    bos_token = "<|begin_of_text|>"
    groundtruth_answers_1 = [item['exe_answer'] for item in gold_list]
    groundtruth_answers_formula_1 = [item['answers'][0] for item in gold_list]
    ## last turn question_list
    question_list_1 = [item['messages'][-1]['content'] for item in gold_list]

    # predicted_answers = load_prediction(prediction_file)

    prompt_list = []
    prompt_path = r"ChatQA/chatqa-questions/convfinqa.txt"
    with open(prompt_path, 'rb') as f:
        for line in f.readlines():
            line = line.decode('utf-8')
            answer = line.strip()
            #answer = bos_token + answer
            prompt_list.append(answer)
    print("包含重复问题个数: ", len(groundtruth_answers_1))
    assert len(prompt_list) == len(groundtruth_answers_1), "Error, prompt not equal data"
    ground_truth_answers = {}
    for i in range(len(prompt_list)):
        if prompt_list[i] not in ground_truth_answers:
            ground_truth_answers[prompt_list[i].strip()] = [groundtruth_answers_1[i], groundtruth_answers_formula_1[i], question_list_1[i]]

    predicted_answers_1 = read_jsonl_files_to_dict("convfinqa", prediction_file)

    print("ground_truth: ", len(ground_truth_answers))
    print("prediction: ", len(predicted_answers_1))
    predicted_answers = []
    groundtruth_answers, groundtruth_answers_formula, question_list = [], [], []
    for key in predicted_answers_1:
        if key in ground_truth_answers:
            groundtruth_answers.append(ground_truth_answers[key][0])
            groundtruth_answers_formula.append(ground_truth_answers[key][1])
            question_list.append(ground_truth_answers[key][2])
            predicted_answers.append(predicted_answers_1[key])

    print(len(predicted_answers), len(groundtruth_answers))
    if len(predicted_answers) != len(groundtruth_answers):
        groundtruth_answers = groundtruth_answers[:len(predicted_answers)]
    out_length = 0
    for item in predicted_answers:
        out_length += len(item.split())
    print(f"输出长度：{out_length/len(predicted_answers)}")
    count_exact_match = 0
    for question, pred, gold, gold_formula in zip(question_list, predicted_answers, groundtruth_answers, groundtruth_answers_formula):
        original_pred = pred
        ## convert 1,000,000 into 1000000
        original_pred = original_pred.replace(",", "")

        ## convert $10 million + $20 million into 10 + 20
        original_pred = original_pred.replace("$", "").replace("million", "").replace("billion", "")

        ## convert 10 (2017) + 20 (2018) into 10 + 20
        pattern = r'\((\b\w+\b)\)'
        original_pred = re.sub(pattern, '', original_pred)

        ## make sure it each token only has one space in between
        original_pred = " ".join(original_pred.split())

        if str(gold) in original_pred:
            count_exact_match += 1

        elif str(gold_formula) in original_pred:
            count_exact_match += 1

        elif _is_float(gold) and (str(round(float(gold), 3)) in original_pred or str(round(float(gold), 2)) in original_pred):
            count_exact_match += 1

        elif "percent" in question and (str(float(gold)*100) in original_pred or str(round(float(gold)*100, 1)) in original_pred or str(round(float(gold)*100, 2)) in original_pred):
            count_exact_match += 1

        elif str(gold).endswith(".0") and str(int(gold)) in original_pred:
            ## gold is a integer like 80.0 then convert it into 80
            count_exact_match += 1

        elif "decrease" in original_pred and _is_float(gold) and gold < 0 and (str(-1 * gold) in original_pred):
            ## for the case where model generates something like a decrese of 10 million, while gold is -10.
            count_exact_match += 1

    print("accuracy of exact match: %.4f" % (count_exact_match/len(predicted_answers)))
    return count_exact_match/len(predicted_answers)

def run_main(args):
    datasets = ["doc2dial", "quac", "qrecc", "coqa", "doqa_cooking", "doqa_movies", "doqa_travel", "convfinqa", "sqa", "topiocqa", "hybridial", "inscit"]
    excel_file = args.output.split(":")[0]
    table_name = args.output.split(":")[-1]
    write_row = args.row
    # datasets = ["quac"]
    idx = 4
    for data_name in datasets:
        print("============================================================")
        print(data_name)
        prompt_path = r"ChatQA/chatqa-questions/{}.txt".format(data_name)
        # prediction_path = r"D:\RAG\code\evaluation_new\chatqa-result-0324\res_rag\questions_2_new_split\{}".format(data_name)
        base_path = args.base_path
        if data_name in ["doc2dial", "qrecc", "topiocqa", "inscit", "coqa", "hybridial", "sqa"]:
            print("-"*80)
            ground_1 = load_groundtruth_file(data_name, prompt_path)
            # prediction_1 = read_jsonl_files_to_dict(prediction_path)
            prediction_1 = read_jsonl_files_to_dict(data_name, base_path)
            ground, prediction = get_real_truth_answer(ground_1, prediction_1)
            res = evaluate_f1(ground, prediction)
            write_number_to_excel(file_path=excel_file, value=res, row=write_row, column=idx, sheet_name=table_name)
            out_length = 0
            ground_length = 0
            for item in prediction:
                out_length += len(item.split())
            for item in ground:
                ground_length += len(item[0].split())
            print(f"输入长度：{ground_length/len(ground)}")
            print(f"输出长度：{out_length/len(prediction)}")
        if data_name in ["quac", "doqa_cooking", "doqa_movies", "doqa_travel"]:
            print("-"*80)
            ground_1 = load_groundtruth_file(data_name, prompt_path)
            # prediction_1 = read_jsonl_files_to_dict(prediction_path)
            prediction_1 = read_jsonl_files_to_dict(data_name, base_path)
            ground, prediction = get_real_truth_answer(ground_1, prediction_1)
            res = evaluate_f1(ground, prediction)
            write_number_to_excel(file_path=excel_file, value=res, row=write_row, column=idx, sheet_name=table_name)
            out_length = 0
            ground_length = 0
            for item in ground:
                ground_length += len(item[0].split())
            print(f"输入长度：{ground_length/len(ground)}")
            for item in prediction:
                out_length += len(item.split())
            print(f"输出长度：{out_length/len(prediction)}")
        if data_name == "convfinqa":
            print("-"*80)
            res = evaluate_convfinqa(path_dict[data_name], base_path)
            write_number_to_excel(file_path=excel_file, value=res, row=write_row, column=idx, sheet_name=table_name)
        idx += 1

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QwQ-32B")
    parser.add_argument("--base-path", type=str, default="")
    parser.add_argument("--row", type=int, default=2)
    parser.add_argument("--output", type=str, default="/home/docker/epai/wangxinjing/yuan_eval_rag/evaluation_rag/res-excel/test.xlsx:kimi")
    args = parser.parse_args()
    run_main(args)
