import json
import os
import logging
import argparse
import time
import re
# from get_boxed_eval import get_boxed_answer_text

def load_a_file(a_file, content=None):
    if not content:
        content = []
    with open(a_file, 'r', encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        content.extend(lines)
    return content


def get_file_list(folder, file_type_list):
    filelist = []
    for dirpath, _, filenames in os.walk(folder):
        for file in filenames:
            file_type = file.split('.')[-1]
            if file_type in file_type_list:
                file_fullname = os.path.join(dirpath, file)
                filelist.append(file_fullname)

    return filelist


def process_gen_files(gen_path_dir, len_ori, file_type_list):
    txt_files_lst = get_file_list(gen_path_dir, file_type_list)
    txt_files_lst = sorted(txt_files_lst, key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)

    content = []
    for i in range(len(txt_files_lst)):
        a_file = txt_files_lst[i]
        content = load_a_file(a_file, content)
    diff_len = len(content) - len_ori

    if diff_len > len(txt_files_lst) or diff_len <= 0:
        return content

    content_all = []
    for i in range(diff_len):
        a_file = txt_files_lst[i]
        content = load_a_file(a_file)
        content_all.extend(content[:-1])
    for i in range(diff_len, len(txt_files_lst)):
        a_file = txt_files_lst[i]
        content = load_a_file(a_file)
        content_all.extend(content)
    return content_all


def deal_box(str):
    # 定位出boxed{出现的地方
    start = str.rfind('boxed{') # 定位出boxed{出现的地方

    # 定位出最后一个}出现的地方
    end = str.rfind('}') # 定位出最后一个}出现的地方

    # 截取这两个位置之间的字符串
    if start == -1:
        res = "None"
    else:
        if str[start+6:].count('{') < str[start+6:].count('}'):
            res = str[start+6:end]
        else:
            res = str[start+6:]

    return res


def get_boxed_answer_text(a_text):
    a_text = a_text.strip().replace('\n', '<n>')
    if not a_text.strip():
        return None

    solution = a_text.replace('<think>', '<eog>').replace('</think>', '</eog>')

    if '<eog>' in solution and '</eog>' not in solution:
        return None
    elif "</eog>" not in solution:
        # answer = None
        realsolution = solution
    else:
        realsolution = solution.split("</eog>")[1]
        # if not realsolution.replace('<n>', '\n').strip():
        #     realsolution = '<n>'.join(solution.split("</eog>")[0].replace('<n>', '\n').strip().split('\n')[-3:])
    realsolution = realsolution.replace('<n>', '\n').strip()
    realsolution = re.sub(r'\n[\n\s]*(?:\n|$)', '\n', realsolution,re.DOTALL)
    realsolution = re.sub(r"boxed\s*\{", "boxed{", realsolution)
    realsolution = realsolution.replace('\n\n', '\n').replace('\n', '<n>')
    solution_split = realsolution.split('<n>')
    answer = None
    if 'boxed{' in realsolution:
        for item in reversed(solution_split):
            if 'boxed{' in item:
                if item.count('boxed{') == 1:
                    answer = deal_box(item)
                    answer = answer.replace(' ', '')
                elif item.count('boxed{') > 1:
                    answer = item
                break
    if not answer:
        answer = '\n'.join(solution_split[-5:])
        # answer = realsolution

    return answer




def is_numeric_string(s):
    if s.count('.') == 1:  # 允许只有一个小数点
        s = s.replace('.', '')
    return s.isdigit()


def judge_answer(generate_ans, ans_true):
    
    ans_true = ans_true.replace('<n>', '\n').strip().replace(' ', '').replace(',', '')
    # generate_ans = get_boxed_answer_text(answer)
    ans_flag = False
    if generate_ans:
        generate_ans = generate_ans.replace('\!', '').replace('\;', '').replace('\,', '').replace('\\%', '').replace('\\%', '').replace('\\ ', '')
        generate_ans = generate_ans.replace('\\]', '').replace('\\[', '').replace('\\(', '').replace('\\)', '').strip().rstrip('.')
        generate_ans = generate_ans.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\$", "").replace("\\mbox{cm}^2", "")
        #if re.findall(r'^[A-Z]$', ans_true):
        #    ans_lst = re.findall(r'(?:[A-D])(?![a-z])', generate_ans)
        if re.findall(r'[A-Z]',ans_true) and  re.findall(r'[A-Z]',ans_true)[0] == ans_true.strip():
            ans_lst = re.findall(r'(?<![a-zA-Z\d\+\-\*\/_=])(?:[A-Z])(?![a-zA-Z\d\+\-\*\/_=])', generate_ans.replace(' ', ''))
        else:
            ans_lst = re.findall(r'(?:(?:[-+]?\d+\.?\d+/\d+\.?\d+)|(?:[-+]?\d+/\d+)|(?:[-+]?\d+\.\d+)|(?:\d{1,5}(?:,\d{3})+(?:\.\d+)*?)|(?:[-+]?\d+))(?![\d+\-*/=()√₀\²³‰¼½¾_×¬^:±÷∶∧∨∑∏∪∷√≡≠><≤≥])', generate_ans)
        if len(ans_lst) == 1:
            generate_ans = ans_lst[0]
        generate_ans = generate_ans.replace(',', '').replace(' ', '')
        if generate_ans == ans_true:
            ans_flag = True

        elif is_numeric_string(ans_true):
            try:
                if float(ans_true) == float(generate_ans):
                    ans_flag = True
                elif abs(float(ans_true)) == abs(float(generate_ans)):
                    ans_flag = True
            except Exception as e:
                print(e)
        elif re.findall(r'[A-Z]',ans_true)[0] == ans_true.strip():
            find_ans = re.findall(r'(?<![a-zA-Z\d\+\-\*\/])[A-Z](?![a-zA-Z\d\+\-\*\/])',generate_ans)
            if len(find_ans) == 1:
                generate_ans = find_ans[0]
                if generate_ans == ans_true:
                    ans_flag = True


    return ans_flag
    


def get_boxed_answer(input_path, origin_file_path, output_path, file_type_list = ['jsonl', 'txt']):
    if not input_path or not os.path.exists(input_path):
        logging.error("The input evaluation path is incorrect.")
        return

    if not origin_file_path or not os.path.exists(origin_file_path):
        logging.error("The original file path is incorrect.")
        return
    file_name = os.path.basename(input_path).split('.')[0]
    print(f"Processing the {file_name} .....")
    # 提取原始问题和答案
    try:
        
        # file_name = '_'.join(os.path.basename(origin_file_path).split('.')[0].split('_')[:-1])
        print(f"origin_file_path:{origin_file_path}")
        content_qa = load_a_file(origin_file_path)
        content_yuan_all_dict = {}
        for i in range(len(content_qa)):
            g = re.sub(r'(<n>)+', '<n>', content_qa[i]).strip()
            g = g.replace('<sep>', '[SEP]')
            # g = re.sub(r"\s*(s\d{3})\s*", "\\1", g)
            # g = content_qa[i].strip()
            if g == "":
                continue
            g = g.split('[SEP]')
            
            content_yuan_all_dict[g[0].strip()] = g[1].replace(' ', '')
        len_ori = len(content_qa)
        print(f'len_ori:{len_ori}')
    except Exception as e:
        print('Error: ', e)

    if os.path.isfile(input_path):
        qa_content = load_a_file(input_path)
    # elif file_type_list == ['txt']:
    #     qa_content = []
    #     txt_files_lst = get_file_list(input_path, file_type_list)
    #     for i in range(len(txt_files_lst)):
    #         a_file = txt_files_lst[i]
    #         qa_content = load_a_file(a_file, qa_content)
    else:
        # qa_content = process_gen_files(input_path, len_ori, file_type_list)
        qa_content = []
        # 若原始问题无重复，则可以直接去重
        txt_files_lst = get_file_list(input_path, file_type_list)
        for i in range(len(txt_files_lst)):
            a_file = txt_files_lst[i]
            qa_content = load_a_file(a_file, qa_content)
        # qa_content = list(set(qa_content))
    
    print(f"len(qa_content) : {len(qa_content)}")
    
    
    output_file_true = os.path.join(output_path, f"{file_name}_judge", f"{file_name}_true.jsonl")
    output_file_false = os.path.join(output_path, f"{file_name}_judge", f"{file_name}_false.jsonl")
    os.makedirs(os.path.dirname(output_file_true), exist_ok=True)

    ans_true_num = 0

    dic = {}
    for line in qa_content:
        line = line.strip()
        if not line.strip():
            continue
        try:
            line = line.replace('<|begin_of_sentence|><|User|>', '').replace('<|Assistant|>', '[SEP]').replace('<think>','<eog>').replace('</think>', '</eog>').replace('<eod>', '').replace('<|end_of_sentence|>', '')
            line = line.replace('<sep>', '[SEP]')
            question, solution = line.split('[SEP]', 1)
            if '<eog>' in solution:
                solution = '<eog>' + solution.split('<eog>')[-1]
        except Exception as e:
            print(f"Error: {e}, line: {line}")
            continue
        
        question = re.sub(r'(?:<n>)+', '<n>', question)
        ans_true = ''
        for key in list(content_yuan_all_dict.keys()):
            if question in key or key in question:
                ans_true = content_yuan_all_dict[key]
                break
        if not ans_true:
            print(f'No correct answer found for question: "{question}".')
            continue
        dic = {"question": question, "answer": solution}
        solution = solution.replace('<think>', '<eog>').replace('</think>', '</eog>').replace('\b', '\\b')
        answer = get_boxed_answer_text(solution)
        ans_true = ans_true.replace('Answer:', '').replace('Answer：', '').replace('Answers:', '').replace('Answers：', '').strip().rstrip('.')
        if '<n>A.' in question and '<n>B.' in question and re.findall(r'^\d$', str(ans_true)):
            choice_dct = dict(zip(range(26), "ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
            ans_true = choice_dct[int(ans_true)]


        ans_true = ans_true.replace('答案：', '').replace('。', '')
        ans_flag = judge_answer(answer, ans_true)
        save_file = output_file_true if ans_flag else output_file_false
        # 添加一个新字段"predict"
        dic['predict'] = answer
        dic['ans_true'] = ans_true
        if ans_flag:
            ans_true_num += 1
        with open(save_file, 'a', encoding='utf-8') as f_new:
            f_new.write(json.dumps(dic, ensure_ascii=False) + '\n')
    print(f"ans_true_num: {ans_true_num}")
    print(f"accuracy: {round((ans_true_num * 100)/len(qa_content), 4)}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="", help='input file or folder')
    parser.add_argument("--origin_path", type=str, default="", help='origin file')
    parser.add_argument("--output_path", type=str, default="", help='output folder')
    parser.add_argument("--len_ori", type=str, default="1319", help='Number of lines in the pending evaluation files, default 1319(gsm8k test number)')
    args = parser.parse_args()
    assert args.input_path and args.output_path, "input_path and output_path must be specified"
    print(f"input_path :{args.input_path}\noutput_path :{args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)

    get_boxed_answer(args.input_path, args.origin_path, args.output_path)



if __name__ == "__main__":
    main()

# python judge_gsm_choice_easy.py --input_path /eval/datasets/aime-2024/ --output_path /your/output/path --origin_path /your/aime-2024.txt
