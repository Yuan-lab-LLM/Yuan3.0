import json
import os
import re
import logging
import argparse


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



def get_boxed_answer(input_path, output_path, len_ori, file_type_list = ['jsonl']):
    if not input_path or not os.path.exists(input_path):
        logging.error("The input evaluation path is incorrect.")
        return

    file_name = '.'.join(os.path.basename(input_path).split('.')[:-1])
    print(f"Processing the {file_name} .....")
    
    if os.path.isfile(input_path):
        qa_content = load_a_file(input_path)
    elif input_path == ['txt']:
        qa_content = []
        txt_files_lst = get_file_list(input_path, file_type_list)
        for i in range(len(txt_files_lst)):
            a_file = txt_files_lst[i]
            qa_content = load_a_file(a_file, qa_content)
    else:
        # qa_content = process_gen_files(input_path, len_ori, file_type_list)
        
        # 若原始问题无重复，则可以直接去重
        txt_files_lst = get_file_list(input_path, file_type_list)
        for i in range(len(txt_files_lst)):
            a_file = txt_files_lst[i]
            qa_content = load_a_file(a_file, qa_content)
        qa_content = list(set(qa_content))

    print(f"len(qa_content) : {len(qa_content)}")
    
    save_file = os.path.join(output_path, f"{file_name}.jsonl")

    with open(save_file, 'w', encoding='utf-8') as f_new:
        dic = {}
        for line in qa_content:
            line = line.strip()
            if not line.strip():
                continue
            try:
                line = line.replace('<sep>', '[SEP]')
                question, solution = line.split('[SEP]', 1)
            except Exception as e:
                print(f"Error: {e}, line: {line}")
                continue
            solution = solution.replace('<think>', '<eog>').replace('</think>', '</eog>')
            dic = {"question": question, "solution": solution}
            
            if "</eog>" not in solution:
                answer = "None"
            else:
                realsolution = solution.split("</eog>")[1]
                solution_split = realsolution.split('<n>')
                answer = "None"
                for item in reversed(solution_split):
                    if 'boxed{' in item:
                        answer = deal_box(item)
                        break
                        
            # 添加一个新字段"predict"
            dic['predict'] = answer

            # 写入新文件
            f_new.write(json.dumps(dic, ensure_ascii=False) + '\n')
    print(f"Save to {save_file} successfully.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="", help='input file or folder')
    parser.add_argument("--output_path", type=str, default="", help='output folder')
    parser.add_argument("--len_ori", type=str, default="1319", help='Number of lines in the pending evaluation files, default 1319(gsm8k test number)')
    args = parser.parse_args()
    assert args.input_path and args.output_path, "input_path and output_path must be specified"
    print(f"input_path :{args.input_path}\output_path :{args.output_path}")

    get_boxed_answer(args.input_path, args.output_path, int(args.len_ori), file_type_list = ['jsonl'])



if __name__ == "__main__":
    main()

# python get_boxed_eval.py --input_path /your/aime-2024.txt --output_path /your/output/path
