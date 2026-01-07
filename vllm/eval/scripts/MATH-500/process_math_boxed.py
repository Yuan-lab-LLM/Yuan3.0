import json
import os
import logging
import argparse
import time
import re
import json
from get_boxed_eval import get_boxed_answer_text

def load_a_file(a_file, content=None):
    if not content:
        content = []
    with open(a_file, 'r', encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        try:
            data = json.loads(lines[0])
            print(data)
            for line in lines:
                data = json.loads(line)
                # if data['solutions']['final_solution']:
                if "solutions" in data and 'final_solution' in data['solutions']:

                    content.append(data['solutions']['final_solution'])
                else:
                    content.append("")
        except:
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


def is_numeric_string(s):
    if s.count('.') == 1:  # 允许只有一个小数点
        s = s.replace('.', '')
    return s.isdigit()


def _fix_fracs(string):
    if "\\frac" not in string:
        return string

    substrs = string.split("\\frac")

    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def process_text(text):
    if not text:
        return text
    origin_text = text
    text = text.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\$", "").replace("$", "").replace('\\ ', '').replace("tfrac", "frac").replace("dfrac", "frac").replace("\\left", "").replace("\\right", "").replace("\!", "").replace('\\mbox{inches}^2', '').replace("\\mbox{cm}^2", "").replace('\;', '').replace('\,', '').replace('\\%', '').replace('%', '')
    
    origin_text = text.replace('\\cup', '∪').replace('\\cap', '∩').replace("\\leqslant", "\\leq").replace("\\geqslant", "\\geq").replace("\\le ", "\\leq").replace("\\ge ", "\\geq")
    
    if 'frac' in text:
        try:
            text = _fix_fracs(origin_text)
        except Exception as e:
            print('Error: ', e)
            text = origin_text

    if 'sqrt' in text:
        try:
            text = _fix_sqrt(origin_text)
        except Exception as e:
            print('Error: ', e)
            text = origin_text
    
    text = text.replace('\\]', '').replace('\\[', '').replace('\\(', '').replace('\\)', '').strip().rstrip('.')
    
    
    return text


def get_boxed_answer(input_path, origin_file_path, output_path, file_type_list = ['jsonl', 'txt']):
    if not input_path or not os.path.exists(input_path):
        
        logging.error("The input evaluation path is incorrect." + f"input_path:{input_path};{os.path.exists(input_path)}; {not input_path}; {not input_path or not os.path.exists(input_path)}")
        return None, None

    if not origin_file_path or not os.path.exists(origin_file_path):
        logging.error("The original file path is incorrect.")
        return None, None

    # 提取原始问题和答案
    try:
        
        # file_name = '_'.join(os.path.basename(origin_file_path).split('.')[0].split('_')[:-1])
        print(f"origin_file_path:{origin_file_path}")
        content_qa = load_a_file(origin_file_path)
        content_yuan_all_dict = {}
        for i in range(len(content_qa)):
            g = re.sub(r'(<n>)+', '<n>', content_qa[i]).strip()
            # g = re.sub(r"\s*(s\d{3})\s*", "\\1", g)
            # g = content_qa[i].strip()
            if g == "":
                continue
            g = g.replace('<SEP>', '<sep>').replace('<sep>', '[SEP]').split('[SEP]')
            g[0] = g[0].replace('<n>', '\n').strip() 
            content_yuan_all_dict[g[0].strip()] = g[1] #.replace(' ', '')
        len_ori = len(content_qa)
        print(f'len_ori:{len_ori}')
    except Exception as e:
        print('Error: ', e)
    file_name = os.path.basename(input_path).split('.')[0]
    print(f"Processing the {file_name} .....")
    if os.path.isfile(input_path):
        qa_content = load_a_file(input_path)
    else:
        # qa_content = process_gen_files(input_path, len_ori, file_type_list)
        qa_content = []
        # file_name = os.path.basename(os.path.dirname(input_path))
        # 若原始问题无重复，则可以直接去重
        txt_files_lst = get_file_list(input_path, file_type_list)
        for i in range(len(txt_files_lst)):
            a_file = txt_files_lst[i]
            qa_content = load_a_file(a_file, qa_content)
        # qa_content = list(set(qa_content))
    
    print(f"len(qa_content) : {len(qa_content)}")
    
    
    # output_file_true = os.path.join(output_path, f"{file_name}_true.jsonl")
    # output_file_false = os.path.join(output_path, f"{file_name}_false.jsonl")
    output_prefix = os.path.join(output_path, f"{file_name}_")

    content_generate = []
    for line in qa_content:
        line = line.strip()
        if not line.strip():
            continue
        try:
            line = line.replace('<|begin_of_sentence|><|User|>', '').replace('<|Assistant|>', '[SEP]').replace('<think>','<eog>').replace('</think>', '</eog>').replace('<|end_of_sentence|>', '').replace('<SEP>', '<sep>').replace('<sep>', '[SEP]')
            #if '<eop>' in line:
            line = line.replace('<eop>', '')
            question, solution = line.split('[SEP]', 1)
            # question = question.split('<n>Question:<n>')[-1]
            question = question.strip()
            #if '<eog>' in solution and '</eog>' in solution:
            #    solution = '<eog>' + solution.split('<eog>')[-1]
            #else:
            solution = solution.split('<eog>')[-1].split('[SEP]')[-1]
            solution = solution.replace('<eod>', '')
        except Exception as e:
            print(f"Error: {e}, line: {line}")
            continue
        
        question = re.sub(r'(?:<n>)+', '<n>', question)
        question = question.replace('<n>', '\n').strip()
        ans_true = ''
        for key in list(content_yuan_all_dict.keys()):
            if question in key or key in question:
                ans_true = content_yuan_all_dict[key]
                break
        if ans_true == '':
            print(f'No correct answer found for question: "{question}".')
            continue
        solution_new = line.split('[SEP]', 1)[1]
        # solution_new = solution if solution_new.endswith('<eod>') else solution_new
        dic = {"question": question, "answer": solution_new}
        solution = solution.replace('<think>', '<eog>').replace('</think>', '</eog>')
        # print(f"solution: {solution}")
        answer = get_boxed_answer_text(solution)
        ans_true = ans_true.replace('Answer:', '').replace('Answer：', '').replace('Answers:', '').replace('Answers：', '').strip().rstrip('.')
        ans_true = ans_true.replace('答案：', '').replace('。', '')

        ans_true = process_text(ans_true)# .replace(' ', '')
        
        if answer:
            answer = process_text(answer)
            if is_numeric_string(ans_true):
                ans_lst = re.findall(r'(?:[-+]?\d+\.?\d+/\d+\.?\d+)|(?:[-+]?\d+/\d+)|(?:[-+]?\d+\.\d+)|(?:\d{1,5}(?:,\d{3})+(?:\.\d+)*?)|(?:[-+]?\d+)', answer)
                ans_lst_2 = re.findall(r'(?:[-+]?\d+\.?\d+/\d+\.?\d+)|(?:[-+]?\d+/\d+)|(?:[-+]?\d+\.\d+)|(?:\d{1,5}(?:,\d{3})+(?:\.\d+)*?)|(?:\d+_\d)|(?:[-+]?\d+)', answer)
                if len(ans_lst) == 1:
                    answer = ans_lst[0]
                elif len(ans_lst_2) == 1:
                    answer = ans_lst[0].split('_')[0]
        #if not solution_new.endswith('<eod>'):
        #    dic['predict'] = "None"
        #else:
        #    dic['predict'] = answer
        dic['predict'] = answer 
        dic['ans_true'] = ans_true
        content_generate.append(dic)
        with open(output_prefix + '_getboxed.jsonl', 'a', encoding='utf-8') as f:
            json.dump(dic, f, ensure_ascii=False)
            f.write('\n')
    print(f"len(content_generate) : {len(content_generate)}")
    return content_generate, output_prefix


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


