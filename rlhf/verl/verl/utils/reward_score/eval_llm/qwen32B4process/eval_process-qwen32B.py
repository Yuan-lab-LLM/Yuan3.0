#!/usr/bin/env python
# -*- coding: utf-8 -*-

# QWEN3-32B AS SUPERVISED MODEL

import os
import re
import json
import time
import threading
import argparse
from openai import OpenAI
import numpy as np
from thefuzz import fuzz, process
from concurrent import futures
from .tools import TellMeTime
import multiprocessing
from .logger import Logging
from .tools import combine_small_lst, combine_small_lst_mllm

logging = Logging(limit_level='debug', colorful=True)
lock = threading.Lock()
LLM_PROCESS_METHOD_FLAG = ['llm_math','llm_choice','mllm_normal','mllm_choice']

def print_thread_count():
    # 获取当前活跃的线程数
    thread_count = threading.active_count()
    print(f"当前活跃线程数: {thread_count}")

    # 获取所有线程的详细信息
    print("所有线程信息:")
    for thread in threading.enumerate():
        print(f"  - {thread.name} (ID: {thread.ident}, 状态: {'alive' if thread.is_alive() else 'dead'})")


def locate_verification_start(text: str, custom_patterns: list = None) -> int:

    MODAL_VERBS = [
        'should', 'must', 'need to', 'have to', 'will', 'would',
        'can', 'could', 'might', 'may', 'us', "let's"
    ]

    VERBS = [
        'double-check', 'cross-verify', 'double check', 'cross verify', 're-verify', 're-check',
        'check', 'confirm', 'verify', 'validate', 'examine', 'review',
        'inspect', 'recap', 'verification', "think again", "try again",
        'make sure', 'reconfirm', 'authenticate', 're-evaluate'
    ]

    # 构建灵活的模式结构（按长度排序确保优先匹配长短语）
    base_pattern = (
        r'(?i)(?<!\S)'  # 前面必须是单词边界（空格或开头）
        r'(?:' + '|'.join(sorted(MODAL_VERBS, key=len, reverse=True)) + r')?\s*'
        r'(' + '|'.join(sorted(VERBS, key=len, reverse=True)) + r')\b'
    )

    # 合并自定义模式
    patterns = [base_pattern]
    if custom_patterns:
        patterns.extend(custom_patterns)

    # 搜索匹配项
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            return match.start(1)  # 返回验证动词的起始位置

    return -1


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

        realsolution = solution
    else:
        realsolution = solution.split("</eog>")[1]

    realsolution = realsolution.replace('<n>', '\n').strip()
    realsolution = re.sub(r'\n[\n\s]*(?:\n|$)', '\n', realsolution,re.DOTALL)
    realsolution = re.sub(r"boxed\s*\{", "boxed{", realsolution)
    realsolution = realsolution.replace('\n\n', '\n').replace('\n', '<n>')
    solution_split = realsolution.split('<n>')
    answer = None
    if 'boxed{' in realsolution:
        for item in reversed(solution_split):
            if 'boxed{' in item:
                answer = item
                break
    if not answer:
        answer = '\n'.join(solution_split[-5:])

    return answer


def get_answer(line):
    items = line.strip().replace('\n','<n>').split('<n>')
    result = []

    # 倒序遍历查找
    for i, item in enumerate(reversed(items)):
        if not item.strip():
            continue
        if 'boxed' in item:
            result.append(item)
        elif result:
            break

    if result:
        return '\n'.join(reversed(result)).strip()
    else:
        return ''

def qwen_vllm_inference(input_text, api, model_path="Qwen3-32B", max_tokens=2048, temperature=0.6, top_p=0.95):

    input_text = input_text.replace('<n>', '\n')
    openai_api_key = "EMPTY"

    openai_api_base = f"http://{api}/v1"
    CLIENT = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        timeout=1000000,
    )
    response = CLIENT.chat.completions.create(
        model=model_path,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": input_text},
            ],
        }
        ],
        extra_body= {
            "include_stop_str_in_output": True,
            "skip_special_tokens": False
        },
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,

    )
    text = response.choices[0].message.content#.replace('\n','<n>')
    think_text = response.choices[0].message.reasoning_content#.replace('\n','<n>')
    think_text = '' if not think_text else think_text
    if text:
        res_text = '<think>' + think_text + '</think>' + text
        res_text = res_text.replace('<|im_end|>', '')
    else:
        res_text = think_text

    return res_text


def get_insert_verify_res_onlyend(sft_think_text, insert_verify, reward_method):
    if not sft_think_text:
        data = {"matchs": [], "sft_think_text_verify": sft_think_text, "end_index_lst": {}}
        return data
    if not insert_verify:
        data = {"matchs": [], "sft_think_text_verify": sft_think_text, "end_index_lst": {}}
        return data
    verify_location_res = insert_verify.split('</think>')[-1]
    if "InputText does not contain content providing the correct answer to the question" in verify_location_res or "输入文本不包含给出了问题最终答案的内容" in verify_location_res:
        data = {"matchs": [], "sft_think_text_verify": sft_think_text, "end_index_lst": {}}
        return data

    verify_split = re.split(r'(?<=\n)(?:[\d\.\s]*ANSWER location[\s\*\d]*[:\.：]|[\n\s]*\d+[:\.：])', '\n' + verify_location_res, flags=re.IGNORECASE | re.DOTALL)
    verify_split = [item.strip().split('<get_answer>')[0] for item in verify_split[1:] if item.strip()]
    flag_language = re.findall(r'[\u4e00-\u9fa5]', sft_think_text)
    Insert_verify_blocks = []
    for item in verify_split:
        if re.findall(r'[\u4e00-\u9fa5]', item) and not flag_language:
            continue
        end_text = re.findall(r'ANSWER location[\s\*\d]*:(.*?)$', item, flags=re.IGNORECASE | re.DOTALL)
        end_text = [sub_text for sub_text in end_text if sub_text.strip()]
        end_text = item if not end_text else end_text[0]
        if not end_text:
            continue
        end_text = end_text.strip().strip('[').strip(']').strip()
        Insert_verify_blocks.append({'end': end_text})

    end_index_lst = {}
    if 'mllm' in reward_method:
        sft_think_text_split = combine_small_lst_mllm(sft_think_text)
    else:
        sft_think_text_split = combine_small_lst(sft_think_text)

    for idx, block_dict in enumerate(Insert_verify_blocks):
        end_text = block_dict['end'].replace('<n>', '\n').strip().strip('"').strip('[').strip(']')
        end_text = end_text[-40:] if len(end_text) > 50 else end_text
        new_text_split = [item for i, item in enumerate(sft_think_text_split) if i not in end_index_lst.keys()]

        best_match_end = process.extractOne(end_text, new_text_split, scorer=fuzz.partial_ratio)
        if not best_match_end:
            continue
        find_index = 0
        end_index = None

        while find_index < len(sft_think_text_split):
            end_index = sft_think_text_split[find_index:].index(best_match_end[0]) + find_index
            if end_index not in end_index_lst.keys():
                break
            else:
                if end_index_lst.keys():
                    find_index = max(end_index_lst.keys()) + 1
                else:
                    find_index = end_index + 1
        if end_index == None:
            continue
        end_similarity = fuzz.partial_ratio(end_text, best_match_end[0].replace('</verify>',''))

        if end_index in end_index_lst:
            end_index_lst[end_index].append(best_match_end[0])
        else:
            end_index_lst[end_index] = [best_match_end[0]]

        Insert_verify_blocks[idx]["end_match_text"] = best_match_end[0]
        Insert_verify_blocks[idx]["end_match_index"] = end_index
        Insert_verify_blocks[idx]["end_similarity"] = end_similarity
        if end_similarity < 65:
            continue

        sft_think_text_split[end_index] += '</verify>'
    if 'mllm' in reward_method:
        sft_think_text_verify = ''.join(sft_think_text_split)
    else:
        sft_think_text_verify = '\n'.join(sft_think_text_split)

    data = {"matchs": Insert_verify_blocks, "sft_think_text_verify": sft_think_text_verify, "end_index_lst": end_index_lst}

    return data


def merge_closest_sum(input_text, target_length):
    split_lst = input_text.replace('<n>','\n').split('\n')

    if len(split_lst) < 2:
        return [input_text]
    if len(input_text) < target_length * 2:
        target_length = len(input_text) // 2 + 1
    result = []
    combine_text = split_lst[0]
    for index,item in enumerate(split_lst):
        if index == 0:
            continue
        combine_text = combine_text + '\n' + item
        length = len(combine_text)
        if re.findall(r'^[\-\*\+\s]*?(?:option|choice|)[\s\*]*[ABCDEF][\s\*]*[:\.、：]', item, flags=re.IGNORECASE) and len(item.strip()) < 100 and index != len(split_lst)-1:
            continue
        if combine_text.strip().endswith((',', ':', '+', '/', '[', '{', '(', ';', '，', '：', '【', '（')) and index != len(split_lst)-1 and length < target_length*1.2:
            continue
        if len(item.strip()) < 20 and not item.strip().endswith((',', ':', '+', '/', '[', '{', '(', ';', '，', '：', '【', '（')) and index != len(split_lst)-1 and length < target_length*1.2:
            continue
        if length > target_length or index == len(split_lst)-1:
            result.append(combine_text)
            combine_text = ''
    res_new = []
    if len(result[-1])<target_length/3:
        res_new.extend(result[:-1])
        res_new[-1] = res_new[-1] + '' + result[-1]
    else:
        res_new = result

    return res_new


def process_split_text(idx, split_text, question, ans_end, api, answer_location_prompt, reward_method):
    try:
        question_block = answer_location_prompt.format(
            Question_text=question,
            Solution_text=ans_end,
            InputText=split_text
        )

        generate_text_block = qwen_vllm_inference(question_block, api)
        generate_text_block_new = generate_text_block.split('</think>')[-1]
    except Exception as e:
        print(f"Error occurred during inference: {e}")
        generate_text_block = ''
        generate_text_block_new = ''
    try:
        if generate_text_block_new:
            res_insert = get_insert_verify_res_onlyend(split_text, generate_text_block_new, reward_method)
            res_insert["question_block"] = question_block
            res_insert["generate_block"] = generate_text_block
        else:
            res_insert = {"matchs": [], "sft_think_text_verify": split_text, "end_index_lst": {}, "question_block": question_block, "generate_block": generate_text_block}
    except Exception as e:
        res_insert = {"matchs": [], "sft_think_text_verify": split_text, "end_index_lst": {}, "question_block": question_block, "generate_block": generate_text_block}
        logging.error(f"An error occurred while processing the line: {e}")
        logging.error(f"question: {question}, ans_end: {ans_end}, api: {api}")
    return res_insert



def process_with_multithreading_map(split_think_text_merge, question, ans_end, api, answer_location_prompt, reward_method):
    sft_think_text_verify = ''
    res_insert_all = []

    # 准备参数列表
    params = [(idx, text, question, ans_end, api, answer_location_prompt, reward_method) for idx, text in enumerate(split_think_text_merge)]

    # 使用map处理
    global_executor = futures.ThreadPoolExecutor(max_workers=len(split_think_text_merge))
    try:
        results = global_executor.map(lambda p: process_split_text(*p), params)

        for result in results:
            sft_think_text_verify = sft_think_text_verify + '' + result["sft_think_text_verify"]
            res_insert_all.append(result)
    except Exception as e:
        logging.error(f"An error occurred while processing the line: {e}")
        logging.error(f"question: {question}, ans_end: {ans_end}, api: {api}")
    finally:
        global_executor.shutdown(wait=True)

    return sft_think_text_verify, res_insert_all


def process_line(question_ord, data):
    logging.debug(f"Processing ord:{question_ord}")
    try:
        question, solution, answer_true, api, result_score = data["question"], data["response"], data["answer"], data["reward_vllm_api"],data["acc_score"]
        question = question.replace('Choose the correct answer from the options after each question.\n', '')

        # enable_think_flag = True if question.strip().endswith('<think>') else False
        enable_think_flag = data["enable_thinking_flag"]
        question = question.replace('<｜User｜>', '<|User|>').replace('<｜Assistant｜>', '<|Assistant|>').replace('<think>', '').strip()
        question = question.split('<|User|>')[-1].split('<|Assistant|>')[0]
        solution = solution.replace('<｜User｜>', '').replace('<｜Assistant｜>', '').replace('<｜end▁of▁sentence｜>', '').replace('<|end_of_sentence|><eod>', '').strip()


        # ans_end = answer_true
        flag_insert = True
        if enable_think_flag:
            solution = '<think>'+solution.strip()
            if "</think>" in solution:
                think_text = solution.split("<think>")[1].split("</think>")[0].strip().replace('<n>', '\n').strip()
                answer_text = solution.split("</think>")[1].replace('<|end_of_sentence|>', '').replace('<n>', '\n').strip()
            else:
                think_text = solution.split("<think>")[1].strip().replace('<n>', '\n').strip()
                answer_text = ""
        else:
            think_text = solution.replace('<n>', '\n').strip()
            answer_text = ""

        ans_end = ''
        if result_score == 1 and answer_text:
            if "**Final Answer**" in answer_text:
                ans_end = answer_text.split("**Final Answer**")[-1].strip()

            ans_end = get_answer(answer_text) if not ans_end else get_answer(ans_end)

            if not ans_end:
                ans_end = get_boxed_answer_text(answer_text)
        
        answer_true_new = get_boxed_answer_text(answer_true)
        if data["reward_method"] in ["llm_choice","mllm_choice"]:
            from .insert_1st_answer_prompt.answer_location_prompt_Cn_1 import question_template as answer_location_prompt_true
            from .insert_1st_answer_prompt.answer_location_prompt_Cn_false import question_template as answer_location_prompt_false
            answer_location_prompt = answer_location_prompt_true if result_score == 1 else answer_location_prompt_false
            if re.findall(r'[A-Z]+', answer_true_new) and re.findall(r'[A-Z]+', answer_true_new)[0] == answer_true_new:
                answer_true = "The answer is \\boxed{" + answer_true_new + "}."
            flag_insert = flag_insert and not (result_score != 1 and len(think_text) > 25000)
        else:
            from .insert_1st_answer_prompt.answer_location_prompt_4 import question_template as answer_location_prompt
        if not ans_end:
            ans_end = answer_true
    except Exception as e:
        logging.error(f"Error parsing input text: {e}")
        data["sft_think_text_original"] = ""
        data["sft_think_text_final"] = ""
        data["matchs"] = []
        data["flag_insert"] = False
        return data

    if flag_insert:
        print(f"Start insert block")
        res_insert_all = []
        if len(think_text) > 1400:
            split_think_text_merge = merge_closest_sum(think_text, 1000)
        else:
            split_think_text_merge = [think_text]
        sft_think_text_verify, res_insert_all = process_with_multithreading_map(split_think_text_merge, question, ans_end, api, answer_location_prompt, data["reward_method"])

        sft_think_text_verify = sft_think_text_verify.replace('</verify>', '<1st_answer>', 1)


        if result_score == 1 and "<1st_answer>" in sft_think_text_verify and not sft_think_text_verify.strip().endswith(('</verify>', '<1st_answer>')):
            sft_think_text_verify = sft_think_text_verify + '</verify>'


        sft_think_text_final = sft_think_text_verify


        print(f"Success insert block.")

        data["sft_think_text_original"] = think_text
        data["sft_think_text_final"] = sft_think_text_final
        data["matchs"] = res_insert_all
        data["flag_insert"] = flag_insert
    else:
        logging.warning(f"Not insert block, start finalize")
        data["sft_think_text_original"] = think_text
        data["sft_think_text_final"] = think_text
        data["matchs"] = []
        data["flag_insert"] = flag_insert

    logging.info(f"..............Processed ord:{question_ord}")
    return data

def thread_func(questions):
    print(f"Hello World!!!!!")
    results = []
    with futures.ThreadPoolExecutor(max_workers=len(questions)) as executor:

        # 创建 Future 对象到问题索引的映射
        future_to_index = {}
        future_results = []

        for q_ord, q in enumerate(questions):
            future = executor.submit(process_line, q_ord, q)
            future_results.append(future)
            future_to_index[future] = (q_ord, q)  # 存储索引和原始问题

        for future in futures.as_completed(future_results):
            # 通过 future 对象获取对应的索引和问题
            q_ord, original_question = future_to_index[future]

            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"问题索引 {q_ord} 生成异常: {exc}")
                print(f"对应的问题是: {original_question}")
                results.append(original_question)
    return results


def process_input_lst(questions):
    content = {}
    all_results = []
    reward_api_lst = [data["reward_vllm_api"] for data in questions if data["reward_method"] in LLM_PROCESS_METHOD_FLAG]
    base_n = len(reward_api_lst)
    questions = list(map(lambda x: {**x[1], 'index': x[0]}, enumerate(questions)))
    questions = sorted(
        questions,
        key=lambda d: len(d.get("response", "")),
        reverse=True
    )
    def zigzag_map(n, total):
        block = n // total
        offset = n % total
        return offset if block % 2 == 0 else total -1 - offset
    for idx, data in enumerate(questions):
        
        # TODO: add LLM_PROCESS_METHOD_FLAG
        if data["reward_method"] not in LLM_PROCESS_METHOD_FLAG:
            all_results.append(data)
            continue
        
        data["reward_vllm_api"] = reward_api_lst[zigzag_map(idx, base_n)]
        api = data["reward_vllm_api"]
        if api not in content:
            content[api] = [data]
        else:
            content[api].append(data)


    def split_list_2(lst, n=4):
        """使用numpy将列表平均分成n份"""
        return np.array_split(lst, n)
    num_processes = 128 # num_process 需整除节点数*4
    split_num = num_processes // len(content.keys())
    print(f"num_processes: {num_processes}; split_num:{split_num}")
    input_lst = []
    for key, value in content.items():
        if split_num > 0:
            value_split = split_list_2(value, split_num)
            for item in value_split:
                input_lst.append(list(item))
        else:
            input_lst.append(value)

    with multiprocessing.Pool(processes=num_processes) as pool:
        # 使用 imap 流式获取结果，节省内存
        for result in pool.imap(thread_func, input_lst):
            all_results.extend(result)

    results_sorted = sorted(all_results, key=lambda x: x["index"])

    logging.info("All question completed.........")

    return results_sorted

def verify_score(marked_txt, acc_score, v_exp, v_max):
    """acc_score can be -1, 1 """
    v = marked_txt.count('</verify>')
    if v == 0.0: # acc_score=-1.0表明反思错误或失败，直接返回-1。
        R_ver = acc_score
    elif v_max == v_exp:
        R_ver = 1.0
    elif v_max <= 2:
        R_ver = 1.0
    else:
        if 0 < v <= v_exp:
            R_ver = 1.0
        elif v_exp < v <= v_max:
            R_ver = 1 - ((v-v_exp) / (v_max - v_exp)) # 根据反思数量进行惩罚
        else:
            R_ver = 0.0 # 存在</verify>时，反思正确，按照反思数量进行惩罚，分值范围（0,1）
    return R_ver


def process_scoring(reward_output, v_exp=2, v_max=10):
    acc_score = reward_output['acc_score']
    R_side = -1.0 if any([reward_output['format'] < 0.0, reward_output['lang_consistency_ratio'] < 0.0, reward_output['repetition_penalty'] < 0.0]) else 0.0
    response = reward_output['response']
    marked_txt = reward_output['sft_think_text_final']
    if '<1st_answer>' not in marked_txt: 
        R_tri = [-1.0, -1.0, -1.0]
    elif marked_txt.strip().endswith('<1st_answer>'):
        if '</think>' in response: # 只有长思维做过程监督
            # think部分输出正常但无verify内容，verify部分直接使用acc分数
            R_tri = [1.0, acc_score, acc_score]
        else:
            R_tri = [1.0, -1.0, -1.0]
    else:
        # verify数量惩罚的下限使用2个，且acc分数为-1时，verify部分直接返回-1分
        R_tri = [1.0, verify_score(marked_txt, acc_score, v_exp, v_max), acc_score]
    R = sum(R_tri) + R_side
    return R, R_tri
