import json
import os
import random
import uuid
import time
import subprocess
import ast
import requests
import re
from math import tanh
from math_verify import LatexExtractionConfig, parse, verify,StringExtractionConfig,ExprExtractionConfig
from latex2sympy2_extended import NormalizationConfig
from verl.utils.reward_score.eval_llm.score_apps import format_trans,compute_score
from openai import OpenAI
'''
数学--score
'''

PROMPT_TEMPLATE='以下两个字段的内容分别是一个问题和某位学生的解题过程，请从解题过程提取答案，并以boxed{{}}的格式输出答案。若无法提取答案，则输出 \\boxed{{None}}。当存在多个答案时，将所有的答案都放在一个boxed{{}}内进行输出。要求仅提取选项，不提取选项内容。\n\n问题：{question}\n学生答案：{predict}'

def math_verify_extract(content):
    ind = content.rfind('\\boxed')
    if ind == -1 and 'boxed' in content:
        ind=content.rfind('boxed')
        print(ind)
        print('********')
        content=f'$\\{content[ind:]}$'.replace('<n>','\n').replace('\,',',').replace('\\%','').replace('%','')
    elif 'boxed' not in content:
        content=f'${content}$'.replace('<n>','\n').replace('\,',',').replace('\\%','').replace('%','')
    else:
        content=f'${content[ind:]}$'.replace('<n>','\n').replace('\,',',').replace('\\%','').replace('%','')

    answer_parsed = parse(
        content,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    equations=False, # disable warning
                    boxed="all",
                    units=True,
                ),
                # Ensures that boxed is tried first
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            ),
            ExprExtractionConfig()
        ],
        extraction_mode="first_match",
    )
    return answer_parsed

def math_verify_reward(content,sol):
    gold_parsed=math_verify_extract(sol)
    answer_parsed=math_verify_extract(content)

    if verify(answer_parsed, gold_parsed,float_rounding=5,numeric_precision=15):
            return True

    return False

def llm_score_math_verify(predict,answer):
    answer = answer.replace("<n>","\n")
    predict = predict.replace("<n>","\n")

    if '\\boxed' not in predict:
        return -1.0
    if answer == predict:
        return 1.0
    else:
        return 1.0 if math_verify_reward(predict,answer) else -1.0

# llm_choice

# llm_choice math verify
def math_verify_extract2(content):
    ind = content.rfind('\\boxed')
    if ind == -1 and 'boxed' in content:
        ind=content.rfind('boxed')
        content=f'$\\{content[ind:]}$'.replace('<n>','\n').replace('\,',',').replace('\\%','').replace('%','')
    elif 'boxed' not in content:
        content=f'${content}$'.replace('<n>','\n').replace('\,',',').replace('\\%','').replace('%','')
    else:
        content=f'${content[ind:]}$'.replace('<n>','\n').replace('\,',',').replace('\\%','').replace('%','')

    answer_parsed = parse(
        content,
        extraction_config=[
            StringExtractionConfig()
        ],
        extraction_mode="first_match",
    )
    return answer_parsed

def math_verify_reward2(content,sol):
    gold_parsed=math_verify_extract2(sol)
    answer_parsed=math_verify_extract2(content)

    if verify(answer_parsed, gold_parsed,float_rounding=5,numeric_precision=15):
            return True

    return False

def llm_score_choice_verify(predict,answer):
    answer = answer.replace("<n>","\n")
    if 'boxed' not in answer:
        answer1 = '\\boxed{'+answer+'}'
    else:
        answer1 = answer
    answer2 = 'The final answer is: '+answer
    predict = predict.replace("<n>","\n")

    if '<|end_of_sentence|>' not in predict:
        return -1.0
    if answer == predict:
        return 1.0
    else:
        return 1.0 if (math_verify_reward(predict,answer1) or math_verify_reward2(predict,answer2)) else -1.0


# llm_choice model#
def parse_output(output):
    pattern = r"\\?boxed{(.*?)}"
    answer_matches = re.findall(pattern, output)
    if not answer_matches:
        return None
    else:
        return answer_matches[-1].replace(', ','').replace(',','')


def generate(prompt,port,timeout,max_tokens):
    openai_api_key = "EMPTY"
    openai_api_base = f"http://{port}/v1"
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base, timeout=timeout)
    models = client.models.list()
    judge_model = models.data[0].id
    prompt = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        model = judge_model,
        messages=prompt,
        max_tokens=max_tokens,
        temperature=0.6,
        top_p=0.95,
        extra_body={
            "include_stop_str_in_output": True,
            "skip_special_tokens": False
        },
    )
    response = completion.choices[0].message.content

    return response


def llm_choice_qwen_vllm(reward_input, api, timeout, max_tokens):

    try:
        question = reward_input['question'].replace('<n>','\n').replace("<｜User｜>","").replace("<｜Assistant｜>","").replace("<think>","")
        response = reward_input['response'].replace('<n>','\n').replace("<｜end▁of▁sentence｜>","")
        answer = reward_input['answer'].replace('<n>','\n')

        index1 = response.find('**Answer:**')
        index2 = response.find('**Final Answer**')
        index3 = response.find('</think>')

        indexes = [index1, index2, index3]
        valid_indexes = [i for i in indexes if i != -1]
        if valid_indexes:
            index = min(valid_indexes)
        else:
            index = -1

        predict=response[:index]
        prompt=PROMPT_TEMPLATE.format(question=question,predict=predict)

        outputs=generate(prompt, api, timeout, max_tokens)

        extracted_ans=parse_output(outputs)
        if not extracted_ans:
            print("No box")
            return -1.0

        flag= (sorted(extracted_ans) == sorted(answer))
        if flag:
            return 1.0
        else:
            return -1.0

    except Exception as e:
        print(e,"error1>>>>>>>>")
        return -1.0


# 代码score

CODE_SAVE_PATH = './code_tmp/middle_py/'
if not os.path.exists(CODE_SAVE_PATH):
    os.makedirs(CODE_SAVE_PATH)

def check_force_code(ans_text, unittest_lst, check_num = 5):
    check_num = len(unittest_lst)
    if check_num ==0 :
        return float(0)
    ans_text = ans_text.replace("<eop>", "")
    if not ans_text:
        return float(0)
    ans_text = ans_text.replace('\n', '<n>')
    code_ans = re.findall(r'(?<=[`\']{3}python)[^`]*?(?=[`\']{3})', ans_text)
    if not code_ans:
        return float(0)
    ans_code = re.sub(r'\s*<n>\s*$','',code_ans[-1])
    ans_code = re.sub(r'^\s*<n>\s*','',ans_code)
    ans_code = re.sub(r'\\\s*<n>','\\<n>',ans_code)

    matches_input = re.findall(r'input\(\)', ans_code)
    # ans_code是最基本的代码片段
    if len(matches_input) == 1:
        total_unittest = len(unittest_lst)
        if check_num == 0 or check_num > total_unittest:
            check_num = total_unittest
        if total_unittest > check_num:
            check_unittest = random.sample(unittest_lst, check_num)
        else:
            check_unittest = unittest_lst[:]
        pass_unittest = 0
        for test_case in check_unittest:
            test_case_input = test_case['input']
            test_case_output = test_case['output']

            if type(test_case_input) is not str:
                ans_code_unit = ans_code.replace('input()',f"{test_case_input}")
            else:
                ans_code_unit = ans_code.replace('input()',f"'{test_case_input}'")
            ans_code_unit = ans_code_unit.replace("<n>", "\n")
            ans_solve = "\n# python code\n" + ans_code_unit + "\n"
            ans_solve = ans_solve.replace('，',',')
            ans_solve = f'''{ans_solve}'''
            try:
                unique_id = str(uuid.uuid1())
                with open(f'./code_tmp/middle_py/{unique_id}.py', "w",encoding='utf-8') as file:
                    file.write(ans_solve)
                du_cmd = ['timeout','6s', 'python3', f'./code_tmp/middle_py/{unique_id}.py']
                process = subprocess.Popen(du_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True) # 将错误信息丢弃
                ans_value, stderr = process.communicate()
            except Exception as e:
                pass_unittest = pass_unittest
            else:
                if ans_value.strip() == test_case_output.strip():
                    pass_unittest = pass_unittest + 1
                try:
                    os.remove(f'./code_tmp/middle_py/{unique_id}.py')
                except Exception:
                    pass_unittest = pass_unittest
    else:
        total_unittest = len(unittest_lst)
        if check_num == 0 or check_num > total_unittest:
            check_num = total_unittest
        if total_unittest > check_num:
            check_unittest = random.sample(unittest_lst, check_num)
        else:
            check_unittest = unittest_lst[:]
        pass_unittest = 0
        for test_case in check_unittest:
            test_case_input = test_case['input']
            test_case_output = test_case['output']

            ans_code_unit = ans_code

            ans_code_unit = re.sub(r'[\\]*<n>', '\n', ans_code_unit)
            ans_code_unit = ans_code_unit.replace("<n>", "\n")
            ans_solve = "\n# python code\n" + ans_code_unit + "\n"
            ans_solve = ans_solve.replace('，',',')
            ans_solve = f'''{ans_solve}'''
            try:
                unique_id = str(uuid.uuid4())
                with open(f'./code_tmp/middle_py/{unique_id}.py', "w",encoding='utf-8') as file:
                    file.write(ans_solve)
                du_cmd = ['timeout','6s', 'python3', f'./code_tmp/middle_py/{unique_id}.py']
                process = subprocess.Popen(du_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True) # 将错误信息丢弃
                ans_value, stderr = process.communicate(input=str(test_case_input))
            except Exception as e:
                pass_unittest = pass_unittest
            else:
                if ans_value.strip() == test_case_output.strip():
                    pass_unittest = pass_unittest + 1
                try:
                    os.remove(f'./code_tmp/middle_py/{unique_id}.py')
                except Exception:
                    pass_unittest = pass_unittest

    return float(pass_unittest)/float(check_num)

def check_correctness(code_text, unittest_lst, check_num = 5):
    check_num = len(unittest_lst)
    if not code_text:
        return float(0)
    code_text = code_text.replace('<n>', '\n').strip().replace('\n', '<n>')
    code_text = re.sub(r'\\\s*<n>', '\\<n>', code_text)

    total_unittest = len(unittest_lst)
    check_unittest = unittest_lst[:]

    pass_unittest = 0
    for test_case in check_unittest:
        ans_code_unit = code_text + '\n' + test_case
        ans_code_unit = ans_code_unit.replace("<n>", "\n")
        ans_solve = "\n# python code\n" + ans_code_unit + "\n"
        ans_solve = ans_solve.replace('，',',')
        ans_solve = f'''{ans_solve}'''
        script_path = f'{CODE_SAVE_PATH}/{str(uuid.uuid1())}.py'

        try:
            with open(script_path, "w",encoding='utf-8') as file:
                file.write(ans_solve)
            time.sleep(0.1)
            result = subprocess.run(['python', script_path], capture_output=True, text=True, timeout=6)
            if result.returncode == 0:
                pass_unittest = pass_unittest + 1
        except subprocess.TimeoutExpired:
            pass_unittest = pass_unittest
        except Exception as e:
            pass_unittest = pass_unittest
        finally:
            try:
                time.sleep(0.1)
                os.remove(script_path)
            except Exception:
                pass_unittest = pass_unittest

    return float(pass_unittest)/float(check_num)


def exec_unittest(middle_text, only_unittest):
    middle_text = middle_text.replace('<eop>', '')
    middle_text = middle_text.replace('<n>', '\n')
    if "```" not in middle_text:
        return float(0)
    only_unittest = only_unittest.replace('<n>', '\n').strip().replace('\n', '<n>')
    pattern = r'```python.*?\n(.*?)```'
    code_match = re.search(pattern, middle_text.replace('<n>', '\n'), re.DOTALL)

    if not code_match:
        split_lst = middle_text.split('```python')
        if len(split_lst) == 1:
            return float(0)
        split_lst = [item for item in split_lst if item.strip()]
        only_code = split_lst[-1].strip('\n')
    else:
        only_code = code_match.group(1)
    if not only_code:
        return float(0)
    unittest_lst = only_unittest.replace('\n', '<n>').split('<n>')
    check_result = check_correctness(only_code, unittest_lst)
    return check_result

def convert_string_to_dicts_with_eval(input_string):
    """
    使用 eval() 将包含字典格式字符串的列表转换为字典列表。
    如果列表中的元素已经是字典，则直接返回。

    参数:
        input_string (str): 包含字典格式字符串的列表，例如：
                            '["{\'input\': \'Sunny\', \'output\': \'Cloudy\'}", ...]'

    返回:
        List[dict]: 转换后的字典列表。
    """
    # 将输入字符串解析为 Python 列表
    json_list = ast.literal_eval(input_string)

    # 检查列表中的每个元素是否是字符串
    if all(isinstance(item, str) for item in json_list):
        # 如果所有元素都是字符串，则使用 eval() 将每个字符串转换为字典
        dict_list = [ast.literal_eval(item) for item in json_list]
    else:
        # 如果列表中已经包含字典，则直接返回
        dict_list = json_list
        # 遍历列表中的每个字典
        for item in dict_list:
            input_value = item.get('input', None)
            if isinstance(input_value, str) and input_value.startswith('[') and input_value.endswith(']'):
                # 如果 'input' 的值是字符串，并且以 [] 开头和结尾，则尝试将其转换为列表
                try:
                    item['input'] = ast.literal_eval(input_value)
                except Exception as e:
                    pass  # 如果解析失败，保留原始字符串
    return dict_list



def check_list_elements(input_list):
    """
    检查列表中的每个元素是否是字符串且包含 'assert'
    :param input_list: 输入的列表
    :return: 返回一个布尔值列表，每个位置对应输入列表中元素的检查结果
    """
    result = []
    for item in input_list:
        # 检查是否是字符串
        if isinstance(item, str):
            # 检查是否包含 'assert'
            if 'assert' in item:
                result.append(True)
            else:
                result.append(False)
        else:
            result.append(False)
    return result

def llm_score_code(ans_text, unittest_text):
    if "assert" in unittest_text: # humaneval/mbpp/leetcode
        ans_text = ans_text.replace('<eop>', '')
        pass_rate = exec_unittest(ans_text, unittest_text)
    elif all(check_list_elements(unittest_text)): # assert(str) list
        ans_text = ans_text.replace('<eop>', '')
        unittest_text = "<n>".join(map(str, unittest_text))
        pass_rate = exec_unittest(ans_text, unittest_text)
    else: # atcoder
        if type(unittest_text) is str:
            # 判断是否是字符串类型，则需要解析
            if unittest_text.startswith('<n>'):
                every_unittest = unittest_text.split('<n>')[1:] # list字符串  [{"input": "5 1 1", "output": "0"},{"input": "3 100 1000", "output": "900"},{"input": "3 100 100", "output": "900"},{"input": "1 1 1", "output": "999999999"}]
                every_unittest = [ast.literal_eval(item) for item in every_unittest] # list 每个元素是{'input': '2 5 7', 'output': '5'}
            else:
                # every_unittest=eval(unittest_text)
                every_unittest=convert_string_to_dicts_with_eval(unittest_text)
            pass_rate = check_force_code(ans_text, every_unittest)
        elif type(unittest_text) is list:
            pass_rate = check_force_code(ans_text, unittest_text)
        else:
            return float(0)
            # assert unittest_text, "Unit test type error"
    return pass_rate


def apps_score_code(ans_text, unittest_text):
    solution_str,ground_truth = format_trans(ans_text, unittest_text)
    res = compute_score(solution_str, ground_truth, continuous=True, timeout=6, num_check=-1, debug=False)
    print(float(res[0]))
    return float(res[0])


def extract_solution_nosearch(solution_str):
    if "<|begin_of_sentence|>" in solution_str:
        solution_str = solution_str.replace("<|begin_of_sentence|>","")
    if "<|User|>" in solution_str:
        solution_str = solution_str.replace("<|User|>","")
    if "<｜User｜>" in solution_str:
        solution_str = solution_str.replace("<｜User｜>","")
    if "<think>" in solution_str:
        solution_str = solution_str.replace("<think>","")
    if "<｜Assistant｜>" in solution_str:
        solution_str = solution_str.replace("<｜Assistant｜>","")

    if "<|Assistant|>" in solution_str:
        solution_str = solution_str.replace("<|Assistant|>","")
    if solution_str.endswith("<|endoftext|>"):
        solution_str = solution_str[:-len("<|endoftext|>")]
    elif solution_str.endswith("<|im_end|>"):
        solution_str = solution_str[:-len("<|im_end|>")]
    else:
        solution_str = solution_str
    if "<|end_of_sentence|>" in solution_str:
        solution_str = solution_str.replace("<|end_of_sentence|>", "")
    if "<eod>" in solution_str:
        solution_str = solution_str.replace("<eod>", "")
    if "</think>" in solution_str:
        solution_str=solution_str.split("</think>")[-1]
    if "<｜end▁of▁sentence｜>" in solution_str:
        solution_str=solution_str.replace("<｜end▁of▁sentence｜>","")
    return solution_str.strip()

# llm_general
DEFAULT__PROMPT_TEMPLATE="""
You are a skilled little expert at scoring responses. You should evaluate given responses based on the given judging criteria.\n Given the context of the conversation (the last round is the User's query) and multiple responses from the Assistant, you need to refer to the [General Evaluation Criteria] to score the responses. Based on the general evaluation criteria, state potential other specific criteria to the query, the weights of different criteria, and then provide an overall comprehensive score upon them.\n Each score is an integer between 1 and 10, with a higher score indicating that the response meets the relevant criteria more closely. For example, a score of 1 means the response does not meet the criteria at all, a score of 6 means the response meets only some parts, and a score of 10 means the response perfectly meets the evaluation criteria.\n Before scoring, please analyze step by step. Your scoring needs to be as strict as possible.

#### Evaluation Criteria ####
1. Instruction Adherence:\n - Fully Adhered (9-10 points): The response fully complies with all instructions and requirements of the question.\n - Partially Adhered (6-8 points): The response meets most of the instructions but has some omissions or misunderstandings.\n - Basically Adhered (3-5 points): The response meets some instructions, but the main requirements are not fulfilled.\n - Not Adhered (1-2 points): The response does not meet any instructions.\n Example: If the question requires three examples and the response provides only one, it falls under "Partially Adhered."
2. Usefulness:\n - Highly Useful (9-10 points): The response provides comprehensive and accurate information, fully addressing the issue.\n - Useful but Incomplete (6-8 points): The response provides some useful information, but lacks details or accuracy.\n - Limited Usefulness (3-5 points): The response offers little useful information, with most content being irrelevant or incorrect.\n - Useless or Incorrect (1-2 points): The response is completely irrelevant or incorrect.\n Example: If there are factual errors in the response but the overall direction is correct, it falls under "Useful but Incomplete."
3. Level of Detail:\n - Very Detailed (9-10 points): The response includes ample details covering all aspects of the issue.\n - Detailed but Slightly Lacking (6-8 points): The response is fairly detailed but misses some important details.\n - Basically Detailed (3-5 points): The response provides some details but is not thorough enough overall.\n - Not Detailed (1-2 points): The response is very brief and lacks necessary details.\n Example: If the response provides only a simple conclusion without an explanation, it falls under "Not Detailed."
4. Relevance:\n - Highly Relevant (9-10 points): The response is highly relevant to the question, with information closely aligned with the topic.\n - Generally Relevant (6-8 points): The response is generally relevant but includes some unnecessary information.\n - Partially Relevant (3-5 points): The response has a lot of content that deviates from the topic.\n - Not Relevant (1-2 points): The response is completely irrelevant.\n Example: If the response strays from the topic but still provides some relevant information, it falls under "Partially Relevant."

#### Conversation Context ####\n{instruction}\n
#### Responses to be Scored ####
{responses}
#### Output Format Requirements ####

Output with three lines
Specific Criteria: <Other potential criteria specific to the query and the context, and the weights of each criteria>.
Analysis: <Compare different responses based on given Criteria>.
Scores: <the overall comprehensive score of all responses in order, separate by comma in the boxed, e.g., \\boxed{{x, x}} if there exists 2 responeses>.
"""

PROMPT_TEMPLATE_CN="""
你是给回答评分的专家。你应该根据给定的评价标准来评价给出的回答。\n给定对话的上下文（最后一轮是用户的查询）和来自助手的多个回答，您需要参考[一般评价标准]来对回答进行评分。基于一般评价标准，向查询声明其他潜在的特定标准，以及不同标准的权重，然后在它们>之上提供一个整体的综合得分。\n每个分数为1~10之间的整数，分数越高表示回答越符合相关标准。例如，1分表示回答完全不符合标准，6分表示回答只符合部分标准，10分表示回答完全符合评价标准。\n评分前，请一步一步分析。你的评分要尽可能严格。

####评价标准####
1. 指令遵循：\n -完全遵循（9-10分）：回答完全符合问题的所有指示和要求。\n -部分遵循（6-8分）：回答符合大部分指示，但有一些遗漏或误解。\n -基本遵循（3-5分）：回答部分符合说明，但主要要求未满足。\n -不遵循（1-2分）：回答不符合任何指示。例子：如果问题>需要三个例子，而回答只提供了一个，它就属于“部分遵循”。
2. 有用性：\n -非常有用（9-10分）：回复提供了全面准确的信息，充分解决了问题。\n -有用但不完整（6-8分）：回答提供了一些有用的信息，但缺乏细节或准确性。有用性有限（3-5分）：回答提供的有用信息很少，大部分内容不相关或不正确。\n—无用或不正确（1-2分）：回
答完全无关或不正确。例子：如果回答中有事实错误，但总体方向是正确的，则属于“有用但不完整”。
3. 详细程度：\n -非常详细（9-10分）：回答包括了足够的细节，涵盖了问题的各个方面。\n -详细但略显不足（6-8分）：回答相当详细，但遗漏了一些重要细节。\n -基本详细（3-5分）：回答提供了一些细节，但总体上不够彻底。不详细（1-2分）：回答非常简短，缺乏必要的>细节。示例：如果回答只提供了一个简单的结论而没有解释，则属于“不详细”。
4. 相关性：\n -高度相关（9-10分）：回答与问题高度相关，信息与主题紧密相关。\n—一般相关（6-8分）：回答一般相关，但包括一些不必要的信息。\n—部分相关（3-5分）：回答有很多偏离主题的内容。\n—不相关（1-2分）：回答完全不相关。示例：如果回复偏离了主题，但仍
然提供了一些相关信息，则属于“部分相关”。

####对话上下文####
{instruction}\n
####待评分的回答####
{responses}
####输出格式要求####
三行输出
特定评价标准：<特定于查询和上下文的其他潜在标准，以及每个标准的权重>。
分析：<根据给定的标准比较不同的回答>。
得分：<所有回答的综合得分，以逗号分隔，如有2个回答>，则为\\boxed{{x, x}}。
"""

def format_responses(response_list):
    formatted_string = ""
    for i, response in enumerate(response_list, start=1):
        response=extract_solution_nosearch(response)
        formatted_string += f"[The Begin of Response {i}]\n{response}\n[The End of Response {i}]\n"
    return formatted_string


def format_responses_cn(response_list):
    formatted_string = ""
    for i, response in enumerate(response_list, start=1):
        response=extract_solution_nosearch(response)
        formatted_string += f"[回答{i}开始]\n{response}\n[回答{i}结束]\n"
    return formatted_string

def check_language(s):
    # 去除字符串中的标点符号和空格，只保留字母和汉字
    filtered = [c for c in s if c.isalpha() or '\u4e00' <= c <= '\u9fff']

    if not filtered:  # 处理空字符串或全是标点符号的情况
        return "中文"  # 默认返回中文

    total = len(filtered)
    english_count = sum(1 for c in filtered if c.isalpha() and c.isascii())
    english_ratio = english_count / total

    return "英文" if english_ratio > 0.9 else "中文"


def extract_composite_scores(composite_score_str):

    # 使用正则表达式查找所有匹配的boxed内容
    matches = re.findall(r"\\boxed\{([^\}]+)\}", composite_score_str)
    if not matches:
        print("No boxed scores found")
        print(composite_score_str)
        return None

    # 选择最后一个匹配的内容
    scores_str = matches[-1]
    scores = scores_str.split(',')

    # Check if each score is a valid float
    try:
        scores = [float(score.strip()) for score in scores]
    except Exception as e:
        print("Scores within boxed are not valid float")
        return None

    return scores


#######
def llm_general_reward(reward_input, api, timeout, max_tokens):

    try:
        data=reward_input
        response_list=reward_input["response_list"]
        instruction=extract_solution_nosearch(reward_input['question'].replace('<n>','\n'))

        if not data["language"] :
            data["language"]=""

        if 'zh' in data['language'] or 'cn' in data['language'] or check_language(instruction) == "中文":
            responses = format_responses_cn(response_list)
            input_text = PROMPT_TEMPLATE_CN.format(instruction=instruction,responses=responses).strip()
        else:
            responses = format_responses(response_list)
            input_text = DEFAULT__PROMPT_TEMPLATE.format(instruction=instruction,responses=responses).strip()
        

        openai_api_key = "EMPTY"
        openai_api_base = f"http://{api}/v1"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            timeout=timeout,
        )
        models=client.models.list()
        judge_model = models.data[0].id


        messages=[{"role": "user", "content": input_text}]
        response = client.chat.completions.create(
            model=judge_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=1e-6,
            top_p=1.0,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                },
        )
        text = response.choices[0].message.content
        
        scores = extract_composite_scores(text)
        if scores is None or len(scores) != len(response_list):
            return [-1.0] * len(response_list), None
        scores = [(2/9)*x - (11/9) for x in scores]
        return scores, text

    except Exception as e:
        print(e,"error1>>>>>>>>")
        return [-1.0] * len(response_list), None

# len-reward

def len_reward(acc_reward_outputs, threshold, max_gen_len):
    """Compute length-based rewards to discourage overthinking and promote token efficiency."""

    contents = [output["response"] for output in acc_reward_outputs]

    # First check correctness of answers
    correctness = [True if float(output["reward_score"]) >= threshold else False for output in acc_reward_outputs]
    pass_rate = sum(correctness)/len(correctness)

    # Calculate lengths
    lengths=[len(output["response_id"]) for output in acc_reward_outputs]
    correct_lengths = [lengths[i] for i in range(len(lengths)) if correctness[i]]
    if correct_lengths:
        ave_length = sum(correct_lengths) / len(correct_lengths)
    else:
        ave_length = 0.0

    L_budget = pass_rate * ave_length + (1-pass_rate) * max_gen_len

    # Calculate rewards
    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = (length - L_budget) / L_budget

        if is_correct:
            reward = max(tanh(-0.5*lambda_val+0.5),0.1)
        else:
            reward = min(tanh(0.9*lambda_val-0.1),-0.1)

        rewards.append(float(reward))

    for i, output in enumerate(acc_reward_outputs):
        output['len_reward'] = rewards[i]

    return acc_reward_outputs


# repetition-penalty

def get_repetition_penalty_reward(reward_output, ngram_size=20, max_penalty=-1, lowest_score=-1.0):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(completion, ngram_size):
        return zip(*[completion[i:] for i in range(ngram_size)])
    
    completion = reward_output["response_id"]
    if completion == "":
        reward = 0.0
        return reward
    if len(completion) < ngram_size:
        reward = 0.0
        return reward

    ngrams = set()
    total = 0
    for ng in zipngram(completion, ngram_size):
        ngrams.add(ng)
        total += 1

    scaling = 1 - len(ngrams) / total
    reward = scaling * max_penalty

    # if not only_apply_to_wrong and is_correct:
    if reward > -0.2:
        reward = 0.0
    else:
        reward = lowest_score

    return reward


# language_consistency

import jieba

def get_default_allowed_english_words():
    """获取默认的允许英文词列表"""
    allowed_english_words = set()
    # 添加单个英文字母（大小写）
    allowed_english_words.update([chr(i) for i in range(ord('a'), ord('z')+1)])
    allowed_english_words.update([chr(i) for i in range(ord('A'), ord('Z')+1)])
    # 添加常见数学函数和常量
    allowed_english_words.update(['sin', 'cos', 'tan', 'cot', 'sec', 'csc',
                                 'log', 'ln', 'exp', 'sqrt', 'abs', 'max', 'min',
                                 'pi', 'e', 'sum', 'prod', 'int', 'lim', 'inf',
                                 'alpha', 'beta', 'gamma', 'delta', 'theta',
                                 'sigma', 'omega', 'lambda', 'mu', 'nu', 'boxed', 'times', 'frac',
                                 ])
    # # 数学运算符和关系符号
    allowed_english_words.update(["ei", "fh","di", "fg", "dh", "eg",])
    # 数学或物理单位
    allowed_english_words.update(["MHz","Var","dx","lim"])
    # # 代码类关键词
    allowed_english_words.update(["if","else","not","return","continue","false","true","for","find","in","get","count","while","pass",
                                  "python","var","mod","list","join","valueError","elif",
                                  "int","str",
                                  "isinstance","raise","except","try","startwith","and",
                                  "has","isdigit","isupper","islower","re","any","current",
                                  "ones","zeros","left","right","defaultdict","length","break","range",
                                  "queue","append","len","set","math","import","float","num","lower","upper",
                                  "round","ceil","floor","split","replace",
                                  "with","finally","yield","global","is","none","nonlocal","or",
                                  "def","print",
                                  ])
    return allowed_english_words

def language_consistency_penalty(input_text, output_text, allowed_english_words=None):
    """
    使用 langdetect 和 jieba 计算语言一致性惩罚值

    Args:
        input_text: 输入文本
        output_text: 输出文本
        allowed_english_words: 允许的英文单词列表（数学符号、函数等）

    Returns:
        penalty_ratio: 语言不一致惩罚比例 (0-1)
    """
    # 使用默认允许词列表
    if allowed_english_words is None:
        allowed_english_words = get_default_allowed_english_words()

    # 过滤去除latex符号
    # 步骤1: 定义LaTeX命令的正则表达式模式
    # 此模式匹配以反斜杠开头的LaTeX命令，包括命令名和可选参数。
    # 命令名由字母组成，参数可以是方括号[]或花括号{}内的内容。
    # 例如：\alpha, \beta, \frac{1}{2}, \sqrt{x} 等。
    # 参考The Comprehensive LaTeX Symbol List中符号的命令结构[citation:6][citation:8]。
    latex_pattern = re.compile(
        r'\\[a-zA-Z]+\s*(?:\[[^\]]*\])?\s*(?:\{[^}]*\})*'
    )
    # 存储匹配到的LaTeX命令，用于后续恢复
    latex_commands = []

    def replace_with_placeholder(match):
        """将匹配到的LaTeX命令替换为唯一占位符。"""
        # placeholder = f"__LATEX_{len(latex_commands)}__"
        placeholder = ""
        latex_commands.append(match.group(0))  # 保存原始命令
        return placeholder

    output_text = latex_pattern.sub(replace_with_placeholder, output_text)

    # 使用 jieba 对输出文本进行分词
    seg_list = jieba.cut(output_text, cut_all=False)
    words = list(seg_list)

    # 过滤规则：
    # 1. 去除数字、数学运算符号、标点符号、空白字符
    # 2. 去除单个英文字母
    # 3. 只保留有意义的中文词汇和长度≥2的英文单词
    filtered_words = []
    # 匹配需要移除的字符：数字、数学运算符号、标点、空白字符
    remove_pattern = r'[0-9\s\.,!?;:"\'()\[\]{}<>+\-*/=~^&@#$%¥€·…—、，。！？；：""''（）【】《》￥…—\s]+'

    for word in words:
        # 第一步：移除数字、数学运算符号、标点符号、空白字符
        cleaned_word = re.sub(remove_pattern, '', word).strip()

        # 第二步：过滤空字符串和单个英文字母
        if cleaned_word:
            # 判断是否为单个英文字母（a-z, A-Z）
            if not (len(cleaned_word) == 1 and re.match(r'^[a-zA-Z]$', cleaned_word)):
                filtered_words.append(cleaned_word)

    # 计算仅中文词汇和长度≥2的英文单词
    total_words = len(filtered_words)
    if total_words == 0:
        return 0.0

    # 去除英文单词白名单
    inconsistent_count = 0
    # inconsistent_words = []
    for word in filtered_words:
        # 判断单词是否为英文（只包含英文字母）
        if re.fullmatch(r'[a-zA-Z]+', word):
            # 如果英文单词不在允许列表和输入问题中，则视为不一致
            if word.lower() not in allowed_english_words and word not in input_text:
                inconsistent_count += 1

    # 计算不一致比例
    penalty_ratio = inconsistent_count / total_words
    return penalty_ratio

def language_consistency_reward(reward_output, threshold=0.2, lowest_score=-1.0, only_apply_to_en=False):
    """
    语言一致性惩罚函数，检测到语言不一致，则返回分值范围内的最低分。

    """

    response = reward_output['response'].replace('</think>','').split('<|end_of_sentence|>')[0].strip()

    assert 'language' in reward_output
    target_lang = reward_output['language'] # 'zh'/'cn' or 'en'

    # ================== 语言检测 ==================
    # 使用更严格的中文检测（包含常见中文标点）
    zh_pattern = re.compile(
        r'[\u4e00-\u9fff'      # 基本汉字
        r'\u3000-\u303f'      # 中文标点符号
        r'\uff00-\uffef'      # 全角符号
        r'\u3400-\u4dbf'      # 扩展A区汉字
        r']'
    )

    # ================== 内容提取 ==================
    think_text = response

    # 有效性检查
    if len(think_text) == 0: # think内容为空返回0
        return 0.0  # 空内容
    if re.fullmatch(r'[\s\W_]+', think_text, re.UNICODE): # 只包含空白字符、非单词字符和下划线的字符串
        return 0.0  # 无意义符号内容

    if target_lang == 'en': # 如果目标语言是英文
        # ================== 字符分析 ==================
        # 首先排除所有数字和数学运算符号，然后，找到中文字符，英文字符，进行合并
        think_text = re.sub(r'[0-9+\-*/=%^$&{}[\]]', '', think_text)
        think_text = think_text.replace(" ","") # 去除空格

        zh_target_chars = zh_pattern.findall(think_text)
        en_target_chars = re.findall(r'[a-zA-Z.,!?;:"\'()-]', think_text) # 匹配所有英文字母+英文标点符号

        total_chars = len(zh_target_chars+en_target_chars)

        if total_chars < 5:  # 内容过短直接返回0
            return 0.0

        # # 目标语言字符检测
        non_target_count = len(zh_target_chars)


        # 非目标语言字符比例
        ratio = non_target_count / total_chars

        if ratio >= threshold: # 如果不一致比例超过最低限制0.2，直接返回-1，否则返回比例
            return lowest_score
        else:
            return ratio
    else: # 如果目标语言是中文
        if reward_output['reward_method']=="llm_code":
            threshold = 0.3
        # 基本惩罚计算
        input_text = reward_output['question'].split('<|User|>')[-1].split('<|Assistant|>')[0].strip()
        output_text = response
        basic_penalty = language_consistency_penalty(input_text, output_text)
        if basic_penalty >= threshold: # 如果不一致比例超过最低限制0.2，直接返回-1，否则返回比例
            return lowest_score
        else:
            return basic_penalty


def format_reward(reward_output, lowest_score=-1.0):
    response = reward_output['response']
    _open_count = len(re.findall(r"<think>", response))
    _close_count = len(re.findall(r"</think>", response))
    _sep_count = len(re.findall(r"<\|Assistant\|>",response))
    _end_count = len(re.findall(r"<|end_of_sentence|>",response))
    
    if _end_count < 1:
        return lowest_score
    if _close_count != 1 or _open_count > 0 or _sep_count > 0:
        return lowest_score

    if reward_output['reward_method'] == 'llm_general':
        return 0.0

    return 0.0

def format_reward_shortsql(reward_output, lowest_score=-1.0):
    response = reward_output['response']
    _close_count = response.count("</think>")

    if _close_count >= 1:
        return lowest_score

    return 0.0
