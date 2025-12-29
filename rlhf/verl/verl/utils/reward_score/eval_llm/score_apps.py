# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl.utils.reward_score.eval_llm.code_utils import check_correctness as apps_check_correctness
import json
import traceback
import ast
import re


def compute_score(completion, test_cases, continuous=False, timeout=6, num_check=10, debug=False):
    """计算代码通过率，首先全部一次性输入，如果通过则直接返回，如果不通过则对每个单独进行判断

    Args:
        completion (_type_): 包含代码的答案
        test_cases (_type_): 测试用例
        continuous (bool, optional): 是否输出连续值. Defaults to False.
        timeout (int, optional): 超时限制. Defaults to 6.
        num_check (int, optional): 最大测试用例数量. Defaults to 10，-1为全部测试
        debug (bool, optional): 调试开关. Defaults to False.

    Returns:
        _type_: 通过率和元数据
    """
    # try to get code solution from completion. if the completion is pure code, this will not take effect.
    solution = completion.split('```python')[-1].split('```')[0]
    try:
        try:
            if not isinstance(test_cases, dict): # 非字典类型转化为字典
                test_cases = json.loads(test_cases)
        except Exception as e:
            print(f"Error:{e}")

        # Complete check on all in-out pairs first. If there is no failure, per-sample test can be skipped.
        inputs = test_cases["inputs"]
        outputs = test_cases["outputs"]

        for i in range(len(inputs)):
            # 检查字典中是否存在键 'fn_name'
            if 'fn_name' in test_cases:
                # 取出键 'fn_name' 的值
                test_cases_list.append({"inputs": [inputs[i]], "outputs": [outputs[i]], "fn_name": test_cases['fn_name']})
            else:
                test_cases_list.append({"inputs": [inputs[i]], "outputs": [outputs[i]]})

        if continuous:
            # per sample test: if continuous score is needed, test first 10 samples regardless of failures
            # do not test all samples cuz some problems have enormous test cases
            metadata_list = []
            res_list = []
            for test_case_id, test_case in enumerate(test_cases_list):
                res, metadata = apps_check_correctness(in_outs=test_case, generation=solution, timeout=timeout, debug=debug)
                try:
                    metadata = dict(enumerate(metadata))[0]  # metadata can be empty occasionally
                except Exception as e:
                    metadata = {}
                metadata["test_case"] = {}
                metadata["test_case"]["input"] = str(test_case["inputs"][0])
                metadata["test_case"]["output"] = str(test_case["outputs"][0])
                metadata["test_case"]["res"] = str(res)
                metadata_list.append(metadata)
                res_list.extend(res)

                if num_check>0 and test_case_id >= num_check-1:
                    break
            res_count = len(res_list) if len(res_list) > 0 else 1
            success = sum(map(lambda x: x == True, res_list)) / res_count
    except Exception as e:
        traceback.print_exc(10)
        success = False
        metadata_list = None
    return success, metadata_list


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


def custom_serializer(obj): # json.dumps无法处理set，需转化为list
    """
    自定义序列化函数，支持多种类型：
    - set: 转换为 list
    - tuple: 转换为 list
    - datetime: 转换为 ISO 格式的字符串
    - decimal.Decimal: 转换为浮点数
    - bytes: 转换为 Base64 编码的字符串
    - numpy 数组: 转换为列表
    - 自定义对象: 转换为对象的 __dict__ 属性（如果有的话）
    - 显式处理字符串和数字（可选）
    """
    if isinstance(obj, set):
        return list(obj)
    else:
        return obj

def parse_assert_statement(assert_str):
    """
    将 assert 语句字符串解析为 (输入, 输出, 函数名) 三元组。

    参数:
        assert_str (str): assert 语句字符串，例如 "assert findMedian([[0, 1], [1, 1]], 2) == [1.0, None]"

    返回:
        tuple: (输入, 输出, 函数名) 三元组
    """
    # 去除注释部分
    assert_str = assert_str.strip()
    # 使用正则表达式匹配函数名、输入参数和预期输出
    # \s+ 表示匹配一个或多个空白字符（空格、制表符等）,\w+ 匹配一个或多个字母、数字或下划线，表示函数名
    # .* 匹配一个或多个字符，表示函数的输入参数。
    # \s* 表示匹配零个或多个空白字符。
    # .* 匹配任意字符（除换行符外）零次或多次，表示预期输出
    # [^=]+匹配一个或多个不等于 = 的字符
    pattern = r"assert\s+(\w+)\((.*)\)\s*==\s*(.*)"
    pattern_with_is = r"assert\s+(\w+)\((.*)\)\s*is\s+([^=]+)\s*==\s*(True|False)"
    match = re.match(pattern, assert_str)
    
    if not match:
        print("pattern_with_is: ",assert_str)
        match = re.match(pattern_with_is, assert_str)
        if not match:
            return json.dumps(None),json.dumps(None),json.dumps(None) # 无法解析
    # 提取函数名、输入参数和预期输出
    func_name = match.group(1)  # 函数名
    input_args = match.group(2).strip()  # 输入参数
    expected_output = match.group(3).strip().split('#')[0].strip()  # 预期输出,去除注释部分

    # 处理参数部分，将其转换为元组
    try:
        # 包裹参数部分以形成元组
        input_args = ast.literal_eval(f'({input_args})')
    except:
        # 处理可能的语法错误，例如单个参数时;或者是一段可执行代码
        try:
            input_args = eval(input_args)
        except:
            pass
    # 如果args_tuple是元组转换为list（对于多个参数情况）,如果本身是list则不进行转换
    if isinstance(input_args, tuple):
        input_args = list(input_args)
        try:
            input_args = "\n".join(json.dumps(custom_serializer(item)) for item in input_args)
        except Exception as e:
            input_args = json.dumps(None) # 无法解析
        # input_args = [[item] for item in input_args]
    else:
        try:
            input_args = json.dumps(input_args)
        except Exception as e:
            input_args = json.dumps(None) # 无法解析
    # 处理输出部分
    try:
        expected_output = ast.literal_eval(expected_output)  # 安全解析预期输出
    except:
        pass
    if isinstance(expected_output, tuple):
        try:
            expected_output = json.dumps(expected_output)
        except Exception as e:
            expected_output = json.dumps(None) # 无法解析

    elif isinstance(expected_output, list):
        if len(expected_output)==0:
            expected_output = str(expected_output)
        else:
            expected_output = [item for item in expected_output]
            try:
                expected_output = json.dumps(expected_output)
            except Exception as e:
                expected_output = json.dumps(None) # 无法解析
    else:
        try:
            expected_output = json.dumps(expected_output)
        except Exception as e:
            expected_output = json.dumps(None) # 无法解析

    return input_args, expected_output, func_name


def format_trans(solution_str, ground_truth):
    """转换输入格式为json.loads可读取

    Args:
        solution_str (_type_): 答案字符串
        ground_truth (_type_): 测试用例字符串

    Returns:
        _type_: _description_
    """
    solution_str = solution_str.replace('<eop>', '')
    solution_str = solution_str.replace('<n>', '\n')
    if len(ground_truth)==0:
        gt_dict = {"inputs":[],"outputs":[]}
        return solution_str, gt_dict

    if "assert" in ground_truth: # assert str split by <n> humaneval/mbpp/leetcode
        if '<n>' in ground_truth:
            ground_truth_str = ground_truth.replace('<n>', '\n').strip().replace('\n', '<n>')
            unittest_lst = ground_truth_str.split('<n>')
        else:
            try:
                unittest_lst = eval(ground_truth)
            except:
                print("Unevaluated!")
                gt_dict = {"inputs":[],"outputs":[]}
                return solution_str, gt_dict

        input_lst = []
        outputlst = []
        for test_case in unittest_lst:
            input_args, expected_output, func_name = parse_assert_statement(test_case)
            input_lst.append(input_args)
            outputlst.append(expected_output)
        
        gt_dict = {"inputs":input_lst,"outputs":outputlst,"fn_name":func_name} # 单个输入或者无funcname，则转换为list
        
        return solution_str, gt_dict
    elif all(check_list_elements(ground_truth)): # assert list
        unittest_lst = ground_truth
        input_lst = []
        outputlst = []
        for test_case in unittest_lst:
            input_args, expected_output, func_name = parse_assert_statement(test_case)
            input_lst.append(input_args)
            outputlst.append(expected_output)
        gt_dict = {"inputs":input_lst,"outputs":outputlst,"fn_name":func_name} # 单个输入或者无funcname，则转换为list

        return solution_str,gt_dict
    else: # atcoder/codeforces
        # 初始化结果字典
        result = {"inputs": [], "outputs": []}
        if type(ground_truth) is str:
            # 判断是否是字符串类型，则需要解析
            if ground_truth.startswith('<n>'):
                every_unittest = ground_truth.split('<n>')[1:] # list字符串  [{"input": "5 1 1", "output": "0"},{"input": "3 100 1000", "output": "900"},{"input": "3 100 100", "output": "900"},{"input": "1 1 1", "output": "999999999"}]
                every_unittest = [ast.literal_eval(item) for item in every_unittest] # list 每个元素是{'input': '2 5 7', 'output': '5'}
            else:
                every_unittest=convert_string_to_dicts_with_eval(ground_truth)

            # 遍历列表，提取输入和输出
            for item in every_unittest:
                result["inputs"].append(item["input"])
                result["outputs"].append(item["output"])
        elif type(ground_truth) is list:

            # 遍历列表，提取输入和输出
            for item in ground_truth:
                result["inputs"].append(item["input"])
                result["outputs"].append(item["output"])
        
        gt_dict = {"inputs":list(map(str, result["inputs"])),"outputs":list(map(str, result["outputs"]))}
        return solution_str, gt_dict
