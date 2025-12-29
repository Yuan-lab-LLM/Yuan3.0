# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Preprocess the math dataset to parquet format
data: {"question": 问题, "answer": 答案, "language": 语言, "prompt": "", "reward_method": "llm_math"}
"""

import argparse
import os
import re
import json
import datasets
import pandas as pd
import multiprocessing
from pathlib import Path
import base64

SEED = 12345678

def get_file_list(folder, file_type_list):
    filelist = []
    for dirpath, _, filenames in os.walk(folder):
        for file in filenames:
            file_type = file.split('.')[-1]
            if file_type in file_type_list:
                file_fullname = os.path.join(dirpath, file)
                filelist.append(file_fullname)

    return filelist


def safe_column_selection(df, columns_to_keep):
    """安全地选择存在的列"""
    # 只保留存在的列
    existing_columns = [col for col in columns_to_keep if col in df.columns]

    # 可选：报告缺失的列
    missing_columns = set(columns_to_keep) - set(df.columns)
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
        print(f"Using only existing columns: {existing_columns}")

    # 如果没有列可选，返回空DataFrame或抛出异常
    if not existing_columns:
        return df

    return df[existing_columns]


def read_image_to_dict(image_path, base64_flag=False):
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    if base64_flag:
        image_bytes = base64.b64encode(image_bytes).decode('utf-8')
    # 创建字典存储图片数据
    image_dict = {
        'bytes': image_bytes,
        'path': image_path  # 字节大小
    }

    return image_dict


def process_file(index, input_text, output_dir, data_source, split_type = "train", flag_image=0):
    text_split = input_text.strip().split()
    expect_len = 0
    enable_thinking_flag = False
    if len(text_split) == 1:
        # data_source = "None"
        input_file = text_split[0]
        select_number = 1000
    elif len(text_split) == 2:
        # data_source = "None"
        select_number, input_file = text_split
        if not os.path.isfile(input_file):
            print(f"Error: Invalid line format: {input_text}")
            return

    elif len(text_split) == 3:
        select_number, input_file, data_source = text_split
    elif len(text_split) == 4:
        select_number, input_file, data_source, enable_thinking_flag = text_split
        enable_thinking_flag = True if enable_thinking_flag in ["True", "true", "TRUE", True] else False
    elif len(text_split) == 5:
        select_number, input_file, data_source, enable_thinking_flag, expect_len = text_split
        enable_thinking_flag = True if enable_thinking_flag in ["True", "true", "TRUE", True] else False
        expect_len = int(expect_len)
    else:
        print(f"Error: Invalid line format: {input_text}")
        return

    select_number = int(select_number)
    print(f"Processing idx:{index}; file {input_file}; writing to {output_dir}")
    input_file_split = list(Path(os.path.splitext(input_file)[0]).parts)
    file_name =  '_'.join(input_file_split) if len(input_file_split) < 3 else '_'.join(input_file_split[-3:])
    print(f"file_name: {file_name}")

    output_file = os.path.join(output_dir, file_name + '_' + split_type + f'_{str(enable_thinking_flag)}.parquet')
    origin_select_file = os.path.join(output_dir.replace('parquet_files', 'origin_select_files'), file_name + '_' + split_type + f'_{str(enable_thinking_flag)}.jsonl')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(origin_select_file), exist_ok=True)
    print(f"======================================")
    try:
        data = []
        error_num = 0
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()  # 去除首尾空白字符
                if not line:  # 跳过空行
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    error_num += 1
                    print(f"第{line_num}行解析错误: {e}")
                    print(f"错误内容: {line}")
        print(f"错误行数: {error_num}")
        print("使用lines模式读取")
    except Exception as e:
        print(f"逐行读取失败: {e}，尝试直接加载整个文件")
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print("使用直接加载模式读取")
        except Exception as e2:
            print(f"所有读取方式都失败: {e2}")
            data = []
    df_input = pd.DataFrame(data)

    if 'sql' in data_source:
        columns_to_keep = ["db_id", "question", "answer", "table", "yuan_input", "question_id"]
    elif 'llm_general' == data_source:
        columns_to_keep = ["input"]
    else:
        columns_to_keep = ["question", "answer", "language", "prompt", "reward_method", "source_language", "response", "extra_info"]
    if flag_image:
        columns_to_keep += ["imagepath", "copy_image_path", "image_path", "real_image_path"]
    columns_to_keep += ["expect_len"]
    df_input = safe_column_selection(df_input, columns_to_keep)
    try:
        df_input['answer'] = df_input['answer'].astype(str)
    except:
        print(f"The input file {input_file} no answer")

    for col in df_input.columns:
        if df_input[col].dtype == 'object':
            # 将列表、字典等复杂类型转换为JSON字符串
            df_input[col] = df_input[col].apply(
                lambda x: json.dumps(x, ensure_ascii=False)
                if isinstance(x, (list, dict, tuple)) and x is not None
                else x
            )
    original_rows = len(df_input)
    print(f"Original dataset: {original_rows} rows")
    if original_rows < select_number:
        rep_num = int(select_number / original_rows) + 1
        df_input = pd.concat([df_input] * rep_num, ignore_index=True)

    dataset = datasets.Dataset.from_pandas(df_input)
    select_dataset = dataset.shuffle(seed=SEED).select(range(select_number))
    # select_dataset = dataset

    # 将数据转换为列表
    data_list = list(select_dataset)

    # 按行写入JSONL文件
    with open(origin_select_file, 'w', encoding='utf-8') as f:
        for item in data_list:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')

    print(f"Select dataset: {len(select_dataset)} rows; selected number: {select_number}")

    def make_map_fn(split):
        def process_fn(example, idx):
            if "input" in example:
                input_text = eval(example.pop('input'))
                example["question"] = input_text[0]["content"][0]["text"]
                example["answer"] = ''
            if "reward_method" not in example or data_source != "None":
                example["reward_method"] = data_source
            if idx == 0:
                print("reward_method:" + example["reward_method"])

            if "question" in example:
                question_raw = example.pop("question")
            else:
                question_raw = ""
            if "source_language" in example:
                example['language'] = example.pop('source_language')
            if "answer" in example:
                answer_raw = example.pop("answer")
            elif data_source in ["sft_nogeo", "sft_knowledge", "sft_grounding", "mllm_general2"] and 'response' in example:
                answer_raw = example.pop("response")
            else:
                print(f"input_file:{input_file}")
                answer_raw = ""
            if 'response' in example:
                example.pop("response")
            solution = answer_raw
            instruction_following = ""
            if "prompt" in example:
                prompt = example.pop("prompt")
            else:
                prompt = ""
            if 'yuan_input' in example:
                question = example.pop("yuan_input") + " " + instruction_following
                middct = {"db_id": example.pop("db_id"), "table":example.pop("table"), "question_id":example.pop("question_id"), "yuan_input": question}
            else:
                question = question_raw if question_raw else prompt
                middct = {}
            if "language" not in example or not example['language']:
                language = 'cn' if bool(re.search(r'[\u4e00-\u9fa5]', question)) else "en"
                example['language'] = language
            if flag_image and '<image>' not in question:
                question = '<image>' + question
            if isinstance(solution, dict):
                solution = json.dumps(solution, ensure_ascii=False)
                answer_raw = json.dumps(answer_raw, ensure_ascii=False)
            elif not isinstance(solution, str):
                solution = str(solution)
                answer_raw = str(answer_raw)

            if "expect_len" in example:
                expect_len_line = example.pop("expect_len")
            else:
                expect_len_line = expect_len
                if idx % 1000 == 0:
                    print(f"expect_len:{expect_len_line}")

            if "expect_len_list " in example:
                expect_len_list = example.pop("expect_len_list ")
                expect_max_len = example.pop("expect_max_len")
            else:
                expect_len_list = [expect_len_line] * 5
                expect_max_len = expect_len_line

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question.replace('<n>', '\n'),
                    }
                ],
                "ability": data_source,
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "enable_thinking_flag": enable_thinking_flag,
                    "expect_len": float(expect_len_line),
                    "answer": answer_raw,
                    "question": question,
                    **middct
                },
            }

            if "extra_info" in example:
                extra_info = example.pop("extra_info")
                if isinstance(extra_info, str):
                    extra_info = eval(extra_info)
                data["extra_info"]["sft_response_text"] = extra_info["sft_response_text"]
                data["extra_info"]["sft_response_token"] = extra_info["sft_response_token"]

            if flag_image:
                data['images'] = []
                if 'image_path' in example:
                    images_path = example.pop("image_path")
                if "imagepath" in example:
                    images_path = example.pop("imagepath")
                if 'copy_image_path' in example:
                    images_path = example.pop("copy_image_path")
                if "real_image_path" in example:
                    images_path = example.pop("real_image_path")

                data["extra_info"]["image_path"] = images_path

                if isinstance(images_path, str) and images_path.startswith('['):
                    try:
                        images_path = eval(images_path)
                    except Exception as e:
                        print(f"Error: {e}; image_path:{images_path} is not a list")
                        images_path = [images_path]
                else:
                    images_path = [images_path]


                for image_path in images_path:
                    if image_path.startswith('images'):
                        image_path = os.path.join(os.path.dirname(input_file), image_path)

                    if not os.path.exists(image_path) or not os.path.isfile(image_path):
                        raise FileNotFoundError(f"Image file not found: {image_path}")
                    img_dct = read_image_to_dict(image_path)
                    data['images'].append(img_dct)
                assert len(images_path) == len(data['images'])
                question_image = question.count('<image')
                if len(images_path) > question_image:
                    question = '<image>' * (len(images_path) - question_image) + question
                    data["prompt"][0]["content"] = question

            return data

        return process_fn

    select_dataset = select_dataset.map(function=make_map_fn(split_type), with_indices=True)
    select_dataset.to_parquet(output_file)
    print(f"Finish processing idx:{index}; writing to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="./gsm8k.jsonl", help='Input paths')
    parser.add_argument("--output_path", default="./gsm8k_process/", help='Output path')
    parser.add_argument("--split_type", default="train", help='Split type')
    parser.add_argument("--data_source", default="None")
    parser.add_argument("--flag_image", default=0, help='Select number')
    args = parser.parse_args()
    if not args.output_path:
        args.output_path = os.path.basename(args.input_path)

    with open(args.input_path, 'r', encoding='utf-8') as f:
        input_lst = f.readlines()

    with multiprocessing.Pool(processes=20) as pool:
        pool.starmap(process_file, [(i, line.strip(), args.output_path, args.data_source, args.split_type, int(args.flag_image)) for i, line in enumerate(input_lst)])



if __name__ == "__main__":
    main()
