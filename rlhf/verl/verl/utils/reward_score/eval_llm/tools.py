#!/usr/bin/python
# -*- coding: UTF-8 -*-

# USED WHEN GPT-OSS AS SUPERVISED MODEL

import time
from .logger import Logging
import re


logging = Logging(limit_level='debug', colorful=True)
__sep_note = "<n>"
def clean_tab(msg_text):
    msg_text = msg_text.replace("\n", __sep_note).replace("\r",__sep_note)
    return msg_text


class TellMeTime():
    def __init__(self):
        self.start_time = None

    def record_start(self):
        self.start_time = time.time()

    def time_passed(self):
        elapsed_time = time.time() - self.start_time
        minutes, seconds = divmod(elapsed_time, 60)
        hours, minutes = divmod(minutes, 60)
        logging.info(
            f"Time Passed：{int(hours)} H {int(minutes)} M {int(seconds)} S")


def count_words_and_characters(item):

    if isinstance(item, str):
        # 计算英文单词数量（假设单词由空格或标点分隔）
        words = re.findall(r'\b\w+\b', item)
        word_count = len(words)

        # 计算汉字数量
        chinese_characters = re.findall(r'[\u4e00-\u9fff]', item)
        chinese_character_count = len(chinese_characters)

    else:
        word_count = 0
        chinese_character_count = 0
    return word_count, chinese_character_count


def split_step_text(a_text):

    flag_chi = re.findall(r'[\u4e00-\u9fa5]{2,}', a_text)
    a_text = a_text.replace('<n>', '\n')#.strip()
    def and_strip(text, split_text, bound_str):
        for idx in range(len(split_text)-1):
            split_text[idx] = split_text[idx] + bound_str
        if split_text and not text.endswith(bound_str) and split_text[-1].endswith(bound_str):
            split_text[-1] = split_text[-1][:-1]
        if split_text and not text.startswith(bound_str) and split_text[0].startswith(bound_str):
            split_text[0] = split_text[0][1:]
        split_text_sub = []
        for sub_text in split_text:
            if not sub_text.strip():
                if not split_text_sub:
                    split_text_sub.append(sub_text)
                else:
                    split_text_sub[-1] += sub_text
            else:
                split_text_sub.append(sub_text)
        return split_text_sub

    split_text = re.split(r'([\n\s]*步骤\s*[\d一二三四五六七八九十]+\s*[:：，,])|([\n\s]*Step\s*\d+\s*[:：，,])|([\n\s]*第\s*[\d一二三四五六七八九十]+\s*步\s*[:：，,])|([\n\s]*(?:首先，|其次，|接下来，|然后，|综上，|因此，|最后，)[\n\s]*)', a_text, flags=re.DOTALL)
    count_words_en, count_words_cn = count_words_and_characters(a_text)
    if len(split_text) == 1:
        split_text = a_text.split('\n')
        split_text = and_strip(a_text, split_text, '\n')

    if len(split_text) == 1:
        if flag_chi and count_words_cn > 30:
            split_text = a_text.split('。')
            split_text = and_strip(a_text, split_text, '。')

        elif not flag_chi and count_words_en > 30:
            split_text = re.split('(?<!\d)\.(?!\d)', a_text)
            split_text = and_strip(a_text, split_text, '.')
    # 对于比较长的步骤进一步拆分
    split_text_new_1 = []
    for text in split_text:
        if not text:
            continue
        count_words_en, count_words_cn = count_words_and_characters(text.strip())
        if count_words_en + count_words_cn > 30:
            split_text_mid = []
            if '\n' in text.strip():
                split_text = text.split('\n')
                split_text_mid = and_strip(text, split_text_mid, '\n')
            elif '。' in text.rstrip('。'):
                split_text = text.split('。')
                split_text_mid = and_strip(text, split_text_mid, '。')
            elif '.' in text.rstrip('.') and not flag_chi:
                split_text_mid = re.split('(?<!\d)\.(?!\d)', text)
                split_text_mid = and_strip(text, split_text_mid, '.')
            if split_text_mid:
                split_text_new_1.extend(split_text_mid)
            else:
                split_text_new_1.append(text)

        else:
            split_text_new_1.append(text)

    if len(split_text_new_1) < 2:
        return split_text_new_1

    # 合并包含步骤、因此等内容的元素
    split_text_new_2 = []
    i = 0
    while i < len(split_text_new_1):
        text = split_text_new_1[i]
        flag = False
        if re.findall(r'^(?:(?:[\n\s]*步骤\s*[\d一二三四五六七八九十]+\s*[:：，,])|(?:[\n\s]*Step\s*\d+\s*[:：，,])|(?:[\n\s]*第\s*[\d一二三四五六七八九十]+\s*步\s*[:：，,])|(?:[\n\s]*(?:首先，|其次，|接下来，|然后，|综上，|因此，|最后，)[\n\s]*))$', text, flags=re.DOTALL):# |(?:[\n\s]*\d+[、.]\s*(?!\d))

            for j in range(i + 1 , len(split_text_new_1)):
                if not split_text_new_1[j]:
                    continue

                split_text_new_2.append(text + split_text_new_1[j])
                i = j
                flag = True
                break
        if not flag:
            split_text_new_2.append(text)
        i = i + 1

    if not re.findall(r'^(?:Step\s+\d+\s*:|[\n\s]*The answer is)',split_text_new_2[-1], flags=re.DOTALL) and re.findall(r'[\n\s]*The answer is', split_text_new_2[-1], flags=re.DOTALL):
        mid_idx = split_text_new_2[-1].find('The answer is')
        if mid_idx !=-1 and mid_idx>0:
            text_end = split_text_new_2[-1][mid_idx:]
            split_text_new_2[-1] = split_text_new_2[-1][:mid_idx]
            split_text_new_2.append(text_end)
    split_text = split_text_new_2
    return split_text


def combine_small_lst_mllm(input_text, max_words = 50, min_words = 20):
    if not input_text:
        return input_text

    middle_split = input_text.replace('<n>', '\n').split('\n')
    split_length = len(middle_split)
    input_lst = ['']
    for idx, middle_text in enumerate(middle_split):
        if len(middle_text) > 1200:
            input_lst.extend(split_step_text(middle_text.replace('<n>','\n')))
        elif len(middle_text.strip()) == 0:
            input_lst[-1] = input_lst[-1] + middle_text
        else:
            input_lst.append(middle_text)
        if idx != split_length - 1:
            input_lst[-1] = input_lst[-1] + '\n'
    if len(input_lst) < 2:
        return input_lst

    combined_lst = [input_lst[0]]

    idx = 1
    while idx < len(input_lst):
        current_line = input_lst[idx]
        words_num = len(current_line.replace(' ', '').replace('\n', ''))
        last_words_num = len(combined_lst[-1].replace(' ', '').replace('\n', ''))
        if words_num < min_words:
            if current_line.strip().startswith(('\\]','\\)','\\}',']',')','}',',')) or words_num == 0 or idx == len(input_lst) - 1:
                combined_lst[-1] = combined_lst[-1] + '' + current_line
            elif combined_lst[-1].strip().endswith((':','：','{','[', '(', '（')):
                combined_lst[-1] = combined_lst[-1] + '' + current_line
            elif not combined_lst[-1].strip().endswith(('.','?', '!', ']','}',')', '。', '？', '！', '）')) and words_num + last_words_num < max_words:
                combined_lst[-1] = combined_lst[-1] + '' + current_line
            elif last_words_num < min_words:
                combined_lst[-1] = combined_lst[-1] + '' + current_line
            else:
                combined_lst.append(current_line)
        else:
            if last_words_num < min_words or combined_lst[-1].strip().endswith((':','：','{','[', '(', '（', ',', '+', '-', '/')):
                combined_lst[-1] = combined_lst[-1] + '' + current_line
            else:
                combined_lst.append(current_line)
        idx += 1
    flag_same = ''.join(combined_lst) == input_text
    assert flag_same, f"The combined string does not match the input text: {[input_text]}"
    return combined_lst


def combine_small_lst(input_text, max_words = 50, min_words = 20):
    if not input_text:
        return input_text
    input_lst = input_text.split('\n')

    if len(input_lst) < 2:
        return input_lst

    combined_lst = [input_lst[0]]

    idx = 1
    while idx < len(input_lst):
        current_line = input_lst[idx]
        words_num = len(current_line.replace(' ', '').replace('\n', ''))
        last_words_num = len(combined_lst[-1].replace(' ', '').replace('\n', ''))
        if words_num < min_words:
            if current_line.strip().startswith(('\\]','\\)','\\}',']',')','}',',')) or words_num == 0 or idx == len(input_lst) - 1:
                combined_lst[-1] = combined_lst[-1] + '\n' + current_line
            elif combined_lst[-1].strip().endswith((':','：','{','[', '(', '（')):
                combined_lst[-1] = combined_lst[-1] + '\n' + current_line
            elif not combined_lst[-1].strip().endswith(('.','?', '!', ']','}',')', '。', '？', '！', '）')) and words_num + last_words_num < max_words:
                combined_lst[-1] = combined_lst[-1] + '\n' + current_line
            elif last_words_num < min_words:
                combined_lst[-1] = combined_lst[-1] + '\n' + current_line
            else:
                combined_lst.append(current_line)
        else:
            if last_words_num < min_words or combined_lst[-1].strip().endswith((':','：','{','[', '(', '（', ',', '+', '-', '/')):
                combined_lst[-1] = combined_lst[-1] + '\n' + current_line
            else:
                combined_lst.append(current_line)
        idx += 1

    return combined_lst
