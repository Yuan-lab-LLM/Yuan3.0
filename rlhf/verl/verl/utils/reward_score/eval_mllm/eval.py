import json
import os
import random
import uuid
import time
import subprocess
import ast
import requests
import re
from math_verify import LatexExtractionConfig, parse, verify,StringExtractionConfig,ExprExtractionConfig
from latex2sympy2_extended import NormalizationConfig

import pandas as pd

from typing import Optional
import itertools



def math_verify_extract(content):
    ind = content.rfind('\\boxed')
    if ind == -1 and 'boxed' in content:
        ind=content.rfind('boxed')
        content=f'$\\{content[ind:]}$'.replace('<n>','\n').replace('\,',',')
    elif 'boxed' not in content:
        content=f'${content}$'.replace('<n>','\n').replace('\,',',')
    else:
        content=f'${content[ind:]}$'.replace('<n>','\n').replace('\,',',')
    if '}\pi' in content:
        content = content.replace('}\pi', '}*\pi')
    if '},' in content:
        content = content.replace('},', '}')
    if '\\).' in content:
        content = content.replace('\\).', '').replace('\\(', '')
    if '\).' in content:
        content = content.replace('\).', '').replace('\(', '')
    match_is = re.search(r'\bis\b\s*(.+)', content)
    if match_is:
        content = match_is.group(1).strip()


    content=replace_fraction(content)
    answer_parsed = parse(
        content,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    equations=False, 
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            ),
            ExprExtractionConfig()
        ],
        extraction_mode="first_match",
    )
    
    return answer_parsed

def math_verify_reward(content,sol):
    content_print = content
    if 'pi' in sol and '\pi' not in sol:
        sol = sol.replace('pi', '\pi')
    if 'sqrt' in sol and '\sqrt' not in sol:
        sol = sol.replace('sqrt', '\sqrt')
    sol = sol.replace('(', '{').replace(')', '}')
    gold_parsed=math_verify_extract(sol)
    answer_parsed=math_verify_extract(content)
    if verify(answer_parsed, gold_parsed,float_rounding=2,numeric_precision=15):
        return True
    return False


def replace_fraction(text):
    if 'frac' not in text:
        return text
    f_t = re.findall(r'(?:\s*\}){3,}', text)
    if f_t:
        return text
    pattern = r'\\[d]?frac\{((?:[^{}]|\{[^{}]*\})+)\}\{((?:[^{}]|\{[^{}]*\})+)\}'
    replacement = r'\1/\2'
    replaced_text = re.sub(pattern, replacement, text)

    return replaced_text


def is_percent_number_match(num1, num2):

    def parse_number(num_str):
        num_str = num_str.replace('\\%', '%')

        parsed_value=[]

        if num_str.endswith('%'):
            try:
                value = float(num_str[:-1].strip()) / 100
                parsed_value.append(value)
                value = float(num_str[:-1].strip())
                parsed_value.append(value)
                return parsed_value
            except ValueError:
                return None
        else:
            try:
                parsed_value.append(float(num_str))
                return parsed_value
            except ValueError:
                return None

    def check_match_with_tolerance(list1, list2, tolerance=1e-10):
        return any(abs(item1 - item2) < tolerance for item1, item2 in itertools.product(list1, list2))

    val1 = parse_number(num1)
    val2 = parse_number(num2)

    if val1 is None or val2 is None:
        return False

    return check_match_with_tolerance(val1,val2)

def normalize_punctuation(text):
    text = text.replace('*', '')
    brackets_to_remove = ['{', '}', '[', ']', '(', ')']
    for bracket in brackets_to_remove:
        text = text.replace(bracket, '')
    text = text.replace('_', '')
    text = text.replace(',', '')
    text = text.replace('and', '')
    text = text.replace(' ', '')
    text = text.replace('\\text', '')
    text = text.replace('\text', '')
    text = text.replace('text', '')
    text = text.replace('$', '')
    text = text.rstrip('.')
    text = text.replace('√', '')
    text = text.replace('\\(','')
    text = text.replace('\\)','')
    text = text.replace('\\','')
    
    return text


def remove_units(text):
    latex_units = re.findall(r'\\[a-zA-Z]+\s*\{\s*[^}]*\s*\}', text)
    protected = {}
    
    for i, unit in enumerate(latex_units):
        placeholder = f"__LATEX_UNIT_{i}__"
        protected[placeholder] = unit
        text = text.replace(unit, placeholder)
    
    units = [
        'square centimeter', 'square meter', 'square kilometer', 'square inch', 'square foot',
        'cubic centimeter', 'cubic meter', 'cubic inch', 'cubic foot',
        'meters per second', 'kilometers per hour', 'miles per hour',
        'millimeter', 'centimeter', 'meter', 'kilometer', 'inch', 'foot', 'yard', 'mile',
        '毫米', '厘米', '米', '千米', '英寸', '英尺', '码', '英里',
        'square millimeter', 'square centimeter', 'square meter', 'square kilometer',
        'square inch', 'square foot', 'square yard', 'square mile',
        '平方毫米', '平方厘米', '平方米', '平方千米',
        'cubic millimeter', 'cubic centimeter', 'cubic meter', 'cubic inch', 'cubic foot',
        'milliliter', 'liter', 'gallon',
        '立方毫米', '立方厘米', '立方米',
        'second', 'minute', 'hour', 'day', 'week', 'month', 'year',
        '秒', '分钟', '小时', '天', '周', '月', '年',
        'milligram', 'gram', 'kilogram', 'pound', 'ounce',
        '毫克', '克', '千克', '磅', '盎司',
        'degree', 'radian',
        '度', '弧度',
        'ampere', 'volt', 'watt', 'ohm', 'hertz',
        '安培', '伏特', '瓦特', '欧姆', '赫兹',
        'pascal', 'joule', 'newton', 'mole', 'kelvin', 'celsius',
        'mm', 'cm', 'km', 'ft', 'yd', 'mi',
        'mm²', 'cm²', 'm²', 'km²', 'in²', 'ft²', 'yd²', 'mi²',
        'mm^2', 'cm^2', 'm^2', 'km^2', 'in^2', 'ft^2', 'yd^2', 'mi^2',
        'mm³', 'cm³', 'm³', 'km³', 'in³', 'ft³', 'yd³', 'mi³',
        'mm^3', 'cm^3', 'm^3', 'km^3', 'in^3', 'ft^3', 'yd^3', 'mi^3',
        'kWh', 'KWH', 'kws', 
        'yuan', 'million', 'Baht per hour', 'Baht', 'baht', 'books', 'centimeters', 'ml', 'ML', 'mL','Ml', 'YUAN', 'panels', 'charity galas',
        '^\circ',
        '(C)', 'N·m', 'cars.', 'MT/s', 'Gbps', 'patients', 'parameters', 'unit/s', 'cameras', 'blue', 'elephants', 'psi', 'routes', 'PM', 'MB/s', '分米', 'floors', '平方厘米', 'types', 'tens', 'Hz', 'scenarios', 'room-months', 'revolutions', 'feet', '米', 'centimeters', 'actors', 'blocks', 'mm.', 'months.', 'rotations', 'dm.', 'yr', 'pc', 'birds', 'coins', 'questions', 'figures', 'Million', '厘米', '°', 'colors', 'Train', 'clusters', 'item', 'volts', '米/秒', 'euros', 'top', 'cm', 'connections', 'ppm', 'tiles', 'games', 'TWh', '[升]', 'pieces', 'players', 'seeds', 'fans', 'PHP', 'shares', 'adventures', 'μm', 'year.', 'bacteria', 'bits/minute', 'bushels', 'months', 'Umeå.', 'stars', '(neon)', 'claims', 'trees', 'words', 'calls', 'pounds', 'minutes', 'buckets', 'incidents', 'degrees.', 'mg/mL', 'kilometers', 'year', 'cells/μL', 'files', 'calories', '¾', 'dm', 'turns', 'episodes', 'onwards', 'N⋅m', 'km/h', 'mg/L', 'pm', 'GB', 'ft/s', 'hearts', '天', 'rad/day', 'lbs', 'quarters', 'MP.', 'sessions', 'calories/day', 'props', 'reports', 'ms', 'horizons', 'unit', 'joules', 'diapers', 'am', 'GB/s', 'dB', 'hectares', 'combinations', 'CE', 'stimuli', 'tickets', 'meters', '元', '小时', 'centimeter', 'strokes.', 'pages', 'rectangles', '千米', 'category', 'individuals.', 'License', 'kcal', 'cm.', 'MW', 'seconds.', 'metres', 'ideals', '平方分米', 'offspring', 'pairs', 'bytes', 'angles', 'decimeters', 'mcg/dL', 'Mbps', 'events', 'month', 'faces', 'steps', '元', 'projects', 'professionals', 'Umeå', 'dollars', '平方米', 'W', 'km', 'cents', 'boys', 'hour', 'orbits', 'minutes.', 'cases', 'packages', 'shots', 'times', 'seasons', 'cars', 'signatures', 'years.', 'days', 'yuan', 'participants', 'gallons', 'mSv', 'cm/week', 'reinforcements', 'frames', 'vehicles', 'days.', 'RPM', 'matches', 'strips', 'countries', 'each', 'nanometers.', 'dancers', 'extras', 'kg/ha', 'rounds', 'week', 'GPa', 'way', 'US$', 'mph', '平方分米', 'm/s', 'tourists', 'developers', 'towels', '岁', 'meters.', 'triangles', 'decibel-seconds', 'visits', 'mm', 'MP', 'credits.', 'channels', 'stops.', 'rows', 'games.', 'KiB', 'equations', 'only', 'squares', 'centuries', 'kWh', 'Yuan.', 'dogs', 'positions', 'meter', 'sides', 'pots', 'mL', 'octaves', 'references', 'years', 'liters', 'violations/year', 'USD', 'surfers', 'minute', 'points', 'races', 'tutors', 'audiobooks', 'items', 'households', 'panels', 'µm', 'barrels', 'July', 'fps', 'segments', 'ppb', 't-shirts', 'strokes', 'copies', 'N/kg', 'μJ', 'outfits', 'JOPLIN', '°C/h', 'attendees', 'units', '厘米', '(Yuan)', 'Mg/Day', '⅜', 'INR/kg', 'sports', '(个)', 'beats', 'dBr', 'subgroups', 'in.', 'edges', 'mg', 'grams', 'BPM', 'knots', 'members', 'customers', 'chapters', '或', 'pixels', 'puzzles', 'BC', 'ft', 'second', 'notebooks.', 'miles', '(methods)', 'ml', 'cm', '海里/时', 'goals', 'kilograms', 'animals', 'iterations', 'referrals', 'level', 'mpg', 'season', 'ways', 'weeks', 'Degrees', 'MWh', 'hours', 'tables', 'Sthree', 'Yuan', 'pigs', 'milliseconds', '米', 'bits', 'Loudon', 'kilometer', 'generators', 'ft.', 'machines.', 'white', 'petals', 'points.', 'cycle/year', 'holes', '(cm)', 'activities', 'TB', 'bushels/year', 'thousand', 'DPI', 'degrees', 'Janka', 'mg/day', 'seconds', 'guests', 'cubes.', 'rhombuses', 'MHz', 'dimensions', 'books', 'beds', 'NITkg', 'cent', 'kg', 'knots.', 'day', 'месяц', 'hops', 'watermelons', 'archers', 'passengers', '(A)', 'inches', 'MPa', 'PEU', 'cubes', 'teeth', '头', 'balls', 'people', 'million', 'sheets', 'acres', 'objects', 'goals.', 'nanometers', 'kN', 'November', 'tons', 'inch', 'μg/m³'

    ]
    
    units.sort(key=len, reverse=True)
    unit_patterns = []
    for unit in units:
        unit_patterns.append(rf'\b{re.escape(unit)}\b')
    
    combined_pattern = '|'.join(unit_patterns)
    text = re.sub(combined_pattern, '', text, flags=re.IGNORECASE)

    for placeholder, unit in protected.items():
        text = text.replace(placeholder, unit)
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text


from verl.utils.reward_score.eval_mllm.utilities import *

def levenshtein_distance_mllm(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def get_most_similar_mllm(prediction, choices):

    distances = [levenshtein_distance_mllm(prediction, choice) for choice in choices]
    ind = distances.index(min(distances))
    return choices[ind], ind


def safe_equal(prediction, answer):

    try:
        if prediction == answer:
            return True
        return False
    except Exception as e:
        print(e)
        return False


def extract_options(text):

    dict_options = extract_dict_options(text)
    if dict_options:
        return dict_options
    
    patterns = [
        r'[A-Z]\.\s*([^\nA-Z]+?)(?=\s*[A-Z][\.:\)]|\s*$|\n|$)',
        r'[A-Z]\)\s*([^\nA-Z]+?)(?=\s*[A-Z][\.:\)]|\s*$|\n|$)',
        r'[A-Z]:\s*([^\nA-Z]+?)(?=\s*[A-Z][\.:\)]|\s*$|\n|$)',
        r'[A-Z]\.\s*([^A-Z]+?)(?=\s*[A-Z][\.:\)]|\s*$)',
        r'[A-Z]\)\s*([^A-Z]+?)(?=\s*[A-Z][\.:\)]|\s*$)',
        r'[A-Z]:\s*([^A-Z]+?)(?=\s*[A-Z][\.:\)]|\s*$)',
        r'Choices:\s*\(\s*(?:\n\s*([^\n\)]+?)){2,12}\s*\)',
        r'(?<!\w)(?:\n\s*([^\n]+?)){2,12}(?=\n\s*[A-Z]|\n\s*$|\s*$)',
    ]
    
    all_options = []
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        if matches:
            cleaned_options = []
            for option in matches:
                cleaned = clean_option_text(option)
                if cleaned and is_valid_option(cleaned):
                    cleaned_options.append(cleaned)
            
            if 2 <= len(cleaned_options) <= 10:
                if len(cleaned_options) > len(all_options):
                    all_options = cleaned_options
    
    return all_options if all_options else []

def extract_dict_options(text):
    dict_patterns = [
        r"\{[^{}]*['\"][A-Za-z]['\"][^{}]*:[\s]*['\"]([^'\"]+)['\"][^{}]*\}",
        r"choices\s*\{[^{}]*['\"][A-Za-z]['\"][^{}]*:[\s]*['\"]([^'\"]+)['\"][^{}]*\}",
    ]
    
    for pattern in dict_patterns:
        dict_match = re.search(pattern, text)
        if dict_match:
            dict_text = dict_match.group(0)
            value_pattern = r"['\"][A-Za-z]['\"][\s]*:[\s]*['\"]([^'\"]+)['\"]"
            values = re.findall(value_pattern, dict_text)
            if values:
                return [clean_option_text(v) for v in values]
    
    kv_pattern = r"[A-Za-z][\s]*:[\s]*['\"]([^'\"]+)['\"]"
    kv_matches = re.findall(kv_pattern, text)
    if len(kv_matches) >= 2:
        option_letters = re.findall(r'([A-Za-z])[\s]*:[\s]*[\'"]', text)
        if len(option_letters) >= 2:
            unique_letters = sorted(set(option_letters))
            if len(unique_letters) >= 2:
                return [clean_option_text(v) for v in kv_matches]
    
    return []

def clean_option_text(option):
    if not option:
        return ""
    cleaned = option.strip()
    
    cleaned = re.sub(r'^[\'\"\(\{\[<]|[\'\"\)\}\]>\.;,]$', '', cleaned)
    cleaned = re.sub(r'^[A-Za-z]\s*[\.:\)]\s*', '', cleaned)
    
    return cleaned.strip()

def is_valid_option(option):
    invalid_patterns = [
        r'^选择题$', r'^填空题$', r'^解答题$', r'^计算题$',
        
    ]
    
    for pattern in invalid_patterns:
        if re.search(pattern, option, re.IGNORECASE):
            return False
    
    return True

def validate_option_order(text, option_count):
    expected_options = [chr(65 + i) for i in range(option_count)]  
    formats = [
        [f"{opt}." for opt in expected_options],  
        [f"{opt})" for opt in expected_options],  
        [f"({opt})" for opt in expected_options],  
        [f"{opt}:" for opt in expected_options]   
    ]
    
    for fmt in formats:
        found_all = all(marker in text for marker in fmt)
        if found_all:
            return True
    
    return False

def extract_options_long(text):
    patterns = [
        r'[A-J]\.\s*(.*?)(?=\s*[A-J]\.|\s*$)',  
        r'\([A-J]\)\s*(.*?)(?=\s*\([A-J]\)|\s*$)',  
        r'[A-J]\)\s*(.*?)(?=\s*[A-J]\)|\s*$)',  
        r'[A-J]:\s*(.*?)(?=\s*[A-J]:|\s*$)'  
    ]

    options = []

    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches and 2 <= len(matches) <= 10:  
            options = [match.strip() for match in matches]
            if validate_option_order(text, len(options)):
                break

    if not options:
        options = extract_options_by_line_long(text)

    return options

def extract_options_by_line_long(text):
    option_lines = []

    lines = text.split('\n')

    for line in lines:
        line = line.strip()
        option_match = re.match(r'^([A-J])[\.\):\s]*|^\(([A-J])\)\s*', line)
        if option_match:
            content = re.sub(r'^[A-J][\.\):\s]*|^\([A-J]\)\s*', '', line)
            if content:
                option_lines.append(content.strip())

    if 2 <= len(option_lines) <= 10 and validate_option_order(text, len(option_lines)):
        return option_lines

    return []

def extract_options_advanced(text):


    newline_patterns = [
        r'<n>',           
        r'<br\s*/?>',     
        r'\\r\\n',        
        r'\\n',           
        r'\\\\n',         
        r'&#10;',         
        r'&#13;',         
    ]
    
    normalized_text = text
    for pattern in newline_patterns:
        normalized_text = re.sub(pattern, '\n', normalized_text, flags=re.IGNORECASE)
     
    text = normalized_text
    text = text.replace('\(','(')
    text = text.replace('\)',')')
    text = text.replace('\\(','(')
    text = text.replace('\\)',')')
    options = extract_dict_options(text)
    if len(options) > 1:
        return options
    
    options = extract_options(text)
    if len(options) > 1:
        return options
    
    lines = text.split('\n')
    option_lines = []
    
    for line in lines:
        if (re.search(r'[A-Z][\.:\)]\s*', line) or 
            re.search(r"[A-Za-z]\s*:\s*['\"]", line)):
            option_lines.append(line)
    
    if option_lines:
        option_text = ' '.join(option_lines)
        options = extract_options(option_text)
        if not options or len(options) == 0:
            options = extract_dict_options(option_text)
    if len(options) > 1:
        return options

    options = extract_options_long(text)
    if len(options) > 1:
        return options
    
    options = extract_options_by_line_long(text)

    return options


def normalize_extracted_answer_mllm(extraction, choices, question_type='multi_choice'):
    if question_type == 'multi_choice':
        if isinstance(extraction, str):
            extraction = extraction.strip()
        else:
            try:
                extraction = str(extraction)
            except:
                extraction = ''
        letter = re.findall(r'[A-Z]', extraction)
        if len(letter) > 0:
            extraction = letter[0].upper()

        options = [chr(ord('A') + i) for i in range(len(choices))]

        if extraction in options:
            ind = options.index(extraction)
            extraction = choices[ind]
            key = options[ind]
        else:
            extraction, ind = get_most_similar_mllm(extraction, choices)
            key = options[ind]
        assert extraction in choices


    return extraction, key


def is_not_alphanumeric_manual(char):
    if char == '':
        return True
    if 'a' <= char <= 'z':
        return False
    if 'A' <= char <= 'Z':
        return False
    if '0' <= char <= '9':
        return False
    
    return True

def is_multiple_choice_question(text):
    patterns = [
        r'^[A-Z][\.\)\:]',  # A. A) A:
        r'^\([A-Z]\)',      # (A) (B)
    ]
    
    option_count = 0
    found_options = set()
    
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        for pattern in patterns:
            if re.match(pattern, line):
                option_char = line[0] if line[0].isupper() else line[1] if len(line) > 1 else ''
                if option_char and option_char.isupper():
                    found_options.add(option_char)
                    option_count += 1
                break
    
    if option_count >= 2:
        sorted_options = sorted(found_options)
        for i in range(1, len(sorted_options)):
            if ord(sorted_options[i]) - ord(sorted_options[i-1]) != 1:
                return False
        return True
    
    return False

def normalize_latex_extended(text):

    latex_commands = re.findall(r'\\[a-zA-Z]+', text)
    protected = {}
    
    for i, cmd in enumerate(latex_commands):
        placeholder = f"__LATEX_CMD_{i}__"
        protected[placeholder] = cmd
        text = text.replace(cmd, placeholder)
    

    replacements = [

        (r'√\s*(\d+)', r'\\sqrt{\1}'),
        (r'√\s*\{([^}]+)\}', r'\\sqrt{\1}'),
        (r'sqrt\s*(\d+)', r'\\sqrt{\1}'),
        (r'sqrt\s*\{([^}]+)\}', r'\\sqrt{\1}'),

        (r'π', r'\\pi'),
        (r'\bpi\b', r'\\pi'),
        (r'α', r'\\alpha'),
        (r'β', r'\\beta'),
        (r'γ', r'\\gamma'),
        (r'θ', r'\\theta'),
        (r'δ', r'\\delta'),
        (r'ε', r'\\epsilon'),

        (r'±', r'\\pm'),
        (r'\+/-', r'\\pm'),
        (r'×', r'\\times'),
        (r'÷', r'\\div'),
        (r'≤', r'\\leq'),
        (r'>=', r'\\geq'),
        (r'≥', r'\\geq'),
        (r'≈', r'\\approx'),
        (r'≠', r'\\neq'),
        (r'!=', r'\\neq'),
        (r'∞', r'\\infty'),
        (r'inf', r'\\infty'),
        (r'∂', r'\\partial'),

        (r'⋅', r'\\cdot'),
        (r'∗', r'\\ast'),
        (r'∑', r'\\sum'),
        (r'∫', r'\\int'),
        (r'lim', r'\\lim'),

        (r'½', r'\\frac{1}{2}'),
        (r'¼', r'\\frac{1}{4}'),
        (r'¾', r'\\frac{3}{4}'),
        (r'(\d+)/(\d+)', r'\\frac{\1}{\2}'),
    ]
    
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    
    for placeholder, cmd in protected.items():
        text = text.replace(placeholder, cmd)
    
    return text


def normalize_numbers_extended(text):
    word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
        'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
        'eighteen': '18', 'nineteen': '19', 'twenty': '20',
        'thirty': '30', 'forty': '40', 'fifty': '50', 
        'sixty': '60', 'seventy': '70', 'eighty': '80', 'ninety': '90',
        'hundred': '00', 'thousand': '000', 'million': '000000'
    }
    
    compound_pattern = r'(\w+)-(\w+)'
    def replace_compound(match):
        first = match.group(1).lower()
        second = match.group(2).lower()
        if first in word_to_digit and second in word_to_digit:
            return str(int(word_to_digit[first]) + int(word_to_digit[second]))
        return match.group(0)
    
    text = re.sub(compound_pattern, replace_compound, text, flags=re.IGNORECASE)
    
    for word, digit in word_to_digit.items():
        text = re.sub(r'\b' + word + r'\b', digit, text, flags=re.IGNORECASE)
    
    roman_to_arabic = {
        'I': '1', 'II': '2', 'III': '3', 'IV': '4', 'V': '5',
        'VI': '6', 'VII': '7', 'VIII': '8', 'IX': '9', 'X': '10'
    }
    
    for roman, arabic in roman_to_arabic.items():
        text = re.sub(r'\b' + roman + r'\b', arabic, text)
    
    return text

def extract_and_combine_latex_ordered(text):
    stack = []
    extracted = []
    combined = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == '\\' and i + 5 <= n and text[i+1:i+5] == 'text':
            # 记录 \text 之前的非 \text 内容
            combined.append(text[:i])
            # 跳过 \text
            i += 5
            if i < n and text[i] == '{':
                stack.append(i)
                i += 1
            # 提取 \text{...} 内容
            brace_level = 0
            content_start = i
            while i < n:
                if text[i] == '{':
                    brace_level += 1
                elif text[i] == '}':
                    if brace_level == 0:
                        break
                    brace_level -= 1
                i += 1
            if i < n and text[i] == '}':
                extracted_content = text[content_start:i]
                extracted.append(extracted_content)
                combined.append(extracted_content)  # 按顺序拼接
                i += 1
                text = text[i:]
                i = 0
                n = len(text)
        else:
            i += 1
    # 添加剩余的非 \text 内容
    if text:
        combined.append(text)
    return ''.join(combined)

def normalize_month_abbreviations_extended(text):
    month_mapping = {
        'Jan': 'January', 'Feb': 'February', 'Mar': 'March', 'Apr': 'April',
        'May': 'May', 'Jun': 'June', 'Jul': 'July', 'Aug': 'August',
        'Sep': 'September', 'Oct': 'October', 'Nov': 'November', 'Dec': 'December',
        'Jan.': 'January', 'Feb.': 'February', 'Mar.': 'March', 'Apr.': 'April',
        'Jun.': 'June', 'Jul.': 'July', 'Aug.': 'August', 'Sep.': 'September',
        'Oct.': 'October', 'Nov.': 'November', 'Dec.': 'December'
    }
    
    for abbrev, full in month_mapping.items():
        text = re.sub(r'\b' + re.escape(abbrev) + r'\b', full, text, flags=re.IGNORECASE)
    
    return text


def apply_preprocessing_extended(text):
    processed = str(text)

    step_functions = {
        'remove_units': remove_units,  
        'normalize_latex_extended': normalize_latex_extended,  
        'normalize_month_abbreviations_extended': normalize_month_abbreviations_extended, 
        'normalize_numbers_extended': normalize_numbers_extended, 
    }


    preprocessing_steps_base = [
            'normalize_latex_extended',
            'remove_units',
            'normalize_numbers_extended',
            'normalize_month_abbreviations_extended',
        ]

    for step in preprocessing_steps_base:
        processed = step_functions[step](processed)
    
    return processed.strip()

def mllm_normal_score(question,predict,answer):
    base_predict = predict

    answer_origin = str(answer).replace("degrees","").replace("cm^2","").replace("cm^2^","").replace("厘","").replace("平方","").replace("cm^{2}","").replace("√","\sqrt").replace("<n>","\n").replace('cm²', '').replace('米', '').replace('°', '').replace('cm', '').replace('()', '').replace('$','').replace('±','\\pm').replace('μg/m³', '').replace('。', '').replace('元', '').strip()
    predict_origin = str(predict).replace("degrees","").replace("cm^2","").replace("cm^2^","").replace("厘","").replace("平方","").replace("cm^{2}","").replace("√","\sqrt").replace("<n>","\n").replace('cm²', '').replace('米', '').replace('°', '').replace('cm', '').replace('()', '').replace('$','').replace('±','\\pm').replace('μg/m³', '').replace('。', '').replace('元', '').strip()

    answer = apply_preprocessing_extended(answer_origin)
    predict = apply_preprocessing_extended(predict_origin)

    answer = remove_units(answer)
    predict = remove_units(predict)
    answer = '\\boxed{'+answer+'}'

    if answer == predict:
        return 1.0
    else:
        pattern = r"\\?boxed{(.*?)}"
        pattern = r"\\?boxed{(.+)}"
        answer_matches = re.findall(pattern, answer)
        if not answer_matches:
            return 0.0
        else:
            predict_matches = re.findall(pattern, predict)
            if not predict_matches:
                return 0.0
            else:
                final_answer = predict_matches[-1]
                answer_m=answer_matches[-1]
                final_answer = remove_units(final_answer)
                answer_m = remove_units(answer_m)
                if predict_matches[-1]==answer_matches[-1] or normalize_punctuation(predict_matches[-1]).lower().strip() == normalize_punctuation(answer_matches[-1]).lower().strip():
                    return 1.0
                elif final_answer.rstrip('.').lower().strip() == answer_m.rstrip('.').lower().strip() or final_answer.rstrip('^').lower().strip() == answer_m.rstrip('^').lower().strip() or final_answer.replace(' ','').lower().strip() == answer_m.replace(' ','').lower().strip() or final_answer.rstrip('m.').lower().strip() == answer_m.rstrip('m.').lower().strip():
                    return 1.0
                elif answer_matches[-1].lower().strip().rstrip('.') == 'yes' or answer_matches[-1].lower().strip().rstrip('.') == 'no':
                    single_answer = re.split(r'[,.]', predict_matches[-1])[0]
                    if single_answer.lower().strip().rstrip('.') == answer_matches[-1].lower().strip().rstrip('.') :
                        return 1.0
                elif final_answer.endswith('.0.'):
                    final_answer = final_answer[:-3]
                    if final_answer == answer_m:
                        return 1.0
                elif answer_m.endswith('.0.'):
                    answer_m = answer_m[:-3]
                    if final_answer == answer_m:
                        return 1.0
                else:
                    if 'text{' in final_answer:
                        final_answer = extract_and_combine_latex_ordered(final_answer)
                        final_answer = final_answer.replace('\\','')
                        final_answer = final_answer.replace(' ','')
                        answer_m = answer_m.replace(' ','')
                        if final_answer.lower().strip() == answer_m.lower().strip() or math_verify_reward(final_answer,answer_m) or normalize_punctuation(final_answer).lower().strip() == normalize_punctuation(answer_m).lower().strip():
                            return 1.0

                        else:
                            final_answer=re.sub(r'text\{[^}]*\}', '', predict_matches[-1])
                            final_answer=final_answer.replace('\\','').replace(" ","").rstrip(",")
                            if math_verify_reward(final_answer,answer) :
                                return 1.0
                            else:
                                return 0.0
                    try:
                        return 1.0 if math_verify_reward(predict,answer) else 0.0
                    except:
                        print("mllm_normal_score打分出现问题，检查mllm_normal_score打分")
                        return 0.0



def mllm_choice_score(question,predict,answer):
    base_predict = predict
    
    if 'boxed{' not in answer:
        answer = '\\boxed{'+answer+'}'

    if answer == predict:
        return 1.0
    else:
        answer = answer.replace('option A.', 'A')
        answer = answer.replace('option B.', 'B')
        answer = answer.replace('option C.', 'C')
        answer = answer.replace('option D.', 'D')
        pattern = r"\\?boxed{(.+)}"
        answer_matches = re.findall(pattern, answer)
        if not answer_matches:
            return 0.0

        else:
            predict_matches = re.findall(pattern, predict)
            if not predict_matches:
                return 0.0
            else:
                final_answer = predict_matches[-1]
                answer_m=answer_matches[-1]

                answer_m = str(answer_m).replace("degrees","").replace("cm^2","").replace("cm^2^","").replace("厘","").replace("平方","").replace("cm^{2}","").replace("√","\sqrt").replace("<n>","\n").replace('cm²', '').replace('米', '').replace('°', '').replace('cm', '').replace('()', '').replace('$','').replace('±','\\pm').replace('μg/m³', '').replace('。', '').replace('元', '').strip()
                final_answer = str(final_answer).replace("degrees","").replace("cm^2","").replace("cm^2^","").replace("厘","").replace("平方","").replace("cm^{2}","").replace("√","\sqrt").replace("<n>","\n").replace('cm²', '').replace('米', '').replace('°', '').replace('cm', '').replace('()', '').replace('$','').replace('±','\\pm').replace('μg/m³', '').replace('。', '').replace('元', '').replace('^\\circ', '').replace('\\circ', '').replace('circ', '').strip()

                final_answer = remove_units(final_answer)
                answer_m = remove_units(answer_m)
                
                choice_extracted = extract_options_advanced(question)

                answer_letters = ['A','B','C','D','E','F','G','H']
                if len(predict_matches[-1]) > 1:
                    second_letter_final_answer = predict_matches[-1][1]
                else:
                    second_letter_final_answer = ''


                if len(answer_matches[-1]) > 1:
                    second_letter_answer_m = answer_matches[-1][1]
                else:
                    second_letter_answer_m = ''


                if len(choice_extracted) > 1:
                    extracted_prediction, extracted_letter = normalize_extracted_answer_mllm(predict_matches[-1], choice_extracted) 
                    extracted_answer_m, extracted_letter_answer_m = normalize_extracted_answer_mllm(answer_matches[-1], choice_extracted)
                else:
                    extracted_prediction, extracted_letter = "", ""
                    extracted_answer_m, extracted_letter_answer_m = "", ""

                if predict_matches[-1]==answer_matches[-1] or normalize_punctuation(predict_matches[-1]).lower().strip() == normalize_punctuation(answer_matches[-1]).lower().strip():
                    return 1.0

                elif predict_matches[-1][0] in answer_letters and answer_matches[-1][0] in answer_letters and is_not_alphanumeric_manual(second_letter_final_answer) and is_not_alphanumeric_manual(second_letter_final_answer) and predict_matches[-1][0].lower().strip() == answer_matches[-1][0].lower().strip():
                    return 1.0
                elif safe_equal(extracted_prediction.lower().strip(), answer_matches[-1].lower().strip()) or safe_equal(extracted_letter.lower().strip(), answer_matches[-1].lower().strip()) or remove_units(answer_matches[-1]).lower().strip() == remove_units(extracted_prediction).lower().strip() or normalize_punctuation(answer_matches[-1]).lower().strip() == normalize_punctuation(extracted_prediction).lower().strip():
                    return 1.0

                elif safe_equal(extracted_answer_m.lower().strip(), extracted_prediction.lower().strip()) and len(choice_extracted) > 1:
                    return 1.0
                elif final_answer.rstrip('.').lower().strip() == answer_m.rstrip('.').lower().strip() or final_answer.rstrip('^').lower().strip() == answer_m.rstrip('^').lower().strip() or final_answer.replace(' ','').lower().strip() == answer_m.replace(' ','').lower().strip() or final_answer.rstrip('m.').lower().strip() == answer_m.rstrip('m.').lower().strip():
                    return 1.0
                elif answer_matches[-1].lower().strip().rstrip('.') == 'yes' or answer_matches[-1].lower().strip().rstrip('.') == 'no':
                    single_answer = re.split(r'[,.]', predict_matches[-1])[0]
                    if single_answer.lower().strip().rstrip('.') == answer_matches[-1].lower().strip().rstrip('.') :
                        return 1.0
                elif final_answer.endswith('.0.'):
                    final_answer = final_answer[:-3]
                    if final_answer == answer_m:
                        return 1.0

                elif answer_m.endswith('.0.'):
                    answer_m = answer_m[:-3]
                    if final_answer == answer_m:
                        return 1.0
                else:
                    if 'text{' in final_answer:
                        final_answer = extract_and_combine_latex_ordered(final_answer)
                        final_answer = final_answer.replace('\\','')
                        final_answer = final_answer.replace(' ','')
                        answer_m = answer_m.replace(' ','')
                        if final_answer.lower().strip() == answer_m.lower().strip() or math_verify_reward(final_answer,answer_m) or normalize_punctuation(final_answer).lower().strip() == normalize_punctuation(answer_m).lower().strip():
                            return 1.0

                        else:
                            final_answer=re.sub(r'text\{[^}]*\}', '', predict_matches[-1])
                            final_answer=final_answer.replace('\\','').replace(" ","").rstrip(",")
                            if math_verify_reward(final_answer,answer) :
                                return 1.0
                            else:
                                return 0.0

                    try:
                        return 1.0 if math_verify_reward(predict,answer) else 0.0
                    except:
                        print("mllm_choice_score打分出现问题，检查mllm_choice_score打分")
                        return 0.0



def mllm_score_math_verify_eval_distance(question,predict,answer):
    def replace_english_numbers_with_arabic(input_string):
        # 定义英文数字到阿拉伯数字的映射
        number_mapping = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
            'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
            'eighteen': '18', 'nineteen': '19', 'twenty': '20', 'thirty': '30',
            'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
            'eighty': '80', 'ninety': '90',
        }

        # 将输入字符串分割为单词
        words = input_string.split()

        # 替换英文数字为阿拉伯数字
        for i, word in enumerate(words):
            if word.lower() in number_mapping:
                words[i] = number_mapping[word.lower()]

        # 重新组合字符串
        output_string = ' '.join(words)
        return output_string

    def extract_response(text):
        matches = re.findall(r'boxed\{([^}]+)\}', text)

        text =  matches[-1] if matches else text
        extract_answers = text
        extract_answers = extract_answers.replace('\\\\text{', '')
        extract_answers = extract_answers.replace('\\\text{', '')
        extract_answers = extract_answers.replace('\\text{', '')
        extract_answers = extract_answers.replace('\text{', '')
        extract_answers = extract_answers.replace('text{', '')
        extract_answers = extract_answers.strip()
        return extract_answers

    answer = replace_english_numbers_with_arabic(answer)
    predict = replace_english_numbers_with_arabic(extract_response(predict))
    if answer == predict:
        return 1.0
    else:
        final_answer = predict.replace('<|end_of_sentence|>','').replace('<eod>','')
        answer_matches = answer
        if 'text{' in final_answer:
            matches = re.findall(r'\\text{((?:[^{}]|\{[^{}]*\})*)}', final_answer)
            answer_extracted_from_text = ''
            for match_text in matches:
                answer_extracted_from_text += match_text
            final_answer = answer_extracted_from_text
        final_answer = final_answer.replace('\\','')

        if final_answer.lower().strip() == answer_matches.lower().strip():
            return 1.0
        else:
            if not bool(re.search('[a-zA-Z]', final_answer)) and ':' not in final_answer:
                return 1.0 if math_verify_reward(predict,answer) else 0.0
            else:
                dist = levenshtein_distance_mllm(final_answer.lower().strip(),answer_matches.lower().strip())
                length = max( len(answer_matches.upper()), len(final_answer.upper()) )
                dist_norm = 0.0 if length == 0 else float(dist) / float(length)

                question_result = 1 - dist_norm
                score = question_result*2-1
                if score <= 0:
                    return 0.0
                else:
                    return 1.0



def mllm_score_math_verify_grounding(question,predict,answer):
    def parse_bbox(bbox_str):
        """解析边界框字符串，返回(s1, s2, s3, s4)格式的元组，失败返回None"""
        # 提取[s...]中的内容
        match = re.search(r'\[s(\d+),s(\d+),s(\d+),s(\d+)\]', bbox_str)
        if not match:
            return None

        try:
            # 转换为整数
            return (
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
                int(match.group(4))
            )
        except ValueError:
            return None

    def calculate_iou(bbox1, bbox2):
        """计算两个边界框的交并比(IOU)"""
        s1, t1, s2, t2 = bbox1
        s1_g, t1_g, s2_g, t2_g = bbox2

        # 计算交集区域
        inter_s = max(s1, s1_g)
        inter_t = max(t1, t1_g)
        inter_e_s = min(s2, s2_g)
        inter_e_t = min(t2, t2_g)

        # 计算交集面积
        inter_area = max(0, inter_e_s - inter_s) * max(0, inter_e_t - inter_t)

        # 计算每个边界框的面积
        area1 = (s2 - s1) * (t2 - t1)
        area2 = (s2_g - s1_g) * (t2_g - t1_g)

        # 计算并集面积
        union_area = area1 + area2 - inter_area

        # 避免除以零
        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def validate_format(answer):
        """验证答案格式是否为'<obj></obj><box></box>'"""
        # 使用正则表达式验证完整格式
        pattern = r'^<obj>(.*?)</obj><box>(.*?)</box>$'
        return re.match(pattern, answer) is not None

    def validate_obj_only_format(answer):
        """验证答案格式是否仅包含'<obj></obj>'部分"""
        pattern = r'^<obj>(.*?)</obj>$'
        return re.match(pattern, answer) is not None


    reference = str(answer).strip() # 格式：'<obj>woman brushing teeth in stripe shirt</obj><box>[s805,s440,s999,s999]</box>'
    candidate = str(predict).replace('<|end_of_sentence|>','').replace('<eod>','').strip()

    if reference == candidate:
        return 1.0
    else:
        # 检查参考答案是否仅包含<obj>部分
        if validate_obj_only_format(reference):
            # 提取参考物体名称
            reference_obj = re.search(r'<obj>(.*?)</obj>', reference).group(1)

            # 检查候选答案格式是否正确，两种格式都不正确则返回0
            if not validate_format(candidate) and not validate_obj_only_format(candidate):
                return 0

            # 提取候选物体名称
            candidate_obj = re.search(r'<obj>(.*?)</obj>', candidate).group(1)

            # 对比物体名称，通过字符串距离计算
            dist = levenshtein_distance_mllm(candidate_obj,reference_obj)
            length = max( len(candidate_obj.upper()), len(reference_obj.upper()) )
            dist_norm = 0.0 if length == 0 else float(dist) / float(length)

            question_result = 1 - dist_norm
            score = question_result*2-1


            return 1.0 if score > 0 else 0

        # 检查格式是否正确
        if not validate_format(reference):
            return 0
        if not validate_format(candidate):
            return 0

        # 提取物体和边界框部分
        candidate_obj = re.search(r'<obj>(.*?)</obj>', candidate).group(1)
        candidate_bbox_str = re.search(r'<box>(.*?)</box>', candidate).group(1)

        reference_obj = re.search(r'<obj>(.*?)</obj>', reference).group(1)
        reference_bbox_str = re.search(r'<box>(.*?)</box>', reference).group(1)

        # 解析参考边界框
        reference_bbox = parse_bbox(reference_bbox_str)
        if not reference_bbox:
            return 0

        # 解析候选边界框
        candidate_bbox = parse_bbox(candidate_bbox_str)
        if not candidate_bbox:
            return 0

        # 检查物体名称是否一致
        if candidate_obj != reference_obj:
            return 0

        # 计算IOU并检查阈值
        iou = calculate_iou(candidate_bbox, reference_bbox)

        # iou条件满足
        if iou > 0.7:
            return 1.0

        # iou条件不满足
        return 0
