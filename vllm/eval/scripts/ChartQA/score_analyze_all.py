import json
import re
import argparse
from word2number import w2n
from typing import Optional

def extract_boxed_content_regex(text):
    """
    使用正则表达式提取\\boxed{}中的内容（简化版本，可能不处理复杂嵌套）

    参数:
        text (str): 输入文本

    返回:
        list: 包含所有\\boxed{}中内容的列表
    """
    # 这个正则表达式可以处理简单的嵌套情况
    pattern = r'boxed\{((?:[^{}]|\{[^{}]*\})*)\}'
    matches = re.findall(pattern, text)
    if 'text{' in text:
        pattern_test = r'text\{((?:[^{}]|\{[^{}]*\})*)\}'
        match_text = re.findall(pattern_test, matches[0])
        return match_text
    return matches


def normalize_text(text):
    """将文本转换为小写并去除多余空格"""
    if text is None:
        return ""
    return str(text).lower().strip()

def word_to_number(text):
    """使用word2number将英文数字单词转换为数字"""
    try:
        # 首先尝试直接转换整个文本
        number = w2n.word_to_num(text)
        return str(number)
    except ValueError:
        # 如果整个文本转换失败，尝试提取数字单词部分
        pass

    # 提取可能的数字单词部分
    words = re.findall(r'[a-z\-]+', text.lower())
    if words:
        try:
            # 尝试将单词组合转换为数字
            combined_text = ' '.join(words)
            number = w2n.word_to_num(combined_text)
            return str(number)
        except ValueError:
            pass

    return None

def extract_number(text):
    """从文本中提取数字（包括小数和分数）"""
    if text is None:
        return None

    text = str(text).lower().strip()

    # 去除常见的单位符号
    text = re.sub(r'[\%\\\/\$\€\£\¥\°\'\"]', '', text)
    text = re.sub(r'\b(minutes|mins|hours|hrs|days|weeks|months|years|percent|percentage|°|degrees)\b', '', text)

    # 处理分数形式
    fraction_match = re.search(r'(\d+)\s*\/\s*(\d+)', text)
    if fraction_match:
        numerator = int(fraction_match.group(1))
        denominator = int(fraction_match.group(2))
        if denominator != 0:
            return str(numerator / denominator)

    # 提取数字（包括小数）
    number_match = re.search(r'[-+]?\d*\.?\d+', text)
    if number_match:
        return number_match.group(0)

    # 尝试将英文数字转换为数字
    word_number = word_to_number(text)
    if word_number:
        return word_number

    return None

def strict_match(extraction, annotation):
    """严格模式匹配"""
    ext_norm = normalize_text(extraction)
    ann_norm = normalize_text(annotation)
    return ext_norm == ann_norm

def lenient_match(extraction, annotation):
    """宽松模式匹配"""
    # 首先尝试严格匹配
    if strict_match(extraction, annotation):
        return True
    
    
    if 'answer:' in extraction:
        extraction = extraction.split('answer:')[1].replace('<|end_of_sentence|>','')
        if strict_match(extraction, annotation):
            return True
        extraction = extraction.split('.')[0]
        if strict_match(extraction, annotation):
            return True


    if 'answer is' in extraction:
        extraction = extraction.split('answer is')[1].replace('<|end_of_sentence|>','')
        if strict_match(extraction, annotation):
            return True
        extraction = extraction.split('.')[0]
        if strict_match(extraction, annotation):
            return True
 
    if 'Answer:' in extraction:
        extraction = extraction.split('Answer:')[1].replace('<|end_of_sentence|>','')
        if strict_match(extraction, annotation):
            return True
        extraction = extraction.split('.')[0]
        if strict_match(extraction, annotation):
            return True

    if 'Answer is' in extraction:
        print(extraction)
        extraction = extraction.split('Answer is')[1].replace('<|end_of_sentence|>','')
        if strict_match(extraction, annotation):
            return True
        print(extraction)
        extraction = extraction.split('.')[0]
        if strict_match(extraction, annotation):
            return True

    if annotation =='A' or annotation =='B' or annotation =='C' or annotation =='D':
        if 'A.' in extraction or 'B.' in extraction or 'C.' in extraction or 'D.' in extraction:
            extract_letter = extraction.split('.')[0]
            if strict_match(extract_letter,annotation):
                return True

    ext_num = extraction
    ann_num = annotation

    if extraction is not None and annotation is not None:
        try:
            # 转换为浮点数进行比较（处理小数精度问题）
            ext_float = float(ext_num)
            ann_float = float(ann_num)

            # 允许小的浮点误差
            return abs(ext_float - ann_float) < 0.001
        except (ValueError, TypeError):
            pass

    return False

def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    "Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct."

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    
    prediction_extract = extract_number(prediction)
    target_extract = extract_number(target)
    
    if prediction_extract is not None and target_extract is not None:

        prediction_extract_float = _to_float(prediction_extract)
        target_extract_float = _to_float(target_extract)
    else:
        prediction_extract_float = prediction_extract
        target_extract_float = target_extract

    if prediction_float is not None and target_float is not None and target_float != 0:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    elif prediction_extract_float is not None and target_extract_float is not None and target_extract_float!= 0:
        relative_change = abs(prediction_extract_float - target_extract_float) / abs(target_extract_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()

def evaluate_relaxed_accuracy(data):
    """Evaluate accuracy with relaxed correctness for chartqa"""
    scores = []
    for key, entry in data.items():
        #extract = extract_boxed_content_regex(entry.get('response', ''))
        #if len(extract) >= 1:
        #    extraction = extract_boxed_content_regex(entry.get('response', ''))[0]
        #else:
        extraction = entry.get('extraction', '')
        annotation = entry.get('annotation', '')
        extraction = extraction.replace('<eod>','')
        extraction = extraction.replace('<|end_of_sentence|>','')
        extraction = extraction.replace('}', '')
 
        if extraction is None or annotation is None:
            scores.append(0)
            continue
            
        # For chartqa, annotation might be a list or single value
        if isinstance(annotation, list):
            # Get the maximum score across all annotations
            score = max([
                int(relaxed_correctness(str(ann), str(extraction)))
                for ann in annotation
            ])
            scores.append(score)
        else:
            score = int(relaxed_correctness(str(annotation), str(extraction)))
            scores.append(score)
        print(extraction, '|||', annotation)
        print(score)
        print('============================')
    print(sum(scores), len(scores)) 
    return sum(scores) / len(scores) * 100 if scores else 0

def relaxed_correctness_strict(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    "Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct."

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)


    if prediction_float is not None and target_float is not None and target_float != 0:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()

def evaluate_relaxed_accuracy_strict(data):
    """Evaluate accuracy with relaxed correctness for chartqa"""
    scores = []
    for key, entry in data.items():
        extraction = entry.get('extraction', '')
        annotation = entry.get('annotation', '')

        if extraction is None or annotation is None:
            scores.append(0)
            continue

        # For chartqa, annotation might be a list or single value
        if isinstance(annotation, list):
            # Get the maximum score across all annotations
            score = max([
                int(relaxed_correctness_strict(str(ann), str(extraction)))
                for ann in annotation
            ])
            scores.append(score)
        else:
            score = int(relaxed_correctness_strict(str(annotation), str(extraction)))
            scores.append(score)

    return sum(scores) / len(scores) * 100 if scores else 0




def levenshtein_distance(s1, s2):
    """Calculate Levenshtein distance between two strings"""
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

def evaluate_docvqa_accuracy(data):
    """Evaluate accuracy for docvqa using Levenshtein distance"""
    scores = []
    
    for key, entry in data.items():
        extraction = entry.get('extraction', '')
        annotation = entry.get('annotation', '')
        extraction = extraction.replace('<eod>','') 
        extraction = extraction.replace('<|end_of_sentence|>','')
        extraction = extraction.replace('}', '')
        print(annotation, '|||||', extraction)
        if extraction is None or annotation is None:
            scores.append(0)
            continue
            
        # For docvqa, annotation should be a list
        if not isinstance(annotation, list):
            annotation = [annotation]
            
        values = []
        for answer in annotation:
            # Preprocess both the answers - gt and prediction
            gt_answer = ' '.join(str(answer).strip().lower().split())
            det_answer = ' '.join(str(extraction).strip().lower().split())
            
            dist = levenshtein_distance(gt_answer, det_answer)
            length = max(len(str(answer)), len(str(extraction)))
            values.append(0.0 if length == 0 else float(dist) / float(length))
        
        question_result = 1 - min(values) if values else 0
        
        if question_result < 0.5:
            question_result = 0
        print(question_result)
        scores.append(question_result)
    
    return sum(scores) / len(scores) * 100 if scores else 0

def calculate_accuracy(json_file_path):
    """计算准确率并保存错误数据"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return 0, 0, 0, {}

    # Check if data is a list with one element containing the actual data
    if isinstance(data, list) and len(data) == 1:
        data = data[0]
    
    strict_correct = 0
    lenient_correct = 0
    special_accuracy = 0
    total = len(data)

    # 存储错误数据
    wrong_data = {
        'strict_wrong_lenient_correct': {},  # 严格错误但宽松正确
        'both_wrong': {}                     # 两种模式都错误
    }

    print(f"Processing {total} entries...")
    print("-" * 80)

    # Calculate standard accuracies
    for key, entry in data.items():
        extraction = entry.get('extraction', '')
        annotation = entry.get('annotation', '')
        print(extraction, '|||', annotation)
        # 跳过无效条目
        if extraction is None or annotation is None:
            continue

        strict_match_result = strict_match(extraction, annotation)
        lenient_match_result = lenient_match(extraction, annotation)

        strict_correct += 1 if strict_match_result else 0
        lenient_correct += 1 if lenient_match_result else 0

        # 记录错误数据
        if not strict_match_result:
            error_entry = {
                'extraction': extraction,
                'annotation': annotation,
                'strict_match': strict_match_result,
                'lenient_match': lenient_match_result,
                'extracted_number': extract_number(extraction),
                'annotation_number': extract_number(annotation)
            }

            if lenient_match_result:
                wrong_data['strict_wrong_lenient_correct'][key] = error_entry
            else:
                wrong_data['both_wrong'][key] = error_entry
        print(strict_match_result, lenient_match_result)
        print('='*50)

    # Calculate special accuracy based on file path
    if 'chartqa' in json_file_path.lower() or 'human' in json_file_path.lower() or 'augmented' in json_file_path.lower():
        print("Using chartqa relaxed accuracy evaluation...")
        special_accuracy_strict = evaluate_relaxed_accuracy_strict(data)
        special_accuracy_loose = evaluate_relaxed_accuracy(data)
    elif 'docvqa' in json_file_path.lower() or '-doc-' in json_file_path.lower():
        print("Using docvqa Levenshtein distance evaluation...")
        special_accuracy = evaluate_docvqa_accuracy(data)
        special_accuracy_strict = 0
        special_accuracy_loose = 0
    else:
        special_accuracy = lenient_correct / total * 100 if total > 0 else 0
        special_accuracy_strict = 0
        special_accuracy_loose = 0

    # 保存错误数据到新JSON文件
    #wrong_file_path = json_file_path.replace('.json', '_wrong.json')
    #try:
    #    with open(wrong_file_path, 'w', encoding='utf-8') as f:
    #        json.dump(wrong_data, f, indent=2, ensure_ascii=False)
    #    print(f"\nWrong data saved to: {wrong_file_path}")
    #except Exception as e:
    #    print(f"Error saving wrong data: {e}")

    strict_accuracy = strict_correct / total * 100 if total > 0 else 0
    lenient_accuracy = lenient_correct / total * 100 if total > 0 else 0

    # 打印错误统计
    print(f"\nError Statistics:")
    print(f"Strict wrong but lenient correct: {len(wrong_data['strict_wrong_lenient_correct'])}")
    print(f"Both strict and lenient wrong: {len(wrong_data['both_wrong'])}")

    return strict_accuracy, lenient_accuracy, special_accuracy_strict, special_accuracy_loose , special_accuracy, wrong_data

def main():
    parser = argparse.ArgumentParser(description='Calculate accuracy scores for extraction vs annotation')
    parser.add_argument('--output-file', required=True, help='Path to the JSON file')
    args = parser.parse_args()

    strict_acc, lenient_acc, special_acc_strict, special_acc_loose, special_acc, wrong_data = calculate_accuracy(args.output_file)

    print("\n" + "="*80)
    print("RESULTS:")
    score_dict = {}
    if 'chartqa' in args.output_file.lower() or 'human' in args.output_file.lower() or 'augmented' in args.output_file.lower():
        print(f"ChartQA Loose Accuracy: {special_acc_loose:.2f}%")
        print(f"ChartQA Strict Accuracy: {special_acc_strict:.2f}%")
        score_dict['ChartQA Loose Accuracy'] = special_acc_loose
        score_dict['ChartQA Strict Strict'] = special_acc_strict
    elif 'docvqa' in args.output_file.lower() or '-doc-' in args.output_file.lower():
        print(f"DocVQA Levenshtein Accuracy: {special_acc:.2f}%")
        score_dict['DocVQA'] = special_acc
    else:
        print("RESULTS:")
        print(f"Strict Accuracy: {strict_acc:.2f}%")
        print(f"Loose Accuracy: {lenient_acc:.2f}%")
        score_dict['Loose Accuracy'] = lenient_acc
        score_dict['Strict Strict'] = strict_acc
        #print(f"Special Accuracy: {special_acc:.2f}%")
    
    print("="*80)
    score_file = args.output_file.replace('.json','_score.json')
    with open(score_file, 'w', encoding='utf-8') as f:
        json.dump(score_dict, f, ensure_ascii=False, indent=4)  # indent 参数用于美化输出
    '''# 打印一些错误示例
    if wrong_data['strict_wrong_lenient_correct']:
        print("\nExamples of strict wrong but lenient correct:")
        for i, (key, entry) in enumerate(list(wrong_data['strict_wrong_lenient_correct'].items())[:3]):
            print(f"  {i+1}. Key: {key}")
            print(f"     Extraction: '{entry['extraction']}'")
            print(f"     Annotation: '{entry['annotation']}'")

    if wrong_data['both_wrong']:
        print("\nExamples of both strict and lenient wrong:")
        for i, (key, entry) in enumerate(list(wrong_data['both_wrong'].items())[:3]):
            print(f"  {i+1}. Key: {key}")
            print(f"     Extraction: '{entry['extraction']}'")
            print(f"     Annotation: '{entry['annotation']}'")'''

if __name__ == "__main__":
    main()
