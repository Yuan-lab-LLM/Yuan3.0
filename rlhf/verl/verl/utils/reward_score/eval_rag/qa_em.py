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

import re
import string
import random
from verl.utils.reward_score.eval_rag.instruction_following_eval.evaluation_lib import *
from openai import OpenAI
from collections import Counter
from sacrebleu.metrics import BLEU
import json
import requests
from rouge_score import rouge_scorer
import time
from datetime import datetime

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


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
    if "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]

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



def get_f1_score(pred_items, gold_items):
    common = Counter(gold_items) & Counter(pred_items)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def f1_check(prediction, golden_answers):
    if not isinstance(golden_answers, list):
        golden_answers = [str(golden_answers)]
    normalized_prediction = normalize_answer(prediction).split()
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer).split()
        f1 = get_f1_score(normalized_prediction, golden_answer)
        score = f1 if f1 > score else score
    return score 

def get_inputexample(instructions):
    import json
    if isinstance(instructions['kwargs'], str):
        kwargs = json.loads(instructions['kwargs'])
    elif isinstance(instructions['kwargs'], list):
        kwargs = instructions['kwargs']
    return InputExample(key=instructions['key'],
            instruction_id_list=instructions["instruction_id_list"],
            prompt=instructions["prompt"],
            kwargs=kwargs)

def ifeval_check(prediction, instructions):
    inp = get_inputexample(instructions)
    return eval_instruction_following_strict(inp, prediction)

def compute_score_ifeval(reward_input):
    def evaluate(ifeval, f1):
        if f1 > 0.8:
            result = 1 if ifeval else -0.4
        elif 0.5 < f1 < 0.8:
            result = 0 if ifeval else -0.6
        else:
            result = -0.4 if ifeval else -1
        return result
    solution_str = reward_input['response']
    ground_truth = reward_input['answer']
    answer = extract_solution_nosearch(solution_str=solution_str)
    if answer is None:
        return float(-1.0)
    else:
        ground, instructions = ground_truth["ground_truth"], ground_truth["instructions"]
        ifeval = ifeval_check(answer, instructions)
        f1 = f1_check(answer, ground)
        return float(evaluate(ifeval, f1))

def judge_by_model(reward_input, args):
    """
    The scoring function for frames, alpaca, simpleqa, using LLM evaluate the answer.
    """
    def evaluate_prompt_frames(question, answer, ground):
        evaluation_prompt = f"""===Task===
    I need your help in evaluating an answer provided by an LLM against a ground truth answer. Your task is to determine if the ground truth answer is present in the LLM's response. Please analyze the provided data and make a decision.
    ===Instructions===
    1. Carefully compare the "Predicted Answer" with the "Ground Truth Answer".
    2. Consider the substance of the answers - look for equivalent information or correct answers.
    Do not focus on exact wording unless the exact wording is crucial to the meaning.
    3. Your final decision should be based on whether the meaning and the vital facts of the "Ground Truth Answer" are present in the "Predicted Answer:"
    ===Input Data===
    - Question: {question}
    - Predicted Answer: {answer}
    - Ground Truth Answer: {ground}
    ===Output Format===
    Provide your final evaluation in the following format:
    Just return the letters "\\box{{1}}" or "\\box{{-1}}", with no text around it.
    Please proceed with the evaluation."""
        prompt = [{"role": "user", "content": evaluation_prompt}]
        return prompt
    
    def evaluate_prompt_alpacaeval(question, output_1, output_2):
        evaluation_prompt=f"""===Task===
    You are a highly efficient assistant, who evaluates and rank large language models (LLMs) based on the quality of their responses to given prompts. This process will create a leaderboard reflecting the most accurate and human-preferred answers.
    ===Instructions===
    I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding responses. Your task is to assess these responses, ranking the models in order of preference from a human perspective. 
    ===Input Data===
    - Prompt: {question}
    - Response_1: {output_1}
    - Response_2: {output_2}
    Response_1 and Response_2 are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.
    ===Output Format===
    Evaluate and rank the models based on the quality and relevance of their outputs. Provide your final evaluation in the following format:
    If Response_1 is better than Response_2, Just return the letters "\\box{{1}}", with no text around it; if Response_2 is better than Response_1, return the letters "\\box{{-1}}", with no text around it; if Response_1 is same as Response_2, return the letters "\\box{{0}}", with no text around it.
    Now please proceed with the evaluation.
        """
        prompt = [{"role": "user", "content": evaluation_prompt}]
        return prompt
    
    def evaluate_prompt_simpleqa(question, predicted_answer, target):
        evaluation_prompt = f"""===Task===
    Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
    First, I will give examples of each grade, and then you will grade a new example.

    ===Instructions===
    1. The following are examples of CORRECT predicted answers.
    ```
    Question: What are the names of Barack Obama's children?
    Gold target: Malia Obama and Sasha Obama
    Predicted answer 1: sasha and malia obama
    Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
    Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
    ```
    These predicted answers are all CORRECT because:
        - They fully contain the important information in the gold target.
        - They do not contain any information that contradicts the gold target.
        - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
        - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.

    2. The following are examples of INCORRECT predicted answers.
    ```
    Question: What are the names of Barack Obama's children?
    Gold target: Malia and Sasha
    Predicted answer 1: Malia.
    Predicted answer 2: Malia, Sasha, and Susan.
    Predicted answer 3: Barack Obama does not have any children.
    Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
    Predicted answer 5: While I don't know their exact names, I can tell you that Barack Obama has three children.
    Predicted answer 6: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
    Predicted answer 7: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
    ```
    These predicted answers are all INCORRECT because:
        - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.

    3. The following are examples of NOT_ATTEMPTED predicted answers.
    ```
    Question: What are the names of Barack Obama's children?
    Gold target: Malia and Sasha
    Predicted answer 1: I don't know.
    Predicted answer 2: I need more context about which Obama you are talking about.
    Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
    Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
    ```
    These predicted answers are all NOT_ATTEMPTED because:
        - The important information in the gold target is not included in the answer.
        - No statements in the answer contradict the gold target.

    ===Input Data===
    Question: {question}
    Gold target: {target}
    Predicted answer: {predicted_answer}

    ===Output Format===
    Grade the predicted answer of this new question as one of:
    1: CORRECT
    0: INCORRECT
    -1: NOT_ATTEMPTED
    Just return the letters "\\box{{1}}", "\\box{{0}}", or "\\box{{-1}}", with no text around it.
    """
        prompt = [{"role": "user", "content": evaluation_prompt}]
        return prompt
    
    openai_api_key = "EMPTY"  # 请替换为您的 API 密钥
    openai_api_base = f"http://{reward_input['reward_vllm_api']}/v1"  # 本地服务地址
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base, timeout=args.timeout)
    models = client.models.list()
    judge_model = models.data[0].id
    
    try:
        reward_input["answer"]=json.loads(reward_input["answer"])
    except:
        try:
            reward_input["answer"]=eval(reward_input["answer"])
        except:
            reward_input["answer"]=reward_input["answer"]


    if reward_input['reward_method'] == "rag_frames":
        messages = evaluate_prompt_frames(extract_solution_nosearch(reward_input['answer']['question']), extract_solution_nosearch(reward_input['response']), reward_input['answer']['ground_truth'])
    elif reward_input['reward_method'] == "rag_alpaca":
        messages = evaluate_prompt_alpacaeval(extract_solution_nosearch(reward_input['answer']['question']), extract_solution_nosearch(reward_input['response']), reward_input['answer']['ground_truth'])
    elif reward_input['reward_method'] == "rag_simpleqa":
        messages = evaluate_prompt_simpleqa(extract_solution_nosearch(reward_input['answer']['question']), extract_solution_nosearch(reward_input['response']), reward_input['answer']['ground_truth'])
    else:
        raise NotImplementedError
    max_tokens = args.max_tokens if 'max_tokens' in args else 32
    try:
        completion = client.chat.completions.create(
            model = judge_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7, ##0.6
            top_p=0.8, ##0.95
            extra_body={
            "include_stop_str_in_output": True,
            "skip_special_tokens": False,
            "top_k":20, ##
            "repetition_penalty": 1.05 ##
        },

        )
        score = completion.choices[0].message.content
        result = re.search(r"\{(-?\d+(?:\.\d+)?)\}", score)
        if result:
            return float(result.group(1))
        return float(-1)
    except Exception as e:
        print("========Model Eval Error========")
        print(f"Question: {reward_input['answer']['question']}")
        print(f"GroundTruth: {reward_input['answer']['ground_truth']}")
        print(f"LLM Repsone: {extract_solution_nosearch(reward_input['response'])}")
        return float(-1)

def compute_score_model(reward_input, args):
    score = judge_by_model(reward_input, args)
    return score

def compute_score_rag(reward_input):
    '''
    alpha_dict = {
        "doc2dial": 0.3,
        "quac": 0.4,
        "qrecc": 0.3,
        "coqa": 0.7,
        "doqa": 0.4,
        "cfqa": 0.6,
        "sqa": 0.7,
        "tcqa": 0.5,
        "hdial": 0.5,
        "inscit": 0.2,
    }
    '''
    alpha_dict = {
        "doc2dial": 0.1,
        "quac": 0.1,
        "qrecc": 0.1,
        "coqa": 0.1,
        "doqa": 0.1,
        "cfqa": 0.6,
        "sqa": 0.1,
        "tcqa": 0.5,
        "hdial": 0.1,
        "inscit": 0.2,
    }
    alpha = alpha_dict[reward_input['answer']['data_source'].lower()]
    answer = extract_solution_nosearch(solution_str=reward_input['response'])
    ground = reward_input['answer']['ground_truth']
    f1 = f1_check(answer, ground)
    if f1 >= alpha:
        return 2*float(f1)-1
    else:
        return float(-1)


def get_f1_score_mdata(reward_input):
    def parse_answer(text):
        text = text.strip().replace("</think>", "").replace("<think>", "")
        answer_idx = text.rfind("Answer:")
        if answer_idx == -1:
            return text
        text = text[answer_idx+len("Answer:"):].strip()
        return text
    def parse_response(text):
        answer_idx = text.rfind("<|Assistant|>")
        if answer_idx != -1:
            text = text[answer_idx+len("<|Assistant|>"):]
        else:
            text = text.strip()

        text = extract_solution_nosearch(text)
        
        match = re.search(r'\\boxed\{([^}]+)\}', text)
        text =  match.group(1) if match else text
        text = parse_answer(text)
        if len(text) == 1 or len(text) == 0:
            return text

        if text[0] in ['A', 'B', 'C', 'D'] and (text[1] == '.' or text[1] ==":"):
            return text[0]
        return text
    
    re_art = re.compile(r'\b(a|an|the)\b')
    re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')
    def normalize_answer(s):
        """
        Lower text and remove punctuation, articles and extra whitespace.
        """
        s = s.lower()
        s = re_punc.sub(' ', s)
        s = re_art.sub(' ', s)
        s = ' '.join(s.split())
        return s
    def rag_pair_en(pred_items, gold_items):
        common = Counter(gold_items) & Counter(pred_items)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_items)
        recall = 1.0 * num_same / len(gold_items)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
    guess = parse_response(reward_input['response']).strip()

    if not isinstance(reward_input['answer'], str):
        if isinstance(reward_input['answer'], (dict, list)):
            # 如果是字典或列表，转换为JSON字符串
            reward_input['answer'] = json.dumps(reward_input['answer'])
        else:
            # 其他类型直接转字符串
            reward_input['answer'] = str(reward_input['answer'])
    answer = parse_answer(reward_input['answer']).strip()
    
    if guess == "":
        return -1.0

    if answer in ['A', 'B', 'C', 'D']:
        score = 1.0 if answer.lower() == guess.lower() else -1.0
        return score
    g_tokens = normalize_answer(guess).split()
    a_tokens = normalize_answer(answer).split()
    score = rag_pair_en(g_tokens, a_tokens)
    if score > 0.8:
        return 1.0
    elif score < 0.5:
        return -1.0
    else:
        return 0

def compute_score_mrag(reward_input):
    score = get_f1_score_mdata(reward_input)
    return float(score)


def parser_choice(solution_str):
    solution_str = solution_str.replace("(", " ").replace(")", " ")
    if len(solution_str) == 1:
        return solution_str
    matches = re.findall(r'[^A-D]*([A-D])(?=\s*[:.\s]|$)', solution_str)
    if matches:
        last_match = matches[-1]
        return last_match[0]
    return solution_str

def compute_mrag_bench(reward_input):
    solution_str = reward_input['response'].strip()
    solution_str = parser_choice(solution_str)
    gt_choice = reward_input['answer']
    gt_choice = parser_choice(gt_choice)
    if solution_str.lower() == gt_choice.lower():
        return 1
    return -1


def extract_tqa_answer_list(model_output):
    """
    Extract the answer list from the model output to compute accuracy
    """
    model_output = model_output.replace('\n',' ')
    ret = re.match('.*({\s*[\"\']answer[\"\']\:.*}).*',model_output)
    if ret is not None:
        answer_str = ret.group(1)
        try:
            try:
                answer_item = json.loads(answer_str)
            except:
                answer_str = re.sub('[\"\']+',"\"",answer_str)
                answer_item = eval(answer_str)
            predicted_answer = answer_item['answer']
            if type(predicted_answer) != list and type(predicted_answer) == str:
                predicted_answer = [predicted_answer]
            elif type(predicted_answer) != list and type(predicted_answer) in [float,int]:
                predicted_answer = [str(predicted_answer)]
            else:
                pass
            _ = set(predicted_answer)
        # The answer is considered to be wrong if we can not extract answer list from the json str
        except Exception:
            predicted_answer = []
        return predicted_answer
    else:
        return []

def evaluate_tqa_questions(answer, ground_truth):
    answer_list = ground_truth
    predicted_answer_list = extract_tqa_answer_list(answer)
    if predicted_answer_list == []: return -1
    return 1 if set(predicted_answer_list) == set(answer_list) else 0

def evaluate_text_generation_questions(answer, ground_truth):
    gold_answer = ground_truth
    bleu = BLEU()
    bleu_score = bleu.corpus_score([answer], [[gold_answer]])
    return 2*bleu_score.score/100 - 1

def compute_score_mmtab(reward_input):
    data_source = reward_input['answer']['data_source']
    solution_str = reward_input['response']
    ground_truth = reward_input['answer']['answer_list']
    if reward_input['imagepath']:
        if data_source == "fetaqa":
            ground_truth = ground_truth
        else:
            ground_truth = extract_tqa_answer_list(ground_truth)
    else:
        if data_source == "fetaqa":
            ground_truth = ground_truth[0]
        else:
            ground_truth = ground_truth

    answer = extract_solution_nosearch(solution_str)
    if data_source == "fetaqa":
        return evaluate_text_generation_questions(answer, ground_truth)
    else:
        return evaluate_tqa_questions(answer, ground_truth)



def compute_score_summary(reward_input):
    def compute_summary_rouge(answer, summary):
        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
            score = scorer.score(summary, answer)
            rouge1_score=score['rouge1'].fmeasure
            rouge2_score=score['rouge2'].fmeasure
        except:
            rouge1_score=0
            rouge2_score=0

        return {"rouge1": rouge1_score, "rouge2": rouge2_score}


    def compute_summary_bert(answer: str, summary: str, url: str):
        payload = {"answer": answer, "summary": summary}
        try:
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
            else:
                print(f"failed to request, status code: {response.status_code}")
                print(f"Details: {response.text}")
                with open("./error_log.txt", "a", encoding="utf-8") as f:
                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: failed to request, status code: {response.status_code}, details: {response.text}")
                result = {"error": f"请求失败，状态码: {response.status_code}", "details": response.text}
        except Exception as e:
            print(f"failed to request: {e}")
            with open("./error_log.txt", "a", encoding="utf-8") as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: failed to request: {str(e)}")
            result = {"error": f"发生错误: {str(e)}"}
        return result
    solution_str = reward_input['response']
    ground_truth = reward_input['answer']

    url = f"http://{reward_input['reward_vllm_api']}/compute-bert-score"
    weights = [0.25, 0.25, 0.5]
    summary = ground_truth
    answer = extract_solution_nosearch(solution_str=solution_str)
    
    rouge_res = compute_summary_rouge(answer, summary)
    rouge1, rouge2 = rouge_res['rouge1'], rouge_res['rouge2']
    bert_f1 = compute_summary_bert(answer, summary, url)
    if 'error' in bert_f1:
        final_score = 0.5*rouge1 + 0.5*rouge2
    else:
        final_score = weights[0]*rouge1 + weights[1]*rouge2 + weights[2]*bert_f1['bert_score']
    return 2*final_score - 1

