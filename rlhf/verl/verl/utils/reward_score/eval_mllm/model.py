import time
import json
import re,os
import argparse
import requests

from verl.utils.reward_score.eval_mllm.eval import mllm_choice_score, mllm_normal_score, mllm_score_math_verify_eval_distance, mllm_score_math_verify_grounding

def mllm_reward_model_acc(reward_input):

    reward_method = reward_input['reward_method']
    start_time = time.time()

    if reward_method == 'mllm_normal':
        iscore = mllm_normal_score(reward_input['question'], reward_input['response'], reward_input['answer'])
    elif reward_method == 'mllm_choice':
        iscore = mllm_choice_score(reward_input['question'], reward_input['response'], reward_input['answer']) 
    elif reward_method =="mllm_knowledge":
        iscore = mllm_score_math_verify_eval_distance(reward_input['question'], reward_input['response'], reward_input['answer'])
    elif reward_method == "mllm_grounding":
        iscore = mllm_score_math_verify_grounding(reward_input['question'], reward_input['response'], reward_input['answer'])
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    if iscore != 1.0:
        iscore = -1.0
    return iscore
    
def mllm_reward_model(reward_input,args):
    acc_score = mllm_reward_model_acc(reward_input)
    process_score, outputs, extracted_ans = 1.0, 1.0, 1.0 
    format_score = 1.0
    lang_consistency_ratio = 1.0
    repetition_penalty = 1.0
    if acc_score == 1.0 and format_score == 1.0 and process_score == 1.0:
        reward_score = 1.0
    else:
        reward_score = -1.0

    result = {'reward_score':reward_score,
              'acc_score':acc_score,
              'repetition_penalty':repetition_penalty,
              'format':format_score,
              'lang_consistency_ratio':lang_consistency_ratio,
              }

    return result


