import time
import numpy as np
from verl.utils.reward_score.eval_llm.eval import llm_score_math_verify, llm_score_code, apps_score_code, llm_choice_qwen_vllm, get_repetition_penalty_reward, language_consistency_reward, format_reward, llm_score_choice_verify
from verl.utils.reward_score.eval_llm.llm_tool_reward_bfcl import llm_tool_reward_bfcl
from verl.utils.reward_score.eval_llm.eval_process import process_scoring

LLM_PROCESS_METHOD_FLAG = ['llm_math','llm_choice','mllm_normal','mllm_choice']

def llm_reward_model_acc(reward_input,args):
    
    reward_method=reward_input["reward_method"]
    if reward_method == 'llm_code':
        iscore_ = llm_score_code(reward_input["response"],reward_input["answer"])
        iscore = 1.0 if iscore_ == 1.0 else -1.0
    elif reward_method == 'llm_math':
        iscore = llm_score_math_verify(reward_input["response"],reward_input["answer"])
    elif reward_method == 'llm_choice':
        iscore = llm_score_choice_verify(reward_input["response"],reward_input["answer"])

    elif reward_method == "llm_tool_reward":
        if reward_input["data_source"]=="bfcl":
            iscore = llm_tool_reward_bfcl(reward_input)
        elif reward_input["data_source"]=="json":
            pass ##取消此类
    else:
        raise NotImplementedError("该方法尚未实现")

    return iscore

def llm_reward_model(reward_input,args):
    acc_score = llm_reward_model_acc(reward_input,args)

    if reward_input['reward_method'] in LLM_PROCESS_METHOD_FLAG and reward_input['enable_thinking_flag']:
        repetition_penalty = get_repetition_penalty_reward(reward_input, lowest_score=-1.0)
        lang_consistency_ratio = language_consistency_reward(reward_input, lowest_score=-1.0)
        format_score = format_reward(reward_input, lowest_score=-1.0)

        if acc_score == 1.0 and repetition_penalty >= 0.0 and format_score >= 0.0 and lang_consistency_ratio >= 0.0:
            reward_score = 1.0
        else:
            reward_score = -1.0
    else:
        reward_score = acc_score
        repetition_penalty = 0.0
        lang_consistency_ratio = language_consistency_reward(reward_input, lowest_score=-1.0)
        format_score = 0.0


    result = {'reward_score':reward_score,
              'acc_score':acc_score,
              'repetition_penalty':repetition_penalty,
              'format':format_score,
              'lang_consistency_ratio':lang_consistency_ratio
              }

    return result

def llm_process_reward_process_scoring(group_input):

    v_exp = 2
    v_max = 10
    v_counts = []
    acc_scores = []
    res_lst = []

    if group_input[0][0]['reward_method'] not in LLM_PROCESS_METHOD_FLAG or not group_input[0][0]['enable_thinking_flag']:
        for _input in group_input:
            output_dict, res = _input
            res['process_stuff'] = None
            res['process_scores'] = None
            res['process_vlimits'] = None
            res_lst.append(res)
        return res_lst

    for _input in group_input:
        output_dict, res = _input
        if res['acc_score'] > 0.0:
            try:
                v = output_dict['sft_think_text_final'].count('</verify>')
                v_counts.append(v)
            except:
                pass
        acc_scores.append(res['acc_score'])
    if v_counts:
        v_max = np.max(v_counts)
        v_exp = np.min(v_counts)
    pass_rate = np.mean([0.0 if x < 0.0 else x for x in acc_scores])
    if pass_rate >= 0.5:
        v_exp = 2
        v_max = 10

    for _input in group_input:
        output_dict, res = _input

        try:
            reward_score, process_scores = process_scoring(output_dict, v_exp, v_max)
        except Exception as e:
            reward_score = None
            process_scores = None
            print(f"generate process_line exception: {e}")

        if reward_score is not None:
            res['reward_score'] = reward_score
        # for debug
        res['process_stuff'] = output_dict
        res['process_scores'] = process_scores
        res['process_vlimits'] = (v_exp, v_max)
        res_lst.append(res)

    return res_lst

