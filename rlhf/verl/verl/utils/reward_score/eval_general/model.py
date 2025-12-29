import time
from verl.utils.reward_score.eval_llm.eval import llm_score_math_verify, llm_score_code, llm_general_reward, apps_score_code, llm_choice_qwen_vllm, get_repetition_penalty_reward, language_consistency_reward, format_reward
from verl.utils.reward_score.eval_general.mllm_general import mllm_general_reward


def general_reward_model_acc(reward_input,args):

    reward_method = reward_input['reward_method']
    start_time = time.time()
    
    if reward_input.get('reward_flag', None) == False:
        acc_score = [-42]*len(reward_input['response_list'])
        judgement = None
        iscore = (acc_score,judgement)
        return iscore

    if reward_method == 'llm_general':
        iscore = llm_general_reward(reward_input, reward_input['reward_vllm_api'], args.timeout, args.max_tokens)
    elif reward_method == "mllm_general2":
        iscore = mllm_general_reward(reward_input, reward_input['reward_vllm_api'], args.timeout, args.max_tokens)
    else:
        raise NotImplementedError("该方法尚未实现")
    end_time = time.time()
    elapsed_time = end_time - start_time

    return iscore


def general_reward_model(reward_input,args):

    acc_score= [-42]*len(reward_input['response_list'])
    judgement = None

    ## TODO:当前验证数据不适合做格式检查
    lang_consistency_ratio = language_consistency_reward(reward_input, lowest_score=-1.0)

    repetition_penalty = 0.0
    format_score = 0.0
    
    reward_score=42.0

    result = {'reward_score':reward_score,
              'acc_score':acc_score,
              'llm_general_judgement':judgement,
              'repetition_penalty':repetition_penalty,
              'format':format_score,
              'lang_consistency_ratio':lang_consistency_ratio
              }

    return result

