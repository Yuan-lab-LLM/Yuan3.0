import time
from verl.utils.reward_score.eval_sql.spider_eval_rl import spider_eval_one_json
from verl.utils.reward_score.eval_sql.bird_eval_rl import bird_eval_one_json
from verl.utils.reward_score.eval_llm.eval import get_repetition_penalty_reward, language_consistency_reward, format_reward, format_reward_shortsql


def sql_reward_model_acc(reward_input):

    reward_method = reward_input['reward_method']
    if reward_method == 'sql_spider':
        iscore_ = spider_eval_one_json(reward_input)
    elif reward_method == "sql_bird":
        iscore_ = bird_eval_one_json(reward_input)
    else:
        raise NotImplementedError("该方法尚未实现")

    iscore = 1.0 if iscore_ == 1.0 else -1.0
    return iscore


def sql_reward_model(reward_input,args):

    acc_score = sql_reward_model_acc(reward_input)
    enable_thinking_flag = reward_input["enable_thinking_flag"]
    if not enable_thinking_flag:
        format_score = format_reward_shortsql(reward_input, lowest_score=-1.0)
    else:
        format_score = 1.0 

    repetition_penalty = 0.0
    lang_consistency_ratio = language_consistency_reward(reward_input, lowest_score=-1.0)
    if repetition_penalty >= 0.0 and format_score >= 0.0:
        reward_score = acc_score
    else:
        reward_score = -1.0


    result = {'reward_score':reward_score,
              'acc_score':acc_score,
              'repetition_penalty':repetition_penalty,
              'format':format_score,
              'lang_consistency_ratio':lang_consistency_ratio
              }

    return result
