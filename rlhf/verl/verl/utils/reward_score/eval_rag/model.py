import time
from verl.utils.reward_score.eval_rag.qa_em import *
from verl.utils.reward_score.eval_llm.eval import get_repetition_penalty_reward, language_consistency_reward, format_reward

def rag_reward_model(reward_input,args):
    reward_method = reward_input['reward_method']

    if reward_method == "rag_ifeval":
        acc_score = compute_score_ifeval(reward_input)
    elif reward_method == "rag_frames" or reward_method == "rag_alpaca" or reward_method == "rag_simpleqa":
        acc_score = None
    elif reward_method == "rag_chatqa":
        acc_score = compute_score_rag(reward_input)
    elif reward_method == "rag_mrag":
        acc_score = compute_score_mrag(reward_input)
    elif reward_method == "rag_mrag_bench":
        acc_score = compute_mrag_bench(reward_input)
    elif reward_method == "rag_summary":
        acc_score = compute_score_summary(reward_input)
    elif reward_method == "rag_mmtab":
        acc_score = compute_score_mmtab(reward_input)
    else:
        raise NotImplementedError("该方法尚未实现")
    
    repetition_penalty=0.0
    format_score = 0.0
    lang_consistency_ratio = language_consistency_reward(reward_input, lowest_score=-1.0) 
    reward_score = acc_score
    #必须包含reward_score, acc_score
    result = {'reward_score':reward_score,
              'acc_score':acc_score,
              'format':format_score,
              'lang_consistency_ratio':lang_consistency_ratio,
              'repetition_penalty':repetition_penalty
              }


    return result


