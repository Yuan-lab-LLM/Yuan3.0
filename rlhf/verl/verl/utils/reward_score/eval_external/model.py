from verl.utils.reward_score.eval_llm.eval import llm_choice_qwen_vllm
from verl.utils.reward_score.eval_general.model import general_reward_model_acc 
from verl.utils.reward_score.eval_rag.qa_em import compute_score_model
from verl.utils.reward_score.eval_llm.eval_process import process_line
import time

PROCESS_REWARD_METHOD = ['llm_math','llm_choice','mllm_normal','mllm_choice']

def llm_process(data,args,q_ord):
    reward_input=data[0]
    score=data[1]
    reward_method=reward_input["reward_method"]
    enable_thinking_flag = reward_input["enable_thinking_flag"]
    start_time=time.time()

    if reward_method=="llm_general" or reward_method=="mllm_general2":
        iscore=general_reward_model_acc(reward_input, args)
        score["acc_score"],score["llm_general_judgement"]=iscore

    elif reward_method == "rag_frames" or reward_method == "rag_alpaca" or reward_method == "rag_simpleqa":
        iscore = compute_score_model(reward_input, args)
        score["acc_score"]=iscore
        score["reward_score"] =iscore
    
    elif reward_method in PROCESS_REWARD_METHOD and enable_thinking_flag:
        _input={**reward_input, **score}
        result=process_line(q_ord, _input)
        data[0]=result

    else:
        pass
    
    end_time=time.time()
    elapsed_time=end_time-start_time
    data[0]["external_elapsed_time"]=elapsed_time
    data[0]["start_time"] = start_time
    data[0]["end_time"] = end_time

    return data

    
