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
# from . import gsm8k, math, prime_math, prime_code

from verl.utils.reward_score.eval_mllm.model import mllm_reward_model
from verl.utils.reward_score.eval_llm.model import llm_reward_model
from verl.utils.reward_score.eval_general.model import general_reward_model
from verl.utils.reward_score.eval_rag.model import rag_reward_model
from verl.utils.reward_score.eval_sql.model import sql_reward_model

import json
import ast

MLLM_METHOD_FLAG = ['mllm_normal', 'mllm_choice', 'mllm_knowledge', 'mllm_grounding']
LLM_METHOD_FLAG = ['llm_math', 'llm_code', 'llm_choice', 'llm_tool_reward']
GENERAL_METHOD_FLAG = ['llm_general', 'mllm_general1', 'mllm_general2']
RAG_METHOD_FLAG = ['rag_ifeval', 'rag_frames', 'rag_alpaca', 'rag_simpleqa', 'rag_chatqa', 'rag_mrag', 'rag_mrag_bench', "rag_summary","rag_mmtab"]
SQL_METHOD_FLAG = ['sql_bird', 'sql_spider']
PROCESS_REWARD_METHOD = ['llm_math','llm_choice','mllm_normal','mllm_choice']

def calculate_overlong_reward(dct, overlong_buffer_len, max_resp_len, response_length, overlong_penalty_factor=1.0, overlong_buffer_enable=True):

    if not overlong_buffer_enable:
        dct["overlong_reward"] = 0.0
        dct["overlong"] = False
        return dct
    
    if response_length > max_resp_len:
        dct["overlong_reward"] = -1.0
        dct["overlong"] = True
    else:
        expected_len = max_resp_len - overlong_buffer_len
        exceed_len = response_length - expected_len
        overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
        dct["overlong_reward"] = overlong_reward
        dct["overlong"] = (overlong_reward<0.0)
        
    return dct


def compute_reward_score(
    reward_input,
    args,
    overlong_buffer_cfg=None,
    max_resp_len=None,
    extra_info=None,
):
    """Compute the score for a given solution based on the data source.

    Args:
        reward_input(dict):
            prompt_str(question),
            solution_str(response),
            response_ids,
            reward_method,
            vllm_api,

        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.  

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    reward_method = reward_input["reward_method"]
    enable_thinking_flag = reward_input["enable_thinking_flag"]
    valid_response_length=len(reward_input["response_id"])
    overlong_buffer_enable = True if overlong_buffer_cfg and overlong_buffer_cfg.enable and (reward_method not in PROCESS_REWARD_METHOD or not enable_thinking_flag) else False  
    
    # For MLLM
    if reward_method in MLLM_METHOD_FLAG:
        reward_input["answer"]=try_parse_dict(reward_input["answer"])
        res = mllm_reward_model(reward_input,args)
        if reward_method in ['mllm_knowledge', 'mllm_grounding']:
            res = calculate_overlong_reward(res, overlong_buffer_len=2048, max_resp_len=4096, response_length=valid_response_length, overlong_buffer_enable=overlong_buffer_enable)
        elif 'mllm' in reward_method and not enable_thinking_flag:
            res = calculate_overlong_reward(res, overlong_buffer_len=1024, max_resp_len=4096, response_length=valid_response_length, overlong_buffer_enable=overlong_buffer_enable)

    # For LLM
    elif reward_method in LLM_METHOD_FLAG:
        res = llm_reward_model(reward_input,args)
        
        if reward_method == "llm_tool_reward":
            res = calculate_overlong_reward(res, overlong_buffer_len=2048, max_resp_len=4096, response_length=valid_response_length, overlong_buffer_enable=overlong_buffer_enable)
        elif reward_method == "llm_code" and enable_thinking_flag:
            res = calculate_overlong_reward(res, overlong_buffer_len=4096, max_resp_len=16384, response_length=valid_response_length, overlong_buffer_enable=overlong_buffer_enable)
        elif reward_method == "llm_code" and not enable_thinking_flag:
            res = calculate_overlong_reward(res, overlong_buffer_len=4096, max_resp_len=8192, response_length=valid_response_length, overlong_buffer_enable=overlong_buffer_enable)
        elif not enable_thinking_flag:
            res = calculate_overlong_reward(res, overlong_buffer_len=2048, max_resp_len=4096, response_length=valid_response_length, overlong_buffer_enable=overlong_buffer_enable)
    # For General
    elif reward_method in GENERAL_METHOD_FLAG:
        res = general_reward_model(reward_input,args)
        if "mllm" in reward_method:
            res = calculate_overlong_reward(res, overlong_buffer_len=2048, max_resp_len=4096, response_length=valid_response_length, overlong_buffer_enable=overlong_buffer_enable)
        else:
            res = calculate_overlong_reward(res, overlong_buffer_len=3072, max_resp_len=4096, response_length=valid_response_length, overlong_buffer_enable=overlong_buffer_enable)
    # For Rag
    elif reward_method in RAG_METHOD_FLAG:
        reward_input["answer"]=try_parse_dict(reward_input["answer"])
        res = rag_reward_model(reward_input,args)
        res = calculate_overlong_reward(res, overlong_buffer_len=2048, max_resp_len=4096, response_length=valid_response_length, overlong_buffer_enable=overlong_buffer_enable)
    # For SQL
    elif reward_method in SQL_METHOD_FLAG:
        res = sql_reward_model(reward_input,args)
        if enable_thinking_flag:
            overlong_buffer_len = 4096
            # max_resp_len = 20480
            max_resp_len = 16384
        else:
            overlong_buffer_len = 2048
            max_resp_len = 4096
        res = calculate_overlong_reward(res, overlong_buffer_len, max_resp_len, valid_response_length, overlong_buffer_enable)
    else:
        raise NotImplementedError("该方法尚未实现")
    
    if "overlong" not in res:
        res["overlong_reward"] = 0.0
        res["overlong"] = False

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])


def try_parse_dict(input_str):
    try:
        parsed = json.loads(input_str)
        if isinstance(parsed, dict):
            return parsed
        else:
            return input_str
    except:
        try:
            parsed = eval(input_str)
            if isinstance(parsed, dict):
                return parsed
            else:
                return input_str
        except:
            return input_str
