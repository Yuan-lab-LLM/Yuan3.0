import ast
import json
from math import e
import math

def single(tool_modeloutput,tool_answer):
    #  'func_name': func_name,
    #  'args': args,
    #  'keywords': keywords
    if   tool_modeloutput['func_name']==tool_answer['func_name']:
        accuracy_score=0.7
        if tool_modeloutput['keywords']==tool_answer['keywords']:
            accuracy_score=1.0
        return accuracy_score
    else:
        return 0.2


def _parse_tool_calls(tool_str: str) -> list | None:
    try:
        # 解析字符串为AST语法树
        parsed_ast = ast.parse(tool_str, mode='eval')
        # 检查解析结果是否为列表表达式
        if not isinstance(parsed_ast.body, ast.List):
            return None
        # 检查列表元素是否均为函数调用（工具调用本质是函数调用）
        list_elements = parsed_ast.body.elts
        
        # 检查列表元素是否均为函数调用
        for item in list_elements:
            if not isinstance(item, ast.Call):
                return None
        # 返回解析后的工具调用列表（保留函数名、参数名、参数值）
        tool_calls = []
        for call in list_elements:
            func_name_parts = []
            node = call.func
            while isinstance(node, ast.Attribute):
                func_name_parts.append(node.attr)
                node = node.value
            if isinstance(node, ast.Name):
                func_name_parts.append(node.id)
            else:
                return None
            func_name = ".".join(reversed(func_name_parts))  # 函数名（如ln、ls）
            args = []  # 位置参数（本题示例无位置参数，预留兼容）
            keywords = {}  # 关键字参数（如target='current_config'）
            # 处理位置参数
            for arg in call.args:
                if isinstance(arg, ast.Constant):
                    args.append(arg.value)
            # 处理关键字参数
            for kw in call.keywords:
                kw_arg_name = kw.arg  # 参数名（如target）
                if isinstance(kw.value, ast.Constant):
                    kw_arg_value = kw.value.value  # 参数值（如current_config）
                    keywords[kw_arg_name] = kw_arg_value
            tool_calls.append({
                'func_name': func_name,
                'args': args,
                'keywords': keywords
            })
        if tool_calls == []:
            tool_calls = None
            print(f"tool_calls 为空")
        return tool_calls
    except (SyntaxError, AttributeError, TypeError) as e:
        # 解析失败（语法错误、属性不存在、类型错误），判定为无工具调用
        return None
    except Exception as e:
        # 处理其他所有异常
        print(f"其他异常：{e}")
        return None

def llm_tool_reward_bfcl(reward_input):
    """
    根据模型输出(modeloutput)与标准答案(answer)的工具调用情况，返回[-1, 1]之间的评分
    打分规则：
modeloutput：有工具
1）格式；0.2 判断标准：能否被ast解析
2）准确性；1 判断标准：是否与标准答案匹配
3）答案不正确：-1 判断标准：不与标准答案匹配
modeloutput没有工具；
1）生成内容没有工具调用；1
2）生成了工具调用；-1；
    """
    modeloutput=reward_input['response'].replace("<eod>","")
    answer=reward_input['answer']
    modeloutput=modeloutput.replace('<1st_answer>[[','[')
    modeloutput=modeloutput.replace('<1st_answer>','')
    modeloutput=modeloutput.replace(']]<tool_calls><|end_of_sentence|>',']')
    modeloutput=modeloutput.replace('<tool_calls><|end_of_sentence|>','')
    modeloutput=modeloutput.replace(']]</tool_calls><|end_of_sentence|>',']')
    modeloutput=modeloutput.replace('</tool_calls><|end_of_sentence|>','')
    modeloutput=modeloutput.replace('<tool_calls>','')
    modeloutput=modeloutput.replace('<tool_calls>[[','[')    
    modeloutput=modeloutput.replace('<|endoftext|>','')    
    modeloutput=modeloutput.replace('<n>','\n').strip()
    answer=answer.replace('<n>','\n').strip()
    
    if('</think>' in answer):
        answer=answer.split('</think>')[-1]
    if(modeloutput=='[]'):
        modeloutput='error'
    # 解析标准答案和模型输出的工具调用
    answer_tools = _parse_tool_calls(answer)  
    model_tools = _parse_tool_calls(modeloutput) 
    # 步骤2：判断模型输出是否存在工具调用，按规则打分
    if model_tools is not None and answer_tools is not None:
        # 情况A：模型输出有工具调用
        # 计算格式分：能解析（model_tools非None）则得0.5
        final=0
        # 计算准确性分：与标准答案工具调用完全匹配得1，否则得-1
        # 无论是单轮还是多轮本质上逻辑是一样的。tool1 --output      tool2 -----answer_tools
        for j in range(0,len(answer_tools)):
        
            score=0
            if(j==len(model_tools)):
                break
            for i in range(0,len(answer_tools)):
                print(j,i,len(answer_tools))
                score=max(single(model_tools[j],answer_tools[i]),score)
            final+=score
        final=final/len(answer_tools)
        if(len(model_tools)>(len(answer_tools))):
            exponent = 1 - (len(model_tools) / len(answer_tools))
            final = final * math.exp(exponent)  # math.exp(x) 等价于 e^x
        return final
    else:
        # 情况B：模型输出无工具调用
        # 若标准答案也无工具调用，得1；若标准答案有工具调用，得-1
        if answer_tools is None and model_tools is None:
            final_score = 1.0
        else:
            final_score = -1.0
        return final_score
