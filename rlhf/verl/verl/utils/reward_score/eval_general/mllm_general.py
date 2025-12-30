from openai import OpenAI
import base64
from io import BytesIO
import re

PROMPT_TEMPLATE="""
Given a question and a reference image, please analyze in detail the provided answers (Answer 1, Answer 2, ... Answer n). Evaluate them based on the following three core dimensions:
1. Semantic accuracy: How well the answer reflects the visual content of the image
2. Correctness: Whether the answer is logically and factually correct
3. Clarity: Whether the answer is clearly and fluently expressed
You may also consider additional dimensions if you find them relevant (e.g., reasoning ability, attention to detail, multimodal grounding, etc.). For each dimension, provide a score from 1 to 10 for each answer, and briefly explain your reasoning. Then, for each answer:
a) Compute total raw score by summing all dimension scores
b) Calculate max_possible = 10 * (number of dimensions used)
c) Compute normalized_score = (total_raw_score / max_possible) * 10
(Round normalized_score to one decimal place)
Enclose your full reasoning within <think> and </think> tags. Then, in the <answer> tag, output exactly the normalized scores for all answers in the format:
\\boxed{{X,X,...,X}} where X represents the normalized scores. No other text is allowed in the <answer> section.
Example format:
<think>
Evaluation:
1. Semantic accuracy: Answer 1 (8/10) - ...; Answer 2 (6/10) - ...; Answer 3 (9/10) - ...
2. Correctness: Answer 1 (7/10) - ...; Answer 2 (8/10) - ...; Answer 3 (9/10) - ...
3. Clarity: Answer 1 (9/10) - ...; Answer 2 (7/10) - ...; Answer 3 (8/10) - ...
[Additional dimensions if any]: Answer 1 (6/10) - ...; Answer 2 (7/10) - ...; Answer 3 (9/10) - ...
Total raw scores:
Answer 1: 8+7+9+6=30
Answer 2: 6+8+7+7=28
Answer 3: 9+9+8+9=35
max_possible = 10*4 = 40
Normalized scores:
Answer 1: (30/40)*10 = 7.5
Answer 2: (28/40)*10 = 7.0
Answer 3: (35/40)*10 = 8.8
</think>
<answer>\\boxed{{7.5,7.0,8.8}}</answer>

**Note: In the example above, scores and the final answer are placeholders meant only to demonstrate the format. Your actual evaluation should be based on the quality of the given answers.**

Your task is provided as follows:
Question: [{query}]
{answers}
"""
PROMPT_TEMPLATE_CN="""
给出一个问题和一个参考图像，请详细分析提供的答案（答案1，答案2，……答案n)。根据以下三个核心维度进行评估：
1. 语义准确性：答案反映图像视觉内容的程度
2. 正确性：答案在逻辑上和事实上是否正确
3. 清晰性：是否清晰流畅地表达了答案
你也可以考虑额外的维度，如果你发现它们是相关的（例如，推理能力，对细节的关注，多模态定位等）。对于每个维度，为每个答案提供1到10分的分数，并简要解释你的理由。然后，对于每个答案：
a)将所有维度分数相加，计算总原始分数total_raw_score
b)计算总分max_possible = 10 *（使用的维度数）
c)计算归一化分数normalized_score = (total_raw_score / max_possible) * 10
（将normalized_score四舍五入到小数点后一位）
在<think>和</think>标签中包含完整的推理。然后，在<answer>和</answer>标签中，准确输出所有答案的归一化分数，格式为：
\\boxed{{X,X,...,X}}，其中X表示归一化分数。<answer>部分中不允许有其他文本。
示例格式:
<think>
评价:
1. 语义准确性：答案1(8/10)-…；答案2(6/10)-…答案3(9/10)-…
2. 正确性：答案1(7/10)-…；答案2(8/10)-…答案3(9/10)-…
3. 清晰性：答案1(9/10)-…答案2(7/10)-…；答案3(8/10)-…
[附加维度（如有）]：答案1(6/10)-…；答案2(7/10)-…；答案3(9/10)-…
总原始分数：
答案1:8+7+9+6=30
答案2:6+8+7+7=28
答案3:9+9+8+9=35
Max_possible = 10*4 = 40
归一化分数:
答案1:(30/40)*10 = 7.5
答案2:(28/40)*10 = 7.0
答案3:(35/40)*10 = 8.8
</think>
<answer>\\boxed{{7.5,7.0,8.8}}</answer>

**注意：在上面的例子中，分数和最终答案是占位符，仅用于演示格式。你的实际评价应该基于所给答案的质量**

你的任务如下：
问题: [{query}]
{answers}
"""

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

def format_responses(response_list):
    formatted_string = ""
    for i, response in enumerate(response_list, start=1):
        response=extract_solution_nosearch(response)
        formatted_string += f"Answer {i}: [{response}]\n"
    return formatted_string

def format_responses_cn(response_list):
    formatted_string = ""
    for i, response in enumerate(response_list, start=1):
        response=extract_solution_nosearch(response)
        formatted_string += f"答案{i}: [{response}]\n"
    return formatted_string


def extract_composite_scores(composite_score_str):

    # 使用正则表达式查找所有匹配的boxed内容
    matches = re.findall(r"\\boxed\{([^\}]+)\}", composite_score_str)
    if not matches:
        print("No boxed scores found")
        return None

    # 选择最后一个匹配的内容
    scores_str = matches[-1]
    scores = scores_str.split(',')

    # Check if each score is a valid float
    try:
        scores = [float(score.strip()) for score in scores]
    except Exception as e:
        # raise ValueError("Scores within boxed are not valid float")
        print("Scores within boxed are not valid float")
        return None

    return scores


def encode_image_base64(image):
    with BytesIO() as buffer:
        if image.mode in ('P', 'RGBA', 'LA', '1', 'L', 'CMYK', 'I', 'F'):
            image = image.convert('RGB')
        image.save(buffer, "JPEG")
        data = buffer.getvalue()
        encoded_image = base64.b64encode(data)
        encoded_image_text = encoded_image.decode('utf-8')
        image_url =f'data:image/jpeg;base64,{encoded_image_text}'
        return image_url


def mllm_general_reward(reward_input, api, timeout, max_tokens):

    try:
        instruction = extract_solution_nosearch(reward_input['question'].replace('<n>','\n'))
        response_list = reward_input['response_list']
        responses = format_responses(response_list)
        if reward_input["language"]=="cn":
            prompt=PROMPT_TEMPLATE_CN.format(query=instruction,answers=responses)
        else:
            prompt=PROMPT_TEMPLATE.format(query=instruction,answers=responses)

        image = reward_input["multi_modal_data"]["image"][0] ##1张图片/是否有多张图片的情况
        image_url=encode_image_base64(image)


        input_text = prompt.strip().replace('\r','')

        openai_api_key = "EMPTY"
        openai_api_base = f"http://{api}/v1"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            timeout=timeout,
        )
        models=client.models.list()
        judge_model = models.data[0].id

        response = client.chat.completions.create(
            model=judge_model,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url":image_url
                            },
                    },
                    {"type": "text", "text": input_text},
                ],
            },
            ],
            extra_body= {
                "include_stop_str_in_output": True,
                "skip_special_tokens": False
            },
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,

        )
        text = response.choices[0].message.content

        # extracting scores as a list over here
        scores = extract_composite_scores(text)
        if scores is None or len(scores) != len(response_list):
            return [-1.0] * len(response_list), None
        # projecting socres into [-1,1]
        scores = [(2/9)*x - (11/9) for x in scores]
        return scores, text

    except Exception as e:
        print(e,"error1>>>>>>>>")
        return [-100.0] * len(response_list), None


