from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import time
import json
import os
import argparse
from process_math_boxed import get_boxed_answer


Gpu_nums = int(os.environ.get("GPU_NUMS", 2))
Max_tokens = int(os.environ.get("MAX_TOKENS", 32768))
Batch_size = int(os.environ.get("BATCH_SIZE", 240))
Prompt_flag = os.environ.get("PROMPT", False)

def batch_infer(input_lst, output_prefix):
    tokenizer = AutoTokenizer.from_pretrained("/your/model_path/deepseek_r1")
    sampling_params = SamplingParams(temperature=0.6, max_tokens=Max_tokens, repetition_penalty=1.0)
    llm = LLM(model="/your/model_path/deepseek_r1", tensor_parallel_size=Gpu_nums, gpu_memory_utilization=0.9)

    start  = time.time()
    i = 0
    llm_inputs = []
    record = []
    input_text_list = []
    solution_list = []
    ans_true_lst = []
    predict_list = []
    per_batch_size = Batch_size

    output_file_true = output_prefix + "true.jsonl"
    output_file_false = output_prefix + "false.jsonl"
    
    # 只有在数学问题的时候才使用
    if Prompt_flag:
        prompt_template = ""
    else:
        prompt_template = ""

    print("------------------------------------------------")
    print(prompt_template)
    ans_true_num = 0
    # Sandra is offering free tutoring for school subjects. She tutors 5 students every 2 hours and tutors for 4 hours a day. If she tutors for 80% of the days from June 1st to July 15th, and takes the rest of the days off, how many students does she tutor during this period?<sep>To solve this problem.<answer>360
    lines_len = len(input_lst)
    for dic in input_lst:
        try:
            i = i + 1
            print(i)
            predict = dic['predict'] # 提取的boxed中的答案
            question = dic["question"]
            ans_true = dic['ans_true']
            solution = dic['answer']
            answer = predict if predict else "None"
            if not solution.endswith('<eod>'):
                answer = "None"

            # answer = predict if predict else solution
            input_text = "以下两个字段的内容分别是一个问题的标准答案和某位学生的解题答案，请从数学的意义上评判学生最终答案的数值部分是否与标准答案是相等或者等价的，可不用在意单位的差异。如果'是'请直接输出:boxed{是}，否则直接输出boxed{否}。\n\n标准答案：" + ans_true + "\n学生答案："+ answer
            print(input_text)
            messages = [
                {"role": "user", "content": input_text}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            record.append(answer)
            input_text_list.append(question)
            llm_inputs.append(text)
            solution_list.append(solution)
            predict_list.append(predict)
            ans_true_lst.append(ans_true)
            # if len(llm_inputs) % per_batch_size != 0 and idx < lines_len -1:
            if len(llm_inputs) < per_batch_size and i < lines_len:
                print(f'current line number = {i}')
                continue

            # generate outputs
            outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
            for output, info, intext, solution_text, predict_text, ans_true_text in zip(outputs, record, input_text_list, solution_list, predict_list, ans_true_lst):
                dic = {}
                generated_text = output.outputs[0].text.replace('\n','<n>')
                dic['question'] = intext
                dic['answer'] = solution_text
                dic['predict'] = predict_text
                dic['ans_true'] = ans_true_text
                dic['judge'] = generated_text
                judge_text = generated_text.split('</think>')[-1]
                if "boxed{" in judge_text:
                    judge_text = "boxed{" + judge_text.split("boxed{")[-1].strip()
                # if "boxed{是}" in generated_text:
                if "boxed{是}" in judge_text and info!="None":
                    ans_flag = True
                    ans_true_num += 1
                else:
                    ans_flag = False
                output_file = output_file_true if ans_flag else output_file_false
                json_str = json.dumps(dic, ensure_ascii=False)
                with open(output_file,'a',encoding='utf-8') as wr:
                    wr.writelines(json_str+'\n')
            llm_inputs = []
            record = []
            input_text_list = []
            solution_list = []
            predict_list = []
            ans_true_lst = []

        except Exception as e:
            print(e,"error1>>>>>>>>")

    end = time.time()
    print(f"ans_true_num: {ans_true_num}")
    print(f"accuracy: {round((ans_true_num * 100)/len(input_lst), 4)}%")
    print(f"Save to {output_prefix} successfully.")
    print("time:", end - start)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="", help='input file or folder')
    parser.add_argument("--origin_path", type=str, default="", help='origin file')
    parser.add_argument("--output_path", type=str, default="", help='output folder')
    args = parser.parse_args()
    assert args.input_path and args.output_path, "input_path and output_path must be specified"
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    content_generate, output_prefix = get_boxed_answer(args.input_path, args.origin_path, args.output_path)
    batch_infer(content_generate, output_prefix)


if __name__ == '__main__':
    main()


# GPU_NUMS=2 MAX_TOKENS=4096 BATCH_SIZE=480 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=6,7 python judge_with_vllm_model_math.py --input_path /your/math-en --output_path /your/output/path --origin_path /your/math125_test_qa.txt


