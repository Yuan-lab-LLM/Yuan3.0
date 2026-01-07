from PIL import Image
from io import BytesIO
import base64
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from multiprocessing import Pool
from functools import partial
from pathlib import Path
import PIL
import json
import argparse
import requests
import pdb
import os
import time
import re
from functools import partial
import multiprocessing
from tqdm import tqdm  # 导入 tqdm 进度条
import time  # 用于计算运行时间
import uuid


_T = TypeVar("_T")
class MediaIO(ABC, Generic[_T]):

    @abstractmethod
    def load_bytes(self, data: bytes) -> _T:
        raise NotImplementedError

    @abstractmethod
    def load_base64(self, media_type: str, data: str) -> _T:
        """
        List of media types:
        https://www.iana.org/assignments/media-types/media-types.xhtml
        """
        raise NotImplementedError

    @abstractmethod
    def load_file(self, filepath: Path) -> _T:
        raise NotImplementedError

class ImageMediaIO(MediaIO[Image.Image]):

    def __init__(self, *, image_mode: str = "RGB") -> None:
        super().__init__()

        self.image_mode = image_mode

    def load_bytes(self, data: bytes) -> Image.Image:
        image = Image.open(BytesIO(data))
        image.load()
        return image.convert(self.image_mode)

    def load_base64(self, media_type: str, data: str) -> Image.Image:
        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> Image.Image:
        image = Image.open(filepath)
        image.load()
        return image.convert(self.image_mode)

    def encode_base64(
        self,
        media: Image.Image,
        *,
        image_format: str = "JPEG",
    ) -> str:
        image = media

        with BytesIO() as buffer:
            image = image.convert(self.image_mode)
            image.save(buffer, image_format)
            data = buffer.getvalue()

        return base64.b64encode(data).decode('utf-8')

def encode_image_base64(
    image: Image.Image,
    *,
    image_mode: str = "RGB",
    format: str = "JPEG",
) -> str:
    """
    Encode a pillow image to base64 format.

    By default, the image is converted into RGB format before being encoded.
    """
    image_io = ImageMediaIO(image_mode=image_mode)
    return image_io.encode_base64(image, image_format=format)

ds_collections = {


    'vqav2_val_5w': {
        'train': 'eval/datasets/vqav2/vqav2_train.jsonl',
        'test': 'eval/datasets/vqav2/vqav2_val_5w.jsonl',
        'question': 'eval/datasets/vqav2/v2_OpenEnded_mscoco_val2014_questions_5w.json',
        'annotation': 'eval/datasets/vqav2/v2_mscoco_val2014_annotations_5w.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },

    'vqav2_testdev': {
        'train': 'eval/datasets/vqav2/vqav2_train.jsonl',
        'test': 'eval/datasets/vqav2/vqav2_testdev.jsonl',
        'question': 'eval/datasets/vqav2/v2_OpenEnded_mscoco_test-dev2015_questions.json',
        'metric': None,
        'max_new_tokens': 10,
    },

    'textvqa_val': {
        #'train': 'eval/datasets/textvqa/textvqa_train.jsonl',
        'train': 'eval/datasets/textvqa/textvqa_train.jsonl',
        #'test': 'eval/datasets/textvqa/textvqa_val.jsonl',
        'test': 'eval/datasets/textvqa/textvqa_val.jsonl',
        #'eval/datasets/textvqa/textvqa_val.jsonl',
        #'question': 'eval/datasets/textvqa/textvqa_val_questions.json',
        'question': 'eval/datasets/textvqa/textvqa_val_questions.json',
        #'annotation': 'eval/datasets/textvqa/textvqa_val_annotations.json',
        'annotation': 'eval/datasets/textvqa/textvqa_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },


    'docvqa_val': {
        #'train': 'eval/datasets/docvqa/train.jsonl',
        'train': 'eval/datasets/docvqa/train.jsonl',
        #'test': 'eval/datasets/docvqa/val.jsonl',
        'test': 'eval/datasets/docvqa/val.jsonl',
        #eval/datasets/docvqa/val.jsonl',
        'annotation': 'eval/datasets/docvqa/val/val_v1.0.json',
        'metric': 'anls',
        'max_new_tokens': 100,
    },
    'docvqa_test': {
        'train': 'eval/datasets/docvqa/train.jsonl',
        #'train': 'eval/datasets/docvqa/train.jsonl',
        'test': 'eval/datasets/docvqa/test.jsonl',
        #'test': 'eval/datasets/docvqa/test.jsonl',
        'metric': None,
        'max_new_tokens': 100,
    },

    'chartqa_test_human': {
        'train': 'eval/datasets/chartqa/train_human.jsonl',
        #'train': 'eval/datasets/chartqa/train_human.jsonl',
        'test': 'eval/datasets/chartqa/test_human.jsonl',
        #'test': 'eval/datasets/chartqa/test_human.jsonl',
        #eval/datasets/chartqa/test_human.jsonl',
        'metric': 'relaxed_accuracy',
        'max_new_tokens': 100,
    },
    'chartqa_test_augmented': {
        'train': 'eval/datasets/chartqa/train_augmented.jsonl',
        #'train': 'eval/datasets/chartqa/train_augmented.jsonl',
        'test': 'eval/datasets/chartqa/test_augmented.jsonl',
        #'test': 'eval/datasets/chartqa/test_augmented.jsonl',
        #eval/datasets/chartqa/test_augmented.jsonl',
        'metric': 'relaxed_accuracy',
        'max_new_tokens': 100,
    },
    'ai2diagram_test': {
        'train': 'eval/datasets/ai2diagram/train.jsonl',
        #'train': 'eval/datasets/ai2diagram/train.jsonl',
        'test': 'eval/datasetsset_3rd_batch_0516/AI2D/ai2d-all/ai2d/AI2D_test_vlmevalkit.jsonl',
        #'test': 'eval/datasetsset_3rd_batch_0516/AI2D/ai2d-all/ai2d/AI2D_test_vlmevalkit.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    },
    'mathvista':{
        'train': 'eval/datasets/MathVista/testmini.json',
        #'train': 'eval/datasets/MathVista/testmini.json',
        'test': 'eval/datasets/MathVista/testmini.json',
        #'test': 'eval/datasets/MathVista/testmini.json',
        'metric': 'math_score'
    }

}

def get_prompt(dataset):

    print(dataset)
    if 'textvqa' in dataset:
        prompt = 'please answer the following question in short based on the texts in the picture, '
    elif 'vqav2' in dataset:
        prompt = 'Please directly answer the question in short, '
    elif 'docvqa' in dataset:
        prompt = 'Please answer the following question in short based on the document in the picture, '
    elif 'chart' in dataset:
        prompt = 'Please answer the following question in short based on the chart in the picture, '
    elif 'ai2d' in args.dataset:
        prompt = '' #使用VLMevalkit提供的jsonl文件进行评价，带有提示词。
    elif 'math' in args.dataset:
        prompt = 'Please answer the following questionin short. If there are options with letters after the question, please give the letter of the answer directly, '
    else:
        raise NotImplementedError
        
    return prompt

def add_prefix_to_list(items):
    # 使用列表推导式生成新的列表
    return [f"{chr(65 + i)}. {item}" for i, item in enumerate(items)]

def get_dataset(test_path, prompt, dataset):
    with open(test_path, 'r', encoding='utf-8') as file:
        test_dataset = json.load(file)
    prompt = prompt
    test_dataset_list = []
    test_list = []
    for key, value in test_dataset.items():
        test_list.append(value)
    for idx in range(len(test_list)):
        data = test_list[idx]
        answer_type = data['answer_type']
        if answer_type == 'integer':
            prompt_math = 'Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end. Question: '
        elif answer_type == 'float':
            prompt_math = 'Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end. Question: '
        elif answer_type == 'text':
            prompt_math = 'Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end. Question: '
        else:
            prompt_math = 'Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end. Question: '

        #if answer_type == 'integer':
            #prompt_math = 'Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end,'
        #    prompt_math = 'Please answer the question and give step by step reasoning before you answer. Use the format \"Final answer: ..\" and put the final answer in a latex box \"\\boxed{}\" in the end. Hint: the question requires an integer answer and provide the final value, e.g., 1, 2, 3\n\nQuestion: '
        #elif answer_type == 'float':
            #prompt_math = 'Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end,'
        #    prompt_math = 'Please answer the question and give step by step reasoning before you answer. Use the format \"Final answer: ..\" and put the final answer in a latex box \"\\boxed{}\" in the end. Hint: the question requires a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4\n\nQuestion: '
        #elif answer_type == 'text':
            #prompt_math = 'Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end,'
        #    prompt_math = 'Please answer the question and give step by step reasoning before you answer. Use the format \"Final answer: ..\" and put the final answer in a latex box \"\\boxed{}\" in the end. Hint: the question requires a correct option letter, e.g., A, B, C, D\n\nQuestion: '
        #else:
            #prompt_math = 'Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end,'
        #    prompt_math = 'Please answer the question and give step by step reasoning before you answer. Use the format \"Final answer: ..\" and put the final answer in a latex box \"\\boxed{}\" in the end. Hint: the question requires a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4]\n\nQuestion: '
        #image, question, pid, annotation = 'eval/datasets/MathVista/' + data['image'], data['question'], data['pid'], data['answer']
        prompt_pre = "Your task is to answer the question and give step by step reasoning before you answer, and when you're ready to answer, please use the format \"Final answer: ..\" and put the final answer in a latex box \"\\boxed{}\"\n\n"

        image, question, pid, annotation = 'eval/datasets/MathVista/' + data['image'], data['question'], data['pid'], data['answer']
        question = question[0].lower()+question[1:]
        if data['choices'] is not None:
            options_text_str = ''
            result = add_prefix_to_list(data['choices'])
            options_text_str = "\n".join(result)
            #for item in data['choices']:
            #    options_text_str += '\n'+str(item)

            question = question[0].lower()+question[1:]
            question = question + options_text_str
        else:
            question = question[0].lower()+question[1:]
        #print(image)
        COT_INSTRUCTION = (
            'Your task is to answer the question below. '
            "Give step by step reasoning before you answer, and when you're ready to answer, "
            "please use the format \"Final answer: ..\""
            '\n\n'
            'Question:'
            '\n\n'
            '{question_text}'
        )

        assert os.path.exists(image)
            
        idata_dict = {
            #'question': COT_INSTRUCTION.format(question_text=question),
            #'question': prompt_math + question,
            'question': prompt_pre + prompt_math + question,
            #'question': '<|begin_of_sentence|><|User|>' + prompt_pre + question,
            'question_id': pid,
            'annotation': annotation,
            'image': image,
            'idx': idx
        }

        test_dataset_list.append(idata_dict)
    return test_dataset_list

def extract_last_box_content_or_original(text):
    matches = re.findall(r'boxed\{([^}]+)\}', text)
    return matches[-1] if matches else text


def generate_answer1(question, port, vllm_api, case_name, topk, topp, temperature, max_tokens):

    #pdb.set_trace()
    prompt = '<image>' + question['question'] + '<sep>'
    #question_id = idata['question_id']
    annotation = question['annotation']
    image_path = question['image']
    #assert len(image_path) == 1
    idx = question['idx']
    api_index = idx % len(vllm_api)
    api = vllm_api[api_index]
    print(api, case_name)

    image = PIL.Image.open(image_path)
    base64_image = encode_image_base64(image)
    #prompt = '<image>请描述这张图片的内容<sep>'
    #pdb.set_trace()
    image_url = f"data:image/jpeg;base64,{base64_image}"
    raw_json_data = {
        "prompt": prompt,
        "logprobs": 1,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": topp,
        "top_k": topk,
        "image_url": [image_url],
        }
    #    "image_url": [image_url],
    #    "stream": True,
    json_data = json.dumps(raw_json_data)
    headers = {
    "Content-Type": "application/json",
    }
    
    
    try:
        response = requests.post(api, data=json_data, headers=headers)
    except:
        print('[ERROR!!!!!!!!!!!!!]', question, image_path)
    output = response.text
    #print(prompt, output, annotation)
    if '<sep>' in output:
        full_answers = [str(output.split('<sep>')[1].replace('<eod>',''))]
        extract_answers = extract_last_box_content_or_original(full_answers[0])
        extract_answers = extract_answers.replace('\\\\text{', '')
        extract_answers = extract_answers.replace('\\\text{', '')
        extract_answers = extract_answers.replace('\\text{', '')
        extract_answers = extract_answers.replace('\text{', '')
        extract_answers = extract_answers.replace('text{', '')
        extract_answers = extract_answers.strip()
    else:
        extract_answers = output
    print(question['question_id'], api, idx, question['question'], annotation)
    #print(image_path, prompt, output, annotation, full_answers)
    
    outputs_dict = {
                            'pid': question['question_id'],
                            'type': 'short_cot',
                            'question': question,
                            'full_answer': full_answers,
                            'extraction': extract_answers,
                            'response': full_answers,
                            'annotation': annotation,
                            'image_path': image_path,
                            'index': idx
                        }

    #print(prompt, output, annotation, full_answers)
    return outputs_dict
    #print(prompt, output, annotation, full_answers)
    

    # print(f"question:{[question]}")
    # prompt = question.replace('<n>', '\n').replace('[SEP]', '<sep>')
    # # raw_json_data = {
    # #     "prompt": prompt,
    # #     "logprobs": 1,
    # #     "max_tokens": max_tokens,
    # #     "temperature": temperature,
    # #     "use_beam_search": False,
    # #     "top_p": top_p,
    # #     "top_k": -1,
    # #     "stop": "<eod>"
    # #     }
    # # repetition_penalty
    # raw_json_data = {
            # "prompt": prompt,
            # "logprobs": 1,
            # "use_beam_search": False,
            # "include_stop_str_in_output": True,
            # "skip_special_tokens": False,
            # "max_tokens": 2048,
            # "top_k": 1,
            # "stop": "<eod>"
            # }
    # json_data = json.dumps(raw_json_data)
    # headers = {
        # "Content-Type": "application/json",
    # }
    # try:
        # response = requests.post(f'http://localhost:{port.strip()}/generate', data=json_data, headers=headers)
        # output = json.loads(response.text)['text'][0]
        # print(f'output:{response.text}')
        # print(f'type output:{type(output)}')
        # output = output.replace('<sep> ', '[SEP]').replace('<sep>', '[SEP]').replace('\n','<n>').replace('<eod>','').strip()
    # except Exception as e:
        # output = None
        # print(f'Error: {e}')

from openai import OpenAI

def generate_answer(question, port, vllm_api, case_name, topk, topp, temperature, max_tokens, model_path):

    #pdb.set_trace()
    prompt = question['question']
    annotation = question['annotation']
    image_path = question['image']
    idx = question['idx']
    api_index = idx % len(vllm_api)
    api = vllm_api[api_index]
    print(api, case_name)
    image = PIL.Image.open(image_path)
    base64_image = encode_image_base64(image)
    image_url = f"data:image/jpeg;base64,{base64_image}"
    
    openai_api_key = "EMPTY"
    openai_api_base = api

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        timeout=1000000,
    )
    num = 0
    '''response = client.chat.completions.create(
                model=model_path,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{prompt}"},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }],
                max_completion_tokens=max_tokens,
                max_tokens=max_tokens,
                temperature=0.000001,
                stop='<eod>',
            )'''
    while num <= 5:
        try:
            response = client.chat.completions.create(
                model=model_path,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{prompt}"},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }],
                extra_body= {
                    "add_generation_prompt": True,  # 控制prompt最后是否拼接<|Assistant|>
                    "chat_template_kwargs": {"enable_thinking": False},  # 控制长短思维
                    "include_stop_str_in_output": True,
                    "skip_special_tokens": False
                },

                max_tokens=max_tokens,
                temperature=temperature,
                top_p=topp,
            )
            break
        except Exception as e:
            print('[ERROR]', question, e)
            num += 1
            exit()

    output = response.choices[0].message.content

    if '<sep>' in output or True:
        #output = output
        full_answers = [output]  
        #full_answers = [str(output.split('<sep>')[1].replace('<eod>',''))]
        extract_answers = extract_last_box_content_or_original(full_answers[0])
        extract_answers = extract_answers.replace('\\\\text{', '')
        extract_answers = extract_answers.replace('\\\text{', '')
        extract_answers = extract_answers.replace('\\text{', '')
        extract_answers = extract_answers.replace('\text{', '')
        extract_answers = extract_answers.replace('text{', '')
        extract_answers = extract_answers.strip()
    #else:
    #    extract_answers = output 
    #print(question['question_id'], api, idx, question['question'], annotation)

    
    outputs_dict = {
                            'pid': question['question_id'],
                            'question': question,
                            'extraction': extract_answers,
                            'response': response.choices[0].message.content,
                            'annotation': annotation,
                            'image_path': image_path,
                            'index': idx
                        }

    print('temperature:', temperature, '  topp:', topp )
    print(outputs_dict)
    return outputs_dict

def handle_async_request(func, batch, output_path, batch_num):
    #import pdb
    #pdb.set_trace()
    #func(batch[1])
    with Pool(processes=batch_num) as pool:
        # 使用 map_async 提交异步请求
        async_result = pool.map_async(func, batch)

        # 获取异步结果
        all_result = async_result.get()
        print(all_result)
        json.dump(all_result,
            open(output_path, 'w',encoding='utf-8'),
            ensure_ascii=False)
    return all_result

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="", help="")
    
    parser.add_argument("--dataset", type=str, default='mathvista', help="")
    #parser.add_argument("--dataset", type=str, default='docvqa_val', help="")
    parser.add_argument("--output_dir", type=str, default="", help="")
    parser.add_argument("--output", type=str, default="", help="")
    # parser.add_argument("--port", type=str, default="8002")
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--topp", type=float, default=0.0)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--max_tokens", type=float, default=8192)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--batch_num", type=int, default=128)
    parser.add_argument("--case_name", type=str, default='')
    parser.add_argument("--model_path", type=str, default="")
    # parser.add_argument("--use_url", type=str, default="")
    parser.add_argument("--url", type=str, default="localhost", help="server url")
    parser.add_argument("--port", type=str, default="8000", help="Port number for model API")
    args = parser.parse_args()

    prompt = get_prompt(args.dataset)
    test_dataset = get_dataset(ds_collections[args.dataset]['test'], prompt, args.dataset)
    print(len(test_dataset))
    print('Get dataset:', args.dataset, len(test_dataset))

    result_list = []
    #for idata in test_dataset:
    #    outputs_dict = generate_answer(idata, port=args.port, temperature=args.temperature, top_p=args.topp, top_k=args.topk, max_tokens=args.max_tokens)
    #    result_list.append(outputs_dict)
    batch_num = args.batch_num
    start_time = time.strftime('%y%m%d%H%M%S', time.localtime())
    time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime()) 
    # 异步处理每一批次
    url = f"http://{args.url}:{args.port}/v1"
    vllm_api = [
            #'http://192.168.13.20:8002/v1',
            #'http://172.20.0.230:8001/v1',
            url,
            ] 
    
    round_num = args.round
    topk = args.topk
    topp = args.topp
    temperature = args.temperature
    max_tokens = args.max_tokens
    #print(topk, topp, temperature, max_tokens, batch_num)
    result_multi_round = []
    for i in range(round_num):
        case_name = args.case_name + '-' + str(i+1)
        print(case_name)
        out_dir = args.output_dir
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        result_file = case_name + '-' + str(batch_num) + '-' + args.dataset + '-' + time_prefix + '_short_chat.json'
        result_file_multi = case_name + '-' + str(batch_num) + '-' + args.dataset + '-' + time_prefix + '_multi.json'
        output_path = os.path.join(out_dir, result_file)
        print(output_path)
        output_path_multi = os.path.join(out_dir, result_file_multi)
        print(output_path_multi)
        func = partial(generate_answer, port=args.port, vllm_api=vllm_api, case_name = case_name, topk = topk, topp = topp, temperature = temperature, max_tokens = max_tokens, model_path=args.model_path)
        print(batch_num)
        all_result = handle_async_request(func, test_dataset, output_path, batch_num)

        temp = {}
        for data_item in all_result:
            pid = data_item['pid']
            temp[pid] = data_item
        result_multi_round.append(temp)

        '''temp = {}
        with multiprocessing.Pool(processes=batch_num) as pool:
            #pool = multiprocessing.Pool(processes=128)
            func = partial(generate_answer, port=args.port, vllm_api=vllm_api, case_name = case_name, topk = topk, topp = topp, temperature = temperature, max_tokens = max_tokens)
            results = pool.imap(func, test_dataset)

            # 使用 tqdm 包装 results 以显示进度条
            for i, result in enumerate(tqdm(results, total=len(test_dataset))):
                #all_result.append(result)
                pid = result['pid']
                temp[pid] = result
                if (i + 1) % 2 == 0:
                    with open(output_path, 'w') as f:
                        print(i + 1, ' samples saved!', output_path)
                        json.dump(temp, f)'''


        json.dump(temp, open(output_path, 'w'), indent=4)
        json.dump(result_multi_round, open(output_path_multi, 'w'), indent=4)
        print('Multi results saved to {}'.format(output_path_multi))
        print('Results saved to {}'.format(output_path))
        end_time = time.strftime('%y%m%d%H%M%S', time.localtime())
        print('start:', start_time, ' -- end:', end_time)
        #if args.cot:
        #    cmd = f'python eval/mathvista/extract_answer.py --output_file {results_file} --output_dir {args.out_dir} --quick_extract'
        #else:
        #cmd = f'python ../eval/mathvista/extract_answer.py --output_file {result_file} --output_dir {out_dir} --quick_extract'
        #print(cmd)
        #os.system(cmd)

        cmd = f'python eval/scripts/score/mathvista/calculate_score_detailed_nokey.py --output_file {result_file} --output_dir {out_dir} --score_file {result_file[:-5]}_score.json'
        print(cmd)

        # model_name = os.path.basename(args.model_path)
        # exec_file_name = f'eval/scripts/exec_score/{case_name}_{model_name}_mathvista.sh'
        exec_file_name = f'eval/scripts/exec_score/{uuid.uuid4()}.sh'

        with open(exec_file_name, 'a', encoding='utf-8') as f_out:
            f_out.write(cmd + '\n')

            