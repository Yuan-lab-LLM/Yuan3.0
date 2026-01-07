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
import re
from functools import partial
import multiprocessing
from tqdm import tqdm  # 导入 tqdm 进度条
import time  # 用于计算运行时间
import copy
import shutil

import datetime
import threading
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed  # 线程池默认的线程数是cpu核的数量
from openai import OpenAI, AsyncOpenAI, APIError

Max_tokens = int(os.environ.get("MAX_TOKENS", 4096))
model_path = os.environ.get("MODEL_PATH", "")
enable_thinking = os.environ.get("enable_thinking", '')

if enable_thinking == 'True':
    enable_thinking = True
elif enable_thinking == 'False':
    enable_thinking = False
else:
    print('environ get error')
    exit()
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input file path")
    # parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--port", type=str, default="8001", help="Port number for model API")
    parser.add_argument("--rep_num", type=int, default=1, help="Number of repetitions for each request")
    parser.add_argument("--out_file_name", type=str, default="output.jsonl", help="Name for the output file")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing")
    parser.add_argument("--temperature", type=float, default=0.6, help="temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="top_p")
    parser.add_argument("--url", type=str, default="localhost", help="server url")
    return parser.parse_args()

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

def generate_answer(data, openai_api_base, temperature, top_p, output_path, lock):
    try:
        if 'image_id' in data:
            prompt = data['input'].replace('<n>','\n')
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prompt}"},
                    # {"type": "image_url", "image_url": {"url": image_url1}},
                ],
            }]
            image_path =  data['image_id'] + '.jpg'
            image = PIL.Image.open(image_path)
            base64_image = encode_image_base64(image)
            every_image_url = f"data:image/jpeg;base64,{base64_image}"
            messages[0]['content'].append({"type": "image_url", "image_url": {"url": every_image_url}})
        else:
            prompt = data['question'].replace('<n>','\n')     
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prompt}"},
                    # {"type": "image_url", "image_url": {"url": image_url1}},
                ],
            }]
            for num in range(len(data['imagepath'])):
                if 'baseline/images' in data['imagepath'][num] or 'gt/images' in data['imagepath'][num] or 'retrieve/images' in data['imagepath'][num]:
                    image_path = f'eval/datasets/mm-bench/' + data['imagepath'][num]
                elif 'images/doc_matrix' in data['imagepath'][num]:
                    image_path = f'eval/datasets/mrag/docmatrix/' + data['imagepath'][num]
                else:
                    image_path = f'eval/datasets/mrag/the_cauldron/' + data['imagepath'][num]
                image = PIL.Image.open(image_path)
                base64_image = encode_image_base64(image)
                every_image_url = f"data:image/jpeg;base64,{base64_image}"
                messages[0]['content'].append({"type": "image_url", "image_url": {"url": every_image_url}})
    except Exception as e:
        print(f"catch data error: {e}")
        return

    openai_api_key = "EMPTY"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        timeout=1000000,
    )

    try:
        response = client.chat.completions.create(
            model=model_path,
            messages=messages,
            extra_body= {
                "add_generation_prompt": True,  # 控制prompt最后是否拼接<|Assistant|>
                "chat_template_kwargs": {"enable_thinking": enable_thinking},  # 控制长短思维
                "include_stop_str_in_output": True,
                "skip_special_tokens": False
            },            
            max_completion_tokens=Max_tokens,
            max_tokens=Max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    except APIError as e:
        print(f"hhb APIError： {e.status_code} - {e}")
        return
    except Exception as e:
        print(f"catch error: {e}")
        return
    try:
        lock.acquire()
        with open(output_path, 'a', encoding='utf-8') as f_out:
            data['solution'] = response.choices[0].message.content
            json_str = json.dumps(data, ensure_ascii=False)
            f_out.writelines(json_str+'\n')
        lock.release()
    except Exception as e:
        lock.release()



def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    openai_api_base = f"http://{args.url}:{args.port}/v1"

    start_time_total = datetime.datetime.now()
    data_list = []


    with open(args.input, 'r',encoding='utf-8') as f_in:
        for line in f_in:
            for i in range(1):
                one_json = json.loads(line)
                data_list.append(one_json)        

    max_workers = args.batch_size
    output_path = args.out_file_name
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logging.info(f"Output will be saved to: {output_path}")

    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for data in data_list:
            executor.submit(generate_answer, data, openai_api_base, args.temperature, args.top_p, output_path, lock)

    end_time_total = datetime.datetime.now()
    end_time_diff_total = end_time_total - start_time_total
    end_seconds_total = int(end_time_diff_total.total_seconds())
    print(f'{args.input} cost = {end_seconds_total} s')


if __name__ == '__main__':
    main()


