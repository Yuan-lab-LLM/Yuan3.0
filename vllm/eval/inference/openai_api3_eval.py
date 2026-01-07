import json
import time
import datetime
import argparse
import os
import base64
import threading
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed  # 线程池默认的线程数是cpu核的数量
from openai import OpenAI, AsyncOpenAI

Max_tokens = int(os.environ.get("MAX_TOKENS", 16384))
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
    parser.add_argument("--port", type=str, default="8000", help="Port number for model API")
    parser.add_argument("--rep_num", type=int, default=1, help="Number of repetitions for each request")
    parser.add_argument("--out_file_name", type=str, default="output.jsonl", help="Name for the output file")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing")
    parser.add_argument("--temperature", type=float, default=0.6, help="temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="top_p")
    parser.add_argument("--url", type=str, default="localhost", help="server url")
    return parser.parse_args()

def batch_infer(prompts, client, temperature, top_p, output_path,lock):
    try:
        if '<sep>' in prompts:
            deal_prompts = prompts.split('<sep>')[0].replace('<n>','\n')
        else:
            deal_prompts = prompts.split('[SEP]')[0].replace('<n>','\n')
        response = client.chat.completions.create(
            model=model_path,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": deal_prompts},
                ],
            }
            ],
            extra_body= {
                "add_generation_prompt": True,  # 控制prompt最后是否拼接<|Assistant|>
                "chat_template_kwargs": {"enable_thinking": enable_thinking},  # 控制长短思维
                "include_stop_str_in_output": True,
                "skip_special_tokens": False
            },
            # extra_body= {
            #     "include_stop_str_in_output": True,
            #     "skip_special_tokens": False
            # },
            max_tokens=Max_tokens,
            temperature=temperature,
            top_p=top_p,

        )
        text = response.choices[0].message.content.replace('\n','<n>')
        if '<sep>' not in prompts and '[SEP]' not in prompts:
            all = prompts + '[SEP]' + text
        else:
            prompts = prompts.replace('<sep>','[SEP]').split('[SEP]')[0].replace('\n','<n>')
            all = prompts + '[SEP]' + text
            
        # all = deal_prompts.replace('\n','<n>') + text
        # logging.info(f"one request done......")
        try:
            lock.acquire()
            with open(output_path, 'a', encoding='utf-8') as f_out:
                f_out.write(all+'\n')
            lock.release()
        except Exception as e:
            lock.release()
    except Exception as e:
        print('data error >>>>',e)



def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = f"http://{args.url}:{args.port}/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        timeout=1000000,
    )
    start_time_total = datetime.datetime.now()
    data_list = []


    with open(args.input, 'r',encoding='utf-8') as f_in:
        for line in f_in:
            data_list.append(line.strip())
    lock = threading.Lock()
    max_workers = args.batch_size
    output_path = args.out_file_name
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logging.info(f"Output will be saved to: {output_path}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for prompts in data_list:
            executor.submit(batch_infer, prompts, client, args.temperature, args.top_p, output_path, lock)



    end_time_total = datetime.datetime.now()
    end_time_diff_total = end_time_total - start_time_total
    end_milliseconds_total = int(end_time_diff_total.total_seconds())
    print(f'{args.input} cost = {end_milliseconds_total} s')


if __name__ == '__main__':
    main()


