import json
from typing import Optional
import re,os

def parse_code_block(string: str, lang: str) -> Optional[str]:
    code_pattern = fr"```{lang}\n(.*?)\n```"
    match = re.search(code_pattern, string, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return string
        #return parse_first_func(string, lang)
def parse_first_func(code: str, lang: str) -> Optional[str]:
    assert lang == "python", "Only python is supported for now. TODO: Rust"
    code_lines = code.split("\n")
    def_i = 0
    last_i = 0
    for i, line in enumerate(code_lines):
        if line.startswith("def "):
            if def_i == 0:
                def_i = i
            else:
                break
        if line == "" and def_i != 0:
            last_i = i
            break

    if last_i == 0:
        last_i = len(code_lines) - 1

    if def_i == 0:
        return None

    return "\n".join(code_lines[def_i:last_i+1])

def generate_func(code):
    #x =code[code.find("```python")+9:]
    x = code
    x = x.replace('\nFIX = """\nAdd more test cases.\n"""','')
    x = x.replace('\n\n\ndef','\ndef')
    x = x.replace('\n    \n\n    ','\n    \n    ')
    x = x.replace('\n\n    ','\n    ')
    x = x.replace('\n\n\n','\n\n')
    x = x.replace('"""\n        ','"""\n    ')
    x = x.replace('\n```python','')
    if x.count('"""') >= 2:
        x = x.replace(x[x.find('"""'):x.find('"""',x.find('"""')+1)+3],'')
    elif x.count("'''") >= 2:
        x = x.replace(x[x.find("'''"):x.find("'''",x.find("'''")+1)+3],'')
    x_func = parse_code_block(x, "python")
    if x_func is not None:
        x_func = x[0:x.find(x_func[0:10])]+x_func
    else:
        x_func = x
    if x_func.count('\ndef') > 1 and x_func.count('\n```')<1 and x_func.count('\n#')<1 and x_func.count('\nif')<1:
        x_func = x_func.replace(x_func[x_func.find('\ndef',x_func.find('\ndef')+1):],'\n')
    if x_func.find('\nprint')>0:
        x_func = x_func[0:x_func.find('\nprint')]
    if x_func.find('\n# 示例')>0:
        x_func = x_func[0:x_func.find('\n# 示例')]
    if x_func.find('\n```')>0:
        x_func = x_func[0:x_func.find('\n```')]
    if x_func.find('\n# 测试')>0:
        x_func = x_func[0:x_func.find('\n# 测试')]
    if x_func.find('\n# 单元测试')>0:
        x_func = x_func[0:x_func.find('\n# 单元测试')]
    if x_func.find('\n#')>0:
        x_func = x_func[0:x_func.find('\n#')]
    if x_func.find('\nif')>0:
        x_func = x_func[0:x_func.find('\nif')]
    #print(x_func)
    return x_func

def getCode(response):

    pattern = r'```python.*?```'

    return re.findall(pattern, response)

def clean_tab(msg_text):
    __sep_note = "<n>"
    msg_text = msg_text.replace("\n", __sep_note)
    msg_text = msg_text.replace(__sep_note + __sep_note, __sep_note)
    msg_text = msg_text.replace(__sep_note + __sep_note, __sep_note)
    msg_text = msg_text.replace(__sep_note + __sep_note, __sep_note)
    return msg_text


def get_file_list(folder, file_type_list):
    filelist = []
    for dirpath, _, filenames in os.walk(folder):
        for file in filenames:
            file_type = file.split('.')[-1]
            if file_type in file_type_list:
                file_fullname = os.path.join(dirpath, file)
                filelist.append(file_fullname)

    return filelist


import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--case-name', type=str, help='Path')
parser.add_argument('--input-path', type=str, help='Path')
parser.add_argument('--output-path', type=str, help='Path2')

args = parser.parse_args()

path = args.input_path
dirname= args.output_path

ori_path= './datasets/HUMANEVAL/HumanEval-instructions-en.jsonl'
id_list = {}
with open(ori_path,'r') as file:
    for line in file.readlines():
        data = json.loads(line)
        task_id = data['task_id']
        prompt = clean_tab(data['prompt'].replace('\n','<n>')).split('Response')[0].replace('<n><n>','<n>').replace('<n><n>','<n>')
        # prompt = clean_tab(data['prompt'].replace('\n','<n>')).split('The function signature is as follows')[0].replace('<n><n>','<n>').replace('<n><n>','<n>')
        #print(prompt)
        id_list[prompt]=task_id



if not os.path.exists(dirname):
    os.mkdir(dirname)

path2 = os.path.join(dirname,'samples.jsonl')

#idir = os.listdir(path)

total_dict={}

#for subpath in idir:
#    if args.case_name+'_' not in subpath:
#        continue
#    subsamples = os.listdir(os.path.join(path,subpath))
if os.path.isfile(path):
    files_lst = [path]
else:
    files_lst = get_file_list(path,['jsonl', 'txt'])
for subsample_file in files_lst:
    # print(subsample_file)
    with open(subsample_file,'r') as file:
        for data in file.readlines():
            #data = json.loads(data)
            
            data = data.replace('<|begin_of_sentence|><|User|>', '').replace('<|Assistant|>', '[SEP]').replace('<think></think>', '').replace('<think>','<eog>').replace('</think>', '</eog>').replace('<eod>', '').replace('<|end_of_sentence|>', '').replace('[SEP]', '<sep>').replace('<eop>', '').replace('<n><sep>', '<sep>')
            # data = data.replace('<sep>', '[SEP]',1).replace('<sep>', '').replace('[SEP]','<sep>')
            solution = data.replace('\n','<n>')
            prompt = solution.split('The function signature is as follows')[0].replace('<n><n>','<n>').replace('<n><n>','<n>')
            # prompt = solution.split('<n>Response:')[0].replace('<n><n>','<n>').replace('<n><n>','<n>')
            # print(id_list)
            #print(prompt)
            task_id = id_list[prompt]
            if task_id in total_dict:
                continue
            # print(task_id)
            if task_id in total_dict:
                continue

            if '### Solution Code' in solution:
                solution = solution.split('### Solution Code')[1]
            elif '</eog>' in solution:
                solution = solution.split('</eog>')[1]
            elif '<eog>' in solution:
                solution = solution.split('<eog>')[1]
            else:
                solution = solution.split('<sep>')[1]
                #solution = solution[solution.find("<sep>")+len("<sep>"):]

            if "```python" not in solution:
                solution = "```python<n>"+solution.strip()+"<n>```"
            else:
                if not re.search(r'```python.*?```', solution, re.DOTALL):
                    solution = solution.strip()+"<n>```"
            # code_text = re.findall(r'```python.*?```',solution)[-1]
            # if code_text:
            #    solution = code_text
            # code = solution
            try:
                #code = getCode(solution)[-1].replace('[SEP]','').replace('<n>','\n').replace('<sep>','').replace('<eop>','')
                #print(f"code:{[code]}")
                code_lst = getCode(solution)
                # code = '\n'.join(code_lst).replace('[SEP]','').replace('<n>','\n').replace('<sep>','').replace('<eop>','')
                for code_item in code_lst[::-1]:
                    code = code_item.replace('[SEP]','').replace('<n>','\n').replace('<sep>','').replace('<eop>','')
                    if '\ndef ' in code:
                        break
            except:
                try:
                    code = re.findall(r'```python.*?',solution)[-1].replace('[SEP]','').replace('<n>','\n').replace('<sep>','').replace('<eop>','')
                except:
                    code = ""
            # print(f"code:{[code]}")

            #code = code[code.find("```python")+9:code.find("```",code.find("```python")+1)]
            code=code[code.find("```python")+9:].replace('<eop>','')

            func = generate_func(code)
            x = dict(task_id=task_id,completion=func)
            total_dict[task_id]=x
print(f"total_dict:{len(total_dict)}")
count=0
for i in range(164):
    if 'HumanEval/'+str(i) not in total_dict:
        count+=1
        print(i)
        total_dict[i]=dict(task_id='HumanEval/'+str(i),completion='')

print(f'还有{count}道题没推理完成')
with open(path2,'w',encoding='utf-8') as file2:
    for key in total_dict:
        x = total_dict[key]
        file2.write(json.dumps(x,ensure_ascii=False)+'\n')
        file2.flush()

