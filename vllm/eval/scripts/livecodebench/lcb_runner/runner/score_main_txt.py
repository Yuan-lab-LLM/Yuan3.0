import os
import json

from lcb_runner.runner.parser import get_args
from lcb_runner.utils.scenarios import Scenario
from lcb_runner.lm_styles import LanguageModelStore
from lcb_runner.runner.runner_utils import build_runner
from lcb_runner.utils.path_utils import get_output_path
from lcb_runner.evaluation import extract_instance_results
from lcb_runner.runner.scenario_router import (
    build_prompt_benchmark,
    combine_results,
    sort_and_extract_save_results,
    get_metrics,
)
from lcb_runner.utils.extraction_utils import extract_code_yuan
from datetime import datetime
from lcb_runner.benchmarks import load_code_generation_dataset
from lcb_runner.prompts import format_prompt_generation


def get_file_list(folder, file_type_list):
    filelist = []
    for dirpath, _, filenames in os.walk(folder):
        for file in filenames:
            file_type = file.split('.')[-1]
            if file_type in file_type_list:
                file_fullname = os.path.join(dirpath, file)
                filelist.append(file_fullname)

    return filelist


def main():
    args = get_args()

    model = LanguageModelStore[args.model]
    # benchmark, format_prompt = build_prompt_benchmark(args)
    benchmark = load_code_generation_dataset(args.release_version)
    # if args.debug:
    #     print(f"Running with {len(benchmark)} instances in debug mode")
    #     benchmark = benchmark[:2]
    start_date = datetime.strptime("2024-08-01", "%Y-%m-%d")
    part_benchmark = []
    for problem in benchmark:
        if problem.contest_date<start_date:
            continue

        part_benchmark.append(problem)
    benchmark = part_benchmark
    output_path = get_output_path(model.model_repr, args)
    eval_file = output_path.replace(".json", "_eval.json")
    eval_all_file = output_path.replace(".json", "_eval_all.json")

    ppath = args.input_file
    print(f"ppath:{ppath}")
    if os.path.isfile(ppath):
        files_lst = [ppath]
    else:
        files_lst = get_file_list(ppath, ['txt', 'jsonl'])
    txt_save_results = []
    for input_path in files_lst:
        print(f"input_path:{input_path}")
        with open(input_path, 'r', encoding='utf-8') as file:
            # 读取所有行
            txt_save_results.extend(file.readlines())

    question_dict = {}
    for line in txt_save_results:
        line = line.strip()
        if not line:
            continue  # 跳过空行
        # 按照 "[SEP]" 分割问题和答案
        line = line.replace('<|begin_of_sentence|><|User|>', '').replace('<|Assistant|>', '[SEP]').replace('<think>','<eog>').replace('</think>', '</eog>')
        parts = line.replace('<sep>','[SEP]').split('[SEP]')
        # if len(parts) != 2:
        #     print(f"Warning: Line '{line}' does not contain exactly one '[SEP]'. Skipping.")
        #     continue
        question, answer = parts[0].strip(), parts[1].strip()

        # 如果问题已经存在于字典中，将答案添加到对应的列表中
        if question in question_dict:
            question_dict[question].append(line)
        else:
            # 如果问题不存在于字典中，创建一个新的键值对
            question_dict[question] = [line]

    txt_save_results_cores = []
    tmp_benchmark = []
    for problem in benchmark:
        # format_prompt = format_prompt_generation(problem, LanguageModelStore["Yuan_M32_generate_model"].model_style)
        format_prompt = f"{problem.question_content}<sep>"
        format_prompt = format_prompt.replace("\r\n", "<n>").replace("\n", "<n>")
        for extract_question, extract_lines in question_dict.items():
            if extract_question.split("<n>")[0] in format_prompt:
                # print(format_prompt)
                txt_save_results_cores.append(extract_lines)
                tmp_benchmark.append(problem)
                break

    txt_save_results = txt_save_results_cores
    print(f"len(txt_save_results)")
    benchmark = tmp_benchmark
    assert len(benchmark)==len(txt_save_results), "生成答案数量与问题数量不匹配"
    # txt_save_results = [txt_save_results[i:i + 10] for i in range(0, len(txt_save_results), 10)] # 按照10等分切片
    save_results = []
    for instance, line_res_list_1 in zip(benchmark, txt_save_results):
        assert len(line_res_list_1)>=args.n, "生成长度小于指定长度"
        outputs_list,extracted_list = [],[]
        # parts = line_res_2.split("[SEP]")
        # answer = parts[1].strip().replace('<eop>', '').replace('<n>', '\n')
        # outputs_list.append(answer)
        # extracted_list.append(extract_code_yuan(answer, model.model_style))
        for line_res in line_res_list_1[:args.n]:

            parts = line_res.replace('<sep>','[SEP]').split("[SEP]")
            answer = parts[1].strip().replace('<eop>', '').replace('<n>', '\n')
            outputs_list.append(answer)
            extracted_list.append(extract_code_yuan(answer, model.model_style))
        save_results.append(instance.insert_output(outputs_list, extracted_list))

    benchmark = sorted(benchmark, key=lambda x: x.question_id)
    save_results = sorted(save_results, key=lambda x: x["question_id"])
    combined_results = [
        (save_result_instance["output_list"], save_result_instance["code_list"])
        for save_result_instance in save_results
    ]

    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=4)

    if args.evaluate:
        metrics = get_metrics(args.scenario, args, benchmark, combined_results)
        graded = extract_instance_results(metrics[1])
        old_eval_all_results = []
        old_eval_results = []

        if args.scenario == Scenario.codegeneration:
            if metrics:
                metadatas = metrics[2]
            else:
                metadatas = [[] for _ in benchmark]
            save_eval_results = [
                instance.insert_output_evaluation(
                    outputs_list, extracted_list, graded_list, metadata=meta
                )
                for instance, (outputs_list, extracted_list), graded_list, meta in zip(
                    benchmark, combined_results, graded, metadatas
                )
            ]
            if metrics and old_eval_results:
                old_eval_results
                metrics[2] = old_eval_results[2] + metrics[2]
        elif args.scenario == Scenario.selfrepair:
            metadatas = metrics[2]
            with open(
                f"output/{model.model_repr}/{Scenario.codegeneration}_{args.codegen_n}_{args.temperature}_eval_all.json"
            ) as f:
                code_gen_evals = json.load(f)
            original_code_lists = [
                code_gen_eval["code_list"] for code_gen_eval in code_gen_evals
            ]

            save_eval_results = [
                instance.insert_output_evaluation(
                    outputs_list,
                    extracted_list,
                    graded_list,
                    metadata=meta,
                    original_code_list=original_code_list,
                )
                for instance, (
                    outputs_list,
                    extracted_list,
                ), graded_list, meta, original_code_list in zip(
                    benchmark, combined_results, graded, metadatas, original_code_lists
                )
            ]

        else:
            save_eval_results = [
                instance.insert_output_evaluation(
                    outputs_list, extracted_list, graded_list
                )
                for instance, (outputs_list, extracted_list), graded_list in zip(
                    benchmark, combined_results, graded
                )
            ]

        save_eval_results = old_eval_all_results + save_eval_results

        with open(eval_file, "w") as f:
            json.dump(metrics, f, indent=4)

        with open(eval_all_file, "w") as f:
            json.dump(save_eval_results, f, indent=4)


if __name__ == "__main__":
    main()

