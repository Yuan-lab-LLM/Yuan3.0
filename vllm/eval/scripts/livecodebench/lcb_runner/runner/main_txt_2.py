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

    ##修改此处的path
    with open(ppath, 'r', encoding='utf-8') as file:
        # 读取所有行
        txt_save_results = file.readlines()

    # if args.n == 1:
    #### 每个问题生成1个答案
    # 答案匹配
    # txt_save_results_cores = []
    # for problem in benchmark:
    #     # format_prompt = format_prompt_generation(problem, LanguageModelStore["Yuan_M32_generate_model"].model_style)
    #     format_prompt = f"{problem.question_content}<sep>"
    #     format_prompt = format_prompt.replace("\r\n", "<n>").replace("\n", "<n>")
    #     for gen_line in txt_save_results:
    #         if gen_line.split("<n>")[0] in format_prompt:
    #             # print(format_prompt)
    #             txt_save_results_cores.append(gen_line)
    #             break
    # txt_save_results = txt_save_results_cores


    # save_results = []
    # for instance, line_res in zip(benchmark, txt_save_results):
    #     # outputs_list,extracted_list = [],[]
    #     parts = line_res.split("[SEP]")
    #     answer = parts[1].strip().replace('<eop>', '').replace('<n>', '\n')
    #     # outputs_list.append(answer)
    #     # extracted_list.append(extract_code_yuan(answer, model.model_style))
    #     # instance.insert_output([answer], [extract_code_yuan(answer, model.model_style)])
    #     save_results.append(instance.insert_output([answer], [extract_code_yuan(answer, model.model_style)]))
    #######################################################################
    # else:
    #### 每个问题生成n个答案
    # 答案匹配
    question_dict = {}
    for line in txt_save_results:
        line = line.strip()
        if not line:
            continue  # 跳过空行
        # 按照 "[SEP]" 分割问题和答案
        line = line.replace('<|begin_of_sentence|><|User|>', '').replace('<|Assistant|>', '[SEP]').replace('<think>','<eog>').replace('</think>', '</eog>')
        parts = line.replace('<sep>','[SEP]').split('[SEP]')
        # line = line.split('<n>Question:<n>',1)[-1]
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
    #######################################################################

    # if args.continue_existing or args.continue_existing_with_eval:
    #     if os.path.exists(output_path):
    #         with open(output_path, "r") as f:
    #             old_save_results = json.load(f)
    #     elif os.path.exists(eval_all_file):
    #         with open(eval_all_file, "r") as f:
    #             old_save_results = json.load(f)
    #     else:
    #         print(
    #             f"File {output_path} does not exist in --continue_existing, starting from scratch"
    #         )
    #         old_save_results = []

    #     old_save_results = [
    #         instance
    #         for instance in old_save_results
    #         if instance["output_list"] and [x for x in instance["output_list"] if x]
    #     ]
    #     # temp_results = []
    #     # for instance in old_save_results: # 替换原始生成的code_list
    #     #     if instance["output_list"] and [x for x in instance["output_list"] if x]:
    #     #         instance["code_list"] = [extract_code_yuan(output, model.model_style) for output in instance["output_list"]]
    #     #         temp_results.append(instance)
    #     # old_save_results = temp_results

    #     old_save_results_question_ids = [
    #         instance["question_id"] for instance in old_save_results
    #     ]
    #     remaining_benchmark = [
    #         instance
    #         for instance in benchmark
    #         if instance.question_id not in old_save_results_question_ids
    #     ]
    #     print(
    #         f"Found {len(old_save_results)} existing generations, continuing with {len(remaining_benchmark)} remaining"
    #     )
    # else:
    #     old_save_results = []
    #     remaining_benchmark = benchmark

    # if len(remaining_benchmark) > 0:
    #     runner = build_runner(args, model)
    #     results: list[list[str]] = runner.run_main(remaining_benchmark, format_prompt)
    # else:
    #     results = []

    # combined_results = combine_results(
    #     args.scenario, results, model, args.cot_code_execution
    # )

    # save_results = [
    #     instance.insert_output(outputs_list, extracted_list)
    #     for instance, (outputs_list, extracted_list) in zip(
    #         remaining_benchmark, combined_results
    #     )
    # ]

    # if args.continue_existing or args.continue_existing_with_eval:
    #     save_results += old_save_results

    # save_results, combined_results = sort_and_extract_save_results(
    #     args.scenario, save_results
    # )
    benchmark = sorted(benchmark, key=lambda x: x.question_id)
    save_results = sorted(save_results, key=lambda x: x["question_id"])
    combined_results = [
        (save_result_instance["output_list"], save_result_instance["code_list"])
        for save_result_instance in save_results
    ]

    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=4)

    # for i in range(len(combined_results)):
    #     for j in range(len(combined_results[i][1])):
    #         if "def solve()" in combined_results[i][1][j]:
    #             from lcb_runner.utils.extraction_utils import extract_code, LMStyle

    #             combined_results[i][1][j] = extract_code(
    #                 combined_results[i][0][j], LMStyle.Gemini
    #             )
    #             if "\nsolve()" not in combined_results[i][1][j]:
    #                 combined_results[i][1][j] += "\n\nsolve()"

    #                 # combined_results[i][1][j] += "\n\nsolve()"
    #                 print(combined_results[i][1][j])

    if args.evaluate:
        # if args.continue_existing_with_eval and os.path.exists(eval_all_file):
        #     with open(eval_all_file) as fp:
        #         old_eval_all_results = json.load(fp)

        #     if os.path.exists(eval_file):
        #         with open(eval_file) as fp:
        #             old_eval_results = json.load(fp)
        #     else:
        #         old_eval_results = None

        #     old_eval_results_question_ids = [
        #         instance["question_id"] for instance in old_eval_all_results
        #     ]
        #     remaining_indices = [
        #         idx
        #         for idx in range(len(benchmark))
        #         if benchmark[idx].question_id not in old_eval_results_question_ids
        #     ]
        #     benchmark = [benchmark[idx] for idx in remaining_indices]
        #     combined_results = [combined_results[idx] for idx in remaining_indices]

        #     old_eval_size = len(old_eval_results_question_ids)
        #     new_eval_size = len(benchmark)

        #     if new_eval_size == 0:
        #         return

        #     print(f"Found {old_eval_size}, running evals for {new_eval_size} problems")

        #     metrics = get_metrics(args.scenario, args, benchmark, combined_results)
        #     graded = extract_instance_results(metrics[1])

        #     if old_eval_results:
        #         for key in metrics[0]:
        #             if key in old_eval_results[0]:
        #                 if key != "detail":
        #                     metrics[0][key] = (
        #                         old_eval_size * old_eval_results[0][key]
        #                         + new_eval_size * metrics[0][key]
        #                     )
        #                     metrics[0][key] /= old_eval_size + new_eval_size

        #         for key in metrics[0]["detail"]:
        #             if key in old_eval_results[0]["detail"]:
        #                 metrics[0]["detail"][key] = {
        #                     **metrics[0]["detail"][key],
        #                     **old_eval_results[0]["detail"][key],
        #                 }
        #         metrics[1] = {**metrics[1], **old_eval_results[1]}
        #     else:
        #         print("Old eval file not present, cannot update eval file")
        #         metrics = {}

        # else:
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
