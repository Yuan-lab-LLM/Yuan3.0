## 我们在众多评测集上对Yuan3.0 Flash模型进行了评估测试，详细的汇总如下：
<table style="border-collapse: collapse;">
  <tr>
    <th style="border: 1px solid white; padding: 8px;">序号</th>
    <th style="border: 1px solid white; padding: 8px;">benchmark名称</th>
    <th style="border: 1px solid white; padding: 8px;">数据路径</th>
    <th style="border: 1px solid white; padding: 8px;">推理脚本</th>
    <th style="border: 1px solid white; padding: 8px;">评测脚本</th>
  </tr>
  
  <!-- ChatRAG -->
  <tr>
    <td style="border: 1px solid white; padding: 8px;">1</td>
    <td style="border: 1px solid white; padding: 8px;">ChatRAG</td>
    <td style="border: 1px solid white; padding: 8px;">eval/datasets/ChatRAG</td>
    <td style="border: 1px solid white; padding: 8px;">eval/inference/openai_api3_eval.py</td>
    <td style="border: 1px solid white; padding: 8px;">eval/scripts/ChatRAG/eval_chatqa.sh</td>
  </tr>
  
  <!-- MMTab -->
  <tr>
    <td style="border: 1px solid white; padding: 8px;">2</td>
    <td style="border: 1px solid white; padding: 8px;">MMTab</td>
    <td style="border: 1px solid white; padding: 8px;">eval/datasets/MMTab</td>
    <td style="border: 1px solid white; padding: 8px;">eval/inference/yuanvl_api_multi_eval.py</td>
    <td style="border: 1px solid white; padding: 8px;">eval/scripts/MMTab/table.sh</td>
  </tr>
  
  <!-- SummEval -->
  <tr>
    <td style="border: 1px solid white; padding: 8px;">3</td>
    <td style="border: 1px solid white; padding: 8px;">SummEval</td>
    <td style="border: 1px solid white; padding: 8px;">eval/datasets/SummEval</td>
    <td style="border: 1px solid white; padding: 8px;">eval/inference/openai_api3_table.py</td>
    <td style="border: 1px solid white; padding: 8px;">eval/scripts/Summary/*.py</td>
  </tr>
  
  <!-- BFCL -->
  <tr>
    <td style="border: 1px solid white; padding: 8px;">4</td>
    <td style="border: 1px solid white; padding: 8px;">BFCL</td>
    <td style="border: 1px solid white; padding: 8px;">
      <a href="https://github.com/ShishirPatil/gorilla/tree/cd9429ccf3d4d04156affe883c495b3b047e6b64" style="color: inherit; text-decoration: none;">
        BFCL (GitHub)
      </a>
    </td>
    <td style="border: 1px solid white; padding: 8px;">eval/inference/generate.sh</td>
    <td style="border: 1px solid white; padding: 8px;">eval/scripts/BFCL/score.sh</td>
  </tr>
  
  <!-- MATH-500 -->
  <tr>
    <td style="border: 1px solid white; padding: 8px;">5</td>
    <td style="border: 1px solid white; padding: 8px;">MATH-500</td>
    <td style="border: 1px solid white; padding: 8px;">eval/datasets/MATH-500</td>
    <td style="border: 1px solid white; padding: 8px;">eval/inference/openai_api3_eval.py</td>
    <td style="border: 1px solid white; padding: 8px;">eval/scripts/MATH-500/judge_with_vllm_model_math.py</td>
  </tr>
  
  <!-- AIME2024 -->
  <tr>
    <td style="border: 1px solid white; padding: 8px;">6</td>
    <td style="border: 1px solid white; padding: 8px;">AIME2024</td>
    <td style="border: 1px solid white; padding: 8px;">eval/datasets/AIME2024</td>
    <td style="border: 1px solid white; padding: 8px;">eval/inference/openai_api3_eval.py</td>
    <td style="border: 1px solid white; padding: 8px;">eval/scripts/AIME2024/eval_easy_lzy.sh</td>
  </tr>
  
  <!-- livecodebench -->
  <tr>
    <td style="border: 1px solid white; padding: 8px;">7</td>
    <td style="border: 1px solid white; padding: 8px;">livecodebench</td>
    <td style="border: 1px solid white; padding: 8px;">eval/datasets/livecodebench</td>
    <td style="border: 1px solid white; padding: 8px;">eval/inference/openai_api3_livecode.py</td>
    <td style="border: 1px solid white; padding: 8px;">eval/scripts/livecodebench/eval_livecodebench_v6.sh</td>
  </tr>
  
  <!-- HumanEval -->
  <tr>
    <td style="border: 1px solid white; padding: 8px;">8</td>
    <td style="border: 1px solid white; padding: 8px;">HumanEval</td>
    <td style="border: 1px solid white; padding: 8px;">eval/datasets/HumanEval</td>
    <td style="border: 1px solid white; padding: 8px;">eval/inference/openai_api3_eval.py</td>
    <td style="border: 1px solid white; padding: 8px;">eval/scripts/HumanEval/score_humaneval_ly.sh</td>
  </tr>
  
  <!-- MMLU -->
  <tr>
    <td style="border: 1px solid white; padding: 8px;">9</td>
    <td style="border: 1px solid white; padding: 8px;">MMLU</td>
    <td style="border: 1px solid white; padding: 8px;">eval/datasets/MMLU</td>
    <td style="border: 1px solid white; padding: 8px;">eval/inference/openai_api3_eval.py</td>
    <td style="border: 1px solid white; padding: 8px;">eval/scripts/MMLU/eval_easy_lzy.sh</td>
  </tr>
  
  <!-- MMLU Pro -->
  <tr>
    <td style="border: 1px solid white; padding: 8px;">10</td>
    <td style="border: 1px solid white; padding: 8px;">MMLU Pro</td>
    <td style="border: 1px solid white; padding: 8px;">eval/datasets/MMLU-Pro</td>
    <td style="border: 1px solid white; padding: 8px;">eval/inference/openai_api3_eval.py</td>
    <td style="border: 1px solid white; padding: 8px;">eval/scripts/MMLU-Pro/eval_easy_lzy.sh</td>
  </tr>
  
  <!-- GPQA-Diamond -->
  <tr>
    <td style="border: 1px solid white; padding: 8px;">11</td>
    <td style="border: 1px solid white; padding: 8px;">GPQA-Diamond</td>
    <td style="border: 1px solid white; padding: 8px;">eval/datasets/GPQA-Diamond</td>
    <td style="border: 1px solid white; padding: 8px;">eval/inference/openai_api3_eval.py</td>
    <td style="border: 1px solid white; padding: 8px;">eval/scripts/GPQA-Diamond/eval_easy_lzy.sh</td>
  </tr>
  
  <!-- ChartQA -->
  <tr>
    <td style="border: 1px solid white; padding: 8px;">12</td>
    <td style="border: 1px solid white; padding: 8px;">ChartQA</td>
    <td style="border: 1px solid white; padding: 8px;">eval/datasets/ChartQA</td>
    <td style="border: 1px solid white; padding: 8px;">
      eval/inference/chartqa_test_augmented.py<br>
      eval/inference/chartqa_test_human.py
    </td>
    <td style="border: 1px solid white; padding: 8px;">eval/scripts/ChartQA/score_analyze_all.py</td>
  </tr>
  
  <!-- DocVQA -->
  <tr>
    <td style="border: 1px solid white; padding: 8px;">13</td>
    <td style="border: 1px solid white; padding: 8px;">DocVQA</td>
    <td style="border: 1px solid white; padding: 8px;">eval/datasets/DocVQA</td>
    <td style="border: 1px solid white; padding: 8px;">
      eval/inference/docvqa_val.py<br>
      eval/inference/chat_docvqa_oneword_chat.py
    </td>
    <td style="border: 1px solid white; padding: 8px;">eval/scripts/DocVQA/score_analyze_all.py</td>
  </tr>
  
  <!-- AI2D -->
  <tr>
    <td style="border: 1px solid white; padding: 8px;">14</td>
    <td style="border: 1px solid white; padding: 8px;">AI2D</td>
    <td style="border: 1px solid white; padding: 8px;">eval/datasets/AI2D</td>
    <td style="border: 1px solid white; padding: 8px;">eval/inference/ai2diagram_test.py</td>
    <td style="border: 1px solid white; padding: 8px;">eval/scripts/AI2D/score_analyze_all.py</td>
  </tr>
  
  <!-- MathVista -->
  <tr>
    <td style="border: 1px solid white; padding: 8px;">15</td>
    <td style="border: 1px solid white; padding: 8px;">MathVista</td>
    <td style="border: 1px solid white; padding: 8px;">eval/datasets/MathVista</td>
    <td style="border: 1px solid white; padding: 8px;">eval/inference/yuanvl_eval_mathvista.py</td>
    <td style="border: 1px solid white; padding: 8px;">eval/scripts/MathVista/calculate_score_detailed_nokey.py</td>
  </tr>
</table>

*注：ChatRAG、MMTab以及SummEval评测集只是示例，全量的数据请去各自的项目中下载*


# Example

### 以AIME2024任务为例，推理脚本如下
```bash
enable_thinking=True MAX_TOKENS=16384 MODEL_PATH=/your/model/path python eval/scripts/openai_api3_eval.py --input eval/datasets/aime-2024/2024_II_cle_001.txt --out_file_name result/aime-2024/2024_II_cle_001.txt --port 8001
```
```bash
参数释义：
enable_thinking： 控制模型是否以思考模式输出，取值范围：[True, False]
MAX_TOKENS：生成文本最大长度
MODEL_PATH：模型的绝对路径
input：待评测文件
out_file_name：结果文件
port：模型服务端口，根据vllm或sglang服务端启动时设置的端口来决定这里的赋值
```

### 以AIME2024任务为例，打分脚本如下
```bash
python script/judge_gsm_choice_easy.py --input_path result/aime-2024/2024_II_cle_001.txt --output_path test_out/ --origin_path eval/datasets/aime-2024/AIME_2024_I_cle.txt
```


