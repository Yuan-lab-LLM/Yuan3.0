We conducted evaluations on the Yuan3.0 Flash model across multiple evaluation datasets, and the detailed summary is as follows:
<table style="border-collapse: collapse;">
  <tr>
    <th style="border: 1px solid white; padding: 8px;">Index</th>
    <th style="border: 1px solid white; padding: 8px;">Benchmark Name</th>
    <th style="border: 1px solid white; padding: 8px;">Data Path</th>
    <th style="border: 1px solid white; padding: 8px;">Reasoning Script</th>
    <th style="border: 1px solid white; padding: 8px;">Evaluation Script</th>
    <th style="border: 1px solid white; padding: 8px;">Evaluation Guide</th>
  </tr>
  
  <!-- ChatRAG -->
  <tr>
    <td style="border: 1px solid white; padding: 8px;">1</td>
    <td style="border: 1px solid white; padding: 8px;">ChatRAG</td>
    <td style="border: 1px solid white; padding: 8px;">eval/datasets/ChatRAG</td>
    <td style="border: 1px solid white; padding: 8px;">eval/inference/openai_api3_eval.py</td>
    <td style="border: 1px solid white; padding: 8px;">eval/scripts/ChatRAG/eval_chatqa.sh</td>
    <td style="border: 1px solid white; padding: 8px;">eval/docs/README-ChatRAG.md</td>
  </tr>
  
  <!-- MMTab -->
  <tr>
    <td style="border: 1px solid white; padding: 8px;">2</td>
    <td style="border: 1px solid white; padding: 8px;">MMTab</td>
    <td style="border: 1px solid white; padding: 8px;">eval/datasets/MMTab</td>
    <td style="border: 1px solid white; padding: 8px;">eval/inference/yuanvl_api_multi_eval.py</td>
    <td style="border: 1px solid white; padding: 8px;">eval/scripts/MMTab/table.sh</td>
    <td style="border: 1px solid white; padding: 8px;">eval/docs/README-MMTAB.md</td>
  </tr>
  
  <!-- SummEval -->
  <tr>
    <td style="border: 1px solid white; padding: 8px;">3</td>
    <td style="border: 1px solid white; padding: 8px;">SummEval</td>
    <td style="border: 1px solid white; padding: 8px;">eval/datasets/SummEval</td>
    <td style="border: 1px solid white; padding: 8px;">eval/inference/openai_api3_table.py</td>
    <td style="border: 1px solid white; padding: 8px;">eval/scripts/Summary/*.py</td>
    <td style="border: 1px solid white; padding: 8px;">eval/docs/README-SummEval.md</td>
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
    <td style="border: 1px solid white; padding: 8px;">eval/docs/README-BFCL.md</td>
  </tr>
  
  <!-- MATH-500 -->
  <tr>
    <td style="border: 1px solid white; padding: 8px;">5</td>
    <td style="border: 1px solid white; padding: 8px;">MATH-500</td>
    <td style="border: 1px solid white; padding: 8px;">eval/datasets/MATH-500</td>
    <td style="border: 1px solid white; padding: 8px;">eval/inference/openai_api3_eval.py</td>
    <td style="border: 1px solid white; padding: 8px;">eval/scripts/MATH-500/judge_with_vllm_model_math.py</td>
    <td style="border: 1px solid white; padding: 8px;">eval/docs/README-MATH-500.md</td>
  </tr>
  
  <!-- AIME2024 -->
  <tr>
    <td style="border: 1px solid white; padding: 8px;">6</td>
    <td style="border: 1px solid white; padding: 8px;">AIME2024</td>
    <td style="border: 1px solid white; padding: 8px;">eval/datasets/AIME2024</td>
    <td style="border: 1px solid white; padding: 8px;">eval/inference/openai_api3_eval.py</td>
    <td style="border: 1px solid white; padding: 8px;">eval/scripts/AIME2024/eval_easy_lzy.sh</td>
    <td style="border: 1px solid white; padding: 8px;">eval/docs/README-AIME2024.md</td>
  </tr>
  
  <!-- livecodebench -->
  <tr>
    <td style="border: 1px solid white; padding: 8px;">7</td>
    <td style="border: 1px solid white; padding: 8px;">livecodebench</td>
    <td style="border: 1px solid white; padding: 8px;">eval/datasets/livecodebench</td>
    <td style="border: 1px solid white; padding: 8px;">eval/inference/openai_api3_livecode.py</td>
    <td style="border: 1px solid white; padding: 8px;">eval/scripts/livecodebench/eval_livecodebench_v6.sh</td>
    <td style="border: 1px solid white; padding: 8px;">eval/docs/README-livecodebench.md</td>
  </tr>
  
  <!-- HumanEval -->
  <tr>
    <td style="border: 1px solid white; padding: 8px;">8</td>
    <td style="border: 1px solid white; padding: 8px;">HumanEval</td>
    <td style="border: 1px solid white; padding: 8px;">eval/datasets/HumanEval</td>
    <td style="border: 1px solid white; padding: 8px;">eval/inference/openai_api3_eval.py</td>
    <td style="border: 1px solid white; padding: 8px;">eval/scripts/HumanEval/score_humaneval_ly.sh</td>
    <td style="border: 1px solid white; padding: 8px;">eval/docs/README-HumanEval.md</td>
  </tr>
  
  <!-- MMLU -->
  <tr>
    <td style="border: 1px solid white; padding: 8px;">9</td>
    <td style="border: 1px solid white; padding: 8px;">MMLU</td>
    <td style="border: 1px solid white; padding: 8px;">eval/datasets/MMLU</td>
    <td style="border: 1px solid white; padding: 8px;">eval/inference/openai_api3_eval.py</td>
    <td style="border: 1px solid white; padding: 8px;">eval/scripts/MMLU/eval_easy_lzy.sh</td>
    <td style="border: 1px solid white; padding: 8px;">eval/docs/README-MMLU.md</td>
  </tr>
  
  <!-- MMLU Pro -->
  <tr>
    <td style="border: 1px solid white; padding: 8px;">10</td>
    <td style="border: 1px solid white; padding: 8px;">MMLU Pro</td>
    <td style="border: 1px solid white; padding: 8px;">eval/datasets/MMLU-Pro</td>
    <td style="border: 1px solid white; padding: 8px;">eval/inference/openai_api3_eval.py</td>
    <td style="border: 1px solid white; padding: 8px;">eval/scripts/MMLU-Pro/eval_easy_lzy.sh</td>
    <td style="border: 1px solid white; padding: 8px;">eval/docs/README-MMLU-Pro.md</td>
  </tr>
  
  <!-- GPQA-Diamond -->
  <tr>
    <td style="border: 1px solid white; padding: 8px;">11</td>
    <td style="border: 1px solid white; padding: 8px;">GPQA-Diamond</td>
    <td style="border: 1px solid white; padding: 8px;">eval/datasets/GPQA-Diamond</td>
    <td style="border: 1px solid white; padding: 8px;">eval/inference/openai_api3_eval.py</td>
    <td style="border: 1px solid white; padding: 8px;">eval/scripts/GPQA-Diamond/eval_easy_lzy.sh</td>
    <td style="border: 1px solid white; padding: 8px;">eval/docs/README-GPQA-Diamond.md</td>
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
    <td style="border: 1px solid white; padding: 8px;">eval/docs/README-ChartQA.md</td>
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
    <td style="border: 1px solid white; padding: 8px;">eval/docs/README-DocVQA.md</td>
  </tr>
  
  <!-- AI2D -->
  <tr>
    <td style="border: 1px solid white; padding: 8px;">14</td>
    <td style="border: 1px solid white; padding: 8px;">AI2D</td>
    <td style="border: 1px solid white; padding: 8px;">eval/datasets/AI2D</td>
    <td style="border: 1px solid white; padding: 8px;">eval/inference/ai2diagram_test.py</td>
    <td style="border: 1px solid white; padding: 8px;">eval/scripts/AI2D/score_analyze_all.py</td>
    <td style="border: 1px solid white; padding: 8px;">eval/docs/README-AI2D.md</td>
  </tr>
  
  <!-- MathVista -->
  <tr>
    <td style="border: 1px solid white; padding: 8px;">15</td>
    <td style="border: 1px solid white; padding: 8px;">MathVista</td>
    <td style="border: 1px solid white; padding: 8px;">eval/datasets/MathVista</td>
    <td style="border: 1px solid white; padding: 8px;">eval/inference/yuanvl_eval_mathvista.py</td>
    <td style="border: 1px solid white; padding: 8px;">eval/scripts/MathVista/calculate_score_detailed_nokey.py</td>
    <td style="border: 1px solid white; padding: 8px;">eval/docs/README-MathVista.md</td>
  </tr>
</table>


*Note: The ChatRAG, MMTab, and SummEval evaluation sets are only subsets of the full datasets. For the complete datasets, please download them from their respective projects*


# Example

Taking the AIME2024 task as an example, the inference script is as follows:
```bash
enable_thinking=True MAX_TOKENS=16384 MODEL_PATH=/your/model/path python eval/scripts/openai_api3_eval.py --input eval/datasets/aime-2024/2024_II_cle_001.txt --out_file_name result/aime-2024/2024_II_cle_001.txt --port 8001
```
```bash
Parameter interpretation:
enable_thinking: whether the model outputs by thinking mode, with a value range of:[True, False]
MAX_TOKENS: maximum length of generated text
MODEL_PATH: absolute path of the model
input: file to be evaluated
out_file_name: file of results
port: the model service port is determined when the vllm server is started
```

Taking the AIME2024 task as an example, the evaluation script is as follows:
```bash
python script/judge_gsm_choice_easy.py --input_path result/aime-2024/2024_II_cle_001.txt --output_path test_out/ --origin_path eval/datasets/aime-2024/AIME_2024_I_cle.txt
```

