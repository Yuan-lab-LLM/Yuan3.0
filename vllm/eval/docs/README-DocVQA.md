# DocVQA Evaluation Pipeline

## Inference

### Step 1: Prepare Directory
Create the scoring script directory:
```bash
mkdir -p eval/scripts/exec_score
```

### Step 2: Run Inference Scripts

#### Short Reasoning
```bash
python eval/inference/chat_docvqa_oneword_chat.py \
  --batch_num 64 \
  --round 1 \
  --topk 1 \
  --topp 0.0000001 \
  --temperature 0.000001 \
  --output_dir /your/output/path \
  --max_tokens 8192 \
  --case_name test \
  --model_path /your/model/path \
  --port 8001
```

#### Long Reasoning
```bash
python eval/inference/chat_docvqa_val_long.py \
  --batch_num 64 \
  --round 1 \
  --topk 1 \
  --topp 0.0000001 \
  --temperature 0.000001 \
  --output_dir /your/output/path \
  --max_tokens 16384 \
  --case_name test \
  --model_path /your/model/path \
  --port 8001
```

> Both scripts generate:
> - Inference results (saved to `--output_dir`)
> - A scoring command script (saved in `eval/scripts/exec_score/`)  
>   The filename is a UUID (e.g., `669125dc-b61d-4845-a41d-9d2987f3f5da.sh`)

---

## Parameter Reference

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Description</th>
      <th>Recommended Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>--batch_num</code></td>
      <td>Inference batch concurrency</td>
      <td><code>64</code></td>
    </tr>
    <tr>
      <td><code>--round</code></td>
      <td>Number of passes over the dataset</td>
      <td><code>1</code></td>
    </tr>
    <tr>
      <td><code>--topk</code></td>
      <td>Top-k sampling</td>
      <td><code>1</code></td>
    </tr>
    <tr>
      <td><code>--topp</code></td>
      <td>Top-p (nucleus) sampling</td>
      <td><code>0.0000001</code></td>
    </tr>
    <tr>
      <td><code>--temperature</code></td>
      <td>Sampling temperature</td>
      <td><code>0.000001</code></td>
    </tr>
    <tr>
      <td><code>--output_dir</code></td>
      <td>Output directory for inference results</td>
      <td>e.g., <code>/your/output/path</code></td>
    </tr>
    <tr>
      <td><code>--max_tokens</code></td>
      <td>Maximum number of output tokens</td>
      <td><code>8192</code> (short), <code>16384</code> (long)</td>
    </tr>
    <tr>
      <td><code>--case_name</code></td>
      <td>Test case name</td>
      <td>e.g., <code>test</code></td>
    </tr>
    <tr>
      <td><code>--model_path</code></td>
      <td>Path to the model</td>
      <td>e.g., <code>/your/model/path</code></td>
    </tr>
    <tr>
      <td><code>--port</code></td>
      <td>Port number of the running model service</td>
      <td>e.g., <code>8001</code></td>
    </tr>
  </tbody>
</table>

---

## Scoring

### Step 1: Execute the Scoring Script
Run the auto-generated scoring script (replace with your actual UUID filename):
```bash
bash eval/scripts/exec_score/669125dc-b61d-4845-a41d-9d2987f3f5da.sh
```

> The filename is randomly generated using `uuid4`—use the one actually created during inference.

### Step 2: View the Evaluation Score

- **For short reasoning**, inspect the score file:
  ```bash
  head /your/output/path/*docvqa_val-short-oneword-*score.json
  ```

- **For long reasoning**, inspect the score file:
  ```bash
  head /your/output/path/*docvqa_val-short_cot-*score.json
  ```

In either case, the evaluation metric is reported under the key:
```json
"DocVQA"
```

This value is your final DocVQA evaluation score.

---