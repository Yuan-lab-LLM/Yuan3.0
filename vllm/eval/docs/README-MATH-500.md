# MATH-500 Evaluation Operation Guide

## 1. Environment Configuration

<table>
  <tbody>
    <tr>
      <td>Required Dependencies</td>
      <td>
        <ul>
          <li>The <code>vllm</code> Python module must be installed.</li>
          <li>A CUDA-enabled environment with <strong>two GPU cards</strong> is required.</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

---

## 2. Input Data Processing

### Step 1: Combine Input Files (if necessary)

- If your input data consists of **a single file**, skip this step.
- If you have **multiple input files**, combine them into one using the `cat` command. Ensure filenames do not conflict.

**Example:**
```bash
cat result/MATH-500/HuggingFaceH4_MATH-500_standard* > result/MATH-500_all/HuggingFaceH4_MATH-500_standard_all.txt
```

---

## 3. Run Evaluation Script

### Step 1: Create Output Directory  
Create a directory to store evaluation results, for example:
```bash
mkdir -p eval/eval_output/MATH-500
```

### Step 2: Configure Scoring Model Path  
Open `eval/scripts/MATH-500/judge_with_vllm_model_math.py` and update the scoring model path to your desired model location.

### Step 3: Execute the Evaluation Script  
Navigate to the script’s directory and run the following command:

```bash
cd eval/scripts/MATH-500/

GPU_NUMS=2 \
MAX_TOKENS=4096 \
BATCH_SIZE=480 \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
CUDA_VISIBLE_DEVICES=0,1 \
python judge_with_vllm_model_math.py \
  --input_path result/MATH-500_all/ \
  --output_path eval/eval_output/MATH-500 \
  --origin_path eval/datasets/MATH-500/HuggingFaceH4_MATH-500_standard_001.txt
```

### Parameter Reference

<table>
  <thead>
    <tr>
      <th>Parameter / Env Var</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>GPU_NUMS=2</code></td>
      <td>Number of GPUs to use for evaluation.</td>
    </tr>
    <tr>
      <td><code>MAX_TOKENS=4096</code></td>
      <td>Maximum number of tokens allowed per model inference.</td>
    </tr>
    <tr>
      <td><code>BATCH_SIZE=480</code></td>
      <td>Batch size for processing inputs.</td>
    </tr>
    <tr>
      <td><code>PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python</code></td>
      <td>Forces use of the pure-Python Protobuf implementation (avoids C++ extension issues).</td>
    </tr>
    <tr>
      <td><code>VLLM_WORKER_MULTIPROC_METHOD=spawn</code></td>
      <td>Sets multiprocessing start method for vLLM workers.</td>
    </tr>
    <tr>
      <td><code>CUDA_VISIBLE_DEVICES=0,1</code></td>
      <td>Specifies which GPU devices to use (adjust as needed).</td>
    </tr>
    <tr>
      <td><code>--input_path</code></td>
      <td>Directory containing your combined prediction files.</td>
    </tr>
    <tr>
      <td><code>--output_path</code></td>
      <td>Directory where evaluation results will be saved.</td>
    </tr>
    <tr>
      <td><code>--origin_path</code></td>
      <td>Absolute path to the original ground-truth file:<br><code>eval/datasets/MATH-500/HuggingFaceH4_MATH-500_standard_001.txt</code></td>
    </tr>
  </tbody>
</table>

---

## 4. View Evaluation Results

After execution completes, the script will print evaluation metrics directly to the terminal.

Look for the **`accuracy`** value in the output. For example:
```text
accuracy: 96.6667%
```

The numeric value (e.g., `96.6667`) is your final MATH-500 evaluation score.

---