# MMLU Evaluation Operation Guide

## 1. Environment Configuration

Standard environment setup is sufficient—no special dependencies or configurations are required.

---

## 2. Input Data Processing

### Step 1: Combine Input Files (if necessary)

- If your input data consists of **a single file**, skip this step.
- If you have **multiple input files**, combine them into one using the `cat` command. Ensure that filenames do not conflict.

**Example:**
```bash
cat result/MMLU/mmlu_test_qa* > result/MMLU/mmlu_test_qa_all.txt
```

### Step 2: Prepare File Lists

In the directory containing the evaluation script (`eval/scripts/MMLU/`), create two files:

- `files_eval_your.txt`
- `files_origin_your.txt`

Populate them as follows:

<table>
  <thead>
    <tr>
      <th>File</th>
      <th>Content</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>files_eval_your.txt</code></td>
      <td>Absolute path(s) to your combined input data file(s), one per line.<br>Example: <code>/full/path/to/result/MMLU/mmlu_test_qa_all.txt</code></td>
    </tr>
    <tr>
      <td><code>files_origin_your.txt</code></td>
      <td>Absolute path to the original reference data file:<br><code>eval/datasets/MMLU/mmlu_test_qa.txt</code></td>
    </tr>
  </tbody>
</table>

> Both files must reside in `eval/scripts/MMLU/`.

---

## 3. Run Evaluation Script

### Step 1: Create Output Directory  
Create a directory to store evaluation results, for example:
```bash
mkdir -p eval/eval_output/MMLU
```

### Step 2: Configure Output Path  
Open `eval/scripts/MMLU/eval_easy_lzy.sh` and modify the `OUTPUT_PATH` variable to point to your desired output directory:
```bash
OUTPUT_PATH="eval/eval_output/MMLU"
```

### Step 3: Execute the Script  
Navigate to the script’s directory and run it:
```bash
cd eval/scripts/MMLU/
bash eval_easy_lzy.sh
```

---

## 4. View Evaluation Results

After execution, go to your output directory:
```bash
cd eval/eval_output/MMLU/
```

- If your input file is named `mmlu_test_qa_all.txt`, inspect the last line of:
  ```
  result_mmlu_test_qa_all_judge/result.txt
  ```

- In general, for an input file named `xxx.txt`, check:
  ```
  result_xxx_judge/result.txt
  ```

Look for a line containing:
```text
accuracy: 96.6667%
```

The numeric value (e.g., `96.6667`) is your final MMLU evaluation score.

---