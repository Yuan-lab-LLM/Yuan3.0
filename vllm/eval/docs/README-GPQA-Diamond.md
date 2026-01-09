# GPQA-Diamond Evaluation Operation Guide

## 1. Environment Configuration

Standard environment setup is sufficient—no special dependencies or configurations are required.

---

## 2. Input Data Processing

### Step 1: Combine Input Files (if necessary)

- If your input data consists of **a single file**, skip this step.
- If you have **multiple input files**, combine them into one using the `cat` command. Ensure that filenames do not conflict.

**Example:**
```bash
cat result/GPQA-Diamond/gpqa_diamond_shuffle* > result/GPQA-Diamond/gpqa_diamond_shuffle_all.txt
```

### Step 2: Prepare File Lists

In the directory containing the evaluation script (`eval/scripts/GPQA-Diamond/`), create two files:

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
      <td>Absolute path(s) to your combined input data file(s), one per line.<br>Example: <code>/full/path/to/result/GPQA-Diamond/gpqa_diamond_shuffle_all.txt</code></td>
    </tr>
    <tr>
      <td><code>files_origin_your.txt</code></td>
      <td>Absolute path to the original reference data file:<br><code>eval/datasets/GPQA-Diamond/gpqa_diamond_shuffle_001.txt</code></td>
    </tr>
  </tbody>
</table>

> Both files must reside in `eval/scripts/GPQA-Diamond/`.

---

## 3. Run Evaluation Script

### Step 1: Create Output Directory  
Create a directory to store evaluation results, for example:
```bash
mkdir -p eval/eval_output/GPQA-Diamond
```

### Step 2: Configure Output Path  
Open `eval/scripts/GPQA-Diamond/eval_easy_lzy.sh` and modify the `OUTPUT_PATH` variable to point to your desired output directory:
```bash
OUTPUT_PATH="eval/eval_output/GPQA-Diamond"
```

### Step 3: Execute the Script  
Navigate to the script’s directory and run it:
```bash
cd eval/scripts/GPQA-Diamond/
bash eval_easy_lzy.sh
```

---

## 4. View Evaluation Results

After execution, go to your output directory:
```bash
cd eval/eval_output/GPQA-Diamond/
```

- If your input file is named `gpqa_diamond_shuffle_all.txt`, inspect the last line of:
  ```
  result_gpqa_diamond_shuffle_all_judge/result.txt
  ```

- In general, for an input file named `xxx.txt`, check:
  ```
  result_xxx_judge/result.txt
  ```

Look for a line containing:
```text
accuracy: 96.6667%
```

The numeric value (e.g., `96.6667`) is your final GPQA-Diamond evaluation score.

---