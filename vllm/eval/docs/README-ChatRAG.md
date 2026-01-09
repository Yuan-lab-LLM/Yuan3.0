# ChatRAG Evaluation Guide

## 1. Data Preparation

Merge input files that belong to the same dataset into a single file using the following script:

```bash
for file in *_aa.txt; do
    if [[ -f "$file" ]]; then
        prefix="${file%_aa.txt}"
        echo "Merging files for dataset: $prefix"
        ls "${prefix}"_*.txt 2>/dev/null | sort | xargs cat > "${prefix}.txt"
    fi
done
echo "All datasets merged!"
```

This combines all split parts (e.g., `dataset_aa.txt`, `dataset_ab.txt`, ...) into one consolidated file per dataset (e.g., `dataset.txt`).

---

## 2. Modify and Run the Scoring Script

Edit the scoring Bash script (`scripts/ChatRAG/eval_chatqa.sh`) to configure the following parameters:

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>--base-path</code></td>
      <td>Path to the directory containing your result files.</td>
    </tr>
    <tr>
      <td><code>--output</code></td>
      <td>Output Excel file in the format: <code>filename.xlsx:sheetname</code>.</td>
    </tr>
    <tr>
      <td><code>--row</code></td>
      <td>Specifies which row in the Excel sheet to write the results to.</td>
    </tr>
  </tbody>
</table>

After updating the paths and sheet names, run the script:
```bash
bash scripts/ChatRAG/eval_chatqa.sh
```

---

## 3. View Scoring Results

Use the following Python command to inspect the generated Excel file:

```python
python3 -c "
import pandas as pd
file_path = 'scripts/ChatRAG/ChatQA/test1.xlsx'
sheet_name = '1230'
try:
    xl = pd.ExcelFile(file_path)
    print('All worksheets in the file:', xl.sheet_names)
    if sheet_name in xl.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f'\nContent of worksheet \"{sheet_name}\" (first 10 rows):')
        print(df.head(10))
        print(f'\nWorksheet shape: {df.shape}')
    else:
        print(f'Error: Worksheet \"{sheet_name}\" does not exist')
        print('Available worksheets:', xl.sheet_names)
except Exception as e:
    print(f'Error: {e}')
"
```

### Result Format Explanation

Each written row contains the following columns:

```
f1  Avg. (10 Results)  Avg. (6 Results)  D2D  QuAC  QReCC  CoQA  doqa_cooking  doqa_movies  doqa_travel  CPQA  SQA  TCQA  Hdial  INSCIT
```

These need to be post-processed into the final format:

```
f1  Avg. (10 Results)  Avg. (6 Results)  D2D  QuAC  QReCC  CoQA  DOQA  CPQA  SQA  TCQA  Hdial  INSCIT
```

Where:

- **DOQA** = average of `doqa_cooking`, `doqa_movies`, and `doqa_travel`
- **Avg. (10 Results)** = average across 10 datasets **after** combining the three DOQA sub-datasets into one
- **Avg. (6 Results)** = average of the following six datasets:  
  `D2D`, `CoQA`, `DOQA`, `TCQA`, `Hdial`, `INSCIT`

> Ensure that the Excel column headers exactly match the expected dataset names as shown above.

---