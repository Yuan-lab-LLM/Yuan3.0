# ChatRAG
## 一、Data Preparation
Merge the same data in the input into a single file:
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
## 二、Modify and run the scoring bash script
You only need to modify the contents in the bash file, including the paths and table names.
--base-path  result file
--output The Excel file to write to, format: filename:sheetname
--row Specifies which row to write

Run the bash script after making the modifications：
```bash scripts/ChatRAG/eval_chatqa.sh```

## 三、View scoring results
Run the code below to view the Excel spreadsheet：
```bash
python3 -c "
import pandas as pd
file_path = 'scripts/ChatRAG/ChatQA/test1.xlsx'
sheet_name = '1230'
try:
    xl = pd.ExcelFile(file_path)
    print('All worksheets in the file:', xl.sheet_names)
    if sheet_name in xl.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f'\nworksheets \"{sheet_name}\" The first 10 lines of content:')
        print(df.head(10))
        print(f'\nworksheets shape: {df.shape}')
    else:
        print(f'error: worksheets \"{sheet_name}\" Does not exist')
        print('Available worksheets:', xl.sheet_names)
except Exception as e:
    print(f'error: {e}')
"
```
Each line written is as follows:
f1 Avg. (10 Results) Avg. (6 Results) D2D  QuAC  QReCC  CoQA  doqa_cooking  doqa_movies  doqa_travel  CPQA  SQA  TCQA  Hdial    INSCIT
Needs to be processed into:
f1 Avg. (10 Results) Avg. (6 Results) D2D  QuAC  QReCC  CoQA  DOQA  CPQA  SQA  TCQA  Hdial    INSCIT
DOQA is the average of doqa_cooking, doqa_movies, and doqa_travel
10 avg results are the average values of 10 datasets after combining doqa
6 avg results are the average values of d2d, coqa, doqa, tcqa, hdial, and inscit
The Excel headers should correspond exactly to the dataset in the screenshot.