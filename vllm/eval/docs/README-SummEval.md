# Summary Scoring：
## File Location：
    eval/scripts/SummEval
## Input File Location：
    eval/scripts/SummEval/data
## Environment Requirements：
    transformers：4.35.2 
    huggingface-hub: 0.36.0

###  Step 1：
```bash
cd eval/scripts/SummEval
```
Main required packages：rouge_score、bert_score、summac、openpyxl 
### If you encounter NLTK issues, run
```bash
python nltk.py  # Install required NLTK packages
```
### Step 2：rouge、bert、summac score
#### Rouge score: ~2 minuts
```bash
python3 eval_rouge.py 
        --path data 
        --excel-file outputs/test.xlsx:summ1106 
        --write-row 2
```

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Description</th>
      <th>Location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>--path</code></td>
      <td>Input path</td>
      <td> data</td>
    </tr>
    <tr>
      <td><code>--excel-file</code></td>
      <td>result files</td>
      <td>outputs/test.xlsx:summ1106</td>
    </tr>
  </tbody>
</table>

How to view results： 
1. View the final scores of each part directly in the terminal (recommended)
2. Recorded in rouge/test.xlsx:summ1106
####  bert socre：~5minuts
```bash
CUDA_VISIBLE_DEVICES=0 python3 eval_bert.py 
                              --path data 
                              --output outputs
```

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Description</th>
      <th>Location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>CUDA_VISIBLE_DEVICES</code></td>
      <td>GPUs（0~7）</td>
    </tr>
    <tr>
      <td><code>--path</code></td>
      <td>result files</td>
      <td> data</td>
    </tr>
<tr>
      <td><code>output</code></td>
      <td>Input path</td>
      <td>outputs/bert.txt</td>
    </tr>
  </tbody>
</table>

How to view results： 
1. View the final scores of each part directly in the terminal (recommended)
2. Recorded in outputs/bert.txt

#### summac score：Time varies
Typically the longest for the pubmed-summarization task, around 3 hours.
Multi-GPU parallel scoring is recommended.
```bash
CUDA_VISIBLE_DEVICES=0 python3 eval_summac.py 
                              --path data 
                              --output outputs
```
<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Description</th>
      <th>Location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>CUDA_VISIBLE_DEVICES</code></td>
      <td>GPUs（0~7）</td>
    </tr>
    <tr>
      <td><code>--path</code></td>
      <td>result files</td>
      <td> data</td>
    </tr>
<tr>
      <td><code>output</code></td>
      <td>Input path</td>
      <td>outputs/summac.txt</td>
    </tr>
  </tbody>
</table>

##### eval_summac.py:Can be switched according to task requirements
```bash
eval_summac.py    #Scoring for all datasets(xsum、big_patent_10_percent、cnn_dailymail、billsum、pubmed-summarization、samsum、wikihow)
```

Scoring for a specific dataset (multi-GPU execution to improve efficiency):
```bash
eval_summac_xsum.py 
eval_summac_big.py  
eval_summac_cnn.py 
eval_summac_billsum.py 
eval_summac_pub.py  
eval_summac_samsum.py 
eval_summac_wiki.py
```
How to view results： 
1. View the final scores of each part directly in the terminal (recommended)
2. Recorded in outputs/summac.txt