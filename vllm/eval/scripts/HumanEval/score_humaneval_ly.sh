# /bin/bash

declare -a FILES_LST=()
mapfile -t FILES_LST < files_eval.txt
# mapfile -t FILES_LST < rag_files.txt
echo "First FILES_LST: ${FILES_LST[0]}"

BASE_DIR=./
SCRIPT_PATH=code_solution_process_mjw_ly.py

OUTPUT_PATH=/your/output/path

NUM_FILES=${#FILES_LST[@]}
for ((i=0; i<NUM_FILES; i++)); do
    INPUT_FILE="${FILES_LST[$i]}"
    file_name=$(basename $INPUT_FILE .txt)

    folder_name=$(basename $(dirname $(dirname $INPUT_FILE)))
    CASENAME=${folder_name}/${file_name}_judge
    mkdir -p $OUTPUT_PATH/$CASENAME
    cd $BASE_DIR && python $SCRIPT_PATH --input-path  $INPUT_FILE  --case-name  $CASENAME   --output-path  $OUTPUT_PATH/$CASENAME
    evaluate_functional_correctness -p $BASE_DIR/datasets/HUMANEVAL/HumanEval.jsonl.gz   $OUTPUT_PATH/$CASENAME/samples.jsonl  2>&1 | tee  $OUTPUT_PATH/$CASENAME/result.txt

done

