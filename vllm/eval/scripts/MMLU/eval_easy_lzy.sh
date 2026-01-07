# /bin/bash

declare -a FILES_LST=()
mapfile -t FILES_LST < files_eval_your.txt
echo "First FILES_LST: ${FILES_LST[0]}"


declare -a ORIGIN_FILES=()
mapfile -t ORIGIN_FILES < files_origin_your.txt
echo "First ORIGIN_FILES: ${ORIGIN_FILES[0]}"



CODA_PATH=your/path
SCRIPT_PATH=judge_gsm_choice_easy.py
OUTPUT_PATH=your/output/path
mkdir -p $OUTPUT_PATH

NUM_FILES=${#FILES_LST[@]}
for ((i=0; i<NUM_FILES; i++)); do
    INPUT_FILE="${FILES_LST[$i]}"
    ORIGIN_FILE="${ORIGIN_FILES[$i]}"
    file_name=$(basename $INPUT_FILE .txt)
    #folder_name=$(basename $(dirname $INPUT_FILE))
    #返回倒数第二级目录
    folder_name=$(basename $(dirname $(dirname $INPUT_FILE)))
    #返回倒数第三级目录
    #folder_name=$(basename $(dirname $(dirname $(dirname $INPUT_FILE))))    
    output_file_name=${folder_name}_${file_name}_judge
    mkdir -p $OUTPUT_PATH/$output_file_name

    cd $CODA_PATH && python $SCRIPT_PATH --input_path $INPUT_FILE --output_path $OUTPUT_PATH/$output_file_name --origin_path $ORIGIN_FILE 2>&1 | tee $OUTPUT_PATH/$output_file_name/result.txt
    
done



