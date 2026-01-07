# /bin/bash

declare -a FILES_LST=()
mapfile -t FILES_LST < files_eval.txt
# mapfile -t FILES_LST < rag_files.txt
echo "First FILES_LST: ${FILES_LST[0]}"

BASE_DIR=./
OUTPUT_DIR=/your/output/dir

NUM_FILES=${#FILES_LST[@]}
for ((i=0; i<NUM_FILES; i++)); do
    INPUT_FILE="${FILES_LST[$i]}"
    basename=$(basename $INPUT_FILE)

    folder_name=$(basename $(dirname $(dirname $(dirname $INPUT_FILE))))
    output_file_name=${folder_name}_judge/${basename}_judge
    OUTPUT_PATH=$OUTPUT_DIR/$output_file_name
    mkdir -p $OUTPUT_PATH
    cd $BASE_DIR && python -m lcb_runner.runner.score_main_txt_2 --model Yuan_M32_generate_model --scenario codegeneration --evaluate --release_version release_v6 --n 1 --input_file $INPUT_FILE --output_path $OUTPUT_PATH 2>&1 | tee -a $OUTPUT_PATH/score_$basename.log &
    if (( i % 4 == 0 )); then
        wait  # 等待所有后台进程结束
    fi
done



