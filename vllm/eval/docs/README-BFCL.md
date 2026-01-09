#### Requirements

git pull https://github.com/ShishirPatil/gorilla/tree/cd9429ccf3d4d04156affe883c495b3b047e6b64

```
cd gorilla/berkeley-function-call-leaderboard

pip install -e .
```

Modify the source code

```
gorilla/berkeley-function-call-leaderboard/bfcl_eval/model_handler/api_inference/qwen.py

gorilla/berkeley-function-call-leaderboard/bfcl_eval/model_handler/api_inference/openai_completion.py
```

Replace the base_url with the specified API service.

#### Inference

```
bfcl generate --model model_path --test-category all --num-threads 16 --include-input-log 
```

Replace `model_path` with a Qwen series model (e.g., `qwen3-8b`) and save the results to `gorilla/berkeley-function-call-leaderboard/result`

#### Score

```
python3 -m bfcl_eval.eval_checker.eval_runner --model model_path --test-category all --result-dir result
```

Replace `model_path` with a Qwen series model (e.g., `qwen3-8b`) and save the results to `gorilla/berkeley-function-call-leaderboard/score`.

The scores are aggregated in `gorilla/berkeley-function-call-leaderboard/score/data_overall.csv`.