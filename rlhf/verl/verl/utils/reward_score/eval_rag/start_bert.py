from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bert_score import BERTScorer

# 初始化FastAPI应用
app = FastAPI(title="BERTScore Summary Evaluator")

# 初始化BERTScorer（只初始化一次以提高性能）
try:
    bert_scorer = BERTScorer(lang="en", device="cuda")  # 使用CUDA加速
except Exception as e:
    # 如果CUDA不可用，回退到CPU
    bert_scorer = BERTScorer(lang="en", device="cpu")
    print(f"Using CPU instead of CUDA: {str(e)}")

# 定义请求体模型
class ScoreRequest(BaseModel):
    answer: str       # 原始文本
    summary: str      # 待评估的摘要

# 定义响应模型
class ScoreResponse(BaseModel):
    bert_score: float # BERTScore的F1分数
    message: str      # 响应消息

# 定义评分端点
@app.post("/compute-bert-score", response_model=ScoreResponse)
async def compute_bert_score(request: ScoreRequest):
    try:
        # 检查输入是否为空
        if not request.answer or not request.summary:
            raise HTTPException(status_code=400, detail="Answer and summary cannot be empty")
        
        # 计算BERTScore
        P, R, F1 = bert_scorer.score([request.answer], [request.summary])
        f1_score = F1.mean().item()
        
        return {
            "bert_score": round(f1_score, 4),
            "message": "BERTScore computed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing BERTScore: {str(e)}")

# 根路径测试端点
@app.get("/")
async def root():
    return {"message": "BERTScore Summary Evaluator API is running. Use POST /compute-bert-score to get scores."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8019)

# uvicorn main:app --host 0.0.0.0 --port 8018
