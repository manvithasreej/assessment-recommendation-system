from fastapi import FastAPI
from pydantic import BaseModel
from model_utils import recommend_assessments

app = FastAPI()

# ---------- HEALTH ENDPOINT ----------
@app.get("/health")
def health_check():
    return {"status": "ok"}

# ---------- REQUEST SCHEMA ----------
class RecommendRequest(BaseModel):
    query: str

# ---------- RECOMMEND ENDPOINT ----------
@app.post("/recommend")
def recommend(request: RecommendRequest):
    results = recommend_assessments(request.query)
    return {
        "recommendations": results
    }
