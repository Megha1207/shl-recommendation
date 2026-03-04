import sys
import os

# Allow imports when running from api/
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI
from pydantic import BaseModel

from retrieval.hybrid import hybrid_search
from retrieval.rerank import balance_and_rerank
from llm.query_understanding import understand_query


app = FastAPI(title="SHL Assessment Recommendation API")


# -----------------------------
# Request Model
# -----------------------------
class QueryRequest(BaseModel):
    query: str
    top_k: int = 10


# -----------------------------
# Health Check
# -----------------------------
@app.get("/health")
def health():
    return {"status": "healthy"}


# -----------------------------
# Recommendation Endpoint
# -----------------------------
@app.post("/recommend")
def recommend(request: QueryRequest):

    user_query = request.query.strip()
    top_k = int(request.top_k)

    # 1️⃣ Query understanding
    structured = understand_query(user_query)
    expanded_query = structured.get("expanded_query", user_query)

    # 2️⃣ Hybrid retrieval
    candidates = hybrid_search(expanded_query, top_k=50)

    # 3️⃣ Reranking
    final_results = balance_and_rerank(
        candidates,
        query=expanded_query,
        top_k=top_k
    )

    # Ensure only serializable fields are returned
    clean_results = []
    for r in final_results:
        clean_results.append({
            "name": r.get("name"),
            "url": r.get("url"),
            "description": r.get("description"),
            "duration": r.get("duration"),
            "test_types": r.get("test_types")
        })

    return {
        "query": user_query,
        "structured_query": structured,
        "recommended_assessments": clean_results
    }