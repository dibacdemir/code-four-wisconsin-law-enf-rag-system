import os
import sys

# Add src to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from retrieval.vector_store import hybrid_search
from generation.llm_client import get_llm_response
import logging

app = FastAPI(title="Wisconsin Legal Chat RAG")

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class QueryRequest(BaseModel):
    question: str
    doc_type_filter: str | None = None  # optional: "statute", "case_law"

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()


class QueryResponse(BaseModel):
    answer: str
    sources: list
    confidence: float = 0.0


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        # Build metadata filter if doc_type specified
        where_filter = None
        if request.doc_type_filter:
            where_filter = {"doc_type": request.doc_type_filter}

        # Retrieve relevant chunks
        results = hybrid_search(
            request.question,
            n_results=5,
            where_filter=where_filter,
        )

        if not results or not results.get("documents"):
            raise HTTPException(status_code=404, detail="No relevant documents found for this query.")

        # Generate response
        response = get_llm_response(request.question, results)

        logger.info("Query: %s | Sources returned: %d", request.question, len(response["sources"]))
        return QueryResponse(
            answer=response["answer"],
            sources=[{k: str(v) for k, v in s.items()} for s in response["sources"]],
            confidence=response.get("confidence", results.get("confidence", 0.0)),

        )

    except HTTPException:
        raise  # re-raise intentional HTTP errors as-is
    except Exception as e:
        logger.error("Query failed: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)