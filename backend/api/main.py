import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone
from pydantic import BaseModel

from backend.agents.sql_agent import SafetyError
from backend.graph import rag_graph

load_dotenv(override=True)

DATABASE_URL = os.environ.get("DATABASE_URL", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
ELASTICSEARCH_URL = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")

log = logging.getLogger(__name__)

RAGAS_RESULTS_PATH = Path(__file__).parent.parent.parent / "data" / "ragas_results.json"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    strategy: str
    confidence: float
    retry_count: int
    reranked_docs: list[dict]
    latency_ms: dict


class FeedbackRequest(BaseModel):
    query_id: str
    thumbs_up: bool


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def _init_feedback_table() -> None:
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id         SERIAL PRIMARY KEY,
                query_id   TEXT      NOT NULL,
                thumbs_up  BOOL      NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
    conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_feedback_table()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Agentic RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    init_state = {
        "query":          req.question,
        "strategy":       "",
        "retrieved_docs": [],
        "reranked_docs":  [],
        "answer":         "",
        "confidence":     0.0,
        "retry_count":    0,
        "retry_reason":   None,
        "query_vector":   None,
        "latency_ms":     {},
    }
    try:
        result = await rag_graph.ainvoke(init_state)
    except SafetyError as exc:
        raise HTTPException(status_code=422, detail=f"Safety check failed: {exc}")
    except Exception as exc:
        log.exception("Unhandled error during graph invocation")
        raise HTTPException(status_code=500, detail=str(exc))

    return QueryResponse(
        answer=result.get("answer", ""),
        strategy=result.get("strategy", ""),
        confidence=result.get("confidence", 0.0),
        retry_count=result.get("retry_count", 0),
        reranked_docs=result.get("reranked_docs", []),
        latency_ms=result.get("latency_ms", {}),
    )


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feedback (query_id, thumbs_up) VALUES (%s, %s)",
                (req.query_id, req.thumbs_up),
            )
    finally:
        conn.close()
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    if not RAGAS_RESULTS_PATH.exists():
        return {}
    return json.loads(RAGAS_RESULTS_PATH.read_text())


@app.get("/health")
async def health():
    status: dict[str, str] = {}

    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pc.list_indexes()
        status["pinecone"] = "ok"
    except Exception as exc:
        log.warning("Pinecone health check failed: %s", exc)
        status["pinecone"] = "error"

    try:
        es = Elasticsearch(ELASTICSEARCH_URL)
        es.cluster.health()
        status["elasticsearch"] = "ok"
    except Exception as exc:
        log.warning("Elasticsearch health check failed: %s", exc)
        status["elasticsearch"] = "error"

    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.close()
        status["postgres"] = "ok"
    except Exception as exc:
        log.warning("Postgres health check failed: %s", exc)
        status["postgres"] = "error"

    return status
