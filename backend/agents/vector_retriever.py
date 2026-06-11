import logging
import os
import time

import cohere
import psycopg2
from dotenv import load_dotenv
from pinecone import Pinecone

from backend.agents.state import AgentState

load_dotenv(override=True)

COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "agentic-rag")
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# Index was built with HNSW ef_construction=128, m=16 (Pinecone serverless default pod settings)
TOP_K = 10

log = logging.getLogger(__name__)

_co: cohere.Client | None = None
_index = None
_pg_conn = None


def _get_clients():
    global _co, _index
    if _co is None:
        _co = cohere.Client(api_key=COHERE_API_KEY)
    if _index is None:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _index = pc.Index(PINECONE_INDEX)
    return _co, _index


def _get_pg_conn():
    global _pg_conn
    if _pg_conn is None or _pg_conn.closed:
        _pg_conn = psycopg2.connect(DATABASE_URL)
        _pg_conn.autocommit = True
    return _pg_conn


def _fetch_texts(matches) -> dict[tuple, str]:
    """Batch-fetch text from chunks table for all Pinecone matches.

    Returns a dict keyed by (passage_id, chunk_idx) -> text.
    """
    pairs = tuple(
        (str(m.metadata.get("passage_id", "")), int(m.metadata.get("chunk_idx", 0)))
        for m in matches
        if m.metadata.get("passage_id") is not None
    )
    if not pairs:
        return {}

    conn = _get_pg_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT passage_id, chunk_idx, text FROM chunks WHERE (passage_id, chunk_idx) IN %s",
            (pairs,),
        )
        return {(row[0], int(row[1])): row[2] for row in cur.fetchall()}


def run(state: AgentState) -> AgentState:
    query = state["query"]
    t0 = time.perf_counter()

    co, index = _get_clients()

    embed_resp = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query",
    )
    query_vector = embed_resp.embeddings[0]

    result = index.query(
        vector=query_vector,
        top_k=TOP_K,
        filter={"source": "msmarco"},
        include_metadata=True,
    )

    text_map = _fetch_texts(result.matches)

    retrieved_docs = [
        {
            "text": text_map.get(
                (str(m.metadata.get("passage_id", "")), int(m.metadata.get("chunk_idx", 0))),
                "",
            ),
            "score":    float(m.score),
            "source":   "msmarco",
            "chunk_id": m.id,
        }
        for m in result.matches
    ]

    latency_ms = dict(state.get("latency_ms") or {})
    latency_ms["vector_retriever"] = round((time.perf_counter() - t0) * 1000, 1)

    log.info(
        "VectorRetriever: query=%r docs=%d top_score=%.4f latency=%.1fms",
        query,
        len(retrieved_docs),
        retrieved_docs[0]["score"] if retrieved_docs else 0.0,
        latency_ms["vector_retriever"],
    )

    return {**state, "retrieved_docs": retrieved_docs, "latency_ms": latency_ms}
