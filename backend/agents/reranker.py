import logging
import os
import time

import cohere
from dotenv import load_dotenv

from backend.agents.state import AgentState

load_dotenv(override=True)

COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")

log = logging.getLogger(__name__)

_co: cohere.Client | None = None


def _get_co() -> cohere.Client:
    global _co
    if _co is None:
        _co = cohere.Client(api_key=COHERE_API_KEY)
    return _co


def run(state: AgentState) -> AgentState:
    retrieved_docs = state.get("retrieved_docs") or []
    t0 = time.perf_counter()

    if not retrieved_docs:
        latency_ms = dict(state.get("latency_ms") or {})
        latency_ms["reranker"] = round((time.perf_counter() - t0) * 1000, 1)
        return {**state, "reranked_docs": [], "latency_ms": latency_ms}

    query = state["query"]
    co = _get_co()

    results = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=[doc["text"] for doc in retrieved_docs],
        top_n=3,
    )

    reranked_docs = [
        {**retrieved_docs[r.index], "rerank_score": r.relevance_score}
        for r in results.results
    ]

    latency_ms = dict(state.get("latency_ms") or {})
    latency_ms["reranker"] = round((time.perf_counter() - t0) * 1000, 1)

    log.info(
        "Reranker: query=%r input=%d output=%d top_score=%.4f latency=%.1fms",
        query,
        len(retrieved_docs),
        len(reranked_docs),
        reranked_docs[0]["rerank_score"] if reranked_docs else 0.0,
        latency_ms["reranker"],
    )

    return {**state, "reranked_docs": reranked_docs, "latency_ms": latency_ms}
