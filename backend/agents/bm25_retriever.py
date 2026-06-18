import logging
import os
import time

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

from backend.agents.state import AgentState

load_dotenv(override=True)

# BM25 wins on OOV tokens — entity names, product names, version
# numbers — where dense vectors underperform.

ELASTICSEARCH_URL = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")
INDEX = "medical_topics"
TOP_K = 10

log = logging.getLogger(__name__)

_es: Elasticsearch | None = None


def _get_es() -> Elasticsearch:
    global _es
    if _es is None:
        _es = Elasticsearch(ELASTICSEARCH_URL)
    return _es


def run(state: AgentState) -> AgentState:
    query = state["query"]
    t0 = time.perf_counter()

    es = _get_es()

    resp = es.search(
        index=INDEX,
        query={
            "multi_match": {
                "query": query,
                "fields": ["title^2", "synonyms^1.5", "body"],
            }
        },
        size=TOP_K,
    )

    hits = resp["hits"]["hits"]
    retrieved_docs = [
        {
            "text":     hit["_source"].get("body", ""),
            "score":    float(hit["_score"]),
            "source":   "medlineplus",
            "chunk_id": hit["_id"],
        }
        for hit in hits
    ]

    latency_ms = dict(state.get("latency_ms") or {})
    latency_ms["bm25_retriever"] = round((time.perf_counter() - t0) * 1000, 1)

    log.info(
        "BM25Retriever: query=%r docs=%d top_score=%.4f latency=%.1fms",
        query,
        len(retrieved_docs),
        retrieved_docs[0]["score"] if retrieved_docs else 0.0,
        latency_ms["bm25_retriever"],
    )

    return {**state, "retrieved_docs": retrieved_docs, "latency_ms": latency_ms}
