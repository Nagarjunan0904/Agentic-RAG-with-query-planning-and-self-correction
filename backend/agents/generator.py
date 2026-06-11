import logging
import os
import time

import anthropic
from dotenv import load_dotenv

from backend.agents.state import AgentState

load_dotenv(override=True)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-5"

log = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def run(state: AgentState) -> AgentState:
    query = state["query"]
    reranked_docs = state.get("reranked_docs") or []
    t0 = time.perf_counter()

    context = "\n".join(
        f"[{doc['source']}] {doc['text']}" for doc in reranked_docs
    )

    prompt = (
        f"Answer the question based only on the context below.\n"
        f"Be concise and factual. If the context doesn't contain "
        f"enough information, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

    message = _get_client().messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = message.content[0].text.strip()

    latency_ms = dict(state.get("latency_ms") or {})
    latency_ms["generator"] = round((time.perf_counter() - t0) * 1000, 1)

    log.info(
        "Generator: query=%r docs=%d latency=%.1fms",
        query,
        len(reranked_docs),
        latency_ms["generator"],
    )

    return {**state, "answer": answer, "latency_ms": latency_ms}
