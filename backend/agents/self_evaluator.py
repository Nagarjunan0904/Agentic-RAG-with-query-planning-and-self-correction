import logging
import os
import time
from typing import List

import anthropic
import cohere
import instructor
from dotenv import load_dotenv
from pydantic import BaseModel

from backend.agents.state import AgentState

load_dotenv(override=True)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")
MODEL = "claude-sonnet-4-5"
CONFIDENCE_THRESHOLD = 0.65
MAX_RETRIES = 2

log = logging.getLogger(__name__)

_anthropic_client: anthropic.Anthropic | None = None
_co: cohere.Client | None = None


def _get_anthropic() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _anthropic_client


def _get_co() -> cohere.Client:
    global _co
    if _co is None:
        _co = cohere.Client(api_key=COHERE_API_KEY)
    return _co


class EvalResult(BaseModel):
    score: int  # 1-5
    reasoning: str
    unsupported_claims: List[str]


def compute_confidence(rerank_scores: list, answer_score: int) -> float:
    retrieval_conf = sum(rerank_scores[:3]) / (3 * max(rerank_scores[:3], default=1))
    answer_conf = (answer_score - 1) / 4
    return 0.4 * retrieval_conf + 0.6 * answer_conf


def _evaluate_answer(context: str, answer: str) -> EvalResult:
    client = instructor.from_anthropic(_get_anthropic())
    prompt = (
        f"Given the retrieved context and the generated answer below,\n"
        f"rate the answer on a scale of 1-5 for factual support from the context.\n\n"
        f"Context: {context}\n\n"
        f"Answer: {answer}\n\n"
        f'Return JSON only: {{"score": 1-5, "reasoning": "...", "unsupported_claims": ["..."]}}\n'
        f"Score meaning: 1=completely unsupported, 3=partially supported, 5=fully supported"
    )
    return client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
        response_model=EvalResult,
    )


def _generate_hypothetical_doc(query: str) -> str:
    message = _get_anthropic().messages.create(
        model=MODEL,
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"Write a short factual paragraph answering: {query}",
        }],
    )
    return message.content[0].text.strip()


def run(state: AgentState) -> AgentState:
    reranked_docs = state.get("reranked_docs") or []
    answer = state.get("answer", "")
    query = state["query"]
    t0 = time.perf_counter()

    context = "\n".join(doc["text"] for doc in reranked_docs)

    eval_result = _evaluate_answer(context, answer)
    rerank_scores = [doc.get("rerank_score", 0.0) for doc in reranked_docs]
    confidence = compute_confidence(rerank_scores, eval_result.score)

    log.info(
        "Evaluator: score=%d confidence=%.4f reasoning=%r",
        eval_result.score,
        confidence,
        eval_result.reasoning,
    )

    updates: dict = {"confidence": confidence}

    retry_count = state.get("retry_count") or 0
    if confidence < CONFIDENCE_THRESHOLD and retry_count < MAX_RETRIES:
        hyp_doc = _generate_hypothetical_doc(query)
        embed_resp = _get_co().embed(
            texts=[hyp_doc],
            model="embed-english-v3.0",
            input_type="search_document",
        )
        updates["query_vector"] = embed_resp.embeddings[0]
        updates["retry_count"] = retry_count + 1
        updates["retry_reason"] = (
            f"Low confidence ({confidence:.2f}): {eval_result.reasoning}"
        )
        log.info(
            "Evaluator: HyDE retry %d — %s",
            retry_count + 1,
            updates["retry_reason"],
        )

    latency_ms = dict(state.get("latency_ms") or {})
    latency_ms["evaluator"] = round((time.perf_counter() - t0) * 1000, 1)
    updates["latency_ms"] = latency_ms

    return {**state, **updates}
