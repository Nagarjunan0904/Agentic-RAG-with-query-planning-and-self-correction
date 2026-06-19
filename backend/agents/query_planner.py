import logging
import os
import time
from typing import Literal

import anthropic
import instructor
from dotenv import load_dotenv
from pydantic import BaseModel

from backend.agents.state import AgentState

load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-5"

log = logging.getLogger(__name__)

PROMPT = """Classify the user query into exactly one of three retrieval strategies:

- SEMANTIC: Use when the user wants an explanation, mechanism, comparison, or multi-faceted answer about a medical topic — e.g. what causes X, how does Y work, what are the symptoms of Z, how is X treated, what is the difference between X and Y. The answer requires synthesising across multiple sources and cannot be satisfied by a single reference entry.

- KEYWORD: Use when the user is looking up a single named medical term, condition, drug, or test — even when phrased as "What is X?" — and the answer is the canonical reference summary for that exact term. Bare-term queries (a condition or drug name with no verb) are always KEYWORD. The key signal is that the user wants the encyclopaedia/glossary entry for one specific thing, not an explanation that requires reasoning or synthesis.

- STRUCTURED: Use when the query requires aggregation, counting, ranking, or filtering over structured patient or clinical records — e.g. "how many", "most common", "average", "top N", queries about the patient database.

Examples:
Q: What is hypertension?
Strategy: KEYWORD

Q: hypertension
Strategy: KEYWORD

Q: asthma treatment guidelines
Strategy: KEYWORD

Q: What is A1C?
Strategy: KEYWORD

Q: What are the symptoms of type 2 diabetes and how does it differ from type 1?
Strategy: SEMANTIC

Q: How is Alzheimer's disease treated?
Strategy: SEMANTIC

Q: What causes chronic kidney disease?
Strategy: SEMANTIC

Q: How many patients are in the database?
Strategy: STRUCTURED

Q: What is the most common condition among patients?
Strategy: STRUCTURED

Q: What is the most prescribed medication?
Strategy: STRUCTURED

Now classify this query:
Q: {query}
Strategy:"""


class ClassificationResult(BaseModel):
    strategy: Literal['SEMANTIC', 'KEYWORD', 'STRUCTURED']
    reasoning: str


_STRUCTURED_TERMS = {"count", "sum", "average", "total", "top", "highest", "lowest", "how many"}
_KEYWORD_PREFIXES = ("who ", "when ", "where ", "which person", "who is")


def _heuristic(query: str) -> str:
    q = query.lower().strip()
    if any(term in q for term in _STRUCTURED_TERMS):
        return "STRUCTURED"
    if q.startswith(_KEYWORD_PREFIXES):
        return "KEYWORD"
    return "SEMANTIC"


def run(state: AgentState) -> AgentState:
    query = state["query"]
    t0 = time.perf_counter()
    strategy: str

    try:
        client = instructor.from_anthropic(anthropic.Anthropic(api_key=ANTHROPIC_API_KEY))
        result: ClassificationResult = client.messages.create(
            model=MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": PROMPT.format(query=query)}],
            response_model=ClassificationResult,
        )
        strategy = result.strategy
        log.info("Planner [Claude]: query=%r strategy=%s reasoning=%s", query, strategy, result.reasoning)
    except Exception as exc:
        log.warning("Claude call failed (%s); falling back to heuristic.", exc)
        strategy = _heuristic(query)
        log.info("Planner [heuristic]: query=%r strategy=%s", query, strategy)

    latency_ms = dict(state.get("latency_ms") or {})
    latency_ms["planner"] = round((time.perf_counter() - t0) * 1000, 1)

    return {**state, "strategy": strategy, "latency_ms": latency_ms}
