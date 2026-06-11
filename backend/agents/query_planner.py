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
- SEMANTIC: Questions about concepts, explanations, descriptions, comparisons, or anything requiring contextual understanding
- KEYWORD: Factual lookups about specific named entities — who, when, where, which person
- STRUCTURED: Queries requiring aggregation, counting, ranking, or computations over structured data

Examples:
Q: What are the health benefits of green tea?
Strategy: SEMANTIC

Q: How does photosynthesis work?
Strategy: SEMANTIC

Q: What is machine learning?
Strategy: SEMANTIC

Q: Who is the CEO of Apple?
Strategy: KEYWORD

Q: When was the Eiffel Tower built?
Strategy: KEYWORD

Q: Where is the Louvre museum located?
Strategy: KEYWORD

Q: How many tracks are in the database?
Strategy: STRUCTURED

Q: What is the total revenue by genre?
Strategy: STRUCTURED

Q: Show the top 5 customers by total purchase amount?
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
