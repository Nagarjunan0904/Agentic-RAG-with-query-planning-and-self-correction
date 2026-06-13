"""
RAGAS evaluation pipeline for the Agentic RAG system.

Usage:
    python -m backend.evaluation.ragas_pipeline                      # all 100 questions
    python -m backend.evaluation.ragas_pipeline --limit 10           # first 10 (smoke-test)
    python -m backend.evaluation.ragas_pipeline --offset 66 --limit 34  # STRUCTURED only (Q67-100)
    python -m backend.evaluation.ragas_pipeline --refresh            # force re-run RAG inference
"""

import argparse
import asyncio
import json
import logging
import os
import time
import warnings
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

warnings.filterwarnings("ignore")  # suppress NumPy 1.x/2.x compat warnings
load_dotenv(override=True)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
COHERE_API_KEY    = os.environ.get("COHERE_API_KEY", "")

ROOT                = Path(__file__).parent.parent.parent
EVAL_QUESTIONS_PATH = ROOT / "data" / "eval_questions.json"
RAG_CACHE_PATH      = ROOT / "data" / "rag_inference_cache.json"
RAGAS_RESULTS_PATH  = ROOT / "data" / "ragas_results.json"

METRIC_NAMES  = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
JUDGE_MODEL   = "claude-haiku-4-5"
MAX_RETRIES   = 3
RETRY_BASE_S  = 5  # seconds; doubles each attempt (5 → 10 → 20)

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Cohere embeddings — avoids the broken langchain_community import chain
# ---------------------------------------------------------------------------

from langchain_core.embeddings import Embeddings  # noqa: E402


class _CohereEmbeddings(Embeddings):
    """Thin Cohere SDK wrapper implementing the LangChain Embeddings interface."""

    def __init__(self, api_key: str, model: str = "embed-english-v3.0") -> None:
        import cohere
        self._co    = cohere.Client(api_key=api_key)
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Use search_query for both methods: RAGAS compares question-to-question
        # (generated hypotheticals vs original query), so both sides must share
        # the same embedding space. search_document vs search_query produces
        # near-zero cosine similarity and breaks answer_relevancy scoring.
        resp = self._co.embed(texts=texts, model=self._model, input_type="search_query")
        return resp.embeddings

    def embed_query(self, text: str) -> list[float]:
        resp = self._co.embed(texts=[text], model=self._model, input_type="search_query")
        return resp.embeddings[0]


# ---------------------------------------------------------------------------
# RAG inference
# ---------------------------------------------------------------------------

async def _run_one(q: dict) -> dict:
    from backend.graph import rag_graph  # deferred to avoid circular import at module level

    init_state = {
        "query":          q["question"],
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
    result = await rag_graph.ainvoke(init_state)
    contexts = [d["text"] for d in result.get("reranked_docs", []) if d.get("text")]
    return {
        "id":           q["id"],
        "category":     q["category"],
        "question":     q["question"],
        "answer":       result.get("answer", ""),
        "contexts":     contexts or ["no context"],
        "ground_truth": q["ground_truth"],
    }


async def collect_rag_rows(questions: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for q in tqdm(questions, desc="RAG inference", unit="q"):
        try:
            row = await _run_one(q)
        except Exception as exc:
            log.warning("Question %d failed: %s", q["id"], exc)
            row = {
                "id":           q["id"],
                "category":     q["category"],
                "question":     q["question"],
                "answer":       "",
                "contexts":     ["no context"],
                "ground_truth": q["ground_truth"],
            }
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Inference cache
# ---------------------------------------------------------------------------

def _load_cache(question_ids: list[int]) -> list[dict] | None:
    """Return cached rows if they exist and cover exactly the requested IDs."""
    if not RAG_CACHE_PATH.exists():
        return None
    try:
        cache = json.loads(RAG_CACHE_PATH.read_text())
        if cache.get("question_ids") == question_ids:
            print(f"Cache hit — loaded {len(cache['rows'])} rows from {RAG_CACHE_PATH.name}")
            return cache["rows"]
    except Exception as exc:
        log.warning("Cache load failed (%s), ignoring.", exc)
    return None


def _save_cache(question_ids: list[int], rows: list[dict]) -> None:
    payload = {"question_ids": question_ids, "rows": rows}
    RAG_CACHE_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Inference cache saved -> {RAG_CACHE_PATH.name}")


# ---------------------------------------------------------------------------
# RAGAS evaluation (synchronous — run via asyncio.to_thread)
# ---------------------------------------------------------------------------

def run_ragas(rows: list[dict]) -> dict:
    import math
    from langchain_anthropic import ChatAnthropic
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    print(f"\nRunning RAGAS evaluation on {len(rows)} samples (judge: {JUDGE_MODEL}) ...")

    samples = [
        SingleTurnSample(
            user_input=row["question"],
            response=row["answer"],
            retrieved_contexts=row["contexts"],
            reference=row["ground_truth"],
        )
        for row in rows
    ]
    dataset = EvaluationDataset(samples=samples)

    llm = ChatAnthropic(model=JUDGE_MODEL, api_key=ANTHROPIC_API_KEY)
    emb = _CohereEmbeddings(api_key=COHERE_API_KEY)

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_emb = LangchainEmbeddingsWrapper(emb)

    try:
        from ragas.metrics import (Faithfulness, AnswerRelevancy,
                                   ContextRecall, ContextPrecision)
        metrics = [
            Faithfulness(llm=ragas_llm),
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
            ContextRecall(llm=ragas_llm),
            ContextPrecision(llm=ragas_llm),
        ]
    except (ImportError, TypeError):
        # RAGAS < 0.2 or missing class exports — mutate the module-level singletons
        from ragas.metrics import (faithfulness, answer_relevancy,
                                   context_recall, context_precision)
        faithfulness.llm            = ragas_llm
        answer_relevancy.llm        = ragas_llm
        answer_relevancy.embeddings = ragas_emb
        context_recall.llm          = ragas_llm
        context_precision.llm       = ragas_llm
        metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

    # Retry loop — handles credit exhaustion and transient rate limits
    result = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                show_progress=True,
                raise_exceptions=False,
            )
            break
        except Exception as exc:
            name = type(exc).__name__
            msg  = str(exc).lower()
            retriable = (
                "BadRequest"  in name or
                "RateLimit"   in name or
                "Timeout"     in name or
                "credit"      in msg  or
                "rate"        in msg  or
                "timeout"     in msg
            )
            if retriable and attempt < MAX_RETRIES:
                wait = RETRY_BASE_S * (2 ** attempt)
                print(f"Judge call failed ({name}), retrying in {wait}s "
                      f"(attempt {attempt + 1}/{MAX_RETRIES}) ...")
                time.sleep(wait)
            else:
                raise

    df = result.to_pandas()

    def _safe_mean(col: str):
        if col not in df.columns:
            return None
        v = df[col].mean(skipna=True)
        return None if math.isnan(v) else round(float(v), 4)

    def _safe_val(col: str, idx: int):
        if col not in df.columns:
            return None
        v = df.iloc[idx][col]
        try:
            f = float(v)
            return None if math.isnan(f) else round(f, 4)
        except (TypeError, ValueError):
            return None

    # Overall metric averages
    per_metric = {m: _safe_mean(m) for m in METRIC_NAMES}

    # Per-question scores
    per_question = []
    for i, row in enumerate(rows):
        scores = {m: _safe_val(m, i) for m in METRIC_NAMES}
        per_question.append({
            "id":       row["id"],
            "category": row["category"],
            "question": row["question"],
            "scores":   scores,
        })

    # Category breakdown
    category_breakdown = {}
    for cat in ("SEMANTIC", "KEYWORD", "STRUCTURED"):
        cat_pq = [pq for pq in per_question if pq["category"] == cat]
        if not cat_pq:
            continue
        cat_bd = {}
        for m in METRIC_NAMES:
            vals = [pq["scores"][m] for pq in cat_pq if pq["scores"].get(m) is not None]
            cat_bd[m] = round(sum(vals) / len(vals), 4) if vals else None
        category_breakdown[cat] = cat_bd

    return {
        "per_metric":         per_metric,
        "per_question":       per_question,
        "category_breakdown": category_breakdown,
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    return f"{v:.4f}" if v is not None else " n/a "


def print_summary(output: dict) -> None:
    pm = output["per_metric"]
    cb = output["category_breakdown"]

    print("\n## RAGAS Evaluation Results\n")
    print("### Overall\n")
    print(f"{'Metric':<25} {'Score':>7}")
    print("-" * 34)
    for m in METRIC_NAMES:
        label = m.replace("_", " ").title()
        print(f"{label:<25} {_fmt(pm.get(m)):>7}")

    print("\n### Per Category\n")
    header = f"{'Category':<12}" + "".join(f" {m.replace('_',' ').title():>18}" for m in METRIC_NAMES)
    print(header)
    print("-" * len(header))
    for cat in ("SEMANTIC", "KEYWORD", "STRUCTURED"):
        bd = cb.get(cat, {})
        row_str = f"{cat:<12}" + "".join(f" {_fmt(bd.get(m)):>18}" for m in METRIC_NAMES)
        print(row_str)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS evaluation pipeline")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip the first N questions before applying --limit")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate up to N questions after --offset (omit for all remaining)")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-run RAG inference even if cache exists")
    args = parser.parse_args()

    with open(EVAL_QUESTIONS_PATH) as f:
        questions: list[dict] = json.load(f)

    questions = questions[args.offset:]
    if args.limit:
        questions = questions[: args.limit]

    print(f"Loaded {len(questions)} questions from {EVAL_QUESTIONS_PATH.name}")

    # Phase 1 — RAG inference (with cache)
    q_ids = [q["id"] for q in questions]
    rows  = None if args.refresh else _load_cache(q_ids)

    if rows is None:
        rows = await collect_rag_rows(questions)
        _save_cache(q_ids, rows)

    # Phase 2 — synchronous RAGAS judge (run in thread to avoid nested asyncio.run)
    output = await asyncio.to_thread(run_ragas, rows)

    # Save
    RAGAS_RESULTS_PATH.write_text(json.dumps(output, indent=2))
    print(f"Results saved -> {RAGAS_RESULTS_PATH}")

    # Print summary
    print_summary(output)


if __name__ == "__main__":
    asyncio.run(main())
