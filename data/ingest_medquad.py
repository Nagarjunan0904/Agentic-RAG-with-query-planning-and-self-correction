"""
Ingest MedQuAD medical Q&A pairs into Pinecone (vectors) and Postgres (chunks table).
Source: keivalya/MedQuad-MedicalQnADataset (HuggingFace, ~16 k rows, no auth needed)

Embedding strategy: each row is embedded as "Q: <question>\\nA: <answer>", which gives
the passage vector rich semantic context from both the question intent and the answer
content. Long Q+A pairs are split at 512 tokens with 50-token overlap (same splitter as
the previous MS MARCO ingest) so the chunks table schema stays identical.

input_type="search_query" is used at both ingest and query time (symmetric space), which
avoids the asymmetric-embedding mismatch that existed in the prior MS MARCO pipeline.
"""

import logging
import os
import time
from pathlib import Path

import cohere
import psycopg2
from datasets import load_dataset
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
COHERE_API_KEY   = os.environ["COHERE_API_KEY"]
DATABASE_URL     = os.environ["DATABASE_URL"]

INDEX_NAME      = os.environ.get("PINECONE_INDEX", "agentic-rag")
HF_DATASET      = "keivalya/MedQuad-MedicalQnADataset"
EMBED_BATCH     = 48
LOG_EVERY       = 500
CHECKPOINT_FILE = Path(__file__).parent / "medquad_checkpoint.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _read_checkpoint() -> int:
    try:
        return int(CHECKPOINT_FILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        return 0


def _write_checkpoint(rows_done: int) -> None:
    CHECKPOINT_FILE.write_text(str(rows_done))


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512,
        chunk_overlap=50,
    )


def _ensure_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id          SERIAL PRIMARY KEY,
                source      TEXT,
                passage_id  TEXT,
                chunk_idx   INT,
                text        TEXT,
                created_at  TIMESTAMP DEFAULT NOW()
            )
            """
        )
    conn.commit()


def _wipe_existing(index, conn) -> None:
    """Delete all Pinecone vectors and stale chunk rows before a fresh ingest."""
    log.info("Wiping Pinecone index '%s' — deleting all existing vectors...", INDEX_NAME)
    index.delete(delete_all=True)
    log.info("Pinecone wipe complete — all vectors removed from index '%s'.", INDEX_NAME)

    # Remove any prior medquad or msmarco rows so the chunks table stays consistent
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE source IN ('msmarco', 'medquad')")
        deleted = cur.rowcount
    conn.commit()
    log.info("Removed %d stale rows from the chunks table.", deleted)

    # Brief pause so the serverless index has time to settle before we start upserting
    time.sleep(3)


def _embed_and_store(co, index, conn, texts: list[str], meta: list[dict]) -> None:
    for attempt in range(5):
        try:
            resp = co.embed(
                texts=texts,
                model="embed-english-v3.0",
                input_type="search_query",  # symmetric: same type used at query time
            )
            break
        except Exception as exc:
            status  = getattr(exc, "status_code", None)
            exc_str = str(exc)
            is_429  = status == 429 or "429" in exc_str or "too many requests" in exc_str.lower()
            wait    = 60 if is_429 else 10
            error_type = "429 rate-limit" if is_429 else f"502/ApiError ({type(exc).__name__})"
            if attempt == 4:
                log.error("Embed failed after 5 attempts (%s): %s", error_type, exc)
                raise
            log.warning("Retry %d/5 after %s – waiting %ds...", attempt + 1, error_type, wait)
            time.sleep(wait)

    embeddings = resp.embeddings
    time.sleep(1.5)  # stay within Cohere free-tier rate limits

    vectors = [
        {
            "id": f"{m['passage_id']}_{m['chunk_idx']}",
            "values": emb,
            "metadata": {
                "source":     "medquad",
                "passage_id": str(m["passage_id"]),
                "chunk_idx":  m["chunk_idx"],
            },
        }
        for m, emb in zip(meta, embeddings)
    ]
    index.upsert(vectors=vectors)

    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (source, passage_id, chunk_idx, text) VALUES (%s, %s, %s, %s)",
            [("medquad", m["passage_id"], m["chunk_idx"], m["text"]) for m in meta],
        )
    conn.commit()


def main() -> None:
    co    = cohere.Client(api_key=COHERE_API_KEY)
    index = Pinecone(api_key=PINECONE_API_KEY).Index(INDEX_NAME)
    conn  = psycopg2.connect(DATABASE_URL)
    _ensure_table(conn)

    splitter  = _splitter()
    start_row = _read_checkpoint()

    if start_row == 0:
        _wipe_existing(index, conn)
    else:
        log.info("Resuming from row %d (checkpoint found) — skipping wipe.", start_row)

    log.info("Loading MedQuAD dataset: %s", HF_DATASET)
    ds = load_dataset(HF_DATASET, split="train")

    cols    = ds.column_names
    col_map = {c.lower(): c for c in cols}
    q_col   = col_map.get("question")
    a_col   = col_map.get("answer")
    if not q_col or not a_col:
        raise ValueError(f"Cannot find Question/Answer columns. Got: {cols}")

    log.info(
        "Columns: %s | question='%s'  answer='%s' | total rows: %d",
        cols, q_col, a_col, len(ds),
    )

    pending_texts: list[str] = []
    pending_meta:  list[dict] = []
    total     = 0
    rows_done = start_row

    for row_idx, row in enumerate(ds):
        if row_idx < start_row:
            continue

        question = (row.get(q_col) or "").strip()
        answer   = (row.get(a_col) or "").strip()
        if not answer:
            rows_done += 1
            continue

        # Prepend the question so every chunk carries the semantic intent of the Q+A pair
        combined = f"Q: {question}\nA: {answer}" if question else answer

        for chunk_idx, chunk_text in enumerate(splitter.split_text(combined)):
            pending_texts.append(chunk_text)
            pending_meta.append(
                {"passage_id": str(row_idx), "chunk_idx": chunk_idx, "text": chunk_text}
            )

        while len(pending_texts) >= EMBED_BATCH:
            batch_texts = pending_texts[:EMBED_BATCH]
            batch_meta  = pending_meta[:EMBED_BATCH]
            del pending_texts[:EMBED_BATCH]
            del pending_meta[:EMBED_BATCH]

            _embed_and_store(co, index, conn, batch_texts, batch_meta)
            _write_checkpoint(rows_done)

            prev, total = total, total + len(batch_texts)
            if total // LOG_EVERY > prev // LOG_EVERY:
                log.info(
                    "Upserted %d vectors (row %d / %d)...",
                    total, rows_done, len(ds),
                )

        rows_done += 1

    # flush tail batch
    if pending_texts:
        _embed_and_store(co, index, conn, pending_texts, pending_meta)
        _write_checkpoint(rows_done)
        total += len(pending_texts)

    conn.close()
    log.info("Done. Total vectors upserted: %d from %d MedQuAD rows.", total, rows_done)
    CHECKPOINT_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
