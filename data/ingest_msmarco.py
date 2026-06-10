"""
Ingest MS MARCO passages into Pinecone (vectors) and Postgres (chunks table).

Extra deps not in requirements.txt:
    pip install ir-datasets tiktoken
"""

import logging
import os
import time
from itertools import islice
from pathlib import Path

import cohere
import psycopg2
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
COHERE_API_KEY = os.environ["COHERE_API_KEY"]
DATABASE_URL = os.environ["DATABASE_URL"]

INDEX_NAME = "agentic-rag"
TOP_K = 50_000
EMBED_BATCH = 48
LOG_EVERY = 1_000
CHECKPOINT_FILE = Path(__file__).parent / "ingest_checkpoint.txt"

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


def _write_checkpoint(passages_done: int) -> None:
    CHECKPOINT_FILE.write_text(str(passages_done))


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


def _embed_and_store(co, index, conn, texts: list[str], meta: list[dict]) -> None:
    for attempt in range(5):
        try:
            resp = co.embed(
                texts=texts,
                model="embed-english-v3.0",
                input_type="search_document",
            )
            break
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            exc_str = str(exc)
            is_429 = status == 429 or "429" in exc_str or "too many requests" in exc_str.lower()
            wait = 60 if is_429 else 10
            error_type = "429 rate-limit" if is_429 else f"502/ApiError ({type(exc).__name__})"
            if attempt == 4:
                log.error("Embed failed after 5 attempts (%s): %s", error_type, exc)
                raise
            log.warning("Retry %d/5 after %s – waiting %ds...", attempt + 1, error_type, wait)
            time.sleep(wait)

    embeddings = resp.embeddings
    time.sleep(1.5)

    vectors = [
        {
            "id": f"{m['passage_id']}_{m['chunk_idx']}",
            "values": emb,
            "metadata": {
                "source": "msmarco",
                "passage_id": str(m["passage_id"]),
                "chunk_idx": m["chunk_idx"],
            },
        }
        for m, emb in zip(meta, embeddings)
    ]
    index.upsert(vectors=vectors)

    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (source, passage_id, chunk_idx, text) VALUES (%s, %s, %s, %s)",
            [("msmarco", m["passage_id"], m["chunk_idx"], m["text"]) for m in meta],
        )
    conn.commit()


def main() -> None:
    import ir_datasets  # imported here so the module is usable without it installed

    co = cohere.Client(api_key=COHERE_API_KEY)
    index = Pinecone(api_key=PINECONE_API_KEY).Index(INDEX_NAME)
    conn = psycopg2.connect(DATABASE_URL)
    _ensure_table(conn)

    splitter = _splitter()

    start_passage = _read_checkpoint()
    if start_passage:
        log.info("Resuming from passage %d / %d (checkpoint found)...", start_passage, TOP_K)

    log.info("Streaming MS MARCO passages (top %d)...", TOP_K)
    dataset = ir_datasets.load("msmarco-passage")

    pending_texts: list[str] = []
    pending_meta: list[dict] = []
    total = 0
    passages_done = start_passage

    # islice(iter, start, stop) skips `start` docs then yields up to index `stop`
    for doc in islice(dataset.docs_iter(), start_passage, TOP_K):
        passages_done += 1
        for chunk_idx, chunk_text in enumerate(splitter.split_text(doc.text)):
            pending_texts.append(chunk_text)
            pending_meta.append(
                {"passage_id": doc.doc_id, "chunk_idx": chunk_idx, "text": chunk_text}
            )

        while len(pending_texts) >= EMBED_BATCH:
            batch_texts = pending_texts[:EMBED_BATCH]
            batch_meta = pending_meta[:EMBED_BATCH]
            del pending_texts[:EMBED_BATCH]
            del pending_meta[:EMBED_BATCH]

            _embed_and_store(co, index, conn, batch_texts, batch_meta)
            _write_checkpoint(passages_done)

            prev, total = total, total + len(batch_texts)
            if total // LOG_EVERY > prev // LOG_EVERY:
                log.info("Upserted %d vectors (passage %d / %d)...", total, passages_done, TOP_K)

    # flush tail
    if pending_texts:
        _embed_and_store(co, index, conn, pending_texts, pending_meta)
        _write_checkpoint(passages_done)
        total += len(pending_texts)

    conn.close()
    log.info("Done. Total vectors upserted: %d", total)
    CHECKPOINT_FILE.unlink(missing_ok=True)  # clean up on successful completion


if __name__ == "__main__":
    main()
