"""
Ingest Wikipedia 20220301.en (top 20K articles) into Elasticsearch.

Extra dep not in requirements.txt:
    pip install datasets
"""

import logging
import os
from itertools import islice

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

load_dotenv()

ELASTICSEARCH_URL = os.environ["ELASTICSEARCH_URL"]

INDEX_NAME = "wikipedia"
TOP_K = 20_000
BATCH_SIZE = 500
LOG_EVERY = 1_000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _create_index(es: Elasticsearch) -> None:
    if es.indices.exists(index=INDEX_NAME).body:
        log.info("Index '%s' already exists, skipping creation.", INDEX_NAME)
        return
    es.indices.create(
        index=INDEX_NAME,
        mappings={
            "properties": {
                "title": {"type": "text", "analyzer": "english"},
                "body":  {"type": "text", "analyzer": "english"},
            }
        },
    )
    log.info("Created index '%s' with English analyzer.", INDEX_NAME)


def _batched(iterable, n: int):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch


def _doc_actions(articles: list[dict]):
    for article in articles:
        yield {
            "_index": INDEX_NAME,
            "_id": article["id"],
            "_source": {
                "title": article["title"],
                "body": article["text"],
            },
        }


def _verify(es: Elasticsearch) -> None:
    query = "Pinecone database"
    resp = es.search(
        index=INDEX_NAME,
        query={
            "multi_match": {
                "query": query,
                "fields": ["title", "body"],
            }
        },
        size=3,
    )
    hits = resp["hits"]["hits"]
    print(f"\nTest query: '{query}' — top {len(hits)} results:")
    for i, hit in enumerate(hits, 1):
        print(f"  {i}. [score={hit['_score']:.4f}] {hit['_source']['title']}")


def main() -> None:
    from datasets import load_dataset  # deferred so the file is importable without it

    es = Elasticsearch(ELASTICSEARCH_URL)
    _create_index(es)

    log.info("Streaming Wikipedia 20220301.en (top %d articles)...", TOP_K)
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )

    total = 0
    for batch in _batched(islice(dataset, TOP_K), BATCH_SIZE):
        bulk(es, _doc_actions(batch))
        prev, total = total, total + len(batch)
        if total // LOG_EVERY > prev // LOG_EVERY:
            log.info("Indexed %d / %d documents...", total, TOP_K)

    log.info("Done. Total documents indexed: %d", total)
    _verify(es)


if __name__ == "__main__":
    main()
