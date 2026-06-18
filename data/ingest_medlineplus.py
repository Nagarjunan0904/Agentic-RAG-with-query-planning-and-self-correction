"""
Ingest MedlinePlus English health topics into Elasticsearch (BM25 index).
Source: data/medlineplus_raw/mplus_topics_2026-06-18.xml (NIH/NLM bulk export)

Document structure per topic:
  title    — topic title (boosted x2 at query time)
  synonyms — also-called alternate names (boosted x1.5 at query time)
  body     — HTML-stripped full-summary + synonyms + MeSH terms joined together
  url      — canonical MedlinePlus URL (keyword, not searched)

The old 'wikipedia' index (different schema) is deleted if present.
The 'medical_topics' index is always recreated so re-runs are idempotent.
"""

import html
import logging
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

load_dotenv()

ELASTICSEARCH_URL = os.environ["ELASTICSEARCH_URL"]

INDEX_NAME  = "medical_topics"
XML_PATH    = Path(__file__).parent / "medlineplus_raw" / "mplus_topics_2026-06-18.xml"
BATCH_SIZE  = 500
LOG_EVERY   = 500

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _strip_html(raw: str) -> str:
    """Strip HTML tags and decode HTML entities, collapse whitespace."""
    if not raw:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_topics(xml_path: Path) -> list[dict]:
    """Parse the MedlinePlus XML export; return English-only topic dicts."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    log.info(
        "Parsing %s (total=%s topics across all languages)...",
        xml_path.name, root.get("total", "?"),
    )

    topics = []
    for ht in root.findall("health-topic"):
        if ht.get("language") != "English":
            continue

        title    = ht.get("title", "").strip()
        topic_id = ht.get("id", "")
        url      = ht.get("url", "")

        # Alternate names
        also_called = [
            e.text.strip() for e in ht.findall("also-called") if e.text and e.text.strip()
        ]

        # Full-summary body (stored as escaped HTML in the text node)
        fs_elem      = ht.find("full-summary")
        full_summary = _strip_html(fs_elem.text if fs_elem is not None else "")

        # MeSH descriptor terms
        mesh_terms = [
            d.text.strip()
            for mh in ht.findall("mesh-heading")
            for d in mh.findall("descriptor")
            if d.text and d.text.strip()
        ]

        # body = full prose + synonyms inline + MeSH terms
        # Including synonyms and MeSH in body gives BM25 natural term coverage
        # even when the retriever only queries the "body" field.
        body_parts = [full_summary]
        if also_called:
            body_parts.append("Also known as: " + ", ".join(also_called))
        if mesh_terms:
            body_parts.append("Medical terms: " + ", ".join(mesh_terms))

        topics.append({
            "id":       topic_id,
            "title":    title,
            "synonyms": " ".join(also_called),
            "body":     "\n".join(body_parts),
            "url":      url,
        })

    log.info("Extracted %d English topics.", len(topics))
    return topics


def _setup_index(es: Elasticsearch) -> None:
    # Remove the old wikipedia index if it still exists (different schema)
    if es.indices.exists(index="wikipedia").body:
        es.indices.delete(index="wikipedia")
        log.info("Deleted legacy 'wikipedia' index.")

    # Always recreate medical_topics so re-runs are idempotent
    if es.indices.exists(index=INDEX_NAME).body:
        es.indices.delete(index=INDEX_NAME)
        log.info("Deleted existing '%s' index for fresh ingest.", INDEX_NAME)

    es.indices.create(
        index=INDEX_NAME,
        mappings={
            "properties": {
                "title":    {"type": "text", "analyzer": "english"},
                "synonyms": {"type": "text", "analyzer": "english"},
                "body":     {"type": "text", "analyzer": "english"},
                "url":      {"type": "keyword"},
            }
        },
    )
    log.info("Created index '%s' with English analyzer.", INDEX_NAME)


def _doc_actions(topics: list[dict]):
    for t in topics:
        yield {
            "_index": INDEX_NAME,
            "_id":    t["id"],
            "_source": {
                "title":    t["title"],
                "synonyms": t["synonyms"],
                "body":     t["body"],
                "url":      t["url"],
            },
        }


def _batched(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _verify(es: Elasticsearch) -> None:
    query = "diabetes"
    resp = es.search(
        index=INDEX_NAME,
        query={
            "multi_match": {
                "query":  query,
                "fields": ["title^2", "synonyms^1.5", "body"],
            }
        },
        size=3,
    )
    hits = resp["hits"]["hits"]
    print(f"\nTest query: '{query}' — top {len(hits)} results:")
    for i, hit in enumerate(hits, 1):
        src = hit["_source"]
        syn = f"  also-called: {src['synonyms']}" if src.get("synonyms") else ""
        print(f"  {i}. [score={hit['_score']:.4f}] {src['title']}")
        if syn:
            print(syn)


def main() -> None:
    es = Elasticsearch(ELASTICSEARCH_URL)

    _setup_index(es)

    topics = _parse_topics(XML_PATH)

    total = 0
    for batch in _batched(topics, BATCH_SIZE):
        bulk(es, _doc_actions(batch))
        prev, total = total, total + len(batch)
        if total // LOG_EVERY > prev // LOG_EVERY:
            log.info("Indexed %d / %d documents...", total, len(topics))

    log.info("Done. Total documents indexed: %d", total)
    es.indices.refresh(index=INDEX_NAME)
    _verify(es)


if __name__ == "__main__":
    main()
