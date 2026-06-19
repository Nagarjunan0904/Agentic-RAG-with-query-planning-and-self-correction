# Agentic Healthcare RAG with Query Planning & Self-Correction

A production-style Retrieval-Augmented Generation system grounded in NIH medical data, capable of answering clinical questions, performing medical terminology lookups, and querying synthetic patient records. The system routes each query to one of three retrieval paths — semantic vector search, BM25 keyword search, or SQL — using a Claude-powered query planner, then applies Cohere reranking and a self-correction loop before generating the final answer.

**Live demo** → [https://agentic-rag-with-query-planning-and.vercel.app/](https://agentic-rag-with-query-planning-and.vercel.app/)  
**API docs** → [https://agentic-rag-with-query-planning-and-self-correct-production.up.railway.app/docs](https://agentic-rag-with-query-planning-and-self-correct-production.up.railway.app/docs)

---

## Architecture

```
User query
    │
    ▼
┌─────────────┐
│ Query       │  Claude classifies intent:
│ Planner     │  SEMANTIC / KEYWORD / STRUCTURED
└──────┬──────┘
       │
   ┌───┴────────────────────────┐
   │                            │                          │
   ▼                            ▼                          ▼
┌──────────────┐   ┌────────────────────┐   ┌─────────────────────┐
│ Vector       │   │ BM25 Retriever     │   │ SQL Agent           │
│ Retriever    │   │                    │   │                     │
│ Pinecone     │   │ Elasticsearch      │   │ PostgreSQL          │
│ (MedQuAD)    │   │ (MedlinePlus)      │   │ (Synthea patients)  │
└──────┬───────┘   └────────┬───────────┘   └──────────┬──────────┘
       └───────────────────┬┘                           │
                           ▼                            │
                 ┌──────────────────┐                   │
                 │ Cohere Reranker  │◄──────────────────┘
                 └────────┬─────────┘
                          ▼
                 ┌──────────────────┐
                 │ Generator        │  Claude synthesises answer
                 └────────┬─────────┘
                          ▼
                 ┌──────────────────┐
                 │ Self-Evaluator   │  Confidence score
                 └────────┬─────────┘
                          │
              ┌───────────┴──────────────┐
        high confidence             low confidence
              │                          │
              ▼                          ▼
         Return answer            Re-retrieve (HyDE)
                                  → Generator → Answer
```

### Query Routing

The query planner (Claude Sonnet) classifies by *intent*, not surface phrasing. A distinction sharpened during testing:

- **KEYWORD** fires for single named-entity lookups — even "What is hypertension?" — where the user wants the canonical reference summary for one specific term. Bare-term queries ("hypertension", "asthma treatment guidelines") are always KEYWORD.
- **SEMANTIC** fires for multi-faceted explanatory questions — "What are the symptoms of type 2 diabetes and how does it differ from type 1?" — where the answer requires synthesising across multiple sources.

Misrouting a bare-term lookup to SEMANTIC retrieves fragmentary QA pairs instead of the authoritative MedlinePlus topic summary; this boundary was tested and explicitly encoded in the planner prompt's few-shot examples.

---

## Retrieval Corpora

### SEMANTIC — MedQuAD (Pinecone HNSW)
- **Source:** [keivalya/MedQuad-MedicalQnADataset](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset) — 16,407 NIH medical Q&A pairs covering diseases, symptoms, treatments, genetics, and clinical research
- **Indexing:** Each `Q: <question>\nA: <answer>` pair embedded with `cohere embed-english-v3.0` (`input_type="search_query"`, symmetric space) and upserted into a Pinecone serverless index; 21,785 vectors after 512-token chunking
- **Best for:** "What are the symptoms of X?", "How is Y treated?", "What causes Z?"

### KEYWORD — MedlinePlus health topics (Elasticsearch BM25)
- **Source:** [MedlinePlus bulk XML export](https://medlineplus.gov/xml.html) (NIH/NLM) — 1,017 English health topic summaries covering diseases, drugs, medical tests, and wellness topics
- **Indexing:** Title, alternate names (also-called), full-summary (HTML-stripped), and MeSH descriptor terms indexed into an Elasticsearch `medical_topics` index with the English analyzer; title boosted ×2, synonyms ×1.5
- **Best for:** Bare condition/drug name lookups ("hypertension", "A1C"), single-topic queries ("asthma treatment guidelines"), medical encyclopedia-style lookups

### STRUCTURED — Synthea synthetic patient database (PostgreSQL)
- **Source:** [Synthea](https://synthea.mitre.org/) — 111 synthetic patients with realistic medical histories across 11 clinical tables
- **Schema:** `patients`, `encounters`, `conditions`, `medications`, `procedures`, `observations`, `allergies`, `immunizations`, `careplans`, `organizations`, `providers`
- **Access:** Read-only `rag_reader` role; Claude generates safe SELECT-only SQL from a dynamically extracted schema
- **Best for:** "How many patients have diabetes?", "What is the most prescribed medication?", aggregate queries over patient records

---

## RAGAS Evaluation Results

Evaluated on 100 healthcare questions (33 SEMANTIC, 33 KEYWORD, 34 STRUCTURED) using Claude Haiku as the RAGAS judge and Cohere embeddings for answer relevancy scoring.

| Metric | SEMANTIC | KEYWORD | STRUCTURED | Overall |
|--------|:--------:|:-------:|:----------:|:-------:|
| Faithfulness | 0.97 | 0.97 | 0.16† | 0.69 |
| Answer Relevancy | 0.79 | 0.84 | 0.95 | 0.86 |
| Context Recall | 0.46 | 0.48 | 1.00 | 0.65 |
| Context Precision | 0.72 | 0.93 | 0.97 | 0.88 |

> **†STRUCTURED Faithfulness** is a known RAGAS artifact: SQL results return raw tuples (e.g. `('Epoetin Alfa', 863)`) which the generator correctly converts to prose, but RAGAS cannot trace prose claims back to raw tuple text. Retrieval quality metrics (Context Recall = 1.00, Context Precision = 0.97) confirm the SQL path is working correctly.

**Highlights:**
- KEYWORD path achieves 0.97 faithfulness and 0.93 context precision — MedlinePlus summaries are authoritative and concise, ideal for BM25 retrieval
- STRUCTURED path achieves perfect context recall (1.00) — every SQL query retrieves exactly the right data
- SEMANTIC context recall (0.46) reflects the difficulty of open-ended medical QA; the retriever surfaces relevant passages but NIH answers span multiple MedQuAD records

---

## Example Queries

| Strategy | Example queries |
|----------|----------------|
| **SEMANTIC** | "What are the symptoms of type 2 diabetes?" · "How is Alzheimer's disease treated?" · "What causes chronic kidney disease?" |
| **KEYWORD** | "hypertension" · "asthma treatment guidelines" · "What is A1C?" |
| **STRUCTURED** | "What is the most common condition among patients?" · "What is the most prescribed medication?" · "How many emergency encounters are recorded?" |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM / Judge | Claude Sonnet 4.6 (generation + planning) · Claude Haiku 4.5 (RAGAS judge) |
| Embeddings | Cohere `embed-english-v3.0` |
| Reranking | Cohere `rerank-english-v3.0` |
| Vector store | Pinecone serverless (HNSW) |
| Keyword search | Elasticsearch 8 (BM25, English analyzer) |
| Structured DB | PostgreSQL 15 (Synthea schema, 11 tables) |
| Orchestration | LangGraph |
| Backend | FastAPI + Uvicorn |
| Frontend | React + Vite + Tailwind CSS |
| Evaluation | RAGAS 0.2 |
| Deployment | Railway (backend) · Vercel (frontend) · Docker (local infra) |

---

## Local Setup

**Prerequisites:** Python 3.11+, Docker Desktop, Node 18+

```bash
# 1. Clone and install Python deps
git clone <repo>
cd agentic-rag-with-query-planning-and-self-correction
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Fill in: ANTHROPIC_API_KEY, COHERE_API_KEY, PINECONE_API_KEY, LANGSMITH_API_KEY

# 3. Start local infrastructure
docker-compose up -d postgres elasticsearch

# 4. Ingest data (one-time)
python data/ingest_synthea.py          # PostgreSQL — Synthea patient data
python data/ingest_medlineplus.py      # Elasticsearch — MedlinePlus topics
python data/ingest_medquad.py          # Pinecone — MedQuAD QA pairs (~24 min)

# 5. Run backend
uvicorn backend.api.main:app --reload --port 8000

# 6. Run frontend
cd frontend && npm install && npm run dev
```

`.env.example`:
```
ANTHROPIC_API_KEY=
COHERE_API_KEY=
PINECONE_API_KEY=
PINECONE_INDEX=agentic-rag
DATABASE_URL=postgresql://postgres:postgres@localhost:5433/ragdb
ELASTICSEARCH_URL=http://localhost:9200
LANGSMITH_API_KEY=
```

### Running RAGAS Evaluation

```bash
# All 100 questions
python -m backend.evaluation.ragas_pipeline

# Smoke test (first 10)
python -m backend.evaluation.ragas_pipeline --limit 10

# STRUCTURED only (Q67–100)
python -m backend.evaluation.ragas_pipeline --offset 66 --limit 34

# Force re-run RAG inference (ignore cache)
python -m backend.evaluation.ragas_pipeline --refresh
```

---

## Deployment

Deployed on a zero-fixed-cost stack suitable for portfolio and low-traffic production use:

| Service | Platform | URL |
|---------|----------|-----|
| Frontend | Vercel | https://agentic-rag-with-query-planning-and.vercel.app/ |
| Backend API | Railway | https://agentic-rag-with-query-planning-and-self-correct-production.up.railway.app/docs |
| PostgreSQL | Railway managed Postgres | — |
| Pinecone | Pinecone serverless (free tier) | — |
| Elasticsearch | Railway Docker service | — |

**Deployment Architecture:**  
All backend services (FastAPI, PostgreSQL, Elasticsearch) run as separate Railway services within a single Railway project, with Railway Serverless enabled on the stateless FastAPI service so it scales to zero between requests. Pinecone is the one fully-managed external service — shared between local dev and production via the same `PINECONE_API_KEY` and `PINECONE_INDEX`, so no separate re-indexing step is needed when switching environments.

---

## Key Technical Decisions

**Why Railway + Vercel instead of AWS?**  
AWS managed equivalents (RDS, OpenSearch, ECS, CloudFront) would cost $40–70/month at minimum for this configuration. Railway's usage-based pricing and Vercel's free tier keep this at ~$0/month for a portfolio project with intermittent traffic, with no meaningful capability trade-off at this scale.

**Why the healthcare domain?**  
Healthcare is a high-stakes RAG application where query planning and self-correction are genuinely meaningful — the difference between routing a clinical question to semantic search versus a patient-record aggregate query to SQL matters for answer quality. Generic demo domains (e-commerce, Wikipedia) flatten this distinction. MedQuAD, MedlinePlus, and Synthea are all freely available NIH/MITRE datasets with no licensing restrictions.

**Why Claude Haiku for RAGAS judging?**  
At RAGAS evaluation scale (hundreds of LLM judge calls per evaluation run), Haiku is approximately 10–20× cheaper than Sonnet with equivalent judgment quality for the structured scoring tasks RAGAS uses (faithfulness decomposition, relevancy assessment). Sonnet is reserved for user-facing generation where response quality is directly visible.

**Why symmetric Cohere embeddings (`input_type="search_query"` for both ingestion and retrieval)?**  
Cohere's asymmetric mode (`search_document` at ingestion, `search_query` at retrieval) is intended for passage retrieval from long documents. For MedQuAD Q&A pairs and short MedlinePlus summaries, the query and passage are semantically similar in length and structure. Using `search_query` for both sides keeps them in the same embedding space and avoids near-zero cosine similarity that would otherwise break RAGAS `answer_relevancy` scoring.

---

## Known Limitations & Potential Improvements

**Context Recall is the most actionable improvement area** per RAGAS results (SEMANTIC 0.46, KEYWORD 0.48). Open-ended medical questions often span multiple MedQuAD records, and the top-10 Pinecone retrieval window doesn't always surface all relevant passages. Targeted improvements:

- **Larger retrieval window:** Increase `TOP_K` from 10 to 20–30 for SEMANTIC queries before reranking, trading slightly higher latency for better recall.
- **Query expansion / HyDE at retrieval time:** Generating a hypothetical answer before embedding the query improves passage recall for explanatory questions (HyDE is already used in the self-correction retry path; applying it on the first pass for SEMANTIC is the natural next step).
- **Finer MedQuAD chunking:** Condition-specific sub-answers (symptoms vs. treatment vs. prognosis) could be split into separate indexed documents rather than chunked from a single Q+A blob, giving the reranker more precise targets.

**STRUCTURED Faithfulness (0.16) is a RAGAS scoring artifact**, not a system defect — see footnote in the evaluation table. Context Recall 1.00 and Context Precision 0.97 confirm the SQL retrieval path is correct.

**Query planner edge cases:** The KEYWORD/SEMANTIC boundary required prompt iteration during testing ("What is hypertension?" initially misclassified as SEMANTIC). The current prompt uses intent-based definitions and medical-domain few-shot examples to address this, but novel phrasings may still misclassify at the margins.
