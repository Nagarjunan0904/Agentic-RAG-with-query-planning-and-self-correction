import logging
import os
import re
import time
from urllib.parse import urlparse

import anthropic
import psycopg2
from dotenv import load_dotenv

from backend.agents.state import AgentState

load_dotenv(override=True)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
DATABASE_URL = os.environ.get("DATABASE_URL", "")
MODEL = "claude-sonnet-4-5"
RAG_READER_PASS = "rag_reader"

log = logging.getLogger(__name__)

_FORBIDDEN = re.compile(r"\b(DROP|DELETE|INSERT|UPDATE|TRUNCATE|ALTER)\b", re.IGNORECASE)

_pg_conn = None


class SafetyError(Exception):
    pass


def _rag_reader_url() -> str:
    parsed = urlparse(DATABASE_URL)
    port = parsed.port or 5432
    return f"postgresql://rag_reader:{RAG_READER_PASS}@{parsed.hostname}:{port}{parsed.path}"


def _get_pg_conn():
    global _pg_conn
    if _pg_conn is None or _pg_conn.closed:
        _pg_conn = psycopg2.connect(_rag_reader_url())
        _pg_conn.autocommit = True
    return _pg_conn


def _extract_schema() -> str:
    conn = _get_pg_conn()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name != 'chunks'
            ORDER BY table_name, ordinal_position
        """)
        rows = cur.fetchall()

    tables: dict[str, list[str]] = {}
    for table_name, column_name, data_type in rows:
        tables.setdefault(table_name, []).append(f"{column_name} {data_type}")

    schema_parts = []
    conn = _get_pg_conn()
    with conn.cursor() as cur:
        for table_name, cols in sorted(tables.items()):
            # Quote every identifier so Claude sees the exact casing required
            quoted_cols = [f'"{col.split()[0]}" {" ".join(col.split()[1:])}' for col in cols]
            col_str = ", ".join(quoted_cols)
            try:
                cur.execute(f'SELECT * FROM "{table_name}" LIMIT 1')
                sample_row = cur.fetchone()
            except Exception:
                sample_row = None
            if sample_row is not None:
                schema_parts.append(
                    f'Table: "{table_name}" ({col_str}) | Sample: {sample_row}'
                )
            else:
                schema_parts.append(f'Table: "{table_name}" ({col_str})')

    return "\n".join(schema_parts)


def _generate_sql(schema: str, query: str) -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = (
        f"You are a PostgreSQL expert. Given the following database schema:\n\n"
        f"{schema}\n\n"
        f"Write a valid PostgreSQL SELECT query to answer this question:\n"
        f"{query}\n\n"
        f"Rules:\n"
        f'- Always wrap every table name and column name in double quotes.\n'
        f'- Never use unquoted identifiers — PostgreSQL identifiers are case-sensitive.\n'
        f"- Always SELECT all columns needed to fully answer the question. "
        f"For aggregation queries, include both the grouped column and the aggregate result column.\n"
        f"- Example of correct style:\n"
        f'  SELECT "description", COUNT(*) AS cnt FROM "conditions"\n'
        f'  GROUP BY "description" ORDER BY cnt DESC LIMIT 5\n'
        f"Return only valid PostgreSQL SELECT SQL. "
        f"No markdown, no explanation, no ```sql fences."
    )
    message = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    sql = message.content[0].text.strip()
    sql = re.sub(r"```\w*", "", sql).replace("```", "").strip()
    return sql


def _check_safe(sql: str) -> None:
    if not sql.strip().upper().startswith("SELECT"):
        raise SafetyError(sql)
    if _FORBIDDEN.search(sql):
        raise SafetyError(sql)


def run(state: AgentState) -> AgentState:
    query = state["query"]
    t0 = time.perf_counter()
    retrieved_docs = []
    retry_reason = None

    try:
        schema = _extract_schema()
        sql = _generate_sql(schema, query)
        log.info("SQLAgent: generated SQL=%r", sql)

        _check_safe(sql)

        conn = _get_pg_conn()
        with conn.cursor() as cur:
            cur.execute(sql)
            result_rows = cur.fetchall()

        retrieved_docs = [
            {
                "text": f"SQL result: {row}",
                "score": 1.0,
                "source": "sql",
                "chunk_id": f"sql_{i}",
            }
            for i, row in enumerate(result_rows)
        ]
    except SafetyError as exc:
        log.error("SQLAgent: safety check failed for SQL=%r", str(exc))
        retry_reason = f"Safety check failed: {exc}"
    except Exception as exc:
        log.error("SQLAgent: execution error: %s", exc)
        retry_reason = str(exc)

    latency_ms = dict(state.get("latency_ms") or {})
    latency_ms["sql_agent"] = round((time.perf_counter() - t0) * 1000, 1)

    log.info(
        "SQLAgent: query=%r rows=%d latency=%.1fms",
        query,
        len(retrieved_docs),
        latency_ms["sql_agent"],
    )

    result: AgentState = {**state, "retrieved_docs": retrieved_docs, "latency_ms": latency_ms}
    if retry_reason is not None:
        result["retry_reason"] = retry_reason
    return result
