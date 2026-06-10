"""
Download Chinook SQLite database, migrate to PostgreSQL, create read-only rag_reader role.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx
import sqlalchemy as sa
from dotenv import load_dotenv
from sqlalchemy import MetaData, create_engine, inspect, text

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]

CHINOOK_URL = (
    "https://github.com/lerocha/chinook-database/raw/master"
    "/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite"
)
CHINOOK_CACHE = Path(__file__).parent / "chinook.sqlite"
RAG_READER_PASS = "rag_reader"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _download() -> Path:
    if CHINOOK_CACHE.exists():
        log.info("Using cached Chinook DB: %s", CHINOOK_CACHE)
        return CHINOOK_CACHE
    log.info("Downloading Chinook SQLite database...")
    with httpx.Client(follow_redirects=True) as client:
        resp = client.get(CHINOOK_URL, timeout=30)
        resp.raise_for_status()
    CHINOOK_CACHE.write_bytes(resp.content)
    log.info("Saved %d KB -> %s", len(resp.content) // 1024, CHINOOK_CACHE)
    return CHINOOK_CACHE


def _coerce_row(table: sa.Table, mapping) -> dict:
    """Parse SQLite datetime strings into Python datetime objects for PostgreSQL."""
    result = {}
    for col_name, value in mapping.items():
        col = table.c[col_name]
        if (
            value is not None
            and isinstance(col.type, (sa.DateTime, sa.Date))
            and isinstance(value, str)
        ):
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    value = datetime.strptime(value, fmt)
                    break
                except ValueError:
                    pass
        result[col_name] = value
    return result


def _pg_type(t: sa.types.TypeEngine) -> sa.types.TypeEngine:
    """Map SQLite-dialect types to generic equivalents PostgreSQL can compile."""
    cls = type(t).__name__
    length = getattr(t, "length", None)
    if cls == "NVARCHAR":
        return sa.String(length) if length else sa.Text()
    if cls == "NCHAR":
        return sa.CHAR(length) if length else sa.CHAR()
    if cls == "DATETIME":
        return sa.TIMESTAMP()
    if cls == "DATE":
        return sa.DATE()
    if cls == "BOOLEAN":
        return sa.BOOLEAN()
    if cls == "NUMERIC":
        precision = getattr(t, "precision", None)
        scale = getattr(t, "scale", None)
        return sa.NUMERIC(precision, scale)
    if cls == "NullType":
        # SQLite reflection falls back to NullType for unrecognised affinities
        return sa.Text()
    return t


def _normalize_types(metadata: MetaData) -> None:
    """Walk every column in the reflected metadata and replace any
    SQLite-specific type objects with generic SQLAlchemy types so that
    PostgreSQL's DDL compiler produces valid SQL."""
    for table in metadata.tables.values():
        for col in table.columns:
            col.type = _pg_type(col.type)


def _migrate(sqlite_path: Path, pg_engine: sa.Engine) -> None:
    sqlite_engine = create_engine(f"sqlite:///{sqlite_path}")
    metadata = MetaData()
    metadata.reflect(bind=sqlite_engine)
    log.info("Reflected %d tables from SQLite.", len(metadata.tables))

    _normalize_types(metadata)

    metadata.drop_all(pg_engine, checkfirst=True)
    metadata.create_all(pg_engine)
    log.info("Created %d tables in PostgreSQL.", len(metadata.tables))

    with sqlite_engine.connect() as src, pg_engine.connect() as dst:
        # Disable FK triggers for the duration of the import so insertion order
        # doesn't matter (handles Employee's self-referential ReportsTo FK).
        dst.execute(text("SET session_replication_role = 'replica'"))
        for table in metadata.sorted_tables:
            rows = src.execute(table.select()).fetchall()
            if not rows:
                continue
            converted = [_coerce_row(table, row._mapping) for row in rows]
            dst.execute(table.insert(), converted)
            log.info("  %-20s %5d rows", table.name, len(rows))
        dst.execute(text("SET session_replication_role = DEFAULT"))
        dst.commit()


def _create_rag_reader(pg_engine: sa.Engine) -> None:
    db_name = urlparse(DATABASE_URL).path.lstrip("/")
    with pg_engine.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'rag_reader'")
        ).fetchone()
        if not exists:
            conn.execute(
                text(f"CREATE ROLE rag_reader WITH LOGIN PASSWORD '{RAG_READER_PASS}'")
            )
            log.info("Created role rag_reader.")
        else:
            log.info("Role rag_reader already exists, updating grants.")
        conn.execute(text(f'GRANT CONNECT ON DATABASE "{db_name}" TO rag_reader'))
        conn.execute(text("GRANT USAGE ON SCHEMA public TO rag_reader"))
        conn.execute(text("GRANT SELECT ON ALL TABLES IN SCHEMA public TO rag_reader"))
        conn.commit()
    log.info("Granted CONNECT + SELECT on all tables to rag_reader.")


def _rag_reader_url() -> str:
    parsed = urlparse(DATABASE_URL)
    port = parsed.port or 5432
    return f"postgresql://rag_reader:{RAG_READER_PASS}@{parsed.hostname}:{port}{parsed.path}"


def verify_schema() -> None:
    engine = create_engine(_rag_reader_url())
    insp = inspect(engine)
    tables = sorted(insp.get_table_names())

    print(f"\nSchema verification — {len(tables)} tables (connected as rag_reader):")
    with engine.connect() as conn:
        for table_name in tables:
            count = conn.execute(
                text(f'SELECT COUNT(*) FROM "{table_name}"')
            ).scalar()
            print(f"  {table_name:<25} {count:>6} rows")

        track_count = conn.execute(
            text('SELECT COUNT(*) FROM "Track"')
        ).scalar()

    if track_count == 3503:
        print(f'\nOK: SELECT COUNT(*) FROM "Track" = {track_count}')
    else:
        raise AssertionError(f'Expected 3503 tracks, got {track_count}')


def main() -> None:
    sqlite_path = _download()
    pg_engine = create_engine(DATABASE_URL)
    _migrate(sqlite_path, pg_engine)
    _create_rag_reader(pg_engine)
    verify_schema()


if __name__ == "__main__":
    main()
