"""
Read Synthea CSV exports from data/synthea_csv/, infer schema,
create matching tables in PostgreSQL, load rows, and create read-only rag_reader role.
"""

import csv
import logging
import os
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import urlparse

import sqlalchemy as sa
from dotenv import load_dotenv
from sqlalchemy import Column, MetaData, Table, create_engine, inspect, text

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
SYNTHEA_CSV_DIR = Path(__file__).parent / "synthea_csv"
RAG_READER_PASS = "rag_reader"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Only infer NUMERIC for columns whose names clearly indicate measurements/amounts.
# Everything else defaults to TEXT so that codes, ZIPs, FIPS, and UUID-style IDs
# are never coerced to numbers.
_NUMERIC_NAME_RE = re.compile(
    r"cost|coverage|expense|lat|lon|revenue|utilization|outstanding|"
    r"income|amount|dispense|quantity|base_",
    re.I,
)


def _infer_pg_type(col_name: str, sample_values: list) -> sa.types.TypeEngine:
    """Infer a PostgreSQL column type from column name and sample non-empty values."""
    non_empty = [v for v in sample_values if v and v.strip()]
    probe = non_empty[:50]

    if not probe:
        return sa.Text()

    if all(_UUID_RE.match(v) for v in probe):
        return sa.Text()

    if all(_TIMESTAMP_RE.match(v) for v in probe):
        return sa.TIMESTAMP()

    if all(_DATE_RE.match(v) for v in probe):
        return sa.DATE()

    if _NUMERIC_NAME_RE.search(col_name):
        try:
            [float(v) for v in probe]
            return sa.NUMERIC()
        except ValueError:
            pass

    return sa.Text()


def _coerce_value(value: str, col_type: sa.types.TypeEngine):
    """Convert a raw CSV string to the appropriate Python type for bulk insert."""
    if value == "" or value is None:
        return None
    if isinstance(col_type, sa.NUMERIC):
        try:
            return Decimal(value)
        except InvalidOperation:
            return None
    if isinstance(col_type, sa.TIMESTAMP):
        from datetime import datetime
        v = value.rstrip("Z").split("+")[0]
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(v, fmt)
            except ValueError:
                pass
        return None
    if isinstance(col_type, sa.DATE):
        from datetime import date
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return value


def _load_csv_files() -> dict:
    """Return {table_name: (headers, rows)} for every CSV in SYNTHEA_CSV_DIR.

    All column names are normalised to lowercase so PostgreSQL identifiers are
    consistent regardless of how Synthea capitalises its headers.
    """
    tables = {}
    for csv_path in sorted(SYNTHEA_CSV_DIR.glob("*.csv")):
        table_name = csv_path.stem.lower()
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            raw_headers = list(reader.fieldnames or [])
            headers = [h.lower() for h in raw_headers]
            rows = [{k.lower(): v for k, v in row.items()} for row in reader]
        tables[table_name] = (headers, rows)
        log.info("Read %-35s %5d rows", csv_path.name, len(rows))
    return tables


def _build_metadata(table_data: dict) -> tuple:
    """Infer PostgreSQL types from all CSV values and build SQLAlchemy MetaData."""
    metadata = MetaData()
    col_type_map: dict[str, dict[str, sa.types.TypeEngine]] = {}

    for table_name, (headers, rows) in table_data.items():
        col_type_map[table_name] = {}
        columns = []
        for col_name in headers:
            # Sample from all rows so rarely-populated columns (e.g. STOP) get typed correctly
            sample = [row.get(col_name, "") for row in rows]
            pg_type = _infer_pg_type(col_name, sample)
            col_type_map[table_name][col_name] = pg_type
            columns.append(Column(col_name, pg_type))
        Table(table_name, metadata, *columns)

    return metadata, col_type_map


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
            print(f"  {table_name:<35} {count:>6} rows")

        patient_count = conn.execute(
            text('SELECT COUNT(*) FROM "patients"')
        ).scalar()

    if patient_count > 0:
        print(f'\nOK: SELECT COUNT(*) FROM "patients" = {patient_count}')
    else:
        raise AssertionError("patients table is empty — ingestion may have failed")


def main() -> None:
    log.info("Loading Synthea CSVs from %s", SYNTHEA_CSV_DIR)
    table_data = _load_csv_files()

    pg_engine = create_engine(DATABASE_URL)

    log.info("Inferring schema and creating %d tables in PostgreSQL...", len(table_data))
    metadata, col_type_map = _build_metadata(table_data)
    metadata.drop_all(pg_engine, checkfirst=True)
    metadata.create_all(pg_engine)
    log.info("Created %d tables in PostgreSQL.", len(metadata.tables))

    log.info("Loading rows into PostgreSQL...")
    with pg_engine.connect() as conn:
        for table_name, (headers, rows) in table_data.items():
            table = metadata.tables[table_name]
            if not rows:
                log.info("  %-35s (empty)", table_name)
                continue
            converted = []
            for raw_row in rows:
                record = {}
                for col_name in headers:
                    col_type = col_type_map[table_name][col_name]
                    record[col_name] = _coerce_value(raw_row.get(col_name, ""), col_type)
                converted.append(record)
            conn.execute(table.insert(), converted)
            log.info("  %-35s %5d rows", table_name, len(converted))
        conn.commit()

    _create_rag_reader(pg_engine)
    verify_schema()


if __name__ == "__main__":
    main()
