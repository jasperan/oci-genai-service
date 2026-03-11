"""Vector table creation and management for Oracle 26ai."""

from __future__ import annotations

import re
from dataclasses import dataclass

_IDENTIFIER_RE = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]{0,127}$')


def validate_identifier(name: str) -> str:
    """Validate a SQL identifier to prevent injection."""
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return name


@dataclass
class VectorTableConfig:
    """Configuration for a vector table in Oracle."""

    table_name: str
    vector_dims: int = 1024
    distance_metric: str = "COSINE"
    id_column: str = "id"
    text_column: str = "text"
    vector_column: str = "embedding"
    metadata_column: str = "metadata"

    def __post_init__(self):
        validate_identifier(self.table_name)
        validate_identifier(self.id_column)
        validate_identifier(self.text_column)
        validate_identifier(self.vector_column)
        validate_identifier(self.metadata_column)


def create_vector_table(conn, config: VectorTableConfig) -> None:
    """Create a vector table in Oracle 26ai with vector index."""
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {config.table_name} (
        {config.id_column} RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
        {config.text_column} CLOB,
        {config.vector_column} VECTOR({config.vector_dims}, FLOAT64),
        {config.metadata_column} JSON,
        created_at TIMESTAMP DEFAULT SYSTIMESTAMP
    )
    """
    with conn.cursor() as cur:
        cur.execute(ddl)

    index_ddl = f"""
    CREATE VECTOR INDEX IF NOT EXISTS idx_{config.table_name}_vec
    ON {config.table_name} ({config.vector_column})
    ORGANIZATION NEIGHBOR PARTITIONS
    DISTANCE {config.distance_metric}
    WITH TARGET ACCURACY 95
    """
    with conn.cursor() as cur:
        cur.execute(index_ddl)
    conn.commit()


def drop_vector_table(conn, table_name: str) -> None:
    """Drop a vector table."""
    validate_identifier(table_name)
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table_name} PURGE")
    conn.commit()
