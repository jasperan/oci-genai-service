"""Vector table creation and management for Oracle 26ai."""

from __future__ import annotations

from dataclasses import dataclass


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
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table_name} PURGE")
    conn.commit()
