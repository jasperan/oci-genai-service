"""Hybrid vector + keyword search for Oracle AI Vector Search."""

from __future__ import annotations


def hybrid_search(
    table_name: str,
    top_k: int = 5,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
    text_column: str = "text",
    vector_column: str = "embedding",
    metadata_column: str = "metadata",
) -> str:
    """Build a hybrid search SQL query combining vector similarity and keyword matching.

    Returns the SQL string (caller provides bind variables for query_vector and query_text).
    """
    return f"""
    SELECT {text_column}, {metadata_column},
        ({vector_weight} * (1 - VECTOR_DISTANCE({vector_column}, :query_vector, COSINE))
         + {keyword_weight} * CASE WHEN LOWER({text_column}) LIKE LOWER('%' || :query_text || '%') THEN 1 ELSE 0 END
        ) AS score
    FROM {table_name}
    ORDER BY score DESC
    FETCH FIRST {top_k} ROWS ONLY
    """
