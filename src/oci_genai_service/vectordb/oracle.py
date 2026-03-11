"""OracleVectorStore -- high-level interface for Oracle AI Vector Search."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

import oracledb

from oci_genai_service.vectordb.tables import VectorTableConfig, create_vector_table, validate_identifier


@dataclass
class SearchResult:
    """A single search result."""

    text: str
    score: float
    metadata: Optional[dict] = None


class OracleVectorStore:
    """High-level vector store backed by Oracle 26ai."""

    def __init__(
        self,
        dsn: str,
        user: str,
        password: str,
        table_name: str = "documents",
        vector_dims: int = 1024,
        embedding_model: Optional[str] = None,
        embed_fn: Optional[Callable] = None,
        auto_create_table: bool = True,
    ):
        validate_identifier(table_name)
        self.conn = oracledb.connect(user=user, password=password, dsn=dsn)
        self.table_config = VectorTableConfig(
            table_name=table_name,
            vector_dims=vector_dims,
        )
        self.embedding_model = embedding_model
        self._embed_fn = embed_fn

        if auto_create_table:
            create_vector_table(self.conn, self.table_config)

    def set_embed_fn(self, fn: Callable[[list[str]], list[list[float]]]) -> None:
        """Set the embedding function used to vectorize text."""
        self._embed_fn = fn

    def add_texts(self, texts: list[str], metadatas: Optional[list[dict]] = None) -> list[str]:
        """Add texts to the vector store. Returns list of IDs."""
        if not self._embed_fn:
            raise RuntimeError("No embedding function set. Use set_embed_fn() or pass embed_fn to constructor.")

        vectors = self._embed_fn(texts)
        ids = []

        with self.conn.cursor() as cur:
            for i, (text, vector) in enumerate(zip(texts, vectors)):
                doc_id = uuid.uuid4().hex
                metadata = metadatas[i] if metadatas else None
                cur.execute(
                    f"""INSERT INTO {self.table_config.table_name}
                    ({self.table_config.id_column}, {self.table_config.text_column},
                     {self.table_config.vector_column}, {self.table_config.metadata_column})
                    VALUES (:id, :text, :vector, :metadata)""",
                    {
                        "id": doc_id,
                        "text": text,
                        "vector": vector,
                        "metadata": json.dumps(metadata) if metadata else None,
                    },
                )
                ids.append(doc_id)
        self.conn.commit()
        return ids

    def search(self, query: str, top_k: int = 5, query_vector: Optional[list[float]] = None) -> list[SearchResult]:
        """Search for similar texts."""
        if query_vector is None:
            if not self._embed_fn:
                raise RuntimeError("No embedding function set.")
            query_vector = self._embed_fn([query])[0]

        with self.conn.cursor() as cur:
            cur.execute(
                f"""SELECT {self.table_config.text_column}, {self.table_config.metadata_column},
                    VECTOR_DISTANCE({self.table_config.vector_column}, :qvec, COSINE) AS distance
                FROM {self.table_config.table_name}
                ORDER BY distance ASC
                FETCH FIRST :top_k ROWS ONLY""",
                {"qvec": query_vector, "top_k": top_k},
            )
            results = []
            for row in cur:
                results.append(SearchResult(
                    text=row[0],
                    score=1 - (row[2] or 0),
                    metadata=json.loads(row[1]) if row[1] else None,
                ))
        return results

    def delete(self, ids: list[str]) -> int:
        """Delete documents by ID."""
        with self.conn.cursor() as cur:
            placeholders = ", ".join([f":id{i}" for i in range(len(ids))])
            cur.execute(
                f"DELETE FROM {self.table_config.table_name} WHERE {self.table_config.id_column} IN ({placeholders})",
                {f"id{i}": doc_id for i, doc_id in enumerate(ids)},
            )
            count = cur.rowcount
        self.conn.commit()
        return count

    def count(self) -> int:
        """Return total number of documents."""
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table_config.table_name}")
            return cur.fetchone()[0]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        """Close the database connection."""
        self.conn.close()
