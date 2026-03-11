"""Retriever — fetches relevant context from the vector store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from oci_genai_service.vectordb.oracle import OracleVectorStore, SearchResult


@dataclass
class RetrievalResult:
    """Retrieved context with scores."""

    chunks: list[SearchResult]
    query: str


class Retriever:
    """Retrieves relevant context from an OracleVectorStore."""

    def __init__(self, store: OracleVectorStore, top_k: int = 5):
        self.store = store
        self.top_k = top_k

    def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """Retrieve the most relevant chunks for a query."""
        k = top_k or self.top_k
        results = self.store.search(query, top_k=k)
        return RetrievalResult(chunks=results, query=query)
