"""Text chunking strategies for RAG pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    """A text chunk with source metadata."""

    text: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


class RecursiveChunker:
    """Recursive text splitter with configurable size and overlap."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, source: str, metadata: Optional[dict] = None) -> list[Chunk]:
        """Split text into chunks with overlap."""
        if not text or not text.strip():
            return []

        meta = metadata or {}
        chunks = []
        start = 0
        idx = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # Try to break at sentence/paragraph boundary
            if end < len(text):
                for sep in ["\n\n", "\n", ". ", " "]:
                    last_sep = chunk_text.rfind(sep)
                    if last_sep > self.chunk_size // 2:
                        chunk_text = chunk_text[:last_sep + len(sep)]
                        break

            chunks.append(Chunk(
                text=chunk_text,
                source=source,
                chunk_index=idx,
                metadata={**meta},
            ))
            idx += 1
            start += len(chunk_text) - self.chunk_overlap
            if len(chunk_text) <= self.chunk_overlap:
                break

        return chunks
