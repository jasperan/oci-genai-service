"""End-to-end RAG pipeline: ingest -> chunk -> embed -> store -> retrieve -> generate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from oci_genai_service.client import GenAIClient
from oci_genai_service.rag.loaders import load_document
from oci_genai_service.rag.chunkers import RecursiveChunker
from oci_genai_service.rag.retriever import Retriever
from oci_genai_service.vectordb.oracle import OracleVectorStore


@dataclass
class RAGResponse:
    """Response from a RAG query with sources."""

    text: str
    sources: list[dict] = field(default_factory=list)


class RAGPipeline:
    """Opinionated RAG pipeline with Docling + Oracle Vector Search."""

    def __init__(
        self,
        client: GenAIClient,
        vector_store: OracleVectorStore,
        chat_model: str = "meta.llama-4-maverick",
        embedding_model: str = "cohere.embed-english-v3.0",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 5,
    ):
        self.client = client
        self.store = vector_store
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.retriever = Retriever(store=vector_store, top_k=top_k)

        def _embed(texts: list[str]) -> list[list[float]]:
            result = self.client.embed(texts, model=self.embedding_model, input_type="SEARCH_DOCUMENT")
            return result.vectors

        self.store.set_embed_fn(_embed)

    def ingest(self, file_path: str, metadata: Optional[dict] = None) -> int:
        """Ingest a document: load -> chunk -> embed -> store. Returns chunk count."""
        text = load_document(file_path)
        chunks = self.chunker.chunk(text, source=file_path, metadata=metadata or {})

        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        metadatas = [{**c.metadata, "source": c.source, "chunk_index": c.chunk_index} for c in chunks]

        self.store.add_texts(texts, metadatas=metadatas)
        return len(chunks)

    def query(self, question: str, system_prompt: Optional[str] = None) -> RAGResponse:
        """Query the RAG pipeline: retrieve -> generate with context."""
        retrieval = self.retriever.retrieve(question)

        context_parts = []
        sources = []
        for result in retrieval.chunks:
            context_parts.append(result.text)
            sources.append({
                "chunk": result.text[:200],
                "score": result.score,
                "metadata": result.metadata,
            })

        context = "\n\n---\n\n".join(context_parts)

        rag_prompt = f"""Answer the question based on the following context. If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}"""

        response = self.client.chat(
            prompt=rag_prompt,
            model=self.chat_model,
            system_prompt=system_prompt or "You are a helpful assistant that answers questions based on provided context. Cite your sources.",
        )

        return RAGResponse(text=response.text, sources=sources)
