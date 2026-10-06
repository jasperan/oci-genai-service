"""Tests for RAG pipeline."""

import sys
import types
from unittest.mock import patch, MagicMock
from oci_genai_service.rag.chunkers import RecursiveChunker, Chunk
from oci_genai_service.rag.loaders import load_document
from oci_genai_service.rag.pipeline import RAGPipeline, RAGResponse


class TestRecursiveChunker:
    def test_splits_text(self):
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        text = "A" * 120
        chunks = chunker.chunk(text, source="test.txt")
        assert len(chunks) >= 2
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(len(c.text) <= 50 for c in chunks)

    def test_preserves_source_metadata(self):
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
        chunks = chunker.chunk("Hello world", source="doc.pdf", metadata={"page": 1})
        assert chunks[0].source == "doc.pdf"
        assert chunks[0].metadata["page"] == 1

    def test_empty_text_returns_empty(self):
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
        chunks = chunker.chunk("", source="empty.txt")
        assert len(chunks) == 0

    def test_no_duplicate_tail_chunk(self):
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        text = "".join(chr(97 + i % 26) for i in range(130))
        chunks = chunker.chunk(text, source="t")
        assert len(chunks) == 2
        assert chunks[0].text + chunks[1].text[20:] == text

    def test_covers_whole_text_and_terminates(self):
        chunker = RecursiveChunker(chunk_size=30, chunk_overlap=10)
        text = " ".join(f"word{i}" for i in range(100))
        chunks = chunker.chunk(text, source="t")
        assert text.endswith(chunks[-1].text)
        assert chunks[0].text == text[: len(chunks[0].text)]

    def test_overlap_must_be_smaller_than_size(self):
        import pytest
        with pytest.raises(ValueError):
            RecursiveChunker(chunk_size=10, chunk_overlap=10)


class TestPipelineQuery:
    def test_query_embeds_with_search_query_input_type(self):
        client = MagicMock()
        client.embed.return_value.vectors = [[0.5, 0.5]]
        client.chat.return_value.text = "answer"
        store = MagicMock()
        store.search.return_value = []
        pipeline = RAGPipeline(client=client, vector_store=store)

        pipeline.query("what?")

        assert client.embed.call_args.kwargs["input_type"] == "SEARCH_QUERY"
        assert store.search.call_args.kwargs["query_vector"] == [0.5, 0.5]


class TestLoadDocument:
    def test_loads_pdf_with_docling(self):
        mock_converter = MagicMock()
        mock_doc = MagicMock()
        mock_doc.export_to_markdown.return_value = "# Title\n\nSome content here."
        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_converter.convert.return_value = mock_result

        mock_converter_cls = MagicMock(return_value=mock_converter)

        # Create a fake docling.document_converter module so the lazy import works
        fake_module = types.ModuleType("docling.document_converter")
        fake_module.DocumentConverter = mock_converter_cls

        with patch.dict(sys.modules, {
            "docling": types.ModuleType("docling"),
            "docling.document_converter": fake_module,
        }):
            text = load_document("test.pdf")
            assert "Some content here" in text
            mock_converter.convert.assert_called_once_with("test.pdf")

    def test_loads_plain_text(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello world")
        text = load_document(str(f))
        assert text == "Hello world"


class TestRAGResponse:
    def test_has_required_fields(self):
        resp = RAGResponse(
            text="The answer is 42.",
            sources=[{"chunk": "some text", "score": 0.95}],
        )
        assert resp.text == "The answer is 42."
        assert len(resp.sources) == 1
