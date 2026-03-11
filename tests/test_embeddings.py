"""Tests for embeddings via native OCI SDK."""

import pytest
from unittest.mock import patch, MagicMock
from oci_genai_service.inference.embeddings import embed_texts, EmbeddingResponse


class TestEmbedTexts:
    @patch("oci_genai_service.inference.embeddings.GenerativeAiInferenceClient")
    @patch("oci_genai_service.inference.embeddings.oci.config.from_file")
    def test_returns_vectors(self, mock_config, mock_client_cls):
        mock_config.return_value = {}
        mock_response = MagicMock()
        mock_response.data.embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_client = MagicMock()
        mock_client.embed_text.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = embed_texts(
            config=MagicMock(config_file="~/.oci/config", profile_name="DEFAULT", region="us-chicago-1"),
            compartment_id="ocid1.test",
            texts=["hello", "world"],
            model="cohere.embed-english-v3.0",
        )
        assert len(result.vectors) == 2
        assert result.vectors[0] == [0.1, 0.2, 0.3]

    @patch("oci_genai_service.inference.embeddings.GenerativeAiInferenceClient")
    @patch("oci_genai_service.inference.embeddings.oci.config.from_file")
    def test_filters_empty_strings(self, mock_config, mock_client_cls):
        mock_config.return_value = {}
        mock_response = MagicMock()
        mock_response.data.embeddings = [[0.1, 0.2]]
        mock_client = MagicMock()
        mock_client.embed_text.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = embed_texts(
            config=MagicMock(config_file="~/.oci/config", profile_name="DEFAULT", region="us-chicago-1"),
            compartment_id="ocid1.test",
            texts=["hello", "", "  "],
            model="cohere.embed-english-v3.0",
        )
        # Verify only non-empty texts were sent
        call_args = mock_client.embed_text.call_args
        details = call_args[0][0]
        assert len(details.inputs) == 1
