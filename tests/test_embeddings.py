"""Tests for embeddings via native OCI SDK."""

import pytest
from unittest.mock import patch, MagicMock
from oci_genai_service.inference.embeddings import embed_texts


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
    def test_blank_inputs_raise(self, mock_config, mock_client_cls):
        # Blank entries must raise rather than be silently dropped, otherwise the
        # returned vectors would no longer align 1:1 with the caller's list.
        with pytest.raises(ValueError, match="blank entries at indices"):
            embed_texts(
                config=MagicMock(config_file="~/.oci/config", profile_name="DEFAULT", region="us-chicago-1"),
                compartment_id="ocid1.test",
                texts=["hello", "", "  "],
                model="cohere.embed-english-v3.0",
            )
        mock_client_cls.assert_not_called()

    @patch("oci_genai_service.inference.embeddings.GenerativeAiInferenceClient")
    @patch("oci_genai_service.inference.embeddings.oci.config.from_file")
    def test_all_inputs_passed_in_order(self, mock_config, mock_client_cls):
        mock_config.return_value = {}
        mock_response = MagicMock()
        mock_response.data.embeddings = [[0.1, 0.2], [0.3, 0.4]]
        mock_client = MagicMock()
        mock_client.embed_text.return_value = mock_response
        mock_client_cls.return_value = mock_client

        embed_texts(
            config=MagicMock(config_file="~/.oci/config", profile_name="DEFAULT", region="us-chicago-1"),
            compartment_id="ocid1.test",
            texts=["hello", "world"],
            model="cohere.embed-english-v3.0",
        )
        details = mock_client.embed_text.call_args[0][0]
        assert details.inputs == ["hello", "world"]
