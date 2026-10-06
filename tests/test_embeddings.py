"""Tests for embeddings via native OCI SDK."""

import pytest
from unittest.mock import patch, MagicMock
from oci_genai_service.auth import AuthConfig
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
            config=AuthConfig(region="us-chicago-1"),
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
                config=AuthConfig(region="us-chicago-1"),
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
            config=AuthConfig(region="us-chicago-1"),
            compartment_id="ocid1.test",
            texts=["hello", "world"],
            model="cohere.embed-english-v3.0",
        )
        details = mock_client.embed_text.call_args[0][0]
        assert details.inputs == ["hello", "world"]


class TestBatching:
    @patch("oci_genai_service.inference.embeddings.GenerativeAiInferenceClient")
    @patch("oci_genai_service.inference.embeddings.oci.config.from_file")
    def test_more_than_96_texts_split_into_requests_and_stay_ordered(self, mock_config, mock_client_cls):
        mock_config.return_value = {}
        mock_client = MagicMock()
        mock_client.embed_text.side_effect = lambda d: MagicMock(
            data=MagicMock(embeddings=[[float(t)] for t in d.inputs])
        )
        mock_client_cls.return_value = mock_client

        texts = [str(i) for i in range(200)]
        result = embed_texts(AuthConfig(), "ocid1.test", texts)

        sizes = [len(c[0][0].inputs) for c in mock_client.embed_text.call_args_list]
        assert sizes == [96, 96, 8]
        assert result.vectors == [[float(i)] for i in range(200)]
        assert result.input_count == 200


class TestAuthTypes:
    @patch("oci_genai_service.inference.embeddings.GenerativeAiInferenceClient")
    @patch("oci_genai_service.inference.embeddings.oci.config.from_file")
    @patch("oci_genai_service.inference.embeddings.oci.auth.signers.InstancePrincipalsSecurityTokenSigner")
    def test_instance_principal_uses_signer_not_config_file(self, mock_signer, mock_from_file, mock_client_cls):
        mock_client_cls.return_value.embed_text.return_value = MagicMock(data=MagicMock(embeddings=[[1.0]]))
        embed_texts(AuthConfig(auth_type="instance_principal"), "ocid1.test", ["hi"])
        mock_from_file.assert_not_called()
        assert mock_client_cls.call_args.kwargs["signer"] is mock_signer.return_value

    @patch("oci_genai_service.inference.embeddings.GenerativeAiInferenceClient")
    @patch("oci_genai_service.inference.embeddings.oci.auth.signers.get_resource_principals_signer")
    def test_resource_principal_uses_signer(self, mock_signer, mock_client_cls):
        mock_client_cls.return_value.embed_text.return_value = MagicMock(data=MagicMock(embeddings=[[1.0]]))
        embed_texts(AuthConfig(auth_type="resource_principal"), "ocid1.test", ["hi"])
        assert mock_client_cls.call_args.kwargs["signer"] is mock_signer.return_value

    def test_unknown_auth_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported auth type"):
            embed_texts(AuthConfig(auth_type="bogus"), "ocid1.test", ["hi"])
