"""Tests for the unified GenAI client."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from oci_genai_service.client import GenAIClient
from oci_genai_service.auth import AuthConfig


class TestGenAIClientInit:
    def test_creates_with_defaults(self):
        client = GenAIClient(compartment_id="ocid1.compartment.oc1..test")
        assert client.config.region == "us-chicago-1"

    def test_creates_with_custom_config(self):
        config = AuthConfig(region="eu-frankfurt-1", profile_name="foosball")
        client = GenAIClient(config=config, compartment_id="ocid1.compartment.oc1..test")
        assert client.config.region == "eu-frankfurt-1"

    def test_creates_with_api_key(self):
        client = GenAIClient(
            api_key="sk-test123",
            compartment_id="ocid1.compartment.oc1..test",
        )
        assert client.config.auth_type == "api_key"


class TestGenAIClientChat:
    @patch("openai.OpenAI")
    def test_chat_returns_text(self, mock_openai_cls):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]
        mock_response.model = "meta.llama-4-maverick"
        mock_response.usage = None
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        client = GenAIClient(api_key="sk-test", compartment_id="ocid1.compartment.oc1..test")
        result = client.chat("Hi", model="meta.llama-4-maverick")
        assert result.text == "Hello!"

    @patch("openai.OpenAI")
    def test_chat_with_system_prompt(self, mock_openai_cls):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="I am helpful"))]
        mock_response.model = "meta.llama-4-maverick"
        mock_response.usage = None
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        client = GenAIClient(api_key="sk-test", compartment_id="ocid1.compartment.oc1..test")
        result = client.chat("Who are you?", model="meta.llama-4-maverick", system_prompt="You are helpful")

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful"
