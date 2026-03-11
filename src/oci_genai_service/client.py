"""Unified GenAI client — the main entry point for oci-genai-service."""

from __future__ import annotations

from typing import Iterator, Optional

from oci_genai_service.auth import AuthConfig, create_auth
from oci_genai_service.inference.chat import ChatResponse, chat_completion


class GenAIClient:
    """Unified client for OCI GenAI Service.

    Auto-routes between oci-openai (OpenAI-compatible models) and native OCI SDK (Cohere).

    Usage:
        client = GenAIClient(compartment_id="ocid1.compartment.oc1..xxx")
        response = client.chat("Hello!", model="meta.llama-4-maverick")
        print(response.text)
    """

    def __init__(
        self,
        config: Optional[AuthConfig] = None,
        compartment_id: Optional[str] = None,
        api_key: Optional[str] = None,
        region: Optional[str] = None,
        profile_name: Optional[str] = None,
    ):
        if config:
            self.config = config
        elif api_key:
            self.config = AuthConfig(auth_type="api_key", api_key=api_key)
        else:
            self.config = AuthConfig.from_env()

        if region:
            self.config.region = region
        if profile_name:
            self.config.profile_name = profile_name
        if compartment_id:
            self.config.compartment_id = compartment_id

    def chat(
        self,
        prompt: str | list[dict],
        model: str = "meta.llama-4-maverick",
        system_prompt: Optional[str] = None,
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[list] = None,
        images: Optional[list[str]] = None,
        **kwargs,
    ) -> ChatResponse | Iterator[str]:
        """Send a chat completion request.

        Args:
            prompt: User message string or list of message dicts.
            model: Model ID (e.g. "meta.llama-4-maverick").
            system_prompt: Optional system prompt.
            stream: If True, returns an iterator of text chunks.
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens in response.
            tools: List of tool definitions for function calling.
            images: List of image paths/URLs for vision models.

        Returns:
            ChatResponse (non-streaming) or Iterator[str] (streaming).
        """
        if images:
            from oci_genai_service.inference.vision import build_vision_messages
            actual_prompt = build_vision_messages(
                prompt if isinstance(prompt, str) else prompt[-1]["content"],
                images=images,
                system_prompt=system_prompt,
            )
            system_prompt = None  # already included in vision messages
        else:
            actual_prompt = prompt

        return chat_completion(
            config=self.config,
            compartment_id=self.config.compartment_id,
            prompt=actual_prompt,
            model=model,
            system_prompt=system_prompt,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            **kwargs,
        )

    def embed(
        self,
        texts: list[str],
        model: str = "cohere.embed-english-v3.0",
        input_type: str = "SEARCH_DOCUMENT",
    ) -> "EmbeddingResponse":
        """Generate text embeddings via native OCI SDK."""
        from oci_genai_service.inference.embeddings import embed_texts, EmbeddingResponse
        return embed_texts(
            config=self.config,
            compartment_id=self.config.compartment_id,
            texts=texts,
            model=model,
            input_type=input_type,
        )
