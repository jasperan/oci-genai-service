"""Chat completions — sync, async, and streaming."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional, Any

from openai.types.chat import ChatCompletion
from oci_openai import OciOpenAI

from oci_genai_service.auth import AuthConfig, create_auth, get_base_url
from oci_genai_service.models import get_model


@dataclass
class ChatResponse:
    """Response from a chat completion."""

    text: str
    model: str
    usage: Optional[dict] = None
    raw: Optional[Any] = None


def chat_completion(
    config: AuthConfig,
    compartment_id: str,
    prompt: str | list[dict],
    model: str,
    system_prompt: Optional[str] = None,
    stream: bool = False,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    tools: Optional[list] = None,
    **kwargs,
) -> ChatResponse | Iterator[str]:
    """Execute a chat completion via the OpenAI-compatible API."""
    auth = create_auth(config)
    client_kwargs = {"base_url": get_base_url(config.region)}

    if config.auth_type == "api_key":
        client_kwargs["api_key"] = config.api_key
        from openai import OpenAI
        openai_client = OpenAI(**client_kwargs)
    else:
        client_kwargs["auth"] = auth
        client_kwargs["compartment_id"] = compartment_id
        openai_client = OciOpenAI(**client_kwargs)

    # Build messages
    if isinstance(prompt, str):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
    else:
        messages = prompt

    request_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": stream,
    }
    if max_tokens:
        request_kwargs["max_tokens"] = max_tokens
    if tools:
        request_kwargs["tools"] = tools
    request_kwargs.update(kwargs)

    if stream:
        return _stream_response(openai_client, request_kwargs)

    response = openai_client.chat.completions.create(**request_kwargs)
    return ChatResponse(
        text=response.choices[0].message.content or "",
        model=response.model,
        usage=dict(response.usage) if response.usage else None,
        raw=response,
    )


def _stream_response(client, request_kwargs) -> Iterator[str]:
    """Yield text chunks from a streaming response."""
    stream = client.chat.completions.create(**request_kwargs)
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
