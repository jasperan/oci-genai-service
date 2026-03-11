"""Model registry with metadata for all OCI GenAI models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


ApiType = Literal["openai-compatible", "native"]


@dataclass(frozen=True)
class ModelInfo:
    """Metadata for an OCI GenAI model."""

    id: str
    name: str
    vendor: str
    capabilities: tuple[str, ...]
    api: ApiType
    context_window: int
    regions: tuple[str, ...] = ("us-chicago-1",)
    embedding_dims: Optional[int] = None


_MODELS: dict[str, ModelInfo] = {}


def _register(m: ModelInfo):
    _MODELS[m.id] = m


# --- Meta Llama ---
_register(ModelInfo(
    id="meta.llama-4-scout", name="Llama 4 Scout", vendor="meta",
    capabilities=("chat", "function_calling", "streaming"),
    api="openai-compatible", context_window=131072,
    regions=("us-chicago-1", "eu-frankfurt-1", "ap-osaka-1"),
))
_register(ModelInfo(
    id="meta.llama-4-maverick", name="Llama 4 Maverick", vendor="meta",
    capabilities=("chat", "function_calling", "streaming"),
    api="openai-compatible", context_window=131072,
    regions=("us-chicago-1", "eu-frankfurt-1", "ap-osaka-1"),
))
_register(ModelInfo(
    id="meta.llama-3.3-70b-instruct", name="Llama 3.3 70B Instruct", vendor="meta",
    capabilities=("chat", "function_calling", "streaming"),
    api="openai-compatible", context_window=131072,
    regions=("us-chicago-1", "eu-frankfurt-1", "ap-osaka-1", "us-ashburn-1"),
))
_register(ModelInfo(
    id="meta.llama-3.2-90b-vision-instruct", name="Llama 3.2 90B Vision", vendor="meta",
    capabilities=("chat", "vision", "streaming"),
    api="openai-compatible", context_window=131072,
    regions=("us-chicago-1",),
))
_register(ModelInfo(
    id="meta.llama-3.2-11b-vision-instruct", name="Llama 3.2 11B Vision", vendor="meta",
    capabilities=("chat", "vision", "streaming"),
    api="openai-compatible", context_window=131072,
    regions=("us-chicago-1",),
))
_register(ModelInfo(
    id="meta.llama-3.1-405b-instruct", name="Llama 3.1 405B Instruct", vendor="meta",
    capabilities=("chat", "function_calling", "streaming"),
    api="openai-compatible", context_window=131072,
    regions=("us-chicago-1",),
))
_register(ModelInfo(
    id="meta.llama-3.1-70b-instruct", name="Llama 3.1 70B Instruct", vendor="meta",
    capabilities=("chat", "function_calling", "streaming"),
    api="openai-compatible", context_window=131072,
    regions=("us-chicago-1", "eu-frankfurt-1"),
))

# --- xAI Grok ---
_register(ModelInfo(
    id="xai.grok-4.1-fast", name="Grok 4.1 Fast", vendor="xai",
    capabilities=("chat", "streaming"),
    api="openai-compatible", context_window=131072,
    regions=("us-chicago-1",),
))
_register(ModelInfo(
    id="xai.grok-4", name="Grok 4", vendor="xai",
    capabilities=("chat", "streaming"),
    api="openai-compatible", context_window=131072,
    regions=("us-chicago-1",),
))
_register(ModelInfo(
    id="xai.grok-4-fast", name="Grok 4 Fast", vendor="xai",
    capabilities=("chat", "streaming"),
    api="openai-compatible", context_window=131072,
    regions=("us-chicago-1",),
))
_register(ModelInfo(
    id="xai.grok-3", name="Grok 3", vendor="xai",
    capabilities=("chat", "streaming"),
    api="openai-compatible", context_window=131072,
    regions=("us-chicago-1",),
))
_register(ModelInfo(
    id="xai.grok-3-mini", name="Grok 3 Mini", vendor="xai",
    capabilities=("chat", "streaming"),
    api="openai-compatible", context_window=131072,
    regions=("us-chicago-1",),
))
_register(ModelInfo(
    id="xai.grok-3-mini-fast", name="Grok 3 Mini Fast", vendor="xai",
    capabilities=("chat", "streaming"),
    api="openai-compatible", context_window=131072,
    regions=("us-chicago-1",),
))
_register(ModelInfo(
    id="xai.grok-code-fast-1", name="Grok Code Fast 1", vendor="xai",
    capabilities=("chat", "streaming", "thinking"),
    api="openai-compatible", context_window=131072,
    regions=("us-chicago-1",),
))

# --- OpenAI gpt-oss ---
_register(ModelInfo(
    id="openai.gpt-oss-120b", name="GPT-OSS 120B", vendor="openai",
    capabilities=("chat", "streaming", "thinking"),
    api="openai-compatible", context_window=131072,
    regions=("us-chicago-1", "eu-frankfurt-1"),
))
_register(ModelInfo(
    id="openai.gpt-oss-20b", name="GPT-OSS 20B", vendor="openai",
    capabilities=("chat", "streaming"),
    api="openai-compatible", context_window=131072,
    regions=("us-chicago-1",),
))

# --- Cohere (native API only) ---
_register(ModelInfo(
    id="cohere.command-a", name="Command A", vendor="cohere",
    capabilities=("chat", "streaming"),
    api="native", context_window=262144,
    regions=("us-chicago-1", "eu-frankfurt-1"),
))
_register(ModelInfo(
    id="cohere.command-r-plus", name="Command R+", vendor="cohere",
    capabilities=("chat", "streaming"),
    api="native", context_window=131072,
    regions=("us-chicago-1",),
))
_register(ModelInfo(
    id="cohere.command-r", name="Command R", vendor="cohere",
    capabilities=("chat", "streaming"),
    api="native", context_window=131072,
    regions=("us-chicago-1",),
))
_register(ModelInfo(
    id="cohere.embed-english-v3.0", name="Embed English v3.0", vendor="cohere",
    capabilities=("embedding",),
    api="native", context_window=512, embedding_dims=1024,
    regions=("us-chicago-1", "eu-frankfurt-1"),
))
_register(ModelInfo(
    id="cohere.embed-multilingual-v3.0", name="Embed Multilingual v3.0", vendor="cohere",
    capabilities=("embedding",),
    api="native", context_window=512, embedding_dims=1024,
    regions=("us-chicago-1", "eu-frankfurt-1"),
))
_register(ModelInfo(
    id="cohere.embed-english-light-v3.0", name="Embed English Light v3.0", vendor="cohere",
    capabilities=("embedding",),
    api="native", context_window=512, embedding_dims=384,
    regions=("us-chicago-1",),
))
_register(ModelInfo(
    id="cohere.embed-multilingual-light-v3.0", name="Embed Multilingual Light v3.0", vendor="cohere",
    capabilities=("embedding",),
    api="native", context_window=512, embedding_dims=384,
    regions=("us-chicago-1",),
))


def get_model(model_id: str) -> ModelInfo:
    """Get model metadata by ID. Raises KeyError if not found."""
    if model_id not in _MODELS:
        raise KeyError(f"Model '{model_id}' not found in registry. Use list_models() to see available models.")
    return _MODELS[model_id]


def list_models(
    vendor: Optional[str] = None,
    capability: Optional[str] = None,
    api: Optional[ApiType] = None,
) -> list[ModelInfo]:
    """List models with optional filters."""
    models = list(_MODELS.values())
    if vendor:
        models = [m for m in models if m.vendor == vendor]
    if capability:
        models = [m for m in models if capability in m.capabilities]
    if api:
        models = [m for m in models if m.api == api]
    return models
