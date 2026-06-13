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


# Region shorthands keep the table below readable.
_US = ("us-chicago-1",)
_US_EU = ("us-chicago-1", "eu-frankfurt-1")
_US_EU_AP = ("us-chicago-1", "eu-frankfurt-1", "ap-osaka-1")
_US_EU_AP_ASH = ("us-chicago-1", "eu-frankfurt-1", "ap-osaka-1", "us-ashburn-1")

# Common capability bundles.
_CHAT_FN = ("chat", "function_calling", "streaming")
_CHAT = ("chat", "streaming")
_CHAT_VISION = ("chat", "vision", "streaming")
_CHAT_THINK = ("chat", "streaming", "thinking")

# Each vendor block supplies defaults (vendor, api, context_window); each row
# overrides only what differs. Rows are (id, name, capabilities, regions, **extra).
_REGISTRY: list[tuple[dict, list[tuple]]] = [
    (
        {"vendor": "meta", "api": "openai-compatible", "context_window": 131072},
        [
            ("meta.llama-4-scout", "Llama 4 Scout", _CHAT_FN, _US_EU_AP),
            ("meta.llama-4-maverick", "Llama 4 Maverick", _CHAT_FN, _US_EU_AP),
            ("meta.llama-3.3-70b-instruct", "Llama 3.3 70B Instruct", _CHAT_FN, _US_EU_AP_ASH),
            ("meta.llama-3.2-90b-vision-instruct", "Llama 3.2 90B Vision", _CHAT_VISION, _US),
            ("meta.llama-3.2-11b-vision-instruct", "Llama 3.2 11B Vision", _CHAT_VISION, _US),
            ("meta.llama-3.1-405b-instruct", "Llama 3.1 405B Instruct", _CHAT_FN, _US),
            ("meta.llama-3.1-70b-instruct", "Llama 3.1 70B Instruct", _CHAT_FN, _US_EU),
        ],
    ),
    (
        {"vendor": "xai", "api": "openai-compatible", "context_window": 131072},
        [
            ("xai.grok-4.1-fast", "Grok 4.1 Fast", _CHAT, _US),
            ("xai.grok-4", "Grok 4", _CHAT, _US),
            ("xai.grok-4-fast", "Grok 4 Fast", _CHAT, _US),
            ("xai.grok-3", "Grok 3", _CHAT, _US),
            ("xai.grok-3-mini", "Grok 3 Mini", _CHAT, _US),
            ("xai.grok-3-mini-fast", "Grok 3 Mini Fast", _CHAT, _US),
            ("xai.grok-code-fast-1", "Grok Code Fast 1", _CHAT_THINK, _US),
        ],
    ),
    (
        {"vendor": "openai", "api": "openai-compatible", "context_window": 131072},
        [
            ("openai.gpt-oss-120b", "GPT-OSS 120B", _CHAT_THINK, _US_EU),
            ("openai.gpt-oss-20b", "GPT-OSS 20B", _CHAT, _US),
        ],
    ),
    (
        # Cohere is native-API only; context_window and embedding_dims vary per row.
        {"vendor": "cohere", "api": "native"},
        [
            ("cohere.command-a", "Command A", _CHAT, _US_EU, {"context_window": 262144}),
            ("cohere.command-r-plus", "Command R+", _CHAT, _US, {"context_window": 131072}),
            ("cohere.command-r", "Command R", _CHAT, _US, {"context_window": 131072}),
            ("cohere.embed-english-v3.0", "Embed English v3.0", ("embedding",), _US_EU,
             {"context_window": 512, "embedding_dims": 1024}),
            ("cohere.embed-multilingual-v3.0", "Embed Multilingual v3.0", ("embedding",), _US_EU,
             {"context_window": 512, "embedding_dims": 1024}),
            ("cohere.embed-english-light-v3.0", "Embed English Light v3.0", ("embedding",), _US,
             {"context_window": 512, "embedding_dims": 384}),
            ("cohere.embed-multilingual-light-v3.0", "Embed Multilingual Light v3.0", ("embedding",), _US,
             {"context_window": 512, "embedding_dims": 384}),
        ],
    ),
]


def _build_registry() -> dict[str, ModelInfo]:
    models: dict[str, ModelInfo] = {}
    for defaults, rows in _REGISTRY:
        for row in rows:
            model_id, name, capabilities, regions = row[:4]
            extra = row[4] if len(row) > 4 else {}
            models[model_id] = ModelInfo(
                id=model_id, name=name, capabilities=capabilities, regions=regions,
                **defaults, **extra,
            )
    return models


_MODELS: dict[str, ModelInfo] = _build_registry()


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
