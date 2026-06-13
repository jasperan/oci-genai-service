"""Text embeddings via native OCI SDK (Cohere models)."""

from __future__ import annotations

from dataclasses import dataclass
import oci
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import (
    EmbedTextDetails,
    OnDemandServingMode,
)

from oci_genai_service.auth import AuthConfig, get_host


@dataclass
class EmbeddingResponse:
    """Response from an embedding request."""

    vectors: list[list[float]]
    model: str
    input_count: int


def embed_texts(
    config: AuthConfig,
    compartment_id: str,
    texts: list[str],
    model: str = "cohere.embed-english-v3.0",
    input_type: str = "SEARCH_DOCUMENT",
    truncate: str = "NONE",
) -> EmbeddingResponse:
    """Generate embeddings for a list of texts using the native OCI SDK.

    The returned vectors align 1:1 with ``texts`` by position, so callers may
    safely zip them together. Blank or whitespace-only inputs raise ``ValueError``
    rather than being silently dropped (which would break that alignment).
    """
    if not texts:
        raise ValueError("texts must contain at least one item")
    blank_indices = [i for i, t in enumerate(texts) if not (t and t.strip())]
    if blank_indices:
        raise ValueError(f"texts contains blank entries at indices {blank_indices}")

    oci_config = oci.config.from_file(config.config_file, config.profile_name)
    client = GenerativeAiInferenceClient(
        config=oci_config,
        service_endpoint=get_host(config.region),
    )

    details = EmbedTextDetails(
        compartment_id=compartment_id,
        inputs=texts,
        serving_mode=OnDemandServingMode(model_id=model),
        input_type=input_type,
        truncate=truncate,
    )

    response = client.embed_text(details)
    return EmbeddingResponse(
        vectors=response.data.embeddings,
        model=model,
        input_count=len(texts),
    )
