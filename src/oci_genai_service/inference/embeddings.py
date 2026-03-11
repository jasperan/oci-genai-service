"""Text embeddings via native OCI SDK (Cohere models)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import oci
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import (
    EmbedTextDetails,
    OnDemandServingMode,
)

from oci_genai_service.auth import AuthConfig


EMBED_ENDPOINT_TEMPLATE = "https://inference.generativeai.{region}.oci.oraclecloud.com"


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
    """Generate embeddings for a list of texts using the native OCI SDK."""
    clean_texts = [t for t in texts if t and t.strip()]

    oci_config = oci.config.from_file(config.config_file, config.profile_name)
    client = GenerativeAiInferenceClient(
        config=oci_config,
        service_endpoint=EMBED_ENDPOINT_TEMPLATE.format(region=config.region),
    )

    details = EmbedTextDetails(
        compartment_id=compartment_id,
        inputs=clean_texts,
        serving_mode=OnDemandServingMode(model_id=model),
        input_type=input_type,
        truncate=truncate,
    )

    response = client.embed_text(details)
    return EmbeddingResponse(
        vectors=response.data.embeddings,
        model=model,
        input_count=len(clean_texts),
    )
