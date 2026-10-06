"""Text embeddings via native OCI SDK (Cohere models)."""

from __future__ import annotations

import os
from dataclasses import dataclass

import oci
import oci.auth.signers
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import (
    EmbedTextDetails,
    OnDemandServingMode,
)

from oci_genai_service.auth import AuthConfig, get_host


# OCI GenAI embed_text accepts at most 96 inputs per request.
MAX_BATCH_SIZE = 96


@dataclass
class EmbeddingResponse:
    """Response from an embedding request."""

    vectors: list[list[float]]
    model: str
    input_count: int


def _build_client(config: AuthConfig) -> GenerativeAiInferenceClient:
    """Build a native inference client honouring ``config.auth_type``.

    ``api_key`` auth only applies to the OpenAI-compatible chat endpoint; the
    native embeddings endpoint has no API-key mode, so it uses the OCI config file.
    """
    endpoint = get_host(config.region)
    auth_type = config.auth_type

    if auth_type == "instance_principal":
        signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
        return GenerativeAiInferenceClient(config={"region": config.region}, signer=signer, service_endpoint=endpoint)
    if auth_type == "resource_principal":
        signer = oci.auth.signers.get_resource_principals_signer()
        return GenerativeAiInferenceClient(config={"region": config.region}, signer=signer, service_endpoint=endpoint)
    if auth_type in ("user_principal", "session", "api_key"):
        oci_config = oci.config.from_file(config.config_file, config.profile_name)
        if auth_type == "session":
            with open(os.path.expanduser(oci_config["security_token_file"])) as f:
                token = f.read().strip()
            private_key = oci.signer.load_private_key_from_file(os.path.expanduser(oci_config["key_file"]))
            signer = oci.auth.signers.SecurityTokenSigner(token, private_key)
            return GenerativeAiInferenceClient(config=oci_config, signer=signer, service_endpoint=endpoint)
        return GenerativeAiInferenceClient(config=oci_config, service_endpoint=endpoint)
    raise ValueError(f"Unsupported auth type: {auth_type}")


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

    client = _build_client(config)

    vectors: list[list[float]] = []
    for start in range(0, len(texts), MAX_BATCH_SIZE):
        details = EmbedTextDetails(
            compartment_id=compartment_id,
            inputs=texts[start:start + MAX_BATCH_SIZE],
            serving_mode=OnDemandServingMode(model_id=model),
            input_type=input_type,
            truncate=truncate,
        )
        vectors.extend(client.embed_text(details).data.embeddings)

    return EmbeddingResponse(
        vectors=vectors,
        model=model,
        input_count=len(texts),
    )
