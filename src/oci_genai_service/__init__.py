"""OCI GenAI Service — Production-ready Python toolkit for Oracle Cloud Generative AI."""

__version__ = "2.0.0"

from oci_genai_service.client import GenAIClient
from oci_genai_service.auth import AuthConfig
from oci_genai_service.models import list_models, get_model

__all__ = ["GenAIClient", "AuthConfig", "list_models", "get_model"]
