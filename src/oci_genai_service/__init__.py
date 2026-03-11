"""OCI GenAI Service — Production-ready Python toolkit for Oracle Cloud Generative AI."""

__version__ = "2.0.1"

from oci_genai_service.client import GenAIClient
from oci_genai_service.auth import AuthConfig
from oci_genai_service.models import list_models, get_model, ModelInfo
from oci_genai_service.inference.tools import tool
from oci_genai_service.inference.chat import ChatResponse
from oci_genai_service.inference.embeddings import EmbeddingResponse
from oci_genai_service.inference.thinking import extract_thinking, ThinkingResponse
from oci_genai_service.vectordb.oracle import OracleVectorStore
from oci_genai_service.rag.pipeline import RAGPipeline, RAGResponse
from oci_genai_service.agents.agent import Agent, AgentResponse
from oci_genai_service.guardrails.moderation import Guardrail, GuardrailResult

__all__ = [
    "GenAIClient", "AuthConfig",
    "list_models", "get_model", "ModelInfo",
    "tool", "ChatResponse", "EmbeddingResponse",
    "extract_thinking", "ThinkingResponse",
    "OracleVectorStore",
    "RAGPipeline", "RAGResponse",
    "Agent", "AgentResponse",
    "Guardrail", "GuardrailResult",
]
