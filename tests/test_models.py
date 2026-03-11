"""Tests for model registry."""

import pytest
from oci_genai_service.models import ModelInfo, list_models, get_model


class TestModelInfo:
    def test_model_has_required_fields(self):
        model = get_model("meta.llama-4-maverick")
        assert model.id == "meta.llama-4-maverick"
        assert model.vendor == "meta"
        assert "chat" in model.capabilities
        assert model.api == "openai-compatible"
        assert model.context_window > 0
        assert len(model.regions) > 0

    def test_cohere_model_uses_native_api(self):
        model = get_model("cohere.embed-english-v3.0")
        assert model.api == "native"
        assert "embedding" in model.capabilities

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError, match="not found"):
            get_model("nonexistent.model")


class TestListModels:
    def test_list_all(self):
        models = list_models()
        assert len(models) > 10

    def test_filter_by_capability(self):
        vision = list_models(capability="vision")
        assert all("vision" in m.capabilities for m in vision)
        assert len(vision) >= 2

    def test_filter_by_vendor(self):
        meta = list_models(vendor="meta")
        assert all(m.vendor == "meta" for m in meta)

    def test_filter_by_api(self):
        native = list_models(api="native")
        assert all(m.api == "native" for m in native)
