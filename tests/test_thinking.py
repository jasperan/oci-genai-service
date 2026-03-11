"""Tests for thinking trace extraction."""

import pytest
from oci_genai_service.inference.thinking import extract_thinking, ThinkingResponse


class TestExtractThinking:
    def test_extracts_think_block(self):
        raw = "<think>Step 1: analyze\nStep 2: compute</think>The answer is 42."
        result = extract_thinking(raw)
        assert result.thinking == "Step 1: analyze\nStep 2: compute"
        assert result.answer == "The answer is 42."

    def test_no_think_block(self):
        raw = "The answer is 42."
        result = extract_thinking(raw)
        assert result.thinking is None
        assert result.answer == "The answer is 42."

    def test_empty_think_block(self):
        raw = "<think></think>Answer here."
        result = extract_thinking(raw)
        assert result.thinking == ""
        assert result.answer == "Answer here."
