"""Tests for vision/multimodal support."""

import pytest
from oci_genai_service.inference.vision import build_vision_messages


class TestBuildVisionMessages:
    def test_single_image_url(self):
        messages = build_vision_messages("Describe this", images=["https://example.com/img.jpg"])
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"

    def test_single_image_base64(self, tmp_path):
        # Create a tiny test image file
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
        messages = build_vision_messages("Describe this", images=[str(img)])
        content = messages[0]["content"]
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/")

    def test_multiple_images(self):
        messages = build_vision_messages("Compare", images=["https://a.com/1.jpg", "https://b.com/2.jpg"])
        content = messages[0]["content"]
        assert len(content) == 3  # text + 2 images

    def test_system_prompt_included(self):
        messages = build_vision_messages("Describe", images=["https://a.com/1.jpg"], system_prompt="Be brief")
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
