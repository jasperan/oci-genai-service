"""Vision/multimodal support for image+text prompts."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Optional


def build_vision_messages(
    prompt: str,
    images: list[str],
    system_prompt: Optional[str] = None,
) -> list[dict]:
    """Build OpenAI-format messages with image content."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    content = [{"type": "text", "text": prompt}]
    for image in images:
        content.append({"type": "image_url", "image_url": {"url": _resolve_image(image)}})

    messages.append({"role": "user", "content": content})
    return messages


def _resolve_image(image: str) -> str:
    """Convert image path or URL to a format the API accepts."""
    if image.startswith(("http://", "https://")):
        return image
    path = Path(image)
    mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"
