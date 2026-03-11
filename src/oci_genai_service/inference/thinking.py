"""Thinking trace extraction for reasoning models (Grok Code, GPT-OSS)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ThinkingResponse:
    """Parsed response separating thinking traces from the answer."""

    thinking: Optional[str]
    answer: str


_THINK_PATTERN = re.compile(r"<think>(.*?)</think>(.*)", re.DOTALL)


def extract_thinking(raw_text: str) -> ThinkingResponse:
    """Extract thinking traces from model output.

    Models like Grok Code and GPT-OSS may include <think>...</think> blocks.
    """
    match = _THINK_PATTERN.match(raw_text.strip())
    if match:
        return ThinkingResponse(
            thinking=match.group(1).strip() if match.group(1).strip() else match.group(1),
            answer=match.group(2).strip(),
        )
    return ThinkingResponse(thinking=None, answer=raw_text.strip())
