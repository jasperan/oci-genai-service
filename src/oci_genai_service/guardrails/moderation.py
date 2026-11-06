"""Heuristic local guardrails — PII detection and prompt-injection screening.

These are lightweight, dependency-free checks (substring and regex matching)
that run entirely client-side. They do not call any OCI service.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

# Separators that may appear between the digits of one number.
_DIGIT_SEPARATOR_RE = re.compile(r"(?<=\d)[\s.\-]+(?=\d)")
_SSN_DASHED_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_SSN_DIGITS_RE = re.compile(r"(?<!\d)\d{9}(?!\d)")
_CARD_DIGITS_RE = re.compile(r"(?<!\d)\d{13,19}(?!\d)")

# Instruction-override phrasing, matched after whitespace/punctuation normalization.
_INJECTION_PATTERNS = (
    re.compile(
        r"\b(ignore|disregard|forget)\b.{0,40}\b(previous|prior|above|all|your)\b"
        r".{0,20}\b(instruction|instructions|prompt|prompts|rule|rules|directive|directives)\b"
    ),
    re.compile(r"\byou are now\b"),
    re.compile(r"\bnew instructions\b"),
)


def _digits_only(text: str) -> str:
    """Collapse separators between digits so grouping does not change detection."""
    return _DIGIT_SEPARATOR_RE.sub("", text)


def _luhn_valid(digits: str) -> bool:
    """Return True when a digit string satisfies the Luhn checksum."""
    total = 0
    for index, character in enumerate(reversed(digits)):
        digit = ord(character) - ord("0")
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _normalize_for_matching(text: str) -> str:
    """NFKC-normalize, lowercase, replace punctuation with spaces, collapse whitespace."""
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""

    blocked: bool
    reason: Optional[str] = None
    flags: list[str] = field(default_factory=list)


class Guardrail:
    """Heuristic content-safety guardrail (local, no OCI calls).

    Usage:
        guard = Guardrail(pii_detection=True, prompt_injection=True)
        result = guard.check("some user input")
        if result.blocked:
            print(result.reason)
    """

    def __init__(
        self,
        pii_detection: bool = False,
        prompt_injection: bool = False,
    ):
        self.pii_detection = pii_detection
        self.prompt_injection = prompt_injection

    def check(self, text: str) -> GuardrailResult:
        """Check text against enabled guardrails."""
        flags = []
        blocked = False
        reasons = []

        if self.prompt_injection:
            normalized = _normalize_for_matching(text)
            if any(pattern.search(normalized) for pattern in _INJECTION_PATTERNS):
                flags.append("prompt_injection")
                blocked = True
                reasons.append("Potential prompt injection detected")

        if self.pii_detection:
            digits = _digits_only(text)
            if _SSN_DASHED_RE.search(text) or _SSN_DIGITS_RE.search(digits):
                flags.append("pii_ssn")
                blocked = True
                reasons.append("Social Security Number detected")
            if any(_luhn_valid(match.group(0)) for match in _CARD_DIGITS_RE.finditer(digits)):
                flags.append("pii_credit_card")
                blocked = True
                reasons.append("Credit card number detected")

        return GuardrailResult(
            blocked=blocked,
            reason="; ".join(reasons) if reasons else None,
            flags=flags,
        )
