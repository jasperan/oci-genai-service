"""OCI AI Guardrails — content moderation, PII detection, prompt injection."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""

    blocked: bool
    reason: Optional[str] = None
    flags: list[str] = field(default_factory=list)


class Guardrail:
    """Content safety guardrail using OCI AI Guardrails.

    Usage:
        guard = Guardrail(content_moderation=True, pii_detection=True)
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
            injection_indicators = [
                "ignore previous instructions",
                "ignore all previous",
                "disregard your instructions",
                "you are now",
                "new instructions:",
            ]
            lower_text = text.lower()
            if any(indicator in lower_text for indicator in injection_indicators):
                flags.append("prompt_injection")
                blocked = True
                reasons.append("Potential prompt injection detected")

        if self.pii_detection:
            if re.search(r"\b\d{3}-\d{2}-\d{4}\b", text):  # SSN
                flags.append("pii_ssn")
                reasons.append("Social Security Number detected")
            if re.search(r"\b\d{16}\b", text):  # Credit card
                flags.append("pii_credit_card")
                reasons.append("Credit card number detected")

        return GuardrailResult(
            blocked=blocked,
            reason="; ".join(reasons) if reasons else None,
            flags=flags,
        )
