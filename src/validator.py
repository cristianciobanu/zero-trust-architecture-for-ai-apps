"""
Output validation and PII redaction.

Scans LLM responses for sensitive data patterns (PII, financial account numbers,
secrets) and redacts them before returning to the user.
"""

import re
from dataclasses import dataclass, field

# Patterns to detect and redact in LLM output
PII_PATTERNS: list[tuple[str, str, str]] = [
    # Credit card numbers (Visa, MC, Amex, etc.)
    (r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
     "[REDACTED_CREDIT_CARD]", "credit_card"),

    # US Social Security Numbers
    (r"\b\d{3}-\d{2}-\d{4}\b",
     "[REDACTED_SSN]", "ssn"),

    # IBAN (international bank account)
    (r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b",
     "[REDACTED_IBAN]", "iban"),

    # Email addresses
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
     "[REDACTED_EMAIL]", "email"),

    # Phone numbers (US and international formats)
    (r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
     "[REDACTED_PHONE]", "phone"),

    # AWS access key IDs
    (r"\bAKIA[0-9A-Z]{16}\b",
     "[REDACTED_AWS_KEY]", "aws_key"),

    # Generic API keys / tokens (long hex or base64 strings)
    (r"\b[A-Za-z0-9]{40,}\b",
     None, "potential_secret"),  # None = flag but don't auto-redact (too broad)
]


@dataclass
class ValidationResult:
    safe_text: str
    was_redacted: bool = False
    has_violations: bool = False
    violations: list[str] = field(default_factory=list)
    redaction_count: int = 0


class OutputValidator:
    def __init__(self) -> None:
        self._compiled_patterns: list[tuple[re.Pattern, str | None, str]] = [
            (re.compile(pattern), replacement, name)
            for pattern, replacement, name in PII_PATTERNS
        ]

    def validate(self, text: str) -> ValidationResult:
        if not text:
            return ValidationResult(safe_text="")

        safe_text = text
        violations: list[str] = []
        redaction_count = 0

        for pattern, replacement, name in self._compiled_patterns:
            matches = pattern.findall(safe_text)
            if matches:
                violations.append(f"{name}:{len(matches)}")
                if replacement is not None:
                    safe_text = pattern.sub(replacement, safe_text)
                    redaction_count += len(matches)

        return ValidationResult(
            safe_text=safe_text,
            was_redacted=redaction_count > 0,
            has_violations=len(violations) > 0,
            violations=violations,
            redaction_count=redaction_count,
        )
