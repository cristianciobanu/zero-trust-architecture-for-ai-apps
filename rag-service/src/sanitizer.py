"""
Prompt injection detection and input sanitization.

Detects common prompt injection patterns and cleans user input
before it reaches the retrieval/LLM pipeline.
"""

import re
from dataclasses import dataclass

# Patterns that indicate prompt injection attempts
INJECTION_PATTERNS: list[tuple[str, str]] = [
    # Direct instruction override attempts
    (r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?|context)",
     "instruction_override"),
    (r"disregard\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)",
     "instruction_override"),
    (r"forget\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)",
     "instruction_override"),

    # System prompt extraction attempts
    (r"(show|reveal|display|print|output|repeat|tell\s+me)\s+(the\s+)?(system\s+prompt|initial\s+prompt|instructions?|your\s+rules)",
     "prompt_extraction"),
    (r"what\s+(are|is)\s+your\s+(system\s+prompt|instructions?|rules|initial\s+prompt)",
     "prompt_extraction"),

    # Role manipulation
    (r"you\s+are\s+now\s+(a|an)\s+",
     "role_manipulation"),
    (r"act\s+as\s+(a|an)\s+",
     "role_manipulation"),
    (r"pretend\s+(to\s+be|you\s+are)\s+",
     "role_manipulation"),
    (r"switch\s+to\s+.*\s+mode",
     "role_manipulation"),

    # Delimiter injection (trying to break out of context)
    (r"```\s*(system|assistant|user)\s*\n",
     "delimiter_injection"),
    (r"\[INST\]|\[/INST\]|<<SYS>>|<\|im_start\|>|<\|im_end\|>",
     "delimiter_injection"),

    # Data exfiltration attempts
    (r"(list|show|give|output)\s+(all|every)\s+(account|credit\s+card|ssn|social\s+security|password|secret|key)",
     "data_exfiltration"),

    # Encoding evasion (base64, hex instructions)
    (r"base64\s*(decode|encode)|decode\s+the\s+following",
     "encoding_evasion"),
]


@dataclass
class SanitizationResult:
    cleaned_text: str
    is_blocked: bool = False
    was_modified: bool = False
    matched_pattern: str | None = None
    original_text: str = ""


class PromptSanitizer:
    def __init__(self) -> None:
        self._compiled_patterns: list[tuple[re.Pattern, str]] = [
            (re.compile(pattern, re.IGNORECASE), name)
            for pattern, name in INJECTION_PATTERNS
        ]

    def check(self, text: str) -> SanitizationResult:
        if not text or not text.strip():
            return SanitizationResult(cleaned_text="", is_blocked=True, matched_pattern="empty_input")

        # Check for injection patterns
        for pattern, name in self._compiled_patterns:
            if pattern.search(text):
                return SanitizationResult(
                    cleaned_text=text,
                    is_blocked=True,
                    was_modified=False,
                    matched_pattern=name,
                    original_text=text,
                )

        # Clean the input (strip control characters, normalize whitespace)
        cleaned = self._clean(text)
        was_modified = cleaned != text

        return SanitizationResult(
            cleaned_text=cleaned,
            is_blocked=False,
            was_modified=was_modified,
            original_text=text,
        )

    def _clean(self, text: str) -> str:
        # Remove null bytes and control characters (keep newlines/tabs)
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        # Collapse excessive whitespace
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()
