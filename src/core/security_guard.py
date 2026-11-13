"""
ShieldGuard Security Layer: Policyholder PII Redaction & Indirect Prompt-Injection (XPIA) Guard.
Sanitizes untrusted commercial insurance PDFs/binders before RAG indexing or LLM inference.

Governing Requirements: REQ-SEC-01
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from src.core.schemas import SanitizedDocumentResult


PII_PATTERNS: List[Tuple[str, re.Pattern[str], str]] = [
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    ("FEIN", re.compile(r"\b\d{2}-\d{7}\b"), "[REDACTED_FEIN]"),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    ("PHONE", re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"), "[REDACTED_PHONE]"),
    ("BANK_ROUTING", re.compile(r"\b(?:routing|aba)\s*#?:?\s*\d{9}\b", re.IGNORECASE), "[REDACTED_ROUTING]"),
]

XPIA_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("IGNORE_PREVIOUS_INSTRUCTIONS", re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.IGNORECASE)),
    ("SYSTEM_PROMPT_EXFILTRATION", re.compile(r"(?:reveal|output|print)\s+(?:your\s+)?system\s+prompt", re.IGNORECASE)),
    ("FORCED_TOOL_OVERRIDE", re.compile(r"override\s+(?:policy|exclusion|sublimit)\s+and\s+approve", re.IGNORECASE)),
    ("HIDDEN_ROLE_SWITCH", re.compile(r"(?:^|\n)\s*(?:system|developer)\s*:\s*you\s+are\s+now", re.IGNORECASE)),
]


def sanitize_insurance_document(raw_text: str) -> SanitizedDocumentResult:
    """
    Redacts sensitive policyholder PII (SSN, FEIN, Email, Phone, Bank Routing)
    and neutralizes indirect prompt injections (XPIA) embedded in untrusted policy PDFs.
    """
    sanitized = raw_text
    redaction_counts: Dict[str, int] = {}

    for label, pattern, replacement in PII_PATTERNS:
        sanitized, count = pattern.subn(replacement, sanitized)
        if count > 0:
            redaction_counts[label] = count

    detected_injections: List[str] = []
    for inj_name, inj_pattern in XPIA_PATTERNS:
        if inj_pattern.search(sanitized):
            detected_injections.append(inj_name)
            sanitized = inj_pattern.sub("[BLOCKED_XPIA_DIRECTIVE]", sanitized)

    total_pii = sum(redaction_counts.values())
    has_injection = len(detected_injections) > 0

    return SanitizedDocumentResult(
        sanitized_text=sanitized,
        pii_entities_redacted=redaction_counts,
        total_pii_redacted=total_pii,
        prompt_injection_detected=has_injection,
        detected_injection_patterns=detected_injections,
        safe_for_llm_ingestion=not has_injection,
    )
