"""Report generator — LLM-powered analysis report generation.

Automatically detects the content domain (finance, politics, technology, etc.)
and adopts the appropriate expert persona for report generation.
"""

import json
import logging
import time
import openai

from prompts.generation import build_generation_messages
from prompts.personas import DomainPersona
from db.repository import get_extracted_entities_for_chunks
from modules.domain_classifier import get_persona_for_query
from modules.privacy import redact_retrieved

logger = logging.getLogger(__name__)


def build_structured_summary(entities: list[dict]) -> str:
    """Build a text summary from extracted entities for the generation prompt."""
    if not entities:
        return "（无结构化数据）"

    lines = []
    for e in entities:
        parts = []
        if e.get("policy_name"):
            parts.append(f"政策：{e['policy_name']}")
        if e.get("policy_level"):
            parts.append(f"级别：{e['policy_level']}")
        if e.get("education_stage"):
            parts.append(f"学段：{e['education_stage']}")
        if e.get("subject_area"):
            parts.append(f"学科：{e['subject_area']}")
        if e.get("institution_name"):
            parts.append(f"机构：{e['institution_name']}")
        if e.get("region"):
            parts.append(f"地区：{e['region']}")
        if e.get("event_date"):
            parts.append(f"日期：{e['event_date']}")
        if e.get("reform_type"):
            parts.append(f"改革类型：{e['reform_type']}")
        if e.get("impact_summary"):
            parts.append(f"影响：{e['impact_summary']}")
        if e.get("person_name"):
            parts.append(f"关键人物：{e['person_name']}")
        if parts:
            lines.append("  |  ".join(parts))
    return "\n".join(lines)


def generate_report(
    client: openai.OpenAI,
    conn,
    query: str,
    retrieved_chunks: list[dict],
    model: str,
    temperature: float = 0.3,
    timeout: int = 120,
) -> dict:
    """Generate an analysis report using DeepSeek with RAG context.

    Automatically detects the content domain and adopts an appropriate
    expert persona (education researcher, general analyst, etc.).

    Returns dict with keys:
        report              — markdown report text
        generation_time_ms  — milliseconds taken
        prompt_system       — system message content (with persona)
        prompt_user         — user message content (with injected context)
        pii_redacted        — number of PII instances redacted
        domain              — detected domain key
        persona_role        — expert role used (first line of system prompt)
    """
    chunk_ids = [int(c["metadata"]["chunk_id"]) for c in retrieved_chunks]
    entities = get_extracted_entities_for_chunks(conn, chunk_ids)

    # Redact PII from retrieved chunks before they reach the LLM API
    redacted_chunks, redaction_count = redact_retrieved(retrieved_chunks)

    from modules.retriever import format_retrieved_context

    retrieved_contexts = format_retrieved_context(redacted_chunks)

    # Collect chunk texts for domain classification
    chunk_texts = [c["document"] for c in retrieved_chunks]

    # Collect entity field names that have values
    entity_fields = []
    edu_keys = {"policy_name", "education_stage", "subject_area", "institution_name", "reform_type"}
    if entities:
        for e in entities:
            for key, val in e.items():
                if val is not None and val != "" and key not in entity_fields and key in edu_keys:
                    entity_fields.append(key)

    # Classify domain and select persona
    persona = get_persona_for_query(query, chunk_texts, entity_fields)

    logger.info(
        "Domain classified: %s | Persona: %s...",
        persona.domain, persona.role[:60],
    )

    structured_summary = build_structured_summary(entities)

    messages = build_generation_messages(
        user_query=query,
        retrieved_contexts=retrieved_contexts,
        structured_summary=structured_summary,
        persona=persona,
    )

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        timeout=timeout,
    )
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    report = response.choices[0].message.content or ""
    logger.info(
        "Report generated in %dms, length=%d chars, PII redacted=%d, domain=%s",
        elapsed_ms, len(report), redaction_count, persona.domain,
    )

    return {
        "report": report,
        "generation_time_ms": elapsed_ms,
        "prompt_system": messages[0]["content"],
        "prompt_user": messages[1]["content"],
        "pii_redacted": redaction_count,
        "domain": persona.domain,
        "persona_role": persona.role,
    }
