"""Domain classifier — detects the knowledge domain from content.

Uses keyword-based classification against the retrieved chunks, user query,
and extracted entity types to determine the most likely domain.

Returns a domain key that maps to a DomainPersona in prompts.personas.
"""

import re
from collections import Counter

from prompts.personas import PERSONAS, DEFAULT_PERSONA, DomainPersona


def _build_keyword_index():
    """Build a mapping from keyword → domain for fast lookup."""
    index = {}
    for domain_key, persona in PERSONAS.items():
        for kw in persona.keywords:
            kw_lower = kw.lower()
            if kw_lower not in index:
                index[kw_lower] = []
            index[kw_lower].append(domain_key)
    return index


_KEYWORD_INDEX = _build_keyword_index()


def classify_domain(
    query: str = "",
    chunk_texts: list[str] | None = None,
    entity_fields: list[str] | None = None,
) -> str:
    """Classify the content domain from available signals."""
    scores: Counter = Counter()

    text_parts = [query]
    if chunk_texts:
        text_parts.extend(chunk_texts)
    combined = "\n".join(text_parts).lower()

    for keyword, domains in _KEYWORD_INDEX.items():
        if keyword in combined:
            for d in domains:
                weight = 2 if len(keyword) >= 4 else 1
                scores[d] += weight

    # Hints from entity fields
    entity_hints = {
        "policy_name": "education",
        "education_stage": "education",
        "subject_area": "education",
        "institution_name": "education",
        "reform_type": "education",
    }
    if entity_fields:
        for field in entity_fields:
            if field in entity_hints:
                scores[entity_hints[field]] += 5

    if scores:
        best_domain, best_score = scores.most_common(1)[0]
        if best_score >= 2:
            return best_domain

    return "general"


def get_persona_for_query(
    query: str,
    chunk_texts: list[str] | None = None,
    entity_fields: list[str] | None = None,
) -> DomainPersona:
    """Classify domain and return the corresponding DomainPersona."""
    from prompts.personas import get_persona

    domain = classify_domain(query, chunk_texts, entity_fields)
    return get_persona(domain)


def get_classification_detail(
    query: str = "",
    chunk_texts: list[str] | None = None,
    entity_fields: list[str] | None = None,
) -> dict:
    """Return detailed classification info including all domain scores."""
    scores: Counter = Counter()

    text_parts = [query]
    if chunk_texts:
        text_parts.extend(chunk_texts)
    combined = "\n".join(text_parts).lower()

    for keyword, domains in _KEYWORD_INDEX.items():
        if keyword in combined:
            for d in domains:
                weight = 2 if len(keyword) >= 4 else 1
                scores[d] += weight

    if entity_fields:
        entity_hints = {
            "policy_name": "education",
            "education_stage": "education",
            "subject_area": "education",
            "institution_name": "education",
            "reform_type": "education",
        }
        for field in entity_fields:
            if field in entity_hints:
                scores[entity_hints[field]] += 5

    best_domain = scores.most_common(1)[0][0] if scores else "general"
    best_score = scores.most_common(1)[0][1] if scores else 0

    return {
        "domain": best_domain,
        "score": best_score,
        "all_scores": dict(scores.most_common()),
        "persona_role": get_persona_for_query(query, chunk_texts, entity_fields).role,
    }
