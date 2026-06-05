"""LLM-based information extraction — unstructured text → structured JSON → SQLite."""

import json
import logging
import openai

import config
from prompts.extraction import build_extraction_messages
from utils.helpers import safe_json_parse

logger = logging.getLogger(__name__)


def extract_from_chunk(
    client: openai.OpenAI,
    chunk_text: str,
    model: str,
    temperature: float = 0.1,
    timeout: int = 60,
) -> dict | None:
    """Send a chunk to DeepSeek for structured field extraction.

    Returns the parsed entity dict, or None if extraction failed.
    """
    messages = build_extraction_messages(chunk_text)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=timeout,
            **config.deepseek_chat_options(),
        )
        raw = response.choices[0].message.content
        logger.debug("Extraction raw response: %s", raw[:200])

        entity = safe_json_parse(raw)
        if entity is None:
            logger.warning("Failed to parse JSON from extraction response: %s", raw[:300])
            return None

        # Ensure confidence_score is numeric
        for field in ("confidence_score",):
            if field in entity and entity[field] is not None:
                try:
                    entity[field] = float(entity[field])
                except (TypeError, ValueError):
                    entity[field] = None

        entity["_extraction_raw"] = raw
        return entity

    except openai.APIError as e:
        logger.error("API error during extraction: %s", e)
        return None
    except Exception as e:
        logger.error("Unexpected error during extraction: %s", e)
        return None


def run_extraction_pipeline(
    client: openai.OpenAI,
    chunks: list,
    conn,
    repo,
    model: str,
    temperature: float = 0.1,
) -> list[dict]:
    """Run extraction on all chunks, store results in SQLite.

    Returns list of parsed entity dicts.
    """
    results = []
    success_count = 0

    for chunk in chunks:
        entity = extract_from_chunk(
            client=client,
            chunk_text=chunk.content,
            model=model,
            temperature=temperature,
        )

        if entity is None:
            entity = {
                "policy_name": None, "policy_level": None,
                "education_stage": None, "subject_area": None,
                "institution_name": None, "person_name": None,
                "event_date": None, "reform_type": None,
                "impact_summary": None, "region": None,
                "keywords": None,
                "confidence_score": 0.0,
            }
            raw = json.dumps({"error": "extraction_failed"}, ensure_ascii=False)
        else:
            raw = entity.pop("_extraction_raw", "")
            success_count += 1

        repo.insert_extracted_entity(
            conn=conn,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            entity=entity,
            extraction_raw=raw,
            model=model,
        )
        results.append(entity)

    logger.info(
        "Extraction complete: %d/%d chunks succeeded (%.1f%%)",
        success_count,
        len(chunks),
        success_count / len(chunks) * 100 if chunks else 0,
    )
    return results
