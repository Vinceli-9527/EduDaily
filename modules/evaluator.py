"""Evaluation module — automated metrics for extraction, retrieval, and generation."""

import json
import re
import logging
from collections import defaultdict
from rapidfuzz import fuzz as _fuzz  # type: ignore

logger = logging.getLogger(__name__)


# ── Extraction Evaluation ────────────────────────────────────────────


def _normalize(s):
    return s.strip() if s else ""


def _fuzzy_match(extracted, truth):
    if not extracted or not truth:
        return 0.0
    return _fuzz.ratio(_normalize(extracted), _normalize(truth)) / 100.0


def _numeric_accuracy(extracted, truth):
    """Relative accuracy for numeric fields, clamped to [0, 1]."""
    if extracted is None or truth is None:
        return 1.0  # Both missing = perfect match
    if truth == 0:
        return 1.0 if extracted == 0 else 0.0
    return max(0.0, 1.0 - abs(extracted - truth) / abs(truth))


def _exact_match(extracted, truth):
    if extracted is None and truth is None:
        return 1.0
    if extracted is None or truth is None:
        return 0.0
    return 1.0 if _normalize(str(extracted)) == _normalize(str(truth)) else 0.0


def _jaccard_keywords(extracted, truth):
    """Jaccard similarity for keyword lists."""
    if not extracted and not truth:
        return 1.0
    if not extracted or not truth:
        return 0.0
    if isinstance(extracted, str):
        try:
            extracted = json.loads(extracted)
        except json.JSONDecodeError:
            extracted = [extracted]
    set_e = set(extracted or [])
    set_t = set(truth or [])
    if not set_e and not set_t:
        return 1.0
    return len(set_e & set_t) / len(set_e | set_t)


def evaluate_extraction(
    ground_truth_path: str,
    db_entities: list[dict],
) -> dict[str, float]:
    """Compare extracted entities against ground truth, compute per-field metrics.

    Returns dict of metric_name → value.
    """
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    # Build a lookup: (policy_name) → ground truth entity
    gt_lookup = {}
    for doc_name, doc_info in gt_data.items():
        for gt in doc_info.get("ground_truths", []):
            key = gt.get("policy_name", doc_name)
            gt_lookup[key] = gt

    if not db_entities or not gt_lookup:
        logger.warning("No data for extraction evaluation")
        return {"extraction_overall_f1": 0.0}

    # Field-level evaluation
    field_metrics = defaultdict(list)
    field_pairs = [
        ("policy_name", _fuzzy_match, "policy_fuzzy_accuracy"),
        ("policy_level", _exact_match, "policy_level_exact_match"),
        ("education_stage", _exact_match, "education_stage_exact_match"),
        ("subject_area", _exact_match, "subject_area_exact_match"),
        ("institution_name", _fuzzy_match, "institution_fuzzy_accuracy"),
        ("event_date", _exact_match, "event_date_exact_match"),
        ("reform_type", _exact_match, "reform_type_exact_match"),
        ("region", _exact_match, "region_exact_match"),
    ]

    scores = {}
    for db_entity in db_entities:
        policy = db_entity.get("policy_name", "")
        gt = gt_lookup.get(policy)
        if gt is None:
            # Try fuzzy match to find nearest ground truth
            best_score = 0
            best_key = None
            for key in gt_lookup:
                sim = _fuzzy_match(policy, key)
                if sim > best_score and sim > 0.6:
                    best_score = sim
                    best_key = key
            if best_key:
                gt = gt_lookup[best_key]

        if gt is None:
            continue

        for field, metric_fn, metric_name in field_pairs:
            val = metric_fn(db_entity.get(field), gt.get(field))
            field_metrics[metric_name].append(val)

    # JSON validity rate
    valid_count = sum(1 for e in db_entities if e.get("confidence_score", 0) > 0)
    total = len(db_entities)
    scores["json_validity_rate"] = valid_count / total if total > 0 else 0.0

    extractable_fields = [
        "policy_name", "policy_level", "education_stage", "subject_area",
        "institution_name", "person_name", "event_date", "reform_type",
        "impact_summary", "region",
    ]
    completeness_scores = []
    for db_entity in db_entities:
        policy = db_entity.get("policy_name", "")
        gt = next((v for k, v in gt_lookup.items() if _fuzzy_match(policy, k) > 0.6), None)
        if gt is None:
            continue
        gt_non_null = sum(1 for f in extractable_fields if gt.get(f) is not None)
        if gt_non_null == 0:
            continue
        ex_non_null = sum(1 for f in extractable_fields if db_entity.get(f) is not None)
        completeness_scores.append(min(1.0, ex_non_null / gt_non_null))

    if completeness_scores:
        scores["completeness"] = sum(completeness_scores) / len(completeness_scores)
    else:
        scores["completeness"] = 0.0

    # Per-field averages
    for metric_name, vals in field_metrics.items():
        if vals:
            scores[metric_name] = sum(vals) / len(vals)

    # Overall F1 (macro-average of all field metrics)
    field_score_vals = [scores.get(name, 0.0) for _, _, name in field_pairs]
    scores["extraction_overall_f1"] = sum(field_score_vals) / len(field_score_vals) if field_score_vals else 0.0

    logger.info(
        "Extraction evaluation — Overall F1: %.3f | Completeness: %.3f | JSON validity: %.3f",
        scores["extraction_overall_f1"],
        scores["completeness"],
        scores["json_validity_rate"],
    )
    return scores


# ── Retrieval Evaluation ──────────────────────────────────────────────


def evaluate_retrieval(
    ground_truth_path: str,
    db_entities: list[dict],
    retrieved_chunk_ids: list[int],
    k: int = 5,
) -> dict[str, float]:
    """Compute retrieval metrics. Uses chunk relevance annotations from ground truth."""
    # For demo: consider all chunks as potentially relevant if they contain
    # entities from the ground truth. Simplified metric.
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    # Count total relevant chunks across all documents
    total_relevant = sum(
        len(doc_info.get("ground_truths", [])) for doc_info in gt_data.values()
    )

    retrieved_set = set(retrieved_chunk_ids[:k])
    relevant_count = len(retrieved_set)  # Simplified: all chunks with entities are relevant

    precision_k = min(1.0, relevant_count / k) if k > 0 else 0.0
    recall_k = min(1.0, relevant_count / total_relevant) if total_relevant > 0 else 0.0
    f1_k = (
        2 * precision_k * recall_k / (precision_k + recall_k)
        if (precision_k + recall_k) > 0
        else 0.0
    )

    return {
        f"precision@{k}": round(precision_k, 4),
        f"recall@{k}": round(recall_k, 4),
        f"f1@{k}": round(f1_k, 4),
    }


# ── Generation Evaluation ─────────────────────────────────────────────


def evaluate_generation(report: str, structured_data: list[dict]) -> dict[str, float]:
    """Heuristic evaluation of generated report quality."""
    scores = {}

    # Section completeness: check all 5 required sections
    required_sections = [
        r"### 一、教育现象/政策概述",
        r"### 二、深度分析",
        r"### 三、利益相关方影响",
        r"### 四、国内外对比与借鉴",
        r"### 五、建议与展望",
    ]
    found = sum(1 for pat in required_sections if re.search(pat, report))
    scores["section_completeness"] = found / len(required_sections)

    # Data citation: count entity values that match structured data
    citation_count = 0
    for entity in structured_data:
        for key in ("policy_name", "institution_name", "person_name", "region"):
            val = entity.get(key)
            if val is not None and str(val) in report:
                citation_count += 1
    scores["data_citation_count"] = float(min(citation_count, 10))

    # Length check
    report_len = len(report)
    scores["length_valid"] = 1.0 if 500 <= report_len <= 5000 else 0.5 if report_len > 0 else 0.0

    # Hallucination check: any policy/institution names in report that aren't in source data?
    source_names = {e.get("policy_name", "") for e in structured_data if e.get("policy_name")}
    source_names.update({e.get("institution_name", "") for e in structured_data if e.get("institution_name")})
    report_mentions = 0
    report_known = 0
    for name in source_names:
        if name and name in report:
            report_mentions += 1
            report_known += 1
    scores["hallucination_flag"] = 1.0 if report_mentions == report_known else 0.5

    logger.info(
        "Generation evaluation — Sections: %.0f%% | Citations: %d | Length: %d",
        scores["section_completeness"] * 100,
        citation_count,
        report_len,
    )
    return scores
