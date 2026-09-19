from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import SETTINGS, Settings
from .cost_conservative import (
    INITIAL_STAGE, VERIFICATION_STAGE, _load_cache, record_context,
    verification_triggers,
)
from .relevance_v8 import material_model_disagreement
from .storage import read_table


DIAGNOSTIC_VERSION = "relevance_v8_architecture_disagreement_audit_v1"
AUTO_INCLUDE_CATEGORIES = [
    "incidental_klsm_connection", "generic_running", "another_event_subject",
    "promotion_or_commercial", "weak_or_image_dependent", "policy_boundary",
    "researcher_label_error", "other",
]
DETERMINISTIC_CATEGORIES = [
    "hashtags_encode_participation", "hashtags_encode_result_or_distance",
    "hashtags_encode_experience_or_feeling", "hashtags_encode_event_information",
    "generic_hashtag_dump", "another_event", "promotion",
    "insufficient_evidence", "researcher_label_error", "other",
]
HASHTAG_PATTERN_CATEGORIES = [
    "klscm_participation", "race_category_or_distance",
    "result_pb_finisher_status", "preparation", "feeling_or_experience",
    "event_information", "generic_running_only", "another_event",
    "promotional_content", "insufficient_evidence",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value):
    if hasattr(value, "tolist"): return value.tolist()
    return value


def _confidence_band(value: object) -> str:
    number = float(value)
    if number < .90: return "below_0.90"
    if number < .95: return "0.90_to_below_0.95"
    if number < .99: return "0.95_to_below_0.99"
    return "0.99_to_1.00"


def _length_band(text: object) -> str:
    words = len(str(text).split())
    return "very_short_lt8" if words < 8 else "short_8_24" if words < 25 else "medium_25_74" if words < 75 else "long_75_plus"


def classify_hashtag_pattern(hashtags: object) -> str:
    """Descriptive, deterministic support category; never changes a label."""
    if isinstance(hashtags, str):
        try: values = json.loads(hashtags)
        except json.JSONDecodeError: values = re.findall(r"#?([\w]+)", hashtags)
    else: values = _jsonable(hashtags) or []
    text = " ".join(str(value).casefold().lstrip("#") for value in values)
    patterns = [
        ("result_pb_finisher_status", r"\b(pb|personalbest|finisher|finished|finishline|sub\d|podium|medal|completed)\b"),
        ("race_category_or_distance", r"\b(5k|10k|21k|42k|42km|halfmarathon|fullmarathon|hm|fm|kidsdash)\b"),
        ("preparation", r"\b(training|train|longrun|roadtoklscm|preparation|raceready)\b"),
        ("feeling_or_experience", r"\b(proud|happy|happiness|love|fun|excited|tired|pain|strong|achievement)\b"),
        ("event_information", r"\b(racekit|expo|flagoff|route|pacers?|registration|roadclosure|racedayinfo)\b"),
        ("promotional_content", r"\b(sale|discount|promo|sponsor|brand|buy|shop|giveaway)\b"),
        ("another_event", r"\b(penang|putrajaya|boston|tokyo|berlin|singapore|standardcharteredhk|scsm)\b"),
        ("klscm_participation", r"\b(klscm\d*|klmarathon|raceday|runner|marathoner|participant)\b"),
        ("generic_running_only", r"\b(running|run|runner|jogging|fitness|workout)\b"),
    ]
    for category, pattern in patterns:
        if re.search(pattern, text): return category
    return "insufficient_evidence"


def _assessment_fields(prefix: str, assessment: dict | None) -> dict:
    names = ["event_connection", "primary_content_type", "secondary_content_types",
        "meaningful_content_present", "event_link_evidence", "analytical_content_evidence",
        "confidence", "contradiction_present", "contradiction_note", "image_dependent",
        "requires_review_recommendation", "review_reason", "language_observed",
        "code_switching_note", "short_explanation"]
    return {f"{prefix}_{name}": (_jsonable(assessment.get(name)) if assessment else "") for name in names}


def create_disagreement_package(settings: Settings = SETTINGS) -> dict:
    audit_root = settings.output_dir / "relevance_v8_single_researcher_architecture_audit_v2"
    output = settings.output_dir / "relevance_v8_architecture_disagreement_audit_v1"
    if output.exists():
        raise FileExistsError(f"Disagreement package exists; refusing overwrite: {output}")
    blind_path = audit_root / "relevance_v8_architecture_audit_blind_v2.csv"
    key_path = audit_root / "relevance_v8_architecture_audit_key_v2.csv"
    production_path = settings.output_dir / "relevance_v8_single_researcher_cost_conservative" / "production_v1" / "relevance_v8_cost_conservative_decisions_v1.parquet"
    documents_path = settings.output_dir / "documents.parquet"
    blind = pd.read_csv(blind_path, dtype=str, keep_default_na=False)
    key = pd.read_csv(key_path, dtype=str, keep_default_na=False)
    if len(blind) != 125 or blind.researcher_label.str.lower().isin({"include", "exclude"}).sum() != 125:
        raise ValueError("Completed 125-row architecture audit is required")
    if set(map(tuple, blind[["audit_id", "document_id"]].values)) != set(map(tuple, key[["audit_id", "document_id"]].values)):
        raise ValueError("Blind audit and hidden v2 key do not align")
    if key.audit_arm.value_counts().to_dict() != {"automatic_include": 75, "deterministic_auto_exclude": 50}:
        raise ValueError("Hidden v2 key must contain the exact 75/50 architecture arms")
    merged = blind.merge(key, on=["audit_id", "document_id"], suffixes=("", "_key"), validate="one_to_one")
    expected = merged.audit_arm.map({"automatic_include": "include", "deterministic_auto_exclude": "exclude"})
    disagreements = merged[merged.researcher_label.str.lower().ne(expected)].copy()
    if len(disagreements) != 26:
        raise ValueError(f"Expected exactly 26 disagreements, found {len(disagreements)}")
    production = read_table(production_path).set_index("document_id")
    documents = read_table(documents_path).set_index("document_id")
    rows = []
    for audit in disagreements.itertuples(index=False):
        production_row = production.loc[audit.document_id]
        document = documents.loc[audit.document_id]
        luna = terra = None
        if audit.audit_arm == "automatic_include":
            context = record_context(type("Record", (), {**document.to_dict(), "document_id": audit.document_id})())
            cached = _load_cache(settings, INITIAL_STAGE, settings.chat_model, context)
            if cached is None: raise FileNotFoundError(f"Missing Luna cache for {audit.document_id}")
            luna = cached[0]
            triggers = verification_triggers(luna)
            if bool(production_row.get("terra_verification_eligible", False)):
                verification_text = json.dumps({"caption": json.loads(context), "first_assessment": luna}, ensure_ascii=False)
                cached_terra = _load_cache(settings, VERIFICATION_STAGE, settings.stronger_model, verification_text)
                if cached_terra is None: raise FileNotFoundError(f"Missing Terra cache for {audit.document_id}")
                terra = cached_terra[0]
        else:
            triggers = []
        hashtags = _jsonable(document.get("hashtags", []))
        record = {
            "audit_id": audit.audit_id, "document_id": audit.document_id,
            "disagreement_arm": "automatic_include_researcher_exclude" if audit.audit_arm == "automatic_include" else "deterministic_exclude_researcher_include",
            "original_caption": document.original_text,
            "normalized_text": document.get(
                "normalized_text",
                document.get("cleaned_text", document.get("linguistic_text", "")),
            ),
            "semantic_text": document.get("semantic_text", ""),
            "hashtags": json.dumps(hashtags, ensure_ascii=False),
            "recorded_language": document.get("primary_language", audit.primary_language),
            "event_year": document.get("event_year", audit.event_year), "source": document.get("source", audit.source),
            "production_route": production_row.get("v8_routing_status", ""),
            "production_routing_reason": production_row.get("v8_routing_reason", ""),
            "review_route_source": production_row.get("review_route_source", ""),
            "event_connection": production_row.get("event_connection", ""),
            "primary_content_type": production_row.get("primary_content_type", ""),
            "meaningful_content_present": production_row.get("meaningful_content_present", ""),
            "confidence": production_row.get("confidence", ""),
            "verification_triggers": "|".join(triggers),
            "final_automatic_decision": production_row.get("v8_automatic_decision", ""),
            "researcher_blind_decision": audit.researcher_label,
            "researcher_evidence": audit.evidence_span, "researcher_rationale": audit.rationale,
            "researcher_comments": audit.comments,
            "assessment_path": "luna_plus_terra" if terra else "luna_only" if luna else "deterministic_no_model",
            "confidence_band": _confidence_band(production_row.get("confidence", 1.0)) if luna else "not_applicable",
            "caption_length_words": len(str(document.original_text).split()), "caption_length_band": _length_band(document.original_text),
            "connection_support_mode": "hashtag_supported" if production_row.get("event_connection", "") == "supported" else "text_explicit" if production_row.get("event_connection", "") == "explicit" else str(production_row.get("event_connection", "")),
            "promotion_indicator": bool((production_row.get("primary_content_type", "") == "promotional_only") or re.search(r"\b(buy|sale|discount|promo|shop|register now)\b", str(document.original_text), re.I)),
            "image_dependent": production_row.get("image_dependent", ""),
            "luna_terra_material_disagreement": material_model_disagreement(luna, terra) if luna and terra else False,
            "deterministic_hashtag_pattern": classify_hashtag_pattern(hashtags) if audit.audit_arm == "deterministic_auto_exclude" else "",
            **_assessment_fields("luna", luna), **_assessment_fields("terra", terra),
            "allowed_diagnostic_categories": "|".join(AUTO_INCLUDE_CATEGORIES if audit.audit_arm == "automatic_include" else DETERMINISTIC_CATEGORIES),
            "diagnostic_category": "", "diagnostic_note": "",
        }
        rows.append(record)
    report = pd.DataFrame(rows).sort_values(["disagreement_arm", "audit_id"]).reset_index(drop=True)
    auto = report[report.disagreement_arm.eq("automatic_include_researcher_exclude")]
    deterministic = report[report.disagreement_arm.eq("deterministic_exclude_researcher_include")]
    summaries = {
        "version": DIAGNOSTIC_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(), "api_calls": 0,
        "records": len(report), "counts_by_arm": report.disagreement_arm.value_counts().to_dict(),
        "automatic_include_patterns": {
            "assessment_path": auto.assessment_path.value_counts().to_dict(),
            "event_connection": auto.event_connection.value_counts().to_dict(),
            "primary_content_type": auto.primary_content_type.value_counts().to_dict(),
            "confidence_band": auto.confidence_band.value_counts().to_dict(),
            "connection_support_mode": auto.connection_support_mode.value_counts().to_dict(),
            "caption_length_band": auto.caption_length_band.value_counts().to_dict(),
            "promotion_indicator": auto.promotion_indicator.value_counts().to_dict(),
            "image_dependent": auto.image_dependent.astype(str).value_counts().to_dict(),
            "luna_terra_material_disagreement": auto.luna_terra_material_disagreement.value_counts().to_dict(),
        },
        "deterministic_exclusion_patterns": {
            category: int(deterministic.deterministic_hashtag_pattern.eq(category).sum())
            for category in HASHTAG_PATTERN_CATEGORIES
        },
        "source_hashes": {str(path): _sha256(path) for path in [blind_path, key_path, production_path, documents_path]},
    }
    output.mkdir(parents=True)
    report_path = output / "relevance_v8_architecture_disagreements_v1.csv"
    report.to_csv(report_path, index=False, encoding="utf-8-sig")
    summary_path = output / "relevance_v8_architecture_disagreement_summary_v1.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"version": DIAGNOSTIC_VERSION, "created_at_utc": summaries["created_at_utc"], "api_calls": 0,
        "records": 26, "annotation_fields_preserved": True,
        "output_hashes": {report_path.name: _sha256(report_path), summary_path.name: _sha256(summary_path)}}
    (output / "manifest_v1.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summaries
