from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

from .schemas import V8_EXCLUDED_CONTENT_TYPES, V8_INCLUDED_CONTENT_TYPES

RELEVANCE_POLICY_VERSION = "v8_event_experience_binary"
RELEVANCE_PROMPT_VERSION = "v8"
RELEVANCE_SCHEMA_VERSION = "v8"
RELEVANCE_ROUTING_VERSION = "v8"
RELEVANCE_VALIDATION_VERSION = "v8_binary"
RELEVANCE_CACHE_NAMESPACE = "relevance-v8"
RELEVANCE_VERIFICATION_CACHE_NAMESPACE = "relevance-verification-v8"
AUTO_INCLUDE_THRESHOLD = 0.80
AUTO_EXCLUDE_THRESHOLD = 0.93
WEAK_EMPTY_EXCLUDE_THRESHOLD = 0.98
FINAL_LABELS = {"include", "exclude"}
INCLUDED = set(V8_INCLUDED_CONTENT_TYPES)
EXCLUDED = set(V8_EXCLUDED_CONTENT_TYPES)

_SUFFIX = {
    "event_route_weather_facilities": "event_route_weather_or_facilities",
    "event_safety_medical": "event_safety_or_medical",
    "event_cost_value": "event_cost_or_value",
}


def derive_reason_code(assessment: dict) -> str:
    connection = assessment["event_connection"]
    content = assessment["primary_content_type"]
    if assessment.get("contradiction_present"):
        return "contradictory_evidence"
    if assessment.get("image_dependent"):
        return "image_dependent"
    if content in INCLUDED and connection in {"explicit", "supported"}:
        return f"{connection}_{_SUFFIX.get(content, content)}"
    if content == "generic_running":
        return "generic_running_without_event_link"
    if content == "other_event":
        if "event_evaluation" in assessment.get("secondary_content_types", []):
            return "meaningful_klscm_comparison"
        return "different_event"
    if content == "promotional_only":
        return "promotion_without_meaningful_experience"
    if content == "spam_or_unrelated":
        return "spam_or_unrelated"
    if content == "no_meaningful_content":
        return "hashtag_without_meaningful_content" if connection == "weak" else "insufficient_text"
    return "insufficient_event_context"


def content_family(content_type: str) -> str:
    return "included" if content_type in INCLUDED else "excluded"


def material_model_disagreement(first: dict, second: dict) -> bool:
    if content_family(first["primary_content_type"]) != content_family(second["primary_content_type"]):
        return True
    if ({first["event_connection"], second["event_connection"]} & {"explicit", "supported"}
            and "none" in {first["event_connection"], second["event_connection"]}):
        return True
    if bool(first["meaningful_content_present"]) != bool(second["meaningful_content_present"]):
        return True
    if bool(first.get("image_dependent")) != bool(second.get("image_dependent")):
        return True
    return False


def _result(assessment: dict, automatic_decision: str, status: str, routing_reason: str) -> dict:
    final = automatic_decision if status != "pending_review" else "unresolved"
    include_topics = final == "include"
    include_sentiment = include_topics and assessment["primary_content_type"] != "event_information"
    assessment_aliases = {f"v8_{key}": value for key, value in assessment.items()}
    return {
        **assessment, **assessment_aliases,
        "v8_reason_code": derive_reason_code(assessment),
        "v8_automatic_decision": automatic_decision,
        "v8_routing_status": status,
        "v8_routing_reason": routing_reason,
        "v8_human_final_inclusion": "",
        "v8_final_inclusion": final,
        "v8_final_decision_source": "automatic" if final != "unresolved" else "pending_human_review",
        "v8_include_in_topics": include_topics,
        "v8_include_in_sentiment": include_sentiment,
        "v8_policy_version": RELEVANCE_POLICY_VERSION,
        "v8_schema_version": RELEVANCE_SCHEMA_VERSION,
        "v8_routing_version": RELEVANCE_ROUTING_VERSION,
    }


def route_assessment(assessment: dict, verification: dict | None = None,
                     include_threshold: float = AUTO_INCLUDE_THRESHOLD,
                     exclude_threshold: float = AUTO_EXCLUDE_THRESHOLD,
                     weak_empty_threshold: float = WEAK_EMPTY_EXCLUDE_THRESHOLD) -> dict:
    chosen = verification or assessment
    confidence = float(chosen["confidence"])
    if verification is not None and material_model_disagreement(assessment, verification):
        return _result(chosen, "review", "pending_review", "material_model_disagreement")
    if chosen.get("contradiction_present"):
        return _result(chosen, "review", "pending_review", "contradictory_evidence")
    if chosen.get("image_dependent"):
        return _result(chosen, "review", "pending_review", "image_dependent")
    if chosen["event_connection"] == "weak":
        deterministic_empty = (
            not chosen["meaningful_content_present"]
            and chosen["primary_content_type"] == "no_meaningful_content"
            and not chosen.get("event_link_evidence", "").strip()
            and not chosen.get("analytical_content_evidence", "").strip()
            and confidence >= weak_empty_threshold
        )
        if deterministic_empty:
            return _result(chosen, "exclude", "automatic_exclude_weak_empty", "weak_no_meaningful_content")
        return _result(chosen, "review", "pending_review", "weak_event_connection")
    family = content_family(chosen["primary_content_type"])
    if (chosen["event_connection"] in {"explicit", "supported"} and family == "included"
            and chosen["meaningful_content_present"] and confidence >= include_threshold
            and chosen["event_link_evidence"].strip() and chosen["analytical_content_evidence"].strip()):
        return _result(chosen, "include", "automatic_include", "included_event_content")
    safe_linked_exclusion = chosen["primary_content_type"] in {
        "promotional_only", "spam_or_unrelated", "no_meaningful_content"
    }
    if (family == "excluded" and confidence >= exclude_threshold
            and (chosen["event_connection"] == "none" or safe_linked_exclusion)):
        return _result(chosen, "exclude", "automatic_exclude", "strong_exclusion_evidence")
    return _result(chosen, "review", "pending_review", "threshold_or_field_conflict")


def apply_human_decision(row: dict, decision: str, notes: str = "") -> dict:
    decision = str(decision).strip().lower()
    if decision not in FINAL_LABELS:
        raise ValueError("Human final inclusion must be include or exclude")
    output = dict(row)
    output["v8_human_final_inclusion"] = decision
    output["v8_review_notes"] = notes
    output["v8_routing_status"] = "human_resolved"
    output["v8_final_inclusion"] = decision
    output["v8_final_decision_source"] = "human_review"
    output["v8_include_in_topics"] = decision == "include"
    output["v8_include_in_sentiment"] = (
        decision == "include" and output["primary_content_type"] != "event_information"
    )
    return output


def _binary_scores(frame: pd.DataFrame, prediction: str) -> dict:
    y_true = frame.human_final_inclusion
    y_pred = frame[prediction]
    precision, recall, included_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=["include"], average="macro", zero_division=0)
    excluded_f1 = precision_recall_fscore_support(
        y_true, y_pred, labels=["exclude"], average="macro", zero_division=0)[2]
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=["include", "exclude"], average="macro", zero_division=0)[2]
    return {"inclusion_precision": float(precision), "inclusion_recall": float(recall),
            "included_class_f1": float(included_f1), "excluded_class_f1": float(excluded_f1),
            "binary_macro_f1": float(macro)}


def validation_report_v8(reviewer: pd.DataFrame, predictions: pd.DataFrame,
                         output_dir: Path, evaluation_split: str = "calibration") -> dict:
    originals = reviewer[reviewer.repeat_of.fillna("").eq("")].copy()
    originals["human_final_inclusion"] = originals.human_final_inclusion.fillna("").str.strip().str.lower()
    invalid = ~originals.human_final_inclusion.isin(FINAL_LABELS)
    if invalid.any():
        raise ValueError(f"Missing or malformed final human labels: {int(invalid.sum())}")
    if originals.document_id.duplicated().any():
        raise ValueError("Repeated validation records entered the unique evaluation set")
    merged = predictions.merge(originals[["document_id", "human_final_inclusion"]], on="document_id", how="inner")
    if len(merged) != len(originals):
        raise ValueError("Reviewer and prediction records do not align")
    if not merged.v8_final_inclusion.isin(FINAL_LABELS | {"unresolved"}).all():
        raise ValueError("Malformed final predictions")
    automatic = merged[merged.v8_routing_status.str.startswith("automatic")].copy()
    routed = merged[merged.v8_final_inclusion.eq("unresolved")].copy()
    automatic_view = {
        "automatic_coverage": float(len(automatic) / max(1, len(merged))),
        "automatic_inclusion_precision": float(automatic.loc[automatic.v8_final_inclusion.eq("include"), "human_final_inclusion"].eq("include").mean()) if automatic.v8_final_inclusion.eq("include").any() else None,
        "automatic_exclusion_precision": float(automatic.loc[automatic.v8_final_inclusion.eq("exclude"), "human_final_inclusion"].eq("exclude").mean()) if automatic.v8_final_inclusion.eq("exclude").any() else None,
        "automatic_error_count": int(automatic.human_final_inclusion.ne(automatic.v8_final_inclusion).sum()),
        "relevant_automatically_excluded": int((automatic.human_final_inclusion.eq("include") & automatic.v8_final_inclusion.eq("exclude")).sum()),
        "weak_empty_auto_exclusions": int(automatic.v8_routing_status.eq("automatic_exclude_weak_empty").sum()),
    }
    routing_view = {
        "review_rate": float(len(routed) / max(1, len(merged))),
        "relevant_routed_to_review": int(routed.human_final_inclusion.eq("include").sum()),
        "irrelevant_routed_to_review": int(routed.human_final_inclusion.eq("exclude").sum()),
        "false_exclusion_prevention": int(routed.human_final_inclusion.eq("include").sum()),
    }
    classifier_frame = merged.copy()
    classifier_frame["classifier_binary_decision"] = classifier_frame.v8_final_inclusion.replace({"unresolved": "exclude"})
    classifier_scores = _binary_scores(classifier_frame, "classifier_binary_decision")
    automatic_view["binary_performance_with_review_not_included"] = classifier_scores
    assisted_frame = merged.copy()
    unresolved = assisted_frame.v8_final_inclusion.eq("unresolved")
    assisted_frame.loc[unresolved, "v8_final_inclusion"] = assisted_frame.loc[unresolved, "human_final_inclusion"]
    assisted = _binary_scores(assisted_frame, "v8_final_inclusion")
    classifier_passes = classifier_scores["inclusion_precision"] >= .85 and classifier_scores["inclusion_recall"] >= .90 and classifier_scores["binary_macro_f1"] >= .80
    repeats = reviewer[reviewer.repeat_of.fillna("").ne("")].merge(
        originals[["review_id", "human_final_inclusion"]].rename(columns={"review_id": "repeat_of", "human_final_inclusion": "original_label"}), on="repeat_of", how="left")
    repeat_labels = repeats.human_final_inclusion.fillna("").str.strip().str.lower()
    valid_repeats = repeat_labels.isin(FINAL_LABELS) & repeats.original_label.isin(FINAL_LABELS)
    kappa = float(cohen_kappa_score(repeats.loc[valid_repeats, "original_label"], repeat_labels[valid_repeats])) if valid_repeats.sum() else None
    annotation_passes = None if kappa is None else bool(kappa >= .80)
    report = {"evaluation_split": evaluation_split, "evaluation_n": int(len(merged)),
              "automatic_classifier_performance": automatic_view,
              "routing_performance": routing_view,
              "simulated_human_assisted_performance": assisted,
              "classifier_passes": bool(classifier_passes),
              "intra_reviewer_kappa": kappa,
              "annotation_quality_passes": annotation_passes,
              "overall_validation_ready": bool(classifier_passes and annotation_passes is True)}
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(confusion_matrix(assisted_frame.human_final_inclusion, assisted_frame.v8_final_inclusion,
                                 labels=["include", "exclude"]),
                 index=["include", "exclude"], columns=["include", "exclude"]).to_csv(output_dir / "relevance_v8_binary_confusion_matrix.csv")
    (output_dir / "relevance_v8_validation_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def threshold_analysis(assessments: pd.DataFrame, human_labels: pd.DataFrame,
                       include_thresholds=(.65, .70, .75, .80, .85, .90),
                       exclude_thresholds=(.85, .90, .93, .95, .97, .99)) -> pd.DataFrame:
    gold = dict(zip(human_labels.document_id, human_labels.human_final_inclusion))
    rows = []
    for inc in include_thresholds:
        for exc in exclude_thresholds:
            if exc <= inc:
                continue
            routed = [route_assessment(row._asdict(), include_threshold=inc, exclude_threshold=exc)
                      for row in assessments.itertuples(index=False)]
            frame = pd.DataFrame(routed)
            frame["human_final_inclusion"] = frame.document_id.map(gold)
            review = frame.v8_final_inclusion.eq("unresolved")
            auto_include = frame.v8_final_inclusion.eq("include")
            auto_exclude = frame.v8_final_inclusion.eq("exclude")
            auto_include_precision = float(frame.loc[auto_include, "human_final_inclusion"].eq("include").mean()) if auto_include.any() else None
            auto_exclude_precision = float(frame.loc[auto_exclude, "human_final_inclusion"].eq("exclude").mean()) if auto_exclude.any() else None
            automatic_errors = int((~review & frame.v8_final_inclusion.ne(frame.human_final_inclusion)).sum())
            frame.loc[review, "v8_final_inclusion"] = frame.loc[review, "human_final_inclusion"]
            scores = _binary_scores(frame, "v8_final_inclusion")
            passes = scores["inclusion_precision"] >= .85 and scores["inclusion_recall"] >= .90 and scores["binary_macro_f1"] >= .80
            rows.append({"automatic_inclusion_threshold": inc, "automatic_exclusion_threshold": exc,
                         "automatic_include_n": int(auto_include.sum()), "automatic_exclude_n": int(auto_exclude.sum()),
                         "review_n": int(review.sum()), "review_rate": float(review.mean()),
                         "reviewer_workload_per_1000": float(review.mean() * 1000),
                         "automatic_coverage": float((~review).mean()),
                         "automatic_inclusion_precision": auto_include_precision,
                         "automatic_exclusion_precision": auto_exclude_precision,
                         "automatic_error_count": automatic_errors, **scores,
                         "passes_quality_targets": bool(passes),
                         "simulated_human_assisted": True})
    return pd.DataFrame(rows)


def recommend_operating_point(table: pd.DataFrame) -> dict | None:
    """Choose lowest defensible workload, then recall, then exclusion precision."""
    eligible = table[table.passes_quality_targets].copy()
    eligible = eligible[eligible.automatic_inclusion_precision.fillna(0).ge(.85)]
    eligible = eligible[eligible.automatic_exclusion_precision.fillna(0).ge(.85)]
    if eligible.empty:
        return None
    chosen = eligible.sort_values(
        ["review_rate", "inclusion_recall", "automatic_exclusion_precision"],
        ascending=[True, False, False],
    ).iloc[0]
    return chosen.to_dict()



ERROR_CATEGORIES = {
    "missed_emotional_experience", "missed_physical_experience",
    "missed_short_caption", "hashtag_context_misinterpreted",
    "generic_running_false_positive", "promotion_false_positive",
    "different_event_false_positive", "image_dependent",
    "insufficient_context", "policy_definition_conflict",
    "schema_mapping_error", "routing_error",
}


def build_error_audit(evaluated: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in evaluated.itertuples(index=False):
        gold = row.human_final_inclusion
        predicted = row.v8_final_inclusion
        routed = predicted == "unresolved"
        if not routed and gold == predicted:
            continue
        content = row.primary_content_type
        if routed:
            category = "image_dependent" if row.image_dependent else "insufficient_context"
        elif gold == "include" and predicted == "exclude":
            category = {
                "event_emotional_experience": "missed_emotional_experience",
                "event_physical_experience": "missed_physical_experience",
            }.get(content, "missed_short_caption" if len(str(getattr(row, "caption", ""))) < 60 else "routing_error")
        else:
            category = {
                "generic_running": "generic_running_false_positive",
                "promotional_only": "promotion_false_positive",
                "other_event": "different_event_false_positive",
            }.get(content, "routing_error")
        rows.append({
            "record_id": getattr(row, "record_id", getattr(row, "document_id", "")),
            "caption": getattr(row, "caption", getattr(row, "original_text", "")),
            "human_final_inclusion": gold,
            "model_final_inclusion": predicted,
            "event_connection": row.event_connection,
            "primary_content_type": content,
            "reason_code": row.v8_reason_code,
            "error_category": category,
            "confidence": row.confidence,
            "review_status": row.v8_routing_status,
            "notes": "",
        })
    return pd.DataFrame(rows, columns=[
        "record_id", "caption", "human_final_inclusion", "model_final_inclusion",
        "event_connection", "primary_content_type", "reason_code", "error_category",
        "confidence", "review_status", "notes",
    ])


def subgroup_performance(evaluated: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dimensions = ["primary_language", "event_year", "length_band", "language_status",
                  "event_connection", "primary_content_type"]
    for dimension in dimensions:
        if dimension not in evaluated:
            continue
        for value, group in evaluated.groupby(dimension, dropna=False):
            assisted = group.copy()
            unresolved = assisted.v8_final_inclusion.eq("unresolved")
            assisted.loc[unresolved, "v8_final_inclusion"] = assisted.loc[unresolved, "human_final_inclusion"]
            scores = _binary_scores(assisted, "v8_final_inclusion")
            rows.append({"dimension": dimension, "group": str(value), "n": len(group),
                         "review_rate": float(unresolved.mean()), **scores})
    return pd.DataFrame(rows)




