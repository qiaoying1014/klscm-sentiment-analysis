from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .relevance_v8 import route_assessment


OPERATING_POINT_VERSION = "v8_op1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _boolean(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


def simulate_saved_route(row: pd.Series, include_threshold: float,
                         exclude_threshold: float) -> dict:
    """Re-route saved fields while preserving forced-review invariants."""
    fields = ["document_id", "event_connection", "primary_content_type",
        "secondary_content_types", "meaningful_content_present", "event_link_evidence",
        "analytical_content_evidence", "confidence", "contradiction_present",
        "contradiction_note", "image_dependent", "requires_review_recommendation",
        "review_reason", "language_observed", "code_switching_note", "short_explanation"]
    assessment = {field: row.get(field, "") for field in fields}
    for field in ["meaningful_content_present", "contradiction_present",
                  "image_dependent", "requires_review_recommendation"]:
        assessment[field] = _boolean(assessment[field])
    if (row.get("v8_routing_reason") == "material_model_disagreement" or
            row.get("v8_model_agreement") == "disagree"):
        return {"v8_routing_status": "pending_review",
                "v8_final_inclusion": "unresolved",
                "v8_automatic_decision": "review",
                "v8_routing_reason": "material_model_disagreement"}
    return route_assessment(assessment, include_threshold=include_threshold,
                            exclude_threshold=exclude_threshold)


def route_frame(frame: pd.DataFrame, include_threshold: float,
                exclude_threshold: float) -> pd.DataFrame:
    routed = pd.DataFrame([simulate_saved_route(row, include_threshold,
        exclude_threshold) for _, row in frame.iterrows()])
    routed.insert(0, "record_id", frame.document_id.to_numpy())
    return routed


def verify_frozen_invariants(original: pd.DataFrame, after: pd.DataFrame) -> None:
    protected = ["event_connection", "primary_content_type", "confidence",
        "v8_model_agreement", "image_dependent", "v8_reason_code",
        "final_v8_gold_label"]
    if not original[protected].reset_index(drop=True).equals(
            after[protected].reset_index(drop=True)):
        raise ValueError("Threshold simulation mutated a protected field")


def route_diff(frame: pd.DataFrame, old=(.80, .93), new=(.80, .99)) -> pd.DataFrame:
    old_route = route_frame(frame, *old).set_index("record_id")
    new_route = route_frame(frame, *new).set_index("record_id")
    changed = old_route.v8_routing_status.ne(new_route.v8_routing_status)
    ids = old_route.index[changed]
    base = frame.set_index("document_id").loc[ids]
    result = pd.DataFrame({"record_id": ids, "caption": base.original_text,
        "gold_label": base.final_v8_gold_label,
        "event_connection": base.event_connection,
        "primary_content_type": base.primary_content_type,
        "confidence": base.confidence,
        "old_route": old_route.loc[ids, "v8_routing_status"],
        "new_route": new_route.loc[ids, "v8_routing_status"],
        "old_automatic_decision": old_route.loc[ids, "v8_automatic_decision"],
        "new_automatic_decision": new_route.loc[ids, "v8_automatic_decision"],
        "reason_for_change": [f"excluded-content confidence {float(v):.2f} is below the new 0.99 automatic-exclusion threshold" for v in base.confidence]})
    return result.reset_index(drop=True)


def deterministic_scores(y_true: pd.Series, y_pred: pd.Series) -> dict:
    matrix = confusion_matrix(y_true, y_pred, labels=["include", "exclude"])
    tp_i, fn_i = matrix[0]
    fp_i, tp_e = matrix[1]
    include_precision = tp_i / (tp_i + fp_i) if tp_i + fp_i else 0
    include_recall = tp_i / (tp_i + fn_i) if tp_i + fn_i else 0
    exclude_precision = tp_e / (tp_e + fn_i) if tp_e + fn_i else 0
    exclude_recall = tp_e / (tp_e + fp_i) if tp_e + fp_i else 0
    include_f1 = 2*include_precision*include_recall/(include_precision+include_recall) if include_precision+include_recall else 0
    exclude_f1 = 2*exclude_precision*exclude_recall/(exclude_precision+exclude_recall) if exclude_precision+exclude_recall else 0
    return {"confusion_matrix": matrix.tolist(), "inclusion_precision": include_precision,
        "inclusion_recall": include_recall, "exclusion_precision": exclude_precision,
        "excluded_class_recall": exclude_recall, "included_class_f1": include_f1,
        "excluded_class_f1": exclude_f1, "automatic_binary_macro_f1": (include_f1+exclude_f1)/2,
        "gold_include_support": int((y_true == "include").sum()),
        "gold_exclude_support": int((y_true == "exclude").sum())}


def operating_point_metrics(frame: pd.DataFrame, include_threshold: float,
                            exclude_threshold: float) -> dict:
    routes = route_frame(frame, include_threshold, exclude_threshold)
    gold = frame.final_v8_gold_label.reset_index(drop=True)
    pred = routes.v8_final_inclusion.reset_index(drop=True)
    review = pred.eq("unresolved"); automatic = ~review
    scores = deterministic_scores(gold[automatic], pred[automatic])
    sklearn_f1 = precision_recall_fscore_support(gold[automatic], pred[automatic],
        labels=["include", "exclude"], average="macro", zero_division=0)[2]
    if abs(scores["automatic_binary_macro_f1"] - sklearn_f1) > 1e-12:
        raise AssertionError("Independent macro-F1 implementation disagrees with sklearn")
    ai = pred.eq("include"); ae = pred.eq("exclude")
    assisted = pred.where(~review, gold)
    assisted_scores = deterministic_scores(gold, assisted)
    return {"include_threshold": include_threshold, "exclude_threshold": exclude_threshold,
        "automatic_decision_n": int(automatic.sum()), "automatic_coverage": float(automatic.mean()),
        "automatic_include_n": int(ai.sum()), "automatic_exclude_n": int(ae.sum()),
        "review_n": int(review.sum()), "review_rate": float(review.mean()),
        "reviews_per_1000": float(review.mean()*1000),
        "automatic_inclusion_precision": scores["inclusion_precision"],
        "automatic_inclusion_recall": scores["inclusion_recall"],
        "automatic_exclusion_precision": scores["exclusion_precision"],
        "automatic_false_inclusions": int((ai & gold.eq("exclude")).sum()),
        "automatic_false_exclusions": int((ae & gold.eq("include")).sum()),
        "relevant_automatically_excluded": int((ae & gold.eq("include")).sum()),
        "irrelevant_automatically_included": int((ai & gold.eq("exclude")).sum()),
        "included_class_f1": scores["included_class_f1"],
        "excluded_class_recall": scores["excluded_class_recall"],
        "excluded_class_f1": scores["excluded_class_f1"],
        "automatic_binary_macro_f1": scores["automatic_binary_macro_f1"],
        "automatic_confusion_matrix": scores["confusion_matrix"],
        "automatic_gold_include_support": scores["gold_include_support"],
        "automatic_gold_exclude_support": scores["gold_exclude_support"],
        "simulated_assisted_inclusion_precision": assisted_scores["inclusion_precision"],
        "simulated_assisted_inclusion_recall": assisted_scores["inclusion_recall"],
        "simulated_assisted_included_class_f1": assisted_scores["included_class_f1"],
        "simulated_assisted_excluded_class_f1": assisted_scores["excluded_class_f1"],
        "simulated_assisted_binary_macro_f1": assisted_scores["automatic_binary_macro_f1"],
        "simulated_assisted_confusion_matrix": assisted_scores["confusion_matrix"]}


def write_append_only_manifest(paths: list[Path], manifest_path: Path,
                               metadata: dict) -> None:
    if manifest_path.exists():
        raise FileExistsError("Phase 8 manifest already exists")
    payload = {**metadata, "files": {str(path): sha256(path) for path in paths}}
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
