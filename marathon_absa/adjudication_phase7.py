from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .relevance_v8 import FINAL_LABELS, route_assessment


GOLD_VERSION = "relevance_v8_adjudicated_gold_v1"
FORBIDDEN_SUPERVISOR_FIELDS = {
    "historical_v7_label", "model_prediction", "model_confidence",
    "classifier_prediction", "classifier_confidence", "hidden_validation_key",
}
SUPERVISOR_MUTABLE_FIELDS = {
    "adjudicator_label", "adjudicator_rationale", "adjudicator_name",
    "adjudication_timestamp",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_supervisor_adjudication(
    completed_path: Path,
    blank_package_path: Path,
    reviewer_a_path: Path,
    reviewer_b_path: Path,
    package_manifest_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Validate a completed worksheet without trusting its mutable fields."""
    completed = pd.read_csv(completed_path, dtype=str, keep_default_na=False)
    blank = pd.read_csv(blank_package_path, dtype=str, keep_default_na=False)
    reviewer_a = pd.read_csv(reviewer_a_path, dtype=str, keep_default_na=False)
    reviewer_b = pd.read_csv(reviewer_b_path, dtype=str, keep_default_na=False)
    if len(completed) != 31 or completed.record_id.nunique() != 31:
        raise ValueError("Supervisor worksheet must contain exactly 31 unique records")
    if len(blank) != 31 or blank.record_id.nunique() != 31:
        raise ValueError("Blank supervisor package must contain exactly 31 unique records")
    if set(completed.record_id) != set(blank.record_id):
        missing = sorted(set(blank.record_id) - set(completed.record_id))
        extra = sorted(set(completed.record_id) - set(blank.record_id))
        raise ValueError(f"Supervisor record mismatch; missing={missing}, extra={extra}")
    if FORBIDDEN_SUPERVISOR_FIELDS & set(completed.columns):
        raise ValueError("Classifier or historical fields leaked into supervisor worksheet")
    immutable = [column for column in blank.columns
                 if column not in SUPERVISOR_MUTABLE_FIELDS | {"record_id"}]
    left = completed.set_index("record_id").sort_index()[immutable]
    right = blank.set_index("record_id").sort_index()[immutable]
    if not left.equals(right):
        changed = [c for c in immutable if not left[c].equals(right[c])]
        raise ValueError(f"Supervisor package immutable fields changed: {changed}")
    labels = completed.adjudicator_label.str.strip().str.lower()
    if not labels.isin(FINAL_LABELS).all():
        bad = completed.loc[~labels.isin(FINAL_LABELS), "record_id"].tolist()
        raise ValueError(f"Missing or invalid adjudicator labels: {bad}")
    for field in ["adjudicator_rationale", "adjudicator_name", "adjudication_timestamp"]:
        bad = completed[field].str.strip().eq("")
        if bad.any():
            raise ValueError(f"Missing {field}: {completed.loc[bad, 'record_id'].tolist()}")
    parsed = pd.to_datetime(completed.adjudication_timestamp, errors="coerce", utc=True)
    if parsed.isna().any():
        raise ValueError("Malformed adjudication timestamp")
    agreement = reviewer_a[["record_id", "reviewer_label"]].merge(
        reviewer_b[["record_id", "reviewer_label"]], on="record_id",
        suffixes=("_a", "_b"), validate="one_to_one",
    )
    expected = set(agreement.loc[agreement.reviewer_label_a.ne(agreement.reviewer_label_b), "record_id"])
    if expected != set(completed.record_id):
        raise ValueError("Worksheet is not exactly the reviewer-disagreement set")
    manifest = json.loads(package_manifest_path.read_text(encoding="utf-8-sig"))
    failures = []
    for raw_path, expected_hash in manifest["files"].items():
        path = Path(raw_path)
        if path.resolve() == completed_path.resolve():
            path = blank_package_path
        if not path.exists() or _sha256(path) != expected_hash:
            failures.append(raw_path)
    if failures:
        raise ValueError(f"Supervisor package manifest integrity failed: {failures}")
    report = {
        "status": "passed", "records": 31, "unique_record_ids": 31,
        "expected_disagreements": 31, "immutable_fields_match": True,
        "hidden_fields_absent": True, "package_manifest_integrity": True,
        "completed_file_sha256": _sha256(completed_path),
        "blank_package_sha256": _sha256(blank_package_path),
    }
    return completed, reviewer_a, reviewer_b, report


def finalize_gold(
    completed: pd.DataFrame,
    reviewer_a: pd.DataFrame,
    reviewer_b: pd.DataFrame,
    candidates: pd.DataFrame,
    output_path: Path,
    source_manifest_path: Path,
) -> tuple[pd.DataFrame, dict]:
    if output_path.exists():
        raise FileExistsError(f"Append-only final gold already exists: {output_path}")
    a = reviewer_a.rename(columns={c: f"reviewer_a_{c}" for c in [
        "reviewer_label", "evidence_span", "rationale", "reviewer_confidence"]})
    b = reviewer_b.rename(columns={c: f"reviewer_b_{c}" for c in [
        "reviewer_label", "evidence_span", "rationale", "reviewer_confidence"]})
    keep = ["record_id", "reviewer_a_reviewer_label", "reviewer_a_evidence_span",
            "reviewer_a_rationale", "reviewer_a_reviewer_confidence"]
    gold = a[keep].merge(b[["record_id", "reviewer_b_reviewer_label",
        "reviewer_b_evidence_span", "reviewer_b_rationale",
        "reviewer_b_reviewer_confidence"]], on="record_id", validate="one_to_one")
    gold = gold.merge(candidates[["record_id", "human_label_v7"]], on="record_id",
                      how="left", validate="one_to_one")
    if gold.human_label_v7.isna().any():
        raise ValueError("Historical labels missing for disputed records")
    gold = gold.merge(completed[["record_id", "adjudicator_label",
        "adjudicator_rationale", "adjudicator_name", "adjudication_timestamp"]],
        on="record_id", how="left", validate="one_to_one")
    agreement = gold.reviewer_a_reviewer_label.eq(gold.reviewer_b_reviewer_label)
    if int(agreement.sum()) != 23 or int((~agreement).sum()) != 31:
        raise ValueError("Expected exactly 23 consensus and 31 adjudicated records")
    gold["reviewer_agreement"] = agreement
    gold["final_v8_label"] = gold.reviewer_a_reviewer_label.where(
        agreement, gold.adjudicator_label)
    gold["resolution_method"] = agreement.map(
        {True: "reviewer_consensus", False: "supervisor_adjudication"})
    gold.loc[agreement, ["adjudicator_label", "adjudicator_rationale",
                         "adjudicator_name", "adjudication_timestamp"]] = ""
    gold["gold_version"] = GOLD_VERSION
    gold["finalization_timestamp"] = _utc_now()
    gold["source_manifest_hash"] = _sha256(source_manifest_path)
    gold = gold.rename(columns={
        "human_label_v7": "historical_v7_label",
        "reviewer_a_reviewer_label": "reviewer_a_label",
        "reviewer_a_reviewer_confidence": "reviewer_a_confidence",
        "reviewer_b_reviewer_label": "reviewer_b_label",
        "reviewer_b_reviewer_confidence": "reviewer_b_confidence",
        "adjudicator_name": "adjudicator_identity",
    })
    if len(gold) != 54 or not gold.final_v8_label.isin(FINAL_LABELS).all():
        raise ValueError("Final gold construction failed")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gold.to_csv(output_path, index=False, encoding="utf-8-sig")
    historical = gold.historical_v7_label.replace({"relevant": "include", "irrelevant": "exclude"})
    stats = {
        "total_disputed_records": 54,
        "reviewer_consensus_n": int(agreement.sum()),
        "supervisor_adjudicated_n": int((~agreement).sum()),
        "final_include_n": int(gold.final_v8_label.eq("include").sum()),
        "final_exclude_n": int(gold.final_v8_label.eq("exclude").sum()),
        "historical_exclude_to_include": int((historical.eq("exclude") & gold.final_v8_label.eq("include")).sum()),
        "historical_include_to_exclude": int((historical.eq("include") & gold.final_v8_label.eq("exclude")).sum()),
        "historical_unchanged": int(historical.eq(gold.final_v8_label).sum()),
        "supervisor_selected_reviewer_a": int(gold.loc[~agreement, "final_v8_label"].eq(gold.loc[~agreement, "reviewer_a_label"]).sum()),
        "supervisor_selected_reviewer_b": int(gold.loc[~agreement, "final_v8_label"].eq(gold.loc[~agreement, "reviewer_b_label"]).sum()),
        "supervisor_include_n": int(gold.loc[~agreement, "final_v8_label"].eq("include").sum()),
        "supervisor_exclude_n": int(gold.loc[~agreement, "final_v8_label"].eq("exclude").sum()),
    }
    return gold, stats


def overlay_calibration(
    predictions: pd.DataFrame,
    historical_sample: pd.DataFrame,
    gold: pd.DataFrame,
    output_path: Path,
    audit_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if output_path.exists() or audit_path.exists():
        raise FileExistsError("Append-only calibration overlay output already exists")
    if len(predictions) != 300 or predictions.document_id.nunique() != 300:
        raise ValueError("Frozen prediction set must contain 300 unique IDs")
    base = historical_sample.drop_duplicates("document_id")
    base = base[base.document_id.isin(predictions.document_id)]
    if len(base) != 300:
        raise ValueError("Historical calibration labels do not align to predictions")
    old = base.set_index("document_id").human_relevance.replace(
        {"relevant": "include", "irrelevant": "exclude"})
    overlay = predictions.copy()
    overlay["previous_gold_label"] = overlay.document_id.map(old)
    final_map = gold.set_index("record_id").final_v8_label
    if not set(final_map.index).issubset(set(overlay.document_id)):
        raise ValueError("Unmatched adjudication IDs")
    overlay["final_v8_gold_label"] = overlay.document_id.map(final_map).fillna(
        overlay.previous_gold_label)
    if not predictions.reset_index(drop=True).equals(
            overlay[predictions.columns].reset_index(drop=True)):
        raise ValueError("Frozen classifier columns changed")
    if not overlay.final_v8_gold_label.isin(FINAL_LABELS).all():
        raise ValueError("Overlay contains malformed labels")
    audit = pd.DataFrame({
        "record_id": gold.record_id,
        "previous_gold_label": gold.record_id.map(old),
        "final_v8_gold_label": gold.final_v8_label,
        "resolution_method": gold.resolution_method,
    })
    audit["changed"] = audit.previous_gold_label.ne(audit.final_v8_gold_label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.to_csv(output_path, index=False, encoding="utf-8-sig")
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    return overlay, audit


def _scores(y_true: pd.Series, y_pred: pd.Series) -> dict:
    inc = precision_recall_fscore_support(y_true, y_pred, labels=["include"],
                                         average="macro", zero_division=0)
    exc = precision_recall_fscore_support(y_true, y_pred, labels=["exclude"],
                                         average="macro", zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred,
        labels=["include", "exclude"], average="macro", zero_division=0)[2]
    return {"inclusion_precision": float(inc[0]), "inclusion_recall": float(inc[1]),
            "included_class_f1": float(inc[2]), "excluded_class_f1": float(exc[2]),
            "binary_macro_f1": float(macro),
            "confusion_matrix": confusion_matrix(y_true, y_pred,
                labels=["include", "exclude"]).tolist()}


def evaluate_calibration(overlay: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    auto = overlay.v8_routing_status.str.startswith("automatic")
    review = overlay.v8_final_inclusion.eq("unresolved")
    if not (~auto).equals(review):
        raise ValueError("Unexpected routing status in frozen outputs")
    automatic = overlay.loc[auto].copy()
    auto_scores = _scores(automatic.final_v8_gold_label, automatic.v8_final_inclusion)
    auto_inc = automatic.v8_final_inclusion.eq("include")
    auto_exc = automatic.v8_final_inclusion.eq("exclude")
    auto_metrics = {
        "automatic_decision_n": int(auto.sum()), "automatic_coverage": float(auto.mean()),
        "automatic_include_n": int(auto_inc.sum()), "automatic_exclude_n": int(auto_exc.sum()),
        "automatic_inclusion_precision": auto_scores["inclusion_precision"],
        "automatic_inclusion_recall": auto_scores["inclusion_recall"],
        "automatic_exclusion_precision": float(automatic.loc[auto_exc, "final_v8_gold_label"].eq("exclude").mean()),
        **{k: auto_scores[k] for k in ["included_class_f1", "excluded_class_f1", "binary_macro_f1", "confusion_matrix"]},
    }
    false_inc = auto_inc & automatic.final_v8_gold_label.eq("exclude")
    false_exc = auto_exc & automatic.final_v8_gold_label.eq("include")
    auto_metrics.update({"automatic_false_inclusions": int(false_inc.sum()),
        "automatic_false_exclusions": int(false_exc.sum()),
        "relevant_automatically_excluded": int(false_exc.sum()),
        "irrelevant_automatically_included": int(false_inc.sum())})
    routed = overlay.loc[review]
    reason = routed.v8_routing_reason
    routing = {"review_n": int(review.sum()), "review_rate": float(review.mean()),
        "relevant_routed_to_review": int(routed.final_v8_gold_label.eq("include").sum()),
        "irrelevant_routed_to_review": int(routed.final_v8_gold_label.eq("exclude").sum()),
        "false_exclusion_prevention": int(routed.final_v8_gold_label.eq("include").sum()),
        "material_model_disagreements": int(reason.eq("material_model_disagreement").sum()),
        "image_dependent_reviews": int(reason.eq("image_dependent").sum()),
        "weak_event_connection_reviews": int(reason.eq("weak_event_connection").sum()),
        "weak_empty_auto_exclusions": int(overlay.v8_routing_status.eq("automatic_exclude_weak_empty").sum()),
        "review_workload_per_1000": float(review.mean() * 1000)}
    routing["material_model_disagreements"] = max(
        routing["material_model_disagreements"], int(routed.v8_model_agreement.eq("disagree").sum()))
    assisted_pred = overlay.v8_final_inclusion.where(~review, overlay.final_v8_gold_label)
    assisted = _scores(overlay.final_v8_gold_label, assisted_pred)
    report = {"automatic_only_classifier_performance": auto_metrics,
              "routing_performance": routing,
              "simulated_human_assisted_performance": assisted,
              "legacy_review_as_exclude": "preserved only in previous artifacts; not used here"}
    errors = overlay.loc[(auto & overlay.v8_final_inclusion.ne(overlay.final_v8_gold_label)) | review].copy()
    errors["automatic_prediction"] = errors.v8_final_inclusion
    errors["gold_label"] = errors.final_v8_gold_label
    errors["error_category"] = errors.apply(_error_category, axis=1)
    errors["research_interpretation"] = errors.apply(_interpretation, axis=1)
    columns = ["document_id", "original_text", "gold_label", "automatic_prediction",
        "event_connection", "primary_content_type", "confidence", "v8_reason_code",
        "primary_language", "error_category", "research_interpretation", "v8_routing_status"]
    errors = errors[columns].rename(columns={"document_id": "record_id", "original_text": "caption",
        "v8_reason_code": "reason_code", "primary_language": "language"})
    return report, errors


def _error_category(row: pd.Series) -> str:
    if row.v8_final_inclusion == "unresolved":
        if bool(row.v8_image_dependent): return "image_dependence"
        if row.v8_event_connection == "weak": return "hashtag_context"
        return "routing_issue"
    mapping = {"generic_running": "generic_running", "event_emotional_experience": "emotional_experience",
        "event_physical_experience": "physical_experience", "promotional_only": "promotion_boundary",
        "event_information": "event_information", "other_event": "other_event"}
    content = row.get("v8_primary_content_type", row.get("primary_content_type", ""))
    return mapping.get(content, "short_caption" if len(str(row.original_text)) < 60 else "true_model_error")


def _interpretation(row: pd.Series) -> str:
    if row.v8_final_inclusion == "unresolved":
        return "Appropriately routed for human resolution; not an automatic classifier error."
    return "Saved automatic decision conflicts with the finalized v8 human gold label."


def simulate_thresholds(overlay: pd.DataFrame,
                        include_thresholds=(.65, .70, .75, .80, .85, .90),
                        exclude_thresholds=(.85, .90, .93, .95, .97, .99)) -> pd.DataFrame:
    rows = []
    base_fields = ["document_id", "event_connection", "primary_content_type",
        "secondary_content_types", "meaningful_content_present", "event_link_evidence",
        "analytical_content_evidence", "confidence", "contradiction_present",
        "contradiction_note", "image_dependent", "requires_review_recommendation",
        "review_reason", "language_observed", "code_switching_note", "short_explanation"]
    for inc in include_thresholds:
        for exc in exclude_thresholds:
            if exc <= inc: continue
            routed = []
            violations = 0
            for _, row in overlay.iterrows():
                assessment = {field: row[field] for field in base_fields}
                # CSV booleans must be normalized before applying the frozen router.
                for field in ["meaningful_content_present", "contradiction_present", "image_dependent", "requires_review_recommendation"]:
                    assessment[field] = str(assessment[field]).lower() == "true"
                if row.v8_routing_reason == "material_model_disagreement" or row.v8_model_agreement == "disagree":
                    result = {"v8_final_inclusion": "unresolved", "v8_routing_status": "pending_review", "v8_routing_reason": "material_model_disagreement"}
                else:
                    result = route_assessment(assessment, include_threshold=inc, exclude_threshold=exc)
                routed.append(result)
                if ((assessment["image_dependent"] or assessment["contradiction_present"] or
                     (assessment["event_connection"] == "weak" and result["v8_routing_status"] != "automatic_exclude_weak_empty"))
                    and result["v8_final_inclusion"] != "unresolved"):
                    violations += 1
            frame = pd.DataFrame(routed)
            gold = overlay.final_v8_gold_label.reset_index(drop=True)
            review = frame.v8_final_inclusion.eq("unresolved")
            auto = ~review
            auto_scores = _scores(gold[auto], frame.loc[auto, "v8_final_inclusion"])
            ai = frame.v8_final_inclusion.eq("include"); ae = frame.v8_final_inclusion.eq("exclude")
            assisted_pred = frame.v8_final_inclusion.where(~review, gold)
            assisted = _scores(gold, assisted_pred)
            rows.append({"include_threshold": inc, "exclude_threshold": exc,
                "automatic_coverage": float(auto.mean()), "review_rate": float(review.mean()),
                "reviews_per_1000": float(review.mean()*1000),
                "automatic_inclusion_precision": float(gold[ai].eq("include").mean()) if ai.any() else None,
                "automatic_exclusion_precision": float(gold[ae].eq("exclude").mean()) if ae.any() else None,
                "automatic_false_inclusions": int((ai & gold.eq("exclude")).sum()),
                "automatic_false_exclusions": int((ae & gold.eq("include")).sum()),
                "relevant_automatically_excluded": int((ae & gold.eq("include")).sum()),
                "automatic_binary_macro_f1": auto_scores["binary_macro_f1"],
                "automatic_inclusion_recall": auto_scores["inclusion_recall"],
                "simulated_assisted_inclusion_precision": assisted["inclusion_precision"],
                "simulated_assisted_inclusion_recall": assisted["inclusion_recall"],
                "simulated_assisted_binary_macro_f1": assisted["binary_macro_f1"],
                "routing_rule_violations": violations})
    return pd.DataFrame(rows)


def write_manifest(paths: list[Path], manifest_path: Path, metadata: dict) -> dict:
    if manifest_path.exists():
        raise FileExistsError(f"Append-only manifest already exists: {manifest_path}")
    manifest = {**metadata, "created_at_utc": _utc_now(),
                "files": {str(path): _sha256(path) for path in paths}}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
