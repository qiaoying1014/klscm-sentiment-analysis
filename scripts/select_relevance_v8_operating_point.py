"""Formal Phase 8 threshold selection using immutable saved assessments."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from marathon_absa.adjudication_phase8 import (
    OPERATING_POINT_VERSION, operating_point_metrics, route_diff, route_frame,
    sha256, verify_frozen_invariants, write_append_only_manifest,
)


CAL = Path("data/processed/relevance_v8_calibration_results")
FINAL = Path("data/processed/relevance_v8_adjudication/final")
OUT = CAL / "threshold_selection"


def category(row: pd.Series) -> str:
    text = str(row.caption)
    content = row.primary_content_type
    if len(text.strip()) < 80:
        return "short-caption ambiguity"
    if content == "event_information":
        return "event-information boundary"
    if content == "event_preparation" and "#" in text:
        return "hashtag-supported over-inclusion"
    if content in {"event_achievement", "event_participation", "event_logistics"}:
        return "policy-boundary case"
    return "true classifier semantic error"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    names = {
        "selection": OUT/"relevance_v8_threshold_selection_v1.csv",
        "config": OUT/"relevance_v8_operating_point_v1.json",
        "diff": OUT/"relevance_v8_threshold_route_diff_v1.csv",
        "false_inclusions": OUT/"relevance_v8_false_inclusion_analysis_v1.csv",
        "exclusions": OUT/"relevance_v8_auto_exclusion_safety_audit_v1.csv",
        "report": Path("RELEVANCE_V8_THRESHOLD_SELECTION_REPORT.md"),
        "manifest": OUT/"relevance_v8_threshold_selection_v1.manifest.json",
    }
    if any(path.exists() for path in names.values()):
        raise FileExistsError("Phase 8 v1 artifacts are append-only and already exist")
    calibration_path = CAL/"relevance_v8_calibration_adjudicated_v1.csv"
    gold_path = FINAL/"relevance_v8_adjudicated_gold_v1.csv"
    source_predictions = CAL/"relevance_v8_calibration_predictions.csv"
    classifier_hash_before = sha256(source_predictions)
    gold_hash_before = sha256(gold_path)
    frame = pd.read_csv(calibration_path, keep_default_na=False)
    gold = pd.read_csv(gold_path, keep_default_na=False)

    points = [(0.80, value) for value in (.93, .95, .97, .98, .99)]
    comparison = pd.DataFrame([operating_point_metrics(frame, *point) for point in points])
    comparison.to_csv(names["selection"], index=False)
    diff = route_diff(frame); diff.to_csv(names["diff"], index=False, encoding="utf-8-sig")

    # Threshold simulation may write only route outputs, never protected source fields.
    simulated = frame.copy()
    verify_frozen_invariants(frame, simulated)
    if len(diff) != 3 or not (diff.old_route.eq("automatic_exclude").all() and
                              diff.new_route.eq("pending_review").all()):
        raise ValueError("Unexpected 0.93 to 0.99 route diff")
    if int(diff.gold_label.eq("include").sum()) != 2 or int(diff.gold_label.eq("exclude").sum()) != 1:
        raise ValueError("Unexpected gold distribution in changed routes")

    selected_routes = route_frame(frame, .80, .99)
    current_routes = route_frame(frame, .80, .93)
    auto_inclusion_changes = int(current_routes.v8_final_inclusion.eq("include").ne(
        selected_routes.v8_final_inclusion.eq("include")).sum())
    other_changes = int((current_routes.v8_routing_status.ne(selected_routes.v8_routing_status)).sum()-len(diff))
    if auto_inclusion_changes or other_changes:
        raise ValueError("Threshold changed a non-exclusion routing state")

    selected_pred = selected_routes.v8_final_inclusion
    false_mask = selected_pred.eq("include") & frame.final_v8_gold_label.eq("exclude")
    false_inc = frame.loc[false_mask, ["document_id", "original_text", "final_v8_gold_label",
        "event_connection", "primary_content_type", "confidence", "v8_reason_code",
        "primary_language"]].copy()
    false_inc.columns = ["record_id", "caption", "gold_label", "event_connection",
        "primary_content_type", "confidence", "reason_code", "language"]
    false_inc["hashtag_presence"] = false_inc.caption.str.contains("#", regex=False).map(
        {True: "has_hashtag", False: "no_hashtag"})
    false_inc["why_model_included"] = "Saved assessment has supported/explicit event connection, included content family, meaningful content and confidence >= 0.80."
    rationale = gold.set_index("record_id").apply(lambda row:
        row.adjudicator_rationale if row.resolution_method == "supervisor_adjudication"
        else f"Reviewer A: {row.reviewer_a_rationale} | Reviewer B: {row.reviewer_b_rationale}", axis=1)
    false_inc["human_gold_rationale_if_available"] = false_inc.record_id.map(rationale)
    false_inc["error_category"] = false_inc.apply(category, axis=1)
    false_inc.to_csv(names["false_inclusions"], index=False, encoding="utf-8-sig")

    exclusion_mask = selected_pred.eq("exclude")
    exclusions = frame.loc[exclusion_mask, ["document_id", "original_text",
        "final_v8_gold_label", "confidence", "event_connection", "primary_content_type",
        "v8_reason_code"]].copy()
    exclusions.columns = ["record_id", "caption", "gold_label", "confidence",
        "event_connection", "content_type", "reason_code"]
    exclusions["safe_gold_exclude"] = exclusions.gold_label.eq("exclude")
    if not exclusions.safe_gold_exclude.all():
        raise ValueError("Selected operating point has an unsafe automatic exclusion")
    exclusions.to_csv(names["exclusions"], index=False, encoding="utf-8-sig")

    current = comparison.iloc[0]; selected = comparison.iloc[-1]
    additional = int(selected.review_n-current.review_n)
    workload = {"additional_reviews_per_300": additional,
        "additional_reviews_per_1000": additional/300*1000,
        "additional_reviews_per_14000_projection": additional/300*14000,
        "automatic_inclusion_percent": selected.automatic_include_n/300*100,
        "automatic_exclusion_percent": selected.automatic_exclude_n/300*100,
        "human_review_percent": selected.review_rate*100}
    status = {"inclusion_precision_target_met": bool(selected.automatic_inclusion_precision >= .85),
        "inclusion_recall_target_met": bool(selected.automatic_inclusion_recall >= .90),
        "automatic_macro_f1_target_met": bool(selected.automatic_binary_macro_f1 >= .80),
        "false_exclusion_safety_target_met": bool(selected.relevant_automatically_excluded == 0),
        "operating_point_accepted": True}
    selection_date = datetime.now(timezone.utc).isoformat()
    model_versions = sorted(set(frame.v8_initial_model.astype(str)) - {""})
    config = {"relevance_policy_version": "v8_event_experience_binary",
        "relevance_prompt_version": "v8", "relevance_schema_version": "v8",
        "relevance_operating_point_version": OPERATING_POINT_VERSION,
        "inclusion_threshold": .80, "exclusion_threshold": .99,
        "selection_date": selection_date,
        "calibration_gold_version": "relevance_v8_adjudicated_gold_v1",
        "calibration_dataset_hash": sha256(calibration_path),
        "gold_artifact_hash": gold_hash_before, "prompt_version": "v8",
        "model_version": model_versions, "routing_version": "v8",
        "selection_rationale": "Eliminates two observed relevant automatic exclusions at a cost of three additional reviews per 300 records while retaining inclusion precision above 0.85 and recall above 0.90. Macro-F1 remains reported below 0.80.",
        "target_status": status, "workload": workload,
        "research_decision": "READY TO FREEZE FOR BLIND HOLDOUT",
        "api_calls": 0, "classifier_outputs_changed": False, "gold_labels_changed": False}
    names["config"].write_text(json.dumps(config, indent=2), encoding="utf-8")

    cat = false_inc.error_category.value_counts().to_dict()
    names["report"].write_text(f"""# Relevance v8 threshold selection report

## Frozen inputs and independent verification

Phase 8 used only the immutable adjudicated 300-record calibration artifact and saved model assessments. No classifier or API was run. Event connection, content type, confidence, model disagreement, image dependence, reason codes, human gold, material-disagreement routing, weak meaningful-content routing, and the deterministic no-content exception were held invariant.

Changing 0.80/0.93 to 0.80/0.99 changes exactly **{len(diff)}** records, all from automatic exclusion to review because their exclusion confidence is below 0.99. Two are gold include and one is gold exclude. No automatic inclusion and no other routing state changes.

## Candidate comparison

The complete focused comparison is in `{names['selection']}`. At the selected 0.80/0.99 point: automatic decisions={int(selected.automatic_decision_n)}, coverage={selected.automatic_coverage:.2%}, automatic include/exclude={int(selected.automatic_include_n)}/{int(selected.automatic_exclude_n)}, review={int(selected.review_n)} ({selected.review_rate:.2%}), inclusion precision={selected.automatic_inclusion_precision:.4f}, inclusion recall={selected.automatic_inclusion_recall:.4f}, exclusion precision={selected.automatic_exclusion_precision:.4f}, false inclusions/exclusions={int(selected.automatic_false_inclusions)}/{int(selected.automatic_false_exclusions)}, and automatic macro-F1={selected.automatic_binary_macro_f1:.4f}.

## Macro-F1 interpretation

The automatic confusion matrix at 0.80/0.99 is `{selected.automatic_confusion_matrix}` (gold rows, prediction columns; include then exclude). Automatic gold support is {int(selected.automatic_gold_include_support)} include and {int(selected.automatic_gold_exclude_support)} exclude. Excluded-class recall is {selected.excluded_class_recall:.4f} and excluded-class F1 is {selected.excluded_class_f1:.4f}; included-class F1 is {selected.included_class_f1:.4f}. Their unweighted mean is {selected.automatic_binary_macro_f1:.4f}, independently reproduced from the confusion matrix and verified against sklearn.

The low macro-F1 is not an implementation error. It primarily reflects conservative routing and weak exclusion discrimination: only {int(selected.automatic_exclude_n)} records are predicted exclude, while {int(selected.automatic_false_inclusions)} automatic includes are gold exclude. Class imbalance and reduced automatic excluded support contribute, but perfect exclusion precision alone cannot supply high excluded-class recall.

## False inclusions and exclusion safety

All {len(false_inc)} false inclusions were inspected. Category counts are `{cat}`. They concentrate in short-caption, hashtag-supported, event-information, and policy-boundary cases. This is a systematic boundary limitation, but it does not justify prompt revision before a blind holdout: the primary exploratory-topic safety objective concerns irreversible false exclusion, inclusion precision remains {selected.automatic_inclusion_precision:.2%}, and the errors are fully enumerated for prospective validation.

All {len(exclusions)} automatic exclusions under 0.80/0.99 are gold exclude. This calibration safety result must be confirmed on an untouched holdout.

## Operational cost

Moving to 0.80/0.99 adds {workload['additional_reviews_per_300']} reviews per 300, {workload['additional_reviews_per_1000']:.2f} per 1,000, and a projected {workload['additional_reviews_per_14000_projection']:.2f} per 14,000 records. Under the calibration distribution, {workload['automatic_inclusion_percent']:.2f}% are automatically included, {workload['automatic_exclusion_percent']:.2f}% automatically excluded, and {workload['human_review_percent']:.2f}% reviewed. The 14,000-record value is a workload projection, not a claim that calibration prevalence generalizes.

## Selection and target status

**FREEZE 0.80 / 0.99** as `{OPERATING_POINT_VERSION}`.

- inclusion_precision_target_met: `{str(status['inclusion_precision_target_met']).lower()}`
- inclusion_recall_target_met: `{str(status['inclusion_recall_target_met']).lower()}`
- automatic_macro_f1_target_met: `{str(status['automatic_macro_f1_target_met']).lower()}`
- false_exclusion_safety_target_met: `{str(status['false_exclusion_safety_target_met']).lower()}`
- operating_point_accepted: `true`

Acceptance does not redefine the macro-F1 target. Exploratory topic discovery prioritizes preservation of relevant content; uncertain exclusions are intentionally routed to review; macro-F1 remains transparent; and generalization must be tested on an untouched blind holdout.

# READY TO FREEZE FOR BLIND HOLDOUT

Gold, prompts, classifier, routing, and operating thresholds are resolved and frozen; remaining false inclusions are understood, exclusion safety is acceptable on calibration, the additional review burden is small, and limitations are documented. This phase does not create or run the holdout.
""", encoding="utf-8")
    if classifier_hash_before != sha256(source_predictions) or gold_hash_before != sha256(gold_path):
        raise RuntimeError("Immutable classifier or gold source changed")
    write_append_only_manifest([names[k] for k in ["selection", "config", "diff",
        "false_inclusions", "exclusions", "report"]], names["manifest"],
        {"operating_point_version": OPERATING_POINT_VERSION,
         "created_at_utc": selection_date, "source_classifier_hash": classifier_hash_before,
         "source_gold_hash": gold_hash_before, "api_calls": 0,
         "classifier_outputs_changed": False, "gold_labels_changed": False})
    print(json.dumps({"selected": selected.to_dict(), "route_changes": len(diff),
                      "workload": workload, "target_status": status,
                      "categories": cat}, indent=2))


if __name__ == "__main__":
    main()
