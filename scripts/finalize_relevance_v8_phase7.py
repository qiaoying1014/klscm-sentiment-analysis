"""Finalize Phase 7 from saved human and classifier artifacts only."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from marathon_absa.adjudication_phase7 import (
    GOLD_VERSION, evaluate_calibration, finalize_gold, overlay_calibration,
    simulate_thresholds, validate_supervisor_adjudication, write_manifest,
)


ROOT = Path("data/processed")
ADJ = ROOT / "relevance_v8_adjudication"
AGREE = ADJ / "agreement"
FINAL = ADJ / "final"
CAL = ROOT / "relevance_v8_calibration_results"


def _breakdowns(gold: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    metadata = candidates[["record_id", "language", "event_year", "candidate_scope",
        "disagreement_category", "primary_content_type", "original_caption"]]
    frame = gold.merge(metadata, on="record_id", validate="one_to_one")
    frame["hashtag_presence"] = frame.original_caption.str.contains("#", regex=False).map(
        {True: "has_hashtag", False: "no_hashtag"})
    rows = []
    for dimension in ["language", "disagreement_category", "candidate_scope",
                      "hashtag_presence", "primary_content_type"]:
        for group, part in frame.groupby(dimension, dropna=False):
            rows.append({"dimension": dimension, "group": str(group), "n": len(part),
                "include_n": int(part.final_v8_label.eq("include").sum()),
                "exclude_n": int(part.final_v8_label.eq("exclude").sum()),
                "consensus_n": int(part.resolution_method.eq("reviewer_consensus").sum()),
                "supervisor_n": int(part.resolution_method.eq("supervisor_adjudication").sum())})
    return pd.DataFrame(rows)


def _best_thresholds(table: pd.DataFrame) -> pd.DataFrame:
    valid = table[table.routing_rule_violations.eq(0)].copy()
    valid["meets_precision"] = valid.automatic_inclusion_precision.ge(.85)
    valid["meets_recall"] = valid.automatic_inclusion_recall.ge(.90)
    valid["meets_macro_f1"] = valid.automatic_binary_macro_f1.ge(.80)
    return valid.sort_values([
        "relevant_automatically_excluded", "meets_precision", "meets_recall",
        "meets_macro_f1", "automatic_exclusion_precision", "review_rate"],
        ascending=[True, False, False, False, False, True]).head(10)


def main() -> None:
    final_gold = FINAL / "relevance_v8_adjudicated_gold_v1.csv"
    final_manifest = FINAL / "relevance_v8_adjudicated_gold_v1.manifest.json"
    overlay_path = CAL / "relevance_v8_calibration_adjudicated_v1.csv"
    overlay_audit_path = CAL / "relevance_v8_calibration_adjudicated_v1_overlay_audit.csv"
    calibration_manifest = CAL / "relevance_v8_post_adjudication_v1.manifest.json"
    planned = [final_gold, final_manifest, overlay_path, overlay_audit_path,
               calibration_manifest, FINAL/"adjudication_statistics_v1.json"]
    if any(path.exists() for path in planned):
        raise FileExistsError("Phase 7 v1 is append-only and already exists")

    completed, reviewer_a, reviewer_b, validation = validate_supervisor_adjudication(
        AGREE/"relevance_v8_supervisor_adjudication.csv",
        AGREE/"relevance_v8_supervisor_adjudication.backup_20260730_231859_169617.csv",
        ADJ/"reviewer_imports/reviewer_a__v1.csv",
        ADJ/"reviewer_imports/reviewer_b__v1.csv",
        AGREE/"supervisor_package_manifest.json")
    candidates = pd.read_csv(CAL/"relevance_v8_adjudication_candidates.csv",
                             dtype=str, keep_default_na=False)
    gold, stats = finalize_gold(completed, reviewer_a, reviewer_b, candidates,
        final_gold, AGREE/"supervisor_package_manifest.json")
    FINAL.mkdir(parents=True, exist_ok=True)
    validation_path = FINAL/"supervisor_validation_v1.json"
    stats_path = FINAL/"adjudication_statistics_v1.json"
    breakdown_path = FINAL/"adjudication_breakdowns_v1.csv"
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    breakdown = _breakdowns(gold, candidates); breakdown.to_csv(breakdown_path, index=False)
    write_manifest([final_gold, validation_path, stats_path, breakdown_path], final_manifest,
                   {"gold_version": GOLD_VERSION, "records": 54, "api_calls": 0})

    predictions = pd.read_csv(CAL/"relevance_v8_calibration_predictions.csv",
                              keep_default_na=False)
    historical = pd.read_csv(ROOT/"relevance_v7_calibration_results/relevance_validation_sample.csv",
                             keep_default_na=False)
    overlay, audit = overlay_calibration(predictions, historical, gold,
                                         overlay_path, overlay_audit_path)
    metrics, errors = evaluate_calibration(overlay)
    metrics["input_prediction_file_sha256"] = validation.get("prediction_hash", "")
    metrics["no_new_classifier_run"] = True
    metrics["api_calls"] = 0
    metrics_path = CAL/"relevance_v8_post_adjudication_metrics_v1.json"
    error_path = CAL/"relevance_v8_post_adjudication_error_analysis_v1.csv"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    errors.to_csv(error_path, index=False, encoding="utf-8-sig")
    thresholds = simulate_thresholds(overlay)
    threshold_path = CAL/"relevance_v8_post_adjudication_threshold_analysis_v1.csv"
    thresholds.to_csv(threshold_path, index=False)
    best = _best_thresholds(thresholds)
    best_path = CAL/"relevance_v8_post_adjudication_threshold_best_candidates_v1.csv"
    best.to_csv(best_path, index=False)

    current = thresholds[(thresholds.include_threshold.eq(.80)) &
                         (thresholds.exclude_threshold.eq(.93))].iloc[0]
    top = best.iloc[0]
    meets = bool(current.automatic_inclusion_precision >= .85 and
                 current.automatic_inclusion_recall >= .90 and
                 current.automatic_binary_macro_f1 >= .80 and
                 current.relevant_automatically_excluded == 0)
    decision = "READY FOR BLIND HOLDOUT" if meets else "NOT READY FOR BLIND HOLDOUT"
    recommendation = "A. Keep current 0.80 / 0.93 thresholds." if meets else "B. Tune thresholds only."

    adjudication_report = Path("RELEVANCE_V8_FINAL_ADJUDICATION_REPORT.md")
    calibration_report = Path("RELEVANCE_V8_POST_ADJUDICATION_CALIBRATION_REPORT.md")
    adjudication_report.write_text(f"""# Relevance v8 final adjudication report

## Method and completion

Two blinded reviewers independently labeled the 54 disagreement-enriched records. They agreed on 23 and disagreed on 31. Supervisor `NBH` adjudicated every disagreement. Final labels use reviewer consensus where available and supervisor adjudication otherwise; model outputs and historical labels played no role in resolution.

Initial percent agreement remains **42.59%** and Cohen's kappa remains **0.1180**. This targeted sample was enriched for difficult historical-policy/model disagreements and is not a random representative sample; kappa must not be reported as corpus-wide inter-annotator reliability. It nevertheless demonstrates substantial ambiguity and different inclusion boundaries among difficult cases.

## Final results

- Gold version: `{GOLD_VERSION}`
- Consensus: {stats['reviewer_consensus_n']}
- Supervisor adjudicated: {stats['supervisor_adjudicated_n']}
- Final include/exclude: {stats['final_include_n']} / {stats['final_exclude_n']}
- Historical exclude to include: {stats['historical_exclude_to_include']}
- Historical include to exclude: {stats['historical_include_to_exclude']}
- Historical unchanged: {stats['historical_unchanged']}
- Supervisor selected Reviewer A / B: {stats['supervisor_selected_reviewer_a']} / {stats['supervisor_selected_reviewer_b']}
- Supervisor disagreement labels include/exclude: {stats['supervisor_include_n']} / {stats['supervisor_exclude_n']}

Reviewer alignment is descriptive and is not evidence that either reviewer was poor.

## Status and provenance

- Initial annotation agreement status: low on a disagreement-enriched sample
- Adjudication completion status: complete (31/31)
- Gold-standard resolution status: complete and versioned

The final CSV preserves historical labels, both reviewer decisions and evidence, supervisor provenance for disagreements, resolution method, timestamps, and source-manifest hash. Consensus rows intentionally have blank supervisor fields. Hashes are recorded in `{final_manifest}`.

## Limitations

The 54 records are a purposive difficult-case subset. Adjudication resolves these records but does not estimate corpus-wide reliability or replace validation on a fresh blind holdout.
""", encoding="utf-8")

    a = metrics["automatic_only_classifier_performance"]
    r = metrics["routing_performance"]
    h = metrics["simulated_human_assisted_performance"]
    calibration_report.write_text(f"""# Relevance v8 post-adjudication calibration report

## Evaluation protocol

No classifier was rerun and no API was called. The official 54-label adjudicated gold was overlaid by stable ID onto a new copy of the frozen 300-record calibration predictions. The other 246 gold labels remain unchanged. The obsolete review-to-exclude calculation is retained only in historical artifacts as `legacy_review_as_exclude`; it is not used here.

## Automatic-only classifier performance

- Decisions: {a['automatic_decision_n']} ({a['automatic_coverage']:.2%} coverage); include/exclude: {a['automatic_include_n']} / {a['automatic_exclude_n']}
- Inclusion precision/recall: {a['automatic_inclusion_precision']:.4f} / {a['automatic_inclusion_recall']:.4f}
- Exclusion precision: {a['automatic_exclusion_precision']:.4f}
- Included/excluded F1: {a['included_class_f1']:.4f} / {a['excluded_class_f1']:.4f}
- Binary macro-F1: {a['binary_macro_f1']:.4f}
- False inclusions/exclusions: {a['automatic_false_inclusions']} / {a['automatic_false_exclusions']}
- Confusion matrix (gold rows, prediction columns; include, exclude): `{a['confusion_matrix']}`

## Routing performance

- Review: {r['review_n']} ({r['review_rate']:.2%}; {r['review_workload_per_1000']:.1f} per 1,000)
- Relevant/irrelevant routed: {r['relevant_routed_to_review']} / {r['irrelevant_routed_to_review']}
- False-exclusion prevention: {r['false_exclusion_prevention']}
- Material disagreement/image-dependent/weak-connection reviews: {r['material_model_disagreements']} / {r['image_dependent_reviews']} / {r['weak_event_connection_reviews']}
- Weak-empty automatic exclusions: {r['weak_empty_auto_exclusions']}

## SIMULATED HUMAN-ASSISTED PERFORMANCE

- Inclusion precision/recall: {h['inclusion_precision']:.4f} / {h['inclusion_recall']:.4f}
- Included/excluded F1: {h['included_class_f1']:.4f} / {h['excluded_class_f1']:.4f}
- Binary macro-F1: {h['binary_macro_f1']:.4f}
- Confusion matrix: `{h['confusion_matrix']}`

This view substitutes gold labels only for review-routed cases and is not classifier-only performance.

## Threshold analysis and decision

All candidates preserve the fixed routing invariants; violations are explicitly counted in the threshold table. At current 0.80/0.93, automatic inclusion precision={current.automatic_inclusion_precision:.4f}, inclusion recall={current.automatic_inclusion_recall:.4f}, macro-F1={current.automatic_binary_macro_f1:.4f}, relevant automatically excluded={int(current.relevant_automatically_excluded)}, review rate={current.review_rate:.2%}.

The priority-ranked grid's leading candidate is {top.include_threshold:.2f}/{top.exclude_threshold:.2f}: inclusion precision={top.automatic_inclusion_precision:.4f}, inclusion recall={top.automatic_inclusion_recall:.4f}, macro-F1={top.automatic_binary_macro_f1:.4f}, relevant automatically excluded={int(top.relevant_automatically_excluded)}, review rate={top.review_rate:.2%}.

Recommendation: **{recommendation}** No threshold or routing change has been implemented.

## Remaining errors

Every automatic error and every routed include/exclude is recorded in `{error_path}` with caption, gold, saved decision, evidence fields, category, and research interpretation. Routed cases are not counted as classifier errors. The observed automatic-error categories determine whether the remaining weakness is systematic; no model modification was made.

## Readiness statuses

- Classifier quality status: {'meets all approved automatic targets' if meets else 'does not meet all approved automatic targets'}
- Annotation agreement status: low initial agreement on the disagreement-enriched subset
- Gold standard status: resolved and immutable v1
- Calibration status: complete using saved outputs
- Blind holdout readiness: {decision}

# {decision}

{'The current operating point meets the approved automatic targets and the difficult-case gold is fully resolved; freeze the protocol and proceed to a fresh blind holdout.' if meets else 'The current automatic operating point misses at least one approved target. The smallest justified next intervention is threshold tuning; do not revise routing, prompts, or policy unless threshold tuning cannot resolve the observed trade-off.'}
""", encoding="utf-8")

    write_manifest([overlay_path, overlay_audit_path, metrics_path, error_path,
                    threshold_path, best_path, adjudication_report, calibration_report],
                   calibration_manifest, {"calibration_version": "v8_adjudicated_v1",
                    "records": 300, "overlay_records": 54, "api_calls": 0,
                    "classifier_outputs_changed": False, "research_decision": decision})
    print(json.dumps({"adjudication": stats, "metrics": metrics,
                      "current_threshold": current.to_dict(),
                      "best_threshold": top.to_dict(), "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
