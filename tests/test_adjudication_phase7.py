from pathlib import Path

import pandas as pd
import pytest

from marathon_absa.adjudication_phase7 import (
    evaluate_calibration, finalize_gold, overlay_calibration,
    simulate_thresholds, validate_supervisor_adjudication, write_manifest,
)


def reviewer(ids, labels):
    return pd.DataFrame({"record_id": ids, "reviewer_label": labels,
        "evidence_span": ["e"]*len(ids), "rationale": ["r"]*len(ids),
        "reviewer_confidence": ["0.8"]*len(ids)})


def test_real_supervisor_validation_and_counts():
    base = Path("data/processed/relevance_v8_adjudication")
    completed, a, b, report = validate_supervisor_adjudication(
        base/"agreement/relevance_v8_supervisor_adjudication.csv",
        base/"agreement/relevance_v8_supervisor_adjudication.backup_20260730_231859_169617.csv",
        base/"reviewer_imports/reviewer_a__v1.csv", base/"reviewer_imports/reviewer_b__v1.csv",
        base/"agreement/supervisor_package_manifest.json")
    assert report["status"] == "passed"
    assert len(completed) == 31
    assert (a.set_index("record_id").reviewer_label != b.set_index("record_id").reviewer_label).sum() == 31


@pytest.mark.parametrize("field,value", [("adjudicator_label", ""),
                                           ("adjudicator_label", "maybe")])
def test_supervisor_rejects_missing_or_invalid_label(tmp_path, field, value):
    base = Path("data/processed/relevance_v8_adjudication")
    source = pd.read_csv(base/"agreement/relevance_v8_supervisor_adjudication.csv", dtype=str)
    source.loc[0, field] = value
    changed = tmp_path/"changed.csv"; source.to_csv(changed, index=False)
    with pytest.raises(ValueError, match="adjudicator labels"):
        validate_supervisor_adjudication(changed,
            base/"agreement/relevance_v8_supervisor_adjudication.backup_20260730_231859_169617.csv",
            base/"reviewer_imports/reviewer_a__v1.csv", base/"reviewer_imports/reviewer_b__v1.csv",
            base/"agreement/supervisor_package_manifest.json")


def test_supervisor_rejects_duplicate(tmp_path):
    base = Path("data/processed/relevance_v8_adjudication")
    source = pd.read_csv(base/"agreement/relevance_v8_supervisor_adjudication.csv")
    source.loc[1, "record_id"] = source.loc[0, "record_id"]
    changed = tmp_path/"duplicate.csv"; source.to_csv(changed, index=False)
    with pytest.raises(ValueError, match="31 unique"):
        validate_supervisor_adjudication(changed,
            base/"agreement/relevance_v8_supervisor_adjudication.backup_20260730_231859_169617.csv",
            base/"reviewer_imports/reviewer_a__v1.csv", base/"reviewer_imports/reviewer_b__v1.csv",
            base/"agreement/supervisor_package_manifest.json")


def test_real_finalization_is_54_with_23_consensus_and_append_only(tmp_path):
    base = Path("data/processed/relevance_v8_adjudication")
    completed, a, b, _ = validate_supervisor_adjudication(
        base/"agreement/relevance_v8_supervisor_adjudication.csv",
        base/"agreement/relevance_v8_supervisor_adjudication.backup_20260730_231859_169617.csv",
        base/"reviewer_imports/reviewer_a__v1.csv", base/"reviewer_imports/reviewer_b__v1.csv",
        base/"agreement/supervisor_package_manifest.json")
    candidates = pd.read_csv("data/processed/relevance_v8_calibration_results/relevance_v8_adjudication_candidates.csv")
    path = tmp_path/"gold.csv"
    gold, stats = finalize_gold(completed, a, b, candidates, path,
                               base/"agreement/supervisor_package_manifest.json")
    assert len(gold) == 54
    assert stats["reviewer_consensus_n"] == 23
    assert stats["supervisor_adjudicated_n"] == 31
    with pytest.raises(FileExistsError):
        finalize_gold(completed, a, b, candidates, path,
                      base/"agreement/supervisor_package_manifest.json")


def test_overlay_changes_exactly_54_gold_slots_and_no_frozen_columns(tmp_path):
    predictions = pd.read_csv("data/processed/relevance_v8_calibration_results/relevance_v8_calibration_predictions.csv")
    sample = pd.read_csv("data/processed/relevance_v7_calibration_results/relevance_validation_sample.csv")
    ids = predictions.document_id.iloc[:54]
    gold = pd.DataFrame({"record_id": ids, "final_v8_label": ["include"]*54,
                         "resolution_method": ["reviewer_consensus"]*54})
    overlay, audit = overlay_calibration(predictions, sample, gold,
        tmp_path/"overlay.csv", tmp_path/"audit.csv")
    assert len(overlay) == overlay.document_id.nunique() == 300
    assert len(audit) == 54
    pd.testing.assert_frame_equal(predictions.reset_index(drop=True),
                                  overlay[predictions.columns].reset_index(drop=True))


def test_automatic_metrics_exclude_review_and_never_map_review_to_exclude():
    frame = pd.DataFrame({
        "v8_routing_status": ["automatic_include", "automatic_exclude", "pending_review"],
        "v8_final_inclusion": ["include", "exclude", "unresolved"],
        "final_v8_gold_label": ["include", "include", "include"],
        "v8_routing_reason": ["included_event_content", "strong_exclusion_evidence", "image_dependent"],
        "v8_model_agreement": ["not_checked"]*3, "document_id": ["a","b","c"],
        "original_text": ["a","b","c"], "event_connection": ["supported"]*3,
        "primary_content_type": ["event_information"]*3, "confidence": [.9]*3,
        "v8_reason_code": ["x"]*3, "primary_language": ["English"]*3,
        "v8_image_dependent": [False,False,True]})
    metrics, _ = evaluate_calibration(frame)
    assert metrics["automatic_only_classifier_performance"]["automatic_decision_n"] == 2
    assert metrics["automatic_only_classifier_performance"]["confusion_matrix"] == [[1, 1], [0, 0]]
    assert metrics["simulated_human_assisted_performance"]["inclusion_recall"] == 2/3


def test_threshold_simulation_preserves_forced_review():
    source = pd.read_csv("data/processed/relevance_v8_calibration_results/relevance_v8_calibration_predictions.csv").iloc[:1].copy()
    source["v8_model_agreement"] = "disagree"
    source["final_v8_gold_label"] = "include"
    table = simulate_thresholds(source, (.8,), (.93,))
    assert table.loc[0, "review_rate"] == 1
    assert table.loc[0, "routing_rule_violations"] == 0


def test_manifest_hash_verification_and_no_api_path(tmp_path, monkeypatch):
    artifact = tmp_path/"artifact.txt"; artifact.write_text("immutable", encoding="utf-8")
    manifest_path = tmp_path/"manifest.json"
    manifest = write_manifest([artifact], manifest_path, {"api_calls": 0})
    assert manifest["api_calls"] == 0 and len(manifest["files"][str(artifact)]) == 64
    with pytest.raises(FileExistsError):
        write_manifest([artifact], manifest_path, {})


def test_real_phase7_reports_overlay_and_manifests_are_complete():
    import hashlib
    import json
    final = Path("data/processed/relevance_v8_adjudication/final")
    cal = Path("data/processed/relevance_v8_calibration_results")
    gold = pd.read_csv(final/"relevance_v8_adjudicated_gold_v1.csv")
    overlay = pd.read_csv(cal/"relevance_v8_calibration_adjudicated_v1.csv")
    audit = pd.read_csv(cal/"relevance_v8_calibration_adjudicated_v1_overlay_audit.csv")
    assert len(gold) == 54 and len(overlay) == 300 and len(audit) == 54
    assert overlay.document_id.nunique() == 300
    assert set(overlay.final_v8_gold_label) <= {"include", "exclude"}
    assert Path("RELEVANCE_V8_FINAL_ADJUDICATION_REPORT.md").exists()
    assert Path("RELEVANCE_V8_POST_ADJUDICATION_CALIBRATION_REPORT.md").exists()
    for manifest_path in [final/"relevance_v8_adjudicated_gold_v1.manifest.json",
                          cal/"relevance_v8_post_adjudication_v1.manifest.json"]:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        for raw, expected in manifest["files"].items():
            actual = hashlib.sha256(Path(raw).read_bytes()).hexdigest()
            assert actual == expected
