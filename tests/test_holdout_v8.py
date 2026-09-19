import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from marathon_absa.holdout_v8 import (
    HOLDOUT_SEED, assert_holdout_scoring_allowed, stratified_sample,
)


ROOT = Path("data/processed")
OUT = ROOT/"relevance_v8_holdout"


def test_holdout_excludes_all_identifiable_development_and_calibration_ids():
    holdout = pd.read_csv(OUT/"relevance_v8_blind_holdout_v1.csv")
    report = json.loads((OUT/"relevance_v8_holdout_sampling_report_v1.json").read_text(encoding="utf-8"))
    development = set()
    for recorded_path in report["development_artifact_counts"]:
        path = Path(recorded_path)
        columns = pd.read_csv(path, nrows=0).columns
        id_column = "document_id" if "document_id" in columns else "record_id"
        development.update(pd.read_csv(path, usecols=[id_column], dtype=str, keep_default_na=False)[id_column])
    calibration = set(pd.read_csv(ROOT/"relevance_v8_calibration_results/relevance_v8_calibration_predictions.csv").document_id)
    selected = set(holdout.source_record_id)
    assert not selected & development
    assert not selected & calibration


def test_sampling_frame_has_no_duplicate_leakage():
    frame = pd.read_csv(OUT/"relevance_v8_holdout_sampling_frame_v1.csv")
    assert not frame.document_id.duplicated().any()
    assert not frame.text_hash.duplicated().any()
    usable = frame.near_duplicate_key.fillna("").str.len().ge(8)
    assert not frame.loc[usable, "near_duplicate_key"].duplicated().any()


def test_sampling_is_reproducible_and_stratum_allocation_exact():
    frame = pd.read_csv(OUT/"relevance_v8_holdout_sampling_frame_v1.csv")
    first, allocation = stratified_sample(frame, 300, HOLDOUT_SEED)
    second, _ = stratified_sample(frame, 300, HOLDOUT_SEED)
    assert first.document_id.tolist() == second.document_id.tolist()
    holdout = pd.read_csv(OUT/"relevance_v8_blind_holdout_v1.csv")
    assert set(first.document_id) == set(holdout.source_record_id)
    actual = holdout.sampling_stratum.value_counts().sort_index()
    expected = allocation.set_index("sampling_stratum").sample_n
    expected = expected[expected.gt(0)].sort_index()
    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_reviewer_files_are_independently_randomized_and_blinded():
    a = pd.read_csv(OUT/"relevance_v8_holdout_reviewer_a.csv", keep_default_na=False)
    b = pd.read_csv(OUT/"relevance_v8_holdout_reviewer_b.csv", keep_default_na=False)
    assert set(a.holdout_id) == set(b.holdout_id) and len(a) == len(b) == 300
    assert a.holdout_id.tolist() != b.holdout_id.tolist()
    assert list(a.columns) == ["holdout_id", "caption", "permitted_metadata", "language",
        "label", "evidence_span", "rationale", "confidence", "comments"]
    hidden = ["prediction", "routing", "reason_code", "gold", "content_type", "event_connection"]
    assert not any(token in column.lower() for column in a.columns for token in hidden)
    assert a[["label", "evidence_span", "rationale", "confidence", "comments"]].eq("").all().all()


def test_holdout_membership_and_frozen_op_hashes_verify():
    manifest = json.loads((OUT/"relevance_v8_holdout_freeze_manifest_v1.json").read_text())
    assert manifest["membership_frozen"] is True
    assert manifest["operating_point_version"] == "v8_op1"
    for raw, expected in manifest["files"].items():
        assert hashlib.sha256(Path(raw).read_bytes()).hexdigest() == expected


def test_normal_holdout_scoring_is_blocked_before_gold():
    holdout = pd.read_csv(OUT/"relevance_v8_blind_holdout_v1.csv")
    with pytest.raises(RuntimeError, match="scoring is blocked"):
        assert_holdout_scoring_allowed(set(holdout.source_record_id), ROOT)
    with pytest.raises(RuntimeError, match="scoring is blocked"):
        assert_holdout_scoring_allowed(set(holdout.source_record_id), ROOT,
                                       dedicated_evaluation=True)


def test_predeclared_metrics_and_confirmatory_rule():
    config = json.loads((OUT/"relevance_v8_holdout_predeclared_metrics_v1.json").read_text())
    assert config["success_criteria"] == {"inclusion_precision_min": .85,
        "inclusion_recall_min": .90, "binary_macro_f1_min": .80}
    assert "confirmatory evaluation" in config["confirmatory_rule"]
    state = json.loads((OUT/"holdout_temporal_state_v1.json").read_text())
    assert state["gold_finalized_at"] is None
    assert state["classifier_first_scored_at"] is None


def test_no_holdout_model_outputs_or_api_activity_exist():
    forbidden = list(OUT.glob("*prediction*")) + list(OUT.glob("*metrics_result*")) + list(OUT.glob("*error_analysis*"))
    assert forbidden == []
    report = json.loads((OUT/"relevance_v8_holdout_sampling_report_v1.json").read_text())
    assert report["api_calls"] == 0
    assert report["model_fields_used_for_eligibility_or_sampling"] == []
