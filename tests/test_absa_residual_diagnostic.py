import json

import pandas as pd

from marathon_absa.absa_residual_diagnostic import (
    CANDIDATE_PATH, CATEGORIES, DEVELOPMENT_ROOT, GOLD_PATH, OUTPUT_JSON,
    OUTPUT_MD, _false_negatives, _load, _statuses,
)


def test_residual_diagnostic_reconciles_frozen_counts_and_reviews_all_fps():
    baseline, candidate, gold, sample = _load()
    baseline_status = _statuses(baseline, gold)
    candidate_status = _statuses(candidate, gold)
    assert (baseline_status.evaluation_status.eq("TP").sum(),
            baseline_status.evaluation_status.eq("FP").sum(),
            len(_false_negatives(baseline, gold))) == (84, 134, 16)
    assert (candidate_status.evaluation_status.eq("TP").sum(),
            candidate_status.evaluation_status.eq("FP").sum(),
            len(_false_negatives(candidate, gold))) == (77, 74, 23)
    review = pd.read_csv(DEVELOPMENT_ROOT / "candidate_residual_fp_review_all_74_v1.csv").fillna("")
    assert len(review) == len(CATEGORIES) == 74
    assert review.reviewed.astype(str).str.lower().eq("true").all()
    assert review.candidate_fp_category.ne("").all()
    assert review.researcher_notes.ne("").all()
    assert set(review.confidence) <= {"high", "medium", "low"}


def test_residual_report_is_offline_development_evidence_with_frozen_hashes():
    report = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
    assert OUTPUT_MD.exists()
    assert report["development_evidence"] is True
    assert report["confirmatory_evidence"] is False
    assert report["api_calls"] == 0 and report["inference_run"] is False
    assert report["prompt_modified"] is False
    assert report["fp_reduction"] == {
        "baseline": 134, "candidate": 74, "absolute": 60,
        "percentage": 60 / 134,
    }
    assert report["recall"]["candidate_fn_total"] == 23
    assert report["recall"]["already_fn_under_baseline"] == 11
    assert report["recall"]["new_fn_created_by_candidate"] == 12
    assert report["gold_zero"]["candidate_gold_zero_predicted_mentions_documents"] == 6
    assert report["multi_aspect"]["candidate_predicted_multi_aspect_documents"] == 35
    assert report["input_hashes"]["candidate_predictions_sha256"]
    assert report["input_hashes"]["gold_sha256"]
