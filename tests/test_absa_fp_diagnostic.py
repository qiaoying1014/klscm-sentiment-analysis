from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from marathon_absa.absa_fp_diagnostic import (
    EVALUATION_PATH, FP_CATEGORIES, GOLD_PATH, PREDICTIONS_PATH, SAMPLE_PATH, SEED,
    create_fp_diagnostic, finalize_fp_diagnostic, reconstruct_matching,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_matching_and_development_package_reconcile_without_source_writes(tmp_path):
    source_hashes = {path: digest(path) for path in [SAMPLE_PATH, GOLD_PATH, PREDICTIONS_PATH, EVALUATION_PATH]}
    gold = pd.read_csv(GOLD_PATH).fillna(""); predictions = pd.read_csv(PREDICTIONS_PATH).fillna("")
    tp, fp, fn = reconstruct_matching(gold, predictions)
    assert (len(tp), len(fp), len(fn)) == (84, 134, 16)
    one = tmp_path / "one"; two = tmp_path / "two"
    manifest_one = create_fp_diagnostic(one); manifest_two = create_fp_diagnostic(two)
    fp_table = pd.read_csv(one / "absa_v1_fp_diagnostic_mentions_v1.csv")
    review_one = pd.read_csv(one / "absa_v1_fp_manual_review_sample_v1.csv")
    review_two = pd.read_csv(two / "absa_v1_fp_manual_review_sample_v1.csv")
    tp_review = pd.read_csv(one / "absa_v1_tp_contrast_sample_v1.csv")
    fn_review = pd.read_csv(one / "absa_v1_fn_contrast_sample_v1.csv")
    summary = json.loads((one / "absa_v1_fp_diagnostic_summary_v1.json").read_text(encoding="utf-8"))
    assert len(fp_table) == 134
    assert summary["baseline_reconciliation"] == {"tp":84, "fp":134, "fn":16,
        "gold_mentions":100, "gold_mention_documents":56, "gold_zero_documents":24,
        "predicted_mentions":218, "predicted_mention_documents":65, "predicted_zero_documents":15}
    assert summary["gold_zero_behavior"]["matrix"] == {"gold_zero_predicted_zero":13,
        "gold_zero_predicted_mentions":11, "gold_mentions_predicted_zero":2,
        "gold_mentions_predicted_mentions":54}
    assert summary["over_extraction"]["gold_multi_aspect_documents"] == 27
    assert summary["over_extraction"]["predicted_multi_aspect_documents"] == 43
    assert len(review_one) == 45 and review_one.review_row_id.nunique() == 45
    assert review_one.document_id.value_counts().max() <= 2
    assert review_one.predicted_aspect.nunique() >= 8 and review_one.language.nunique() >= 4
    assert review_one.final_topic_label.nunique() >= 10
    assert len(tp_review) == 15 and len(fn_review) == 16
    pd.testing.assert_frame_equal(review_one, review_two)
    assert manifest_one["manual_sample_seed"] == manifest_two["manual_sample_seed"] == SEED
    assert all(digest(path) == value for path, value in source_hashes.items())


def test_manual_fp_finalizer_requires_complete_controlled_categories(tmp_path):
    package = tmp_path / "package"
    create_fp_diagnostic(package)
    review_path = package / "absa_v1_fp_manual_review_sample_v1.csv"
    review = pd.read_csv(review_path).fillna("")
    with pytest.raises(ValueError, match="reviewed=true"):
        finalize_fp_diagnostic(package)
    review["reviewed"] = True; review["fp_error_category"] = "not_a_category"
    review.to_csv(review_path, index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="Unknown"):
        finalize_fp_diagnostic(package)
    review["fp_error_category"] = FP_CATEGORIES[0]
    review.to_csv(review_path, index=False, encoding="utf-8-sig")
    result = finalize_fp_diagnostic(package)
    assert result["reviewed_fp_rows"] == 45
    assert result["dominant_error_mechanisms"] == [FP_CATEGORIES[0]]
    assert not result["prompt_generated"] and result["api_calls"] == 0
