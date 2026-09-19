import inspect
import json

import numpy as np
import pandas as pd

from marathon_absa.absa_inferential_analysis import (
    DESCRIPTIVE_MANIFEST, DOCUMENTS_PATH, FINALIZATION_MANIFEST, MENTIONS_PATH,
    ROOT, SUPPORT_PATH, _sha, adjust_pvalues, build_document_matrix,
    create_inferential_analysis, wilson_interval,
)


def test_document_level_matrix_and_population_reconcile():
    matrix = pd.read_csv(ROOT / "absa_v1_document_analysis_matrix_v1.csv", dtype={"document_id": str})
    assert matrix.document_id.nunique() == 7704
    assert len(matrix) == 7704 * 20
    assert matrix.groupby("document_id").size().eq(20).all()
    assert int(matrix.groupby("document_id").aspect_present.max().sum()) == 5316
    assert int(matrix.aspect_present.sum()) < 15486


def test_years_and_zero_mention_documents_are_preserved():
    matrix = pd.read_csv(ROOT / "absa_v1_document_analysis_matrix_v1.csv")
    counts = matrix.drop_duplicates("document_id").event_year.value_counts().to_dict()
    assert counts == {2019: 2279, 2023: 1873, 2025: 1828, 2024: 1724}
    zero = matrix.groupby("document_id").aspect_present.sum().eq(0).sum()
    assert zero == 2388


def test_eligibility_is_support_based_and_frozen_before_testing():
    eligibility = pd.read_csv(ROOT / "absa_v1_inferential_eligibility_v1.csv")
    eligible = eligibility[eligibility.primary_inference_eligible]
    excluded = eligibility[~eligibility.primary_inference_eligible]
    assert set(eligible.support_flag) <= {"moderate_support", "high_support"}
    assert eligible.minimum_year_affected_documents.ge(20).all()
    assert not excluded[excluded.support_flag.isin(["low_support", "very_low_support"])].primary_inference_eligible.any()
    manifest = json.loads((ROOT / "absa_v1_inferential_analysis_manifest_v1.json").read_text(encoding="utf-8"))
    assert manifest["eligible_aspects_frozen_before_testing"] == eligible.aspect.tolist()
    source = inspect.getsource(create_inferential_analysis)
    assert source.index("eligible = []") < source.index("_omnibus(aspect_all")


def test_bh_and_holm_corrections_are_correct_and_separate():
    assert adjust_pvalues([.01, .02, .20], "bh") == [0.03, 0.03, 0.20000000000000004]
    assert adjust_pvalues([.01, .02, .20], "holm") == [0.03, 0.04, 0.2]
    for filename, family in [("absa_v1_inferential_aspect_prevalence_v1.csv", "A_aspect_prevalence"),
                             ("absa_v1_inferential_positive_year_v1.csv", "B_positive_sentiment"),
                             ("absa_v1_inferential_negative_year_v1.csv", "C_negative_sentiment")]:
        frame = pd.read_csv(ROOT / filename)
        assert set(frame.analysis_family) == {family}
        assert {"raw_p", "adjusted_p", "significant_fdr"} <= set(frame)


def test_pairwise_only_follows_qualifying_omnibus_and_uses_holm():
    pairs = pd.read_csv(ROOT / "absa_v1_inferential_pairwise_year_v1.csv")
    assert {"raw_p", "adjusted_p", "significant_holm", "difference", "ci_lower", "ci_upper"} <= set(pairs)
    sources = {
        "A_aspect_prevalence": pd.read_csv(ROOT / "absa_v1_inferential_aspect_prevalence_v1.csv"),
        "B_positive_sentiment": pd.read_csv(ROOT / "absa_v1_inferential_positive_year_v1.csv"),
        "C_negative_sentiment": pd.read_csv(ROOT / "absa_v1_inferential_negative_year_v1.csv"),
    }
    for row in pairs[["aspect", "analysis_family"]].drop_duplicates().itertuples():
        source = sources[row.analysis_family]
        assert source[source.aspect.eq(row.aspect)].significant_fdr.all()
        assert len(pairs[(pairs.aspect.eq(row.aspect)) & (pairs.analysis_family.eq(row.analysis_family))]) == 6


def test_effect_sizes_intervals_and_models_are_generated():
    prevalence = pd.read_csv(ROOT / "absa_v1_inferential_aspect_prevalence_v1.csv")
    models = pd.read_csv(ROOT / "absa_v1_inferential_aspect_year_models_v1.csv")
    assert prevalence.effect_size.notna().all()
    assert ((prevalence.ci_lower <= prevalence.prevalence) & (prevalence.prevalence <= prevalence.ci_upper)).all()
    assert {"odds_ratio", "ci_lower", "ci_upper", "reference_year", "comparison_year"} <= set(models)
    assert models.reference_year.eq(2019).all()
    assert wilson_interval(0, 100)[0] == 0


def test_sources_unchanged_and_execution_is_offline():
    manifest = json.loads((ROOT / "absa_v1_inferential_analysis_manifest_v1.json").read_text(encoding="utf-8"))
    for path in [DOCUMENTS_PATH, MENTIONS_PATH, FINALIZATION_MANIFEST, SUPPORT_PATH, DESCRIPTIVE_MANIFEST]:
        assert manifest["source_artifacts"][str(path)]["sha256"] == _sha(path)
    assert manifest["openai_calls"] == manifest["hugging_face_inference"] == manifest["absa_inference"] == 0
    assert manifest["manual_review"] == 0 and manifest["production_predictions_modified"] is False
    source = inspect.getsource(create_inferential_analysis).lower()
    assert "openai_service" not in source and "huggingface" not in source
