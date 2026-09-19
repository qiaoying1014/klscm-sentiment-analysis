import inspect
import json

import numpy as np
import pandas as pd

from marathon_absa.absa_dashboard_data import (
    FROZEN_ELIGIBLE, INFERENTIAL, ROOT, _effect, create_dashboard_data,
)


def test_corpus_and_sentiment_totals_reconcile():
    corpus = json.loads((ROOT / "dashboard_corpus_overview_v1.json").read_text(encoding="utf-8"))
    sentiment = pd.read_csv(ROOT / "dashboard_sentiment_overall_v1.csv")
    assert corpus["total_documents"] == 7704
    assert corpus["mention_bearing_documents"] == 5316
    assert corpus["zero_mention_documents"] == 2388
    assert corpus["total_mentions"] == 15486
    assert sentiment.mention_count.sum() == 15486
    assert np.isclose(sentiment.mention_share.sum(), 1)


def test_aspect_and_year_dimensions_and_document_denominators():
    aspects = pd.read_csv(ROOT / "dashboard_aspect_overview_v1.csv")
    years = pd.read_csv(ROOT / "dashboard_year_aspect_v1.csv")
    assert len(aspects) == aspects.aspect.nunique() == 20
    assert len(years) == 80 and set(years.year) == {2019, 2023, 2024, 2025}
    expected = {2019: 2279, 2023: 1873, 2024: 1724, 2025: 1828}
    assert years.groupby("year").total_documents_year.first().to_dict() == expected
    assert np.allclose(years.document_prevalence, years.affected_documents / years.total_documents_year)
    assert (years.affected_documents <= years.total_documents_year).all()


def test_aspect_sentiment_uses_aspect_mentions_as_denominator():
    sentiment = pd.read_csv(ROOT / "dashboard_aspect_sentiment_v1.csv")
    overview = pd.read_csv(ROOT / "dashboard_aspect_overview_v1.csv").set_index("aspect")
    assert sentiment.mention_count.sum() == 15486
    assert np.allclose(sentiment.groupby("aspect").share_within_aspect.sum(), 1)
    for aspect, group in sentiment.groupby("aspect"):
        assert group.mention_count.sum() == overview.loc[aspect, "mention_count"]
        assert np.allclose(group.share_within_aspect, group.mention_count / group.mention_count.sum())


def test_support_effect_and_eligibility_are_frozen():
    aspects = pd.read_csv(ROOT / "dashboard_aspect_overview_v1.csv")
    assert aspects.loc[aspects.affected_document_count.lt(30), "support_class"].eq("very_low_support").all()
    assert aspects.loc[aspects.affected_document_count.between(30, 99), "support_class"].eq("low_support").all()
    assert aspects.loc[aspects.affected_document_count.between(100, 499), "support_class"].eq("moderate_support").all()
    assert aspects.loc[aspects.affected_document_count.ge(500), "support_class"].eq("high_support").all()
    assert aspects.loc[aspects.inferential_eligible, "aspect"].tolist() == FROZEN_ELIGIBLE
    for row in aspects.itertuples(): assert row.effect_category == _effect(row.cramers_v)


def test_emphasis_policy_has_required_safeguards():
    aspects = pd.read_csv(ROOT / "dashboard_aspect_overview_v1.csv").set_index("aspect")
    expected = {"race_performance", "crowd_community_atmosphere", "physical_experience"}
    assert set(aspects[aspects.recommended_for_emphasis].index) == expected
    calculated = (aspects.inferential_eligible & aspects.year_association_fdr_significant.eq(True) &
                  aspects.cramers_v.ge(.10) & aspects.support_class.isin(["moderate_support", "high_support"]))
    assert calculated.equals(aspects.recommended_for_emphasis)
    assert not aspects.loc["photography_media", "recommended_for_emphasis"]
    assert not aspects[aspects.support_class.isin(["low_support", "very_low_support"])].recommended_for_emphasis.any()


def test_cautions_disable_weather_topic_and_language_inference():
    cautions = pd.read_csv(ROOT / "dashboard_caution_findings_v1.csv")
    metadata = json.loads((ROOT / "dashboard_metadata_v1.json").read_text(encoding="utf-8"))
    language = pd.read_csv(ROOT / "dashboard_language_overview_v1.csv")
    assert "weather_conditions" in set(cautions.item)
    assert metadata["topic_inference_allowed"] is False and metadata["language_inference_allowed"] is False
    assert not language.inferential_comparison_allowed.any()
    assert language.set_index("language").loc[["insufficient_text", "undetermined"], "language_quality_warning"].all()


def test_frozen_confidence_intervals_and_statistics_are_passed_through():
    mart = pd.read_csv(ROOT / "dashboard_year_aspect_v1.csv")
    frozen = pd.read_csv(INFERENTIAL / "absa_v1_inferential_aspect_prevalence_v1.csv")
    joined = mart.merge(frozen, on=["aspect", "year"], suffixes=("_mart", "_frozen"))
    assert np.allclose(joined.document_prevalence_ci_lower, joined.ci_lower)
    assert np.allclose(joined.document_prevalence_ci_upper, joined.ci_upper)
    assert np.allclose(joined.omnibus_adjusted_p, joined.adjusted_p)
    assert np.allclose(joined.cramers_v, joined.effect_size)
    pair = pd.read_csv(ROOT / "dashboard_pairwise_year_v1.csv")
    original = pd.read_csv(INFERENTIAL / "absa_v1_inferential_pairwise_year_v1.csv")
    assert np.allclose(pair.raw_p, original.raw_p)
    assert np.allclose(pair.holm_adjusted_p, original.adjusted_p)
    assert pair.significant_holm.equals(original.significant_holm)


def test_topic_language_and_year_marts_reconcile():
    topics = pd.read_csv(ROOT / "dashboard_topic_overview_v1.csv")
    languages = pd.read_csv(ROOT / "dashboard_language_overview_v1.csv")
    years = pd.read_csv(ROOT / "dashboard_year_overview_v1.csv")
    assert topics.document_count.sum() == 7704
    assert languages.documents.sum() == 7704
    assert years.documents.sum() == 7704 and years.mentions.sum() == 15486
    assert set(["positive_share", "negative_share", "mixed_share", "neutral_share"]) <= set(topics)


def test_manifest_hashes_sources_and_pipeline_contains_no_inference_or_tests():
    manifest = json.loads((ROOT / "dashboard_data_manifest_v1.json").read_text(encoding="utf-8"))
    assert all(item["sha256"] for item in manifest["source_artifacts"].values())
    execution = manifest["execution"]
    assert execution["openai_calls"] == execution["hugging_face_inference"] == execution["absa_inference"] == 0
    assert execution["manual_review"] == execution["new_statistical_hypothesis_tests"] == 0
    source = inspect.getsource(create_dashboard_data).lower()
    for forbidden in ["chi2_contingency", "adjust_pvalues", "wilson_interval", "openai_service", "huggingface"]:
        assert forbidden not in source


def test_dashboard_marts_regenerate_deterministically(tmp_path):
    metadata = json.loads((ROOT / "dashboard_metadata_v1.json").read_text(encoding="utf-8"))
    regenerated = tmp_path / "dashboard"
    create_dashboard_data(regenerated, created_at=metadata["created_at"])
    for name in ["dashboard_aspect_overview_v1.csv", "dashboard_year_aspect_v1.csv",
                 "dashboard_pairwise_year_v1.csv", "dashboard_metadata_v1.json",
                 "dashboard_research_findings_v1.csv"]:
        assert (ROOT / name).read_bytes() == (regenerated / name).read_bytes()
