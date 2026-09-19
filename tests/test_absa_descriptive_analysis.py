import inspect
import json

import numpy as np
import pandas as pd
import pytest

from marathon_absa.absa_descriptive_analysis import (
    DOCUMENTS_PATH, FINALIZATION_MANIFEST, LIMITATION, MENTIONS_PATH, ROOT,
    SENTIMENTS, SUPPORT_THRESHOLDS, YEARS, _sha, create_descriptive_analysis,
)


def test_analysis_reconciles_frozen_document_and_mention_populations():
    summary=json.loads((ROOT/"absa_v1_corpus_summary_v1.json").read_text(encoding="utf-8"))
    assert summary["total_documents"]==7704
    assert summary["mention_bearing_documents"]==5316
    assert summary["zero_mention_documents"]==2388
    assert summary["total_mentions"]==15486
    assert summary["document_population_denominator"]==7704
    assert summary["mention_population_denominator"]==15486
    assert summary["api_calls"]==0


def test_all_frozen_aspects_and_sentiments_are_retained_with_correct_denominators():
    prevalence=pd.read_csv(ROOT/"absa_v1_aspect_prevalence_v1.csv")
    aspect_sentiment=pd.read_csv(ROOT/"absa_v1_aspect_sentiment_v1.csv")
    assert len(prevalence)==20 and prevalence.aspect.nunique()==20
    assert prevalence.mention_count.sum()==15486
    assert prevalence.mention_share_all_mentions.sum()==pytest.approx(1)
    assert len(aspect_sentiment)==20*len(SENTIMENTS)
    assert set(aspect_sentiment.sentiment)==set(SENTIMENTS)
    shares=aspect_sentiment.groupby("aspect").within_aspect_sentiment_share.sum()
    assert np.allclose(shares.to_numpy(),1.0)
    assert aspect_sentiment.mention_count.sum()==15486


def test_year_topic_and_language_tables_include_raw_and_normalized_values():
    year=pd.read_csv(ROOT/"absa_v1_year_aspect_sentiment_v1.csv")
    topic=pd.read_csv(ROOT/"absa_v1_topic_aspect_sentiment_v1.csv")
    language=pd.read_csv(ROOT/"absa_v1_language_aspect_sentiment_v1.csv")
    assert set(year.event_year)==set(YEARS) and len(year)==len(YEARS)*20*len(SENTIMENTS)
    assert year.mention_count.sum()==15486
    assert {"aspect_document_prevalence","aspect_mention_share_within_year","within_year_aspect_sentiment_share"}<=set(year)
    assert topic.mention_count.sum()==15486
    assert {"document_count","aspect_affected_document_count","within_topic_aspect_sentiment_share"}<=set(topic)
    assert language.mention_count.sum()==15486
    assert {"zero_mention_rate","aspect_document_prevalence_within_language","within_language_aspect_sentiment_share"}<=set(language)


def test_support_flags_are_predeclared_and_do_not_remove_sparse_aspects():
    support=pd.read_csv(ROOT/"absa_v1_support_flags_v1.csv")
    assert len(support)==20 and support.aspect.nunique()==20
    assert SUPPORT_THRESHOLDS["very_low_support"]["maximum_affected_documents"]==29
    assert SUPPORT_THRESHOLDS["high_support"]["minimum_affected_documents"]==500
    assert set(support.support_flag)<={"very_low_support","low_support","moderate_support","high_support"}
    assert support.loc[support.aspect.eq("event_information"),"support_flag"].item()=="very_low_support"


def test_manifest_hashes_sources_and_analysis_is_strictly_offline():
    manifest=json.loads((ROOT/"analysis_manifest_v1.json").read_text(encoding="utf-8"))
    assert manifest["source_artifacts"][DOCUMENTS_PATH.name]["sha256"]==_sha(DOCUMENTS_PATH)
    assert manifest["source_artifacts"][MENTIONS_PATH.name]["sha256"]==_sha(MENTIONS_PATH)
    assert manifest["source_artifacts"][FINALIZATION_MANIFEST.name]["sha256"]==_sha(FINALIZATION_MANIFEST)
    assert manifest["api_calls"]==0 and manifest["inferential_testing_performed"] is False
    source=inspect.getsource(create_descriptive_analysis)
    assert "OpenAI" not in source and "huggingface" not in source.lower()
    with pytest.raises(FileExistsError):create_descriptive_analysis()


def test_report_preserves_limitation_and_separates_mentions_from_documents():
    report=(ROOT/"ABSA_V1_DESCRIPTIVE_ANALYSIS_REPORT.md").read_text(encoding="utf-8")
    assert LIMITATION in report
    assert "must not be interpreted as independent posts" in report
    assert "No inferential significance testing was performed" in report
    assert "API calls: 0" in report
