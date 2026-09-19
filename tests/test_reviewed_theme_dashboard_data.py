from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from marathon_absa.reviewed_theme_dashboard_data import (
    INSUFFICIENT_ASPECTS, REVIEW_ROOT, load_reviewed_theme_dashboard_data,
)


def _hashes():
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in REVIEW_ROOT.iterdir() if path.is_file()}


def test_finalized_reviewed_theme_loader_and_frozen_integrity():
    before = _hashes()
    data = load_reviewed_theme_dashboard_data()
    assert _hashes() == before
    assert data.manifest["lifecycle_status"] == "reviewed_taxonomy_finalized"
    assert data.manifest["source_clusters"] == data.manifest["number_reviewed"] == 101
    assert data.manifest["final_reviewed_theme_count"] == len(data.summary) == len(data.taxonomy) == 64
    assert data.summary.aspect.nunique() == 16


def test_pending_or_provisional_package_is_rejected_before_data_loading(tmp_path):
    (tmp_path / "review_manifest.json").write_text(json.dumps({"lifecycle_status": "review_package_generated_pending_researcher_review", "output_hashes": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="not finalized"):
        load_reviewed_theme_dashboard_data(tmp_path)


def test_reviewed_theme_identity_and_zero_cross_aspect_leakage():
    data = load_reviewed_theme_dashboard_data()
    assert data.summary.reviewed_theme_id.is_unique and data.taxonomy.reviewed_theme_id.is_unique
    identity = data.summary.set_index("reviewed_theme_id").aspect
    for frame in [data.taxonomy, data.sentiment, data.years, data.evidence]:
        assert frame.apply(lambda row: identity[row.reviewed_theme_id] == row.aspect, axis=1).all()
    assert data.summary.review_status.eq("reviewed").all()
    assert data.summary.reviewed_theme_label.str.strip().ne("").all()


def test_unique_document_support_and_sentiment_join_are_frozen():
    data = load_reviewed_theme_dashboard_data()
    sentiment = data.sentiment.groupby("reviewed_theme_id").agg(documents=("support_documents", "sum"), share=("share_of_theme_documents", "sum"))
    expected = data.summary.set_index("reviewed_theme_id").support_documents
    assert sentiment.documents.reindex(expected.index).equals(expected)
    assert np.allclose(sentiment.share, 1)
    for name in ["positive", "negative", "mixed", "neutral"]:
        frozen = data.summary.set_index("reviewed_theme_id")[f"{name}_documents"]
        joined = data.sentiment[data.sentiment.document_sentiment.eq(name)].set_index("reviewed_theme_id").support_documents
        assert joined.reindex(frozen.index).equals(frozen)


def test_year_join_and_frozen_prevalence_values():
    data = load_reviewed_theme_dashboard_data()
    assert len(data.years) == 64 * 4 and set(data.years.year.astype(int)) == {2019, 2023, 2024, 2025}
    assert data.years.groupby("reviewed_theme_id").size().eq(4).all()
    denominator = data.years.aspect_documents_that_year.replace(0, np.nan)
    expected = (data.years.support_documents / denominator).fillna(0)
    assert np.allclose(data.years.within_aspect_theme_prevalence, expected)


def test_representative_evidence_lineage_and_multilingual_preservation():
    data = load_reviewed_theme_dashboard_data()
    assert len(data.evidence) == 64 * 5
    assert data.evidence.groupby("reviewed_theme_id").size().eq(5).all()
    assert data.evidence.mention_id.is_unique
    assignment = pd.read_parquet(REVIEW_ROOT / "reviewed_theme_assignments.parquet", columns=["mention_id", "document_id", "aspect", "reviewed_theme_id", "evidence_text", "english_gloss"])
    joined = data.evidence.merge(assignment, on="mention_id", suffixes=("_shown", "_frozen"), validate="one_to_one")
    for field in ["document_id", "aspect", "reviewed_theme_id", "evidence_text", "english_gloss"]:
        assert joined[f"{field}_shown"].fillna("").equals(joined[f"{field}_frozen"].fillna(""))
    assert data.evidence.evidence_text.str.contains("[^\x00-\x7F]", regex=True).any()


def test_insufficient_support_is_explicit_and_has_no_reviewed_themes():
    data = load_reviewed_theme_dashboard_data()
    assert set(data.insufficient_support.aspect) == INSUFFICIENT_ASPECTS
    assert set(data.summary.aspect).isdisjoint(INSUFFICIENT_ASPECTS)
    assert data.insufficient_support.support_mentions.sum() == 154


def test_loader_contains_validation_only_and_no_research_inference_path():
    import marathon_absa.reviewed_theme_dashboard_data as module
    source = inspect.getsource(module).lower()
    for forbidden in ["openai", "sentence_transformer", "hdbscan", "umap(", "chi2_contingency", "multipletests", "requests.", "urlopen("]:
        assert forbidden not in source
