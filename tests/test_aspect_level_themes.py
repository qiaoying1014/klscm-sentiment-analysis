from __future__ import annotations

import hashlib
import inspect

import numpy as np
import pandas as pd

from marathon_absa.aspect_level_themes import (
    DOCUMENTS_PATH, EXPECTED_MENTIONS, MENTIONS_PATH, adaptive_cluster_policy,
    build_outputs, construct_theme_semantic_text, load_and_reconcile,
    _write_frame,
)
import marathon_absa.aspect_level_themes as alta


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_reconciliation_and_lineage_are_exact():
    mentions, documents, _, aspects = load_and_reconcile()
    assert len(mentions) == mentions.mention_id.nunique() == EXPECTED_MENTIONS
    assert len(documents) == documents.document_id.nunique() == 7704
    assert mentions.document_id.nunique() == 5316
    assert documents.mention_count.astype(int).eq(0).sum() == 2388
    assert len(aspects) == len(set(aspects)) == 20
    assert set(mentions.aspect) == set(aspects)
    assert set(mentions.event_year.astype(int)) == {2019, 2023, 2024, 2025}
    assert set(mentions.document_id).issubset(set(documents.document_id))


def test_semantic_text_is_deterministic_multilingual_and_non_destructive():
    row = {"target": "  cuaca panas ", "evidence_text": "很热 🔥  #KLSCM", "english_gloss": "very hot"}
    expected = "cuaca panas | 很热 🔥 #KLSCM | very hot"
    assert construct_theme_semantic_text(row) == expected
    assert construct_theme_semantic_text(row) == expected
    assert row["evidence_text"] == "很热 🔥  #KLSCM"


def test_support_and_noise_policy_are_explicit():
    assert not adaptive_cluster_policy(59, 100)["sufficient_support"]
    assert not adaptive_cluster_policy(100, 39)["sufficient_support"]
    policy = adaptive_cluster_policy(100, 70)
    assert policy["sufficient_support"] and policy["min_cluster_size"] == 10


def test_document_aggregation_deduplicates_mentions_and_preserves_evidence():
    rows = []
    for mention_id, document_id, sentiment in [("m1", "d1", "positive"), ("m2", "d1", "negative"), ("m3", "d2", "positive")]:
        rows.append({"mention_id": mention_id, "document_id": document_id, "aspect": "route_course", "theme_id": 0,
                     "target": "hill", "evidence_text": f"real {mention_id}", "english_gloss": "",
                     "sentiment": sentiment, "event_year": 2025, "primary_language": "English",
                     "theme_semantic_text": "hill route", "theme_status": "stable_cluster"})
    assignments = pd.DataFrame(rows)
    documents = pd.DataFrame({"document_id": ["d1", "d2"]})
    outputs = build_outputs(assignments, documents)
    summary = outputs["theme_summary"].iloc[0]
    assert summary.support_mentions == 3 and summary.support_documents == 2
    assert summary.mixed_document_presence == 1 and summary.positive_document_presence == 1
    evidence = outputs["theme_representative_evidence"]
    assert set(evidence.evidence_text).issubset(set(assignments.evidence_text))
    assert evidence.document_id.nunique() == len(evidence)


def test_loading_never_writes_frozen_artifacts():
    before = {_sha(MENTIONS_PATH), _sha(DOCUMENTS_PATH)}
    load_and_reconcile()
    after = {_sha(MENTIONS_PATH), _sha(DOCUMENTS_PATH)}
    assert before == after


def test_assignment_invariants_with_synthetic_vectors():
    source = pd.DataFrame({"mention_id": ["m1", "m2"], "document_id": ["d1", "d2"],
                           "aspect": ["facilities", "route_course"], "sentiment": ["neutral", "positive"],
                           "evidence_text": ["厕所", "bukit"], "target": ["toilet", "hill"],
                           "english_gloss": ["toilet", "hill"]})
    derived = source.copy()
    derived["theme_id"] = [-1, -1]
    assert derived.mention_id.is_unique
    assert derived.aspect.equals(source.aspect)
    assert derived.sentiment.equals(source.sentiment)
    assert derived.evidence_text.equals(source.evidence_text)
    assert np.array_equal(derived.theme_id.to_numpy(), [-1, -1])


def test_output_hash_generation_is_reproducible(tmp_path):
    frame = pd.DataFrame({"mention_id": ["m1"], "evidence_text": ["真实 evidence 🔥"]})
    hashes = _write_frame(frame, tmp_path / "artifact")
    assert hashes["artifact.csv"] == _sha(tmp_path / "artifact.csv")
    assert hashes["artifact.parquet"] == _sha(tmp_path / "artifact.parquet")
    assert pd.read_parquet(tmp_path / "artifact.parquet").equals(frame)


def test_alta_module_has_no_openai_or_network_execution_path():
    source = inspect.getsource(alta)
    assert "import openai" not in source.casefold() and "from openai" not in source.casefold()
    assert "requests." not in source and "urlopen(" not in source
    assert "local_files_only=True" in source
