import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from marathon_absa.blog_analysis import (
    DECISIONS, EXPECTED_PROMPT_SHA256, INSTRUCTIONS, STAGE,
    _review_sentiment, aggregate_blog_absa, audit_blog_source,
    create_blog_absa_requests, finalize_theme_mappings, ground_blog_evidence,
    included_reviews, verify_frozen_upstream,
)


def test_authoritative_blog_audit_and_stable_lineage(tmp_path):
    result = audit_blog_source(tmp_path)
    assert result["raw_review_count"] == 28
    assert result["usable_review_count"] == 25
    assert result["duplicate_review_count"] == 3
    assert result["chunk_count"] == 56
    assert result["reviews_with_multiple_chunks"] == 10
    assert result["min_chunks_per_review"] == 1
    assert result["max_chunks_per_review"] == 6
    assert all(x["exclusion_reason"] == "exact_duplicate" for x in result["excluded_records"])
    assert all(value == "DOCUMENTATION_REQUIRED" for value in result["provenance"].values())
    reviews, chunks = included_reviews()
    assert reviews.document_id.nunique() == 25
    assert chunks.unit_id.nunique() == 56
    assert set(chunks.document_id) == set(reviews.document_id)


def test_request_package_reuses_frozen_v3_and_isolated_identity(tmp_path, monkeypatch):
    from marathon_absa import blog_analysis
    monkeypatch.setattr(blog_analysis, "REQUESTS", tmp_path / "requests.jsonl")
    monkeypatch.setattr(blog_analysis, "REQUEST_MANIFEST", tmp_path / "request_manifest.json")
    result = create_blog_absa_requests(tmp_path)
    assert result["stage"] == STAGE == "absa_v1_blog_mentions_v1"
    assert result["prompt_sha256"] == EXPECTED_PROMPT_SHA256
    assert hashlib.sha256(INSTRUCTIONS.encode("utf-8")).hexdigest() == EXPECTED_PROMPT_SHA256
    assert result["request_count"] == 56 and result["review_count"] == 25
    assert result["submitted"] is False and result["api_calls"] == 0


def test_frozen_upstream_integrity_quantities_and_hashes(tmp_path):
    result = verify_frozen_upstream(tmp_path)
    assert result["all_passed"] and all(result["checks"].values())


def test_review_level_aggregation_deduplicates_chunks_and_preserves_mentions(tmp_path):
    reviews = pd.DataFrame({"document_id": ["r1", "r2"], "event_year": [2024, 2024], "primary_language": ["English", "English"]})
    chunks = pd.DataFrame({"document_id": ["r1", "r1", "r2"], "unit_id": ["c1", "c2", "c3"]})
    mentions = pd.DataFrame([
        {"review_id": "r1", "chunk_id": "c1", "blog_mention_id": "m1", "aspect": "route_course", "sentiment": "positive"},
        {"review_id": "r1", "chunk_id": "c2", "blog_mention_id": "m2", "aspect": "route_course", "sentiment": "negative"},
        {"review_id": "r2", "chunk_id": "c3", "blog_mention_id": "m3", "aspect": "route_course", "sentiment": "positive"},
    ])
    aggregate_blog_absa(mentions, reviews, chunks, tmp_path)
    presence = pd.read_csv(tmp_path / "blog_review_aspect_presence_v1.csv")
    aspect = pd.read_csv(tmp_path / "blog_aspect_summary_v1.csv")
    assert len(presence) == 2
    assert presence.set_index("review_id").loc["r1", "review_aspect_sentiment"] == "mixed"
    row = aspect.iloc[0]
    assert row.support_reviews == 2 and row.total_included_reviews == 2 and row.review_prevalence == 1.0
    assert row.support_mentions == 3 and row.denominator_unit == "included_parent_reviews"


def test_conflicting_sentiments_become_mixed():
    assert _review_sentiment(pd.Series(["positive", "negative"])) == "mixed"
    assert _review_sentiment(pd.Series(["positive", "positive"])) == "positive"


def test_theme_decision_vocabulary_has_no_automatic_accept():
    assert DECISIONS == {"MATCH_EXISTING", "EMERGENT_BLOG_THEME", "UNCLEAR", "EXCLUDE_FROM_THEME_COMPARISON"}


def test_theme_finalization_fails_while_researcher_rows_unresolved(tmp_path):
    path = tmp_path / "mapping.csv"
    pd.DataFrame([{"review_status": "pending", "mapping_decision": "", "selected_reviewed_theme_id": "",
                   "emergent_blog_theme_label": "", "aspect": "route_course"}]).to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="Unresolved"):
        finalize_theme_mappings(path, tmp_path)


def test_cross_source_module_declares_separate_denominators_and_no_inference():
    source = Path("marathon_absa/cross_source_analysis.py").read_text(encoding="utf-8")
    assert '"pooled_prevalence": False' in source
    assert '"inferential_tests": 0' in source
    assert "instagram_total_documents" in source and "blog_total_reviews" in source


def test_blog_grounding_preserves_exact_and_repairs_unique_case():
    text = "The struggle was real. There were ample water stations."
    exact = ground_blog_evidence(text, "The struggle was real")
    case = ground_blog_evidence(text, "there were ample water stations")
    assert exact["raw_exact_match"] and not exact["evidence_repaired"]
    assert case["evidence_repair_method"] == "unique_case_insensitive_match"
    assert case["final_exact_evidence_text"] == "There were ample water stations"


def test_blog_grounding_repairs_discontinuous_evidence_to_one_exact_enclosing_span():
    text = "Breathing got heavier, runners slowed down. Muscles started to tighten."
    ellipsis = ground_blog_evidence(text, "Breathing got heavier ... Muscles started to tighten")
    assert ellipsis["evidence_repair_method"] == "unique_ordered_anchor_enclosing_span"
    assert ellipsis["final_exact_evidence_text"] in text
    assert ellipsis["final_exact_evidence_text"] == text[:-1]

    text = "Cooling mist helped all the runners in hot weather. A volunteer threw cold water. We need more cooling stations to prevent overheating."
    omitted = ground_blog_evidence(text, "Cooling mist helped all the runners in hot weather. We need more cooling stations to prevent overheating.")
    assert omitted["evidence_repair_method"] == "unique_ordered_anchor_enclosing_span"
    assert omitted["final_exact_evidence_text"] == text


def test_blog_grounding_rejects_ambiguous_or_invented_evidence():
    ambiguous = ground_blog_evidence("Good Route, then another good route", "GOOD ROUTE")
    invented = ground_blog_evidence("The route was difficult", "Volunteers were wonderful")
    assert ambiguous["normalized_match_status"] == "unrecoverable"
    assert invented["normalized_match_status"] == "unrecoverable"
