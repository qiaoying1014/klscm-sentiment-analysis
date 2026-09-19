import pandas as pd
import pytest

from marathon_absa.relabel import (
    POLICY_VERSION,
    build_relabel_sample,
    build_repeat_disagreement_audit,
    eligible_relabel_documents,
    finalize_relabel_sample,
)


def documents_frame() -> pd.DataFrame:
    rows = []
    for index in range(30):
        feeling = "So proud and tired after #KLSCM" if index < 12 else f"Race caption {index}"
        rows.append({
            "document_id": f"d{index}", "source": "instagram",
            "processing_status": "ready", "linguistic_text": feeling.replace("#KLSCM", ""),
            "original_text": feeling, "event_year": str(2023 + index % 3),
            "primary_language": "English" if index % 2 else "Malay",
            "language_status": "ok",
        })
    rows.append({
        "document_id": "hashtags", "source": "instagram", "processing_status": "ready",
        "linguistic_text": "", "original_text": "#KLSCM #running", "event_year": "2024",
        "primary_language": "insufficient_text", "language_status": "insufficient_text",
    })
    return pd.DataFrame(rows)


def test_eligible_relabel_documents_excludes_hashtag_only():
    eligible = eligible_relabel_documents(documents_frame())
    assert "hashtags" not in set(eligible.document_id)
    assert len(eligible) == 30


def test_relabel_sample_is_blind_disjoint_and_contains_repeats():
    reviewer, key = build_relabel_sample(
        documents_frame(), {"d29"}, "pilot", n=12, repeats=3, seed=42,
    )
    assert len(key) == 12
    assert len(reviewer) == 15
    assert reviewer.repeat_of.ne("").sum() == 3
    assert "d29" not in set(key.document_id)
    assert "model_relevance" not in reviewer
    assert reviewer.policy_version.eq(POLICY_VERSION).all()
    assert key.challenge_category.eq("feeling_or_physical_cue").any()


def test_finalize_requires_binary_adjudication_and_reports_repeat_agreement():
    reviewer, _ = build_relabel_sample(
        documents_frame(), set(), "pilot", n=10, repeats=2, seed=43,
    )
    reviewer["human_relevance"] = "relevant"
    reviewer.loc[0, "human_relevance"] = "needs_review"
    with pytest.raises(ValueError, match="Adjudicate every needs_review"):
        finalize_relabel_sample(reviewer, "pilot")
    reviewer.loc[0, "human_relevance"] = "relevant"
    report = finalize_relabel_sample(reviewer, "pilot")
    assert report["base_records"] == 10
    assert report["repeat_records"] == 2
    assert report["repeat_agreement_rate"] == 1.0

def test_repeat_disagreement_audit_preserves_both_blind_labels():
    reviewer, _ = build_relabel_sample(
        documents_frame(), set(), "calibration", n=10, repeats=2, seed=44,
    )
    reviewer["human_relevance"] = "relevant"
    repeat_index = reviewer.index[reviewer.repeat_of.ne("")][0]
    reviewer.loc[repeat_index, "human_relevance"] = "irrelevant"
    audit = build_repeat_disagreement_audit(reviewer)
    assert len(audit) == 1
    assert audit.loc[0, "original_label"] == "relevant"
    assert audit.loc[0, "repeat_label"] == "irrelevant"
    assert audit.loc[0, "adjudicated_label"] == ""