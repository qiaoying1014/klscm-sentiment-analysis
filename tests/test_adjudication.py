from pathlib import Path

import pandas as pd
import pytest

from marathon_absa.adjudication import (
    ANNOTATION_COLUMNS,
    analyze_agreement,
    build_blinded_annotation,
    finalize_gold_standard,
    import_reviewer_file,
    initialize_adjudication_package,
)


def candidate_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "record_id": "d1", "original_caption": "Proud! #KLSCM",
            "language": "English", "event_year": "2024",
            "disagreement_reason": "Hashtag interpretation changed",
            "event_connection": "supported",
            "primary_content_type": "event_emotional_experience",
            "human_label_v7": "exclude", "model_prediction_v8": "include",
            "confidence": "0.99", "short_explanation": "hidden",
            "reason_code": "hidden", "routing_status": "automatic_include",
        },
        {
            "record_id": "d2", "original_caption": "Morning run",
            "language": "English", "event_year": "2024",
            "disagreement_reason": "Historical annotation inconsistent",
            "event_connection": "none",
            "primary_content_type": "generic_running",
            "human_label_v7": "include", "model_prediction_v8": "unresolved",
            "confidence": "0.50", "short_explanation": "hidden",
            "reason_code": "hidden", "routing_status": "pending_review",
        },
        {
            "record_id": "not-disputed", "original_caption": "KLSCM finish",
            "language": "English", "event_year": "2024",
            "disagreement_reason": "",
            "event_connection": "explicit",
            "primary_content_type": "event_achievement",
            "human_label_v7": "include", "model_prediction_v8": "include",
            "confidence": "0.99", "short_explanation": "hidden",
            "reason_code": "hidden", "routing_status": "automatic_include",
        },
    ])


def completed_review(path: Path, labels: dict[str, str]) -> None:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame["reviewer_label"] = frame.record_id.map(labels)
    frame["evidence_span"] = frame.caption
    frame["rationale"] = "Applied the v8 text-only policy."
    frame["reviewer_confidence"] = "0.90"
    frame["optional_comments"] = ""
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def setup_package(tmp_path: Path) -> tuple[Path, Path]:
    candidates_path = tmp_path / "candidates.csv"
    package = tmp_path / "package"
    candidate_frame().to_csv(candidates_path, index=False)
    initialize_adjudication_package(candidates_path, package, seed=9)
    return candidates_path, package


def test_blinded_template_contains_only_permitted_fields_and_disputes():
    blinded = build_blinded_annotation(candidate_frame(), seed=3)
    assert list(blinded.columns) == ANNOTATION_COLUMNS
    assert set(blinded.record_id) == {"d1", "d2"}
    hidden = {
        "human_label_v7", "model_prediction_v8", "confidence",
        "short_explanation", "routing_status", "reason_code",
        "event_connection", "primary_content_type",
    }
    assert hidden.isdisjoint(blinded.columns)


def test_initialization_is_reproducible_and_never_overwrites(tmp_path):
    candidates_path, package = setup_package(tmp_path)
    first = pd.read_csv(package / "relevance_v8_blinded_annotation.csv")
    assert len(first) == 2
    with pytest.raises(FileExistsError):
        initialize_adjudication_package(candidates_path, package, seed=9)


def test_import_is_append_only_and_preserves_provenance(tmp_path):
    _, package = setup_package(tmp_path)
    review = package / "relevance_v8_reviewer_a.csv"
    completed_review(review, {"d1": "include", "d2": "exclude"})
    imported = import_reviewer_file(review, "reviewer-a", "v1", package)
    snapshot = pd.read_csv(imported, dtype=str, keep_default_na=False)
    assert snapshot.reviewer_id.eq("reviewer-a").all()
    assert snapshot.original_file_sha256.str.len().eq(64).all()
    with pytest.raises(FileExistsError):
        import_reviewer_file(review, "reviewer-a", "v1", package)


def test_import_rejects_missing_labels_and_rationales(tmp_path):
    _, package = setup_package(tmp_path)
    review = package / "relevance_v8_reviewer_a.csv"
    with pytest.raises(ValueError):
        import_reviewer_file(review, "reviewer-a", "v1", package)


def imported_pair(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path]:
    candidates, package = setup_package(tmp_path)
    a_file = package / "relevance_v8_reviewer_a.csv"
    b_file = package / "relevance_v8_reviewer_b.csv"
    completed_review(a_file, {"d1": "include", "d2": "exclude"})
    completed_review(b_file, {"d1": "exclude", "d2": "exclude"})
    a = import_reviewer_file(a_file, "reviewer-a", "v1", package)
    b = import_reviewer_file(b_file, "reviewer-b", "v1", package)
    return candidates, package, a, b


def test_agreement_outputs_overall_subgroups_and_only_disagreements(tmp_path):
    candidates, package, a, b = imported_pair(tmp_path)
    output = package / "agreement"
    summary = analyze_agreement(a, b, candidates, output)
    assert summary["records"] == 2
    assert summary["agreements"] == 1
    stats = pd.read_csv(output / "agreement_statistics.csv")
    assert {"overall", "language", "event_connection", "primary_content_type"}.issubset(
        set(stats.dimension)
    )
    report = pd.read_csv(
        output / "relevance_v8_adjudication_report.csv",
        dtype=str, keep_default_na=False,
    )
    assert report.record_id.tolist() == ["d1"]
    assert report.adjudicated_v8_label.eq("").all()


def test_agreement_rejects_same_reviewer_identity(tmp_path):
    candidates, package = setup_package(tmp_path)
    source_a = package / "relevance_v8_reviewer_a.csv"
    source_b = package / "relevance_v8_reviewer_b.csv"
    completed_review(source_a, {"d1": "include", "d2": "exclude"})
    completed_review(source_b, {"d1": "exclude", "d2": "exclude"})
    a = import_reviewer_file(source_a, "same", "v1", package)
    b = import_reviewer_file(source_b, "same", "v2", package)
    with pytest.raises(ValueError, match="independent"):
        analyze_agreement(a, b, candidates, package / "agreement")


def test_finalization_requires_all_disagreement_adjudications(tmp_path):
    candidates, package, a, b = imported_pair(tmp_path)
    agreement = package / "agreement"
    analyze_agreement(a, b, candidates, agreement)
    report = agreement / "relevance_v8_adjudication_report.csv"
    with pytest.raises(ValueError, match="requires include or exclude"):
        finalize_gold_standard(
            a, b, candidates, report, package / "gold", "v1",
        )


def test_final_gold_is_new_append_only_dataset_with_all_provenance(tmp_path):
    candidates, package, a, b = imported_pair(tmp_path)
    agreement = package / "agreement"
    analyze_agreement(a, b, candidates, agreement)
    report_path = agreement / "relevance_v8_adjudication_report.csv"
    report = pd.read_csv(report_path, dtype=str, keep_default_na=False)
    report["adjudicated_v8_label"] = "include"
    report["adjudicator"] = "senior-reviewer"
    report["adjudication_rationale"] = "The hashtag and pride form supported evidence."
    report["adjudicated_at_utc"] = "2026-07-30T00:00:00+00:00"
    report.to_csv(report_path, index=False)
    gold = finalize_gold_standard(
        a, b, candidates, report_path, package / "gold", "v1",
    )
    final = pd.read_csv(gold, dtype=str, keep_default_na=False)
    assert {
        "historical_v7_label", "reviewer1_label", "reviewer2_label",
        "adjudicated_v8_label", "adjudicator", "adjudicated_at_utc",
        "adjudication_rationale",
    }.issubset(final.columns)
    assert dict(zip(final.record_id, final.adjudicated_v8_label)) == {
        "d1": "include", "d2": "exclude",
    }
    with pytest.raises(FileExistsError):
        finalize_gold_standard(
            a, b, candidates, report_path, package / "gold", "v1",
        )
