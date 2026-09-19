from pathlib import Path

import pandas as pd
import pytest

from marathon_absa.adjudication import initialize_adjudication_package
from marathon_absa.adjudication_phase6 import (
    HIDDEN_FIELDS,
    _agreement_row,
    analyze_phase6_agreement,
    import_validated_submission,
    preflight_two_reviewers,
)


def candidates() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "record_id": "d1", "original_caption": "Proud #KLSCM",
            "language": "English", "event_year": "2024",
            "disagreement_reason": "Hashtag interpretation changed",
            "disagreement_category": "Hashtag-supported experience",
            "candidate_scope": "automatic false inclusion",
        },
        {
            "record_id": "d2", "original_caption": "Morning run",
            "language": "Malay", "event_year": "2023",
            "disagreement_reason": "Historical annotation inconsistent",
            "disagreement_category": "Human inconsistency",
            "candidate_scope": "review-routed",
        },
        {
            "record_id": "d3", "original_caption": "KLSCM results",
            "language": "English", "event_year": "2024",
            "disagreement_reason": "Information boundary changed",
            "disagreement_category": "Information boundary",
            "candidate_scope": "automatic false exclusion",
        },
    ])


def package(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    candidate_path = tmp_path / "candidates.csv"
    package_dir = tmp_path / "package"
    candidates().to_csv(candidate_path, index=False)
    initialize_adjudication_package(candidate_path, package_dir, seed=4)
    a = package_dir / "relevance_v8_reviewer_a.csv"
    b = package_dir / "relevance_v8_reviewer_b.csv"
    for path, labels in [
        (a, {"d1": "include", "d2": "exclude", "d3": "include"}),
        (b, {"d1": "include", "d2": "include", "d3": "exclude"}),
    ]:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        frame["reviewer_label"] = frame.record_id.map(labels)
        frame["evidence_span"] = frame.caption
        frame["rationale"] = "Applied v8 policy."
        frame["reviewer_confidence"] = "0.90"
        frame.to_csv(path, index=False)
    return candidate_path, package_dir, a, b


def test_preflight_validates_both_before_import(tmp_path):
    _, package_dir, a, b = package(tmp_path)
    first, second, report = preflight_two_reviewers(
        a, b, package_dir, "reviewer_a", "reviewer_b", "v1", "v1",
    )
    assert len(first) == len(second) == 3
    assert report["overall_status"] == "passed"
    assert report["reviewer_a"]["invalid_label_count"] == 0


@pytest.mark.parametrize("failure", ["duplicate", "missing", "invalid", "caption"])
def test_preflight_rejects_bad_submission(tmp_path, failure):
    _, package_dir, a, b = package(tmp_path)
    frame = pd.read_csv(a, dtype=str, keep_default_na=False)
    if failure == "duplicate":
        frame.loc[1, "record_id"] = frame.loc[0, "record_id"]
    elif failure == "missing":
        frame = frame.iloc[:-1]
    elif failure == "invalid":
        frame.loc[0, "reviewer_label"] = "ambiguous"
    else:
        frame.loc[0, "caption"] = "modified"
    frame.to_csv(a, index=False)
    with pytest.raises(ValueError):
        preflight_two_reviewers(
            a, b, package_dir, "reviewer_a", "reviewer_b", "v1", "v1",
        )
    assert not (package_dir / "reviewer_imports").exists()


def test_preflight_rejects_identity_collision(tmp_path):
    _, package_dir, a, b = package(tmp_path)
    with pytest.raises(ValueError, match="distinct"):
        preflight_two_reviewers(
            a, b, package_dir, "same", "same", "v1", "v1",
        )


def test_import_is_append_only_and_preserves_role(tmp_path):
    _, package_dir, a, b = package(tmp_path)
    first, _, _ = preflight_two_reviewers(
        a, b, package_dir, "reviewer_a", "reviewer_b", "v1", "v1",
    )
    imported = import_validated_submission(
        first, a, "reviewer_a", "independent_human_reviewer_a",
        "v1", package_dir,
    )
    snapshot = pd.read_csv(imported)
    assert snapshot.reviewer_role.eq("independent_human_reviewer_a").all()
    with pytest.raises(FileExistsError):
        import_validated_submission(
            first, a, "reviewer_a", "independent_human_reviewer_a",
            "v1", package_dir,
        )


def imported_pair(tmp_path):
    candidate_path, package_dir, a, b = package(tmp_path)
    first, second, _ = preflight_two_reviewers(
        a, b, package_dir, "reviewer_a", "reviewer_b", "v1", "v1",
    )
    imported_a = import_validated_submission(
        first, a, "reviewer_a", "independent_human_reviewer_a",
        "v1", package_dir,
    )
    imported_b = import_validated_submission(
        second, b, "reviewer_b", "independent_human_reviewer_b",
        "v1", package_dir,
    )
    return candidate_path, package_dir, imported_a, imported_b


def test_agreement_counts_matrix_and_disagreement_only_export(tmp_path):
    candidate_path, package_dir, a, b = imported_pair(tmp_path)
    result = analyze_phase6_agreement(
        a, b, candidate_path, package_dir / "agreement", random_seed=8,
    )
    metrics = result["metrics"]
    assert metrics["agreements"] == 1
    assert metrics["disagreements"] == 2
    assert metrics["confusion_matrix"] == [[1, 1], [1, 0]]
    supervisor = result["supervisor"]
    assert len(supervisor) == 2
    assert supervisor.adjudicator_label.eq("").all()
    assert HIDDEN_FIELDS.isdisjoint(supervisor.columns)


def test_kappa_calculation_matches_sklearn(tmp_path):
    candidate_path, package_dir, a, b = imported_pair(tmp_path)
    result = analyze_phase6_agreement(
        a, b, candidate_path, package_dir / "agreement",
    )
    expected = -0.5
    assert result["metrics"]["cohen_kappa"] == pytest.approx(expected)


def test_subgroup_kappa_is_undefined_for_one_class():
    frame = pd.DataFrame({
        "reviewer_a_label": ["include", "include"],
        "reviewer_b_label": ["include", "include"],
    })
    row = _agreement_row(frame, "language", "English")
    assert row["cohen_kappa"] is None
    assert "kappa_undefined" in row["stability_flag"]
