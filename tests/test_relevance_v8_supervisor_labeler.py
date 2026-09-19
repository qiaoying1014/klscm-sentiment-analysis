from pathlib import Path

import pandas as pd
import pytest

from relevance_v8_supervisor_labeler import (
    REQUIRED_COLUMNS,
    completed_mask,
    initial_index,
    load_adjudications,
    next_pending_index,
    save_adjudication,
)


def sample_frame() -> pd.DataFrame:
    rows = []
    for index in range(2):
        row = {column: "" for column in REQUIRED_COLUMNS}
        row.update({
            "record_id": f"r{index}",
            "original_caption": f"Caption {index} #KLSCM",
            "permitted_metadata": '{"event_year": "2024"}',
            "recorded_language": "English",
            "reviewer_a_label": "include",
            "reviewer_a_evidence_span": "Caption",
            "reviewer_a_rationale": "KLSCM context.",
            "reviewer_a_confidence": "0.80",
            "reviewer_b_label": "exclude",
            "reviewer_b_evidence_span": "Caption",
            "reviewer_b_rationale": "Insufficient content.",
            "reviewer_b_confidence": "0.80",
            "phase4_candidate_origin": "review-routed",
            "phase4_disagreement_category": "Boundary",
        })
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.loc[1, [
        "adjudicator_label",
        "adjudicator_rationale",
        "adjudicator_name",
        "adjudication_timestamp",
    ]] = [
        "exclude",
        "The content is insufficient.",
        "Supervisor",
        "2026-07-30T00:00:00+00:00",
    ]
    return frame


def write_sample(path: Path) -> None:
    sample_frame().to_csv(path, index=False, encoding="utf-8-sig")


def test_completion_and_navigation():
    frame = sample_frame()
    assert completed_mask(frame).tolist() == [False, True]
    assert initial_index(frame) == 0
    assert next_pending_index(frame, 0) == 0


def test_load_requires_supervisor_schema(tmp_path):
    path = tmp_path / "supervisor.csv"
    pd.DataFrame([{"record_id": "r1"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_adjudications(path)


def test_save_adjudication_is_atomic_and_preserves_backup(tmp_path):
    path = tmp_path / "supervisor.csv"
    write_sample(path)
    original = load_adjudications(path)
    timestamp = "2026-07-30T10:00:00+00:00"
    updated, backup = save_adjudication(
        path,
        0,
        "include",
        "Meaningful KLSCM-linked preparation.",
        "Lead Supervisor",
        timestamp=timestamp,
    )
    assert completed_mask(updated).all()
    assert updated.at[0, "adjudication_timestamp"] == timestamp
    assert load_adjudications(backup).equals(original)
    assert load_adjudications(path).at[0, "adjudicator_label"] == "include"


@pytest.mark.parametrize(
    ("label", "rationale", "name", "message"),
    [
        ("", "Reason", "Supervisor", "include or exclude"),
        ("include", "", "Supervisor", "rationale"),
        ("include", "Reason", "", "name"),
    ],
)
def test_save_rejects_incomplete_adjudication(
    tmp_path, label, rationale, name, message
):
    path = tmp_path / "supervisor.csv"
    write_sample(path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match=message):
        save_adjudication(path, 0, label, rationale, name)
    assert path.read_bytes() == before
