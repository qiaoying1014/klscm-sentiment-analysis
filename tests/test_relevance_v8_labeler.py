from pathlib import Path

import pandas as pd
import pytest

from marathon_absa.adjudication import ANNOTATION_COLUMNS
from relevance_v8_labeler import (
    completed_mask,
    initial_index,
    load_review,
    next_pending_index,
    save_review,
    translate_to_english,
)


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "record_id": "r1",
            "caption": "Bangga tamat #KLSCM",
            "language": "Malay",
            "event_year": "2024",
            "reviewer_label": "",
            "evidence_span": "",
            "rationale": "",
            "reviewer_confidence": "",
            "optional_comments": "",
        },
        {
            "record_id": "r2",
            "caption": "Morning run",
            "language": "English",
            "event_year": "2023",
            "reviewer_label": "exclude",
            "evidence_span": "Morning run",
            "rationale": "No KLSCM connection.",
            "reviewer_confidence": "0.90",
            "optional_comments": "",
        },
    ], columns=ANNOTATION_COLUMNS)


def write_sample(path: Path) -> None:
    sample_frame().to_csv(path, index=False, encoding="utf-8-sig")


def test_completion_and_pending_navigation():
    frame = sample_frame()
    assert completed_mask(frame).tolist() == [False, True]
    assert initial_index(frame) == 0
    assert next_pending_index(frame, 0) == 0


def test_load_review_requires_adjudication_schema(tmp_path):
    path = tmp_path / "review.csv"
    pd.DataFrame([{"record_id": "r1"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_review(path)


def test_save_review_is_atomic_and_preserves_backup(tmp_path):
    path = tmp_path / "review.csv"
    write_sample(path)
    original = load_review(path)
    updated, backup = save_review(
        path,
        0,
        "include",
        "Bangga tamat #KLSCM",
        "KLSCM-linked achievement.",
        0.85,
        "Clear case.",
    )
    assert completed_mask(updated).all()
    assert updated.at[0, "reviewer_confidence"] == "0.85"
    assert load_review(backup).equals(original)
    assert load_review(path).at[0, "reviewer_label"] == "include"


@pytest.mark.parametrize(
    ("label", "evidence", "rationale", "confidence", "message"),
    [
        ("", "text", "reason", 0.8, "include or exclude"),
        ("include", "", "reason", 0.8, "evidence span"),
        ("include", "text", "", 0.8, "rationale"),
        ("include", "text", "reason", 1.1, "from 0 to 1"),
    ],
)
def test_save_review_rejects_incomplete_values(
    tmp_path, label, evidence, rationale, confidence, message
):
    path = tmp_path / "review.csv"
    write_sample(path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match=message):
        save_review(path, 0, label, evidence, rationale, confidence)
    assert path.read_bytes() == before


def test_translation_uses_v8_cache_identity():
    class FakeService:
        def structured(
            self, stage, prompt_version, instructions, text, schema, extra="",
        ):
            assert stage == "relevance_v8_human_translation"
            assert prompt_version == "relevance-v8-human-translation-v1"
            assert text == "Bangga tamat #KLSCM"
            assert extra == "Malay"
            return {"english_translation": "Proud to finish #KLSCM"}, {}

    assert translate_to_english(
        "Bangga tamat #KLSCM", "Malay", FakeService()
    ) == "Proud to finish #KLSCM"
