from pathlib import Path

import pandas as pd
import pytest

from policy_alignment_labeler import (
    completed_mask,
    language_requires_translation,
    load_alignment_labels,
    next_incomplete_index,
    save_alignment_label,
    translate_to_english,
)


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "alignment_id": "align-001",
            "document_id": "d1",
            "event_year": "2024",
            "primary_language": "English",
            "language_status": "ok",
            "original_text": "Training for KLSCM 2024",
            "second_coder_label": "",
            "visual_context_required": "",
            "reason_category": "",
            "evidence_span": "",
            "coder_notes": "",
        },
        {
            "alignment_id": "align-002",
            "document_id": "d2",
            "event_year": "2024",
            "primary_language": "Malay",
            "language_status": "ok",
            "original_text": "Generic morning run #KLSCM",
            "second_coder_label": "irrelevant",
            "visual_context_required": "no",
            "reason_category": "incidental_hashtag",
            "evidence_span": "#KLSCM",
            "coder_notes": "",
        },
    ])


def test_completion_and_navigation():
    frame = sample_frame()
    assert completed_mask(frame).tolist() == [False, True]
    assert next_incomplete_index(frame, 0) == 0
    assert next_incomplete_index(frame, 1) == 0


def test_save_alignment_label_is_atomic_and_creates_one_backup(tmp_path):
    path = tmp_path / "alignment.csv"
    original = sample_frame()
    original.to_csv(path, index=False)

    updated, backup = save_alignment_label(
        path, 0, "relevant", "no", "concrete_klscm_journey",
        "Training for KLSCM 2024", "explicit preparation",
    )
    loaded = load_alignment_labels(path)

    assert completed_mask(updated).all()
    assert loaded.at[0, "second_coder_label"] == "relevant"
    assert backup.exists()
    assert load_alignment_labels(backup).equals(original)

    _, same_backup = save_alignment_label(
        path, 1, "irrelevant", "no", "incidental_hashtag",
        "#KLSCM", backup_path=backup,
    )
    assert same_backup == backup


def test_save_requires_exact_evidence_and_preserves_file_on_error(tmp_path):
    path = tmp_path / "alignment.csv"
    sample_frame().to_csv(path, index=False)
    before = path.read_bytes()

    with pytest.raises(ValueError, match="copied exactly"):
        save_alignment_label(
            path, 0, "relevant", "no", "concrete_klscm_journey",
            "not in caption",
        )

    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.backup_*.csv"))


def test_load_requires_blind_alignment_schema(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"alignment_id": "align-001"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_alignment_labels(path)

@pytest.mark.parametrize("language", ["English", "Malay", "Bahasa Melayu", "Chinese", "Mandarin"])
def test_policy_reviewer_languages_do_not_require_translation(language):
    assert not language_requires_translation(language, "caption")


@pytest.mark.parametrize("language", ["Indonesian", "Thai", "Japanese", "Tamil", "Portuguese"])
def test_other_policy_languages_require_translation(language):
    assert language_requires_translation(language, "caption")


def test_policy_translation_uses_separate_cached_contract():
    class FakeService:
        def structured(self, stage, prompt_version, instructions, text, schema, extra=""):
            assert stage == "policy_alignment_translation"
            assert prompt_version == "policy-alignment-translation-v1"
            assert text == "Selamat pagi #KLSCM"
            assert extra == "Indonesian"
            return {"english_translation": "Good morning #KLSCM"}, {"cached": True}

    assert translate_to_english("Selamat pagi #KLSCM", "Indonesian", FakeService()) == "Good morning #KLSCM"