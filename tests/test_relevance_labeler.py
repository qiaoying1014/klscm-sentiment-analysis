from pathlib import Path

import pandas as pd
import pytest

from relevance_labeler import (
    initial_index,
    language_requires_translation,
    load_labels,
    next_pending_index,
    pending_indices,
    save_label,
    translate_to_english,
)


COLUMNS = [
    "review_id",
    "document_id",
    "event_year",
    "primary_language",
    "language_status",
    "original_text",
    "repeat_of",
    "human_relevance",
    "review_notes",
]


def sample_frame(labels: list[str]) -> pd.DataFrame:
    rows = []
    for index, label in enumerate(labels):
        rows.append(
            {
                "review_id": "duplicate-review" if index < 2 else f"r{index}",
                "document_id": "duplicate-document" if index < 2 else f"d{index}",
                "event_year": "2024",
                "primary_language": "English",
                "language_status": "ok",
                "original_text": f"caption {index}",
                "repeat_of": "",
                "human_relevance": label,
                "review_notes": f"note {index}",
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def write_sample(path: Path, labels: list[str]) -> pd.DataFrame:
    frame = sample_frame(labels)
    frame.to_csv(path, index=False)
    return frame


def test_pending_and_initial_index_include_blank_and_malformed_labels():
    frame = sample_frame(["relevant", "", "amabiguous", "irrelevant"])
    assert pending_indices(frame) == [1, 2]
    assert initial_index(frame) == 1


def test_initial_index_is_zero_when_every_row_is_complete():
    assert initial_index(sample_frame(["relevant", "ambiguous", "irrelevant"])) == 0


def test_next_pending_wraps_and_stays_when_complete():
    frame = sample_frame(["", "relevant", "", "irrelevant"])
    assert next_pending_index(frame, 0) == 2
    assert next_pending_index(frame, 2) == 0
    assert next_pending_index(sample_frame(["relevant"]), 0) == 0


@pytest.mark.parametrize(
    "language",
    [
        "English", "Malay", "Bahasa Melayu", "Indonesian", "Bahasa Indonesia",
        "Chinese", "Mandarin", "Mandarin Chinese", "Malay/Indonesian uncertain",
    ],
)
def test_reviewer_ready_languages_do_not_require_translation(language):
    assert not language_requires_translation(language, "some text")


@pytest.mark.parametrize("language", ["Portuguese", "Japanese", "Tamil", "French"])
def test_other_languages_require_translation(language):
    assert language_requires_translation(language, "some text")


def test_empty_or_undetermined_text_does_not_request_translation():
    assert not language_requires_translation("Portuguese", "")
    assert not language_requires_translation("undetermined", "some text")


def test_translation_uses_cached_service_contract():
    class FakeService:
        def structured(
            self, stage, prompt_version, instructions, text, schema, extra="",
        ):
            assert stage == "human_translation"
            assert prompt_version == "human-translation-v1"
            assert text == "Bom dia #KLSCM"
            assert extra == "Portuguese"
            assert schema["required"] == ["english_translation"]
            return {"english_translation": "Good morning #KLSCM"}, {"cached": True}

    assert (
        translate_to_english("Bom dia #KLSCM", "Portuguese", FakeService())
        == "Good morning #KLSCM"
    )

@pytest.mark.parametrize("label", ["relevant", "ambiguous", "irrelevant"])
def test_save_label_updates_position_and_preserves_other_data(tmp_path, label):
    path = tmp_path / "labels.csv"
    original = write_sample(path, ["", ""])

    updated, backup = save_label(path, 1, label)
    loaded = load_labels(path)

    assert updated.at[1, "human_relevance"] == label
    assert loaded.at[1, "human_relevance"] == label
    assert loaded.at[0, "human_relevance"] == ""
    assert loaded["review_notes"].tolist() == original["review_notes"].tolist()
    assert loaded.columns.tolist() == COLUMNS
    assert backup.exists()
    assert load_labels(backup).equals(original)


def test_save_label_reuses_session_backup_and_edits_existing_label(tmp_path):
    path = tmp_path / "labels.csv"
    original = write_sample(path, ["relevant", "irrelevant"])

    _, backup = save_label(path, 0, "ambiguous")
    _, same_backup = save_label(path, 1, "relevant", backup)

    assert same_backup == backup
    assert load_labels(backup).equals(original)
    assert load_labels(path)["human_relevance"].tolist() == [
        "ambiguous",
        "relevant",
    ]


def test_save_label_rejects_invalid_label_without_changing_csv(tmp_path):
    path = tmp_path / "labels.csv"
    write_sample(path, [""])
    before = path.read_bytes()

    with pytest.raises(ValueError, match="Invalid relevance label"):
        save_label(path, 0, "maybe")

    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.backup_*.csv"))


def test_load_labels_requires_expected_schema(tmp_path):
    path = tmp_path / "labels.csv"
    pd.DataFrame([{"human_relevance": ""}]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_labels(path)
