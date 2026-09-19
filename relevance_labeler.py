from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from marathon_absa.config import SETTINGS
from marathon_absa.openai_service import CachedOpenAI


CSV_PATH = Path(
    os.getenv(
        "RELEVANCE_LABEL_CSV",
        "data/processed/relevance_review_queue.csv",
    )
)
VALID_LABELS = ("relevant", "ambiguous", "irrelevant")
RELABEL_LABELS = ("relevant", "needs_review", "irrelevant")
FINAL_LABELS = ("relevant", "irrelevant")
TRANSLATION_SCHEMA = {
    "type": "object",
    "properties": {"english_translation": {"type": "string"}},
    "required": ["english_translation"],
    "additionalProperties": False,
}
TRANSLATION_INSTRUCTIONS = """Translate the supplied social-media text into natural,
faithful English. Preserve names, handles, hashtags, emoji, URLs, line breaks, and
KLSCM event terminology. Do not summarize, classify, explain, censor, or add facts.
Return only the translation in the required JSON field."""
TRANSLATION_EXEMPT_LANGUAGES = {
    "english", "malay", "bahasa melayu", "indonesian", "bahasa indonesia",
    "chinese", "mandarin", "mandarin chinese",
}
REQUIRED_COLUMNS = {
    "document_id", "event_year", "primary_language", "original_text",
    "human_relevance", "review_notes",
}


def load_labels(path: Path) -> pd.DataFrame:
    """Load the labeling CSV while preserving blank fields and string values."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"CSV is missing required columns: {names}")
    return frame


def pending_indices(
    frame: pd.DataFrame, valid_labels: tuple[str, ...] = VALID_LABELS
) -> list[int]:
    """Return row positions whose label is blank or invalid."""
    labels = frame["human_relevance"].astype(str).str.strip().str.lower()
    return frame.index[~labels.isin(valid_labels)].tolist()


def initial_index(
    frame: pd.DataFrame, valid_labels: tuple[str, ...] = VALID_LABELS
) -> int:
    pending = pending_indices(frame, valid_labels)
    return pending[0] if pending else 0


def next_pending_index(
    frame: pd.DataFrame,
    current_index: int,
    valid_labels: tuple[str, ...] = VALID_LABELS,
) -> int:
    """Find the next pending row, wrapping once to the start."""
    pending = pending_indices(frame, valid_labels)
    if not pending:
        return current_index
    following = [index for index in pending if index > current_index]
    return following[0] if following else pending[0]


def language_requires_translation(language: str, text: str) -> bool:
    """Whether a non-empty record falls outside the reviewer-ready languages."""
    normalized = " ".join(str(language).casefold().replace("_", " ").split())
    if not str(text).strip():
        return False
    if normalized in TRANSLATION_EXEMPT_LANGUAGES:
        return False
    if "malay" in normalized and "indonesian" in normalized:
        return False
    return normalized not in {
        "", "undetermined", "no text", "insufficient text",
        "no linguistic content",
    }


def translate_to_english(
    text: str,
    source_language: str,
    service: CachedOpenAI | None = None,
) -> str:
    """Translate text with the project's cached OpenAI service."""
    translator = service or CachedOpenAI(SETTINGS.cache_dir, SETTINGS.chat_model)
    result, _ = translator.structured(
        "human_translation",
        "human-translation-v1",
        TRANSLATION_INSTRUCTIONS,
        text,
        TRANSLATION_SCHEMA,
        extra=source_language,
    )
    return str(result["english_translation"]).strip()


def save_label(
    path: Path,
    row_index: int,
    label: str,
    backup_path: Path | None = None,
    valid_labels: tuple[str, ...] = VALID_LABELS,
) -> tuple[pd.DataFrame, Path]:
    """Update one row and atomically replace the CSV, creating one backup."""
    if label not in valid_labels:
        raise ValueError(f"Invalid relevance label: {label}")
    frame = load_labels(path)
    if row_index < 0 or row_index >= len(frame):
        raise IndexError(f"Row index {row_index} is outside the CSV")
    if backup_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_path = path.with_name(f"{path.stem}.backup_{timestamp}{path.suffix}")
        shutil.copy2(path, backup_path)
    frame.at[row_index, "human_relevance"] = label
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", suffix=".csv",
            prefix=f".{path.stem}_", dir=path.parent, delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            frame.to_csv(temporary, index=False, lineterminator="\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return frame, backup_path


def _display_value(value: str) -> str:
    return value.strip() or "-"


def _autosave_selection(
    csv_path: Path,
    row_index: int,
    widget_key: str,
    valid_labels: tuple[str, ...],
) -> None:
    label = st.session_state.get(widget_key)
    if label not in valid_labels:
        return
    try:
        updated, backup_path = save_label(
            csv_path, row_index, label, st.session_state.get("backup_path"),
            valid_labels,
        )
    except (OSError, ValueError, IndexError, pd.errors.ParserError) as exc:
        st.session_state.save_error = f"Could not save the label: {exc}"
        return
    st.session_state.backup_path = backup_path
    st.session_state.row_index = next_pending_index(
        updated, row_index, valid_labels
    )
    st.session_state.save_message = f'Row {row_index + 1} saved as "{label}".'


def render_app(csv_path: Path = CSV_PATH) -> None:
    st.set_page_config(
        page_title="Human Relevance Labeler", page_icon="label", layout="centered"
    )
    st.title("Human Relevance Labeler")
    with st.sidebar:
        st.header("Feeling-inclusive v7 rubric")
        st.markdown(
            "Label **Relevant** when the text links KLSCM to an event theme, "
            "an emotional or physical experience, or both. A journey stage is "
            "not required. One feeling such as excitement, tiredness, energy, "
            "relief, or pride is sufficient when defensibly attributed to KLSCM.\n\n"
            "Label **Irrelevant** when the feeling or activity is generic, another "
            "event is the subject, or no defensible KLSCM connection exists. "
            "Hashtag-only rows are excluded before this interface.\n\n"
            "Use **Needs review** temporarily; adjudicate it to Relevant or "
            "Irrelevant before finalization."
        )
    st.caption(f"Labels are written directly to `{csv_path.as_posix()}`.")
    if not csv_path.exists():
        st.error(f"CSV not found: {csv_path}")
        st.stop()
    try:
        frame = load_labels(csv_path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not load the labeling CSV: {exc}")
        st.stop()
    if frame.empty:
        st.info("The labeling CSV has no rows.")
        st.stop()
    is_review_queue = "review_status" in frame.columns
    is_v7_relabel = (
        "policy_version" in frame.columns
        and frame["policy_version"].eq("feeling-inclusive-v7").any()
    )
    valid_labels = (
        FINAL_LABELS if is_review_queue
        else RELABEL_LABELS if is_v7_relabel
        else VALID_LABELS
    )
    if "row_index" not in st.session_state:
        st.session_state.row_index = initial_index(frame, valid_labels)
    st.session_state.row_index = max(
        0, min(int(st.session_state.row_index), len(frame) - 1)
    )
    row_index = st.session_state.row_index
    pending = pending_indices(frame, valid_labels)
    completed = len(frame) - len(pending)
    st.progress(
        completed / len(frame),
        text=f"{completed} completed | {len(pending)} pending | {len(frame)} total",
    )
    if pending:
        if row_index in pending:
            pending_position = pending.index(row_index) + 1
            st.info(
                f"Unlabeled document {pending_position} of {len(pending)} | "
                f"{len(pending)} documents still need labels"
            )
        else:
            st.info(f"{len(pending)} documents still need labels")
    else:
        st.success("All documents have valid human relevance labels.")
    if message := st.session_state.pop("save_message", None):
        st.success(message)
    if error := st.session_state.pop("save_error", None):
        st.error(error)

    row = frame.iloc[row_index]
    st.subheader(f"Row {row_index + 1} of {len(frame)}")
    left, right = st.columns(2)
    with left:
        if "review_id" in frame:
            st.markdown(f"**Review ID:** {_display_value(row['review_id'])}")
        st.markdown(f"**Event year:** {_display_value(row['event_year'])}")
        st.markdown(f"**Language:** {_display_value(row['primary_language'])}")
    with right:
        st.markdown(f"**Document ID:** {_display_value(row['document_id'])}")
        if "language_status" in frame:
            st.markdown(
                f"**Language status:** {_display_value(row['language_status'])}"
            )
        if "repeat_of" in frame:
            st.markdown(f"**Repeat of:** {_display_value(row['repeat_of'])}")
    if is_review_queue:
        st.markdown("#### Model decision context")
        context_left, context_right = st.columns(2)
        context_left.markdown(
            f"**Current decision:** {_display_value(row.get('relevance', ''))}"
        )
        context_left.markdown(
            f"**Reason:** {_display_value(row.get('reason_code', ''))}"
        )
        context_right.markdown(
            f"**Confidence:** {_display_value(row.get('confidence', ''))}"
        )
        context_right.markdown(
            f"**Model language:** "
            f"{_display_value(row.get('model_observed_language', ''))}"
        )
        if str(row.get("evidence", "")).strip():
            st.caption(f"Model evidence: {row['evidence']}")
    if "original_label" in frame.columns and "repeat_label" in frame.columns:
        st.markdown("#### Blind-repeat disagreement")
        disagreement_left, disagreement_right = st.columns(2)
        disagreement_left.markdown(
            f"**Original decision:** {_display_value(row['original_label'])}"
        )
        disagreement_right.markdown(
            f"**Repeated decision:** {_display_value(row['repeat_label'])}"
        )
        st.caption(
            "Make a fresh final decision from the caption and rubric. The two "
            "blind decisions remain preserved for reliability reporting."
        )
    st.markdown("#### Original text")
    st.text_area(
        "Original text", value=row["original_text"], height=260, disabled=True,
        label_visibility="collapsed",
    )
    if language_requires_translation(row["primary_language"], row["original_text"]):
        st.markdown("#### English translation")
        auto_translate = st.toggle(
            "Enable automatic translation", value=False,
            help=(
                "Uses the configured OpenAI API and may incur cost. "
                "Successful translations are cached locally."
            ),
            key="auto_translate",
        )
        if auto_translate:
            try:
                with st.spinner("Translating to English..."):
                    translation = translate_to_english(
                        row["original_text"], row["primary_language"]
                    )
            except Exception as exc:
                st.error(
                    "Translation was unavailable. Check OPENAI_API_KEY and the "
                    f"configured model, then try again. Details: {exc}"
                )
            else:
                st.text_area(
                    "English translation", value=translation, height=180,
                    disabled=True, label_visibility="collapsed",
                    key=f"translation_{row_index}",
                )
        else:
            st.info(
                "Turn on automatic translation to show an English version. "
                "This uses the configured OpenAI API."
            )

    current_label = row["human_relevance"].strip().lower()
    selected_index = (
        valid_labels.index(current_label) if current_label in valid_labels else None
    )
    widget_key = f"label_{row_index}"
    st.radio(
        "Human relevance (saves automatically when selected)",
        valid_labels,
        index=selected_index,
        horizontal=True,
        key=widget_key,
        on_change=_autosave_selection,
        args=(csv_path, row_index, widget_key, valid_labels),
    )
    if row["human_relevance"].strip() and current_label not in valid_labels:
        st.warning(
            f'This row has an invalid existing value: "{row["human_relevance"]}". '
            "Choose a valid label to replace it."
        )
    previous_column, next_column = st.columns(2)
    if previous_column.button(
        "Previous", disabled=row_index == 0, use_container_width=True
    ):
        st.session_state.row_index = row_index - 1
        st.rerun()
    if next_column.button(
        "Next", disabled=row_index == len(frame) - 1, use_container_width=True
    ):
        st.session_state.row_index = row_index + 1
        st.rerun()


if __name__ == "__main__":
    render_app()




