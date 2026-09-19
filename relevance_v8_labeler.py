from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from marathon_absa.adjudication import ANNOTATION_COLUMNS, FINAL_LABELS
from marathon_absa.config import SETTINGS
from marathon_absa.openai_service import CachedOpenAI


PACKAGE_DIR = Path("data/processed/relevance_v8_adjudication")
REVIEWER_FILES = {
    "Reviewer A": PACKAGE_DIR / "relevance_v8_reviewer_a.csv",
    "Reviewer B": PACKAGE_DIR / "relevance_v8_reviewer_b.csv",
}
TRANSLATION_SCHEMA = {
    "type": "object",
    "properties": {"english_translation": {"type": "string"}},
    "required": ["english_translation"],
    "additionalProperties": False,
}
TRANSLATION_INSTRUCTIONS = """Translate the supplied social-media caption into
faithful, natural English. Preserve names, handles, hashtags, emoji, URLs, line
breaks, and KLSCM terminology. Do not summarize, classify, explain, censor, or
add facts. Return only the translation in the required JSON field."""


def load_review(path: Path) -> pd.DataFrame:
    """Load a blinded reviewer file without converting blanks to NaN."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(ANNOTATION_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {', '.join(sorted(missing))}"
        )
    return frame


def completed_mask(frame: pd.DataFrame) -> pd.Series:
    """Return records that satisfy the downstream reviewer-import contract."""
    confidence = pd.to_numeric(frame["reviewer_confidence"], errors="coerce")
    return (
        frame["reviewer_label"].str.strip().str.lower().isin(FINAL_LABELS)
        & frame["evidence_span"].str.strip().ne("")
        & frame["rationale"].str.strip().ne("")
        & confidence.between(0, 1)
    )


def initial_index(frame: pd.DataFrame) -> int:
    pending = frame.index[~completed_mask(frame)].tolist()
    return pending[0] if pending else 0


def next_pending_index(frame: pd.DataFrame, current_index: int) -> int:
    pending = frame.index[~completed_mask(frame)].tolist()
    if not pending:
        return current_index
    following = [index for index in pending if index > current_index]
    return following[0] if following else pending[0]


def translate_to_english(
    text: str,
    source_language: str,
    service: CachedOpenAI | None = None,
) -> str:
    """Translate a caption through the project's cache-backed OpenAI service."""
    translator = service or CachedOpenAI(SETTINGS.cache_dir, SETTINGS.chat_model)
    result, _ = translator.structured(
        "relevance_v8_human_translation",
        "relevance-v8-human-translation-v1",
        TRANSLATION_INSTRUCTIONS,
        text,
        TRANSLATION_SCHEMA,
        extra=source_language,
    )
    return str(result["english_translation"]).strip()


def save_review(
    path: Path,
    row_index: int,
    label: str,
    evidence_span: str,
    rationale: str,
    confidence: float | str,
    optional_comments: str = "",
    backup_path: Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Validate and atomically save one blinded review."""
    frame = load_review(path)
    if row_index < 0 or row_index >= len(frame):
        raise IndexError(f"Row index {row_index} is outside the CSV")
    label = str(label).strip().lower()
    evidence_span = str(evidence_span).strip()
    rationale = str(rationale).strip()
    if label not in FINAL_LABELS:
        raise ValueError("Choose include or exclude")
    if not evidence_span:
        raise ValueError("Enter an evidence span from the original caption")
    if not rationale:
        raise ValueError("Enter a brief rationale")
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError) as exc:
        raise ValueError("Confidence must be a number from 0 to 1") from exc
    if not 0 <= confidence_value <= 1:
        raise ValueError("Confidence must be from 0 to 1")

    if backup_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_path = path.with_name(
            f"{path.stem}.backup_{timestamp}{path.suffix}"
        )
        shutil.copy2(path, backup_path)
    frame.loc[row_index, [
        "reviewer_label", "evidence_span", "rationale",
        "reviewer_confidence", "optional_comments",
    ]] = [
        label, evidence_span, rationale, f"{confidence_value:.2f}",
        str(optional_comments).strip(),
    ]
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8-sig", newline="", suffix=".csv",
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


def _clear_record_widgets() -> None:
    for key in list(st.session_state):
        if key.startswith("v8_field_"):
            del st.session_state[key]


def _change_record(index: int) -> None:
    st.session_state.v8_row_index = index
    _clear_record_widgets()


def render_app() -> None:
    st.set_page_config(
        page_title="KLSCM v8 Independent Review",
        page_icon="🏷️",
        layout="centered",
    )
    st.title("KLSCM v8 Independent Review")
    st.caption(
        "Work independently. Historical decisions and model output remain hidden."
    )
    with st.sidebar:
        st.header("Reviewer file")
        reviewer_name = st.selectbox(
            "Choose your assigned file",
            tuple(REVIEWER_FILES),
            key="v8_reviewer_name",
        )
        st.warning(
            "Each reviewer must use only their assigned file. Do not inspect or "
            "edit the other reviewer's answers."
        )
        st.header("v8 decision rubric")
        st.markdown(
            "**Include** when the caption has a reasonable textual KLSCM "
            "connection and meaningful experiential, emotional, physical, "
            "logistical, social, evaluative, organizational, informational, or "
            "journey-related content.\n\n"
            "**Exclude** when there is no defensible KLSCM connection, or the "
            "caption is only generic running/emotion, another event, promotion, "
            "spam, or meaningless content.\n\n"
            "A KLSCM hashtag may support event context when meaningful text "
            "accompanies it. Do not infer unseen image or video content. Base "
            "evidence on the original caption, not its translation."
        )

    csv_path = REVIEWER_FILES[reviewer_name]
    if st.session_state.get("v8_active_path") != str(csv_path):
        st.session_state.v8_active_path = str(csv_path)
        st.session_state.v8_row_index = 0
        st.session_state.v8_backup_path = None
        _clear_record_widgets()
    st.caption(f"Answers are written directly to `{csv_path.as_posix()}`.")
    if not csv_path.exists():
        st.error(f"Reviewer CSV not found: {csv_path}")
        st.stop()
    try:
        frame = load_review(csv_path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not load reviewer CSV: {exc}")
        st.stop()
    if frame.empty:
        st.info("The reviewer CSV has no records.")
        st.stop()

    if "v8_row_index" not in st.session_state:
        st.session_state.v8_row_index = initial_index(frame)
    row_index = max(
        0, min(int(st.session_state.v8_row_index), len(frame) - 1)
    )
    st.session_state.v8_row_index = row_index
    complete = completed_mask(frame)
    completed = int(complete.sum())
    st.progress(
        completed / len(frame),
        text=(
            f"{completed} completed | {len(frame) - completed} pending | "
            f"{len(frame)} total"
        ),
    )
    if complete.all():
        st.success("This reviewer file is complete and ready for import.")
    if message := st.session_state.pop("v8_save_message", None):
        st.success(message)

    row = frame.iloc[row_index]
    st.subheader(f"Record {row_index + 1} of {len(frame)}")
    st.caption(
        f"Record ID: {row['record_id']} | Year: {row['event_year'] or '-'} | "
        f"Language: {row['language'] or '-'}"
    )
    st.text_area(
        "Original caption", value=row["caption"], height=300, disabled=True,
    )
    translate = st.toggle(
        "Translate caption into English",
        value=False,
        key=f"v8_translate_{reviewer_name}_{row_index}",
        help=(
            "Uses the configured OpenAI API and may incur cost on a cache miss. "
            "The original caption is never replaced."
        ),
    )
    if translate:
        try:
            with st.spinner("Translating into English..."):
                translation = translate_to_english(
                    row["caption"], row["language"]
                )
        except Exception as exc:
            st.error(
                "Translation is unavailable. Check OPENAI_API_KEY and the "
                f"configured model. Details: {exc}"
            )
        else:
            st.text_area(
                "English translation", value=translation, height=220,
                disabled=True,
                key=f"v8_translation_{reviewer_name}_{row_index}",
            )

    key_prefix = f"v8_field_{reviewer_name}_{row_index}"
    existing_label = row["reviewer_label"].strip().lower()
    labels = tuple(sorted(FINAL_LABELS, reverse=True))
    selected_index = (
        labels.index(existing_label) if existing_label in FINAL_LABELS else None
    )
    label = st.radio(
        "Decision", labels, index=selected_index, horizontal=True,
        key=f"{key_prefix}_label",
    )
    evidence_span = st.text_area(
        "Evidence span", value=row["evidence_span"],
        help=(
            "Copy the shortest useful words or clauses from the original caption. "
            "For exclusions, record the text that best supports the exclusion."
        ),
        key=f"{key_prefix}_evidence",
    )
    rationale = st.text_area(
        "Rationale", value=row["rationale"],
        help="Briefly explain how the v8 rubric leads to this decision.",
        key=f"{key_prefix}_rationale",
    )
    try:
        confidence_value = float(row["reviewer_confidence"])
    except ValueError:
        confidence_value = 0.80
    confidence = st.slider(
        "Reviewer confidence", min_value=0.0, max_value=1.0,
        value=confidence_value, step=0.05,
        key=f"{key_prefix}_confidence",
    )
    comments = st.text_area(
        "Optional comments", value=row["optional_comments"],
        key=f"{key_prefix}_comments",
    )
    if st.button("Save review", type="primary", use_container_width=True):
        try:
            updated, backup = save_review(
                csv_path, row_index, label or "", evidence_span, rationale,
                confidence, comments, st.session_state.get("v8_backup_path"),
            )
        except (OSError, ValueError, IndexError, pd.errors.ParserError) as exc:
            st.error(f"Could not save review: {exc}")
        else:
            st.session_state.v8_backup_path = backup
            st.session_state.v8_save_message = (
                f'Record {row_index + 1} saved as "{label}".'
            )
            _change_record(next_pending_index(updated, row_index))
            st.rerun()

    previous, next_record = st.columns(2)
    if previous.button(
        "Previous", disabled=row_index == 0, use_container_width=True
    ):
        _change_record(row_index - 1)
        st.rerun()
    if next_record.button(
        "Next", disabled=row_index == len(frame) - 1, use_container_width=True
    ):
        _change_record(row_index + 1)
        st.rerun()


if __name__ == "__main__":
    render_app()
