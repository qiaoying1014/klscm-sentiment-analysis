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
        "POLICY_ALIGNMENT_LABEL_CSV",
        "data/processed/relevance_policy_alignment_second_coder.csv",
    )
)
VALID_LABELS = ("relevant", "irrelevant")
VISUAL_OPTIONS = ("no", "yes")
REASON_CATEGORIES = (
    "concrete_klscm_journey",
    "another_event",
    "generic_running",
    "commercial_promotion",
    "lifestyle_or_spam",
    "incidental_hashtag",
    "image_dependent",
    "insufficient_text",
)
REQUIRED_COLUMNS = {
    "alignment_id", "document_id", "event_year", "primary_language",
    "language_status", "original_text", "second_coder_label",
    "visual_context_required", "reason_category", "evidence_span", "coder_notes",
}
TRANSLATION_SCHEMA = {
    "type": "object",
    "properties": {"english_translation": {"type": "string"}},
    "required": ["english_translation"],
    "additionalProperties": False,
}
TRANSLATION_INSTRUCTIONS = """Translate the supplied social-media caption into faithful natural English. Preserve names, handles, hashtags, emoji, URLs and KLSCM terminology. Do not summarize, classify, explain or add facts."""
TRANSLATION_EXEMPT_LANGUAGES = {"english", "malay", "bahasa melayu", "chinese", "mandarin", "mandarin chinese"}


def load_alignment_labels(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
    return frame


def completed_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        frame.second_coder_label.str.strip().str.lower().isin(VALID_LABELS)
        & frame.visual_context_required.str.strip().str.lower().isin(VISUAL_OPTIONS)
        & frame.reason_category.str.strip().str.lower().isin(REASON_CATEGORIES)
        & frame.evidence_span.str.strip().ne("")
    )


def language_requires_translation(language: str, text: str) -> bool:
    normalized = " ".join(str(language).casefold().replace("_", " ").split())
    if not str(text).strip():
        return False
    return normalized not in TRANSLATION_EXEMPT_LANGUAGES


def translate_to_english(text: str, source_language: str, service: CachedOpenAI | None = None) -> str:
    translator = service or CachedOpenAI(SETTINGS.cache_dir, SETTINGS.chat_model)
    result, _ = translator.structured("policy_alignment_translation", "policy-alignment-translation-v1", TRANSLATION_INSTRUCTIONS, text, TRANSLATION_SCHEMA, extra=source_language)
    return str(result["english_translation"]).strip()

def next_incomplete_index(frame: pd.DataFrame, current_index: int) -> int:
    pending = frame.index[~completed_mask(frame)].tolist()
    if not pending:
        return current_index
    following = [index for index in pending if index > current_index]
    return following[0] if following else pending[0]


def save_alignment_label(
    path: Path,
    row_index: int,
    label: str,
    visual_context_required: str,
    reason_category: str,
    evidence_span: str,
    coder_notes: str = "",
    backup_path: Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    frame = load_alignment_labels(path)
    label = label.strip().lower()
    visual_context_required = visual_context_required.strip().lower()
    reason_category = reason_category.strip().lower()
    evidence_span = evidence_span.strip()
    if label not in VALID_LABELS:
        raise ValueError("Choose relevant or irrelevant")
    if visual_context_required not in VISUAL_OPTIONS:
        raise ValueError("Choose yes or no for visual context")
    if reason_category not in REASON_CATEGORIES:
        raise ValueError("Choose a valid reason category")
    if not evidence_span:
        raise ValueError("Enter a short exact evidence span")
    original_text = str(frame.at[row_index, "original_text"])
    if evidence_span not in original_text:
        raise ValueError("Evidence span must be copied exactly from the original text")
    if backup_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_path = path.with_name(f"{path.stem}.backup_{timestamp}{path.suffix}")
        shutil.copy2(path, backup_path)
    frame.loc[row_index, [
        "second_coder_label", "visual_context_required", "reason_category",
        "evidence_span", "coder_notes",
    ]] = [
        label, visual_context_required, reason_category, evidence_span,
        coder_notes.strip(),
    ]
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


def render_app(csv_path: Path = CSV_PATH) -> None:
    st.set_page_config(page_title="KLSCM Policy Alignment", page_icon="label")
    st.title("KLSCM Policy-Alignment Labeler")
    st.caption("Work independently. Prior human and model labels are hidden.")
    rubric_path = csv_path.with_name("relevance_policy_alignment_rubric.md")
    st.sidebar.header("Annotation rubric")
    if rubric_path.exists():
        st.sidebar.markdown(rubric_path.read_text(encoding="utf-8"))
    else:
        st.sidebar.warning(f"Rubric not found: {rubric_path}")
    if not csv_path.exists():
        st.error(f"CSV not found: {csv_path}")
        st.stop()
    try:
        frame = load_alignment_labels(csv_path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(str(exc))
        st.stop()
    if "row_index" not in st.session_state:
        st.session_state.row_index = next_incomplete_index(frame, -1)
    row_index = max(0, min(int(st.session_state.row_index), len(frame) - 1))
    complete = completed_mask(frame)
    st.progress(
        int(complete.sum()) / len(frame),
        text=f"{int(complete.sum())} completed | {int((~complete).sum())} pending",
    )
    row = frame.iloc[row_index]
    st.subheader(f"Record {row_index + 1} of {len(frame)}")
    st.caption(
        f"ID: {row.alignment_id} | Year: {row.event_year or '-'} | "
        f"Language: {row.primary_language or '-'}"
    )
    st.text_area(
        "Original caption", value=row.original_text, height=300, disabled=True,
    )
    if language_requires_translation(row.primary_language, row.original_text):
        st.markdown("#### Automatic English translation")
        st.caption("This cached translation uses the configured OpenAI API and may incur cost the first time this caption is translated.")
        try:
            with st.spinner("Translating caption..."):
                translation = translate_to_english(row.original_text, row.primary_language)
        except Exception as exc:
            st.error(f"Translation unavailable. Check OPENAI_API_KEY and the configured model. Details: {exc}")
        else:
            st.text_area("English translation", value=translation, height=220, disabled=True)
    with st.form(f"alignment_form_{row_index}"):
        label_value = row.second_coder_label.strip().lower()
        visual_value = row.visual_context_required.strip().lower()
        reason_value = row.reason_category.strip().lower()
        label = st.radio(
            "Substantive relevance", VALID_LABELS,
            index=VALID_LABELS.index(label_value) if label_value in VALID_LABELS else None,
            horizontal=True,
        )
        visual = st.radio(
            "Does the decision require unavailable visual context?", VISUAL_OPTIONS,
            index=VISUAL_OPTIONS.index(visual_value) if visual_value in VISUAL_OPTIONS else None,
            horizontal=True,
        )
        reason = st.selectbox(
            "Reason category", REASON_CATEGORIES,
            index=REASON_CATEGORIES.index(reason_value) if reason_value in REASON_CATEGORIES else None,
            placeholder="Choose one",
        )
        evidence = st.text_input(
            "Short exact evidence span", value=row.evidence_span,
            help="Copy and paste an exact span from the original caption.",
        )
        notes = st.text_area("Optional coder notes", value=row.coder_notes, height=90)
        submitted = st.form_submit_button("Save and next", use_container_width=True)
    if submitted:
        try:
            updated, backup = save_alignment_label(
                csv_path, row_index, label or "", visual or "", reason or "",
                evidence, notes, st.session_state.get("backup_path"),
            )
        except (OSError, ValueError, IndexError, pd.errors.ParserError) as exc:
            st.error(str(exc))
        else:
            st.session_state.backup_path = backup
            st.session_state.row_index = next_incomplete_index(updated, row_index)
            st.rerun()
    previous, following = st.columns(2)
    if previous.button("Previous", disabled=row_index == 0, use_container_width=True):
        st.session_state.row_index = row_index - 1
        st.rerun()
    if following.button("Next", disabled=row_index == len(frame) - 1, use_container_width=True):
        st.session_state.row_index = row_index + 1
        st.rerun()


if __name__ == "__main__":
    render_app()
