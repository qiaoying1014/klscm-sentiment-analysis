from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from relevance_v8_labeler import translate_to_english


CSV_PATH = Path(
    "data/processed/relevance_v8_adjudication/agreement/"
    "relevance_v8_supervisor_adjudication.csv"
)
VALID_LABELS = ("include", "exclude")
REQUIRED_COLUMNS = {
    "record_id",
    "original_caption",
    "permitted_metadata",
    "recorded_language",
    "reviewer_a_label",
    "reviewer_a_evidence_span",
    "reviewer_a_rationale",
    "reviewer_a_confidence",
    "reviewer_a_comments",
    "reviewer_b_label",
    "reviewer_b_evidence_span",
    "reviewer_b_rationale",
    "reviewer_b_confidence",
    "reviewer_b_comments",
    "phase4_candidate_origin",
    "phase4_disagreement_category",
    "adjudicator_label",
    "adjudicator_rationale",
    "adjudicator_name",
    "adjudication_timestamp",
}


def load_adjudications(path: Path) -> pd.DataFrame:
    """Load the supervisor worksheet while preserving blank values."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {', '.join(sorted(missing))}"
        )
    return frame


def completed_mask(frame: pd.DataFrame) -> pd.Series:
    """Return rows containing all fields required by the supervisor protocol."""
    return (
        frame["adjudicator_label"].str.strip().str.lower().isin(VALID_LABELS)
        & frame["adjudicator_rationale"].str.strip().ne("")
        & frame["adjudicator_name"].str.strip().ne("")
        & frame["adjudication_timestamp"].str.strip().ne("")
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


def save_adjudication(
    path: Path,
    row_index: int,
    label: str,
    rationale: str,
    adjudicator_name: str,
    backup_path: Path | None = None,
    timestamp: str | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Validate and atomically save one supervisor decision."""
    frame = load_adjudications(path)
    if row_index < 0 or row_index >= len(frame):
        raise IndexError(f"Row index {row_index} is outside the CSV")
    label = str(label).strip().lower()
    rationale = str(rationale).strip()
    adjudicator_name = str(adjudicator_name).strip()
    if label not in VALID_LABELS:
        raise ValueError("Choose include or exclude")
    if not rationale:
        raise ValueError("Enter a grounded adjudication rationale")
    if not adjudicator_name:
        raise ValueError("Enter the adjudicator name")
    saved_at = timestamp or datetime.now(timezone.utc).isoformat()

    if backup_path is None:
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_path = path.with_name(
            f"{path.stem}.backup_{suffix}{path.suffix}"
        )
        shutil.copy2(path, backup_path)
    frame.loc[row_index, [
        "adjudicator_label",
        "adjudicator_rationale",
        "adjudicator_name",
        "adjudication_timestamp",
    ]] = [label, rationale, adjudicator_name, saved_at]

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8-sig",
            newline="",
            suffix=".csv",
            prefix=f".{path.stem}_",
            dir=path.parent,
            delete=False,
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
        if key.startswith("supervisor_field_"):
            del st.session_state[key]


def _change_record(index: int) -> None:
    st.session_state.supervisor_row_index = index
    _clear_record_widgets()


def _display_review(row: pd.Series, reviewer: str) -> None:
    prefix = f"reviewer_{reviewer}"
    st.markdown(f"### Reviewer {reviewer.upper()}")
    st.markdown(f"**Decision:** `{row[f'{prefix}_label']}`")
    st.markdown(f"**Evidence:** {row[f'{prefix}_evidence_span'] or '-'}")
    st.markdown(f"**Rationale:** {row[f'{prefix}_rationale'] or '-'}")
    st.markdown(f"**Confidence:** {row[f'{prefix}_confidence'] or '-'}")
    st.markdown(f"**Comments:** {row[f'{prefix}_comments'] or '-'}")


def render_app(csv_path: Path = CSV_PATH) -> None:
    st.set_page_config(
        page_title="KLSCM v8 Supervisor Adjudication",
        page_icon="⚖️",
        layout="wide",
    )
    st.title("KLSCM v8 Supervisor Adjudication")
    st.caption(
        "Adjudicate only reviewer disagreements. Decide independently from the "
        "official v8 policy; do not optimize agreement with either reviewer."
    )
    with st.sidebar:
        st.header("Adjudicator")
        adjudicator_name = st.text_input(
            "Name",
            key="supervisor_adjudicator_name",
            help="Saved with every adjudication.",
        )
        st.header("v8 decision rubric")
        st.markdown(
            "**Include** when the available caption text and permitted metadata "
            "establish a reasonable KLSCM connection plus meaningful "
            "experiential, emotional, physical, logistical, social, evaluative, "
            "organizational, informational, or journey-related content.\n\n"
            "**Exclude** when there is no defensible KLSCM connection, or the "
            "caption is only generic running/emotion, another event, promotion, "
            "spam, or meaningless content.\n\n"
            "A hashtag may support event context when meaningful text accompanies "
            "it. Do not infer unseen images or videos. Translation is an aid; "
            "ground the decision in the original caption."
        )
        st.info(
            "Classifier predictions, routing, historical labels, and Phase 4 "
            "proposed labels are intentionally unavailable."
        )

    st.caption(f"Decisions are written directly to `{csv_path.as_posix()}`.")
    if not csv_path.exists():
        st.error(f"Supervisor worksheet not found: {csv_path}")
        st.stop()
    try:
        frame = load_adjudications(csv_path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not load supervisor worksheet: {exc}")
        st.stop()
    if frame.empty:
        st.info("The supervisor worksheet has no disagreement records.")
        st.stop()

    if "supervisor_row_index" not in st.session_state:
        st.session_state.supervisor_row_index = initial_index(frame)
    row_index = max(
        0, min(int(st.session_state.supervisor_row_index), len(frame) - 1)
    )
    st.session_state.supervisor_row_index = row_index
    complete = completed_mask(frame)
    completed = int(complete.sum())
    st.progress(
        completed / len(frame),
        text=(
            f"{completed} adjudicated | {len(frame) - completed} pending | "
            f"{len(frame)} disagreements"
        ),
    )
    if complete.all():
        st.success("All disagreement records have been adjudicated.")
    if message := st.session_state.pop("supervisor_save_message", None):
        st.success(message)

    row = frame.iloc[row_index]
    st.subheader(f"Disagreement {row_index + 1} of {len(frame)}")
    st.caption(
        f"Record ID: {row['record_id']} | "
        f"Language: {row['recorded_language'] or '-'} | "
        f"Metadata: {row['permitted_metadata'] or '-'}"
    )
    st.text_area(
        "Original caption",
        value=row["original_caption"],
        height=280,
        disabled=True,
    )
    translate = st.toggle(
        "Translate caption into English",
        value=False,
        key=f"supervisor_translate_{row_index}",
        help=(
            "Uses the configured OpenAI API and may incur cost on a cache miss. "
            "The original caption is never replaced."
        ),
    )
    if translate:
        try:
            with st.spinner("Translating into English..."):
                translation = translate_to_english(
                    row["original_caption"], row["recorded_language"]
                )
        except Exception as exc:
            st.error(
                "Translation is unavailable. Check OPENAI_API_KEY and the "
                f"configured model. Details: {exc}"
            )
        else:
            st.text_area(
                "English translation",
                value=translation,
                height=220,
                disabled=True,
                key=f"supervisor_translation_{row_index}",
            )

    review_a, review_b = st.columns(2)
    with review_a:
        _display_review(row, "a")
    with review_b:
        _display_review(row, "b")
    st.caption(
        f"Candidate origin: {row['phase4_candidate_origin'] or '-'} | "
        f"Disagreement category: "
        f"{row['phase4_disagreement_category'] or '-'}"
    )
    st.divider()

    prefix = f"supervisor_field_{row_index}"
    existing_label = row["adjudicator_label"].strip().lower()
    selected_index = (
        VALID_LABELS.index(existing_label)
        if existing_label in VALID_LABELS
        else None
    )
    label = st.radio(
        "Final adjudicated label",
        VALID_LABELS,
        index=selected_index,
        horizontal=True,
        key=f"{prefix}_label",
    )
    rationale = st.text_area(
        "Adjudication rationale",
        value=row["adjudicator_rationale"],
        help=(
            "State briefly why the final decision follows the v8 policy. "
            "Resolve the boundary issue; do not merely choose a reviewer."
        ),
        key=f"{prefix}_rationale",
    )
    if row["adjudication_timestamp"].strip():
        st.caption(
            f"Last saved by {row['adjudicator_name']} at "
            f"{row['adjudication_timestamp']}"
        )
    if st.button(
        "Save adjudication",
        type="primary",
        use_container_width=True,
    ):
        try:
            updated, backup = save_adjudication(
                csv_path,
                row_index,
                label or "",
                rationale,
                adjudicator_name,
                st.session_state.get("supervisor_backup_path"),
            )
        except (OSError, ValueError, IndexError, pd.errors.ParserError) as exc:
            st.error(f"Could not save adjudication: {exc}")
        else:
            st.session_state.supervisor_backup_path = backup
            st.session_state.supervisor_save_message = (
                f'Disagreement {row_index + 1} saved as "{label}".'
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
        "Next", disabled=row_index == len(frame) - 1,
        use_container_width=True,
    ):
        _change_record(row_index + 1)
        st.rerun()


if __name__ == "__main__":
    render_app()
