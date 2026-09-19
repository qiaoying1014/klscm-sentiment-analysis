from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from marathon_absa.single_researcher_audit import AUDIT_COLUMNS, VALID_LABELS, atomic_save_annotations
from relevance_v8_labeler import translate_to_english


AUDIT_PATH = Path("data/processed/relevance_v8_single_researcher_architecture_audit_v2/relevance_v8_architecture_audit_blind_v2.csv")
REVIEW_PATH = Path("data/processed/relevance_v8_single_researcher_architecture_audit_v2/relevance_v8_operational_review_queue_v2.csv")


def load_audit(path: Path = AUDIT_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(AUDIT_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Audit CSV missing columns: {', '.join(sorted(missing))}")
    if any(token in column.lower() for column in frame.columns for token in ("automatic", "prediction", "routing", "model", "gold", "historical")):
        raise ValueError("Hidden prediction field detected in visible audit")
    return frame


def completed_mask(frame: pd.DataFrame, strict_audit: bool = True) -> pd.Series:
    confidence = pd.to_numeric(frame.confidence, errors="coerce")
    complete = frame.researcher_label.str.lower().isin(VALID_LABELS) & confidence.between(0, 1)
    return complete & frame.rationale.str.strip().ne("") if strict_audit else complete


def save_audit(path: Path, index: int, label: str, evidence: str,
               rationale: str, confidence: float, comments: str,
               backup_path: Path | None = None, strict_audit: bool = True) -> tuple[pd.DataFrame, Path]:
    frame = load_audit(path)
    label = str(label).strip().lower()
    if label not in VALID_LABELS:
        raise ValueError("Choose include or exclude")
    if strict_audit and not str(rationale).strip():
        raise ValueError("Enter a short rationale")
    if strict_audit and label == "include" and not str(evidence).strip():
        raise ValueError("Include decisions require grounded evidence from the original caption")
    if not 0 <= float(confidence) <= 1:
        raise ValueError("Confidence must be between 0 and 1")
    if backup_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = path.with_name(f"{path.stem}.session_backup_{stamp}.csv")
        shutil.copy2(path, backup_path)
    frame.loc[index, ["researcher_label", "evidence_span", "rationale", "confidence", "comments"]] = [label, str(evidence).strip(), str(rationale).strip(), f"{float(confidence):.2f}", str(comments).strip()]
    atomic_save_annotations(path, frame)
    return frame, backup_path


def render_app() -> None:
    st.set_page_config(page_title="KLSCM single-researcher relevance audit", page_icon="🔎")
    st.title("KLSCM relevance quality audit")
    st.caption("The production route and audit arm are hidden. Label only from the original record.")
    with st.sidebar:
        activity = st.selectbox("Activity", ["Random quality audit", "Operational review queue"])
        st.header("v8 event-experience rubric")
        st.markdown("**Include** defensible KLSCM-linked meaningful experience, participation, preparation, logistics, organization, atmosphere, safety, results, achievement, facilities, or event information.\n\n**Exclude** generic or unrelated running, another event, pure promotion, spam, incidental KLSCM mention, hashtag stuffing, hashtag-only text, or no defensible KLSCM connection.\n\nDo not infer unseen image content. Translation is optional and never replaces original evidence.")
    active_path = AUDIT_PATH if activity == "Random quality audit" else REVIEW_PATH
    if st.session_state.get("sra_active_path") != str(active_path):
        st.session_state.sra_active_path = str(active_path); st.session_state.sra_index = 0; st.session_state.sra_backup = None
    if not active_path.exists():
        st.error(f"Annotation file not found: {active_path}"); st.stop()
    frame = load_audit(active_path)
    strict_audit = activity == "Random quality audit"
    complete = completed_mask(frame, strict_audit)
    if "sra_index" not in st.session_state:
        pending = frame.index[~complete].tolist(); st.session_state.sra_index = pending[0] if pending else 0
    index = min(max(int(st.session_state.sra_index), 0), len(frame)-1)
    row = frame.iloc[index]
    st.progress(float(complete.mean()), text=f"{int(complete.sum())} complete · {len(frame)-int(complete.sum())} pending · {len(frame)} total")
    st.caption(f"Audit ID: {row.audit_id} · Source: {row.source or '-'} · Year: {row.event_year or '-'} · Language: {row.primary_language or '-'}")
    st.text_area("Original caption", row.original_text, height=280, disabled=True)
    if st.toggle("Translate into English (may incur API cost on cache miss)", key=f"translate_{index}"):
        st.text_area("English translation", translate_to_english(row.original_text, row.primary_language), disabled=True)
    label = st.radio("Researcher decision", ["include", "exclude"], index=None, horizontal=True, key=f"label_{index}")
    evidence = st.text_area("Evidence span", value=row.evidence_span, key=f"evidence_{index}")
    rationale = st.text_area("Short rationale (required)", value=row.rationale, key=f"rationale_{index}")
    confidence = st.slider("Confidence", 0.0, 1.0, float(row.confidence or .80), .05, key=f"confidence_{index}")
    comments = st.text_area("Optional comments", value=row.comments, key=f"comments_{index}")
    if st.button("Save and continue", type="primary", use_container_width=True):
        updated, backup = save_audit(active_path, index, label or "", evidence, rationale, confidence, comments, st.session_state.get("sra_backup"), strict_audit=activity == "Random quality audit")
        st.session_state.sra_backup = backup
        pending = updated.index[~completed_mask(updated, strict_audit)].tolist()
        st.session_state.sra_index = next((i for i in pending if i > index), pending[0] if pending else index)
        st.rerun()
    left, right = st.columns(2)
    if left.button("Previous", disabled=index == 0): st.session_state.sra_index = index-1; st.rerun()
    if right.button("Next", disabled=index == len(frame)-1): st.session_state.sra_index = index+1; st.rerun()


if __name__ == "__main__":
    render_app()
