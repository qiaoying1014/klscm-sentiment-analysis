"""Autosaving Streamlit review for the controlled c1-refined experiment."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from marathon_absa.topic_discovery import REVIEW_LABELS, TOPIC_QUALITY_LABELS

ROOT = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT / "data" / "processed" / "topic_discovery_v1" / "c1_refined_v1"
REVIEW_PATH = PACKAGE_DIR / "topic_review_c1_refined_v1.csv"
OUTLIER_PATH = PACKAGE_DIR / "topic_outlier_c1_refined_v1.csv"


def render_distribution(label: str, value: str) -> None:
    st.markdown(f"**{label}**")
    st.json(json.loads(value))


def render_documents(label: str, value: str) -> None:
    st.markdown(f"**{label}**")
    for index, item in enumerate(json.loads(value), 1):
        caption = item.get("caption", "") if isinstance(item, dict) else str(item)
        st.markdown(f"{index}. {caption}")


st.set_page_config(page_title="KLSCM c1-refined review", layout="wide")
st.title("KLSCM c1-refined topic-level interpretation")
st.caption("The final model decision is unresolved. Compare themes; do not classify individual records. Original c1 is frozen.")

if not REVIEW_PATH.exists():
    st.info("Prepare the c1 topic-review package before starting review.")
    st.stop()

data = pd.read_csv(REVIEW_PATH, keep_default_na=False)
reviewed = data.topic_relevance.isin(REVIEW_LABELS) & data.topic_name.str.strip().ne("")
st.progress(float(reviewed.mean()), text=f"{int(reviewed.sum())} / {len(data)} topics reviewed")

topic_id = st.selectbox("Topic", data.topic_id.tolist(), format_func=lambda value: f"Topic {value}")
position = data.index[data.topic_id.eq(topic_id)][0]
row = data.loc[position]
st.header(f"Topic {topic_id} · {int(row.topic_size):,} records · {row.corpus_percentage:.2f}%")
st.subheader(row.bertopic_name)
st.markdown("**Top terms**")
st.write(", ".join(json.loads(row.top_terms)))

left, right = st.columns(2)
with left:
    render_documents("Representative documents", row.representative_documents)
with right:
    render_documents("Fixed-seed random documents", row.random_documents)

with st.expander("Topic metadata", expanded=True):
    a, b, c = st.columns(3)
    with a:
        render_distribution("Language", row.language_distribution)
    with b:
        render_distribution("Event year", row.year_distribution)
    with c:
        render_distribution("Previous relevance state", row.relevance_distribution)

relevance_options = ["", *REVIEW_LABELS]
quality_options = ["", *TOPIC_QUALITY_LABELS]
topic_relevance = st.selectbox("Topic relevance *", relevance_options, index=relevance_options.index(row.topic_relevance) if row.topic_relevance in relevance_options else 0)
topic_name = st.text_input("Human topic name *", value=row.topic_name)
topic_quality = st.selectbox("Topic quality", quality_options, index=quality_options.index(row.topic_quality) if row.topic_quality in quality_options else 0)
notes = st.text_area("Optional notes", value=row.notes)

new_values = [topic_relevance, topic_name, topic_quality, notes]
old_values = [row.topic_relevance, row.topic_name, row.topic_quality, row.notes]
if new_values != old_values:
    data.loc[position, ["topic_relevance", "topic_name", "topic_quality", "notes"]] = new_values
    data.loc[position, "reviewed_at"] = datetime.now(timezone.utc).isoformat()
    data.to_csv(REVIEW_PATH, index=False, encoding="utf-8-sig")
    st.toast("Review saved automatically")

st.divider()
st.subheader("Topic -1: outliers/unassigned documents")
st.caption("This is not an irrelevance category and is not part of the normal-topic review count.")
if OUTLIER_PATH.exists():
    outlier = pd.read_csv(OUTLIER_PATH, keep_default_na=False).iloc[0]
    st.write(f"{int(outlier.topic_size):,} records · {outlier.corpus_percentage:.2f}%")
    with st.expander("Show outlier diagnostics and reproducible sample"):
        render_distribution("Language", outlier.language_distribution)
        render_distribution("Event year", outlier.year_distribution)
        render_distribution("Previous relevance state", outlier.relevance_distribution)
        render_documents("Random outlier captions", outlier.random_documents)
