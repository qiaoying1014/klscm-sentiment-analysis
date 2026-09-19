"""Compact autosaving c1 versus c1-refined model-selection review."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from marathon_absa.topic_discovery import (
    INTERPRETABILITY_CHANGE_LABELS, SEMANTIC_RELATION_LABELS,
    TOPIC_QUALITY_LABELS,
)

ROOT = Path(__file__).resolve().parent
PATH = ROOT / "data" / "processed" / "topic_discovery_v1" / "model_selection_review_v1" / "model_selection_compact_review_v1.csv"


def show_documents(title: str, value: str) -> None:
    st.markdown(f"**{title}**")
    for index, item in enumerate(json.loads(value), 1):
        caption = item.get("caption", "") if isinstance(item, dict) else str(item)
        st.markdown(f"{index}. {caption}")


st.set_page_config(page_title="KLSCM model selection", layout="wide")
st.title("c1 versus c1-refined compact comparison")
st.caption("Compare selected topics only. Do not create a new taxonomy or classify records.")
if not PATH.exists():
    st.info("Run topic-discovery-model-selection-prepare first.")
    st.stop()

data = pd.read_csv(PATH, keep_default_na=False)
complete = (
    data.interpretability_change.isin(INTERPRETABILITY_CHANGE_LABELS)
    & data.semantic_relation.isin(SEMANTIC_RELATION_LABELS)
    & data.refined_topic_quality.isin(TOPIC_QUALITY_LABELS)
)
st.progress(float(complete.mean()), text=f"{int(complete.sum())} / {len(data)} selected topics reviewed")
topic_id = st.selectbox("Refined topic", data.refined_topic_id.tolist(), format_func=lambda value: f"Refined topic {value}")
index = data.index[data.refined_topic_id.eq(topic_id)][0]
row = data.loc[index]
st.header(f"Refined topic {topic_id} · {int(row.topic_size):,} documents")
st.write("Selection reasons:", ", ".join(json.loads(row.selection_reasons)))
st.markdown("**Top terms**")
st.write(", ".join(json.loads(row.top_terms)))

left, right = st.columns(2)
with left:
    show_documents("Representative documents", row.representative_documents)
with right:
    show_documents("Fixed-seed random documents", row.random_documents)

st.subheader("Original c1 contributions")
st.write(f"Inherited from c1 outliers: {row.c1_outlier_inheritance_percentage:.2f}%")
contributions = pd.DataFrame(json.loads(row.c1_contributions))
st.dataframe(contributions, hide_index=True, use_container_width=True)

change_options = ["", *INTERPRETABILITY_CHANGE_LABELS]
relation_options = ["", *SEMANTIC_RELATION_LABELS]
quality_options = ["", *TOPIC_QUALITY_LABELS]
change = st.selectbox("Interpretability change *", change_options, index=change_options.index(row.interpretability_change) if row.interpretability_change in change_options else 0)
relation = st.selectbox("Semantic relation *", relation_options, index=relation_options.index(row.semantic_relation) if row.semantic_relation in relation_options else 0)
quality = st.selectbox("Refined topic quality *", quality_options, index=quality_options.index(row.refined_topic_quality) if row.refined_topic_quality in quality_options else 0)
note = st.text_area("Optional note", value=row.note)
new = [change, relation, quality, note]
old = [row.interpretability_change, row.semantic_relation, row.refined_topic_quality, row.note]
if new != old:
    data.loc[index, ["interpretability_change", "semantic_relation", "refined_topic_quality", "note"]] = new
    data.loc[index, "reviewed_at"] = datetime.now(timezone.utc).isoformat()
    data.to_csv(PATH, index=False, encoding="utf-8-sig")
    st.toast("Comparison saved automatically")
