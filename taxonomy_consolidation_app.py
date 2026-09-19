"""Autosaving final c1 taxonomy-consolidation interface."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from marathon_absa.final_taxonomy import TAXONOMY_ACTIONS

ROOT = Path(__file__).resolve().parent
PATH = ROOT / "data" / "processed" / "topic_discovery_v1" / "final_taxonomy_v1" / "final_taxonomy_worksheet_v1.csv"


def show_documents(title: str, value: str) -> None:
    st.markdown(f"**{title}**")
    for index, item in enumerate(json.loads(value), 1):
        caption = item.get("caption", "") if isinstance(item, dict) else str(item)
        st.markdown(f"{index}. {caption}")


st.set_page_config(page_title="KLSCM final taxonomy", layout="wide")
st.title("Final c1 taxonomy consolidation")
st.caption("c1 is final. Consolidate existing interpretations only; do not relabel records.")
if not PATH.exists():
    st.info("Run topic-taxonomy-prepare first.")
    st.stop()

data = pd.read_csv(PATH, keep_default_na=False)
complete = data.final_taxonomy_action.isin(TAXONOMY_ACTIONS)
st.progress(float(complete.mean()), text=f"{int(complete.sum())} / 40 taxonomy actions complete")
topic_id = st.selectbox("c1 topic", data.topic_id.tolist(), format_func=lambda value: f"Topic {value}")
index = data.index[data.topic_id.eq(topic_id)][0]
row = data.loc[index]
st.header(f"Topic {topic_id}: {row.human_topic_name}")
st.write(f"{int(row.topic_size):,} documents · {row.corpus_percentage:.2f}% · {row.topic_relevance} · {row.topic_quality}")
st.markdown("**Top terms**")
st.write(", ".join(json.loads(row.top_terms)))
left, right = st.columns(2)
with left:
    show_documents("Representative captions", row.representative_documents)
with right:
    show_documents("Fixed-seed random captions", row.random_documents)

st.subheader("Suggested overlaps—not decisions")
suggestions = pd.DataFrame(json.loads(row.suggested_merge_candidates))
if suggestions.empty:
    st.write("No top-ranked suggestion for this topic.")
else:
    st.dataframe(suggestions, hide_index=True, use_container_width=True)
if row.topic_relevance == "mixed_topic":
    st.write("Possible related substantive topic IDs:", json.loads(row.possible_related_substantive_topics))
    st.json(json.loads(row.relevance_distribution))

action_options = ["", *TAXONOMY_ACTIONS]
action = st.selectbox("Final taxonomy action *", action_options, index=action_options.index(row.final_taxonomy_action) if row.final_taxonomy_action in action_options else 0)
name = st.text_input("Final topic name (leave blank to keep existing name)", value=row.final_topic_name)
group = st.text_input("Level 1 parent / final topic group", value=row.final_topic_group)
target = st.text_input("Merge target topic ID (required only for merge)", value=row.merge_target)
note = st.text_area("Optional taxonomy note", value=row.taxonomy_note)
new = [action, name, target, group, note]
old = [row.final_taxonomy_action, row.final_topic_name, row.merge_target, row.final_topic_group, row.taxonomy_note]
if new != old:
    data.loc[index, ["final_taxonomy_action", "final_topic_name", "merge_target", "final_topic_group", "taxonomy_note"]] = new
    data.to_csv(PATH, index=False, encoding="utf-8-sig")
    st.toast("Taxonomy decision saved automatically")

st.divider()
st.caption("Topic -1: 5,703 preserved unassigned outliers (41.50%). It is not a coherent topic and is not labelled irrelevant.")
