from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

PATH = Path("data/processed/blog_analysis_v1/blog_theme_mapping_candidates_v1.csv")
EDITABLE = ["mapping_decision", "selected_reviewed_theme_id", "emergent_blog_theme_label", "researcher_note", "review_status"]


def atomic_save(frame: pd.DataFrame) -> None:
    PATH.parent.mkdir(parents=True, exist_ok=True)
    if PATH.exists() and "backup_created" not in st.session_state:
        backup = PATH.with_name(f"{PATH.stem}.backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv")
        shutil.copy2(PATH, backup); st.session_state.backup_created = str(backup)
    temporary = PATH.with_suffix(".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, PATH)


st.set_page_config(page_title="Blog theme mapping review", layout="wide")
st.title("Blog theme mapping review")
st.caption("Similarity scores are candidate-ranking aids only. No candidate is accepted automatically.")
if not PATH.exists():
    st.error("Create and finalize blog ABSA, then generate theme candidates first."); st.stop()
data = pd.read_csv(PATH, keep_default_na=False)
status_filter = st.selectbox("Rows", ["pending", "reviewed", "all"])
view = data if status_filter == "all" else data[data.review_status.eq(status_filter)]
if view.empty:
    st.info("No rows in this filter."); st.stop()
row_index = st.selectbox("Mention", view.index, format_func=lambda i: f"{i+1}: {view.loc[i, 'aspect']} — {view.loc[i, 'target']}")
row = data.loc[row_index]
st.subheader(f"{row.get('aspect')} · review {row.get('review_id')}")
st.write(row.get("evidence_text")); st.caption(f"English gloss: {row.get('english_gloss') or '—'}")
candidate_ids = [row.get(f"candidate_theme_id_{i}", "") for i in range(1, 4) if row.get(f"candidate_theme_id_{i}", "")]
for i in range(1, 4):
    if row.get(f"candidate_theme_id_{i}", ""): st.write(f"{i}. {row.get(f'candidate_theme_label_{i}')} · cosine {float(row.get(f'candidate_similarity_{i}', 0)):.3f}")
decision = st.selectbox("Decision", ["", "MATCH_EXISTING", "EMERGENT_BLOG_THEME", "UNCLEAR", "EXCLUDE_FROM_THEME_COMPARISON"], index=max(0, ["", "MATCH_EXISTING", "EMERGENT_BLOG_THEME", "UNCLEAR", "EXCLUDE_FROM_THEME_COMPARISON"].index(row.mapping_decision) if row.mapping_decision in ["", "MATCH_EXISTING", "EMERGENT_BLOG_THEME", "UNCLEAR", "EXCLUDE_FROM_THEME_COMPARISON"] else 0))
taxonomy = pd.read_csv("data/processed/absa_v1/aspect_level_themes_review_v1/reviewed_theme_taxonomy.csv", keep_default_na=False)
same_aspect = taxonomy[taxonomy.aspect.eq(row.aspect) & taxonomy.review_status.eq("reviewed")]
theme_labels = same_aspect.set_index("reviewed_theme_id").reviewed_theme_label.to_dict()
theme_options = [""] + list(dict.fromkeys(candidate_ids + list(theme_labels)))
selected = st.selectbox("Existing same-aspect theme", theme_options,
                        index=theme_options.index(row.selected_reviewed_theme_id) if row.selected_reviewed_theme_id in theme_options else 0,
                        format_func=lambda value: f"{theme_labels.get(value, value)} ({value})" if value else "")
emergent = st.text_input("Blog-emergent theme label", value=row.emergent_blog_theme_label)
note = st.text_area("Researcher note", value=row.researcher_note)
if st.button("Save reviewed decision", type="primary"):
    if decision == "MATCH_EXISTING" and not selected: st.error("Select a same-aspect reviewed theme.")
    elif decision == "EMERGENT_BLOG_THEME" and not emergent.strip(): st.error("Provide a human-readable emergent label.")
    elif not decision or decision == "UNCLEAR": st.error("This row remains unresolved and cannot be marked reviewed.")
    else:
        values = [decision, selected, emergent.strip(), note, "reviewed"]
        for column, value in zip(EDITABLE, values): data.loc[row_index, column] = value
        atomic_save(data); st.success("Decision saved atomically."); st.rerun()
st.metric("Reviewed", f"{data.review_status.eq('reviewed').sum()} / {len(data)}")
