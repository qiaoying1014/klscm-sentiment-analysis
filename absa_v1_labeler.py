from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from marathon_absa.absa_v1 import (
    DRAFT_ANNOTATIONS_PATH, DRAFT_PROGRESS_PATH, ONTOLOGY_PATH, annotation_paths,
    save_document_annotations, stable_mention_id,
)

historical = "--historical-150" in sys.argv
paths = annotation_paths(historical_150=historical)
SAMPLE_PATH, REVIEW_PATH, PROGRESS_PATH = paths["sample"], paths["annotations"], paths["progress"]

st.set_page_config(page_title="KLSCM ABSA V1 Review", layout="wide")
st.title("KLSCM ABSA V1 — researcher review")
st.caption("Current 80-document AI-assisted audit" if not historical else "Historical blinded 150-document package")
st.warning("AI drafts are provisional. A document becomes researcher-confirmed only after an explicit review decision.")

if not SAMPLE_PATH.exists():
    st.error("The frozen sample is missing.")
    st.stop()

sample = pd.read_csv(SAMPLE_PATH)
ontology = json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8"))
aspects = [a["id"] for a in ontology["aspects"]]
aspect_help = {a["id"]: f'{a["name"]}: {a["definition"]}' for a in ontology["aspects"]}
review_rows = pd.read_csv(REVIEW_PATH).fillna("").to_dict("records") if REVIEW_PATH.exists() else []
progress = (pd.read_csv(PROGRESS_PATH).fillna("") if PROGRESS_PATH.exists()
            else pd.DataFrame(columns=["document_id", "annotation_status", "no_evaluative_aspect_mention",
                                       "review_decision", "reviewed_at_utc"]))
draft_rows = (pd.read_csv(DRAFT_ANNOTATIONS_PATH).fillna("").to_dict("records")
              if not historical and DRAFT_ANNOTATIONS_PATH.exists() else [])
draft_progress = (pd.read_csv(DRAFT_PROGRESS_PATH).fillna("") if not historical and DRAFT_PROGRESS_PATH.exists()
                  else pd.DataFrame())

complete_ids = set(progress.loc[progress.annotation_status.eq("complete"), "document_id"].astype(str))
mode = st.sidebar.radio("Show", ["All", "Pending only", "Completed only"])
visible = sample.copy()
if mode == "Pending only":
    visible = visible[~visible.document_id.astype(str).isin(complete_ids)]
elif mode == "Completed only":
    visible = visible[visible.document_id.astype(str).isin(complete_ids)]
if visible.empty:
    st.success("No documents match this filter.")
    st.stop()

position = st.sidebar.number_input("Document in current view", 1, len(visible), 1) - 1
row = visible.iloc[position]
document_id = str(row.document_id)
st.sidebar.write(f"Researcher-confirmed: {len(complete_ids)} / {len(sample)}")
st.progress(len(complete_ids) / len(sample))
st.write({"sample_order": int(row.sample_order), "document_id": document_id,
          "topic": row.final_topic_name, "language": row.primary_language,
          "event_year": row.event_year, "source": row.source})
st.text_area("Original caption", str(row.original_text), height=260, disabled=True)

saved = [r for r in review_rows if str(r["document_id"]) == document_id]
draft = [r for r in draft_rows if str(r["document_id"]) == document_id]
draft_none = False
if not draft_progress.empty:
    match = draft_progress[draft_progress.document_id.astype(str).eq(document_id)]
    draft_none = bool(not match.empty and str(match.iloc[0].get("ai_draft_no_evaluative_aspect_mention", "")).lower() == "true")
base_rows = saved if saved else draft
locked = document_id in complete_ids
if locked:
    decision = progress.loc[progress.document_id.astype(str).eq(document_id), "review_decision"].iloc[0]
    st.success(f"Researcher-confirmed and locked: {decision}")
elif draft or draft_none:
    st.info(f"AI draft loaded: {len(draft)} mention(s). Review every field before confirming.")
else:
    st.info("No AI draft is available for this record.")

with st.expander("Frozen aspect guide"):
    for aspect_id in aspects:
        st.markdown(f"**{aspect_id}** — {aspect_help[aspect_id]}")

default_none = (str(progress.loc[progress.document_id.astype(str).eq(document_id),
                                 "no_evaluative_aspect_mention"].iloc[0]).lower() == "true"
                if document_id in complete_ids else draft_none)
no_mention = st.checkbox("No evaluative aspect mention", value=default_none,
                         key=f"none-{document_id}", disabled=locked)
count = st.number_input("Mention rows", 0, 12, len(base_rows) or (0 if no_mention else 1),
                        disabled=no_mention or locked, key=f"count-{document_id}")
new_rows = []
for i in range(int(count)):
    old = base_rows[i] if i < len(base_rows) else {}
    with st.expander(f"Mention {i + 1}", expanded=True):
        aspect = st.selectbox("Aspect", aspects,
                              index=aspects.index(old.get("aspect")) if old.get("aspect") in aspects else 0,
                              key=f"a-{document_id}-{i}", disabled=locked)
        st.caption(aspect_help[aspect])
        target = st.text_input("Evaluated target", old.get("target", ""), key=f"t-{document_id}-{i}", disabled=locked)
        sentiments = ["positive", "negative", "neutral", "mixed"]
        sentiment = st.selectbox("Sentiment", sentiments,
                                 index=sentiments.index(old.get("sentiment")) if old.get("sentiment") in sentiments else 0,
                                 key=f"s-{document_id}-{i}", disabled=locked)
        evidence = st.text_area("Exact evidence text", old.get("evidence_text", ""),
                                key=f"e-{document_id}-{i}", disabled=locked)
        expressions = ["explicit", "implicit"]
        expression = st.selectbox("Expression", expressions,
                                  index=expressions.index(old.get("expression_type")) if old.get("expression_type") in expressions else 0,
                                  key=f"x-{document_id}-{i}", disabled=locked)
        note = st.text_input("Optional note", old.get("analysis_notes", ""), key=f"n-{document_id}-{i}", disabled=locked)
        if evidence and evidence not in str(row.original_text):
            st.error("Evidence must occur exactly in the original caption.")
        start = str(row.original_text).find(evidence) if evidence else -1
        new_rows.append({"mention_id": stable_mention_id(document_id, i, aspect, evidence),
            "document_id": document_id, "aspect": aspect, "target": target, "sentiment": sentiment,
            "evidence_text": evidence, "evidence_start": start,
            "evidence_end": start + len(evidence) if start >= 0 else -1,
            "expression_type": expression, "language": old.get("language", row.primary_language),
            "english_gloss": old.get("english_gloss", ""),
            "contributing_hashtags": old.get("contributing_hashtags", ""),
            "contributing_emoji": old.get("contributing_emoji", ""),
            "emerging_aspect_name": old.get("emerging_aspect_name", ""), "analysis_notes": note})

options = ["accept_ai_draft_unchanged", "accept_with_edits", "replace_ai_draft", "confirm_no_mention"]
decision = st.selectbox("Researcher decision", options, disabled=locked,
                        index=3 if no_mention else 0, key=f"decision-{document_id}")
if st.button("Confirm, save, and lock document", type="primary", disabled=locked):
    invalid = (not no_mention and (not new_rows or any(not r["target"].strip() or not r["evidence_text"]
                                                       or r["evidence_start"] < 0 for r in new_rows)))
    if invalid:
        st.error("Every mention needs a target and exact evidence, or select no evaluative aspect mention.")
    elif no_mention and decision != "confirm_no_mention":
        st.error("Choose confirm_no_mention when no evaluative mention is selected.")
    elif not no_mention and decision == "confirm_no_mention":
        st.error("confirm_no_mention requires the no-mention checkbox.")
    else:
        save_document_annotations(document_id, [] if no_mention else new_rows, REVIEW_PATH, PROGRESS_PATH)
        progress = progress[progress.document_id.astype(str) != document_id]
        progress = pd.concat([progress, pd.DataFrame([{"document_id": document_id,
            "annotation_status": "complete", "no_evaluative_aspect_mention": bool(no_mention),
            "review_decision": decision, "reviewed_at_utc": datetime.now(timezone.utc).isoformat()}])], ignore_index=True)
        progress.to_csv(PROGRESS_PATH, index=False, encoding="utf-8-sig")
        st.success("Researcher decision saved and locked. Change the document number to continue.")
