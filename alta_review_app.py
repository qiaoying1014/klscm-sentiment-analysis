from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from marathon_absa import aspect_level_themes_review as review
from marathon_absa.alta_review_ui import (
    EDITABLE_COLUMNS,
    aspect_order,
    clusters_for_aspect,
    finalization_ready,
    load_review_data,
    merge_options,
    progress_summary,
    save_review_row,
    validation_issues,
)


REVIEW_ROOT = review.OUTPUT_ROOT
MAPPING_PATH = REVIEW_ROOT / "cluster_review_mapping.csv"
DECISION_ORDER = [
    "KEEP", "RENAME", "MERGE", "UNCLEAR_OTHER",
    "EXCLUDE_FROM_INTERPRETATION",
]
QUALITY_ORDER = ["coherent", "somewhat_mixed", "highly_mixed"]


def _field_key(cluster_key: str, field: str) -> str:
    return f"alta_review::{cluster_key}::{field}"


def _initialize_fields(row: pd.Series) -> None:
    defaults = {
        "theme_quality": row.theme_quality or QUALITY_ORDER[0],
        "review_decision": row.review_decision or DECISION_ORDER[0],
        "researcher_theme_label": row.researcher_theme_label,
        "merge_target_theme_key": row.merge_target_theme_key,
        "researcher_notes": row.researcher_notes,
    }
    for field, value in defaults.items():
        st.session_state.setdefault(_field_key(row.cluster_key, field), value)


def _draft_values(cluster_key: str) -> dict[str, str]:
    return {
        field: str(st.session_state.get(_field_key(cluster_key, field), ""))
        for field in EDITABLE_COLUMNS if field != "review_status"
    }


def _is_dirty(row: pd.Series) -> bool:
    draft = _draft_values(row.cluster_key)
    persisted = {field: str(row[field]) for field in draft}
    if row.review_status != "reviewed":
        initial = {
            "theme_quality": QUALITY_ORDER[0],
            "review_decision": DECISION_ORDER[0],
            "researcher_theme_label": "",
            "merge_target_theme_key": "",
            "researcher_notes": "",
        }
        return draft != initial
    return draft != persisted


def _save(row: pd.Series) -> bool:
    try:
        _, backup = save_review_row(
            MAPPING_PATH,
            row.cluster_key,
            _draft_values(row.cluster_key),
            review_root=REVIEW_ROOT,
            create_backup=not st.session_state.get("alta_backup_created", False),
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(str(exc))
        return False
    st.session_state.alta_backup_created = True
    if backup:
        st.session_state.alta_backup_path = str(backup)
    st.session_state.alta_save_message = f"Saved review for {row.cluster_key}."
    return True


def _set_cluster(cluster_key: str) -> None:
    st.session_state.alta_cluster_key = cluster_key


def _metric_row(summary: dict, aspect_frame: pd.DataFrame, aspect: str) -> None:
    aspect_reviewed = int(aspect_frame.review_status.eq("reviewed").sum())
    columns = st.columns(6)
    columns[0].metric("Machine clusters", summary["total"])
    columns[1].metric("Reviewed", summary["reviewed"])
    columns[2].metric("Pending", summary["pending"])
    columns[3].metric("Complete", f"{summary['percent']:.1f}%")
    columns[4].metric("Current aspect", aspect)
    columns[5].metric("Aspect progress", f"{aspect_reviewed} / {len(aspect_frame)}")
    st.progress(summary["reviewed"] / summary["total"] if summary["total"] else 0)


def _render_progress(mapping: pd.DataFrame, joined: pd.DataFrame) -> None:
    summary = progress_summary(mapping)
    st.subheader("Review Progress")
    st.dataframe(summary["by_aspect"], hide_index=True, width="stretch")
    left, right = st.columns(2)
    decisions = pd.DataFrame([
        {"decision": name, "count": summary["decisions"].get(name, 0)}
        for name in DECISION_ORDER
    ])
    qualities = pd.DataFrame([
        {"theme_quality": name, "count": summary["qualities"].get(name, 0)}
        for name in QUALITY_ORDER
    ])
    left.caption("Workflow decision counts (not research findings)")
    left.dataframe(decisions, hide_index=True, width="stretch")
    right.caption("Workflow quality counts (not research findings)")
    right.dataframe(qualities, hide_index=True, width="stretch")


def _render_reference_sections() -> None:
    st.subheader("Insufficient-Support Aspects")
    st.info(
        "ALTA v1 did not identify sufficient support for stable within-aspect "
        "semantic decomposition under the frozen support criteria."
    )
    path = REVIEW_ROOT / "insufficient_support_summary.csv"
    if path.exists():
        frame = pd.read_csv(path, keep_default_na=False, encoding="utf-8-sig")
        st.dataframe(frame, hide_index=True, width="stretch")
    st.subheader("Noise Summary")
    st.info(
        "Noise represents mentions not assigned to a stable within-aspect semantic "
        "cluster by HDBSCAN. Noise is retained and is not manually redistributed "
        "during researcher theme review."
    )
    path = REVIEW_ROOT / "noise_summary.csv"
    if path.exists():
        st.dataframe(
            pd.read_csv(path, keep_default_na=False, encoding="utf-8-sig"),
            hide_index=True, width="stretch",
        )


def _render_validation(mapping: pd.DataFrame) -> None:
    st.subheader("Validate and Finalize")
    summary = progress_summary(mapping)
    ready, ready_message = finalization_ready(mapping)
    left, right, status = st.columns(3)
    left.metric("Reviewed", f"{summary['reviewed']} / {summary['total']}")
    right.metric("Pending", summary["pending"])
    status.metric("Finalization status", "Ready" if ready else "Not ready")
    if st.button("Validate Review", type="primary"):
        try:
            result = review.validate_review_package(MAPPING_PATH)
            issues = validation_issues(mapping)
            if result["complete"]:
                st.success("Review mapping validation passed.")
            else:
                st.warning("The mapping is structurally valid but incomplete.")
                for issue in issues:
                    st.write(f"- {issue}")
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            st.error(f"Review mapping validation failed: {exc}")
    st.divider()
    st.caption(ready_message)
    confirmed = st.checkbox(
        "I confirm that I have reviewed all cluster decisions and want to generate "
        "the finalized researcher-reviewed taxonomy.",
        disabled=not ready,
    )
    if st.button(
        "Finalize Reviewed Taxonomy", disabled=not (ready and confirmed), type="primary"
    ):
        try:
            result = review.finalize_reviewed_taxonomy(MAPPING_PATH)
            manifest_path = REVIEW_ROOT / "review_manifest.json"
            st.success("Researcher-reviewed taxonomy finalized successfully.")
            st.json({
                "final_reviewed_theme_count": result["reviewed_themes"],
                "output_path": result["output_root"],
                "manifest_path": str(manifest_path),
                "decision_summary": summary["decisions"],
                "frozen_alta_v1_modified": False,
            })
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            st.error(f"Finalization failed: {exc}")


def _render_review(joined: pd.DataFrame, mapping: pd.DataFrame) -> None:
    aspects = aspect_order(joined)
    with st.sidebar:
        st.header("Review navigation")
        aspect = st.selectbox("Aspect", aspects, key="alta_aspect")
        status_filter = st.selectbox("Review status", ["Pending", "All", "Reviewed"])
        decision_filter = st.selectbox("Decision", ["All"] + DECISION_ORDER)
        quality_filter = st.selectbox("Theme quality", ["All"] + QUALITY_ORDER)
    aspect_frame = clusters_for_aspect(joined, aspect)
    filtered = aspect_frame.copy()
    if status_filter != "All":
        filtered = filtered[filtered.review_status.eq(status_filter.lower())]
    if decision_filter != "All":
        filtered = filtered[filtered.review_decision.eq(decision_filter)]
    if quality_filter != "All":
        filtered = filtered[filtered.theme_quality.eq(quality_filter)]
    if filtered.empty:
        st.warning("No clusters match the current filters.")
        return
    keys = filtered.cluster_key.tolist()
    current_key = st.session_state.get("alta_cluster_key")
    if current_key not in keys:
        current_key = keys[0]
        _set_cluster(current_key)
    labels = {
        row.cluster_key: f"Cluster {int(row.theme_id)} — {row.provisional_label} — {int(row.support_documents)} docs"
        for row in filtered.itertuples()
    }
    with st.sidebar:
        selected = st.selectbox(
            "Cluster", keys, index=keys.index(current_key), format_func=labels.get,
            key=f"alta_cluster_picker::{aspect}::{current_key}",
        )
        if selected != current_key:
            _set_cluster(selected)
            current_key = selected
    row = joined.loc[joined.cluster_key.eq(current_key)].iloc[0]
    _initialize_fields(row)
    cluster_index = aspect_frame.index[aspect_frame.cluster_key.eq(current_key)][0]
    _metric_row(progress_summary(mapping), aspect_frame, aspect)
    st.caption(f"Current cluster: Cluster {cluster_index + 1} of {len(aspect_frame)}")
    if message := st.session_state.pop("alta_save_message", None):
        st.success(message)
    if _is_dirty(row):
        st.warning("Unsaved changes. Save this review before relying on it as persisted research data.")

    previous_key = aspect_frame.iloc[max(cluster_index - 1, 0)].cluster_key
    next_key = aspect_frame.iloc[min(cluster_index + 1, len(aspect_frame) - 1)].cluster_key
    with st.sidebar:
        nav_left, nav_right = st.columns(2)
        if nav_left.button("Previous", disabled=cluster_index == 0):
            _set_cluster(previous_key); st.rerun()
        if nav_right.button("Next", disabled=cluster_index == len(aspect_frame) - 1):
            _set_cluster(next_key); st.rerun()

    st.subheader(row.provisional_label)
    st.caption(f"{row.aspect} · theme_id {int(row.theme_id)} · {row.cluster_key}")
    cards = st.columns(4)
    cards[0].metric("Support documents", int(row.support_documents))
    cards[1].metric("Support mentions", int(row.support_mentions))
    cards[2].metric("Share within aspect", f"{100 * float(row.share_of_aspect_documents):.1f}%")
    cards[3].metric("Years present", row.years_present)
    sentiment = pd.DataFrame({
        "sentiment": ["Positive", "Negative", "Mixed", "Neutral"],
        "share": [row.positive_share, row.negative_share, row.mixed_share, row.neutral_share],
    })
    sentiment["share"] = sentiment["share"].map(lambda value: f"{100 * float(value):.1f}%")
    st.dataframe(sentiment, hide_index=True, width="stretch")

    st.subheader("Representative Evidence")
    evidence_ids = [item.strip() for item in str(row.representative_evidence_ids).split("|")]
    for number in range(1, 6):
        evidence = str(row.get(f"representative_evidence_{number}", ""))
        gloss = str(row.get(f"representative_gloss_{number}", ""))
        document_id = str(row.get(f"representative_document_id_{number}", ""))
        with st.expander(f"Example {number}", expanded=number <= 2):
            st.markdown("**Original evidence (frozen source evidence)**")
            st.write(evidence)
            st.markdown("**English gloss (translation aid; not source evidence)**")
            st.write(gloss or "Not available")
            mention_id = evidence_ids[number - 1] if number <= len(evidence_ids) else ""
            st.caption(f"Document ID: {document_id} · Mention ID: {mention_id}")

    with st.expander("Machine-generated interpretation aids"):
        st.write(f"**Provisional machine label:** {row.provisional_label}")
        st.write(f"**Top targets:** {row.top_targets}")
        st.write(f"**Top keyphrases:** {row.top_keyphrases}")
        st.warning(
            "Machine-generated lexical summaries are interpretation aids. Base "
            "researcher decisions primarily on the underlying evidence."
        )
    with st.expander("How to review this cluster"):
        st.markdown(
            "**KEEP** — coherent and already described adequately.  \n"
            "**RENAME** — coherent, but the machine label is awkward, lexical, narrow, or unsuitable for reporting.  \n"
            "**MERGE** — essentially the same discussion as another cluster within this aspect.  \n"
            "**UNCLEAR_OTHER** — meaningfully mixed or unable to support a defensible standalone interpretation.  \n"
            "**EXCLUDE_FROM_INTERPRETATION** — artifacts, duplicated generic material, irrelevant content, or otherwise unsuitable evidence.\n\n"
            "Theme labels should describe **what is discussed**, not the conclusion or sentiment. "
            "For example, prefer “Hills and course elevation” over “The route is too difficult”."
        )

    st.subheader("Researcher Review")
    quality = st.radio(
        "Theme quality", QUALITY_ORDER,
        key=_field_key(current_key, "theme_quality"), horizontal=True,
    )
    decision = st.radio(
        "Review decision", DECISION_ORDER,
        key=_field_key(current_key, "review_decision"), horizontal=True,
    )
    st.text_input(
        "Researcher theme label",
        key=_field_key(current_key, "researcher_theme_label"),
        help=(
            "Use a short, descriptive, neutral, concept-focused label understandable "
            "to nontechnical readers. RENAME requires a label; KEEP may retain the provisional label."
        ),
    )
    if decision == "MERGE":
        options = merge_options(joined, current_key)
        option_keys = options.cluster_key.tolist()
        option_labels = {
            item.cluster_key: f"Cluster {int(item.theme_id)} — {item.provisional_label} — {int(item.support_documents)} docs"
            for item in options.itertuples()
        }
        stored = st.session_state.get(_field_key(current_key, "merge_target_theme_key"), "")
        index = option_keys.index(stored) if stored in option_keys else None
        st.selectbox(
            "Merge target (same aspect only)", option_keys, index=index,
            placeholder="Choose another cluster", format_func=option_labels.get,
            key=_field_key(current_key, "merge_target_theme_key"),
        )
    else:
        st.session_state[_field_key(current_key, "merge_target_theme_key")] = ""
    st.text_area(
        "Researcher notes", key=_field_key(current_key, "researcher_notes"),
        placeholder=(
            "Evidence consistently concerns hilly sections/elevation; provisional "
            "label is lexical, so renamed for reporting clarity."
        ),
    )
    save_col, save_next_col = st.columns(2)
    if save_col.button("Save Review", type="primary", width="stretch"):
        if _save(row): st.rerun()
    if save_next_col.button("Save & Next", type="primary", width="stretch"):
        if _save(row):
            _, refreshed_mapping, refreshed = load_review_data(REVIEW_ROOT)
            remaining = clusters_for_aspect(refreshed, aspect)
            pending = remaining[remaining.review_status.ne("reviewed")]
            if not pending.empty:
                _set_cluster(pending.iloc[0].cluster_key)
            elif cluster_index < len(aspect_frame) - 1:
                _set_cluster(next_key)
                st.session_state.alta_save_message += " Aspect review complete."
            else:
                st.session_state.alta_save_message += " Aspect review complete."
            st.rerun()

    st.subheader("Other clusters in this aspect")
    comparison = aspect_frame[aspect_frame.cluster_key.ne(current_key)][[
        "theme_id", "provisional_label", "support_documents", "theme_quality",
        "review_decision", "researcher_theme_label",
    ]]
    st.dataframe(comparison, hide_index=True, width="stretch")
    with st.expander("Comparison evidence previews"):
        for other in aspect_frame[aspect_frame.cluster_key.ne(current_key)].itertuples():
            st.markdown(f"**Cluster {other.theme_id} — {other.provisional_label}**")
            st.write(str(getattr(other, "representative_evidence_1", "")))


def render_app() -> None:
    st.set_page_config(page_title="KLSCM ALTA Researcher Review", page_icon="🔎", layout="wide")
    st.title("KLSCM ALTA Researcher Review")
    st.caption(
        "Review machine-induced within-aspect semantic clusters and convert them "
        "into researcher-reviewed discussion themes."
    )
    st.info(
        "ALTA themes are derived from model-estimated ABSA mentions. Researcher "
        "review improves interpretability of semantic clusters but does not validate "
        "or correct the upstream ABSA classifier."
    )
    with st.expander("Method limitation and upstream development performance"):
        st.write(review.LIMITATION)
        st.write("Aspect precision ≈ 0.513 · Aspect recall ≈ 0.790")
    try:
        _, mapping, joined = load_review_data(REVIEW_ROOT)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not load ALTA review artifacts: {exc}")
        st.stop()
    page = st.sidebar.radio(
        "Section", ["Cluster Review", "Review Progress", "Insufficient Support & Noise", "Validate & Finalize"]
    )
    st.sidebar.caption(f"Reads `{REVIEW_ROOT / 'cluster_review_workbook.csv'}`")
    st.sidebar.caption(f"Writes `{MAPPING_PATH}` only during review saves")
    if backup := st.session_state.get("alta_backup_path"):
        st.sidebar.caption(f"Session backup: `{backup}`")
    if page == "Cluster Review":
        _render_review(joined, mapping)
    elif page == "Review Progress":
        _render_progress(mapping, joined)
    elif page == "Insufficient Support & Noise":
        _render_reference_sections()
    else:
        _render_validation(mapping)
    with st.sidebar.expander("Export"):
        st.download_button(
            "Download current mapping CSV", MAPPING_PATH.read_bytes(),
            file_name=MAPPING_PATH.name, mime="text/csv",
        )
        summary_csv = progress_summary(mapping)["by_aspect"].to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Download progress summary", summary_csv,
            file_name="alta_review_progress.csv", mime="text/csv",
        )


if __name__ == "__main__":
    render_app()
