"""Organizer-facing display of the verified finalized participant-experience release."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from .participant_experience_dashboard_data import (
    CATEGORIES, ROOT, filter_themes, load_participant_experience_dashboard_data,
)


TITLE = "Participant Experience"
PROVENANCE = "AI-assisted substantive review; not independent human validation."
PAGE_SIZE = 12
COVERAGE = {
    "cross_source": "Instagram + long-form review evidence",
    "instagram_only": "Instagram evidence only",
    "blog_only_emergent": "Long-form review evidence only",
}


@st.cache_data(show_spinner=False)
def _load_release(root_text: str) -> dict:
    """Cache a strict, read-only verification of the finalized release."""
    return load_participant_experience_dashboard_data(Path(root_text))


def _claim(claims: dict, field: str) -> str:
    return claims[field]["text"]


def _matches_keyword(theme: dict, claims: dict, query: str) -> bool:
    if not query:
        return True
    text = " ".join([theme["theme_label"], theme["aspect_label"], *[
        value["text"] for value in claims.values()
    ]]).casefold()
    return query.casefold() in text


def _theme_card(theme: dict, claims: dict) -> None:
    title = f"{theme['aspect_label']} · {theme['theme_label']}"
    with st.expander(title):
        st.caption(f"{theme['experience_classification']} · {COVERAGE[theme['source_coverage']]}")
        st.markdown("**Participant experience**")
        st.write(_claim(claims, "participant_experience_summary"))
        columns = st.columns(3)
        for column, label, field in zip(columns, ("What participants valued", "What participants struggled with", "Mixed or varied experience"),
                                        ("positive_experience_summary", "negative_experience_summary", "mixed_experience_summary")):
            with column:
                st.markdown(f"**{label}**")
                st.write(_claim(claims, field))
        st.markdown("**What organizers should pay attention to**")
        st.write(_claim(claims, "organizer_insight"))
        st.markdown("**Possible organizer consideration**")
        st.write(_claim(claims, "organizer_implication"))
        st.caption(_claim(claims, "evidence_scope_note"))
        st.caption(
            f"Instagram support: {theme['instagram_support_documents']} theme documents; "
            f"long-form support: {theme['blog_support_reviews']} parent reviews. These are separate source units."
        )
        with st.expander("View supporting evidence"):
            st.caption("Examples support the synthesis; they are not the complete participant population.")
            for evidence in theme["representative_evidence"]:
                source = "Instagram" if evidence["source"] == "instagram" else "Long-form review"
                st.markdown(f"**{source} · {evidence['sentiment'].title()}**")
                st.write(evidence["text"])
                if evidence.get("english_gloss") and evidence["english_gloss"] != evidence["text"]:
                    st.caption("Existing English gloss: " + evidence["english_gloss"])
                st.caption(f"Evidence ID: {evidence['evidence_id']} · Source context: {evidence['parent_id']}")


def render_participant_experience(root: Path = ROOT) -> None:
    st.title(TITLE)
    st.write("What participants valued, struggled with, and experienced around KLSCM, synthesized from finalized Instagram and long-form review evidence.")
    try:
        data = _load_release(str(Path(root)))
    except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
        st.error("The verified finalized participant-experience release could not be loaded. Draft candidates and reviews are never displayed here.")
        with st.expander("Validation detail"):
            st.text(str(exc))
        return

    package = data["evidence"]
    themes = package["themes"]
    insights = {item["theme_id"]: item["claims"] for item in data["insights"]}
    source_counts = {coverage: sum(theme["source_coverage"] == coverage for theme in themes) for coverage in COVERAGE}
    metrics = st.columns(4)
    metrics[0].metric("Finalized themes", len(themes), "Verified release")
    metrics[1].metric("Instagram-only", source_counts["instagram_only"], "Separate source coverage")
    metrics[2].metric("Cross-source", source_counts["cross_source"], "Observed in both sources")
    metrics[3].metric("Long-form-only", source_counts["blog_only_emergent"], "Emergent review themes")
    st.caption("Theme counts help navigation only. They do not estimate how many participants had an experience or rank its importance.")

    with st.expander("About this analysis"):
        st.write("Instagram and long-form reviews are separate evidence sources. Themes summarize observed feedback, are descriptive rather than causal, and source coverage is not statistical representativeness. Quantitative ABSA charts elsewhere show sentiment of extracted aspect mentions; they do not measure the share of happy participants.")
        st.caption(PROVENANCE)
        st.caption(package["model_limitation"])

    st.subheader("Explore participant experience")
    st.write("Browse finalized themes by aspect, experience classification, source coverage, or a keyword. No ranking is applied.")
    filters = st.columns(4)
    with filters[0]:
        aspect = st.selectbox("Aspect", [None, *package["aspects"]], format_func=lambda value: "All aspects" if value is None else package["aspects"][value], key="experience_aspect")
    with filters[1]:
        category = st.selectbox("Experience category", [None, *CATEGORIES], format_func=lambda value: "All categories" if value is None else value, key="experience_category")
    with filters[2]:
        coverage = st.selectbox("Source coverage", [None, *COVERAGE], format_func=lambda value: "All source coverage" if value is None else COVERAGE[value], key="experience_source")
    with filters[3]:
        keyword = st.text_input("Search themes", placeholder="e.g. hydration, training, volunteers", key="experience_search")
    selected = [theme for theme in filter_themes(themes, aspect, category, coverage)
                if _matches_keyword(theme, insights[theme["theme_id"]], keyword.strip())]
    st.caption(f"Showing {len(selected)} of {len(themes)} finalized themes.")
    if not selected:
        st.info("No finalized theme matches these filters. This does not mean the experience was absent from the broader participant population.")
        return

    with st.expander("Experience categories", expanded=False):
        for category in CATEGORIES:
            count = sum(theme["experience_classification"] == category for theme in themes)
            st.write(f"**{category}:** {count} themes. Counts are navigation aids, not participant percentages.")
    ordered = sorted(selected, key=lambda item: (item["aspect_label"], item["theme_label"]))
    pages = range(1, (len(ordered) - 1) // PAGE_SIZE + 2)
    page = st.selectbox("Results page", pages, key="experience_results_page")
    start = (page - 1) * PAGE_SIZE
    st.caption(f"Displaying themes {start + 1}–{min(start + PAGE_SIZE, len(ordered))} on this page.")
    for theme in ordered[start:start + PAGE_SIZE]:
        _theme_card(theme, insights[theme["theme_id"]])
