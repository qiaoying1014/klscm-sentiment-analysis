from __future__ import annotations

import html
import io
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud

from marathon_absa.dashboard_data import DashboardData, display_label_map, load_dashboard_data
from marathon_absa.reviewed_theme_dashboard_data import (
    ReviewedThemeDashboardData, load_reviewed_theme_dashboard_data,
)
from marathon_absa.participant_experience_dashboard_data import ROOT as PARTICIPANT_EXPERIENCE_ROOT
from marathon_absa.participant_experience_page import _load_release, render_participant_experience
from marathon_absa.wordcloud_data import (
    evidence_expression_table, evidence_frequency_table, filter_document_ids, filter_evidence_mentions,
    frequency_table, load_wordcloud_sources, tokenize_documents, tokenize_evidence_mentions, top_items,
)

SENTIMENT_COLORS = {"positive": "#159a69", "negative": "#d84a4a", "mixed": "#7557d9", "neutral": "#d59b22"}
ACCENT = "#2563eb"
INK = "#13243a"
GRID = "#dfe6ef"

st.set_page_config(page_title="KLSCM ABSA Research Dashboard", page_icon="K", layout="wide",
                   initial_sidebar_state="expanded")


@st.cache_data(show_spinner=False)
def get_data() -> DashboardData:
    return load_dashboard_data()


@st.cache_data(show_spinner=False)
def get_reviewed_theme_data() -> ReviewedThemeDashboardData:
    return load_reviewed_theme_dashboard_data()


def pct(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}%}"


def plot_style(figure: go.Figure, height: int = 430) -> go.Figure:
    figure.update_layout(
        height=height, margin=dict(l=12, r=18, t=54, b=24), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Arial, sans-serif", color=INK),
        title=dict(font=dict(size=17, color=INK)), legend_title_text="", hoverlabel=dict(font_size=13),
        legend=dict(font=dict(color=INK)),
    )
    figure.update_xaxes(gridcolor=GRID, zeroline=False, tickfont=dict(color=INK), title_font=dict(color=INK))
    figure.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(color=INK), title_font=dict(color=INK))
    return figure


def badge(text: str, tone: str = "default") -> str:
    return f'<span class="badge badge-{tone}">{html.escape(text)}</span>'


def support_text(value: str) -> str:
    return {"high_support": "High support", "moderate_support": "Moderate support",
            "low_support": "Limited support", "very_low_support": "Very limited support"}.get(value, value)


def effect_text(value: str) -> str:
    return {"small": "Small year association", "trivial": "Trivial year association",
            "moderate": "Moderate year association", "strong": "Strong year association",
            "not_tested": "Not formally tested"}.get(value, value)


def metric_card(label: str, value: str, detail: str) -> None:
    st.markdown(f"<div class='metric-card'><p>{html.escape(label)}</p><strong>{html.escape(value)}</strong><span>{html.escape(detail)}</span></div>", unsafe_allow_html=True)


def executive_metric_card(icon: str, label: str, value: str, detail: str, tone: str = "blue") -> None:
    st.markdown(
        f"<div class='executive-metric executive-metric-{html.escape(tone)}'>"
        f"<span class='metric-icon' aria-hidden='true'>{html.escape(icon)}</span>"
        f"<div><p>{html.escape(label)}</p><strong>{html.escape(value)}</strong><small>{html.escape(detail)}</small></div></div>",
        unsafe_allow_html=True,
    )


def page_header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(f"<div class='page-kicker'>{html.escape(kicker)}</div><h1 class='page-title'>{html.escape(title)}</h1><p class='page-subtitle'>{html.escape(subtitle)}</p>", unsafe_allow_html=True)


def section_title(number: str, title: str, subtitle: str) -> None:
    st.markdown(f"<div class='section-label'><span>{html.escape(number)}</span><div><h2>{html.escape(title)}</h2><p>{html.escape(subtitle)}</p></div></div>", unsafe_allow_html=True)


def notice(title: str, body: str, kind: str = "info") -> None:
    st.markdown(f"<div class='notice notice-{kind}' role='note'><strong>{html.escape(title)}</strong><p>{html.escape(body)}</p></div>", unsafe_allow_html=True)


def sentiment_chart(frame: pd.DataFrame, title: str, share_column: str, count_column: str) -> go.Figure:
    shown = frame.copy(); shown["label"] = shown.sentiment.str.title()
    figure = go.Figure()
    for row in shown.itertuples():
        figure.add_trace(go.Bar(name=row.label, y=["Mentions"], x=[getattr(row, share_column)], orientation="h",
            marker_color=SENTIMENT_COLORS[row.sentiment], customdata=[[getattr(row, count_column)]],
            hovertemplate=f"{row.label}: %{{x:.1%}}<br>Count: %{{customdata[0]:,}}<extra></extra>"))
    figure.update_layout(barmode="stack", xaxis_tickformat=".0%", xaxis_range=[0, 1], title=title)
    return plot_style(figure, 250)


def aspect_prevalence_chart(frame: pd.DataFrame, metric: str) -> go.Figure:
    shown = frame.sort_values(metric, ascending=True).copy()
    shown["support"] = shown.support_class.map(support_text)
    if metric == "document_prevalence_all":
        figure = px.bar(shown, x=metric, y="display_label", orientation="h", color="recommended_for_emphasis",
            color_discrete_map={True: ACCENT, False: "#9eafb2"},
            custom_data=["affected_document_count", "support", "effect_category"])
        figure.update_traces(hovertemplate="%{y}<br>%{x:.1%} of posts<br>%{customdata[0]:,} documents<br>%{customdata[1]}<br>%{customdata[2]} effect<extra></extra>")
        figure.update_xaxes(tickformat=".0%", title="Share of analyzed posts")
        title = "Share of analyzed posts containing each aspect"
    else:
        figure = px.bar(shown, x=metric, y="display_label", orientation="h", color="recommended_for_emphasis",
            color_discrete_map={True: ACCENT, False: "#9eafb2"}, custom_data=["support"])
        figure.update_traces(hovertemplate="%{y}<br>%{x:,} model-estimated mentions<br>%{customdata[0]}<extra></extra>")
        figure.update_xaxes(title="Model-estimated aspect mentions")
        title = "Model-estimated mention count by aspect"
    figure.update_layout(title=title, showlegend=False)
    return plot_style(figure, 650)


def aspect_sentiment_chart(frame: pd.DataFrame) -> go.Figure:
    pivot = frame.pivot(index="display_label", columns="sentiment", values="share_within_aspect").fillna(0)
    order = frame.groupby("display_label").mention_count.sum().sort_values().index
    figure = go.Figure()
    for sentiment in ["positive", "negative", "mixed", "neutral"]:
        figure.add_trace(go.Bar(name=sentiment.title(), y=order, x=pivot.loc[order, sentiment], orientation="h",
                                marker_color=SENTIMENT_COLORS[sentiment], hovertemplate=f"{sentiment.title()}: %{{x:.1%}}<extra></extra>"))
    figure.update_layout(barmode="stack", title="Sentiment composition within each aspect")
    figure.update_xaxes(tickformat=".0%", range=[0, 1], title="Share of model-estimated mentions within aspect")
    return plot_style(figure, 650)


def reviewed_theme_prevalence_chart(frame: pd.DataFrame) -> go.Figure:
    shown = frame.sort_values("support_documents", ascending=True)
    figure = px.bar(
        shown, x="support_documents", y="reviewed_theme_label", orientation="h",
        custom_data=["share_of_aspect_documents", "support_mentions"],
    )
    figure.update_traces(
        marker_color=ACCENT,
        hovertemplate="%{y}<br>%{x:,} unique documents<br>%{customdata[0]:.1%} of aspect-bearing documents<br>%{customdata[1]:,} mentions<extra></extra>",
    )
    figure.update_xaxes(title="Unique supporting documents", rangemode="tozero")
    figure.update_yaxes(title=None, automargin=True)
    return plot_style(figure, max(300, 58 * len(shown) + 90))


def render_reviewed_themes(themes: ReviewedThemeDashboardData, aspect: str, display_label: str) -> None:
    section_title("03", "What participants are talking about", "Finalized researcher-reviewed discussion themes within the selected ABSA aspect")
    insufficient = themes.insufficient_support[themes.insufficient_support.aspect.eq(aspect)]
    if not insufficient.empty:
        row = insufficient.iloc[0]
        notice(
            "Insufficient support for stable within-aspect themes",
            f"Stable aspect-level discussion themes were not produced for this aspect because its {int(row.support_mentions):,} mentions across {int(row.support_documents):,} documents did not meet the frozen ALTA minimum-support criteria. Insufficient thematic support does not imply that the aspect is unimportant.",
            "info",
        )
        return
    selected = themes.summary[themes.summary.aspect.eq(aspect)].sort_values("support_documents", ascending=False)
    if selected.empty:
        notice("Reviewed-theme data unavailable", "The finalized reviewed-theme package contains no displayable taxonomy for this aspect.", "warning")
        return
    st.caption("Themes are ordered by unique-document support, not by managerial importance.")
    st.plotly_chart(reviewed_theme_prevalence_chart(selected), width="stretch", key=f"reviewed_theme_overview_{aspect}")
    with st.expander("How were these discussion themes identified?"):
        st.markdown(
            "Discussion themes were derived downstream from frozen mention-level ABSA evidence, embedded locally and clustered separately within each aspect. The 101 machine-induced clusters were then reviewed, renamed, merged or excluded by the researcher, producing 64 finalized discussion themes. Document-level prevalence is primary; evidence uses exact grounded ABSA spans. These displays are descriptive and introduce no new inferential tests."
        )
        st.warning("Reviewed discussion themes remain downstream of model-estimated ABSA assignments and therefore inherit uncertainty from the upstream classifier (development precision 0.513; recall 0.790). Researcher review did not correct every possible upstream classification error.")
        st.caption("BERTopic topics describe broad corpus-wide discourse; ABSA aspects identify evaluated marathon-experience elements; reviewed ALTA themes describe what is discussed within a selected aspect.")
    for row in selected.itertuples():
        heading = f"{row.reviewed_theme_label} · {int(row.support_documents):,} documents · {row.share_of_aspect_documents:.1%} within aspect"
        with st.expander(heading):
            cols = st.columns(3)
            with cols[0]: metric_card("Supporting documents", f"{int(row.support_documents):,}", "Unique document presence")
            with cols[1]: metric_card("Within aspect", f"{row.share_of_aspect_documents:.1%}", f"of {display_label} documents")
            with cols[2]: metric_card("Supporting mentions", f"{int(row.support_mentions):,}", "Secondary descriptive count")
            st.markdown("**Document-level sentiment composition**")
            sentiment = themes.sentiment[themes.sentiment.reviewed_theme_id.eq(row.reviewed_theme_id)].set_index("document_sentiment")
            sentiment_cols = st.columns(4)
            for column, name in zip(sentiment_cols, ["positive", "negative", "mixed", "neutral"]):
                value = sentiment.loc[name]
                with column: st.metric(name.title(), f"{value.share_of_theme_documents:.1%}", f"{int(value.support_documents):,} documents")
            st.markdown("**Across observed editions**")
            years = themes.years[themes.years.reviewed_theme_id.eq(row.reviewed_theme_id)].sort_values("year")
            year_table = years.rename(columns={"year": "Edition", "support_documents": "Supporting documents", "aspect_documents_that_year": "Aspect documents", "within_aspect_theme_prevalence": "Within-aspect prevalence"})
            st.dataframe(year_table[["Edition", "Supporting documents", "Aspect documents", "Within-aspect prevalence"]], hide_index=True, width="stretch", column_config={"Within-aspect prevalence": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
            st.caption("Edition values are descriptive and do not represent a new inferential test.")
            st.markdown("**Representative participant evidence**")
            evidence = themes.evidence[themes.evidence.reviewed_theme_id.eq(row.reviewed_theme_id)].sort_values("mention_id")
            for index, item in enumerate(evidence.itertuples(), 1):
                st.markdown(f"**Evidence {index} · {int(item.event_year)} · {html.escape(str(item.sentiment).title())}**")
                st.markdown(f"> {html.escape(str(item.evidence_text))}")
                if str(item.english_gloss).strip():
                    st.caption(f"English gloss (not a verbatim quotation): {item.english_gloss}")
                metadata = [str(item.primary_language)]
                if str(item.target).strip(): metadata.append(f"Target: {item.target}")
                st.caption(" · ".join(metadata))
    notice("Scope", "These are descriptive reviewed discussion themes only. Researcher perceptions and participant interview propositions have not yet been developed.", "info")
    st.caption("Some model-estimated aspect mentions were not assigned to stable ALTA semantic clusters and are excluded from the reviewed theme taxonomy.")


def trend_chart(frame: pd.DataFrame, selected_aspects: list[str], labels: dict[str, str]) -> go.Figure:
    shown = frame[frame.aspect.isin(selected_aspects)].sort_values("year")
    figure = go.Figure()
    palette = ["#087f8c", "#c75d3d", "#536f8d", "#8d6a9f", "#8a7d35"]
    for index, (aspect, group) in enumerate(shown.groupby("aspect", sort=False)):
        color = palette[index % len(palette)]
        has_ci = group.document_prevalence_ci_lower.notna().all()
        error = dict(type="data", symmetric=False,
            array=group.document_prevalence_ci_upper-group.document_prevalence,
            arrayminus=group.document_prevalence-group.document_prevalence_ci_lower, thickness=1.2, width=4) if has_ci else None
        figure.add_trace(go.Scatter(x=group.year, y=group.document_prevalence, name=labels[aspect], mode="lines+markers",
            line=dict(color=color, width=3), marker=dict(size=8), error_y=error,
            customdata=group[["affected_documents", "total_documents_year"]],
            hovertemplate="%{x}: %{y:.1%}<br>%{customdata[0]:,} / %{customdata[1]:,} posts<extra></extra>"))
    figure.update_layout(title="Document prevalence across observed KLSCM editions", hovermode="x unified")
    figure.update_xaxes(tickmode="array", tickvals=[2019, 2023, 2024, 2025], title="Event edition")
    figure.update_yaxes(tickformat=".0%", rangemode="tozero", title="Share of analyzed posts")
    return plot_style(figure, 460)


def compact_style(figure: go.Figure, height: int, left_margin: int = 8) -> go.Figure:
    plot_style(figure, height)
    figure.update_layout(
        margin=dict(l=left_margin, r=8, t=34, b=14),
        title=None,
        font=dict(family="Arial, sans-serif", color=INK, size=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font_size=10),
        hoverlabel=dict(font_size=11),
    )
    return figure


def render_executive_overview(data: DashboardData) -> None:
    labels = display_label_map(data)
    st.markdown(
        "<div class='executive-header'><div class='executive-kicker'>Frozen research synthesis</div>"
        "<div class='executive-heading-row'><div><h1>Standard Chartered Kuala Lumpur Marathon</h1>"
        "<h2>Sentiment &amp; Perception Analysis Dashboard</h2></div>"
        "<div class='executive-editions'><span>Observed editions</span><strong>2019 · 2023 · 2024 · 2025</strong></div></div>"
        "<p>One-page research brief covering corpus scale, aspect sentiment, edition-level patterns, topics, languages and evidence quality.</p></div>",
        unsafe_allow_html=True,
    )
    try:
        experience_release = _load_release(str(PARTICIPANT_EXPERIENCE_ROOT))
        experience_cols = st.columns([1, 3, 1])
        with experience_cols[0]:
            st.metric("Participant experience", len(experience_release["insights"]), "Verified finalized themes")
        with experience_cols[1]:
            st.markdown("**Participant Experience**  ")
            st.caption("Read the finalized, evidence-grounded synthesis of what participants valued, struggled with and experienced across Instagram and long-form reviews. Theme counts are not participant prevalence or priority rankings.")
        with experience_cols[2]:
            st.button("Explore Participant Experience", key="open_participant_experience",
                      on_click=lambda: st.session_state.update({"dashboard_page": "Participant Experience & Organizer Insights"}))
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        st.caption("Participant Experience is available only when its verified finalized release passes integrity checks.")

    corpus = data.corpus
    positive = float(data.sentiment_overall.loc[data.sentiment_overall.sentiment.eq("positive"), "mention_share"].iloc[0])
    negative = float(data.sentiment_overall.loc[data.sentiment_overall.sentiment.eq("negative"), "mention_share"].iloc[0])
    kpis = [
        ("📚", "Analyzed Posts", f"{corpus['total_documents']:,}", "Substantive posts", "blue"),
        ("🎯", "Posts With Aspects", f"{corpus['mention_bearing_documents']:,}", f"{corpus['mention_bearing_documents']/corpus['total_documents']:.1%} of posts", "blue"),
        ("🧩", "Aspect Mentions", f"{corpus['total_mentions']:,}", "Model-estimated", "blue"),
        ("👍", "Positive Mentions", f"{positive:.1%}", "of aspect mentions", "positive"),
        ("👎", "Negative Mentions", f"{negative:.1%}", "of aspect mentions", "negative"),
        ("🔀", "Multi-Aspect Posts", f"{corpus['multi_aspect_documents']:,}", f"{corpus['multi_aspect_rate_all_documents']:.1%} of posts", "mixed"),
    ]
    st.markdown("<div class='executive-section-label'>Research at a glance</div>", unsafe_allow_html=True)
    for start in (0, 3):
        for column, values in zip(st.columns(3, gap="small"), kpis[start:start + 3]):
            with column:
                executive_metric_card(*values)

    st.markdown("<div class='executive-section-label'>Sentiment and discussion</div>", unsafe_allow_html=True)
    row2 = st.columns([.85, 1.15], gap="small")
    with row2[0], st.container(border=True):
        st.markdown("<h3 class='compact-title'>Overall Aspect Sentiment</h3>", unsafe_allow_html=True)
        sentiment = data.sentiment_overall.copy()
        figure = go.Figure(go.Pie(
            labels=sentiment.sentiment.str.title(), values=sentiment.mention_count, hole=.64,
            marker_colors=[SENTIMENT_COLORS[value] for value in sentiment.sentiment],
            customdata=sentiment.mention_share,
            hovertemplate="%{label}<br>%{value:,} mentions<br>%{customdata:.1%}<extra></extra>",
            textinfo="percent", textfont_size=10,
        ))
        figure.update_layout(showlegend=True)
        st.plotly_chart(compact_style(figure, 250), width="stretch", key="executive_sentiment")
        st.caption("Share of 15,486 model-estimated aspect mentions")
    with row2[1], st.container(border=True):
        st.markdown("<h3 class='compact-title'>Top Discussed Aspects</h3>", unsafe_allow_html=True)
        compact_labels = {"Emotional And Overall Event Experience": "Emotional Experience",
                          "Training, Preparation & Pacing": "Training & Preparation"}
        top = data.aspects.nlargest(6, "document_prevalence_all").sort_values("document_prevalence_all").copy()
        top["compact_label"] = top.display_label.replace(compact_labels)
        figure = px.bar(top, x="document_prevalence_all", y="compact_label", orientation="h", text="document_prevalence_all",
                        custom_data=["affected_document_count"])
        figure.update_traces(marker_color=ACCENT, texttemplate="%{x:.1%}", textposition="outside",
                             hovertemplate="%{y}<br>%{x:.1%} of posts<br>%{customdata[0]:,} documents<extra></extra>")
        figure.update_xaxes(tickformat=".0%", range=[0, float(top.document_prevalence_all.max()) * 1.18], title=None)
        figure.update_yaxes(title=None, tickfont=dict(size=9, color=INK))
        st.plotly_chart(compact_style(figure, 250, 118), width="stretch", key="executive_aspects")
        st.caption("Document prevalence among 7,704 analyzed posts")
    st.markdown("<div class='executive-section-label'>Aspect composition and principal trends</div>", unsafe_allow_html=True)
    row3 = st.columns([1.1, 1], gap="small")
    with row3[0], st.container(border=True):
        st.markdown("<h3 class='compact-title'>Aspect-Based Sentiment</h3>", unsafe_allow_html=True)
        aspects = data.aspects.nlargest(5, "document_prevalence_all").aspect.tolist()
        shown = data.aspect_sentiment[data.aspect_sentiment.aspect.isin(aspects)].copy()
        shown["display_label"] = shown.display_label.replace(compact_labels)
        figure = aspect_sentiment_chart(shown)
        figure.update_xaxes(title=None); figure.update_yaxes(tickfont=dict(size=9, color=INK))
        st.plotly_chart(compact_style(figure, 250, 118), width="stretch", key="executive_aspect_sentiment")
        st.caption("Share of model-estimated mentions within each aspect")

    with row3[1], st.container(border=True):
        st.markdown("<h3 class='compact-title'>Principal Aspect Prevalence</h3>", unsafe_allow_html=True)
        principal = data.aspects.loc[data.aspects.recommended_for_emphasis, "aspect"].tolist()
        figure = trend_chart(data.year_aspect, principal, labels)
        figure.update_yaxes(title=None); figure.update_xaxes(title=None)
        st.plotly_chart(compact_style(figure, 260), width="stretch", key="executive_temporal")
        st.caption("Frozen document prevalence with Wilson 95% confidence intervals")
    st.markdown("<div class='executive-section-label'>Edition signal and evidence synthesis</div>", unsafe_allow_html=True)
    row4 = st.columns([1, 1.25], gap="small")
    with row4[0], st.container(border=True):
        st.markdown("<h3 class='compact-title'>Training &amp; Preparation Sentiment</h3>", unsafe_allow_html=True)
        training = data.year_aspect[data.year_aspect.aspect.eq("training_preparation_pacing")]
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=training.year, y=training.positive_document_share_within_aspect_year, name="Positive", mode="lines+markers", line=dict(color=SENTIMENT_COLORS["positive"], width=2.5)))
        figure.add_trace(go.Scatter(x=training.year, y=training.negative_document_share_within_aspect_year, name="Negative", mode="lines+markers", line=dict(color=SENTIMENT_COLORS["negative"], width=2.5)))
        figure.update_xaxes(tickmode="array", tickvals=[2019, 2023, 2024, 2025], title=None)
        figure.update_yaxes(tickformat=".0%", range=[0, .72], title=None)
        st.plotly_chart(compact_style(figure, 260), width="stretch", key="executive_training")
        st.caption("Document-level positive and negative presence may overlap.")
    with row4[1], st.container(border=True):
        sentiment_shares = data.sentiment_overall.set_index("sentiment").mention_share
        top_aspects = data.aspects.nlargest(3, "document_prevalence_all").display_label.tolist()
        insight_items = []
        for _, finding in data.findings.iterrows():
            if finding.finding_type == "aspect_prevalence":
                insight = f"{labels[finding.aspect]}: {pct(finding['2019_value'])} to {pct(finding['2025_value'])}"
            else:
                insight = f"{labels[finding.aspect]}: more positive and less negative discussion"
            insight_items.append(f"<li>{html.escape(insight)}</li>")
        st.markdown(
            "<div class='ai-summary'>"
            "<div class='ai-summary-heading'><h3>AI Summary &amp; Key Insights</h3>"
            + badge("Frozen research summary", "accent")
            + "</div><div class='ai-summary-source'>Generated from frozen analysis outputs</div>"
            + "<p class='ai-summary-overall'>Across "
            + f"{data.corpus['total_documents']:,} analyzed posts, positive sentiment was the largest share of "
            + f"{data.corpus['total_mentions']:,} model-estimated aspect mentions ({pct(sentiment_shares['positive'])}; "
            + f"negative {pct(sentiment_shares['negative'])}, mixed {pct(sentiment_shares['mixed'])}, "
            + f"neutral {pct(sentiment_shares['neutral'])}). The most prevalent substantive aspects were "
            + html.escape(", ".join(top_aspects[:-1]) + f", and {top_aspects[-1]}.")
            + "</p><ul class='ai-summary-insights'>" + "".join(insight_items) + "</ul>"
            + "<p class='ai-summary-takeaway'><strong>Takeaway.</strong> Later observed editions contained more "
            + "performance-, community-, and physical-experience-related discussion, while training and preparation "
            + "discussion became more positive and less negative.</p>"
            + f"<div class='ai-summary-quality'>Development evidence: precision {data.metadata['model_precision']:.3f} "
            + f"&middot; recall {data.metadata['model_recall']:.3f}</div>"
            + "<p class='ai-summary-caution'>Associations in model-estimated labels, not causal changes in participant "
            + "attitudes. " + html.escape(data.metadata["warnings"]["temporal"]) + "</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='executive-section-label'>Descriptive context and research scope</div>", unsafe_allow_html=True)
    row5 = st.columns([1.15, 1, .9], gap="small")
    with row5[0], st.container(border=True):
        st.markdown("<h3 class='compact-title'>Topic Snapshot " + badge("Descriptive only") + "</h3>", unsafe_allow_html=True)
        topic_ids = [1, 0, 11, 12]
        for row in data.topics.set_index("topic_id").loc[topic_ids].itertuples():
            st.markdown(f"<div class='snapshot-row'><span>{html.escape(row.topic_label)}</span><strong>{html.escape(labels[row.top_aspect_1])} {pct(row.top_aspect_1_document_prevalence)}</strong></div>", unsafe_allow_html=True)
    with row5[1], st.container(border=True):
        st.markdown("<h3 class='compact-title'>Language Snapshot " + badge("Descriptive only") + "</h3>", unsafe_allow_html=True)
        major_languages = ["English", "Malay", "Malay/Indonesian uncertain", "Chinese", "Indonesian"]
        language = data.languages.set_index("language").loc[major_languages]
        for name, row in language.iterrows():
            st.markdown(f"<div class='language-row'><span>{html.escape(name)}</span><strong>{int(row.documents):,}</strong><small>{row.mention_bearing_documents/row.documents:.1%} mention-bearing</small></div>", unsafe_allow_html=True)
    with row5[2], st.container(border=True):
        st.markdown("<h3 class='compact-title'>Research Quality &amp; Scope</h3>", unsafe_allow_html=True)
        quality = [("ABSA precision", "0.513"), ("ABSA recall", "0.790"), ("Human gold", "80 documents"), ("Aspect families", "20"), ("Final topics", "32")]
        st.markdown("<div class='quality-grid'>" + "".join(f"<div><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>" for label, value in quality) + "</div>", unsafe_allow_html=True)
        st.markdown("<p class='quality-warning'>Model-estimated labels; precision target of 0.55 was not met.</p><p class='method-link'>See Methodology in the navigation for full scope and limitations.</p>", unsafe_allow_html=True)

    notice("Interpret with care", "Results reflect model-estimated ABSA labels. Cross-year comparisons should be interpreted alongside changing corpus composition and increasing extraction density.", "warning")


def render_header(data: DashboardData) -> None:
    page_header("Frozen research dataset · ABSA V1", "Standard Chartered Kuala Lumpur Marathon",
                "Sentiment & Perception Analysis Dashboard · Social-media analysis across the 2019, 2023, 2024 and 2025 KLSCM editions")
    notice("Research quality notice", data.metadata["warnings"]["model"], "info")


def render_overview(data: DashboardData) -> None:
    render_header(data)
    section_title("01", "Research overview", "Frozen production corpus and overall model-estimated sentiment")
    selected = st.selectbox("Edition", ["All editions", 2019, 2023, 2024, 2025], key="overview_year")
    if selected == "All editions":
        docs, bearing, mentions, zero = data.corpus["total_documents"], data.corpus["mention_bearing_documents"], data.corpus["total_mentions"], data.corpus["zero_mention_documents"]
        sentiment = data.sentiment_overall[["sentiment", "mention_count", "mention_share"]]
        prevalence = data.aspects
        aspect_sentiment = data.aspect_sentiment
    else:
        year = data.years[data.years.year.eq(selected)].iloc[0]
        docs, bearing, mentions, zero = int(year.documents), int(year.mention_bearing_documents), int(year.mentions), int(year.zero_mention_documents)
        sentiment = pd.DataFrame({"sentiment": ["positive", "negative", "mixed", "neutral"],
            "mention_count": [year.positive_mentions, year.negative_mentions, year.mixed_mentions, year.neutral_mentions],
            "mention_share": [year.positive_share, year.negative_share, year.mixed_share, year.neutral_share]})
        prevalence = data.year_aspect[data.year_aspect.year.eq(selected)].merge(data.aspects[["aspect", "display_label", "support_class", "effect_category", "recommended_for_emphasis"]], on="aspect", suffixes=("", "_aspect"))
        prevalence = prevalence.rename(columns={"document_prevalence": "document_prevalence_all"})
        aspect_sentiment = data.year_aspect_sentiment[data.year_aspect_sentiment.year.eq(selected)].merge(data.aspects[["aspect", "display_label"]], on="aspect", suffixes=("", "_aspect"))
        aspect_sentiment = aspect_sentiment.rename(columns={"share_within_aspect_year": "share_within_aspect"})
    positive = float(sentiment.loc[sentiment.sentiment.eq("positive"), "mention_share"].iloc[0])
    negative = float(sentiment.loc[sentiment.sentiment.eq("negative"), "mention_share"].iloc[0])
    multi = data.corpus["multi_aspect_documents"] if selected == "All editions" else None
    columns = st.columns(6)
    with columns[0]: metric_card("Analyzed posts", f"{docs:,}", "Substantive production documents")
    with columns[1]: metric_card("Posts with aspects", f"{bearing:,}", f"{bearing/docs:.1%} contained ≥1 aspect")
    with columns[2]: metric_card("Aspect mentions", f"{mentions:,}", "Model-estimated mentions, not posts")
    with columns[3]: metric_card("Positive mentions", f"{positive:.1%}", "of model-estimated aspect mentions")
    with columns[4]: metric_card("Negative mentions", f"{negative:.1%}", "of model-estimated aspect mentions")
    with columns[5]: metric_card("Multi-aspect posts", f"{multi:,}" if multi is not None else "All-years only", "Posts containing >1 aspect")
    section_title("02", "Core social-media signals", "Sentiment, discussed aspects and within-aspect composition")
    left, right = st.columns([0.78, 1.22], gap="large")
    with left:
        with st.container(border=True):
            st.plotly_chart(sentiment_chart(sentiment, "Overall Aspect Sentiment", "mention_share", "mention_count"), width="stretch", key=f"sentiment_{selected}")
            st.caption(f"Share of {mentions:,} model-estimated aspect mentions. Counts are available in tooltips.")
    with right:
        with st.container(border=True):
            metric = st.radio("Ranking measure", ["Document prevalence", "Mention count"], horizontal=True)
            chart_metric = "document_prevalence_all" if metric == "Document prevalence" else "mention_count"
            st.plotly_chart(aspect_prevalence_chart(prevalence.head(5), chart_metric), width="stretch", key=f"prevalence_{selected}_{metric}")
            st.caption("Document prevalence = share of analyzed posts containing at least one mention of the aspect.")
    with st.container(border=True):
        major = ["race_performance", "crowd_community_atmosphere", "physical_experience", "emotional_experience", "training_preparation_pacing", "route_course", "photography_media"]
        st.plotly_chart(aspect_sentiment_chart(aspect_sentiment[aspect_sentiment.aspect.isin(major)]), width="stretch", key=f"aspect_sentiment_{selected}")
        st.caption("Denominator: model-estimated mentions within each aspect; not all posts.")
        with st.expander("Accessible aspect sentiment table"):
            st.dataframe(aspect_sentiment[aspect_sentiment.aspect.isin(major)][["display_label", "sentiment", "mention_count", "share_within_aspect"]], hide_index=True, width="stretch")
    section_title("03", "Principal aspect prevalence across editions", "Frozen document prevalence and Wilson 95% confidence intervals")
    with st.container(border=True):
        principal = data.aspects.loc[data.aspects.recommended_for_emphasis, "aspect"].tolist()
        st.plotly_chart(trend_chart(data.year_aspect, principal, display_label_map(data)), width="stretch", key="overview_temporal")
        st.caption(data.metadata["warnings"]["temporal"])
    section_title("04", "Research findings", "Curated frozen findings—not a live AI summary")
    cards = st.columns(2)
    for index, row in enumerate(data.findings.itertuples()):
        with cards[index % 2]:
            st.markdown(f"<article class='finding finding-card'><div>{badge(effect_text(row.effect_category), 'accent')} {badge(support_text(row.support_class))}</div><h3>{html.escape(row.headline)}</h3><p>{html.escape(row.short_interpretation)}</p></article>", unsafe_allow_html=True)
    with st.expander("Methodological cautions"):
        for row in data.cautions.itertuples(): st.markdown(f"**{str(row.item).replace('_',' ').title()}** — {row.reason}")


def render_temporal(data: DashboardData, labels: dict[str, str]) -> None:
    page_header("Edition-level evidence", "Temporal Trends", "Document-level comparisons across the four observed KLSCM editions")
    notice("Cross-edition context", data.metadata["warnings"]["temporal"], "warning")
    defaults = data.aspects.loc[data.aspects.recommended_for_emphasis, "aspect"].tolist()
    selected = st.multiselect("Aspects to compare", data.aspects.aspect.tolist(), default=defaults,
                              format_func=lambda value: labels[value], max_selections=5)
    if not selected:
        st.info("Select at least one aspect to view its frozen document-prevalence estimates.")
    else:
        st.plotly_chart(trend_chart(data.year_aspect, selected, labels), width="stretch")
        st.caption("Error bars are frozen 95% confidence intervals where the aspect was in the inferential set.")
        evidence = data.aspects[data.aspects.aspect.isin(selected)][["display_label", "support_class", "effect_category", "year_association_adjusted_p", "cramers_v", "recommended_for_emphasis"]].copy()
        evidence["support_class"] = evidence.support_class.map(support_text); evidence["effect_category"] = evidence.effect_category.map(effect_text)
        with st.expander("Frozen evidence details"):
            st.dataframe(evidence, hide_index=True, width="stretch")
    st.subheader("Edition-level extraction context")
    context = data.years[["year", "mentions_per_document", "zero_mention_rate"]]
    st.dataframe(context, hide_index=True, width="stretch", column_config={"zero_mention_rate": st.column_config.NumberColumn(format="%.1%%")})

    st.subheader("Training, preparation & pacing sentiment")
    training = data.year_aspect[data.year_aspect.aspect.eq("training_preparation_pacing")]
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=training.year, y=training.positive_document_share_within_aspect_year, name="Positive present", mode="lines+markers", line=dict(color=SENTIMENT_COLORS["positive"], width=3)))
    figure.add_trace(go.Scatter(x=training.year, y=training.negative_document_share_within_aspect_year, name="Negative present", mode="lines+markers", line=dict(color=SENTIMENT_COLORS["negative"], width=3)))
    figure.update_layout(title="Independent sentiment presence among training/preparation-bearing posts")
    figure.update_xaxes(tickmode="array", tickvals=[2019, 2023, 2024, 2025]); figure.update_yaxes(tickformat=".0%", range=[0, .75])
    st.plotly_chart(plot_style(figure, 400), width="stretch")
    st.caption("Positive and negative percentages can overlap because one post may contain distinct sentiment propositions.")


def render_social_media(data: DashboardData) -> None:
    page_header("Descriptive corpus analytics", "Social Media Analytics", "Corpus scale, model extraction density and sentiment by observed edition")
    section_title("01", "Corpus by edition", "Frozen substantive posts and model-estimated aspect mentions")
    left, right = st.columns(2, gap="large")
    with left, st.container(border=True):
        figure = px.bar(data.years, x="year", y="documents", text="documents", custom_data=["mention_bearing_documents"])
        figure.update_traces(marker_color=ACCENT, textposition="outside", hovertemplate="%{x}<br>%{y:,} posts<br>%{customdata[0]:,} mention-bearing<extra></extra>")
        figure.update_layout(title="Analyzed posts by edition"); figure.update_xaxes(type="category", title="Event edition"); figure.update_yaxes(title="Analyzed posts")
        st.plotly_chart(plot_style(figure, 380), width="stretch")
    with right, st.container(border=True):
        figure = px.bar(data.years, x="year", y="mentions", text="mentions")
        figure.update_traces(marker_color="#4f7fd8", textposition="outside", hovertemplate="%{x}<br>%{y:,} model-estimated mentions<extra></extra>")
        figure.update_layout(title="Aspect mentions by edition"); figure.update_xaxes(type="category", title="Event edition"); figure.update_yaxes(title="Aspect mentions")
        st.plotly_chart(plot_style(figure, 380), width="stretch")
    section_title("02", "Extraction profile", "Zero-mention prevalence and model extraction density")
    with st.container(border=True):
        figure = go.Figure()
        figure.add_trace(go.Bar(x=data.years.year, y=data.years.zero_mention_rate, name="Zero-mention rate", marker_color="#d59b22", hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
        figure.add_trace(go.Scatter(x=data.years.year, y=data.years.mentions_per_document, name="Mentions per document", yaxis="y2", mode="lines+markers", line=dict(color=ACCENT,width=3), hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
        figure.update_layout(title="Zero-mention rate and mentions per document", yaxis=dict(title="Zero-mention rate",tickformat=".0%"), yaxis2=dict(title="Mentions per document",overlaying="y",side="right"), hovermode="x unified")
        figure.update_xaxes(type="category",title="Event edition")
        st.plotly_chart(plot_style(figure, 430), width="stretch")
    notice("Interpretation warning", "Extraction density increased across editions. Temporal differences should therefore not be interpreted solely as changes in participant attitudes.", "warning")
    with st.expander("Accessible edition table"):
        st.dataframe(data.years, hide_index=True, width="stretch")


def render_findings(data: DashboardData) -> None:
    page_header("Frozen evidence synthesis", "Research Findings", "Supported findings and explicit boundaries from the completed analysis")
    st.caption("Curated in the frozen research-findings mart; the dashboard does not generate interpretations.")
    for row in data.findings.itertuples():
        st.markdown(f"<article class='finding'><div>{badge(effect_text(row.effect_category), 'accent')} {badge(support_text(row.support_class))}</div><h3>{html.escape(row.headline)}</h3><p>{html.escape(row.short_interpretation)}</p><small>{html.escape(row.caution_note)}</small></article>", unsafe_allow_html=True)
    with st.expander("Findings to interpret cautiously"):
        for row in data.cautions.itertuples():
            st.markdown(f"**{str(row.item).replace('_', ' ').title()}**  \n{row.reason}  \n_{row.dashboard_behavior}_")


def render_aspect_explorer(data: DashboardData, labels: dict[str, str]) -> None:
    page_header("Twenty-family ABSA ontology", "Aspect Analysis", "Document prevalence, mention sentiment and frozen temporal evidence")
    section_title("01", "Aspect ranking", "All 20 aspect families remain visible, including lower-support categories")
    metric = st.radio("Ranking measure", ["Document prevalence", "Mention count"], horizontal=True, key="aspect_page_metric")
    st.plotly_chart(aspect_prevalence_chart(data.aspects, "document_prevalence_all" if metric=="Document prevalence" else "mention_count"), width="stretch")
    section_title("02", "Aspect detail", "Select an aspect to inspect its frozen descriptive and inferential record")
    aspect = st.selectbox("Aspect", data.aspects.aspect.tolist(), format_func=lambda value: labels[value], key="aspect_explorer")
    overview = data.aspects[data.aspects.aspect.eq(aspect)].iloc[0]
    st.markdown(f"<div class='section-heading'><div><h2>{html.escape(overview.display_label)}</h2><p>{overview.document_prevalence_all:.1%} of analyzed posts contained this model-estimated aspect.</p></div><div>{badge(support_text(overview.support_class))} {badge(effect_text(overview.effect_category), 'accent' if overview.recommended_for_emphasis else 'default')}</div></div>", unsafe_allow_html=True)
    cols = st.columns(3)
    with cols[0]: metric_card("Affected posts", f"{overview.affected_document_count:,}", f"{overview.document_prevalence_all:.1%} prevalence")
    with cols[1]: metric_card("Aspect mentions", f"{overview.mention_count:,}", "Model-estimated propositions")
    with cols[2]: metric_card("Mean mentions", f"{overview.mean_mentions_per_affected_document:.2f}", "Per affected post")
    if aspect == "weather_conditions":
        caution = data.cautions[data.cautions.item.eq("weather_conditions")].iloc[0]
        notice("Weather interpretation", caution.reason, "warning")
    if aspect == "photography_media":
        caution = data.cautions[data.cautions.item.eq("photography_media")].iloc[0]
        notice("Descriptive finding", caution.reason, "info")
    left, right = st.columns([.8, 1.2], gap="large")
    sentiment = data.aspect_sentiment[data.aspect_sentiment.aspect.eq(aspect)]
    with left:
        st.plotly_chart(sentiment_chart(sentiment.rename(columns={"share_within_aspect": "share"}), "Sentiment within aspect", "share", "mention_count"), width="stretch")
        st.caption("Denominator: model-estimated mentions within this aspect.")
    with right:
        st.plotly_chart(trend_chart(data.year_aspect, [aspect], labels), width="stretch")
    stats = data.year_aspect[data.year_aspect.aspect.eq(aspect)][["year", "affected_documents", "total_documents_year", "document_prevalence", "document_prevalence_ci_lower", "document_prevalence_ci_upper"]]
    with st.expander("Year estimates and frozen pairwise comparisons"):
        st.dataframe(stats, hide_index=True, width="stretch")
        pairs = data.pairwise[data.pairwise.aspect.eq(aspect)]
        if pairs.empty: st.caption("No frozen pairwise results are available for this aspect/outcome.")
        else: st.dataframe(pairs, hide_index=True, width="stretch")
    render_reviewed_themes(get_reviewed_theme_data(), aspect, overview.display_label)


def render_topic_explorer(data: DashboardData) -> None:
    page_header("Descriptive analysis", "Topic Analysis", "Aspect and sentiment patterns within the frozen 32-topic taxonomy")
    notice("Descriptive only", data.metadata["warnings"]["topic"], "info")
    st.caption("Topics describe broad corpus-wide discourse, while reviewed aspect themes describe what participants discuss within a specific ABSA aspect.")
    topic_ids = data.topics.sort_values("topic_id").topic_id.tolist()
    topic = st.selectbox("Final consolidated topic", topic_ids,
        format_func=lambda value: data.topics.loc[data.topics.topic_id.eq(value), "topic_label"].iloc[0])
    row = data.topics[data.topics.topic_id.eq(topic)].iloc[0]
    cols = st.columns(3)
    with cols[0]: metric_card("Topic documents", f"{row.document_count:,}", "Frozen substantive posts")
    with cols[1]: metric_card("Aspect mentions", f"{row.mention_count:,}", "Model-estimated mentions")
    with cols[2]: metric_card("Leading aspect", str(row.top_aspect_1).replace("_", " ").title(), f"{row.top_aspect_1_document_prevalence:.1%} of topic posts")
    topic_aspects = data.topic_aspect[data.topic_aspect.topic_id.eq(topic)].sort_values("document_prevalence", ascending=True)
    figure = px.bar(topic_aspects, x="document_prevalence", y="aspect_display_label", orientation="h",
                    custom_data=["affected_documents", "mention_count"])
    figure.update_traces(marker_color=ACCENT, hovertemplate="%{y}<br>%{x:.1%} of topic posts<br>%{customdata[0]:,} documents<br>%{customdata[1]:,} mentions<extra></extra>")
    figure.update_xaxes(tickformat=".0%", title="Share of topic posts"); figure.update_layout(title=row.topic_label)
    st.plotly_chart(plot_style(figure, 590), width="stretch")
    with st.expander("Accessible topic-aspect table"):
        st.dataframe(topic_aspects, hide_index=True, width="stretch")


def render_language_explorer(data: DashboardData) -> None:
    page_header("Descriptive analysis", "Language Analysis", "Frozen language categories, aspect prevalence and model-estimated sentiment")
    notice("Descriptive only", data.metadata["warnings"]["language"], "info")
    language = st.selectbox("Frozen language category", data.languages.sort_values("documents", ascending=False).language.tolist())
    row = data.languages[data.languages.language.eq(language)].iloc[0]
    if row.language_quality_warning:
        notice("Language-quality category", "This category reflects insufficient or undetermined text and is not a meaningful language group.", "warning")
    cols = st.columns(4)
    with cols[0]: metric_card("Documents", f"{row.documents:,}", "Frozen language assignment")
    with cols[1]: metric_card("Mention-bearing", f"{row.mention_bearing_documents:,}", f"{row.mention_bearing_documents/row.documents:.1%} of documents")
    with cols[2]: metric_card("Zero mention", f"{row.zero_mention_rate:.1%}", "No model-estimated aspect")
    with cols[3]: metric_card("Aspect mentions", f"{row.mentions:,}", "Model-estimated mentions")
    lang = data.language_aspect[data.language_aspect.language.eq(language)]
    aspects = lang.drop_duplicates("aspect").sort_values("document_prevalence", ascending=True)
    figure = px.bar(aspects, x="document_prevalence", y="aspect_display_label", orientation="h", custom_data=["affected_documents"])
    figure.update_traces(marker_color=ACCENT, hovertemplate="%{y}<br>%{x:.1%} of language-category posts<br>%{customdata[0]:,} documents<extra></extra>")
    figure.update_xaxes(tickformat=".0%", title="Document prevalence"); figure.update_layout(title=f"Aspect prevalence · {language}")
    st.plotly_chart(plot_style(figure, 590), width="stretch")
    with st.expander("Accessible language-aspect table"):
        st.dataframe(lang, hide_index=True, width="stretch")


def render_methodology(data: DashboardData) -> None:
    page_header("Research design", "Methodology & Limitations", "Data lineage, development evidence and frozen statistical procedures")
    section_title("01", "Analysis pipeline", "Completed stages represented in this dashboard")
    pipeline = ["Instagram collection", "Relevance filtering", "Multilingual BERTopic", "Final topic taxonomy",
        "20-family ABSA ontology", "80-document human ABSA gold", "V1 / V2 / V3 controlled development",
        "V3 production inference", "7,704 substantive posts", "Descriptive analysis",
        "Document-level inferential analysis", "Within-aspect ALTA clustering",
        "Researcher-reviewed theme taxonomy", "Dashboard"]
    st.markdown("<div class='pipeline'>"+"".join(f"<div><span>{i+1:02d}</span><strong>{html.escape(step)}</strong></div>" for i,step in enumerate(pipeline))+"</div>", unsafe_allow_html=True)
    st.caption("Parallel long-form branch: parent-review analysis, delegated AI-assisted theme review, and descriptive cross-source comparison. Its completed results appear in Cross-Source Analysis. The statistical procedures below apply to Instagram only.")
    section_title("02", "Research basis and development evidence", "The production system was selected with a known precision limitation")
    cols = st.columns(2, gap="large")
    with cols[0]:
        st.subheader("Research basis")
        st.markdown(f"""
- **Corpus:** {data.corpus['total_documents']:,} substantive KLSCM social-media posts
- **ABSA ontology:** {len(data.metadata['aspect_families'])} frozen aspect families
- **Production mentions:** {data.corpus['total_mentions']:,} model-estimated aspect mentions
- **Observed editions:** {', '.join(map(str, data.metadata['years']))}
- **Statistical unit:** {data.metadata['inferential_unit']}
- **Dataset:** `{data.metadata['dataset_version']}`
- **Reviewed thematic layer:** 64 finalized discussion themes across 16 sufficiently supported aspects
- **Theme prevalence unit:** unique documents within each aspect
""")
    with cols[1]:
        st.subheader("Development evidence")
        st.metric("Aspect precision", f"{data.metadata['model_precision']:.3f}", help="Frozen 80-document development result")
        st.metric("Aspect recall", f"{data.metadata['model_recall']:.3f}", help="Frozen 80-document development result")
        st.caption("The predeclared precision target was 0.55 and was not met. Production is not validation.")
    for title, key in [("Model warning", "model"), ("Statistical warning", "statistical"),
                       ("Temporal warning", "temporal"), ("Language warning", "language"), ("Topic warning", "topic")]:
        notice(title, data.metadata["warnings"][key], "warning" if key in {"model", "statistical", "temporal"} else "info")
    st.markdown("**Frozen statistical design:** document-level Pearson chi-square, Cramer's V, BH-FDR by analysis family and Holm-corrected pairwise comparisons. The dashboard only displays prepared outputs.")
    st.markdown("**Validation status:** The 80-document human sample was reused during prompt development and is development evidence rather than untouched confirmatory validation. Production predictions were not manually verified.")
    section_title("03", "Statistical methodology", "Displayed values are frozen preprocessing outputs")
    methods = [("Unit of inference","Document"),("Temporal grouping","Event edition"),("Confidence intervals","Wilson 95%"),("Omnibus association","Pearson chi-square"),("Effect size","Cramer's V"),("Multiple-comparison correction","Benjamini–Hochberg FDR"),("Pairwise correction","Holm")]
    st.markdown("<div class='method-grid'>"+"".join(f"<div><small>{html.escape(k)}</small><strong>{html.escape(v)}</strong></div>" for k,v in methods)+"</div>",unsafe_allow_html=True)


@st.cache_data(show_spinner="Preparing frozen caption and evidence tokens...")
def get_wordcloud_sources(config_version: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    documents, mentions = load_wordcloud_sources()
    return documents, mentions, tokenize_documents(documents), tokenize_evidence_mentions(mentions)


def _wordcloud_font() -> str | None:
    candidates = [Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"), Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf"), Path("C:/Windows/Fonts/arial.ttf")]
    return str(next((path for path in candidates if path.exists()), "")) or None


@st.cache_data(show_spinner=False)
def wordcloud_png(frequencies: tuple[tuple[str, int], ...], palette: str, width: int = 1000, height: int = 430) -> bytes:
    colors = {
        "overall": ["#0b3b66", "#124e78", "#176b87", "#1d5f9e", "#263f73"],
        "positive": ["#0b5d3b", "#14734b", "#1b8055", "#28643f", "#367a4e"],
        "negative": ["#8f2634", "#a8323e", "#b64145", "#7d2430", "#963b45"],
    }[palette]
    cloud = WordCloud(
        width=width, height=height, background_color="white", prefer_horizontal=.88,
        max_words=120, min_font_size=11, relative_scaling=.45, random_state=37,
        collocations=False, font_path=_wordcloud_font(), margin=3,
        color_func=lambda word, font_size, position, orientation, random_state, **kwargs: random_state.choice(colors),
    ).generate_from_frequencies(dict(frequencies))
    buffer = io.BytesIO(); cloud.to_image().save(buffer, format="PNG")
    return buffer.getvalue()


def render_cross_source(data: DashboardData) -> None:
    page_header("Separate corpora · descriptive triangulation", "Cross-Source Analysis",
                "Finalized long-form reviews alongside the frozen Instagram analysis")
    from marathon_absa.blog_dashboard_data import load_blog_dashboard_data
    try:
        blog = load_blog_dashboard_data()
    except FileNotFoundError:
        st.warning("Finalized long-form review files are unavailable. Complete the blog review and cross-source stages before displaying findings.")
        return
    except (ValueError, KeyError, RuntimeError):
        st.error("Long-form review data failed consistency checks. Refresh the finalized cross-source outputs before displaying findings.")
        return
    n = blog.counts
    st.success("Long-form review analysis and theme review are finalized.")
    left, right = st.columns(2)
    with left:
        st.metric("Instagram documents", f"{data.corpus['total_documents']:,}")
        st.caption("Frozen social-media analysis; its charts and statistical results remain separate.")
    with right:
        st.metric("Long-form parent reviews", str(n['reviews']))
        st.caption(f"{n['chunks']} chunks used only for inference; prevalence counts unique parent reviews.")
    for column, label, value in zip(st.columns(4),
            ["Blog aspect mentions", "Aspects in both sources", "Matched Instagram themes", "Blog-emergent themes"],
            [str(n['mentions']), f"{n['aspects']}/20", f"{n['matched']}/{n['reference_themes']}", str(n['emergent'])]):
        column.metric(label, value)
    st.caption("Blog theme review was AI-assisted and explicitly delegated by the user; it is not independent human validation.")
    labels = display_label_map(data)
    aspect = st.selectbox("Blog / cross-source aspect", ["All", *blog.aspects.aspect.tolist()],
                          format_func=lambda x: "All aspects" if x == "All" else labels.get(x, x), key="blog_aspect")
    st.caption("The aspect filter applies to the tables below. Corpus totals above remain fixed.")
    def selected(frame):
        return frame.copy() if aspect == "All" else frame[frame.aspect.eq(aspect)].copy()
    display = selected(blog.aspects)
    display["Aspect"] = display.aspect.map(labels)
    display["Instagram support"] = display.apply(lambda r: f"{int(r.instagram_support_documents):,}/7,704 posts ({r.instagram_prevalence:.1%})", axis=1)
    display["Blog support"] = display.apply(lambda r: f"{int(r.blog_support_reviews)}/25 reviews ({r.blog_prevalence:.1%})", axis=1)
    st.subheader("Aspect comparison")
    st.dataframe(display[["Aspect", "Instagram support", "Blog support"]], hide_index=True, width="stretch")
    st.caption("Separate denominators. Percentages are descriptive and are not directly comparable population estimates; no pooled prevalence or source-comparison significance tests are calculated.")

    st.subheader("Blog aspect sentiment")
    sentiment = selected(blog.sentiment)
    sentiment["Aspect"] = sentiment.aspect.map(labels)
    sentiment["Sentiment"] = sentiment.review_aspect_sentiment.str.title()
    sentiment["Support"] = sentiment.support_reviews.map(lambda x: f"{x}/25 reviews")
    st.dataframe(sentiment[["Aspect", "Sentiment", "Support"]], hide_index=True, width="stretch")
    st.caption("Sentiment is aggregated within each parent review and aspect. Repeated chunks do not add reviews; a review may discuss several aspects. These are model-estimated classifications.")

    st.subheader("Instagram-derived themes observed in blogs")
    st.caption(f"{n['matched']} of {n['reference_themes']} frozen Instagram themes have blog support; {n['reference_themes']-n['matched']} have none in this sample. Emergent themes are listed separately.")
    matched = selected(blog.matched)
    status = st.radio("Matched-theme visibility", ["All reference themes", "Observed in blogs", "No blog support"], horizontal=True, key="blog_matches")
    if status != "All reference themes":
        matched = matched[matched.present_in_blogs.eq(status == "Observed in blogs")]
    matched["Aspect"] = matched.aspect.map(labels)
    matched["Theme"] = matched.reviewed_theme_label
    matched["Instagram support"] = matched.instagram_support_documents.map(lambda x: f"{x:,}/7,704 posts")
    matched["Blog support"] = matched.blog_support_reviews.map(lambda x: f"{x}/25 reviews")
    st.dataframe(matched[["Aspect", "Theme", "Instagram support", "Blog support"]], hide_index=True, width="stretch")

    st.subheader("Blog-emergent themes")
    st.caption("Blog-emergent themes represent concepts observed in the sampled long-form reviews that were not aligned to the frozen Instagram-derived reviewed theme taxonomy. Support is counted by unique parent review.")
    emergent = selected(blog.emergent).sort_values(["support_reviews", "theme_label"], ascending=[False, True])
    emergent["Aspect"] = emergent.aspect.map(labels)
    emergent["Theme"] = emergent.theme_label
    emergent["Support"] = emergent.support_reviews.map(lambda x: f"{x}/25 reviews")
    emergent["Interpretation"] = emergent.singleton_flag.map({True: "Singleton · cautious interpretation", False: "Observed in multiple sampled reviews"})
    st.dataframe(emergent[["Aspect", "Theme", "Support", "Interpretation"]], hide_index=True, width="stretch")
    st.caption(f"{n['singletons']} of {n['emergent']} emergent themes occur in one sampled review. Low support means fewer than two parent reviews and is not an exclusion rule.")
    if not emergent.empty:
        theme_id = st.selectbox("Emergent theme evidence", emergent.theme_id.tolist(),
            format_func=lambda x: emergent.set_index('theme_id').loc[x, 'theme_label'], key="blog_evidence")
        theme = emergent.set_index('theme_id').loc[theme_id]
        with st.expander("Representative evidence and scope", expanded=True):
            st.write(theme.representative_evidence)
            st.caption(theme.scope_note)
    with st.expander("Long-form sample composition"):
        left, right = st.columns(2)
        left.dataframe(blog.years.rename(columns={"event_year":"Review year", "support_reviews":"Parent reviews"}), hide_index=True, width="stretch")
        right.dataframe(blog.languages.rename(columns={"primary_language":"Language", "support_reviews":"Parent reviews"}), hide_index=True, width="stretch")
        st.caption("These counts describe the 25 sampled parent reviews; they are not Instagram year or language totals.")
    st.subheader("Methodological boundary")
    st.markdown("Instagram and long-form reviews differ in document length and sampling frame. Blogs were analyzed separately using the frozen ABSA ontology and Instagram-derived taxonomy as a reference. Blog alignment and emergent review were completed through delegated AI-assisted interpretation. The 35 blog-emergent themes are complementary to the frozen 64-theme Instagram taxonomy; they do not extend or change it.")
    st.warning("Blog collection provenance remains incompletely documented. The inherited ABSA development precision was approximately 0.513 and recall 0.790. Theme review does not correct upstream ABSA classification error.")


def render_cloud(frame: pd.DataFrame, weighting: str, palette: str, key: str) -> None:
    column = "document_count" if weighting == "Document Frequency" else "count"
    frequencies = tuple((row.token, int(getattr(row, column))) for row in frame.head(120).itertuples())
    if not frequencies:
        st.info("No tokens meet the minimum document-frequency threshold for this selection.")
        return
    st.image(wordcloud_png(frequencies, palette), width="stretch")


def render_word_cloud(data: DashboardData) -> None:
    page_header("Descriptive text exploration only", "Word Cloud", "Descriptive lexical patterns in frozen KLSCM social-media captions")
    notice("Interpretation boundary", "Word frequency is descriptive and can be influenced by repeated hashtags, captions, language structure, and posting style.", "info")
    documents, mentions, tokenized, evidence = get_wordcloud_sources("wordcloud_v1_config_3")
    labels = display_label_map(data)
    text_source = st.radio("Text Source", ["ABSA Evidence", "Full Caption Context"], horizontal=True, key="wc_source")
    filters = st.columns(5)
    with filters[0]: year = st.selectbox("Year", ["All", 2019, 2023, 2024, 2025], key="wc_year")
    with filters[1]: aspect = st.selectbox("Aspect", ["All", *data.aspects.aspect.tolist()], format_func=lambda value: "All" if value == "All" else labels[value], key="wc_aspect")
    with filters[2]: sentiment = st.selectbox("Sentiment", ["All", "positive", "negative", "mixed", "neutral"], format_func=str.title, key="wc_sentiment")
    with filters[3]: topic = st.selectbox("Final Topic", ["All", *sorted(documents.final_consolidated_topic_label.unique())], key="wc_topic")
    with filters[4]: language = st.selectbox("Language", ["All", *sorted(documents.primary_language.unique())], key="wc_language")
    weighting = st.radio("Cloud weighting", ["Document Frequency", "Raw Frequency"], horizontal=True, key="wc_weighting")

    evidence_mode = text_source == "ABSA Evidence"
    if evidence_mode:
        selected_evidence = filter_evidence_mentions(evidence, year, aspect, sentiment, topic, language)
        selected_ids = set(selected_evidence.document_id)
        frequencies = evidence_frequency_table(selected_evidence)
        nonempty_evidence = selected_evidence.evidence_text.str.strip().ne("")
        summary = st.columns(4)
        with summary[0]: metric_card("Matching mentions", f"{len(selected_evidence):,}", "Frozen ABSA evidence rows")
        with summary[1]: metric_card("Unique documents", f"{selected_evidence.document_id.nunique():,}", "Documents contributing evidence")
        with summary[2]: metric_card("Unique evidence spans", f"{selected_evidence.loc[nonempty_evidence, 'evidence_text'].nunique():,}", "Non-empty exact frozen evidence text")
        with summary[3]: metric_card("Unique tokens", f"{len(frequencies):,}", "Minimum document frequency of two")
        main_title = "Filtered ABSA Evidence Cloud"
        context_copy = ("Only exact frozen evidence spans from matching ABSA mention rows are tokenized. "
                        "Sentiment belongs to the aspect-level expression, not necessarily to every individual token.")
        missing_evidence = int((~nonempty_evidence).sum())
    else:
        selected_ids = filter_document_ids(documents, mentions, year, aspect, sentiment, topic, language)
        selected_evidence = pd.DataFrame()
        frequencies = frequency_table(tokenized, selected_ids)
        summary = st.columns(2)
        with summary[0]: metric_card("Matching documents", f"{len(selected_ids):,}", "Entire captions after mention-based filtering")
        with summary[1]: metric_card("Unique tokens", f"{len(frequencies):,}", "Minimum document frequency of two")
        main_title = "Filtered Caption Context"
        context_copy = ("Caption-context clouds show vocabulary from entire posts containing matching ABSA mentions. "
                        "Individual words are not necessarily sentiment-bearing.")
    st.caption(context_copy)
    if evidence_mode and missing_evidence:
        st.caption(f"{missing_evidence:,} matching frozen mention rows have no evidence text and contribute no cloud tokens or expressions.")

    main, table = st.columns([1.55, 1], gap="large")
    with main, st.container(border=True):
        st.subheader(main_title)
        render_cloud(frequencies, weighting, "overall", "filtered")
        st.caption("Word size uses unique matching documents containing the token by default.")
    with table, st.container(border=True):
        st.subheader("Top Terms")
        top = frequencies.head(25).copy()
        top.insert(0, "Rank", range(1, len(top) + 1))
        top["Document Share"] = top.document_count / max(len(selected_ids), 1)
        top = top.rename(columns={"token": "Term", "document_count": "Document Count", "count": "Raw Frequency", "evidence_count": "Evidence Count"})
        columns = ["Rank", "Term", "Document Count", "Document Share"] + (["Evidence Count"] if evidence_mode else []) + ["Raw Frequency"]
        st.dataframe(top[columns], hide_index=True, width="stretch",
                     column_config={"Document Share": st.column_config.NumberColumn(format="%.1%%")})

    if sentiment == "All":
        comparison = st.columns(2, gap="large")
        if evidence_mode:
            positive_rows = filter_evidence_mentions(evidence, year, aspect, "positive", topic, language)
            negative_rows = filter_evidence_mentions(evidence, year, aspect, "negative", topic, language)
            with comparison[0], st.container(border=True):
                st.subheader("Positive Sentiment Expressions")
                render_cloud(evidence_frequency_table(positive_rows), weighting, "positive", "positive")
                st.caption(f"{len(positive_rows):,} exact positive evidence spans across {positive_rows.document_id.nunique():,} documents.")
            with comparison[1], st.container(border=True):
                st.subheader("Negative Sentiment Expressions")
                render_cloud(evidence_frequency_table(negative_rows), weighting, "negative", "negative")
                st.caption(f"{len(negative_rows):,} exact negative evidence spans across {negative_rows.document_id.nunique():,} documents.")
        else:
            positive_ids = filter_document_ids(documents, mentions, year, aspect, "positive", topic, language)
            negative_ids = filter_document_ids(documents, mentions, year, aspect, "negative", topic, language)
            with comparison[0], st.container(border=True):
                st.subheader("Positive-Containing Caption Context")
                render_cloud(frequency_table(tokenized, positive_ids), weighting, "positive", "positive")
                st.caption(f"Entire captions from {len(positive_ids):,} documents containing a matching positive ABSA mention.")
            with comparison[1], st.container(border=True):
                st.subheader("Negative-Containing Caption Context")
                render_cloud(frequency_table(tokenized, negative_ids), weighting, "negative", "negative")
                st.caption(f"Entire captions from {len(negative_ids):,} documents containing a matching negative ABSA mention.")

    if evidence_mode:
        st.subheader("Top Sentiment Expressions")
        expressions = evidence_expression_table(selected_evidence).head(20).copy()
        if len(expressions): expressions["Aspect"] = expressions.Aspect.map(labels)
        st.dataframe(expressions, hide_index=True, width="stretch")
        st.caption("Expressions are conservatively case-folded and whitespace-normalized for counting; displayed wording remains frozen evidence text.")
        st.info("Sentiment is assigned to an aspect-level expression, not necessarily to every individual token. For example, ‘training’ is neutral by itself but may occur in the negative expression ‘not enough training.’")

    source_rows = selected_evidence if evidence_mode else tokenized[tokenized.document_id.isin(selected_ids)]
    hashtags = top_items(source_rows, set(source_rows.document_id), "hashtags", 10)
    emojis = top_items(source_rows, set(source_rows.document_id), "emojis", 10)
    if hashtags or emojis:
        extras = st.columns(2)
        with extras[0]:
            st.markdown("**Meaningful hashtags**")
            st.caption(" · ".join(f"#{tag} ({count:,})" for tag, count in hashtags) or "None after boilerplate removal")
        with extras[1]:
            st.markdown("**Top emojis**")
            st.caption(" · ".join(f"{emoji} ({count:,})" for emoji, count in emojis) or "None")
    notice("Descriptive warning", "Word clouds summarize lexical frequency in the selected social-media captions. Word size reflects document frequency and should not be interpreted as sentiment strength, statistical significance, or aspect prevalence.", "warning")


st.markdown("""
<style>
:root{--navy:#0b1930;--navy2:#112746;--ink:#13243a;--muted:#65748a;--blue:#2563eb;--paper:#f3f6fa;--line:#dfe6ef;--card:#fff}
.stApp{background:var(--paper);color:var(--ink)}
[data-testid="stHeader"]{background:transparent}
[data-testid="stToolbarActions"],[data-testid="stStatusWidget"]{visibility:hidden}
[data-testid="stExpandSidebarButton"]{visibility:visible!important}
[data-testid="stAppViewContainer"] p,[data-testid="stAppViewContainer"] label,[data-testid="stAppViewContainer"] small,[data-testid="stAppViewContainer"] summary{color:#475569}
[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] p{color:#526277!important}
[data-testid="stAppViewContainer"] [data-baseweb="select"] *{color:#0f172a!important}
[data-testid="stAppViewContainer"] [data-baseweb="select"]{background:#fff}
[data-testid="stAppViewContainer"] [role="radiogroup"] label p{color:#26364b!important}
[data-testid="stAppViewContainer"] [data-testid="stExpander"] summary p{color:#26364b!important}
[data-testid="stAppViewContainer"] [data-testid="stDataFrame"]{color:#0f172a}
[data-testid="stSidebar"]{background:var(--navy);border-right:0}[data-testid="stSidebar"] *{color:#dce7f8}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label p,[data-testid="stSidebar"] small{color:#dce7f8!important}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p{color:#fff!important}
[data-testid="stSidebar"] [data-testid="stRadio"] label{padding:.46rem .65rem;border-radius:5px;margin:.08rem 0;font-size:.85rem}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked){background:#1d4f91;color:#fff}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{background:var(--navy2)}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{color:#aebed5}
.block-container{max-width:1500px;padding-top:1.6rem;padding-bottom:4rem}
h1,h2,h3{color:var(--ink);letter-spacing:-.025em}.page-title{font-size:clamp(2rem,3.8vw,3.7rem)!important;line-height:1.03!important;margin:.25rem 0 .5rem!important}.page-subtitle{color:var(--muted);font-size:1rem;margin:0 0 1.25rem;max-width:950px}.page-kicker{color:var(--blue);font-size:.69rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase}
.sidebar-brand{padding:.5rem .2rem 1.25rem;border-bottom:1px solid #29405f;margin-bottom:1rem}.sidebar-brand strong{font-size:1.65rem;letter-spacing:.08em;color:#fff}.sidebar-brand h2{color:#fff;font-size:1rem;margin:.55rem 0 .25rem}.sidebar-brand p{font-size:.74rem;margin:0;color:#9fb0c8}.sidebar-context{margin-top:2rem;padding:1rem;border:1px solid #29405f;border-radius:6px;background:#0f213c}.sidebar-context strong{font-size:.76rem;color:#fff}.sidebar-context p{font-size:.72rem;line-height:1.6;margin:.45rem 0 0;color:#9fb0c8!important}
.section-label{display:flex;gap:.8rem;align-items:flex-start;margin:1.75rem 0 .8rem}.section-label>span{background:var(--blue);color:#fff;font-size:.68rem;font-weight:800;padding:.28rem .38rem;border-radius:4px}.section-label h2{font-size:1.18rem;margin:0}.section-label p{font-size:.8rem;color:var(--muted);margin:.2rem 0 0}
.metric-card{background:var(--card);border:1px solid var(--line);border-top:3px solid var(--blue);border-radius:6px;padding:.85rem .9rem;min-height:122px;box-shadow:0 4px 14px rgba(25,50,85,.045)}.metric-card p{color:var(--muted);margin:0 0 .45rem;font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.04em}.metric-card strong{display:block;font-size:clamp(1.55rem,2.5vw,2.25rem);line-height:1;letter-spacing:-.035em;color:var(--ink)}.metric-card span{display:block;margin-top:.55rem;color:var(--muted);font-size:.72rem;line-height:1.35}
[data-testid="stVerticalBlockBorderWrapper"]{background:#fff;border-color:var(--line)!important;border-radius:6px!important;box-shadow:0 4px 14px rgba(25,50,85,.04)}
.notice{border:1px solid #c9d8ed;border-left:4px solid var(--blue);background:#eef5ff;padding:.75rem .9rem;margin:.8rem 0 1.2rem;border-radius:4px}.notice-warning{border-color:#ead39c;border-left-color:#d59b22;background:#fff9e9}.notice strong{font-size:.78rem}.notice p{margin:.25rem 0 0;color:#4d6076;font-size:.78rem;line-height:1.45}
.finding{border-top:1px solid var(--line);padding:1.1rem 0;max-width:1000px}.finding-card{background:#fff;border:1px solid var(--line);border-left:4px solid var(--blue);border-radius:6px;padding:1rem;margin-bottom:.8rem;min-height:160px}.finding h3{font-size:1.05rem;margin:.65rem 0 .3rem}.finding p{color:#4d6076;margin:.2rem 0;font-size:.84rem}.finding small{display:block;color:var(--muted);margin-top:.55rem;font-size:.74rem}.badge{display:inline-block;border:1px solid #cbd6e4;color:#4d6076;padding:.18rem .45rem;border-radius:4px;font-size:.65rem;font-weight:750;margin-right:.2rem;background:#f8fafc}.badge-accent{border-color:#a8c4f5;color:#174ea6;background:#eef5ff}
.section-heading{display:flex;justify-content:space-between;gap:1rem;align-items:flex-start;margin-bottom:1rem}.section-heading h2{margin:0}.section-heading p{color:var(--muted)}
.pipeline{display:grid;grid-template-columns:repeat(4,1fr);gap:.55rem}.pipeline>div{background:#fff;border:1px solid var(--line);border-radius:5px;padding:.75rem;min-height:68px}.pipeline span{display:block;font-size:.62rem;color:var(--blue);font-weight:800}.pipeline strong{font-size:.78rem;color:var(--ink)}
.method-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:.65rem}.method-grid>div{background:#fff;border:1px solid var(--line);border-radius:5px;padding:.8rem}.method-grid small{display:block;color:var(--muted);font-size:.65rem}.method-grid strong{display:block;color:var(--ink);font-size:.82rem;margin-top:.25rem}
.executive-header{margin:0 0 .85rem;padding:1.05rem 1.15rem;background:linear-gradient(120deg,#fff 0%,#f5f8fd 100%);border:1px solid var(--line);border-left:5px solid var(--blue);border-radius:8px;box-shadow:0 5px 18px rgba(25,50,85,.045)}.executive-kicker{color:var(--blue);font-size:.62rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase}.executive-heading-row{display:flex;align-items:flex-end;justify-content:space-between;gap:1.5rem}.executive-header h1{font-size:1.72rem;margin:.2rem 0 0;line-height:1.08}.executive-header h2{font-size:.94rem;margin:.25rem 0;color:#42546b;letter-spacing:0}.executive-header p{font-size:.76rem;color:var(--muted);margin:.55rem 0 0;max-width:850px}.executive-editions{flex:0 0 auto;text-align:right;background:#eaf1ff;border:1px solid #ccdcf7;border-radius:6px;padding:.55rem .7rem}.executive-editions span{display:block;color:#65748a;font-size:.58rem;font-weight:750;text-transform:uppercase;letter-spacing:.05em}.executive-editions strong{display:block;color:#174ea6;font-size:.72rem;margin-top:.18rem}.executive-section-label{margin:.95rem 0 .42rem;color:#526277;font-size:.64rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase}
.executive-metric{display:flex;gap:.8rem;align-items:center;background:#fff;border:1px solid var(--line);border-radius:8px;padding:.78rem .85rem;min-height:94px;box-shadow:0 3px 12px rgba(25,50,85,.04)}.metric-icon{display:grid;place-items:center;flex:0 0 38px;height:38px;border-radius:8px;background:#eaf1ff;color:#174ea6;font-size:1.08rem;font-family:"Segoe UI Emoji","Apple Color Emoji",sans-serif}.executive-metric-positive .metric-icon{background:#e7f7f0}.executive-metric-negative .metric-icon{background:#fdecec}.executive-metric-mixed .metric-icon{background:#f0edff}.executive-metric p{margin:0;color:var(--muted);font-size:.63rem;font-weight:800;text-transform:uppercase;letter-spacing:.035em}.executive-metric strong{display:block;color:var(--ink);font-size:1.55rem;line-height:1.05;margin:.25rem 0 .18rem;letter-spacing:-.035em}.executive-metric small{display:block;color:var(--muted);font-size:.65rem;line-height:1.2}
h3.compact-title{font-size:.88rem!important;margin:0 0 .15rem!important;letter-spacing:-.01em}.ai-summary{padding:.05rem 0}.ai-summary-heading{display:flex;align-items:center;justify-content:space-between;gap:.45rem}.ai-summary-heading h3{font-size:.88rem!important;margin:0!important;letter-spacing:-.01em}.ai-summary-heading .badge{margin:0;white-space:nowrap}.ai-summary-source{font-size:.58rem;color:var(--muted);margin:.18rem 0 .42rem}.ai-summary-overall,.ai-summary-takeaway{font-size:.65rem;line-height:1.38;margin:.35rem 0;color:#34465d!important}.ai-summary-insights{margin:.35rem 0;padding-left:1rem}.ai-summary-insights li{font-size:.63rem;line-height:1.35;color:#34465d;margin:.16rem 0}.ai-summary-takeaway{background:#eef5ff;border-left:3px solid var(--blue);padding:.42rem .48rem}.ai-summary-quality{font-size:.61rem;font-weight:750;color:#174ea6;margin:.4rem 0 .28rem}.ai-summary-caution{font-size:.58rem;line-height:1.35;color:#72520b!important;background:#fff9e9;padding:.38rem .44rem;margin:.25rem 0 0}.snapshot-row{padding:.46rem 0;border-bottom:1px solid #e8edf4}.snapshot-row:last-child{border-bottom:0}.snapshot-row span{display:block;color:#41536b;font-size:.67rem;line-height:1.25}.snapshot-row strong{display:block;color:var(--ink);font-size:.7rem;margin-top:.16rem}.language-row{display:grid;grid-template-columns:1fr auto;gap:.12rem .5rem;padding:.42rem 0;border-bottom:1px solid #e8edf4}.language-row span{font-size:.7rem;color:#41536b}.language-row strong{font-size:.72rem;color:var(--ink)}.language-row small{grid-column:1/-1;color:var(--muted);font-size:.61rem}.quality-grid{display:grid;grid-template-columns:1fr 1fr;gap:.42rem}.quality-grid>div{background:#f6f8fb;border:1px solid #e4eaf2;border-radius:4px;padding:.45rem}.quality-grid span{display:block;color:var(--muted);font-size:.58rem}.quality-grid strong{display:block;color:var(--ink);font-size:.77rem;margin-top:.12rem}.quality-warning{font-size:.65rem;color:#8a5a00;background:#fff7df;border-left:3px solid #d59b22;padding:.48rem;margin:.55rem 0 .35rem}.method-link{font-size:.62rem;color:var(--muted);margin:.2rem 0}
@media(max-width:900px){.pipeline,.method-grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:700px){.block-container{padding:1rem .8rem}.section-heading{display:block}.metric-card{min-height:100px;margin-bottom:.25rem}.executive-metric{min-height:86px;margin-bottom:.2rem}.executive-heading-row{display:block}.executive-editions{margin-top:.7rem;text-align:left;width:max-content}.executive-header h1{font-size:1.28rem}.page-title{font-size:2rem!important}.pipeline,.method-grid{grid-template-columns:1fr}.section-label{margin-top:1.25rem}}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important;transition:none!important;animation:none!important}}
</style>
""",unsafe_allow_html=True)

try:
    dashboard = get_data()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Dashboard cannot load the frozen research data: {exc}")
    st.stop()

labels = display_label_map(dashboard)
with st.sidebar:
    st.markdown("<div class='sidebar-brand'><strong>KLSCM</strong><h2>Sentiment &amp; Perception Analysis</h2><p>Social Media &amp; Long-form Reviews</p></div>",unsafe_allow_html=True)
    page=st.radio("Research dashboard navigation",["Participant Experience & Organizer Insights","Executive Overview","Overview","Social Media Analytics","Aspect Analysis","Temporal Trends","Topic Analysis","Language Analysis","Word Cloud","Research Findings","Cross-Source Analysis","Methodology"],label_visibility="collapsed",index=1,key="dashboard_page")
    st.markdown("<div class='sidebar-context'><strong>Kuala Lumpur Standard Chartered Marathon</strong><p>Observed editions:<br>2019 · 2023 · 2024 · 2025</p></div>",unsafe_allow_html=True)

pages={"Participant Experience & Organizer Insights":render_participant_experience,"Executive Overview":lambda:render_executive_overview(dashboard),"Overview":lambda:render_overview(dashboard),"Social Media Analytics":lambda:render_social_media(dashboard),
       "Aspect Analysis":lambda:render_aspect_explorer(dashboard,labels),"Temporal Trends":lambda:render_temporal(dashboard,labels),
       "Topic Analysis":lambda:render_topic_explorer(dashboard),"Language Analysis":lambda:render_language_explorer(dashboard),
       "Word Cloud":lambda:render_word_cloud(dashboard),"Research Findings":lambda:render_findings(dashboard),"Cross-Source Analysis":lambda:render_cross_source(dashboard),"Methodology":lambda:render_methodology(dashboard)}
if page not in {"Cross-Source Analysis", "Participant Experience & Organizer Insights"}:
    st.info("Source scope: this page presents the frozen Instagram analysis. Finalized long-form reviews are included separately under Cross-Source Analysis; source counts and sentiment are not pooled.")
pages[page]()
