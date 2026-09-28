from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from marathon_absa.dashboard_data import load_dashboard_data


def test_data_access_loads_and_validates_frozen_contract():
    data = load_dashboard_data()
    assert data.corpus["total_documents"] == 7704
    assert data.corpus["total_mentions"] == 15486
    assert data.aspects.aspect.nunique() == 20
    assert set(data.years.year) == {2019, 2023, 2024, 2025}
    assert data.topics.topic_id.nunique() == 32


def test_default_ranking_and_labels_are_frozen_mart_values():
    data = load_dashboard_data()
    assert data.aspects.iloc[0].aspect == "race_performance"
    assert data.aspects.iloc[0].document_prevalence_all > data.aspects.iloc[1].document_prevalence_all
    assert data.aspects.set_index("aspect").loc["photography_media", "recommended_for_emphasis"] == False
    assert set(data.aspects[data.aspects.recommended_for_emphasis].aspect) == {
        "race_performance", "crowd_community_atmosphere", "physical_experience"}
    assert data.aspects.support_class.str.endswith("_support").all()
    assert data.aspects.effect_category.isin(["trivial", "small", "moderate", "strong", "not_tested"]).all()


def test_research_and_caution_copy_comes_from_frozen_marts():
    data = load_dashboard_data()
    assert len(data.findings) == 4
    assert "weather_conditions" in set(data.cautions.item)
    assert "photography_media" in set(data.cautions.item)
    source = Path("absa_dashboard.py").read_text(encoding="utf-8")
    assert 'data.metadata["warnings"]["topic"]' in source
    assert "data.findings.itertuples()" in source and "data.cautions.itertuples()" in source


def test_denominator_columns_are_distinct_and_preserved():
    data = load_dashboard_data()
    assert "document_prevalence_all" in data.aspects
    assert "share_within_aspect" in data.aspect_sentiment
    assert not pd.Series(data.sentiment_overall.document_share_all).sum() == 1
    assert data.metadata["document_counts_can_overlap_by_sentiment"] is True


def test_frontend_contains_no_statistical_or_model_inference_logic():
    source = Path("absa_dashboard.py").read_text(encoding="utf-8").lower()
    for forbidden in ["chi2_contingency", "cramers_v(", "multipletests(", "wilson_interval(", "openai.", "huggingface_hub", "transformers.", "requests.post"]:
        assert forbidden not in source


def test_streamlit_mvp_smoke_and_visible_warnings():
    app = AppTest.from_file("absa_dashboard.py", default_timeout=30).run()
    assert not app.exception
    assert any("5,425" in item.value for item in app.markdown)
    assert any("12,065" in item.value for item in app.markdown)
    assert any("model-estimated" in item.value.lower() for item in app.markdown)
    assert any("descriptive" in item.value.lower() for item in app.markdown)


def test_weather_selection_surfaces_frozen_caution():
    app = AppTest.from_file("absa_dashboard.py", default_timeout=30).run()
    app.sidebar.radio[0].set_value("Aspect Analysis").run()
    selector = next(item for item in app.selectbox if item.label == "Aspect")
    selector.select("weather_conditions").run()
    assert not app.exception
    assert any("large sentiment variation" in item.value.lower() for item in app.markdown)


def test_navigation_and_unsupported_mockup_features_are_excluded():
    source = Path("absa_dashboard.py").read_text(encoding="utf-8")
    navigation = ["Participant Experience & Organizer Insights", "Executive Overview", "Overview", "Aspect Analysis",
                  "Topic Analysis", "Word Cloud", "Research Findings", "Cross-Source Analysis", "Methodology"]
    app = AppTest.from_file("absa_dashboard.py", default_timeout=30).run()
    assert app.sidebar.radio[0].options == navigation
    assert app.sidebar.radio[0].value == "Executive Overview"
    lower = source.lower()
    for unsupported in ["interview insights", "geospatial analysis", "gis map", "race category", "total interviewees", "locations detected"]:
        assert unsupported not in lower
    assert "2026" not in source


def test_executive_overview_reconciles_frozen_outputs_and_boundaries():
    app = AppTest.from_file("absa_dashboard.py", default_timeout=30).run()
    assert not app.exception
    visible = " ".join(item.value for item in [*app.markdown, *app.caption])
    for expected in ["5,425", "3,899", "12,065", "65.7%", "29.5%"]:
        assert expected in visible
    assert "Document prevalence among 5,425 analyzed posts" in visible
    assert "Share of 12,065 model-estimated aspect mentions" in visible
    assert "Descriptive context and research scope" in visible
    assert "0.513" in visible and "0.790" in visible and "80 documents" in visible
    assert "precision target of 0.55 was not met" in visible


def test_sidebar_reopen_control_and_executive_metric_icons_remain_visible():
    source = Path("absa_dashboard.py").read_text(encoding="utf-8")
    assert 'initial_sidebar_state="expanded"' in source
    assert '[data-testid="stToolbar"]{visibility:hidden}' not in source
    assert '[data-testid="stExpandSidebarButton"]{visibility:visible!important}' in source
    for icon in ["📚", "🎯", "🧩", "👍", "👎"]:
        assert icon in source


def test_executive_overview_uses_active_scope_and_preserves_frozen_findings():
    source = Path("absa_dashboard.py").read_text(encoding="utf-8")
    assert "for _, finding in data.findings.iterrows()" in source
    assert "ACTIVE_YEARS = (2023, 2024, 2025)" in source
    assert 'nlargest(6, "document_prevalence_all")' in source
    executive_source = source.split("def render_executive_overview", 1)[1].split("def render_header", 1)[0].lower()
    for unsupported in ["interview", "geospatial", "gis", "race category", "2019_value"]:
        assert unsupported not in executive_source


def test_executive_ai_summary_is_frozen_traceable_and_cautioned():
    app = AppTest.from_file("absa_dashboard.py", default_timeout=30).run()
    assert not app.exception
    visible = " ".join(item.value for item in [*app.markdown, *app.caption])
    for expected in [
        "AI Summary &amp; Key Insights", "Frozen research summary", "Generated from frozen analysis outputs",
        "Race Performance: 38.3% to 46.2%", "Community &amp; Atmosphere: 25.2% to 33.0%",
        "Physical Experience: 21.5% to 29.2%", "more positive and less negative discussion",
        "Development evidence: precision 0.513", "recall 0.790", "not causal changes",
        "changing corpus composition and increasing extraction density",
    ]:
        assert expected in visible
    assert "The most prevalent substantive aspects were Race Performance, Community &amp; Atmosphere, and Physical Experience." in visible
    source = Path("absa_dashboard.py").read_text(encoding="utf-8")
    assert "data.sentiment_overall.set_index" in source
    assert 'data.aspects.nlargest(3, "document_prevalence_all")' in source
    assert 'data.metadata["warnings"]["temporal"]' in source
    for forbidden in ["openai", "anthropic", "requests.post", "generate_content", "chat.completions"]:
        assert forbidden not in source.lower()


def test_active_scope_hides_pages_and_excludes_2019_from_user_surface():
    source = Path("absa_dashboard.py").read_text(encoding="utf-8")
    assert "ACTIVE_YEARS = (2023, 2024, 2025)" in source
    app = AppTest.from_file("absa_dashboard.py", default_timeout=60).run()
    hidden = {"Temporal Trends", "Social Media Analytics", "Language Analysis"}
    assert not hidden.intersection(app.sidebar.radio[0].options)
    visible = " ".join(item.value for item in [*app.markdown, *app.caption])
    assert "2019" not in visible
    assert "2023" in visible and "2025" in visible


def test_detailed_research_findings_page_remains_available_after_summary_addition():
    app = AppTest.from_file("absa_dashboard.py", default_timeout=30).run()
    app.sidebar.radio[0].set_value("Research Findings").run()
    assert not app.exception
    visible = " ".join(item.value for item in [*app.markdown, *app.caption])
    assert "Curated in the frozen research-findings mart" in visible
    assert "Race-performance discussion became more prevalent across observed editions." in visible
    assert "Training/preparation sentiment shifted toward more positive and less negative discussion." in visible


def test_topic_language_and_methodology_pages_show_frozen_boundaries():
    app = AppTest.from_file("absa_dashboard.py", default_timeout=30).run()
    app.sidebar.radio[0].set_value("Topic Analysis").run()
    assert any("descriptive" in item.value.lower() for item in app.markdown)
    app.sidebar.radio[0].set_value("Methodology").run()
    visible = " ".join(item.value for item in app.markdown)
    metric_values = {item.label: item.value for item in app.metric}
    assert metric_values["Aspect precision"] == "0.513" and metric_values["Aspect recall"] == "0.790"
    assert "0.55" in visible


def test_word_cloud_page_smoke_and_descriptive_boundaries():
    app = AppTest.from_file("absa_dashboard.py", default_timeout=45).run()
    app.sidebar.radio[0].set_value("Word Cloud").run(timeout=45)
    assert not app.exception
    visible = " ".join(item.value for item in [*app.markdown, *app.caption])
    assert "12,065" in visible and "3,899" in visible
    assert "Descriptive text exploration only" in visible
    assert "statistical significance" in visible
    assert next(item for item in app.radio if item.label == "Cloud weighting").value == "Document Frequency"
    assert next(item for item in app.radio if item.label == "Text Source").value == "ABSA Evidence"
    assert "Filtered ABSA Evidence Cloud" in [item.value for item in app.subheader]
    assert "Top Sentiment Expressions" in [item.value for item in app.subheader]


def test_full_caption_context_mode_remains_explicitly_available():
    app = AppTest.from_file("absa_dashboard.py", default_timeout=45).run()
    app.sidebar.radio[0].set_value("Word Cloud").run(timeout=45)
    source = next(item for item in app.radio if item.label == "Text Source")
    source.set_value("Full Caption Context").run(timeout=45)
    assert not app.exception
    visible = " ".join(item.value for item in [*app.markdown, *app.caption])
    assert "Filtered Caption Context" in [item.value for item in app.subheader]
    assert "Individual words are not necessarily sentiment-bearing" in visible


def test_theme_tokens_follow_streamlit_effective_color_scheme():
    source = Path("absa_dashboard.py").read_text(encoding="utf-8")
    assert "--ink:light-dark(#13243a,#e6edf7)" in source
    assert "--paper:light-dark(#f3f6fa,#0e1522)" in source
    assert "--navy:light-dark(#edf2f8,#0b1930)" in source
    assert 'prefers-color-scheme' not in source  # native System mode owns resolution
    assert '[data-testid="stSidebar"][aria-expanded="true"] [data-testid="stSidebarCollapseButton"]{visibility:visible!important}' in source
    assert ':focus-visible{outline:2px solid var(--blue)' in source
    assert '[data-testid="stToolbarActions"],[data-testid="stStatusWidget"]{visibility:hidden}' not in source


def test_executive_plotly_titles_and_theme_do_not_override_native_colors():
    import json
    app = AppTest.from_file("absa_dashboard.py", default_timeout=45).run()
    assert not app.exception
    charts = app.get("plotly_chart")
    assert len(charts) == 5
    for chart in charts:
        spec = json.loads(chart.proto.spec)
        layout = spec["layout"]
        assert layout["title"]["text"] == ""
        assert "undefined" not in json.dumps(spec)
        assert "color" not in layout.get("font", {})
        assert "color" not in layout.get("legend", {}).get("font", {})
        for axis in ("xaxis", "yaxis"):
            assert "gridcolor" not in layout.get(axis, {})
            assert "color" not in layout.get(axis, {}).get("tickfont", {})
        assert chart.proto.theme == "streamlit"


def test_aspect_explorer_displays_only_finalized_reviewed_themes():
    app = AppTest.from_file("absa_dashboard.py", default_timeout=45).run()
    app.sidebar.radio[0].set_value("Aspect Analysis").run(timeout=45)
    selector = next(item for item in app.selectbox if item.label == "Aspect")
    selector.select("route_course").run(timeout=45)
    assert not app.exception
    visible = " ".join(item.value for item in [*app.markdown, *app.caption])
    assert "What participants are talking about" in visible
    assert "Themes are ordered by unique-document support" in visible
    assert "Researcher perceptions and participant interview propositions have not yet been developed" in visible
    assert any("within aspect" in item.label for item in app.expander)
    source = Path("absa_dashboard.py").read_text(encoding="utf-8")
    assert "load_reviewed_theme_dashboard_data" in source
    assert "theme_summary.parquet" not in source and "provisional_label" not in source


def test_insufficient_support_aspect_has_clear_nonempty_state():
    app = AppTest.from_file("absa_dashboard.py", default_timeout=45).run()
    app.sidebar.radio[0].set_value("Aspect Analysis").run(timeout=45)
    selector = next(item for item in app.selectbox if item.label == "Aspect")
    selector.select("facilities").run(timeout=45)
    assert not app.exception
    visible = " ".join(item.value for item in [*app.markdown, *app.caption])
    assert "Insufficient support for stable within-aspect themes" in visible
    assert "57 mentions across 49 documents" in visible
    assert "does not imply that the aspect is unimportant" in visible


def test_topic_explorer_remains_available_with_construct_clarification():
    app = AppTest.from_file("absa_dashboard.py", default_timeout=45).run()
    app.sidebar.radio[0].set_value("Topic Analysis").run(timeout=45)
    assert not app.exception
    visible = " ".join(item.value for item in [*app.markdown, *app.caption])
    assert "broad corpus-wide discourse" in visible
