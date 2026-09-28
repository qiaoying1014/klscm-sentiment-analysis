"""Render-only cleanup must preserve the complete research release."""
import json

import pandas as pd
from streamlit.testing.v1 import AppTest

from marathon_absa.dashboard_data import load_dashboard_data
from marathon_absa.dashboard_presentation import HIDDEN_DISPLAY_ASPECTS, marathon_experience_display


def test_display_projection_preserves_sources_denominators_and_all_topics():
    original = load_dashboard_data()
    snapshot = original.aspects.copy(deep=True)
    shown = marathon_experience_display(original)
    assert HIDDEN_DISPLAY_ASPECTS <= set(original.aspects.aspect)
    assert len(shown.aspects) == 16
    pd.testing.assert_frame_equal(original.aspects, snapshot)
    for field in ("corpus", "metadata", "years", "topics", "sentiment_overall"):
        assert getattr(shown, field) is getattr(original, field)
    assert set(shown.years.year) == {2019, 2023, 2024, 2025}
    assert len(shown.topics) == 32
    for field in ("aspects", "aspect_sentiment", "year_aspect", "year_aspect_sentiment", "topic_aspect", "findings"):
        frame = getattr(shown, field)
        assert not HIDDEN_DISPLAY_ASPECTS.intersection(frame.aspect)
        pd.testing.assert_frame_equal(frame, getattr(original, field).loc[frame.index])


def test_visible_pages_do_not_expose_hidden_aspect_categories():
    data = load_dashboard_data()
    forbidden = set(HIDDEN_DISPLAY_ASPECTS) | set(data.aspects.loc[data.aspects.aspect.isin(HIDDEN_DISPLAY_ASPECTS), "display_label"])
    app = AppTest.from_file("absa_dashboard.py", default_timeout=60).run()
    assert not {"Research Findings", "Cross-Source Analysis", "Temporal Trends", "Language Analysis", "Social Media Analytics"}.intersection(app.sidebar.radio[0].options)
    for page in app.sidebar.radio[0].options:
        app.sidebar.radio[0].set_value(page).run(timeout=60)
        assert not app.exception, page
        # Do not scan participant quotations or broad topic titles: those are frozen source text.
        surface = " ".join(str(item.options) for item in app.selectbox if item.label == "Aspect")
        surface += " ".join(item.label for item in app.expander)
        for chart in app.get("plotly_chart"):
            spec = json.loads(chart.proto.spec)
            surface += " ".join(json.dumps(trace) for trace in spec["data"])
        for label in forbidden:
            assert label not in surface, (page, label)
        visible = " ".join(str(item.value) for item in [*app.markdown, *app.caption, *app.subheader])
        for removed in ("Across observed editions", "Accessible topic-aspect table", "Top Sentiment Expressions", "Descriptive context and research scope"):
            assert removed not in visible, (page, removed)
