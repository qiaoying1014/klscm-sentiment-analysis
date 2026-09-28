"""Display-only aspect selection; never passed to research loaders or pipelines."""
from dataclasses import replace

HIDDEN_DISPLAY_ASPECTS = frozenset({
    "race_performance", "emotional_experience",
    "training_preparation_pacing", "emerging_other",
})


def visible_aspects(frame):
    """Return a display copy, preserving frozen values and denominators."""
    return frame.loc[~frame["aspect"].isin(HIDDEN_DISPLAY_ASPECTS)].copy()


def marathon_experience_display(data):
    """A render-only view; corpus, years, topics and overall sentiment stay intact."""
    return replace(data, **{
        name: visible_aspects(getattr(data, name)) for name in (
            "aspects", "aspect_sentiment", "year_aspect", "year_aspect_sentiment",
            "pairwise", "findings", "topic_aspect", "language_aspect",
        )
    })
