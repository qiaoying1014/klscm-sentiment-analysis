"""Dashboard contract for the verified, read-only participant-experience release."""
import json
import shutil
from pathlib import Path

import pytest

from marathon_absa import participant_experience_dashboard_data as loader
from marathon_absa.participant_experience_page import COVERAGE, PROVENANCE, _matches_keyword


def test_verified_v2_loader_has_complete_finalized_coverage():
    data = loader.load_participant_experience_dashboard_data()
    assert data["provenance"]["version"] == loader.RELEASE_VERSION
    assert data["provenance"]["review_status"] == "finalized"
    assert len(data["insights"]) == 99
    assert len({row["theme_id"] for row in data["insights"]}) == 99
    assert all(set(row["claims"]) == set(loader.FIELDS) for row in data["insights"])
    assert all(theme["source_coverage"] in COVERAGE for theme in data["evidence"]["themes"])
    assert any(claim["text"] == "Insufficient evidence at this level."
               for row in data["insights"] for claim in row["claims"].values())


def test_loader_rejects_invalid_finalized_file_without_candidate_fallback(tmp_path):
    root = tmp_path / "release"
    shutil.copytree(loader.ROOT, root)
    (root / "finalized.json").write_text("[]", encoding="utf-8")
    manifest = loader.read(root / "finalized_manifest.json")
    manifest["hashes"]["finalized.json"] = loader.sha(root / "finalized.json")
    (root / "finalized_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid theme references|incomplete release"):
        loader.load_participant_experience_dashboard_data(root)


def test_filters_and_keyword_only_select_finalized_theme_rows():
    data = loader.load_participant_experience_dashboard_data()
    themes = data["evidence"]["themes"]
    selected = loader.filter_themes(themes, coverage="blog_only_emergent")
    assert selected and all(row["source_coverage"] == "blog_only_emergent" for row in selected)
    insight = next(row for row in data["insights"] if row["theme_id"] == selected[0]["theme_id"])
    assert _matches_keyword(selected[0], insight["claims"], selected[0]["theme_label"])
    assert not _matches_keyword(selected[0], insight["claims"], "unfindable participant experience phrase")


def test_dashboard_page_exposes_finalized_explorer_and_methodology_note():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("absa_dashboard.py", default_timeout=60).run()
    app.sidebar.radio[0].set_value("Participant Experience & Organizer Insights").run(timeout=60)
    assert not app.exception
    visible = " ".join(item.value for item in [*app.markdown, *app.caption, *app.info])
    assert "Showing 99 of 99 finalized themes." in visible
    assert PROVENANCE in visible
    assert "not participant percentages" in visible
    assert {"Aspect", "Experience category", "Source coverage", "Results page"} <= {item.label for item in app.selectbox}
