import json

import pandas as pd
import pytest

from marathon_absa.config import SETTINGS
from marathon_absa.disagreement_audit import (
    AUTO_INCLUDE_CATEGORIES,
    DETERMINISTIC_CATEGORIES,
    classify_hashtag_pattern,
    create_disagreement_package,
)


OUTPUT = (
    SETTINGS.output_dir
    / "relevance_v8_architecture_disagreement_audit_v1"
)


def test_hashtag_pattern_classification_is_deterministic():
    assert classify_hashtag_pattern('["klscm2019", "42km"]') == "race_category_or_distance"
    assert classify_hashtag_pattern('["klscm", "finisher"]') == "result_pb_finisher_status"
    assert classify_hashtag_pattern('["roadtoklscm"]') == "preparation"
    assert classify_hashtag_pattern('["running"]') == "generic_running_only"
    assert classify_hashtag_pattern([]) == "insufficient_evidence"


def test_generated_disagreement_worksheet_is_exact_and_annotation_safe():
    report = pd.read_csv(
        OUTPUT / "relevance_v8_architecture_disagreements_v1.csv",
        dtype=str,
        keep_default_na=False,
    )
    assert len(report) == 26
    assert report["document_id"].is_unique
    assert report["disagreement_arm"].value_counts().to_dict() == {
        "automatic_include_researcher_exclude": 14,
        "deterministic_exclude_researcher_include": 12,
    }
    assert report["diagnostic_category"].eq("").all()
    assert report["diagnostic_note"].eq("").all()
    assert report["researcher_blind_decision"].value_counts().to_dict() == {
        "exclude": 14,
        "include": 12,
    }
    required = {
        "original_caption", "normalized_text", "semantic_text", "hashtags",
        "recorded_language", "event_year", "source", "production_route",
        "luna_event_connection", "terra_event_connection", "event_connection",
        "primary_content_type", "meaningful_content_present", "confidence",
        "verification_triggers", "final_automatic_decision",
        "researcher_evidence", "researcher_rationale", "researcher_comments",
    }
    assert required.issubset(report.columns)
    for row in report.itertuples(index=False):
        expected = (
            AUTO_INCLUDE_CATEGORIES
            if row.disagreement_arm == "automatic_include_researcher_exclude"
            else DETERMINISTIC_CATEGORIES
        )
        assert row.allowed_diagnostic_categories.split("|") == expected


def test_disagreement_package_is_overwrite_protected_and_manifested():
    manifest = json.loads((OUTPUT / "manifest_v1.json").read_text(encoding="utf-8"))
    assert manifest["records"] == 26
    assert manifest["api_calls"] == 0
    assert manifest["annotation_fields_preserved"] is True
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        create_disagreement_package(SETTINGS)
