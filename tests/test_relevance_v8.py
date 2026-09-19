import pandas as pd
import pytest

from marathon_absa.openai_service import RELEVANCE_V8_FEW_SHOTS, RELEVANCE_V8_INSTRUCTIONS
from marathon_absa.relevance_v8 import (
    apply_human_decision, derive_reason_code, material_model_disagreement,
    route_assessment, threshold_analysis, validation_report_v8,
)
from marathon_absa.schemas import RELEVANCE_V8_ASSESSMENT_SCHEMA


def assessment(connection="explicit", content="event_participation", confidence=.95, **overrides):
    value = {
        "document_id": "d1", "event_connection": connection,
        "primary_content_type": content, "secondary_content_types": [],
        "meaningful_content_present": content != "no_meaningful_content",
        "event_link_evidence": "KLSCM" if connection == "explicit" else "#KLSCM" if connection in {"supported", "weak"} else "",
        "analytical_content_evidence": "finished" if content != "no_meaningful_content" else "",
        "confidence": confidence, "contradiction_present": False,
        "contradiction_note": "", "image_dependent": False,
        "requires_review_recommendation": False, "review_reason": "",
        "language_observed": "English", "code_switching_note": "",
        "short_explanation": "grounded",
    }
    value.update(overrides)
    return value


def test_v8_model_schema_does_not_accept_operational_reason_code():
    assert "reason_code" not in RELEVANCE_V8_ASSESSMENT_SCHEMA["properties"]


def test_v8_model_schema_uses_supported_strict_json_schema_keywords():
    secondary_types = RELEVANCE_V8_ASSESSMENT_SCHEMA["properties"]["secondary_content_types"]
    assert "uniqueItems" not in secondary_types


def test_v8_prompt_contains_at_least_thirty_examples():
    assert len(RELEVANCE_V8_FEW_SHOTS) >= 30
    assert "not explicit merely because" in RELEVANCE_V8_INSTRUCTIONS


def test_event_linked_emotional_experience_is_included():
    routed = route_assessment(assessment("supported", "event_emotional_experience"))
    assert routed["v8_final_inclusion"] == "include"
    assert routed["v8_reason_code"] == "supported_event_emotional_experience"


def test_event_linked_physical_experience_is_included():
    assert route_assessment(assessment("supported", "event_physical_experience"))["v8_final_inclusion"] == "include"


def test_generic_emotion_without_connection_is_excluded():
    routed = route_assessment(assessment("none", "no_meaningful_content", .98,
                                        meaningful_content_present=False,
                                        event_link_evidence="", analytical_content_evidence=""))
    assert routed["v8_final_inclusion"] == "exclude"


def test_generic_running_motivation_is_excluded():
    assert route_assessment(assessment("none", "generic_running", .97))["v8_final_inclusion"] == "exclude"


def test_hashtag_plus_meaningful_experience_is_supported_and_included():
    assert route_assessment(assessment("supported", "event_achievement"))["v8_include_in_topics"]


def test_weak_meaningful_caption_enters_review():
    routed = route_assessment(assessment("weak", "event_emotional_experience", .99))
    assert routed["v8_routing_status"] == "pending_review"


def test_weak_empty_caption_uses_separately_reported_strict_exception():
    routed = route_assessment(assessment("weak", "no_meaningful_content", .99,
                                        meaningful_content_present=False,
                                        event_link_evidence="", analytical_content_evidence=""))
    assert routed["v8_routing_status"] == "automatic_exclude_weak_empty"


def test_promotion_only_is_excluded_at_strict_threshold():
    routed = route_assessment(assessment("supported", "promotional_only", .95))
    assert routed["v8_final_inclusion"] == "exclude"


def test_other_marathon_is_excluded():
    assert derive_reason_code(assessment("none", "other_event")) == "different_event"


def test_meaningful_comparison_is_not_mapped_to_different_event():
    value = assessment("explicit", "other_event", secondary_content_types=["event_evaluation"])
    assert derive_reason_code(value) == "meaningful_klscm_comparison"


def test_short_caption_is_not_automatically_rejected():
    assert route_assessment(assessment("supported", "event_achievement"))["v8_final_inclusion"] == "include"


def test_image_dependent_caption_enters_review():
    routed = route_assessment(assessment("weak", "event_emotional_experience", image_dependent=True))
    assert routed["v8_routing_status"] == "pending_review"


def test_code_switched_caption_can_be_supported():
    routed = route_assessment(assessment("supported", "event_physical_experience",
                                        language_observed="Malay-English",
                                        code_switching_note="Malay-English"))
    assert routed["v8_final_inclusion"] == "include"


def test_event_information_enters_topics_but_not_sentiment():
    routed = route_assessment(assessment("explicit", "event_information"))
    assert routed["v8_include_in_topics"] is True
    assert routed["v8_include_in_sentiment"] is False


def test_human_decision_controls_final_eligibility_and_survives_as_field():
    reviewed = apply_human_decision(route_assessment(assessment("weak", "event_emotional_experience")), "include")
    assert reviewed["v8_human_final_inclusion"] == "include"
    assert reviewed["v8_include_in_topics"] is True


def test_included_subtype_disagreement_is_not_material():
    first = assessment("explicit", "event_logistics")
    second = assessment("explicit", "event_organization")
    assert not material_model_disagreement(first, second)


def test_included_vs_excluded_disagreement_is_material():
    assert material_model_disagreement(assessment(), assessment("none", "generic_running"))


def test_validation_blocks_missing_or_malformed_human_labels(tmp_path):
    reviewer = pd.DataFrame([{"review_id": "r1", "document_id": "d1", "repeat_of": "", "human_final_inclusion": ""}])
    predictions = pd.DataFrame([route_assessment(assessment())])
    with pytest.raises(ValueError, match="Missing or malformed"):
        validation_report_v8(reviewer, predictions, tmp_path)


def test_validation_excludes_repeat_from_main_count_and_reports_annotation_unavailable(tmp_path):
    reviewer = pd.DataFrame([
        {"review_id": "r1", "document_id": "d1", "repeat_of": "", "human_final_inclusion": "include"},
    ])
    predictions = pd.DataFrame([route_assessment(assessment())])
    report = validation_report_v8(reviewer, predictions, tmp_path)
    assert report["evaluation_n"] == 1
    assert report["annotation_quality_passes"] is None
    assert report["overall_validation_ready"] is False


def test_review_status_is_routing_not_third_binary_prediction(tmp_path):
    reviewer = pd.DataFrame([{"review_id": "r1", "document_id": "d1", "repeat_of": "", "human_final_inclusion": "include"}])
    predictions = pd.DataFrame([route_assessment(assessment("weak", "event_emotional_experience"))])
    report = validation_report_v8(reviewer, predictions, tmp_path)
    assert report["routing_performance"]["review_rate"] == 1.0
    assert report["simulated_human_assisted_performance"]["inclusion_recall"] == 1.0


def test_threshold_analysis_uses_two_independent_thresholds():
    frame = pd.DataFrame([assessment()])
    labels = pd.DataFrame([{"document_id": "d1", "human_final_inclusion": "include"}])
    table = threshold_analysis(frame, labels, include_thresholds=(.8,), exclude_thresholds=(.93,))
    assert table.loc[0, "automatic_inclusion_threshold"] == .8
    assert table.loc[0, "automatic_exclusion_threshold"] == .93



def test_v8_controlled_calibration_is_isolated_and_writes_required_views(tmp_path, monkeypatch):
    from dataclasses import replace
    from marathon_absa.config import SETTINGS
    from marathon_absa.pipeline import Pipeline
    from marathon_absa.storage import write_table

    settings = replace(SETTINGS, output_dir=tmp_path / "processed", cache_dir=tmp_path / "cache")
    settings.ensure_dirs()
    documents = pd.DataFrame([
        {"document_id": "d1", "source": "instagram", "processing_status": "ready", "event_year": "2024", "primary_language": "English", "language_status": "ok", "original_text": "Finished KLSCM", "linguistic_text": "Finished KLSCM", "hashtags": []},
        {"document_id": "d2", "source": "instagram", "processing_status": "ready", "event_year": "2024", "primary_language": "English", "language_status": "ok", "original_text": "Morning run", "linguistic_text": "Morning run", "hashtags": []},
    ])
    write_table(documents, settings.output_dir / "documents.parquet")
    reviewer = pd.DataFrame([
        {"review_id": "r1", "document_id": "d1", "repeat_of": "", "human_relevance": "relevant"},
        {"review_id": "r2", "document_id": "d2", "repeat_of": "", "human_relevance": "irrelevant"},
    ])
    reviewer.to_csv(settings.output_dir / "relevance_v7_calibration_sample.csv", index=False)
    write_table(pd.DataFrame([{"review_id": "r1", "document_id": "d1"}, {"review_id": "r2", "document_id": "d2"}]), settings.output_dir / "relevance_v7_calibration_key.parquet")
    pipeline = Pipeline(settings)
    scored = pd.DataFrame([
        route_assessment(assessment("explicit", "event_achievement")),
        {**route_assessment(assessment("none", "generic_running", .98)), "document_id": "d2"},
    ])
    monkeypatch.setattr(pipeline, "_score_relevance_v8_candidates", lambda candidates: scored)
    report = pipeline.relevance_v8_score_calibration()
    result_dir = settings.output_dir / "relevance_v8_calibration_results"
    assert report["evaluation_n"] == 2
    assert (result_dir / "relevance_v8_threshold_analysis.csv").exists()
    assert (result_dir / "relevance_v8_error_audit.csv").exists()
    assert not (settings.output_dir / "relevance.parquet").exists()

