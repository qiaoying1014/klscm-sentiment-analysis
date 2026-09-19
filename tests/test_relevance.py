from types import SimpleNamespace
from dataclasses import replace

import pandas as pd
import pytest

from marathon_absa.schemas import RELEVANCE_SCHEMA
from marathon_absa.config import SETTINGS
from marathon_absa.storage import read_table, write_table

from marathon_absa.relevance import (
    apply_human_reviews,
    build_fresh_holdout_sample,
    build_review_queue,
    build_validation_sample,
    deterministic_result,
    needs_adjudication,
    route_results,
    route_verified_result,
    validate_topic_gate,
    binary_metrics_with_bootstrap,
    fresh_holdout_report,
    validation_report,
)


def result(label, confidence, reason="weak_event_connection"):
    return {
        "relevance": label,
        "confidence": confidence,
        "reason_code": reason,
        "evidence": "evidence",
        "english_gloss": "gloss",
        "model_observed_language": "English",
        "code_switching_note": "",
    }


def test_deterministic_hashtag_only_is_auto_excluded():
    row = SimpleNamespace(linguistic_text="", language_status="insufficient_text", hashtags=["klscm2024"])
    routed = deterministic_result(row)
    assert routed["reason_code"] == "hashtag_only"
    assert routed["review_status"] == "auto_excluded"
    assert routed["exclusion_reason"] == "hashtag_only"
    assert not routed["include_in_topics"]


def test_v5_all_relevant_decisions_require_adjudication():
    assert needs_adjudication("relevant", 0.99)
    assert needs_adjudication("relevant", 0.79)
    assert not needs_adjudication("irrelevant", 0.90)
    assert needs_adjudication("irrelevant", 0.89)
    assert needs_adjudication("ambiguous", 1.0)


def test_calibration_sensitive_exclusions_always_receive_adjudication():
    for reason_code in [
        "generic_running", "no_event_connection", "weak_event_connection",
        "hashtag_only", "conflicting_evidence",
    ]:
        assert needs_adjudication("irrelevant", 0.99, reason_code)


def test_sensitive_high_confidence_exclusion_cannot_bypass_adjudication():
    routed = route_results(result("irrelevant", 0.99, "generic_running"))
    assert routed["relevance"] == "ambiguous"
    assert routed["review_status"] == "pending"
    assert routed["adjudication_method"] == "awaiting_adjudication"


def test_adjudication_can_correct_sensitive_false_negative():
    routed = route_results(
        result("irrelevant", 0.99, "generic_running"),
        result("relevant", 0.90, "event_preparation"),
    )
    assert routed["relevance"] == "relevant"
    assert routed["review_status"] == "auto_resolved"
    assert routed["include_in_topics"]

def test_v6_structured_gate_requires_subject_relation_and_no_exclusion():
    initial = result("relevant", 0.99, "event_participation")
    verification = {
        "substantive_relevance": "relevant", "confidence": 0.95,
        "klscm_is_actual_subject": True, "concrete_event_relation": True,
        "exclusion_trigger": "none", "reason_code": "event_participation",
        "evidence": "KLSCM finish", "english_gloss": "KLSCM finish",
        "model_observed_language": "English", "code_switching_note": "",
    }
    accepted = route_verified_result(initial, verification)
    assert accepted["relevance"] == "relevant"
    assert accepted["substantive_relevance"] == "relevant"
    assert accepted["routing_status"] == "auto_resolved"
    for field, value in [("klscm_is_actual_subject", False), ("concrete_event_relation", False)]:
        rejected = route_verified_result(initial, {**verification, field: value})
        assert rejected["relevance"] == "ambiguous"
        assert rejected["routing_status"] == "pending"
    excluded = route_verified_result(initial, {**verification, "exclusion_trigger": "generic_running"})
    assert excluded["relevance"] == "ambiguous"


def test_v6_verification_confidence_is_secondary_routing_signal():
    initial = result("relevant", 0.99, "event_participation")
    verification = {"substantive_relevance": "relevant", "confidence": 0.79, "klscm_is_actual_subject": True, "concrete_event_relation": True, "exclusion_trigger": "none", "reason_code": "event_participation", "evidence": "KLSCM finish", "english_gloss": "", "model_observed_language": "English", "code_switching_note": ""}
    routed = route_verified_result(initial, verification)
    assert routed["relevance"] == "ambiguous"
    assert routed["substantive_relevance"] == "relevant"

def test_stronger_model_routes_borderline_to_human_review():
    routed = route_results(result("ambiguous", 0.7), result("relevant", 0.74))
    assert routed["relevance"] == "ambiguous"
    assert routed["review_status"] == "pending"
    assert not routed["include_in_topics"]


def test_v5_rejects_sub_ninety_percent_relevant_adjudication():
    routed = route_results(
        result("relevant", 0.99, "event_participation"),
        result("relevant", 0.89, "event_participation"),
    )
    assert routed["relevance"] == "ambiguous"
    assert routed["review_status"] == "pending"

def test_stronger_model_can_accept_relevant_record():
    routed = route_results(result("ambiguous", 0.7), result("relevant", 0.90, "event_participation"))
    assert routed["relevance"] == "relevant"
    assert routed["review_status"] == "auto_resolved"
    assert routed["include_in_topics"]


def test_human_review_overrides_model_decision():
    relevance = pd.DataFrame([{
        "document_id": "a", "relevance": "ambiguous", "confidence": 0.6,
        "review_status": "pending", "adjudication_method": "stronger_model",
        "include_in_topics": False, "human_relevance": "", "review_notes": "",
    }])
    queue = pd.DataFrame([{"document_id": "a", "human_relevance": "relevant", "review_notes": "KLSCM preparation"}])
    output = apply_human_reviews(relevance, queue)
    assert output.loc[0, "relevance"] == "relevant"
    assert output.loc[0, "review_status"] == "human_resolved"
    assert bool(output.loc[0, "include_in_topics"])


def test_review_queue_preserves_existing_human_input(tmp_path):
    relevance = pd.DataFrame([{"document_id": "a", "review_status": "pending", "relevance": "ambiguous"}])
    documents = pd.DataFrame([{
        "document_id": "a", "event_year": "2024", "primary_language": "Malay",
        "original_text": "sy daftar KLSCM", "linguistic_text": "sy daftar",
    }])
    path = tmp_path / "queue.csv"
    pd.DataFrame([{"document_id": "a", "human_relevance": "relevant", "review_notes": "checked"}]).to_csv(path, index=False)
    queue = build_review_queue(relevance, documents, path)
    assert queue.loc[0, "human_relevance"] == "relevant"
    assert queue.loc[0, "review_notes"] == "checked"


def test_topic_gate_requires_complete_and_resolved_relevance():
    documents = pd.DataFrame([{"document_id": "a", "source": "instagram", "processing_status": "ready"}])
    with pytest.raises(RuntimeError, match="incomplete"):
        validate_topic_gate(documents, pd.DataFrame(columns=["document_id", "review_status"]))
    relevance = pd.DataFrame([{"document_id": "a", "review_status": "pending"}])
    with pytest.raises(RuntimeError, match="pending"):
        validate_topic_gate(documents, relevance)
    relevance.loc[0, "review_status"] = "human_resolved"
    validate_topic_gate(documents, relevance)


def test_validation_sample_is_blind_and_adds_repeats():
    rows = []
    documents = []
    labels = ["relevant", "ambiguous", "irrelevant"]
    for i in range(30):
        document_id = f"d{i}"
        rows.append({
            "document_id": document_id, "relevance": labels[i % 3], "confidence": 0.9,
            "reason_code": "event_participation" if i % 3 == 0 else "weak_event_connection",
        })
        documents.append({
            "document_id": document_id, "event_year": str(2023 + i % 2),
            "primary_language": "English" if i % 2 else "Malay", "language_status": "ok",
            "original_text": f"caption {i}", "linguistic_text": f"caption words {i}",
        })
    reviewer, key = build_validation_sample(pd.DataFrame(rows), pd.DataFrame(documents), n=18, repeats=3, seed=42)
    assert len(reviewer) == 21
    assert "model_relevance" not in reviewer.columns
    assert "model_relevance" in key.columns
    assert reviewer.repeat_of.ne("").sum() == 3


def test_v4_prompt_requires_explicit_klscm_attribution():
    from marathon_absa.openai_service import (
        RELEVANCE_ADJUDICATION_INSTRUCTIONS,
        RELEVANCE_INSTRUCTIONS,
    )

    assert "explicitly attributes" in RELEVANCE_INSTRUCTIONS
    assert "named event is the main subject" in RELEVANCE_INSTRUCTIONS
    assert "product calls-to-action" in RELEVANCE_ADJUDICATION_INSTRUCTIONS

def test_fresh_holdout_excludes_prior_documents_and_is_reproducible():
    relevance = pd.DataFrame([
        {"document_id": f"d{i}", "relevance": "relevant" if i % 2 else "irrelevant", "confidence": 0.9, "reason_code": "event_participation"}
        for i in range(20)
    ])
    documents = pd.DataFrame([
        {"document_id": f"d{i}", "event_year": "2024", "primary_language": "English", "language_status": "ok", "original_text": f"caption {i}", "linguistic_text": f"caption {i}"}
        for i in range(20)
    ])
    excluded = {"d0", "d1", "d2"}
    first_reviewer, first_key = build_fresh_holdout_sample(relevance, documents, excluded, n=10, repeats=2, seed=42)
    second_reviewer, second_key = build_fresh_holdout_sample(relevance, documents, excluded, n=10, repeats=2, seed=42)
    assert not (set(first_key.document_id) & excluded)
    assert first_key.equals(second_key)
    assert first_reviewer.equals(second_reviewer)
    assert first_key.validation_split.eq("fresh_holdout").all()
    assert "model_relevance" not in first_reviewer.columns
    assert first_reviewer.repeat_of.ne("").sum() == 2

def test_reason_code_schema_matches_policy_constants():
    from marathon_absa.relevance import REASON_CODES
    assert set(RELEVANCE_SCHEMA["properties"]["reason_code"]["enum"]) == REASON_CODES


def test_binary_bootstrap_metrics_are_reproducible_and_support_weights():
    frame = pd.DataFrame({"human": ["relevant", "relevant", "irrelevant", "irrelevant"], "predicted": ["relevant", "irrelevant", "irrelevant", "relevant"], "weight": [1.0, 2.0, 1.0, 3.0]})
    first = binary_metrics_with_bootstrap(frame, "human", "predicted", seed=7, bootstrap_samples=50)
    second = binary_metrics_with_bootstrap(frame, "human", "predicted", seed=7, bootstrap_samples=50)
    weighted = binary_metrics_with_bootstrap(frame, "human", "predicted", seed=7, bootstrap_samples=50, weight_column="weight")
    assert first == second
    assert first["relevant_precision"] == 0.5
    assert weighted["weighted"]
    assert weighted["relevant_precision"] != first["relevant_precision"]


def test_v6_verification_schema_contains_auditable_gate_fields():
    from marathon_absa.schemas import RELEVANCE_VERIFICATION_SCHEMA
    required = set(RELEVANCE_VERIFICATION_SCHEMA["required"])
    assert {"substantive_relevance", "klscm_is_actual_subject", "concrete_event_relation", "exclusion_trigger"}.issubset(required)

def test_validation_report_writes_metrics_and_intra_reviewer_agreement(tmp_path):
    key = pd.DataFrame([
        {"review_id": "r1", "model_relevance": "relevant", "validation_split": "holdout", "event_year": "2024", "primary_language": "English", "language_status": "ok", "length_band": 1},
        {"review_id": "r2", "model_relevance": "irrelevant", "validation_split": "holdout", "event_year": "2024", "primary_language": "Malay", "language_status": "ok", "length_band": 2},
    ])
    reviewer = pd.DataFrame([
        {"review_id": "r1", "repeat_of": "", "human_relevance": "relevant"},
        {"review_id": "r2", "repeat_of": "", "human_relevance": "irrelevant"},
        {"review_id": "repeat-1", "repeat_of": "r1", "human_relevance": "relevant"},
        {"review_id": "repeat-2", "repeat_of": "r2", "human_relevance": "irrelevant"},
    ])
    report = validation_report(reviewer, key, tmp_path)
    assert report["passes_targets"]
    assert report["intra_reviewer_n"] == 2
    assert (tmp_path / "relevance_confusion_matrix.csv").exists()
    assert (tmp_path / "relevance_subgroup_metrics.csv").exists()

def test_rescore_validation_reuses_completed_labels_and_isolates_outputs(
    tmp_path, monkeypatch,
):
    from marathon_absa import pipeline as pipeline_module

    settings = replace(
        SETTINGS,
        output_dir=tmp_path / "processed",
        cache_dir=tmp_path / "cache",
    )
    settings.ensure_dirs()
    documents = pd.DataFrame([
        {"document_id": "d1"},
        {"document_id": "d2"},
    ])
    write_table(documents, settings.output_dir / "documents.parquet")
    reviewer = pd.DataFrame([
        {
            "review_id": "r1", "document_id": "d1", "repeat_of": "",
            "human_relevance": "relevant",
        },
        {
            "review_id": "r2", "document_id": "d2", "repeat_of": "",
            "human_relevance": "irrelevant",
        },
    ])
    reviewer.to_csv(
        settings.output_dir
        / "relevance_validation_sample_before_recalibration.csv",
        index=False,
    )
    key = pd.DataFrame([
        {
            "review_id": "r1", "document_id": "d1",
            "model_relevance": "irrelevant", "model_confidence": 0.9,
            "model_reason_code": "generic_running",
            "validation_split": "holdout", "event_year": "2024",
            "primary_language": "English", "language_status": "ok",
            "length_band": 1,
        },
        {
            "review_id": "r2", "document_id": "d2",
            "model_relevance": "relevant", "model_confidence": 0.9,
            "model_reason_code": "event_participation",
            "validation_split": "holdout", "event_year": "2024",
            "primary_language": "Malay", "language_status": "ok",
            "length_band": 2,
        },
    ])
    key.to_csv(
        settings.output_dir
        / "relevance_validation_key_before_recalibration.csv",
        index=False,
    )
    scored = pd.DataFrame([
        {
            "document_id": "d1", "relevance": "relevant",
            "confidence": 0.95, "reason_code": "event_participation",
            "initial_prompt_version": "relevance-v6",
            "adjudication_prompt_version": "",
        },
        {
            "document_id": "d2", "relevance": "irrelevant",
            "confidence": 0.95, "reason_code": "no_event_connection",
            "initial_prompt_version": "relevance-v6",
            "adjudication_prompt_version": "relevance-verification-v6",
        },
    ])
    pipeline = pipeline_module.Pipeline(settings)
    monkeypatch.setattr(
        pipeline, "_score_relevance_candidates", lambda candidates: scored
    )

    report = pipeline.relevance_rescore_validation("holdout")

    assert report["passes_targets"]
    output_dir = settings.output_dir / "relevance_v6_holdout"
    assert (output_dir / "relevance_validation_metrics.json").exists()
    rescored = read_table(output_dir / "relevance_validation_key.parquet")
    assert rescored.model_relevance.tolist() == ["relevant", "irrelevant"]
    assert not (settings.output_dir / "relevance.parquet").exists()

def test_fresh_holdout_report_uses_binary_gate_and_reports_routing(tmp_path):
    reviewer = pd.DataFrame([
        {"review_id": "r1", "repeat_of": "", "human_relevance": "relevant"},
        {"review_id": "r2", "repeat_of": "", "human_relevance": "irrelevant"},
        {"review_id": "r3", "repeat_of": "", "human_relevance": "ambiguous"},
        {"review_id": "repeat-1", "repeat_of": "r1", "human_relevance": "relevant"},
    ])
    key = pd.DataFrame([
        {"review_id": "r1", "model_relevance": "relevant", "validation_split": "fresh_holdout", "event_year": "2024", "primary_language": "English", "language_status": "ok", "length_band": 1},
        {"review_id": "r2", "model_relevance": "irrelevant", "validation_split": "fresh_holdout", "event_year": "2024", "primary_language": "English", "language_status": "ok", "length_band": 1},
        {"review_id": "r3", "model_relevance": "ambiguous", "validation_split": "fresh_holdout", "event_year": "2024", "primary_language": "English", "language_status": "ok", "length_band": 1},
    ])
    report = fresh_holdout_report(reviewer, key, tmp_path)
    assert report["substantive_binary_n"] == 2
    assert report["substantive_binary_macro_f1"] == 1.0
    assert report["ambiguous_routing_rate"] == pytest.approx(1 / 3)
    assert report["repeat_agreement_rate"] == 1.0
    assert report["passes_targets"]


def test_fresh_holdout_rescore_isolated_and_never_scores_repeats(tmp_path, monkeypatch):
    from marathon_absa import pipeline as pipeline_module
    settings = replace(SETTINGS, output_dir=tmp_path / "processed", cache_dir=tmp_path / "cache")
    settings.ensure_dirs()
    documents = pd.DataFrame([{"document_id": f"d{i}", "event_year": "2024", "primary_language": "English", "language_status": "ok", "original_text": f"caption {i}", "linguistic_text": f"caption {i}", "hashtags": []} for i in range(150)])
    write_table(documents, settings.output_dir / "documents.parquet")
    production = pd.DataFrame([{"document_id": "production-sentinel", "relevance": "irrelevant"}])
    write_table(production, settings.output_dir / "relevance.parquet")
    reviewer = pd.DataFrame([{"review_id": f"fresh-{i+1:04d}", "document_id": f"d{i}", "repeat_of": "", "human_relevance": "relevant" if i < 34 else "irrelevant", "review_notes": ""} for i in range(150)])
    repeats = reviewer.head(15).copy()
    repeats["repeat_of"] = repeats.review_id
    repeats["review_id"] = [f"fresh-repeat-{i+1:03d}" for i in range(15)]
    reviewer = pd.concat([reviewer, repeats], ignore_index=True)
    reviewer_path = settings.output_dir / "relevance_fresh_holdout_sample.csv"
    reviewer.to_csv(reviewer_path, index=False)
    key = pd.DataFrame([{"review_id": f"fresh-{i+1:04d}", "document_id": f"d{i}", "validation_split": "fresh_holdout", "event_year": "2024", "primary_language": "English", "language_status": "ok", "length_band": 1, "model_relevance": "irrelevant", "model_confidence": 0.9, "model_reason_code": "generic_running"} for i in range(150)])
    write_table(key, settings.output_dir / "relevance_fresh_holdout_key.parquet")
    before_reviewer = reviewer_path.read_bytes()
    before_production = (settings.output_dir / "relevance.parquet").read_bytes()
    def fake_score(candidates):
        assert len(candidates) == 150
        rows = []
        for row in candidates.itertuples(index=False):
            index = int(row.document_id[1:])
            label = "relevant" if index < 34 else "irrelevant"
            rows.append({"document_id": row.document_id, "relevance": label, "confidence": 0.99, "reason_code": "event_participation" if label == "relevant" else "no_event_connection", "initial_prompt_version": "relevance-v6", "adjudication_prompt_version": "relevance-verification-v6"})
        return pd.DataFrame(rows)
    pipeline = pipeline_module.Pipeline(settings)
    monkeypatch.setattr(pipeline, "_score_relevance_candidates", fake_score)
    report = pipeline.relevance_rescore_fresh_holdout()
    assert report["passes_targets"]
    result_dir = settings.output_dir / "relevance_v4_fresh_holdout"
    assert (result_dir / "input_hashes.json").exists()
    assert reviewer_path.read_bytes() == before_reviewer
    assert (settings.output_dir / "relevance.parquet").read_bytes() == before_production
    with pytest.raises(FileExistsError):
        pipeline.relevance_rescore_fresh_holdout()

def test_policy_alignment_finalize_validates_and_preserves_append_only_audit(tmp_path):
    from marathon_absa import pipeline as pipeline_module
    settings = replace(SETTINGS, output_dir=tmp_path / "processed", cache_dir=tmp_path / "cache")
    settings.ensure_dirs()
    reviewer = pd.DataFrame([{"alignment_id": f"align-{i+1:03d}", "document_id": f"d{i}", "second_coder_label": "relevant" if i % 2 else "irrelevant", "visual_context_required": "no", "reason_category": "concrete_klscm_journey" if i % 2 else "generic_running", "evidence_span": f"evidence {i}", "coder_notes": "", "original_text": f"caption {i}"} for i in range(60)])
    key = pd.DataFrame([{"alignment_id": f"align-{i+1:03d}", "document_id": f"d{i}", "original_human_label": "relevant" if i % 3 else "irrelevant", "policy_overlay_label": "relevant" if i % 2 else "irrelevant", "alignment_stratum": "test"} for i in range(60)])
    reviewer.to_csv(settings.output_dir / "relevance_policy_alignment_second_coder.csv", index=False)
    key.to_csv(settings.output_dir / "relevance_policy_alignment_key.csv", index=False)
    pipeline = pipeline_module.Pipeline(settings)
    disagreements = pipeline.relevance_policy_alignment_finalize()
    audit_path = settings.output_dir / "relevance_policy_alignment_audit.csv"
    before = audit_path.read_bytes()
    assert {"adjudicated_label", "adjudication_reason", "adjudicated_at"}.issubset(pd.read_csv(audit_path).columns)
    assert len(disagreements) > 0
    with pytest.raises(FileExistsError):
        pipeline.relevance_policy_alignment_finalize()
    assert audit_path.read_bytes() == before

def test_pipeline_rule_only_run_creates_auditable_outputs_without_api(tmp_path, monkeypatch):
    from marathon_absa import pipeline as pipeline_module

    class NoApi:
        def __init__(self, *args, **kwargs):
            pass

        def structured(self, *args, **kwargs):
            raise AssertionError("API must not be called for deterministic records")

    monkeypatch.setattr(pipeline_module, "CachedOpenAI", NoApi)
    settings = replace(SETTINGS, output_dir=tmp_path / "processed", cache_dir=tmp_path / "cache")
    settings.ensure_dirs()
    documents = pd.DataFrame([{
        "document_id": "hash-only", "source": "instagram", "processing_status": "ready",
        "event_year": "2024", "primary_language": "insufficient_text",
        "original_text": "#klscm2024", "linguistic_text": "", "hashtags": ["klscm2024"],
        "language_status": "insufficient_text",
    }])
    write_table(documents, settings.output_dir / "documents.parquet")
    pipeline_module.Pipeline(settings).relevance()
    relevance = read_table(settings.output_dir / "relevance.parquet")
    queue = pd.read_csv(settings.output_dir / "relevance_review_queue.csv", keep_default_na=False)
    assert relevance.loc[0, "reason_code"] == "hashtag_only"
    assert relevance.loc[0, "review_status"] == "auto_excluded"
    assert relevance.loc[0, "exclusion_reason"] == "hashtag_only"
    assert queue.empty
    assert not bool(relevance.loc[0, "include_in_topics"])




