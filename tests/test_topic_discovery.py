import numpy as np
import pandas as pd
import pytest

from marathon_absa.topic_discovery import (
    REVIEW_LABELS, build_c1_review_frames, build_topic_corpus, default_config, multilingual_tokenizer,
    normalize_hashtag, quality_reason, refined_text_fields, semantic_topic_text,
    preserve_topic_annotations, prepare_model_selection_review,
    topic_review_summary, validate_embedding_cache, write_model_selection_summary,
)


def _records(texts, decisions=None):
    decisions = decisions or ["review"] * len(texts)
    decision_rows, document_rows = [], []
    for i, (text, decision) in enumerate(zip(texts, decisions)):
        document_id = f"id-{i}"
        decision_rows.append({"document_id": document_id, "original_text": text, "source": "instagram", "event_year": 2024, "primary_language": "English", "v8_automatic_decision": decision, "v8_routing_status": "test"})
        document_rows.append({"document_id": document_id, "normalized_text": text, "linguistic_text": text, "semantic_text": text, "hashtags": "[]", "detected_language": "English"})
    return pd.DataFrame(decision_rows), pd.DataFrame(document_rows)


def test_quality_rules_preserve_short_meaningful_text_and_hashtags():
    for value in ["First KLSCM!", "PB 21KM", "KLSCM finisher", "Hot and humid", "See you KLSCM2024"]:
        semantic, _ = semantic_topic_text(value)
        assert quality_reason(value, semantic) == ""
    semantic, tags = semantic_topic_text("#KLSCM2024 #HalfMarathon #21KM #PB")
    assert tags == ["KLSCM2024", "HalfMarathon", "21KM", "PB"]
    assert "Half Marathon" in semantic and "KLSCM 2024" in semantic


@pytest.mark.parametrize("text,reason", [("", "empty_text"), ("https://x.test", "url_only"), ("@runner", "mention_only"), ("!!!", "punctuation_only")])
def test_quality_exclusions(text, reason):
    semantic, _ = semantic_topic_text(text)
    assert quality_reason(text, semantic) == reason


def test_relevance_is_metadata_not_gate_and_ids_reconcile():
    decisions, documents = _records(["meaningful text", "review text", "excluded but useful"], ["include", "review", "exclude"])
    result = build_topic_corpus(decisions, documents)
    assert set(result.document_id) == {"id-0", "id-1", "id-2"}
    assert result.include_in_topic_discovery.all()
    assert result.relevance_decision.tolist() == ["include", "review", "exclude"]


def test_exact_duplicates_map_to_representative_without_losing_rows():
    decisions, documents = _records(["Same semantic text", "Same semantic text", "Different"])
    result = build_topic_corpus(decisions, documents)
    assert len(result) == 3
    duplicate = result[result.data_quality_exclusion_reason.eq("exact_duplicate_semantic_text")].iloc[0]
    assert duplicate.duplicate_representative_id == "id-0"


def test_multilingual_chinese_tokenizer_is_not_whitespace_dependent():
    tokens = multilingual_tokenizer("KLSCM 马拉松 很热")
    assert "klscm" in tokens and "马拉" in tokens and "拉松" in tokens


def test_reproducible_configuration_and_absa_block():
    config = default_config()
    assert config["random_seed"] == 42
    assert config["umap"]["random_state"] == 42
    assert config["bertopic"]["nr_topics"] is None


def test_embedding_cache_alignment():
    validate_embedding_cache(["a", "b"], np.zeros((2, 3)))
    with pytest.raises(ValueError):
        validate_embedding_cache(["a"], np.zeros((2, 3)))


def test_hashtag_normalization_is_conservative():
    assert normalize_hashtag("#PersonalBest") == "Personal Best"
    assert normalize_hashtag("#KLSCM2024") == "KLSCM 2024"
    assert normalize_hashtag("#klscm") == "klscm"


def _topic_review_inputs():
    rows, assignments, info, reps = [], [], [], {}
    for topic_id in range(40):
        size = 6 + topic_id
        info.append({"Topic": topic_id, "Name": f"topic-{topic_id}", "Representation": [f"term-{topic_id}"]})
        reps[topic_id] = [f"representative {topic_id}-{i}" for i in range(10)]
        for i in range(size):
            document_id = f"{topic_id}-{i}"
            rows.append({"document_id": document_id, "include_in_topic_discovery": True, "original_caption_text": f"caption {document_id}", "primary_language": "English", "event_year": 2024, "relevance_decision": "review", "source": "instagram"})
            assignments.append({"document_id": document_id, "topic_id": topic_id})
    for i in range(12):
        document_id = f"outlier-{i}"
        rows.append({"document_id": document_id, "include_in_topic_discovery": True, "original_caption_text": f"outlier {i}", "primary_language": "Malay", "event_year": 2019, "relevance_decision": "include", "source": "instagram"})
        assignments.append({"document_id": document_id, "topic_id": -1})
    info.append({"Topic": -1, "Name": "outlier", "Representation": ["outlier"]})
    return pd.DataFrame(rows), pd.DataFrame(assignments), pd.DataFrame(info), reps


def test_c1_review_has_40_topics_outliers_separate_and_reproducible_samples():
    inputs = _topic_review_inputs()
    first = build_c1_review_frames(*inputs)
    second = build_c1_review_frames(*inputs)
    review, outlier, largest = first
    assert review.topic_id.tolist() == list(range(40))
    assert -1 not in set(review.topic_id) and outlier.iloc[0].topic_id == -1
    assert review.representative_documents.map(lambda value: len(__import__("json").loads(value))).eq(5).all()
    assert review.random_documents.tolist() == second[0].random_documents.tolist()
    assert len(largest) == 5
    assert largest.representative_documents.map(lambda value: len(__import__("json").loads(value))).eq(10).all()
    assert largest.random_documents.map(lambda value: len(__import__("json").loads(value))).eq(20).all()


def test_existing_topic_annotations_are_never_overwritten():
    fresh = build_c1_review_frames(*_topic_review_inputs())[0]
    existing = fresh.copy()
    existing.loc[0, ["topic_relevance", "topic_name", "topic_quality", "notes", "reviewed_at"]] = [REVIEW_LABELS[0], "Finish-line experience", "coherent", "keep", "2026-08-17T00:00:00Z"]
    resumed = preserve_topic_annotations(fresh, existing)
    assert resumed.loc[0, "topic_name"] == "Finish-line experience"
    assert resumed.loc[0, "notes"] == "keep"


def test_review_summary_uses_assignment_counts_and_separates_outliers(tmp_path):
    _, assignments, _, _ = _topic_review_inputs()
    review = build_c1_review_frames(*_topic_review_inputs())[0]
    review.loc[0, ["topic_relevance", "topic_name", "topic_quality"]] = ["substantive_klsm_topic", "Theme", "coherent"]
    review_path, assignments_path = tmp_path / "review.csv", tmp_path / "assignments.csv"
    review.to_csv(review_path, index=False)
    assignments.to_csv(assignments_path, index=False)
    summary = topic_review_summary(review_path, assignments_path)
    assert summary["topics_reviewed"] == 1
    assert summary["categories"]["substantive_klsm_topic"]["documents"] == 6
    assert summary["outliers"]["documents"] == 12


def test_topic_review_implementation_has_no_api_client_and_absa_stays_blocked():
    import inspect
    import marathon_absa.topic_discovery as module
    source = inspect.getsource(module.prepare_c1_topic_review)
    assert "OpenAI" not in source and "CachedOpenAI" not in source
    assert default_config()["bertopic"]["nr_topics"] is None


def test_refined_text_separates_embedding_and_representation_cleaning():
    embedding, representation = refined_text_fields("K L S C M 2 0 1 9 and saya dah #HalfMarathon #HalfMarathon 💪📸😊")
    assert "KLSCM 2019" in embedding
    assert "Half Marathon" in embedding
    assert embedding.count("Half Marathon") == 1
    assert "flexed_biceps" not in embedding and "camera_with_flash" not in embedding
    assert "smiling_face" not in embedding
    assert "and" not in representation.split()
    assert "saya" not in representation.split() and "dah" not in representation.split()
    assert "klscm" in representation.split() and "marathon" in representation.split()


def test_refined_hashtag_cap_keeps_domain_signals():
    generic = " ".join(f"#generic{i}" for i in range(20))
    embedding, _ = refined_text_fields(generic + " #KLSCM2025 #42KM")
    assert "KLSCM 2025" in embedding and "42 KM" in embedding
    assert len(embedding.split()) <= 28


def test_model_selection_summary_is_pending_and_then_conservative(tmp_path):
    rows = []
    for topic_id in range(4):
        rows.append({
            "refined_topic_id": topic_id, "topic_size": 100,
            "interpretability_change": "", "semantic_relation": "",
            "refined_topic_quality": "", "note": "", "reviewed_at": "",
        })
    path = tmp_path / "review.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    pending = write_model_selection_summary(path, tmp_path)
    assert pending["recommendation"] == "pending_compact_review"
    complete = pd.read_csv(path, keep_default_na=False)
    complete["interpretability_change"] = ["improved", "improved", "improved", "worse"]
    complete["semantic_relation"] = ["merges_related_topics", "splits_theme_usefully", "recovers_useful_outliers", "unclear"]
    complete["refined_topic_quality"] = "coherent"
    complete.to_csv(path, index=False)
    result = write_model_selection_summary(path, tmp_path)
    assert result["recommendation"] == "recommend_c1_refined"


def test_model_selection_summary_rejects_newer_model_without_clear_gain(tmp_path):
    frame = pd.DataFrame({
        "refined_topic_id": [0, 1], "topic_size": [100, 100],
        "interpretability_change": ["similar", "worse"],
        "semantic_relation": ["preserves_theme", "mixes_unrelated_topics"],
        "refined_topic_quality": ["coherent", "highly_mixed"], "note": ["", ""], "reviewed_at": ["x", "x"],
    })
    path = tmp_path / "review.csv"; frame.to_csv(path, index=False)
    assert write_model_selection_summary(path, tmp_path)["recommendation"] == "recommend_keep_frozen_c1"
