import json
import hashlib
import builtins

import pandas as pd
import pytest

import marathon_absa.absa_v1 as absa_module
from marathon_absa.absa_v1 import (
    ABSA_V1_INSTRUCTIONS, AUDIT_SEED, HISTORICAL_MANIFEST_PATH, HISTORICAL_SAMPLE_PATH,
    SAMPLE_PATH, aggregate_mentions, annotation_columns, annotation_paths, cost_estimate, create_single_researcher_audit,
    create_validation_sample, load_ontology, mention_schema, require_gold_frozen, save_annotations,
    save_document_annotations,
    canonicalize_evidence, evaluate_validation, finalize_batch, stable_mention_id, validate_draft_package,
    validate_mentions, validate_ontology,
    validation_metrics,
)


def test_ontology_schema_and_controlled_size():
    ontology = load_ontology()
    validate_ontology(ontology)
    assert len(ontology["aspects"]) == 20
    assert ontology["aspects"][-1]["id"] == "emerging_other"


def test_zero_and_multiple_mentions_with_different_sentiments():
    ids = [a["id"] for a in load_ontology()["aspects"]]
    assert validate_mentions({"mentions": []}, "KLSCM 2025", ids) == []
    caption = "The route was brutal but the volunteers were amazing."
    common = {"confidence": .9, "expression_type": "explicit", "language": "English",
              "english_gloss": "", "contributing_hashtags": [], "contributing_emoji": [],
              "emerging_aspect_name": "", "analysis_notes": "grounded"}
    result = {"mentions": [
        {**common, "aspect": "route_course", "target": "route", "sentiment": "negative", "evidence_text": "route was brutal"},
        {**common, "aspect": "volunteer_support", "target": "volunteers", "sentiment": "positive", "evidence_text": "volunteers were amazing"},
    ]}
    assert [x["sentiment"] for x in validate_mentions(result, caption, ids)] == ["negative", "positive"]


def test_evidence_multilingual_hashtag_emoji_and_mixed_semantics():
    ids = [a["id"] for a in load_ontology()["aspects"]]
    caption = "Laluan cantik tapi bukit teruk 😭 #suffer"
    mention = {"aspect": "route_course", "target": "Laluan/bukit", "sentiment": "mixed", "confidence": .92,
        "evidence_text": caption, "expression_type": "explicit", "language": "Malay-English mixed",
        "english_gloss": "Beautiful route but terrible hills", "contributing_hashtags": ["#suffer"],
        "contributing_emoji": ["😭"], "emerging_aspect_name": "", "analysis_notes": "same route has both polarities"}
    validated = validate_mentions({"mentions": [mention]}, caption, ids)
    assert validated[0]["evidence_start"] == 0 and validated[0]["sentiment"] == "mixed"
    with pytest.raises(ValueError):
        validate_mentions({"mentions": [{**mention, "evidence_text": "invented"}]}, caption, ids)


def test_schema_is_strict_and_prompt_requires_zero_mentions():
    ids = [a["id"] for a in load_ontology()["aspects"]]
    schema = mention_schema(ids)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["mentions"]["items"]["additionalProperties"] is False
    assert "zero mentions" in ABSA_V1_INSTRUCTIONS


def test_sample_stable_blind_and_scope_provenance(tmp_path):
    rows = []
    for i in range(7704):
        rows.append({"document_id": f"d{i}", "original_text": f"My run {i}", "source": "instagram",
            "event_year": 2024, "primary_language": "English", "language_status": "ok",
            "original_topic_id": i % 32, "final_topic_id": i % 32, "final_topic_name": f"T{i%32}",
            "final_topic_group": "G", "final_taxonomy_action": "keep", "text_length_chars": 10})
    corpus = tmp_path / "corpus.csv"; pd.DataFrame(rows).to_csv(corpus, index=False)
    a = create_validation_sample(corpus, tmp_path / "a", 150, 7)
    b = create_validation_sample(corpus, tmp_path / "b", 150, 7)
    assert a.document_id.tolist() == b.document_id.tolist()
    assert not any("prediction" in c for c in a.columns)
    assert {"original_topic_id", "final_topic_id", "final_topic_name", "final_topic_group"}.issubset(a.columns)


def test_annotation_backup_resume_and_aggregation(tmp_path):
    path = tmp_path / "gold.csv"
    row = {"mention_id": stable_mention_id("d1", 0, "route_course", "bad route"), "document_id": "d1"}
    save_annotations([row], path); save_annotations([row], path)
    assert pd.read_csv(path).iloc[0].document_id == "d1"
    assert list(tmp_path.glob("*.backup_*.csv"))
    mentions = pd.DataFrame([{"document_id":"d1","aspect":"route_course"}, {"document_id":"d1","aspect":"route_course"}])
    agg = aggregate_mentions(mentions, ["aspect"])
    assert agg.iloc[0].aspect_mentions == 2 and agg.iloc[0].documents_with_mention == 1


def test_cost_estimator_local_and_full_run_gate(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.csv"
    pd.DataFrame([{"document_id": str(i), "original_text": "text"} for i in range(10)]).to_csv(corpus, index=False)
    report = cost_estimate(corpus, sample_size=2)
    assert report["api_calls_made"] == 0 and report["new_api_calls_required"] == 10
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="blocked"):
        require_gold_frozen()


def test_validation_metrics_separate_aspect_sentiment_joint_and_grounding():
    gold = pd.DataFrame([{"mention_id":"g1","document_id":"d1","aspect":"route_course","sentiment":"negative"},
                         {"mention_id":"g2","document_id":"d1","aspect":"volunteer_support","sentiment":"positive"}])
    pred = pd.DataFrame([{"mention_id":"p1","document_id":"d1","aspect":"route_course","sentiment":"negative",
                          "evidence_text":"route bad","evidence_start":0,"evidence_end":9},
                         {"mention_id":"p2","document_id":"d1","aspect":"volunteer_support","sentiment":"negative",
                          "evidence_text":"volunteers great","evidence_start":14,"evidence_end":30}])
    result = validation_metrics(gold, pred, {"d1":"route bad but volunteers great"})
    assert result["aspect_detection"]["f1"] == 1
    assert result["matched_sentiment"]["accuracy"] == .5
    assert result["joint_aspect_sentiment"]["f1"] == .5
    assert result["evidence_grounding"]["exact_substring_rate"] == 1


def _audit_corpus():
    rows = []
    for i in range(7704):
        rows.append({"document_id":f"a{i}", "original_text":f"Saya loved my KLSCM run #{i} 😭. But it was long.",
            "source":"instagram" if i % 3 else "blog", "event_year":[2019, 2023, 2024, 2025][i % 4],
            "primary_language":"Malay" if i % 5 == 0 else "English", "detected_language":"Malay",
            "language_status":"mixed" if i % 7 == 0 else "ok", "original_topic_id":i % 32,
            "final_topic_id":i % 32, "final_topic_name":f"Topic {i%32}", "final_topic_group":"Group",
            "final_taxonomy_action":"keep", "text_length_chars":50 + i % 200})
    return pd.DataFrame(rows)


def test_current_audit_exact_composition_stability_scope_and_blinding(tmp_path):
    corpus = tmp_path / "corpus.csv"; _audit_corpus().to_csv(corpus, index=False)
    a = create_single_researcher_audit(corpus, tmp_path / "one", AUDIT_SEED)
    b = create_single_researcher_audit(corpus, tmp_path / "two", AUDIT_SEED)
    assert len(a) == a.document_id.nunique() == 80
    assert a.sample_component.value_counts().to_dict() == {"topic_floor":32, "difficult_content_enrichment":24, "random_component":24}
    assert a.final_topic_id.nunique() == 32
    assert a.document_id.tolist() == b.document_id.tolist()
    assert not any("prediction" in column.lower() for column in a.columns)
    assert set(a.final_topic_group) == {"Group"}


def test_default_paths_resolve_80_audit_and_historical_is_explicit():
    assert annotation_paths()["sample"] == SAMPLE_PATH
    assert "single_researcher_audit_v1" in str(SAMPLE_PATH)
    assert annotation_paths(historical_150=True)["sample"] == HISTORICAL_SAMPLE_PATH


def test_completed_document_cannot_be_overwritten(tmp_path):
    annotations = tmp_path / "annotations.csv"; progress = tmp_path / "progress.csv"
    pd.DataFrame([{"document_id":"d1", "annotation_status":"complete"}]).to_csv(progress, index=False)
    with pytest.raises(ValueError, match="immutable"):
        save_document_annotations("d1", [], annotations, progress)


def test_ai_draft_package_is_complete_but_not_human_gold(tmp_path):
    sample = tmp_path / "sample.csv"
    annotations = tmp_path / "draft.csv"
    progress = tmp_path / "progress.csv"
    pd.DataFrame([
        {"document_id": "d1", "original_text": "The route was excellent."},
        {"document_id": "d2", "original_text": "KLSCM 2025"},
    ]).to_csv(sample, index=False)
    evidence = "route was excellent"
    pd.DataFrame([{**{column: "" for column in annotation_columns()},
        "mention_id": stable_mention_id("d1", 0, "route_course", evidence), "document_id": "d1",
        "aspect": "route_course", "target": "route", "sentiment": "positive",
        "evidence_text": evidence, "evidence_start": 4, "evidence_end": 23,
        "expression_type": "explicit"}]).to_csv(annotations, index=False)
    pd.DataFrame([
        {"document_id": "d1", "annotation_status": "ai_draft_complete", "ai_draft_no_evaluative_aspect_mention": False},
        {"document_id": "d2", "annotation_status": "ai_draft_complete", "ai_draft_no_evaluative_aspect_mention": True},
    ]).to_csv(progress, index=False)
    assert validate_draft_package(sample, annotations, progress) == {
        "documents": 2, "mentions": 1, "zero_mention_documents": 1}


def test_historical_150_artifacts_remain_byte_identical():
    assert hashlib.sha256(HISTORICAL_SAMPLE_PATH.read_bytes()).hexdigest() == "b60e36b20737b43e436718113846293aa4d8f3364e8cedb6c69fa721c36ba8c1"
    assert hashlib.sha256(HISTORICAL_MANIFEST_PATH.read_bytes()).hexdigest() == "f13ca4f70ade783b5b8d8f20cecaeaf40a45846088f9e669bf7bcb0ca44b06f0"


def test_evidence_canonicalization_exact_whitespace_unicode_offset_and_case():
    exact = canonicalize_evidence("The route was great", "route was great")
    assert exact["raw_exact_match"] and not exact["evidence_repaired"]
    whitespace = canonicalize_evidence("A  good run", "A good run")
    assert whitespace["evidence_repair_method"] == "unique_whitespace_normalized_match"
    unicode_match = canonicalize_evidence("Cafe\u0301 run", "Café run")
    assert unicode_match["evidence_repair_method"] == "unique_unicode_nfc_match"
    offset = canonicalize_evidence("Great  route", "Great route", 0, 12)
    assert offset["evidence_repair_method"] == "model_offsets"
    case = canonicalize_evidence("The Route", "the route")
    assert case["evidence_repair_method"] == "unique_case_insensitive_match"
    for result, caption in [(whitespace, "A  good run"), (unicode_match, "Cafe\u0301 run"),
                            (offset, "Great  route"), (case, "The Route")]:
        assert result["final_exact_evidence_text"] in caption
        assert caption[result["final_evidence_start"]:result["final_evidence_end"]] == result["final_exact_evidence_text"]


def test_ambiguous_paraphrased_translated_and_hallucinated_evidence_are_rejected():
    ambiguous = canonicalize_evidence("A  run and A   run", "A run")
    paraphrase = canonicalize_evidence("The route was extremely difficult", "The course was very hard")
    translated = canonicalize_evidence("Laluan sangat cantik", "The route was beautiful")
    hallucinated = canonicalize_evidence("KLSCM 2025", "Volunteers were amazing")
    assert all(result["normalized_match_status"] == "unrecoverable"
               for result in [ambiguous, paraphrase, translated, hallucinated])
    assert all(result["final_exact_evidence_text"] == ""
               for result in [ambiguous, paraphrase, translated, hallucinated])


def test_safe_finalization_keeps_invalid_prediction_and_never_mutates_sources_or_calls_api(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.csv"; sample = tmp_path / "sample.csv"; gold = tmp_path / "gold.csv"
    records = [
        {"document_id":"d1", "original_text":"The route was great", "source":"instagram", "event_year":2025,
         "original_topic_id":1, "final_topic_id":1, "final_topic_name":"Race", "final_topic_group":"Experience",
         "primary_language":"English"},
        {"document_id":"d2", "original_text":"KLSCM 2025", "source":"instagram", "event_year":2025,
         "original_topic_id":1, "final_topic_id":1, "final_topic_name":"Race", "final_topic_group":"Experience",
         "primary_language":"English"},
    ]
    pd.DataFrame(records).to_csv(corpus, index=False)
    pd.DataFrame(records).to_csv(sample, index=False)
    pd.DataFrame([{"mention_id":"g1", "document_id":"d1", "aspect":"route_course",
                  "sentiment":"positive"}]).to_csv(gold, index=False)
    batch = tmp_path / "out" / "batch" / "validation"; batch.mkdir(parents=True)
    common = {"target":"route", "sentiment":"positive", "confidence":.9, "expression_type":"explicit",
              "language":"English", "english_gloss":"", "contributing_hashtags":[], "contributing_emoji":[],
              "emerging_aspect_name":"", "analysis_notes":""}
    results = [("d1", {"mentions":[{**common, "aspect":"route_course", "evidence_text":"route was great"}]}),
               ("d2", {"mentions":[{**common, "aspect":"route_course", "evidence_text":"invented route"}]})]
    raw = batch / "absa_v1_validation_raw_results.jsonl"
    raw.write_text("\n".join(json.dumps({"custom_id":f"absa1-{did}", "response":{"body":{"model":"test",
        "output":[{"content":[{"type":"output_text", "text":json.dumps(payload)}]}]}}}) for did,payload in results), encoding="utf-8")
    monkeypatch.setattr(absa_module, "CORPUS", corpus); monkeypatch.setattr(absa_module, "SAMPLE_PATH", sample)
    monkeypatch.setattr(absa_module, "GOLD_PATH", gold)
    raw_hash = hashlib.sha256(raw.read_bytes()).hexdigest(); gold_hash = hashlib.sha256(gold.read_bytes()).hexdigest()
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name == "openai":
            raise AssertionError("No API import is allowed during finalization")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    path = finalize_batch("validation", tmp_path / "out")
    predictions = pd.read_csv(path).fillna("")
    assert len(predictions) == 2 and (predictions.prediction_error == "unrecoverable_evidence").sum() == 1
    metrics = json.loads((tmp_path / "out" / "absa_validation_metrics_v1.json").read_text())
    grounding = metrics["evidence_grounding"]
    assert grounding["strict_raw_model_exact_count"] == 1
    assert grounding["final_recoverable_exact_span_count"] == 1
    assert metrics["aspect_detection"]["fp"] == 1  # Invalid evidence prediction is not silently dropped.
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == raw_hash
    assert hashlib.sha256(gold.read_bytes()).hexdigest() == gold_hash
    first_prediction_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    assert finalize_batch("validation", tmp_path / "out") == path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == first_prediction_hash


def test_standalone_frozen_evaluation_reports_confusion_subgroups_and_is_immutable(tmp_path, monkeypatch):
    audit = tmp_path / "audit"; audit.mkdir(); output = tmp_path / "out"; output.mkdir()
    sample_path = audit / "sample.csv"; gold_path = audit / "gold.csv"; marker = audit / "gold.frozen.json"
    rows = [{"document_id":f"d{i}", "original_text":"The route was great" if i == 0 else "KLSCM 2025",
             "primary_language":"English" if i < 40 else "Malay"} for i in range(80)]
    pd.DataFrame(rows).to_csv(sample_path, index=False)
    pd.DataFrame([{"mention_id":"g1", "document_id":"d0", "aspect":"route_course",
                  "sentiment":"positive"}]).to_csv(gold_path, index=False)
    marker.write_text("{}", encoding="utf-8")
    predictions_path = output / "absa_validation_predictions_v1.csv"
    pd.DataFrame([{"mention_id":"p1", "document_id":"d0", "aspect":"route_course", "sentiment":"positive",
                  "evidence_text":"route was great", "evidence_start":4, "evidence_end":19,
                  "raw_exact_match":True}]).to_csv(predictions_path, index=False)
    ontology = tmp_path / "ontology.json"; ontology.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(absa_module, "SAMPLE_PATH", sample_path); monkeypatch.setattr(absa_module, "GOLD_PATH", gold_path)
    monkeypatch.setattr(absa_module, "AUDIT_ROOT", audit); monkeypatch.setattr(absa_module, "ONTOLOGY_PATH", ontology)
    (audit / "absa_v1_audit_gold_v1.frozen.json").write_text("{}", encoding="utf-8")
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in [sample_path, gold_path, predictions_path, ontology]}
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name == "openai": raise AssertionError("Evaluation must not import the API client")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    report_path = evaluate_validation(output)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["population"]["gold_documents_total"] == 80
    assert report["population"]["gold_mentions"] == report["population"]["predicted_mentions"] == 1
    assert report["metrics"]["aspect_detection"]["f1"] == 1
    confusion = report["metrics"]["document_diagnostics"]["zero_mention_confusion_gold_rows_predicted_columns"]
    assert confusion == {"gold_zero_predicted_zero":79, "gold_zero_predicted_mentions":0,
                         "gold_mentions_predicted_zero":0, "gold_mentions_predicted_mentions":1}
    assert report["metrics"]["document_diagnostics"]["multilingual_subgroups"]["Malay"]["metrics_reported"]
    assert (output / "absa_validation_evaluation_summary_v1.csv").exists()
    assert (output / "absa_validation_evaluation_subgroups_v1.csv").exists()
    assert evaluate_validation(output) == report_path
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in hashes.items())
