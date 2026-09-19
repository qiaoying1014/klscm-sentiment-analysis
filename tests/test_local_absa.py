import json
from pathlib import Path

import pandas as pd
import pytest

from marathon_absa.local_absa import (
    ABSA_ROOT, AUDIT_JSON, CONFIG_PATH, DATASET_PATH, GOLD_PATH, ONTOLOGY_PATH,
    SAMPLE_PATH, SENTIMENTS, _derived_documents, baseline_evaluate, baseline_train,
    MODEL_REVISION, gpu_preflight, ontology, reject_overlap, sha256,
)


def test_gpu_preflight_refuses_cpu_neural_execution():
    result = gpu_preflight(require_cuda=False)
    if not result["passed"]:
        with pytest.raises(RuntimeError, match="CUDA"):
            gpu_preflight(require_cuda=True)
    else:
        assert result["selected_device"] == result["tensor_x_device"] == result["tensor_y_device"] == "cuda:0"
        assert result["model_execution_device"] == "cuda:0"


def test_training_and_inference_require_cuda(monkeypatch):
    def fail(*, require_cuda=True):
        raise RuntimeError("CUDA required")
    monkeypatch.setattr("marathon_absa.local_absa.gpu_preflight", fail)
    with pytest.raises(RuntimeError, match="CUDA"): baseline_train()
    with pytest.raises(RuntimeError, match="CUDA"): baseline_evaluate()


def test_frozen_ontology_and_sentiment_schema_are_preserved():
    aspects = ontology()
    assert len(aspects) == len({x["id"] for x in aspects}) == 20
    gold = pd.read_csv(GOLD_PATH)
    assert set(gold.aspect) <= {x["id"] for x in aspects}
    assert set(gold.sentiment) <= set(SENTIMENTS) == {"positive", "neutral", "negative", "mixed"}


def test_provenance_overlap_is_rejected():
    ids = set(pd.read_csv(SAMPLE_PATH, dtype={"document_id": str}).document_id)
    with pytest.raises(ValueError, match="overlap"):
        reject_overlap(ids, ids)
    reject_overlap({"train"}, {"evaluation"})


def test_multilabel_representation_preserves_zero_and_rare_aspects():
    data = _derived_documents()
    columns = [c for c in data if c.startswith("aspect__")]
    assert len(data) == data.document_id.nunique() == 80 and len(columns) == 20
    assert (~data.has_any_aspect).sum() == 24
    assert all(column in data for column in columns)
    assert (data[columns].sum(axis=1) > 1).any()


def test_created_artifacts_are_isolated_reproducible_and_preflight_only():
    report = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert report["random_seed"] == config["seed"] == 20260821
    assert report["openai_api_calls"] == report["paid_api_calls"] == 0
    assert report["data"]["clean_supervised_training_documents"] == 0
    assert report["leakage"]["overlap_detected_if_used_for_both"] == 80
    assert config["maximum_documents"] == 80 and config["full_corpus_run_allowed"] is False
    assert DATASET_PATH.parent != GOLD_PATH.parent and DATASET_PATH != GOLD_PATH
    assert ABSA_ROOT not in DATASET_PATH.parents


def test_completed_experiment_is_pinned_capped_and_uses_frozen_matching_inputs():
    root=DATASET_PATH.parent/"zero_shot_nli_v1"
    manifest=json.loads((root/"experiment_manifest_v1.json").read_text(encoding="utf-8"))
    metrics=json.loads((root/"local_absa_evaluation_v1.json").read_text(encoding="utf-8"))
    assert manifest["document_count"] == metrics["document_count"] == 80
    assert manifest["revision"] == metrics["revision"] == MODEL_REVISION
    assert manifest["threshold_grid"] == [0.5,0.6,0.7,0.8,0.9]
    assert manifest["per_aspect_thresholds"] is False
    assert manifest["gold_sha256"] == sha256(GOLD_PATH)
    assert manifest["ontology_sha256"] == sha256(ONTOLOGY_PATH)
    assert metrics["aspect_detection"]["tp"]+metrics["aspect_detection"]["fn"] == 100
    assert metrics["evidence_extraction_supported"] is False
    assert metrics["openai_api_calls"] == metrics["paid_api_calls"] == 0


def test_local_path_has_no_openai_or_hosted_inference_client():
    source=Path("marathon_absa/local_absa.py").read_text(encoding="utf-8")
    assert "from openai" not in source and "OpenAI(" not in source
    assert "InferenceClient" not in source
