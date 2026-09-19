import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from marathon_absa.absa_aspect_ownership_development import ROOT as V3_ROOT
from marathon_absa.absa_production import (
    CONTEXTUAL_CORPUS, EXPECTED_DOCUMENTS, EXPECTED_PROMPT_SHA256,
    FREEZE_PATH, ONTOLOGY_PATH, OUTLIER_CORPUS, PRODUCTION_CACHE_STAGE,
    LOCAL_EVALUATION_PATH, MODEL_SELECTION_JSON, MODEL_SELECTION_MD, POPULATION_PATH, PRE_SUBMISSION_AUDIT_JSON,
    REQUEST_MANIFEST_PATH, REQUEST_PATH, ROOT, SCHEMA_VERSION, create_retry_package, finalize_production,
    V3_PROMPT_PATH, _base_document_result, _population, _reconcile_response_ids, _sha,
    _validate_prepared,
)


def test_production_population_is_exact_frozen_substantive_partition():
    source=_population();ids=set(source.document_id)
    contextual=set(pd.read_csv(CONTEXTUAL_CORPUS,usecols=["document_id"]).document_id.astype(str))
    outliers=set(pd.read_csv(OUTLIER_CORPUS,usecols=["document_id"]).document_id.astype(str))
    assert len(source)==source.document_id.nunique()==EXPECTED_DOCUMENTS==7704
    assert len(contextual)==336 and len(outliers)==5703
    assert EXPECTED_DOCUMENTS+len(contextual)+len(outliers)==13743
    assert not ids&contextual and not ids&outliers


def test_production_freeze_records_known_limitation_and_immutable_configuration():
    freeze=json.loads(FREEZE_PATH.read_text(encoding="utf-8"));manifest=json.loads(REQUEST_MANIFEST_PATH.read_text(encoding="utf-8"))
    normalized_hash=hashlib.sha256(V3_PROMPT_PATH.read_text(encoding="utf-8").encode()).hexdigest()
    assert normalized_hash==freeze["selected_prompt_sha256"]==manifest["prompt_sha256"]==EXPECTED_PROMPT_SHA256
    assert freeze["model"]==manifest["model"]=="gpt-5.6-luna"
    assert freeze["schema"]==manifest["schema"]==SCHEMA_VERSION=="absa_mention_schema_v1"
    assert freeze["ontology_sha256"]==manifest["ontology_sha256"]==_sha(ONTOLOGY_PATH)
    assert freeze["development_decision"]=="INSUFFICIENT_PRECISION_IMPROVEMENT"
    assert freeze["selection_status"]=="selected_for_production_with_known_development_limitation"
    assert freeze["development_evidence"] is True and freeze["confirmatory_evidence"] is False
    assert freeze["manual_review_closed_for_current_phase"] is True
    assert freeze["additional_prompt_tuning_deferred"] is True and freeze["V4_created"] is False
    assert freeze["full_corpus_inference_started"] is False and manifest["submitted"] is False
    assert freeze["full_corpus_submitted"] is False
    assert freeze["local_experiment_status"] == "completed"
    assert freeze["local_baseline_viability"] == "LOCAL_NOT_CURRENTLY_VIABLE"
    assert freeze["local_selected_for_production"] is False
    assert freeze["additional_local_model_development_deferred"] is True
    assert PRODUCTION_CACHE_STAGE not in {"absa_v1_mentions","absa_v1_mentions_precision_development_v1","absa_v1_mentions_aspect_ownership_development_v1"}


def test_production_requests_reconcile_and_match_v3_inference_configuration():
    _,manifest,_=_validate_prepared();seen=set();first=None
    with REQUEST_PATH.open(encoding="utf-8") as handle:
        count=0
        for line in handle:
            item=json.loads(line);count+=1;custom=item["custom_id"]
            assert custom not in seen;seen.add(custom)
            if first is None:first=item
    assert count==len(seen)==manifest["request_count"]==7704
    assert manifest["request_jsonl_sha256"]==_sha(REQUEST_PATH)
    assert manifest["request_jsonl_bytes"]==REQUEST_PATH.stat().st_size
    v3=json.loads(next(iter((V3_ROOT/"batch"/"absa_v1_aspect_ownership_requests.jsonl").read_text(encoding="utf-8").splitlines())))
    for key in ["model","instructions","text"]:assert first["body"][key]==v3["body"][key]
    assert first["method"]==v3["method"]=="POST" and first["url"]==v3["url"]=="/v1/responses"
    assert manifest["request_jsonl_bytes"]<200_000_000 and manifest["batch_chunk_count"]==1


def test_model_selection_closes_local_benchmark_without_mutating_it():
    selection=json.loads(MODEL_SELECTION_JSON.read_text(encoding="utf-8"))
    assert MODEL_SELECTION_MD.exists()
    assert selection["selected"] == "OpenAI V3"
    assert selection["local_experiment_status"] == "completed"
    assert selection["local_viability"] == "LOCAL_NOT_CURRENTLY_VIABLE"
    assert selection["local_selected_for_production"] is False
    assert selection["local_baseline_retained_as_research_benchmark"] is True
    assert selection["local_evaluation_sha256"] == _sha(LOCAL_EVALUATION_PATH)
    assert selection["request_jsonl_sha256"] == _sha(REQUEST_PATH)
    assert selection["submitted"] is False and selection["api_calls"] == 0


def test_response_reconciliation_and_zero_document_preservation_helpers():
    _reconcile_response_ids({"a","b"},["a","b"])
    with pytest.raises(ValueError,match="Duplicate"):_reconcile_response_ids({"a","b"},["a","a"])
    with pytest.raises(ValueError,match="mismatch"):_reconcile_response_ids({"a","b"},["a","c"])
    row=SimpleNamespace(document_id="a",source="instagram",event_year=2025,primary_language="English",
                        original_topic_id=2,original_human_topic_name="Raw",final_topic_id=1,
                        final_topic_name="Topic",final_topic_group="Group")
    result=_base_document_result(row)
    assert result["document_id"]=="a" and result["mention_count"]==0
    assert result["prediction_status"]=="zero_mentions" and result["parse_status"]=="success"
    assert result["source"]=="instagram" and result["raw_c1_topic_id"]==2 and result["taxonomy_status"]=="substantive_absa_ready"


def test_pre_submission_audit_reconciles_schema_metadata_and_offline_guarantees():
    audit=json.loads(PRE_SUBMISSION_AUDIT_JSON.read_text(encoding="utf-8"))
    assert audit["verdict"]=="SAFE_TO_SUBMIT_WITH_OFFLINE_DOWNSTREAM_ANALYSIS"
    assert audit["request_count"]==audit["unique_custom_ids"]==7704
    assert audit["request_jsonl_sha256"]==_sha(REQUEST_PATH)
    assert audit["request_jsonl_bytes"]==REQUEST_PATH.stat().st_size
    assert not audit["malformed_requests"] and not audit["empty_captions"] and not audit["configuration_mismatches"]
    assert set(audit["schema"]["fields"])==set(audit["schema"]["required_fields"])
    assert audit["schema"]["model_returns_numeric_evidence_offsets"] is False
    assert audit["schema"]["evidence_extraction_sufficient_for_planned_analysis"] is True
    assert all(audit["checks"].values()) and audit["api_calls"]==0
    assert all(value for key,value in audit["downstream_offline"].items() if key!="aggregation_visualization_statistics_topic_grouping_reporting_changes_require_inference")
    assert audit["downstream_offline"]["aggregation_visualization_statistics_topic_grouping_reporting_changes_require_inference"] is False


def test_finalizer_is_offline_idempotent_and_never_deletes_raw_files():
    source=inspect.getsource(finalize_production)
    assert "OpenAI" not in source and "files.content" not in source
    assert "unlink(" not in source and "Remove-Item" not in source
    assert "FINALIZATION_MANIFEST_PATH" in source and "raw_files_deleted\":False" in source
    assert "PARSED_RESPONSES_PATH" in source and "validate_json_schema" in source


def test_affected_document_retry_copies_only_frozen_requests(tmp_path,monkeypatch):
    root=tmp_path/"production";(root/"batch").mkdir(parents=True)
    request=root/"batch"/REQUEST_PATH.name
    request.write_text('\n'.join(json.dumps({"custom_id":f"absa1prod-{doc}","body":{}}) for doc in ["a","b","c"])+"\n",encoding="utf-8")
    (root/"absa_v1_production_response_integrity_v1.json").write_text(json.dumps({"missing_document_ids":["b"],"duplicate_document_ids":[]}),encoding="utf-8")
    monkeypatch.setattr("marathon_absa.absa_production._validate_prepared",lambda value:( {},{"request_jsonl_sha256":_sha(request)},request))
    retry=create_retry_package(root);lines=retry.read_text(encoding="utf-8").splitlines()
    assert len(lines)==1 and json.loads(lines[0])["custom_id"]=="absa1prod-b"
    manifest=json.loads((retry.parent/"retry_manifest.json").read_text(encoding="utf-8"))
    assert manifest["request_count"]==1 and manifest["submitted"] is False and manifest["api_calls"]==0


def test_production_namespace_cannot_overwrite_development_and_no_v4_exists():
    assert ROOT.parent.name=="production" and V3_ROOT.parent.name=="development"
    assert not ROOT.is_relative_to(V3_ROOT)
    checked=[Path("marathon_absa"),Path("data/processed/absa_v1/development"),ROOT]
    assert not any("instructions_4" in path.name.lower() for base in checked for path in base.rglob("*"))
