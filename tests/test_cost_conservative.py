from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from marathon_absa.config import SETTINGS
from marathon_absa.cost_conservative import (
    CONFIG_VERSION, build_production_report, cache_key, create_batch, direct_review_result, import_batch,
    initially_relevant_candidate, record_context, terra_required, verification_triggers,
)
from marathon_absa.single_researcher_audit import calculate_metrics


def assessment(connection="explicit", content="event_participation", confidence=.95, **overrides):
    value={"event_connection":connection,"primary_content_type":content,"secondary_content_types":[],
        "meaningful_content_present":content!="no_meaningful_content","event_link_evidence":"KLSCM" if connection=="explicit" else "#KLSCM" if connection in {"supported","weak"} else "",
        "analytical_content_evidence":"finished" if content!="no_meaningful_content" else "","confidence":confidence,
        "contradiction_present":False,"contradiction_note":"","image_dependent":False,"requires_review_recommendation":False,
        "review_reason":"","language_observed":"English","code_switching_note":"","short_explanation":"grounded"}
    value.update(overrides); return value


def test_terra_eligibility_is_triggered_and_substantively_relevant():
    relevant=assessment(confidence=.90)
    assert verification_triggers(relevant)==["confidence_below_0_93"]
    assert initially_relevant_candidate(relevant) and terra_required(relevant)
    excluded=assessment("none","generic_running",.99)
    assert verification_triggers(excluded)
    assert not initially_relevant_candidate(excluded) and not terra_required(excluded)


def test_direct_review_can_never_auto_exclude():
    value=assessment("none","generic_running",.99)
    result=direct_review_result(value,verification_triggers(value))
    assert result["v8_automatic_decision"]=="review"
    assert result["v8_final_inclusion"]=="unresolved"
    assert result["terra_verification_skipped"] is True
    assert result["operational_config_version"]==CONFIG_VERSION


def test_nontriggered_candidate_keeps_normal_behavior():
    assert verification_triggers(assessment(confidence=.99))==[]
    assert not terra_required(assessment(confidence=.99))


def test_cost_conservative_audit_allows_two_but_not_three_false_exclusions():
    rows=[]
    for decision in ["include","exclude"]:
        for i in range(75):
            rows.append({"automatic_decision":decision,"researcher_label":decision,"stratum_population_n":75,"stratum_sample_n":75})
    frame=pd.DataFrame(rows)
    excluded=frame.index[frame.automatic_decision.eq("exclude")]
    frame.loc[excluded[:2],"researcher_label"]="include"
    assert calculate_metrics(frame,2)["acceptance"]["audit_acceptable"]
    frame.loc[excluded[2],"researcher_label"]="include"
    assert not calculate_metrics(frame,2)["acceptance"]["audit_acceptable"]


def test_production_reporting_distinguishes_processing_paths_from_review_outcomes():
    frame = pd.DataFrame([
        {"v8_automatic_decision":"exclude", "v8_routing_status":"automatic_exclude_deterministic", "review_route_source":"deterministic"},
        {"v8_automatic_decision":"review", "v8_routing_status":"pending_review", "review_route_source":"deterministic"},
        {"v8_automatic_decision":"include", "v8_routing_status":"automatic_include", "review_route_source":"post_terra"},
        {"v8_automatic_decision":"review", "v8_routing_status":"pending_review", "review_route_source":"post_terra"},
        {"v8_automatic_decision":"review", "v8_routing_status":"pending_review", "review_route_source":"verification_triggered_terra_ineligible"},
    ])
    report = build_production_report(frame)
    assert report["processing_paths"]["deterministic_no_initial_total"] == 2
    assert report["processing_paths"]["deterministic_human_review"] == 1
    assert report["processing_paths"]["terra_processed_eligible_total"] == 2
    assert report["processing_paths"]["terra_processed_human_review"] == 1
    assert report["routing_reconciliation"]["automatic_includes_plus_automatic_excludes_plus_human_review"] == 5


def test_initial_batch_is_deterministic_valid_and_refuses_overwrite(tmp_path):
    output=tmp_path/"processed"; cache=tmp_path/"cache"; output.mkdir(); cache.mkdir()
    docs=pd.DataFrame([
        {"document_id":"d1","source":"instagram","processing_status":"ready","event_year":"2024","original_text":"KLSCM finish","linguistic_text":"KLSCM finish","hashtags":[],"primary_language":"English","language_status":"ok"},
        {"document_id":"d2","source":"instagram","processing_status":"ready","event_year":"2024","original_text":"#KLSCM","linguistic_text":"","hashtags":["KLSCM"],"primary_language":"English","language_status":"no_text"},
        {"document_id":"d3","source":"instagram","processing_status":"ready","event_year":"2024","original_text":"KLSCM proud","linguistic_text":"KLSCM proud","hashtags":[],"primary_language":"English","language_status":"ok"},
    ])
    docs.to_parquet(output/"documents.parquet",index=False)
    settings=replace(SETTINGS,output_dir=output,cache_dir=cache)
    context=record_context(next(docs[docs.document_id.eq("d1")].itertuples(index=False)))
    cache_path=cache/"relevance_v8_initial"/f"{cache_key('relevance_v8_initial',settings.chat_model,context)}.json"
    cache_path.parent.mkdir(parents=True); cache_path.write_text(json.dumps({"result":assessment(),"metadata":{}}),encoding="utf-8")
    manifest=create_batch("initial",settings,max_estimated_tokens=3000)
    assert manifest["requests_generated"]==1 and manifest["deterministic_no_initial"]==1 and manifest["reusable_cache_hits"]==1
    batch_dir=output/"relevance_v8_single_researcher_cost_conservative"/"batch"/"initial"
    lines=[json.loads(line) for line in next(batch_dir.glob("*.jsonl")).read_text(encoding="utf-8").splitlines()]
    assert lines[0]["url"]=="/v1/responses"
    assert lines[0]["body"]["text"]["format"]["strict"] is True
    assert len({line["custom_id"] for line in lines})==1
    assert manifest["files"]
    with pytest.raises(FileExistsError): create_batch("initial",settings)


def test_batch_import_rejects_unknown_and_accepts_valid_result(tmp_path):
    output=tmp_path/"processed"; cache=tmp_path/"cache"; output.mkdir(); cache.mkdir()
    docs=pd.DataFrame([{"document_id":"d1","source":"instagram","processing_status":"ready","event_year":"2024","original_text":"KLSCM\u2028finish","linguistic_text":"KLSCM\u2028finish","hashtags":[],"primary_language":"English","language_status":"ok"}])
    docs.to_parquet(output/"documents.parquet",index=False); settings=replace(SETTINGS,output_dir=output,cache_dir=cache)
    create_batch("initial",settings); root=output/"relevance_v8_single_researcher_cost_conservative"/"batch"/"initial"; chunk=next(root.glob("*.jsonl")).name
    (root/chunk.replace(".jsonl",".submission.json")).write_text(json.dumps({"batch_id":"b1","imported":False}),encoding="utf-8")
    body={"output":[{"content":[{"type":"output_text","text":json.dumps(assessment())}]}],"usage":{"input_tokens":10,"output_tokens":5}}
    class Files:
        text=""
        def content(self,_): return SimpleNamespace(text=self.text)
    fake=SimpleNamespace(batches=SimpleNamespace(retrieve=lambda _:SimpleNamespace(id="b1",status="completed",output_file_id="f1")),files=Files())
    fake.files.text=json.dumps({"custom_id":"unknown","response":{"status_code":200,"body":body},"error":None})
    with pytest.raises(ValueError,match="Unknown"): import_batch("initial",chunk,settings,fake)
    expected=pd.read_csv(root/"expected_requests_v1.csv").iloc[0].custom_id
    fake.files.text=json.dumps({"custom_id":expected,"response":{"status_code":200,"body":body},"error":None})
    result=import_batch("initial",chunk,settings,fake)
    assert result["imported"] is True
    assert len(list((cache/"relevance_v8_initial").glob("*.json")))==1
