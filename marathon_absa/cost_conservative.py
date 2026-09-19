from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import SETTINGS, Settings
from .openai_service import RELEVANCE_V8_INSTRUCTIONS, RELEVANCE_V8_VERIFICATION_INSTRUCTIONS
from .relevance import deterministic_result
from .relevance_v8 import material_model_disagreement, route_assessment
from .schemas import RELEVANCE_V8_ASSESSMENT_SCHEMA, V8_EXCLUDED_CONTENT_TYPES, V8_INCLUDED_CONTENT_TYPES
from .storage import read_table, write_table


CONFIG_VERSION = "v8_single_researcher_cost_conservative_v1"
OUTPUT_NAME = "relevance_v8_single_researcher_cost_conservative"
INITIAL_STAGE = "relevance_v8_initial"
VERIFICATION_STAGE = "relevance_v8_verification"
PROMPT_VERSION = "v8"
INCLUDE_THRESHOLD = .80
EXCLUDE_THRESHOLD = .99


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_key(stage: str, model: str, text: str) -> str:
    return hashlib.sha256("|".join([stage, PROMPT_VERSION, model, text, ""]).encode("utf-8")).hexdigest()


def record_context(row) -> str:
    hashtags = row.hashtags.tolist() if hasattr(row.hashtags, "tolist") else row.hashtags
    return json.dumps({"document_id": row.document_id, "event_year": row.event_year,
        "original_text": row.original_text, "linguistic_text": row.linguistic_text,
        "hashtags": hashtags, "primary_language": row.primary_language,
        "language_status": row.language_status}, ensure_ascii=False)


def verification_triggers(assessment: dict) -> list[str]:
    triggers = []
    if assessment["event_connection"] in {"weak", "none"}: triggers.append("weak_or_none_event_connection")
    if assessment["primary_content_type"] in V8_EXCLUDED_CONTENT_TYPES: triggers.append("excluded_content_family")
    if float(assessment["confidence"]) < .93: triggers.append("confidence_below_0_93")
    if assessment["contradiction_present"]: triggers.append("contradiction_present")
    if assessment["image_dependent"]: triggers.append("image_dependent")
    if assessment["requires_review_recommendation"]: triggers.append("model_recommends_review")
    return triggers


def initially_relevant_candidate(assessment: dict) -> bool:
    """Deterministic, gold-independent Terra eligibility semantic gate."""
    return bool(
        assessment["event_connection"] in {"explicit", "supported"}
        and assessment["meaningful_content_present"]
        and assessment["primary_content_type"] in V8_INCLUDED_CONTENT_TYPES
    )


def terra_required(assessment: dict) -> bool:
    return bool(verification_triggers(assessment)) and initially_relevant_candidate(assessment)


def direct_review_result(assessment: dict, triggers: list[str]) -> dict:
    routed = route_assessment(assessment, None, INCLUDE_THRESHOLD, EXCLUDE_THRESHOLD)
    routed.update({"v8_automatic_decision": "review", "v8_routing_status": "pending_review",
        "v8_routing_reason": "cost_conservative_direct_human_review",
        "v8_final_inclusion": "unresolved", "v8_final_decision_source": "pending_human_review",
        "v8_include_in_topics": False, "v8_include_in_sentiment": False,
        "verification_required": True, "terra_verification_eligible": False,
        "terra_verification_skipped": True,
        "terra_skip_reason": "initial_assessment_not_substantive_inclusion_candidate",
        "review_route_source": "verification_triggered_terra_ineligible",
        "verification_trigger_reasons": triggers, "operational_config_version": CONFIG_VERSION})
    return routed


def _validate_assessment(value: dict) -> None:
    required = set(RELEVANCE_V8_ASSESSMENT_SCHEMA["required"])
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Assessment does not exactly match the required v8 schema fields")
    properties = RELEVANCE_V8_ASSESSMENT_SCHEMA["properties"]
    for key, spec in properties.items():
        item = value[key]
        if "enum" in spec and item not in spec["enum"]: raise ValueError(f"Invalid {key}")
        if spec.get("type") == "boolean" and not isinstance(item, bool): raise ValueError(f"Invalid {key}")
        if spec.get("type") == "number" and (not isinstance(item, (int, float)) or not spec.get("minimum", 0) <= item <= spec.get("maximum", 1)): raise ValueError(f"Invalid {key}")
        if spec.get("type") == "array" and (not isinstance(item, list) or any(x not in spec["items"]["enum"] for x in item)): raise ValueError(f"Invalid {key}")
        if spec.get("type") == "string" and not isinstance(item, str): raise ValueError(f"Invalid {key}")


def _cache_path(settings: Settings, stage: str, model: str, text: str) -> Path:
    return settings.cache_dir / stage / f"{cache_key(stage, model, text)}.json"


def _load_cache(settings: Settings, stage: str, model: str, text: str) -> tuple[dict, dict] | None:
    path = _cache_path(settings, stage, model, text)
    if not path.exists(): return None
    stored = json.loads(path.read_text(encoding="utf-8")); _validate_assessment(stored["result"])
    return stored["result"], stored.get("metadata", {})


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.stem}_", suffix=".json", dir=path.parent); os.close(fd)
    temporary = Path(raw)
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()


def replay_calibration(settings: Settings = SETTINGS) -> dict:
    root = settings.output_dir / OUTPUT_NAME
    replay_dir = root / "calibration_replay_v1"
    if replay_dir.exists(): raise FileExistsError("Calibration replay exists; refusing overwrite")
    source = settings.output_dir / "relevance_v8_calibration_results" / "relevance_v8_calibration_adjudicated_v1.csv"
    calibration = pd.read_csv(source, dtype=str, keep_default_na=False)
    documents = read_table(settings.output_dir / "documents.parquet").set_index("document_id")
    old_rows=[]; new_rows=[]
    for row in calibration.itertuples(index=False):
        doc=documents.loc[row.document_id]; context=record_context(type("R", (), {**doc.to_dict(), "document_id": row.document_id})())
        initial=_load_cache(settings, INITIAL_STAGE, settings.chat_model, context)
        if initial is None: raise FileNotFoundError(f"Missing saved Luna calibration result: {row.document_id}")
        assessment=initial[0]; triggers=verification_triggers(assessment); verification=None
        if triggers:
            verification_text=json.dumps({"caption":json.loads(context),"first_assessment":assessment},ensure_ascii=False)
            cached=_load_cache(settings, VERIFICATION_STAGE, settings.stronger_model, verification_text)
            if cached is None: raise FileNotFoundError(f"Missing saved Terra calibration result: {row.document_id}")
            verification=cached[0]
        old=route_assessment(assessment, verification, INCLUDE_THRESHOLD, EXCLUDE_THRESHOLD)
        if triggers and not initially_relevant_candidate(assessment): new=direct_review_result(assessment,triggers)
        else:
            new=route_assessment(assessment, verification, INCLUDE_THRESHOLD, EXCLUDE_THRESHOLD)
            new.update({"verification_required":bool(triggers),"terra_verification_eligible":bool(triggers),"terra_verification_skipped":False,"terra_skip_reason":"","review_route_source":"existing_v8_routing","verification_trigger_reasons":triggers,"operational_config_version":CONFIG_VERSION})
        gold=row.final_v8_gold_label
        old_rows.append({"document_id":row.document_id,"gold":gold,"decision":old["v8_automatic_decision"],"terra_called":bool(triggers)})
        new_rows.append({"document_id":row.document_id,"gold":gold,"decision":new["v8_automatic_decision"],"terra_called":bool(triggers and initially_relevant_candidate(assessment)),"trigger_reasons":"|".join(triggers),"primary_language":row.primary_language,"event_year":row.event_year})
    old=pd.DataFrame(old_rows); new=pd.DataFrame(new_rows)
    def metrics(frame):
        auto=frame[frame.decision.isin({"include","exclude"})]
        return {"terra_calls":int(frame.terra_called.sum()),"manual_review":int(frame.decision.eq("review").sum()),"automatic_rows":len(auto),
            "false_exclusions":int(((auto.gold=="include")&(auto.decision=="exclude")).sum()),
            "false_inclusions":int(((auto.gold=="exclude")&(auto.decision=="include")).sum())}
    report={"configuration_version":CONFIG_VERSION,"created_at_utc":utc_now(),"api_calls":0,"source":str(source),"source_sha256":sha256(source),
        "old_v8_op1":metrics(old),"cost_conservative":metrics(new),"terra_calls_avoided":int(old.terra_called.sum()-new.terra_called.sum()),
        "validation_scope":"retrospective development calibration replay; not prospective confirmatory validation"}
    replay_dir.mkdir(parents=True); new.to_csv(replay_dir/"relevance_v8_cost_conservative_replay_rows_v1.csv",index=False,encoding="utf-8-sig")
    _atomic_json(replay_dir/"relevance_v8_cost_conservative_replay_metrics_v1.json",report)
    config={"configuration_version":CONFIG_VERSION,"parent_construct":"v8_event_experience_binary","parent_development_operating_point":"v8_op1",
        "not_v8_op1":True,"statement":"This configuration is NOT v8_op1 and must not be described as the frozen Phase 8 operating point.",
        "luna_model":settings.chat_model,"terra_model":settings.stronger_model,"initial_prompt_version":"v8","verification_prompt_version":"v8","schema_version":"v8",
        "inclusion_threshold":INCLUDE_THRESHOLD,"exclusion_threshold":EXCLUDE_THRESHOLD,"verification_trigger_logic":"existing v8 six-condition trigger",
        "terra_eligibility_rule":"triggered AND event_connection in {explicit,supported} AND meaningful_content_present AND primary_content_type in included family",
        "direct_human_rule":"triggered non-candidates bypass Terra and remain unresolved REVIEW; never automatic exclude",
        "created_at_utc":utc_now(),"calibration_replay":str(replay_dir),"calibration_replay_metrics_sha256":sha256(replay_dir/"relevance_v8_cost_conservative_replay_metrics_v1.json"),
        "api_execution_occurred":False,"production_execution_occurred":False}
    _atomic_json(root/f"{CONFIG_VERSION}.json",config)
    _atomic_json(replay_dir/"manifest_v1.json",{"configuration_version":CONFIG_VERSION,"created_at_utc":utc_now(),"api_calls":0,
        "source_sha256":sha256(source),"output_hashes":{"relevance_v8_cost_conservative_replay_metrics_v1.json":sha256(replay_dir/"relevance_v8_cost_conservative_replay_metrics_v1.json"),
        "relevance_v8_cost_conservative_replay_rows_v1.csv":sha256(replay_dir/"relevance_v8_cost_conservative_replay_rows_v1.csv"),f"../{CONFIG_VERSION}.json":sha256(root/f"{CONFIG_VERSION}.json")},
        "validation_scope":report["validation_scope"]})
    return report


def _production_candidates(settings: Settings):
    docs=read_table(settings.output_dir/"documents.parquet"); candidates=docs[(docs.source=="instagram")&(docs.processing_status=="ready")].copy()
    deterministic=candidates.apply(deterministic_result,axis=1); return candidates,deterministic


def _request(custom_id: str, model: str, instructions: str, text: str) -> dict:
    return {"custom_id":custom_id,"method":"POST","url":"/v1/responses","body":{"model":model,"instructions":instructions,"input":text,
        "text":{"format":{"type":"json_schema","name":custom_id.split("-")[0],"strict":True,"schema":RELEVANCE_V8_ASSESSMENT_SCHEMA}}}}


def create_batch(stage: str, settings: Settings=SETTINGS, max_estimated_tokens: int=1_250_000) -> dict:
    if stage not in {"initial","verification"}: raise ValueError("stage must be initial or verification")
    root=settings.output_dir/OUTPUT_NAME/"batch"/stage
    if root.exists(): raise FileExistsError(f"{stage} batch package exists; refusing overwrite")
    candidates,deterministic=_production_candidates(settings); requests=[]; reused=0
    for row in candidates[deterministic.isna()].itertuples(index=False):
        context=record_context(row); initial=_load_cache(settings,INITIAL_STAGE,settings.chat_model,context)
        if stage=="initial":
            if initial: reused+=1; continue
            requests.append(_request("ini-"+hashlib.sha256(row.document_id.encode()).hexdigest()[:24],settings.chat_model,RELEVANCE_V8_INSTRUCTIONS,context)|{"document_id":row.document_id,"cache_text":context})
        else:
            if initial is None: raise RuntimeError("All Luna initial results must exist before verification batch creation")
            assessment=initial[0]; triggers=verification_triggers(assessment)
            if not triggers or not initially_relevant_candidate(assessment): continue
            text=json.dumps({"caption":json.loads(context),"first_assessment":assessment},ensure_ascii=False)
            if _load_cache(settings,VERIFICATION_STAGE,settings.stronger_model,text): reused+=1; continue
            requests.append(_request("ver-"+hashlib.sha256(row.document_id.encode()).hexdigest()[:24],settings.stronger_model,RELEVANCE_V8_VERIFICATION_INSTRUCTIONS,text)|{"document_id":row.document_id,"cache_text":text})
    if len({r["custom_id"] for r in requests})!=len(requests): raise AssertionError("Duplicate custom IDs")
    estimate_per=2326 if stage=="initial" else 2423; chunk_size=max(1,max_estimated_tokens//estimate_per)
    root.mkdir(parents=True); expected=[]
    for index in range(0,len(requests),chunk_size):
        chunk=requests[index:index+chunk_size]; path=root/f"{stage}_batch_{index//chunk_size+1:03d}.jsonl"
        with path.open("w",encoding="utf-8",newline="\n") as handle:
            for item in chunk:
                exported={k:item[k] for k in ["custom_id","method","url","body"]}; handle.write(json.dumps(exported,ensure_ascii=False)+"\n")
                expected.append({"custom_id":item["custom_id"],"document_id":item["document_id"],"cache_text_sha256":hashlib.sha256(item["cache_text"].encode()).hexdigest(),"chunk":path.name})
    expected_path=root/"expected_requests_v1.csv"; pd.DataFrame(expected).to_csv(expected_path,index=False,encoding="utf-8-sig")
    files=sorted(root.glob("*.jsonl")); manifest={"configuration_version":CONFIG_VERSION,"stage":stage,"created_at_utc":utc_now(),"api_calls":0,"max_estimated_tokens_per_chunk":max_estimated_tokens,
        "records_considered":len(candidates),"deterministic_no_initial":int(deterministic.notna().sum()),"reusable_cache_hits":reused,"requests_generated":len(requests),"estimated_input_tokens":len(requests)*estimate_per,
        "chunk_count":len(files),"files":{p.name:sha256(p) for p in files},"expected_requests_sha256":sha256(expected_path),"account_batch_limit_note":"Configured planning limit; verify the API account tier queue limit before submission."}
    _atomic_json(root/"manifest_v1.json",manifest); return manifest


def _submission_path(root: Path, chunk: str) -> Path: return root/(Path(chunk).stem+".submission.json")


def submit_batch(stage: str, chunk: str, settings: Settings=SETTINGS, client=None) -> dict:
    root=settings.output_dir/OUTPUT_NAME/"batch"/stage; path=root/chunk
    if not path.exists(): raise FileNotFoundError(path)
    submission=_submission_path(root,chunk)
    if submission.exists(): raise FileExistsError("Chunk already submitted; refusing duplicate submission")
    if client is None:
        from openai import OpenAI
        client=OpenAI()
    with path.open("rb") as handle: uploaded=client.files.create(file=handle,purpose="batch")
    batch=client.batches.create(input_file_id=uploaded.id,endpoint="/v1/responses",completion_window="24h",metadata={"config":CONFIG_VERSION,"stage":stage})
    value={"configuration_version":CONFIG_VERSION,"stage":stage,"chunk":chunk,"input_file_id":uploaded.id,"batch_id":batch.id,"submitted_at_utc":utc_now(),"imported":False}
    _atomic_json(submission,value); return value


def batch_status(stage: str, chunk: str, settings: Settings=SETTINGS, client=None) -> dict:
    submission=json.loads(_submission_path(settings.output_dir/OUTPUT_NAME/"batch"/stage,chunk).read_text())
    if client is None:
        from openai import OpenAI
        client=OpenAI()
    batch=client.batches.retrieve(submission["batch_id"])
    return {"batch_id":batch.id,"status":batch.status,"output_file_id":getattr(batch,"output_file_id",None),"error_file_id":getattr(batch,"error_file_id",None)}


def _output_text(body: dict) -> str:
    for output in body.get("output",[]):
        for content in output.get("content",[]):
            if content.get("type")=="output_text": return content.get("text","")
    raise ValueError("Batch response has no output_text")


def _jsonl_lines(text: str):
    """Split JSONL only at its ASCII newline record delimiter."""
    return (line for line in text.split("\n") if line.strip())


def import_batch(stage: str, chunk: str, settings: Settings=SETTINGS, client=None, allow_partial: bool=False) -> dict:
    root=settings.output_dir/OUTPUT_NAME/"batch"/stage; submission_path=_submission_path(root,chunk); submission=json.loads(submission_path.read_text())
    if submission.get("imported"): raise FileExistsError("Batch chunk already imported")
    if client is None:
        from openai import OpenAI
        client=OpenAI()
    batch=client.batches.retrieve(submission["batch_id"])
    if batch.status!="completed": raise RuntimeError(f"Batch is not completed: {batch.status}")
    raw=client.files.content(batch.output_file_id); text=raw.text if hasattr(raw,"text") else raw.read().decode("utf-8")
    rows=[json.loads(line) for line in _jsonl_lines(text)]
    ids=[row["custom_id"] for row in rows]
    if len(ids)!=len(set(ids)): raise ValueError("Duplicate custom IDs in batch output")
    expected=pd.read_csv(root/"expected_requests_v1.csv",dtype=str); expected=expected[expected.chunk==chunk]
    unknown=set(ids)-set(expected.custom_id)
    if unknown: raise ValueError("Unknown custom IDs in batch output")
    if not allow_partial and set(ids)!=set(expected.custom_id): raise ValueError("Batch output is missing expected IDs")
    stage_name=INITIAL_STAGE if stage=="initial" else VERIFICATION_STAGE; model=settings.chat_model if stage=="initial" else settings.stronger_model
    imported=0
    for row in rows:
        if row.get("error") or row.get("response",{}).get("status_code")!=200:
            if allow_partial: continue
            raise ValueError("Batch output contains failed request")
        body=row["response"]["body"]; result=json.loads(_output_text(body)); _validate_assessment(result)
        request_line=None
        for line in _jsonl_lines((root/chunk).read_text(encoding="utf-8")):
            candidate=json.loads(line)
            if candidate["custom_id"]==row["custom_id"]: request_line=candidate; break
        cache_text=request_line["body"]["input"]; cache_path=_cache_path(settings,stage_name,model,cache_text)
        usage=body.get("usage",{}); stored={"result":result,"metadata":{"model":model,"prompt_version":"v8","input_tokens":usage.get("input_tokens"),"output_tokens":usage.get("output_tokens"),"cached":False,"batch":True,"batch_id":batch.id}}
        if cache_path.exists():
            if json.loads(cache_path.read_text(encoding="utf-8"))["result"]!=result: raise FileExistsError("Conflicting cache response; refusing overwrite")
        else: _atomic_json(cache_path,stored)
        imported+=1
    complete = set(ids) == set(expected.custom_id)
    submission.update({"imported":complete,"partial_import":not complete,"imported_at_utc":utc_now(),"output_file_id":batch.output_file_id,"imported_results":imported,"imported_custom_ids":ids}); _atomic_json(submission_path,submission)
    return submission


def finalize_production(settings: Settings=SETTINGS) -> dict:
    root=settings.output_dir/OUTPUT_NAME; output=root/"production_v1"
    if output.exists(): raise FileExistsError("Production finalization exists; refusing overwrite")
    candidates,deterministic=_production_candidates(settings); rows=[]
    for index,row in candidates.iterrows():
        det=deterministic.loc[index]
        if det is not None:
            excluded=det["reason_code"]=="hashtag_only"; rows.append({"document_id":row.document_id,"v8_automatic_decision":"exclude" if excluded else "review","v8_routing_status":"automatic_exclude_deterministic" if excluded else "pending_review","v8_routing_reason":det["reason_code"],"v8_final_inclusion":"exclude" if excluded else "unresolved","v8_include_in_topics":False,"operational_config_version":CONFIG_VERSION,"review_route_source":"deterministic"}); continue
        context=record_context(row); initial=_load_cache(settings,INITIAL_STAGE,settings.chat_model,context)
        if initial is None: raise RuntimeError(f"Missing Luna result: {row.document_id}")
        assessment,meta=initial; triggers=verification_triggers(assessment)
        if triggers and not initially_relevant_candidate(assessment): routed=direct_review_result(assessment,triggers)
        else:
            verification=None
            if triggers:
                text=json.dumps({"caption":json.loads(context),"first_assessment":assessment},ensure_ascii=False); cached=_load_cache(settings,VERIFICATION_STAGE,settings.stronger_model,text)
                if cached is None: raise RuntimeError(f"Missing required Terra result: {row.document_id}")
                verification=cached[0]
            routed=route_assessment(assessment,verification,INCLUDE_THRESHOLD,EXCLUDE_THRESHOLD); routed.update({"verification_required":bool(triggers),"terra_verification_eligible":bool(triggers),"terra_verification_skipped":False,"terra_skip_reason":"","review_route_source":"post_terra" if triggers else "initial_routing","verification_trigger_reasons":triggers,"operational_config_version":CONFIG_VERSION})
        rows.append({"document_id":row.document_id,**routed})
    decisions=pd.DataFrame(rows).merge(candidates[["document_id","original_text","source","event_year","primary_language","language_status"]],on="document_id",validate="one_to_one")
    output.mkdir(parents=True); write_table(decisions,output/"relevance_v8_cost_conservative_decisions_v1.parquet")
    review=decisions[decisions.v8_automatic_decision.eq("review")][["document_id","original_text","source","event_year","primary_language","review_route_source"]].copy()
    review["researcher_label"]=""; review["reason_code"]=""; review["evidence_span"]=""; review["notes"]=""
    review.to_csv(output/"relevance_v8_operational_review_queue_v1.csv",index=False,encoding="utf-8-sig")
    sources=review.review_route_source.value_counts().to_dict(); n=len(review)
    report={"configuration_version":CONFIG_VERSION,"finalized_at_utc":utc_now(),"api_calls":0,"eligible_records":len(decisions),
        "deterministic_auto_excludes":int(decisions.v8_routing_status.eq("automatic_exclude_deterministic").sum()),"deterministic_review":int((decisions.review_route_source=="deterministic").sum()),
        "automatic_includes":int(decisions.v8_automatic_decision.eq("include").sum()),"automatic_excludes":int(decisions.v8_automatic_decision.eq("exclude").sum()),
        "direct_human_skipped_terra":int((decisions.review_route_source=="verification_triggered_terra_ineligible").sum()),"post_terra_review":int((decisions.review_route_source=="post_terra").sum()),
        "total_review_queue":n,"review_sources":sources,"workload_minutes":{"5_seconds":n*5/60,"10_seconds":n*10/60,"20_seconds":n*20/60}}
    _atomic_json(output/"production_manifest_v1.json",report); return report


def build_production_report(decisions: pd.DataFrame) -> dict:
    """Describe finalized routing without changing any record-level decision."""
    eligible = len(decisions)
    automatic_includes = int(decisions.v8_automatic_decision.eq("include").sum())
    automatic_excludes = int(decisions.v8_automatic_decision.eq("exclude").sum())
    human_review = int(decisions.v8_automatic_decision.eq("review").sum())
    deterministic_total = int(decisions.review_route_source.eq("deterministic").sum())
    deterministic_excludes = int(decisions.v8_routing_status.eq("automatic_exclude_deterministic").sum())
    deterministic_review = int((decisions.review_route_source.eq("deterministic") & decisions.v8_automatic_decision.eq("review")).sum())
    terra_processed = int(decisions.review_route_source.eq("post_terra").sum())
    post_terra_review = int((decisions.review_route_source.eq("post_terra") & decisions.v8_automatic_decision.eq("review")).sum())
    skipped_terra_review = int((decisions.review_route_source.eq("verification_triggered_terra_ineligible") & decisions.v8_automatic_decision.eq("review")).sum())
    initial_only = int(decisions.review_route_source.eq("initial_routing").sum())
    review_sources = decisions.loc[decisions.v8_automatic_decision.eq("review"), "review_route_source"].value_counts().to_dict()
    if automatic_includes + automatic_excludes + human_review != eligible:
        raise RuntimeError("Final routing counts do not reconcile to eligible records")
    if deterministic_excludes + deterministic_review != deterministic_total:
        raise RuntimeError("Deterministic routing counts do not reconcile")
    if post_terra_review != int(review_sources.get("post_terra", 0)):
        raise RuntimeError("Post-Terra review count does not reconcile")
    return {
        "report_version": "production_reporting_v2", "configuration_version": CONFIG_VERSION,
        "generated_at_utc": utc_now(), "api_calls": 0, "eligible_records": eligible,
        "routing_outcomes": {"automatic_includes": automatic_includes, "automatic_excludes": automatic_excludes, "human_review": human_review},
        "routing_reconciliation": {"automatic_includes_plus_automatic_excludes_plus_human_review": automatic_includes + automatic_excludes + human_review, "matches_eligible_records": True},
        "processing_paths": {
            "deterministic_no_initial_total": deterministic_total,
            "deterministic_automatic_excludes": deterministic_excludes,
            "deterministic_human_review": deterministic_review,
            "luna_initial_processed_total": eligible - deterministic_total,
            "luna_initial_only_routed_total": initial_only,
            "terra_processed_eligible_total": terra_processed,
            "terra_processed_human_review": post_terra_review,
            "terra_ineligible_triggered_human_review": skipped_terra_review,
        },
        "human_review_sources": review_sources,
        "human_workload": {"records": human_review, "minutes_at_5_seconds_each": human_review*5/60, "minutes_at_10_seconds_each": human_review*10/60, "minutes_at_20_seconds_each": human_review*20/60},
        "semantic_notes": {
            "deterministic_no_initial_total": "All records resolved without Luna: deterministic automatic exclusions plus deterministic human-review routes.",
            "deterministic_human_review": "Only deterministic/no-initial records that require researcher annotation.",
            "terra_processed_eligible_total": "All records eligible for and processed by Terra, regardless of final automatic or review outcome.",
            "terra_processed_human_review": "Only Terra-processed records whose final route remains human review.",
        },
    }


def write_production_report(settings: Settings=SETTINGS) -> dict:
    root=settings.output_dir/OUTPUT_NAME/"production_v1"
    decisions=read_table(root/"relevance_v8_cost_conservative_decisions_v1.parquet")
    path=root/"production_report_v2.json"
    if path.exists(): raise FileExistsError("Production reporting v2 already exists; refusing overwrite")
    report=build_production_report(decisions); _atomic_json(path,report); return report


def finalize_topic_gate(settings: Settings=SETTINGS) -> dict:
    root=settings.output_dir/OUTPUT_NAME; target=root/"final_corpus_v1"
    if target.exists(): raise FileExistsError("Final relevance corpus exists; refusing overwrite")
    decisions=read_table(root/"production_v1"/"relevance_v8_cost_conservative_decisions_v1.parquet")
    audit_root=settings.output_dir/"relevance_v8_single_researcher_architecture_audit_v2"
    review_path=audit_root/"relevance_v8_operational_review_queue_v2.csv"
    metrics_path=audit_root/"final_v2"/"relevance_v8_architecture_audit_metrics_v2.json"
    if not review_path.exists() or not metrics_path.exists(): raise FileNotFoundError("Completed operational review and audit metrics are required")
    review=pd.read_csv(review_path,dtype=str,keep_default_na=False); labels=review.researcher_label.str.lower()
    pending=decisions[decisions.v8_automatic_decision.eq("review")]
    if set(review.document_id)!=set(pending.document_id) or not labels.isin({"include","exclude"}).all(): raise RuntimeError("All and only operational review records must be resolved")
    metrics=json.loads(metrics_path.read_text(encoding="utf-8"))
    if not metrics.get("acceptance",{}).get("audit_acceptable"): raise RuntimeError("Random audit acceptance did not pass")
    mapping=dict(zip(review.document_id,labels)); final=decisions.copy(); mask=final.document_id.isin(mapping)
    final.loc[mask,"v8_final_inclusion"]=final.loc[mask,"document_id"].map(mapping)
    final.loc[mask,"v8_include_in_topics"]=final.loc[mask,"v8_final_inclusion"].eq("include")
    final.loc[mask,"v8_final_decision_source"]="single_researcher_operational_review"
    if final.v8_final_inclusion.eq("unresolved").any(): raise RuntimeError("Unresolved relevance records remain")
    target.mkdir(parents=True); write_table(final,target/"relevance_v8_final_corpus_v1.parquet")
    report={"configuration_version":CONFIG_VERSION,"frozen_at_utc":utc_now(),"records":len(final),"included":int(final.v8_include_in_topics.astype(bool).sum()),
        "excluded":int((~final.v8_include_in_topics.astype(bool)).sum()),"unresolved":0,"audit_metrics_sha256":sha256(metrics_path),"api_calls":0,
        "output_hashes":{name:sha256(target/name) for name in ["relevance_v8_final_corpus_v1.parquet","relevance_v8_final_corpus_v1.csv"]}}
    _atomic_json(target/"final_corpus_manifest_v1.json",report); return report
