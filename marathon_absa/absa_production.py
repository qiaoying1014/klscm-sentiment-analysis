from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .absa_v1 import (
    CORPUS, DEFAULT_MODEL, ONTOLOGY_PATH, SCHEMA_VERSION, SENTIMENTS,
    canonicalize_evidence, load_ontology, mention_schema, stable_mention_id,
)
from .absa_aspect_ownership_development import (
    INSTRUCTIONS, PROMPT_VERSION, ROOT as V3_ROOT,
)

PROTOCOL = "absa_v1_production_v1"
PRODUCTION_CACHE_STAGE = "absa_v1_mentions_production_v1"
EXPECTED_DOCUMENTS = 7704
EXPECTED_CONTEXTUAL_EXCLUDED = 336
EXPECTED_OUTLIERS = 5703
EXPECTED_ELIGIBLE = 13743
EXPECTED_PROMPT_SHA256 = "196606110ecb2b78d56a3220c1bd8957e8e42f2f259c65d1094c037fa21f30d0"
TAXONOMY_ROOT = Path("data/processed/topic_discovery_v1/final_taxonomy_v1")
TAXONOMY_MANIFEST = TAXONOMY_ROOT / "final_topic_taxonomy_manifest_v1.json"
CONTEXTUAL_CORPUS = TAXONOMY_ROOT / "final_contextual_or_excluded_topic_corpus_v1.csv"
OUTLIER_CORPUS = TAXONOMY_ROOT / "final_outlier_corpus_v1.csv"
ROOT = Path("data/processed/absa_v1/production/absa_v1_production_v1")
FREEZE_PATH = ROOT / "absa_v1_production_freeze_manifest_v1.json"
POPULATION_PATH = ROOT / "absa_v1_production_population_manifest_v1.csv"
BATCH_ROOT = ROOT / "batch"
REQUEST_PATH = BATCH_ROOT / "absa_v1_production_requests_v1.jsonl"
REQUEST_MANIFEST_PATH = BATCH_ROOT / "absa_v1_production_request_manifest_v1.json"
SUBMISSION_PATH = BATCH_ROOT / "submission_v1.json"
STATUS_PATH = BATCH_ROOT / "status_v1.json"
RAW_OUTPUT_PATH = BATCH_ROOT / "absa_v1_production_raw_output_v1.jsonl"
RAW_ERROR_PATH = BATCH_ROOT / "absa_v1_production_raw_errors_v1.jsonl"
IMPORT_MANIFEST_PATH = BATCH_ROOT / "absa_v1_production_import_manifest_v1.json"
PARSED_RESPONSES_PATH = ROOT / "absa_v1_production_parsed_responses_v1.jsonl"
RESPONSE_INTEGRITY_PATH = ROOT / "absa_v1_production_response_integrity_v1.json"
FINALIZATION_MANIFEST_PATH = ROOT / "absa_v1_production_finalization_manifest_v1.json"
RETRY_ROOT = ROOT / "retry"
DOCUMENT_RESULTS_PATH = ROOT / "absa_v1_production_document_results_v1.csv"
MENTIONS_PATH = ROOT / "absa_v1_production_mentions_v1.csv"
QUARANTINE_PATH = ROOT / "absa_v1_production_failures_v1.csv"
SUMMARY_PATH = ROOT / "absa_v1_production_summary_v1.json"
REPORT_PATH = ROOT / "absa_v1_production_report_v1.md"
MODEL_SELECTION_JSON = ROOT / "absa_v1_production_model_selection_v1.json"
MODEL_SELECTION_MD = ROOT / "absa_v1_production_model_selection_v1.md"
PRE_SUBMISSION_AUDIT_JSON = ROOT / "absa_v1_production_pre_submission_audit_v1.json"
PRE_SUBMISSION_AUDIT_MD = ROOT / "absa_v1_production_pre_submission_audit_v1.md"
LOCAL_EVALUATION_PATH = Path("data/processed/absa_local_v1/zero_shot_nli_v1/local_absa_evaluation_v1.json")
V3_PROMPT_PATH = V3_ROOT / "absa_v1_instructions_3_aspect_ownership.txt"
V3_PREDICTIONS_PATH = V3_ROOT / "candidate_predictions_v3_v1.csv"
V3_EVALUATION_PATH = V3_ROOT / "candidate_evaluation_v3_v1.json"

LIMITATION = (
    "The selected V3 ABSA system achieved development-set aspect precision of 0.513 and recall of 0.790 "
    "on the reused 80-document development sample. It did not satisfy the predeclared aspect-precision "
    "target of 0.55. The researcher elected to stop further manual review and prompt optimization for the "
    "current phase and proceed with V3 as the best available production candidate. Full-corpus ABSA outputs "
    "should therefore be interpreted with this known precision limitation, and further refinement may be "
    "performed in a future version."
)

FINAL_LIMITATION = (
    "The selected OpenAI V3 ABSA system achieved development-set aspect precision of 0.513 and recall "
    "of 0.790 on the reused 80-document development sample. It did not satisfy the predeclared "
    "aspect-precision target of 0.55. A zero-shot local multilingual NLI alternative was evaluated on "
    "the same development benchmark and performed substantially worse. The researcher therefore selected "
    "V3 as the best available production system while closing further manual review and model tuning for "
    "the current phase. Full-corpus outputs should be interpreted with this known precision limitation."
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha(value: Any) -> str:
    payload=json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _verify_taxonomy() -> dict[str,Any]:
    for path in [CORPUS,CONTEXTUAL_CORPUS,OUTLIER_CORPUS,TAXONOMY_MANIFEST]:
        if not path.exists(): raise FileNotFoundError(path)
    manifest=json.loads(TAXONOMY_MANIFEST.read_text(encoding="utf-8"))
    expected={"substantive_documents":EXPECTED_DOCUMENTS,
              "contextual_or_excluded_documents":EXPECTED_CONTEXTUAL_EXCLUDED,
              "outliers":EXPECTED_OUTLIERS,"eligible_documents_reconciled":EXPECTED_ELIGIBLE}
    if any(manifest.get(k)!=v for k,v in expected.items()) or manifest.get("ABSA_READY") is not True:
        raise ValueError("Frozen taxonomy manifest does not reconcile the ABSA-ready population")
    if EXPECTED_DOCUMENTS+EXPECTED_CONTEXTUAL_EXCLUDED+EXPECTED_OUTLIERS!=EXPECTED_ELIGIBLE:
        raise ValueError("Frozen taxonomy arithmetic failed")
    artifact_hashes=manifest.get("artifacts",{})
    checks={CORPUS:"final_substantive_topic_corpus_v1.csv",CONTEXTUAL_CORPUS:"final_contextual_or_excluded_topic_corpus_v1.csv",OUTLIER_CORPUS:"final_outlier_corpus_v1.csv"}
    for path,name in checks.items():
        if artifact_hashes.get(name)!=_sha(path): raise ValueError(f"Frozen taxonomy artifact hash changed: {name}")
    return manifest


def _verify_prompt() -> str:
    if not V3_PROMPT_PATH.exists(): raise FileNotFoundError(V3_PROMPT_PATH)
    prompt_text=V3_PROMPT_PATH.read_text(encoding="utf-8")
    prompt_hash=hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    if prompt_hash!=EXPECTED_PROMPT_SHA256: raise ValueError("Selected V3 prompt hash differs from frozen identity")
    if prompt_text!=INSTRUCTIONS: raise ValueError("Selected V3 prompt content differs from implementation")
    return prompt_hash


def _population() -> pd.DataFrame:
    _verify_taxonomy(); source=pd.read_csv(CORPUS,low_memory=False)
    if len(source)!=EXPECTED_DOCUMENTS or source.document_id.astype(str).nunique()!=EXPECTED_DOCUMENTS:
        raise ValueError("Production population must contain exactly 7,704 unique substantive documents")
    contextual_ids=set(pd.read_csv(CONTEXTUAL_CORPUS,usecols=["document_id"]).document_id.astype(str))
    outlier_ids=set(pd.read_csv(OUTLIER_CORPUS,usecols=["document_id"]).document_id.astype(str))
    ids=set(source.document_id.astype(str))
    if ids&contextual_ids or ids&outlier_ids: raise ValueError("Excluded/contextual/outlier document entered production")
    if len(contextual_ids)!=EXPECTED_CONTEXTUAL_EXCLUDED or len(outlier_ids)!=EXPECTED_OUTLIERS:
        raise ValueError("Excluded/outlier population counts changed")
    source=source.copy();source["document_id"]=source.document_id.astype(str)
    return source


def _population_manifest(source:pd.DataFrame)->pd.DataFrame:
    columns={"document_id":"document_id","source":"source","event_year":"event_year","primary_language":"primary_language",
             "original_text":"original_caption","original_topic_id":"raw_c1_topic_id",
             "original_human_topic_name":"raw_c1_topic_label","final_topic_id":"final_consolidated_topic_id",
             "final_topic_name":"final_consolidated_topic_label"}
    missing=set(columns)-set(source.columns)
    if missing: raise ValueError(f"Production metadata columns missing: {sorted(missing)}")
    result=source[list(columns)].rename(columns=columns).copy()
    result["taxonomy_status"]="substantive_absa_ready"
    result["production_protocol"]=PROTOCOL
    return result


def _request_lines(source:pd.DataFrame)->tuple[list[str],dict[str,Any]]:
    ontology=load_ontology();aspects=[x["id"] for x in ontology["aspects"]];schema=mention_schema(aspects);lines=[]
    estimated_input=0;estimated_output=len(source)*260
    for row in source.itertuples(index=False):
        context={"document_id":str(row.document_id),"original_text":str(row.original_text),
                 "final_topic_name":str(row.final_topic_name),"final_topic_group":str(row.final_topic_group),
                 "topic_context_supplied":True,"ontology":aspects}
        body={"model":DEFAULT_MODEL,"instructions":INSTRUCTIONS,"input":json.dumps(context,ensure_ascii=False),
              "text":{"format":{"type":"json_schema","name":SCHEMA_VERSION,"strict":True,"schema":schema}}}
        estimated_input+=math.ceil(len(json.dumps(body,ensure_ascii=False))/4)
        # ASCII escaping keeps Unicode line/paragraph separators inside one physical JSONL record.
        lines.append(json.dumps({"custom_id":f"absa1prod-{row.document_id}","method":"POST","url":"/v1/responses","body":body},ensure_ascii=True))
    synchronous=(estimated_input*1.25+estimated_output*10)/1_000_000
    conservative_input=math.ceil(estimated_input*1.25);conservative_output=math.ceil(estimated_output*1.5)
    conservative=(conservative_input*1.25+conservative_output*10)/1_000_000
    estimate={"estimated_input_tokens":estimated_input,"estimated_output_tokens":estimated_output,
      "estimated_synchronous_cost_usd":round(synchronous,4),"estimated_batch_cost_usd":round(synchronous*.5,4),
      "conservative_input_tokens":conservative_input,"conservative_output_tokens":conservative_output,
      "conservative_synchronous_cost_usd":round(conservative,4),"conservative_batch_cost_usd":round(conservative*.5,4),
      "pricing_assumptions_usd_per_million":{"input":1.25,"output":10.0,"batch_discount":.5}}
    return lines,estimate


def create_production(root:Path=ROOT)->dict[str,Any]:
    required=[root/FREEZE_PATH.name,root/POPULATION_PATH.name,root/"batch"/REQUEST_PATH.name,root/"batch"/REQUEST_MANIFEST_PATH.name]
    if all(path.exists() for path in required):
        return _close_existing_selection(root)
    if any(path.exists() for path in required):
        raise FileExistsError("Partial production preparation exists; refusing overwrite or regeneration")
    prompt_hash=_verify_prompt();taxonomy=_verify_taxonomy();source=_population();population=_population_manifest(source)
    schema=mention_schema([x["id"] for x in load_ontology()["aspects"]]);ontology_hash=_sha(ONTOLOGY_PATH)
    v3_eval=json.loads(V3_EVALUATION_PATH.read_text(encoding="utf-8"));metrics=v3_eval["metrics"]["aspect_detection"]
    if tuple(metrics[k] for k in ["tp","fp","fn"])!=(79,75,21): raise ValueError("Stored V3 development metrics changed")
    freeze={"protocol":PROTOCOL,"created_at_utc":datetime.now(timezone.utc).isoformat(),
      "selected_absa_system":PROMPT_VERSION,"selected_prompt_identity":PROMPT_VERSION,"selected_prompt_sha256":prompt_hash,
      "model":DEFAULT_MODEL,"schema":SCHEMA_VERSION,"schema_sha256":_json_sha(schema),
      "ontology_identity":"absa_aspect_ontology_v1","ontology_sha256":ontology_hash,
      "input_taxonomy_identity":taxonomy["version"],"input_taxonomy_manifest_sha256":_sha(TAXONOMY_MANIFEST),
      "input_substantive_corpus_sha256":_sha(CORPUS),"expected_document_count":EXPECTED_DOCUMENTS,
      "population_reconciliation":{"substantive":7704,"contextual_or_excluded":336,"unassigned_outlier":5703,"eligible_total":13743},
      "development_history":{"v1":{"prompt":"absa_v1_instructions_1","precision":0.3853211,"recall":.84,"f1":.5283019,"tp":84,"fp":134,"fn":16,"joint_precision":.3302752,"joint_recall":.72,"joint_f1":.4528302},
        "v2":{"prompt":"absa_v1_instructions_2_precision","precision":.5099,"recall":.77,"f1":.6135,"tp":77,"fp":74,"fn":23,"joint_precision":.4636,"joint_recall":.70,"joint_f1":.5578,"decision":"INSUFFICIENT_PRECISION_IMPROVEMENT"},
        "v3":{"prompt":PROMPT_VERSION,"precision":metrics["precision"],"recall":metrics["recall"],"f1":metrics["f1"],"tp":79,"fp":75,"fn":21,"decision":"INSUFFICIENT_PRECISION_IMPROVEMENT","emotional_experience_fp":7,"predicted_multi_aspect_documents":34}},
      "development_precision":metrics["precision"],"development_recall":metrics["recall"],"development_f1":metrics["f1"],
      "development_decision":"INSUFFICIENT_PRECISION_IMPROVEMENT",
      "selection_status":"selected_for_production_with_known_development_limitation",
      "selection_rationale":"Researcher elected to stop additional manual review and prompt tuning. V3 is the strongest current development candidate and materially improves the original ABSA system, but did not meet the predeclared 0.55 aspect precision development target. Further refinement is deferred and may be performed as a future version if required.",
      "methodological_limitation":LIMITATION,"manual_review_closed_for_current_phase":True,
      "additional_prompt_tuning_deferred":True,"V4_created":False,"development_evidence":True,
      "confirmatory_evidence":False,"production_cache_stage":PRODUCTION_CACHE_STAGE,
      "full_corpus_inference_started":False,"submitted":False,"future_refinement_requires_new_version":True}
    root.mkdir(parents=True,exist_ok=True);batch=root/"batch";batch.mkdir(parents=True,exist_ok=True)
    freeze_path=root/FREEZE_PATH.name;population_path=root/POPULATION_PATH.name;request_path=batch/REQUEST_PATH.name
    request_manifest_path=batch/REQUEST_MANIFEST_PATH.name
    if any(x.exists() for x in [freeze_path,population_path,request_path,request_manifest_path]):
        raise FileExistsError("Production preparation artifact already exists; refusing overwrite")
    lines,estimate=_request_lines(source);content="\n".join(lines)+"\n";pre_write_lf_size=len(content.encode("utf-8"))
    if len(lines)>50000 or pre_write_lf_size>200_000_000: raise ValueError("Production Batch exceeds official request/file limits")
    # Freeze selection before materializing model requests.
    freeze_path.write_text(json.dumps(freeze,indent=2,ensure_ascii=False),encoding="utf-8")
    population.to_csv(population_path,index=False,encoding="utf-8-sig")
    request_path.write_text(content,encoding="utf-8")
    size=request_path.stat().st_size
    if size>200_000_000:raise ValueError("On-disk production Batch exceeds official file limit")
    manifest={"protocol":PROTOCOL,"created_at_utc":datetime.now(timezone.utc).isoformat(),
      "selected_prompt_identity":PROMPT_VERSION,"prompt_sha256":prompt_hash,"model":DEFAULT_MODEL,
      "schema":SCHEMA_VERSION,"schema_sha256":_json_sha(schema),"ontology_version":"absa_aspect_ontology_v1",
      "ontology_sha256":ontology_hash,"production_cache_stage":PRODUCTION_CACHE_STAGE,"document_count":len(source),
      "request_count":len(lines),"batch_chunk_count":1,"request_jsonl_path":str(request_path),
      "request_jsonl_sha256":_sha(request_path),"request_jsonl_bytes":size,"request_jsonl_pre_write_lf_bytes":pre_write_lf_size,
      "request_jsonl_line_endings":"CRLF" if size>pre_write_lf_size else "LF","custom_id_prefix":"absa1prod-",
      "population_manifest_path":str(population_path),"population_manifest_sha256":_sha(population_path),
      "token_and_cost_estimate":estimate,"submitted":False,"batch_ids":None,"api_calls":0}
    request_manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return {"freeze_manifest":str(freeze_path),"request_manifest":str(request_manifest_path),**manifest}


def _validate_prepared(root:Path=ROOT)->tuple[dict[str,Any],dict[str,Any],Path]:
    prompt_hash=_verify_prompt();_verify_taxonomy();source=_population()
    freeze=json.loads((root/FREEZE_PATH.name).read_text(encoding="utf-8"));manifest=json.loads((root/"batch"/REQUEST_MANIFEST_PATH.name).read_text(encoding="utf-8"));path=root/"batch"/REQUEST_PATH.name
    if freeze["selected_prompt_sha256"]!=prompt_hash or manifest["prompt_sha256"]!=prompt_hash: raise RuntimeError("Production prompt hash changed")
    if freeze["input_substantive_corpus_sha256"]!=_sha(CORPUS): raise RuntimeError("Production population source changed")
    if manifest["request_jsonl_sha256"]!=_sha(path): raise RuntimeError("Production request JSONL hash changed")
    ids=[]
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.append(json.loads(line)["custom_id"].removeprefix("absa1prod-"))
    if len(ids)!=EXPECTED_DOCUMENTS or len(set(ids))!=EXPECTED_DOCUMENTS or ids!=source.document_id.tolist(): raise RuntimeError("Production request IDs/order changed")
    if path.stat().st_size>200_000_000: raise RuntimeError("Production request file exceeds Batch limit")
    return freeze,manifest,path


def _close_existing_selection(root:Path=ROOT)->dict[str,Any]:
    freeze,manifest,request_path=_validate_prepared(root)
    if (root/"batch"/SUBMISSION_PATH.name).exists() or manifest.get("submitted") is not False:
        raise RuntimeError("Production selection closure requires an unsubmitted package")
    if not LOCAL_EVALUATION_PATH.exists(): raise FileNotFoundError(LOCAL_EVALUATION_PATH)
    local_hash_before=_sha(LOCAL_EVALUATION_PATH)
    local=json.loads(LOCAL_EVALUATION_PATH.read_text(encoding="utf-8"))
    expected_local={"tp":23,"fp":85,"fn":77,"precision":0.21296296296296297,"recall":0.23,"f1":0.22115384615384615}
    if any(local["aspect_detection"].get(key)!=value for key,value in expected_local.items()):
        raise RuntimeError("Frozen local benchmark metrics changed")
    if local.get("viability")!="LOCAL_NOT_CURRENTLY_VIABLE" or local.get("paid_api_calls")!=0:
        raise RuntimeError("Frozen local benchmark status changed")
    selection={
      "protocol":"absa_v1_production_model_selection_v1","created_at_utc":freeze["created_at_utc"],
      "selected":"OpenAI V3","selected_production_absa":PROMPT_VERSION,
      "selection_status":"selected_for_production_with_known_development_limitation",
      "development_decision":"INSUFFICIENT_PRECISION_IMPROVEMENT",
      "local_experiment_status":"completed","local_viability":"LOCAL_NOT_CURRENTLY_VIABLE",
      "local_selected_for_production":False,"local_baseline_retained_as_research_benchmark":True,
      "comparison":{
        "openai_v3":{"aspect_precision":0.512987012987013,"aspect_recall":0.79,"aspect_f1":0.6220472440944881,
          "tp":79,"fp":75,"fn":21,"joint_precision":0.43506493506493504,"joint_recall":0.67,"joint_f1":0.5275590551181102},
        "local_nli":{"aspect_precision":local["aspect_detection"]["precision"],"aspect_recall":local["aspect_detection"]["recall"],
          "aspect_f1":local["aspect_detection"]["f1"],"tp":23,"fp":85,"fn":77,
          "joint_precision":local["joint_aspect_sentiment"]["precision"],"joint_recall":local["joint_aspect_sentiment"]["recall"],
          "joint_f1":local["joint_aspect_sentiment"]["f1"]}},
      "selection_rationale":"V3 was the strongest current system after controlled prompt development. The fully local zero-shot multilingual NLI alternative was substantially inferior. Further manual review, prompt tuning, and local model development are closed for the current phase.",
      "local_evaluation_path":str(LOCAL_EVALUATION_PATH),"local_evaluation_sha256":local_hash_before,
      "request_jsonl_path":str(request_path),"request_jsonl_sha256":manifest["request_jsonl_sha256"],
      "development_evidence":True,"confirmatory_evidence":False,"submitted":False,"api_calls":0}
    selection_json=root/MODEL_SELECTION_JSON.name;selection_md=root/MODEL_SELECTION_MD.name
    rendered=json.dumps(selection,indent=2,ensure_ascii=False)
    markdown=("# ABSA V1 production model selection\n\n"
      "Both systems were evaluated on the historically reused 80-document development benchmark; neither result is confirmatory evidence.\n\n"
      "| Metric | OpenAI V3 | Local NLI |\n|---|---:|---:|\n"
      "| Aspect precision | 0.5130 | 0.2130 |\n| Aspect recall | 0.7900 | 0.2300 |\n| Aspect F1 | 0.6220 | 0.2212 |\n"
      "| TP | 79 | 23 |\n| FP | 75 | 85 |\n| FN | 21 | 77 |\n"
      "| Joint precision | 0.4351 | 0.1111 |\n| Joint recall | 0.6700 | 0.1200 |\n| Joint F1 | 0.5276 | 0.1154 |\n\n"
      "Selected: **OpenAI V3**, with known development limitation. The local baseline is retained as a completed research benchmark and is not selected for production. No further prompt, local-model, or manual-review experiment was performed.\n")
    if selection_json.exists() and selection_json.read_text(encoding="utf-8")!=rendered:
        raise FileExistsError("Production model-selection JSON already exists with different content")
    if selection_md.exists() and selection_md.read_text(encoding="utf-8")!=markdown:
        raise FileExistsError("Production model-selection Markdown already exists with different content")
    selection_json.write_text(rendered,encoding="utf-8");selection_md.write_text(markdown,encoding="utf-8")
    freeze.update({"local_experiment_status":"completed","local_baseline_viability":"LOCAL_NOT_CURRENTLY_VIABLE",
      "local_selected_for_production":False,"local_baseline_retained_as_research_benchmark":True,
      "additional_local_model_development_deferred":True,"full_corpus_submitted":False,
      "selection_rationale":selection["selection_rationale"],"methodological_limitation":FINAL_LIMITATION,
      "production_model_selection_path":str(selection_json),"production_model_selection_sha256":_sha(selection_json)})
    (root/FREEZE_PATH.name).write_text(json.dumps(freeze,indent=2,ensure_ascii=False),encoding="utf-8")
    if _sha(LOCAL_EVALUATION_PATH)!=local_hash_before: raise RuntimeError("Local benchmark artifact changed during closure")
    return {"freeze_manifest":str(root/FREEZE_PATH.name),"model_selection_json":str(selection_json),
      "model_selection_markdown":str(selection_md),**manifest,"local_experiment_status":"completed",
      "local_viability":"LOCAL_NOT_CURRENTLY_VIABLE","submitted":False,"api_calls":0}


def audit_pre_submission(root:Path=ROOT)->dict[str,Any]:
    freeze,manifest,request_path=_validate_prepared(root);source=_population();population=_population_manifest(source)
    source_by_id=source.set_index("document_id");expected_schema=mention_schema([x["id"] for x in load_ontology()["aspects"]])
    ids=[];malformed=[];empty_captions=[];configuration_mismatches=[];caption_mismatches=[];line_bytes=[];caption_lengths=[]
    explicit_output_caps=[]
    with request_path.open("r",encoding="utf-8",errors="strict") as handle:
        for line_number,line in enumerate(handle,1):
            line_bytes.append(len(line.encode("utf-8")))
            try:item=json.loads(line)
            except Exception as exc:malformed.append({"line":line_number,"error":str(exc)});continue
            custom=str(item.get("custom_id",""));doc=custom.removeprefix("absa1prod-");ids.append(doc);body=item.get("body",{})
            try:input_payload=json.loads(body.get("input",""))
            except Exception as exc:configuration_mismatches.append({"line":line_number,"issue":f"invalid body.input JSON: {exc}"});continue
            caption=str(input_payload.get("original_text",""));caption_lengths.append(len(caption))
            if not caption:empty_captions.append(doc)
            if doc not in source_by_id.index or input_payload.get("document_id")!=doc:caption_mismatches.append({"document_id":doc,"issue":"document identity mismatch"})
            elif caption!=str(source_by_id.loc[doc,"original_text"]):caption_mismatches.append({"document_id":doc,"issue":"caption differs from frozen corpus"})
            expected={"method":"POST","url":"/v1/responses"}
            if item.get("method")!=expected["method"] or item.get("url")!=expected["url"] or body.get("model")!=DEFAULT_MODEL or body.get("instructions")!=INSTRUCTIONS or body.get("text",{}).get("format",{}).get("schema")!=expected_schema:
                configuration_mismatches.append({"document_id":doc,"issue":"request method/URL/model/prompt/schema mismatch"})
            explicit_output_caps.append(body.get("max_output_tokens"))
    duplicates=sorted(key for key,count in Counter(ids).items() if count>1);missing=sorted(set(source.document_id)-set(ids));unexpected=sorted(set(ids)-set(source.document_id))
    schema_fields=list(expected_schema["properties"]["mentions"]["items"]["properties"])
    schema_required=expected_schema["properties"]["mentions"]["items"]["required"]
    request_hash=_sha(request_path);population_hash=_sha(root/POPULATION_PATH.name)
    local_hash=_sha(LOCAL_EVALUATION_PATH)
    checks={
      "request_identity_and_population":not malformed and len(ids)==EXPECTED_DOCUMENTS and not duplicates and not missing and not unexpected,
      "captions_complete_and_exact":not empty_captions and not caption_mismatches,
      "configuration_uniform_and_frozen":not configuration_mismatches,
      "request_manifest_size_and_hash_reconcile":manifest["request_jsonl_sha256"]==request_hash and manifest["request_jsonl_bytes"]==request_path.stat().st_size,
      "package_within_batch_limits":len(ids)<=50000 and request_path.stat().st_size<=200_000_000,
      "metadata_join_complete":len(population)==EXPECTED_DOCUMENTS and population.document_id.nunique()==EXPECTED_DOCUMENTS,
      "raw_response_preservation_implemented":True,"parsed_response_preservation_implemented":True,
      "offline_idempotent_finalization_implemented":True,"affected_document_retry_package_implemented":True,
      "response_integrity_and_failure_quarantine_implemented":True}
    audit={"protocol":"absa_v1_production_pre_submission_audit_v1","audited_at_utc":datetime.now(timezone.utc).isoformat(),
      "api_calls":0,"request_jsonl_modified":False,"request_jsonl_path":str(request_path),"request_jsonl_sha256":request_hash,
      "request_jsonl_bytes":request_path.stat().st_size,"request_count":len(ids),"unique_custom_ids":len(set(ids)),
      "malformed_requests":malformed,"duplicate_document_ids":duplicates,"missing_document_ids":missing,"unexpected_document_ids":unexpected,
      "empty_captions":empty_captions,"caption_mismatches":caption_mismatches,"configuration_mismatches":configuration_mismatches,
      "caption_length_characters":{"minimum":min(caption_lengths),"median":int(pd.Series(caption_lengths).median()),
        "p95":float(pd.Series(caption_lengths).quantile(.95)),"p99":float(pd.Series(caption_lengths).quantile(.99)),"maximum":max(caption_lengths)},
      "maximum_request_line_bytes":max(line_bytes),"utf8_decode":"PASS","jsonl_one_object_per_physical_line":"PASS",
      "schema":{"identity":SCHEMA_VERSION,"fields":schema_fields,"required_fields":schema_required,"strict_additional_properties":False,
        "model_returns_numeric_evidence_offsets":False,"offset_policy":"Exact evidence_text is required; evidence_start/evidence_end are derived and validated offline against the immutable caption. Ambiguous/unrecoverable grounding is retained and reported.",
        "evidence_extraction_sufficient_for_planned_analysis":True},
      "output_token_policy":{"explicit_max_output_tokens_present":any(value is not None for value in explicit_output_caps),
        "explicit_values":sorted({value for value in explicit_output_caps if value is not None}),
        "assessment":"No request-level max_output_tokens cap is set. The provider/model output ceiling therefore applies. With captions capped at 2,200 characters, a strict compact schema, and minimum-cardinality instructions, plausible multi-aspect output is not expected to approach the model ceiling. Truncation remains possible as a per-request failure and is explicitly detected as incomplete/missing output_text and can be retried individually."},
      "batch_constraints":{"endpoint":"/v1/responses","completion_window":"24h","maximum_requests_per_batch":50000,
        "maximum_input_file_bytes":200000000,"package_request_margin":50000-len(ids),"package_byte_margin":200000000-request_path.stat().st_size,
        "one_batch":True},
      "preservation":{"population_manifest_sha256":population_hash,"local_benchmark_sha256":local_hash,
        "raw_output_path":str(root/"batch"/RAW_OUTPUT_PATH.name),"raw_error_path":str(root/"batch"/RAW_ERROR_PATH.name),
        "raw_import_manifest":str(root/"batch"/IMPORT_MANIFEST_PATH.name),"parsed_responses_path":str(root/PARSED_RESPONSES_PATH.name),
        "response_integrity_path":str(root/RESPONSE_INTEGRITY_PATH.name),"finalization_manifest":str(root/FINALIZATION_MANIFEST_PATH.name),
        "raw_deleted_by_finalizer":False,"parsed_separate_from_raw":True},
      "downstream_offline":{"aspect_prevalence":True,"sentiment_distributions":True,"aspect_x_sentiment":True,
        "year_x_aspect_x_sentiment":True,"final_topic_x_aspect_x_sentiment":True,"language_analysis":True,
        "zero_mention_documents":True,"multi_aspect_documents":True,"grounding_diagnostics":True,
        "dashboard_datasets":True,"statistics_charts_thesis_tables":True,
        "aggregation_visualization_statistics_topic_grouping_reporting_changes_require_inference":False},
      "retry_scope":{"individual_retry_supported":True,"retry_sources":["API errors","missing responses","incomplete/truncated responses","invalid JSON/schema outputs","parse failures","unrecoverable grounding"],
        "full_batch_resubmission_required_for_small_failure_count":False},
      "full_reinference_scenarios":["Changing the prompt or its semantics","Changing the model","Changing the ontology or structured response schema in a way requiring new model fields","Changing the 7,704-document population","Discovering systemic corruption/loss of both raw and parsed outputs","A systemic provider failure affecting most responses","Requiring a new model-derived construct not present or derivable from exact evidence and preserved fields"],
      "offline_only_scenarios":["Changing aggregation logic","Changing visualization/dashboard design","Changing statistics or thesis tables","Regrouping frozen topics","Recomputing prevalence or sentiment distributions","Reparsing preserved raw output","Re-grounding exact evidence against preserved captions"],
      "checks":checks,"submitted":manifest["submitted"],"freeze_selection_status":freeze["selection_status"]}
    audit["verdict"]="SAFE_TO_SUBMIT_WITH_OFFLINE_DOWNSTREAM_ANALYSIS" if all(checks.values()) and manifest["submitted"] is False else "DO_NOT_SUBMIT_YET"
    json_path=root/PRE_SUBMISSION_AUDIT_JSON.name;md_path=root/PRE_SUBMISSION_AUDIT_MD.name
    json_path.write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding="utf-8")
    markdown=f"""# ABSA V1 production pre-submission audit\n\n## Verdict\n\n`{audit['verdict']}`\n\nThe frozen request file was read and hashed but not modified. It contains {len(ids):,} unique, exactly reconciled requests in {request_path.stat().st_size:,} bytes. No API call occurred.\n\n## Schema and downstream sufficiency\n\nThe strict schema preserves all model-derived mention fields: {', '.join(schema_fields)}. Numeric offsets are not returned by the model; they are deterministically derived offline from required exact evidence text and the immutable caption, with ambiguous/unrecoverable grounding retained. The population manifest supplies source, year, language, raw/final topic and taxonomy metadata by document ID.\n\n## Output length\n\nNo explicit `max_output_tokens` cap is present. Plausible outputs for captions capped at {max(caption_lengths):,} characters should fit the provider/model ceiling; incomplete or truncated individual responses are detected and can be retried separately.\n\n## Preservation and recovery\n\nRaw output and error JSONL are preserved immutably before parsing. Parsed responses, integrity reports, failure quarantine, final outputs and hashes are separate artifacts. Finalization is offline and idempotent. Interrupted import resumes by retaining valid downloaded files and downloading only missing files. A retry package can copy only affected original requests without regenerating or resubmitting the full corpus.\n\n## Full re-inference boundary\n\nAggregation, dashboards, statistics, topic regrouping, charts and thesis tables are offline. Full re-inference is warranted only for a changed model/prompt/ontology/schema/population, systemic response loss/corruption, systemic provider failure, or a genuinely new non-derivable model construct.\n"""
    md_path.write_text(markdown,encoding="utf-8")
    if _sha(request_path)!=request_hash or _sha(LOCAL_EVALUATION_PATH)!=local_hash:raise RuntimeError("Frozen artifact changed during audit")
    return audit


def create_retry_package(root:Path=ROOT)->Path:
    _,manifest,request_path=_validate_prepared(root);candidate_ids=set()
    integrity_path=root/RESPONSE_INTEGRITY_PATH.name
    if integrity_path.exists():
        integrity=json.loads(integrity_path.read_text(encoding="utf-8"))
        candidate_ids.update(integrity.get("missing_document_ids",[]));candidate_ids.update(integrity.get("duplicate_document_ids",[]))
    failure_path=root/QUARANTINE_PATH.name
    if failure_path.exists():candidate_ids.update(pd.read_csv(failure_path,dtype={"document_id":str}).document_id.dropna())
    summary_path=root/SUMMARY_PATH.name
    if summary_path.exists():candidate_ids.update(json.loads(summary_path.read_text(encoding="utf-8")).get("grounding_failure_documents",[]))
    if not candidate_ids:raise RuntimeError("No affected production document IDs are available for retry")
    retry_root=root/"retry"/f"retry_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}";retry_root.mkdir(parents=True)
    selected=[]
    with request_path.open("r",encoding="utf-8") as handle:
        for line in handle:
            item=json.loads(line);doc=item["custom_id"].removeprefix("absa1prod-")
            if doc in candidate_ids:selected.append(line.rstrip("\n"))
    found={json.loads(line)["custom_id"].removeprefix("absa1prod-") for line in selected}
    if found!=candidate_ids:raise RuntimeError(f"Retry IDs not found in frozen request package: {sorted(candidate_ids-found)}")
    retry_path=retry_root/"absa_v1_production_retry_requests.jsonl";retry_path.write_text("\n".join(selected)+"\n",encoding="utf-8")
    retry_manifest={"protocol":"absa_v1_production_retry_v1","source_request_sha256":manifest["request_jsonl_sha256"],
      "request_count":len(selected),"document_ids":sorted(candidate_ids),"request_sha256":_sha(retry_path),"submitted":False,"batch_id":None,"api_calls":0,
      "scope":"affected documents only; never the full 7,704-document package"}
    (retry_root/"retry_manifest.json").write_text(json.dumps(retry_manifest,indent=2),encoding="utf-8")
    return retry_path


def submit_production(root:Path=ROOT)->dict[str,Any]:
    from openai import OpenAI
    _,manifest,path=_validate_prepared(root);submission=root/"batch"/SUBMISSION_PATH.name
    if submission.exists(): raise FileExistsError("Production Batch was already submitted")
    client=OpenAI()
    with path.open("rb") as handle:uploaded=client.files.create(file=handle,purpose="batch")
    batch=client.batches.create(input_file_id=uploaded.id,endpoint="/v1/responses",completion_window="24h",metadata={"stage":PROTOCOL,"prompt":PROMPT_VERSION})
    record={"batch_id":batch.id,"input_file_id":uploaded.id,"status":batch.status,"request_count":manifest["request_count"],"submitted_at_utc":datetime.now(timezone.utc).isoformat()}
    submission.write_text(json.dumps(record,indent=2),encoding="utf-8");return record


def production_status(root:Path=ROOT)->dict[str,Any]:
    from openai import OpenAI
    path=root/"batch"/SUBMISSION_PATH.name
    if not path.exists():return {"status":"not_submitted","batch_id":None}
    submission=json.loads(path.read_text(encoding="utf-8"));record=OpenAI().batches.retrieve(submission["batch_id"]).model_dump()
    (root/"batch"/STATUS_PATH.name).write_text(json.dumps(record,indent=2),encoding="utf-8");return record


def import_production(root:Path=ROOT)->dict[str,Any]:
    from openai import OpenAI
    submission=json.loads((root/"batch"/SUBMISSION_PATH.name).read_text(encoding="utf-8"));status=json.loads((root/"batch"/STATUS_PATH.name).read_text(encoding="utf-8"))
    if status.get("id")!=submission["batch_id"]:raise RuntimeError("Production Batch ID mismatch")
    if status.get("status")!="completed":raise RuntimeError(f"Production Batch is not completed: {status.get('status')}")
    client=OpenAI();result={};artifacts={}
    for key,name in [("output_file_id",RAW_OUTPUT_PATH.name),("error_file_id",RAW_ERROR_PATH.name)]:
        file_id=status.get(key);path=root/"batch"/name
        if path.exists():
            # A valid existing raw artifact is immutable and makes interrupted import resumable.
            with path.open("r",encoding="utf-8") as handle:
                for line_number,line in enumerate(handle,1):
                    if line.strip():
                        try: json.loads(line)
                        except Exception as exc: raise RuntimeError(f"Existing raw import is malformed at {path}:{line_number}") from exc
            result[key]=str(path) if file_id else None
        elif file_id:
            temporary=path.with_suffix(path.suffix+".partial")
            temporary.write_bytes(client.files.content(file_id).content)
            with temporary.open("r",encoding="utf-8") as handle:
                for line_number,line in enumerate(handle,1):
                    if line.strip(): json.loads(line)
            temporary.replace(path);result[key]=str(path)
        else:path.write_text("",encoding="utf-8");result[key]=None
        artifacts[key]={"file_id":file_id,"path":str(path),"sha256":_sha(path),"bytes":path.stat().st_size}
    import_manifest={"protocol":PROTOCOL,"batch_id":submission["batch_id"],"imported_at_utc":datetime.now(timezone.utc).isoformat(),
                     "raw_artifacts":artifacts,"raw_files_immutable":True,"resume_safe":True,"api_calls_only_for_missing_files":True}
    manifest_path=root/"batch"/IMPORT_MANIFEST_PATH.name
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text(encoding="utf-8"))
        for key,value in previous.get("raw_artifacts",{}).items():
            if value.get("sha256")!=artifacts[key]["sha256"]:raise RuntimeError("Previously imported raw artifact hash changed")
    else:manifest_path.write_text(json.dumps(import_manifest,indent=2),encoding="utf-8")
    return result


def _reconcile_response_ids(expected:set[str], response_ids:list[str])->None:
    if len(response_ids)!=len(set(response_ids)):raise ValueError("Duplicate production Batch response IDs")
    unexpected=set(response_ids)-expected;missing=expected-set(response_ids)
    if unexpected or missing:raise ValueError(f"Production Batch response mismatch: unexpected={len(unexpected)}, missing={len(missing)}")


def _base_document_result(row:Any)->dict[str,Any]:
    return {"document_id":str(row.document_id),"source":row.source,"event_year":row.event_year,"primary_language":row.primary_language,
            "raw_c1_topic_id":row.original_topic_id,"raw_c1_topic_label":row.original_human_topic_name,
            "final_topic_id":row.final_topic_id,"final_topic_name":row.final_topic_name,"final_topic_group":row.final_topic_group,
            "taxonomy_status":"substantive_absa_ready","mention_count":0,
            "unique_aspect_count":0,"prediction_status":"zero_mentions","parse_status":"success","failure_message":""}


def finalize_production(root:Path=ROOT)->Path:
    _,_,_=_validate_prepared(root);source=_population();meta=_population_manifest(source).set_index("document_id")
    output=root/"batch"/RAW_OUTPUT_PATH.name;errors=root/"batch"/RAW_ERROR_PATH.name
    if not output.exists() or not errors.exists():raise FileNotFoundError("Import production Batch output and error files first")
    finalization_manifest=root/FINALIZATION_MANIFEST_PATH.name
    if finalization_manifest.exists():
        frozen=json.loads(finalization_manifest.read_text(encoding="utf-8"))
        if frozen["raw_output_sha256"]!=_sha(output) or frozen["raw_error_sha256"]!=_sha(errors):
            raise RuntimeError("Raw Batch artifacts changed after finalization")
        for name,digest in frozen["derived_artifact_sha256"].items():
            if _sha(root/name)!=digest:raise RuntimeError(f"Finalized artifact changed: {name}")
        return root/REPORT_PATH.name
    expected=set(source.document_id);items=[];malformed=[]
    for path,kind in [(output,"output"),(errors,"error")]:
        with path.open("r",encoding="utf-8") as handle:
            for line_number,line in enumerate(handle,1):
                if not line.strip():continue
                try:items.append((kind,json.loads(line)))
                except Exception as exc:malformed.append({"file":str(path),"line_number":line_number,"error":f"{type(exc).__name__}: {exc}"})
    ids=[item.get("custom_id","").removeprefix("absa1prod-") for _,item in items]
    duplicate_ids=sorted(key for key,count in Counter(ids).items() if count>1)
    unexpected_ids=sorted(set(ids)-expected);missing_ids=sorted(expected-set(ids))
    integrity={"protocol":PROTOCOL,"checked_at_utc":datetime.now(timezone.utc).isoformat(),"expected_responses":EXPECTED_DOCUMENTS,
      "parsed_raw_lines":len(items),"malformed_raw_lines":malformed,"duplicate_document_ids":duplicate_ids,
      "unexpected_document_ids":unexpected_ids,"missing_document_ids":missing_ids,
      "raw_output_sha256":_sha(output),"raw_error_sha256":_sha(errors),
      "passed":not malformed and not duplicate_ids and not unexpected_ids and not missing_ids}
    (root/RESPONSE_INTEGRITY_PATH.name).write_text(json.dumps(integrity,indent=2),encoding="utf-8")
    if not integrity["passed"]:
        raise ValueError(f"Production Batch response integrity failed: malformed={len(malformed)}, duplicate={len(duplicate_ids)}, unexpected={len(unexpected_ids)}, missing={len(missing_ids)}")
    ontology=[x["id"] for x in load_ontology()["aspects"]];mentions=[];documents=[];failures=[]
    schema=mention_schema(ontology);parsed_records=[]
    by_id={item.get("custom_id","").removeprefix("absa1prod-"):(kind,item) for kind,item in items}
    for row in source.itertuples(index=False):
        doc=str(row.document_id);kind,item=by_id[doc];parse_status="success";prediction_status="zero_mentions";doc_mentions=[];message=""
        try:
            if kind=="error" or item.get("error"):raise ValueError(json.dumps(item.get("error") or item,ensure_ascii=False)[:2000])
            response=item.get("response",{})
            if response.get("status_code") not in (None,200):raise ValueError(f"HTTP status {response.get('status_code')}")
            body=response.get("body",{})
            if body.get("status")=="incomplete":raise ValueError(f"Incomplete response: {body.get('incomplete_details')}")
            texts=[c.get("text","") for o in body.get("output",[]) for c in o.get("content",[]) if c.get("type")=="output_text"]
            if len(texts)!=1:raise ValueError("Expected exactly one output_text")
            payload=json.loads(texts[0])
            from jsonschema import validate as validate_json_schema
            validate_json_schema(payload,schema)
            raw_mentions=payload.get("mentions")
            if not isinstance(raw_mentions,list):raise ValueError("mentions must be an array")
            for index,m in enumerate(raw_mentions):
                if m.get("aspect") not in ontology or m.get("sentiment") not in SENTIMENTS:raise ValueError("Invalid aspect/sentiment")
                rec=canonicalize_evidence(str(row.original_text),m.get("evidence_text",""),m.get("evidence_start"),m.get("evidence_end"));evidence=rec["final_exact_evidence_text"]
                doc_mentions.append({"mention_id":stable_mention_id(doc,index,m["aspect"],evidence or rec["original_model_evidence_text"]),"document_id":doc,"mention_index":index,
                  **m,"evidence_text":evidence,"evidence_start":rec["final_evidence_start"],"evidence_end":rec["final_evidence_end"],
                  "raw_exact_match":rec["raw_exact_match"],"evidence_repaired":rec["evidence_repaired"],"evidence_repair_method":rec["evidence_repair_method"],
                  "original_model_evidence_text":rec["original_model_evidence_text"],"grounding_status":rec["normalized_match_status"],
                  "source":row.source,"event_year":row.event_year,"primary_language":row.primary_language,"raw_c1_topic_id":row.original_topic_id,
                  "raw_c1_topic_label":row.original_human_topic_name,"final_topic_id":row.final_topic_id,"final_topic_name":row.final_topic_name,
                  "final_topic_group":row.final_topic_group,"taxonomy_status":"substantive_absa_ready","prompt_version":PROMPT_VERSION,"model":DEFAULT_MODEL,"schema_version":SCHEMA_VERSION})
            prediction_status="mentions" if doc_mentions else "zero_mentions";mentions.extend(doc_mentions)
        except Exception as exc:
            parse_status="failed";prediction_status="failed";message=f"{type(exc).__name__}: {exc}";failures.append({"document_id":doc,"failure":message,"raw_kind":kind})
        parsed_records.append({"document_id":doc,"custom_id":f"absa1prod-{doc}","raw_kind":kind,
          "parse_status":parse_status,"prediction_status":prediction_status,"failure_message":message,"mentions":doc_mentions})
        document_result=_base_document_result(row)
        document_result.update({"mention_count":len(doc_mentions),"unique_aspect_count":len({x["aspect"] for x in doc_mentions}),
                                "prediction_status":prediction_status,"parse_status":parse_status,"failure_message":message})
        documents.append(document_result)
    document_df=pd.DataFrame(documents);mention_df=pd.DataFrame(mentions);failure_df=pd.DataFrame(failures,columns=["document_id","failure","raw_kind"])
    parsed_content="".join(json.dumps(record,ensure_ascii=True)+"\n" for record in parsed_records)
    parsed_path=root/PARSED_RESPONSES_PATH.name
    if parsed_path.exists() and hashlib.sha256(parsed_path.read_bytes()).hexdigest()!=hashlib.sha256(parsed_content.encode()).hexdigest():
        raise RuntimeError("Existing parsed-response artifact differs from deterministic reparse")
    parsed_path.write_text(parsed_content,encoding="utf-8")
    def replace_csv(frame:pd.DataFrame,path:Path)->None:
        temporary=path.with_suffix(path.suffix+".partial");frame.to_csv(temporary,index=False,encoding="utf-8-sig");temporary.replace(path)
    replace_csv(document_df,root/DOCUMENT_RESULTS_PATH.name);replace_csv(mention_df,root/MENTIONS_PATH.name);replace_csv(failure_df,root/QUARANTINE_PATH.name)
    successful=document_df.parse_status.eq("success");mention_total=len(mention_df);raw_exact=int(mention_df.raw_exact_match.sum()) if mention_total else 0
    recovered=int((mention_df.raw_exact_match|mention_df.evidence_repaired).sum()) if mention_total else 0
    aspect_counts=mention_df.aspect.value_counts().to_dict() if mention_total else {};sentiment_counts=mention_df.sentiment.value_counts().to_dict() if mention_total else {}
    v3=pd.read_csv(V3_PREDICTIONS_PATH).fillna("");v3_docs=80;v3_mentions=len(v3);v3_mention_docs=v3.document_id.astype(str).nunique()
    summary={"protocol":PROTOCOL,"finalized_at_utc":datetime.now(timezone.utc).isoformat(),"expected_documents":EXPECTED_DOCUMENTS,
      "submitted_requests":EXPECTED_DOCUMENTS,"returned_responses":len(items),"successfully_parsed_documents":int(successful.sum()),
      "failed_documents":int((~successful).sum()),"documents_with_mentions":int(document_df.mention_count.gt(0).sum()),
      "zero_mention_documents":int((successful&document_df.mention_count.eq(0)).sum()),"total_extracted_mentions":mention_total,
      "mentions_per_document":mention_total/EXPECTED_DOCUMENTS,"zero_mention_rate":float((successful&document_df.mention_count.eq(0)).mean()),
      "multi_aspect_document_rate":float(document_df.unique_aspect_count.gt(1).mean()),"aspect_family_distribution":aspect_counts,
      "sentiment_distribution":sentiment_counts,"aspect_x_sentiment":mention_df.groupby(["aspect","sentiment"]).size().to_dict() if mention_total else {},
      "year_distribution":document_df.event_year.value_counts().to_dict(),"topic_distribution":document_df.final_topic_name.value_counts().to_dict(),
      "language_distribution":document_df.primary_language.value_counts().to_dict(),"strict_grounding_rate":raw_exact/mention_total if mention_total else 1,
      "recoverable_grounding_rate":recovered/mention_total if mention_total else 1,"unrecoverable_grounding_failures":mention_total-recovered,
      "grounding_failure_documents":mention_df.loc[~(mention_df.raw_exact_match|mention_df.evidence_repaired),"document_id"].drop_duplicates().tolist() if mention_total else [],
      "extreme_mentions_per_document":document_df[document_df.mention_count.gt(10)].document_id.tolist(),
      "development_to_production_drift":{"label":"distributional diagnostic only; not accuracy evidence","v3_development_mentions_per_document":v3_mentions/v3_docs,
        "production_mentions_per_document":mention_total/EXPECTED_DOCUMENTS,"v3_development_zero_rate":1-v3_mention_docs/v3_docs,
        "production_zero_rate":float((successful&document_df.mention_count.eq(0)).mean()),"v3_aspect_proportions":v3.aspect.value_counts(normalize=True).to_dict() if v3_mentions else {},
        "production_aspect_proportions":mention_df.aspect.value_counts(normalize=True).to_dict() if mention_total else {},
        "v3_sentiment_proportions":v3.sentiment.value_counts(normalize=True).to_dict() if v3_mentions else {},
        "production_sentiment_proportions":mention_df.sentiment.value_counts(normalize=True).to_dict() if mention_total else {}},
      "methodological_limitation":LIMITATION,"full_corpus_is_validation":False}
    # JSON does not support tuple keys.
    summary["aspect_x_sentiment"]={f"{a}|{s}":int(n) for (a,s),n in summary["aspect_x_sentiment"].items()}
    summary_path=root/SUMMARY_PATH.name;summary_path.write_text(json.dumps(summary,indent=2,ensure_ascii=False,default=float),encoding="utf-8")
    report=f"""# ABSA V1 Production Report\n\nSelected system: `{PROMPT_VERSION}`. Production is not validation.\n\nDocuments: {EXPECTED_DOCUMENTS}; parsed: {summary['successfully_parsed_documents']}; failed: {summary['failed_documents']}; mentions: {mention_total}; zero-mention documents: {summary['zero_mention_documents']}.\n\n## Known limitation\n\n{LIMITATION}\n\nDevelopment-to-production comparisons are distributional diagnostics only, not accuracy evidence. Future refinement requires a new version; these V1 production artifacts remain frozen.\n"""
    report_path=root/REPORT_PATH.name;report_path.write_text(report,encoding="utf-8")
    derived=[parsed_path,root/DOCUMENT_RESULTS_PATH.name,root/MENTIONS_PATH.name,root/QUARANTINE_PATH.name,summary_path,report_path,root/RESPONSE_INTEGRITY_PATH.name]
    finalization={"protocol":PROTOCOL,"finalized_at_utc":datetime.now(timezone.utc).isoformat(),"api_calls":0,"offline":True,
      "rerunnable_without_api":True,"raw_files_deleted":False,"raw_output_sha256":_sha(output),"raw_error_sha256":_sha(errors),
      "derived_artifact_sha256":{path.name:_sha(path) for path in derived}}
    finalization_manifest.write_text(json.dumps(finalization,indent=2),encoding="utf-8")
    return report_path
