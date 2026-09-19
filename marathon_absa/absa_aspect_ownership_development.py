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
    AUDIT_ROOT, CORPUS, DEFAULT_MODEL, GOLD_PATH, ONTOLOGY_PATH, SAMPLE_PATH,
    SCHEMA_VERSION, SENTIMENTS, canonicalize_evidence, load_ontology,
    mention_schema, stable_mention_id, validation_metrics,
)
from .absa_precision_development import (
    BASELINE, CANDIDATE_CACHE_STAGE as V2_CACHE_STAGE,
    CANDIDATE_INSTRUCTIONS as V2_INSTRUCTIONS,
    CANDIDATE_PROMPT_VERSION as V2_PROMPT_VERSION,
    ROOT as V2_ROOT, SUCCESS_GATES,
)
from .absa_residual_diagnostic import _false_negatives, _metric_row, _statuses

EXPERIMENT_ID = "absa_v1_aspect_ownership_prompt_development_v1"
PROMPT_VERSION = "absa_v1_instructions_3_aspect_ownership"
CACHE_STAGE = "absa_v1_mentions_aspect_ownership_development_v1"
ROOT = Path("data/processed/absa_v1/development") / EXPERIMENT_ID
PROMPT_PATH = ROOT / "absa_v1_instructions_3_aspect_ownership.txt"
DIFF_PATH = ROOT / "absa_v1_instructions_2_to_3_diff_v1.txt"
SPEC_PATH = ROOT / "experiment_manifest_v1.json"
BATCH_ROOT = ROOT / "batch"
REQUEST_PATH = BATCH_ROOT / "absa_v1_aspect_ownership_requests.jsonl"
REQUEST_MANIFEST_PATH = BATCH_ROOT / "request_manifest_v1.json"
SUBMISSION_PATH = BATCH_ROOT / "submission.json"
STATUS_PATH = BATCH_ROOT / "status.json"
RAW_PATH = BATCH_ROOT / "absa_v1_aspect_ownership_raw_results.jsonl"
PREDICTIONS_PATH = ROOT / "candidate_predictions_v3_v1.csv"
V2_PREDICTIONS = V2_ROOT / "candidate_predictions_v1.csv"
V2_RESIDUAL = V2_ROOT / "candidate_residual_fp_review_all_74_v1.csv"

OWNERSHIP_POLICY = r"""

ASPECT OWNERSHIP AND MENTION CARDINALITY EXTENSION

Reason at the level of evaluative propositions. An evaluative proposition is one coherent statement or
expression communicating a judgment, affect, experience, preference, praise, complaint, achievement,
difficulty, or other qualifying evaluation. For each proposition, conceptually follow:
EVALUATIVE PROPOSITION -> WHAT IS ACTUALLY EVALUATED? -> PRIMARY ASPECT OWNER -> SENTIMENT TOWARD THAT OWNER
-> MENTION. The primary owner is the aspect whose quality, outcome, condition, behavior, experience, or effect
is actually judged. A related concept in the proposition is not automatically another evaluated aspect.

Emotion, difficulty, achievement, pain, gratitude, excitement, or disappointment may be evidence about an
object-level aspect rather than a second evaluated target. Do not emit emotional_experience merely because
emotional language expresses sentiment toward another owner. "I loved the course" normally belongs to
route_course; love supplies its polarity. "Proud that I beat my previous time" normally belongs to
race_performance. Emit emotional_experience when the emotional state itself is a distinct substantive
experience, such as sustained anxiety throughout the race. Apply the same ownership test to physical language:
preserve substantive pain, impairment, exhaustion, strain, or inability, but do not add physical_experience
when effort language only explains another evaluated outcome.

Do not map encouragement, hope, motivation, future intention, generic pride, or generic finishing language to
race_performance unless actual performance is evaluated. Preserve comparative performance, improvement,
failure against a target, achievement, evaluatively framed DNF, completion pride, and implicit performance
judgment. Community-directed praise, encouragement, gratitude, support, and congratulations remain legitimate
when runners, supporters, or community are the actual owner; ask who is praised or encouraged instead of
suppressing the proposition or inferring an achieved result. Preserve locally owned operational complaints
inside long narratives, including organization, registration, information, facilities, transport, and aid
stations.

One evaluative proposition should normally produce one mention for its primary owner. This is a default, not a
one-aspect-per-sentence rule. Emit multiple families only for multiple distinct evaluated targets. Before a
second mention from the same local proposition, ask whether it is genuinely another evaluated target or only
another interpretation of the same affect/evidence. Keep the best-supported owner in the latter case. Avoid
semantic redundancy, but preserve separate evaluative propositions even when they share an aspect or occur
near one another. Do not minimize raw mention count or collapse distinct gold-policy-valid propositions.

When evaluation clearly exists but an initially considered family is wrong, remap it to the correct supported
owner before considering suppression. Ownership does not require explicit "I like X" syntax. Preserve implicit
achievement, difficulty, comparison, impairment, fear, enjoyment, gratitude, encouragement, pride,
disappointment, strain, and insufficient preparation. Protect Malay colloquial language, Malay/Indonesian
code-switching, Chinese comparison, hashtags, emojis, abbreviations, informal spelling, and race-community
slang. Short or hashtag-led propositions may qualify when evaluation and owner are identifiable; generic affect
without an owner does not.

A prospective statement is not an achieved outcome, but it may contain a legitimate evaluation owned by
community encouragement, a strategy, or a preference. Preserve qualifying neutral selections, strategies,
judgments, comparisons, and experiences under the existing neutral policy; a bare fact remains insufficient.

Small synthetic ownership contrasts:
- "I absolutely loved the course." -> route_course positive; do not automatically add emotional_experience.
- "I felt anxious from start to finish." -> emotional_experience negative because emotion is the target.
- "So proud that I finally beat my previous time." -> primarily race_performance positive.
- "The route was gorgeous, but hydration stations were disappointing." -> two distinct owners; extract both.
- "Good luck everyone - you've got this!" -> no achieved result; preserve community encouragement when the
  frozen ontology supports that owner.
- "My injured leg couldn't generate any power." -> preserve physical_experience negative.

This extension changes attribution and semantic cardinality only. Retain every V2 rule for local entailment,
facts/context, prospective outcomes, sentiment spillover, unsupported inference, multilingual and implicit
evaluation, neutral/mixed sentiment, exact evidence, offsets, ontology, schema, and output structure.
"""

INSTRUCTIONS = V2_INSTRUCTIONS + OWNERSHIP_POLICY


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source() -> pd.DataFrame:
    sample=pd.read_csv(SAMPLE_PATH); corpus=pd.read_csv(CORPUS,low_memory=False)
    ids=sample.document_id.astype(str).tolist()
    if len(ids)!=80 or len(set(ids))!=80: raise ValueError("V3 requires exactly 80 frozen documents")
    corpus.document_id=corpus.document_id.astype(str)
    source=corpus[corpus.document_id.isin(ids)].set_index("document_id").loc[ids].reset_index()
    if source.document_id.tolist()!=ids: raise ValueError("V3 population/order differs from frozen sample")
    return source


def _frozen_hashes() -> dict[str,str]:
    marker=AUDIT_ROOT/"absa_v1_audit_gold_v1.frozen.json"
    v2_prompt=V2_ROOT/"absa_v1_instructions_2_precision.txt"
    for path in [SAMPLE_PATH,GOLD_PATH,marker,ONTOLOGY_PATH,CORPUS,v2_prompt,V2_PREDICTIONS,V2_RESIDUAL]:
        if not path.exists(): raise FileNotFoundError(path)
    return {"sample_sha256":_sha(SAMPLE_PATH),"gold_sha256":_sha(GOLD_PATH),
      "gold_marker_sha256":_sha(marker),"ontology_sha256":_sha(ONTOLOGY_PATH),"corpus_sha256":_sha(CORPUS),
      "v2_prompt_sha256":_sha(v2_prompt),"v2_predictions_sha256":_sha(V2_PREDICTIONS),
      "v2_residual_74_sha256":_sha(V2_RESIDUAL)}


def _estimate(source:pd.DataFrame)->dict[str,Any]:
    overhead=len(INSTRUCTIONS)+len(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    input_tokens=sum(math.ceil((len(str(x))+overhead)/4) for x in source.original_text); output_tokens=len(source)*260
    synchronous=(input_tokens*1.25+output_tokens*10)/1_000_000
    return {"estimated_input_tokens":input_tokens,"estimated_output_tokens":output_tokens,
            "estimated_synchronous_cost_usd":round(synchronous,4),"estimated_batch_cost_usd":round(synchronous*.5,4),
            "pricing_assumptions_usd_per_million":{"input":1.25,"output":10.0,"batch_discount":.5}}


def create_experiment(root:Path=ROOT)->dict[str,Any]:
    source=_source(); hashes=_frozen_hashes(); v3_hash=_text_sha(INSTRUCTIONS)
    if v3_hash==hashes["v2_prompt_sha256"]: raise ValueError("V3 prompt must differ from V2")
    prompt_path=root/PROMPT_PATH.name; diff_path=root/DIFF_PATH.name; spec_path=root/SPEC_PATH.name
    request_path=root/"batch"/REQUEST_PATH.name; request_manifest_path=root/"batch"/REQUEST_MANIFEST_PATH.name
    spec={"experiment_id":EXPERIMENT_ID,"created_at_utc":datetime.now(timezone.utc).isoformat(),
      "candidate_prompt":PROMPT_VERSION,"candidate_cache_stage":CACHE_STAGE,"parent_prompt":V2_PROMPT_VERSION,
      "parent_cache_stage":V2_CACHE_STAGE,"model":DEFAULT_MODEL,"schema":SCHEMA_VERSION,
      "development_evidence":True,"confirmatory_evidence":False,"only_prompt_changed":True,
      "model_unchanged":True,"schema_unchanged":True,"ontology_unchanged":True,"gold_unchanged":True,
      "sample_unchanged":True,"preprocessing_unchanged":True,"matching_evaluator_unchanged":True,
      "inference_parameters_unchanged":True,"postprocessing_added":False,"code_deduplication_added":False,
      "full_corpus_prohibited":True,"candidate_count":1,"v2_prompt_sha256":hashes["v2_prompt_sha256"],
      "v3_prompt_sha256":v3_hash,"success_gates":SUCCESS_GATES,
      "success_gates_frozen_before_inference":True,"frozen_input_hashes":hashes,
      "intervention_scope":"single-proposition aspect ownership and mention cardinality; remap/attribute/collapse rather than broader suppression"}
    immutable=["candidate_prompt","candidate_cache_stage","parent_prompt","model","schema","only_prompt_changed",
               "v2_prompt_sha256","v3_prompt_sha256","success_gates","frozen_input_hashes","intervention_scope"]
    if spec_path.exists():
        old=json.loads(spec_path.read_text(encoding="utf-8"))
        if any(old.get(k)!=spec.get(k) for k in immutable): raise FileExistsError("Frozen V3 specification differs")
    else:
        root.mkdir(parents=True,exist_ok=True); prompt_path.write_text(INSTRUCTIONS,encoding="utf-8")
        diff_path.write_text(OWNERSHIP_POLICY.strip()+"\n",encoding="utf-8")
        spec_path.write_text(json.dumps(spec,indent=2,ensure_ascii=False),encoding="utf-8")
    aspects=[x["id"] for x in load_ontology()["aspects"]]; schema=mention_schema(aspects); lines=[]
    for row in source.itertuples(index=False):
        context={"document_id":str(row.document_id),"original_text":str(row.original_text),
                 "final_topic_name":str(row.final_topic_name),"final_topic_group":str(row.final_topic_group),
                 "topic_context_supplied":True,"ontology":aspects}
        body={"model":DEFAULT_MODEL,"instructions":INSTRUCTIONS,"input":json.dumps(context,ensure_ascii=False),
              "text":{"format":{"type":"json_schema","name":SCHEMA_VERSION,"strict":True,"schema":schema}}}
        lines.append(json.dumps({"custom_id":f"absa1own-{row.document_id}","method":"POST","url":"/v1/responses","body":body},ensure_ascii=False))
    content="\n".join(lines)+"\n"; request_path.parent.mkdir(parents=True,exist_ok=True)
    if request_path.exists() and request_path.read_text(encoding="utf-8")!=content: raise FileExistsError("Existing V3 requests differ")
    request_path.write_text(content,encoding="utf-8"); costs=_estimate(source)
    request_manifest={"experiment_id":EXPERIMENT_ID,"document_count":80,"request_count":80,
      "document_ids_sha256":hashlib.sha256("\n".join(source.document_id).encode()).hexdigest(),
      "request_sha256":_sha(request_path),"model":DEFAULT_MODEL,"prompt_version":PROMPT_VERSION,
      "prompt_sha256":v3_hash,"parent_prompt_sha256":hashes["v2_prompt_sha256"],"schema_version":SCHEMA_VERSION,
      "cache_stage":CACHE_STAGE,"submitted":False,"development_evidence":True,"confirmatory_evidence":False,
      "full_corpus":False,"cost_estimate":costs}
    request_manifest_path.write_text(json.dumps(request_manifest,indent=2),encoding="utf-8")
    return {"experiment_manifest":str(spec_path),"prompt_artifact":str(prompt_path),"prompt_diff":str(diff_path),
            "request_path":str(request_path),**request_manifest}


def _validate(root:Path=ROOT)->tuple[dict[str,Any],dict[str,Any],Path]:
    spec=json.loads((root/SPEC_PATH.name).read_text(encoding="utf-8")); path=root/"batch"/REQUEST_PATH.name
    manifest=json.loads((root/"batch"/REQUEST_MANIFEST_PATH.name).read_text(encoding="utf-8"))
    if spec["success_gates"]!=SUCCESS_GATES or not spec["success_gates_frozen_before_inference"]:
        raise RuntimeError("V3 success gates changed")
    if spec["frozen_input_hashes"]!=_frozen_hashes(): raise RuntimeError("Frozen input changed")
    prompt_text=(root/PROMPT_PATH.name).read_text(encoding="utf-8")
    if _text_sha(prompt_text)!=spec["v3_prompt_sha256"]: raise RuntimeError("V3 prompt hash changed")
    if _sha(path)!=manifest["request_sha256"]: raise RuntimeError("V3 request hash changed")
    rows=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    ids={x["custom_id"].removeprefix("absa1own-") for x in rows}
    if len(rows)!=80 or ids!=set(_source().document_id): raise RuntimeError("V3 population is not frozen 80")
    if any(x["body"]["model"]!=DEFAULT_MODEL for x in rows): raise RuntimeError("V3 model changed")
    return spec,manifest,path


def submit_experiment(root:Path=ROOT)->dict[str,Any]:
    from openai import OpenAI
    _,manifest,path=_validate(root); submission=root/"batch"/SUBMISSION_PATH.name
    if submission.exists(): raise FileExistsError("V3 Batch already submitted")
    client=OpenAI()
    with path.open("rb") as handle: uploaded=client.files.create(file=handle,purpose="batch")
    batch=client.batches.create(input_file_id=uploaded.id,endpoint="/v1/responses",completion_window="24h",
                                metadata={"stage":EXPERIMENT_ID,"prompt":PROMPT_VERSION})
    record={"batch_id":batch.id,"input_file_id":uploaded.id,"status":batch.status,
            "request_count":manifest["request_count"],"submitted_at_utc":datetime.now(timezone.utc).isoformat()}
    submission.write_text(json.dumps(record,indent=2),encoding="utf-8"); return record


def experiment_status(root:Path=ROOT)->dict[str,Any]:
    from openai import OpenAI
    sub=json.loads((root/"batch"/SUBMISSION_PATH.name).read_text(encoding="utf-8"))
    record=OpenAI().batches.retrieve(sub["batch_id"]).model_dump()
    (root/"batch"/STATUS_PATH.name).write_text(json.dumps(record,indent=2),encoding="utf-8"); return record


def import_experiment(root:Path=ROOT)->Path:
    from openai import OpenAI
    status=json.loads((root/"batch"/STATUS_PATH.name).read_text(encoding="utf-8"))
    if status.get("status")!="completed" or not status.get("output_file_id"): raise RuntimeError("V3 Batch not complete")
    path=root/"batch"/RAW_PATH.name
    if path.exists(): raise FileExistsError("V3 raw output already exists")
    path.write_bytes(OpenAI().files.content(status["output_file_id"]).content); return path


def _parse_predictions(root:Path)->pd.DataFrame:
    raw=root/"batch"/RAW_PATH.name; corpus=pd.read_csv(CORPUS,low_memory=False)
    corpus.document_id=corpus.document_id.astype(str); corpus=corpus.set_index("document_id")
    allowed=set(_source().document_id); aspects=[x["id"] for x in load_ontology()["aspects"]]
    rows=[]; processed=set()
    for line in raw.read_text(encoding="utf-8").splitlines():
        item=json.loads(line); doc=item["custom_id"].removeprefix("absa1own-")
        if doc not in allowed or doc in processed: raise ValueError("Unknown/duplicate V3 document")
        processed.add(doc); body=item.get("response",{}).get("body",{})
        text=[c.get("text","") for o in body.get("output",[]) for c in o.get("content",[]) if c.get("type")=="output_text"]
        if len(text)!=1: raise ValueError("V3 result requires one output_text")
        mentions=json.loads(text[0]).get("mentions"); caption=str(corpus.loc[doc].original_text)
        if not isinstance(mentions,list): raise ValueError("mentions must be array")
        for i,m in enumerate(mentions):
            if m.get("aspect") not in aspects or m.get("sentiment") not in SENTIMENTS: raise ValueError("Invalid V3 mention")
            rec=canonicalize_evidence(caption,m.get("evidence_text",""),m.get("evidence_start"),m.get("evidence_end")); evidence=rec["final_exact_evidence_text"]
            rows.append({"mention_id":stable_mention_id(doc,i,m["aspect"],evidence or rec["original_model_evidence_text"]),
              "document_id":doc,**m,"evidence_text":evidence,"evidence_start":rec["final_evidence_start"],"evidence_end":rec["final_evidence_end"],
              "raw_exact_match":rec["raw_exact_match"],"evidence_repaired":rec["evidence_repaired"],
              "original_model_evidence_text":rec["original_model_evidence_text"],"prompt_version":PROMPT_VERSION,
              "schema_version":SCHEMA_VERSION,"model":body.get("model",DEFAULT_MODEL)})
    if processed!=allowed: raise ValueError("V3 results do not cover frozen 80")
    columns=["mention_id","document_id","aspect","target","sentiment","confidence","evidence_text","expression_type","language","english_gloss",
      "contributing_hashtags","contributing_emoji","emerging_aspect_name","analysis_notes","evidence_start","evidence_end","raw_exact_match",
      "evidence_repaired","original_model_evidence_text","prompt_version","schema_version","model"]
    return pd.DataFrame(rows,columns=columns)


def _decision(metrics:dict[str,Any])->str:
    a=metrics["aspect_detection"];j=metrics["joint_aspect_sentiment"]
    zero=metrics["document_diagnostics"]["zero_mention_confusion_gold_rows_predicted_columns"]["gold_zero_predicted_mentions"]
    if a["precision"]>=.55 and a["recall"]<.75:return "TOO_CONSERVATIVE"
    if a["precision"]>=.55 and a["recall"]>=.75 and j["precision"]>.3302752 and j["f1"]>=.4528302 and zero<=7:
        return "PASS_FOR_CONFIRMATORY_TESTING_PENDING_QUALITATIVE_SAFEGUARD"
    if a["precision"]>.5099337748344371 and a["precision"]<.55:return "INSUFFICIENT_PRECISION_IMPROVEMENT"
    return "FAILED"


def finalize_experiment(root:Path=ROOT)->Path:
    spec,_,_=_validate(root); v3=_parse_predictions(root); sample=pd.read_csv(SAMPLE_PATH).fillna("");gold=pd.read_csv(GOLD_PATH).fillna("")
    v1=pd.read_csv("data/processed/absa_v1/absa_validation_predictions_v1.csv").fillna("");v2=pd.read_csv(V2_PREDICTIONS).fillna("")
    for frame in [sample,gold,v1,v2,v3]:frame.document_id=frame.document_id.astype(str)
    meta=sample.set_index("document_id");captions=meta.original_text.astype(str).to_dict();languages=meta.primary_language.astype(str).to_dict()
    m1=validation_metrics(gold,v1,captions,languages);m2=validation_metrics(gold,v2,captions,languages);m3=validation_metrics(gold,v3,captions,languages)
    outputs=[root/"candidate_evaluation_v3_v1.json",root/"v1_v2_v3_comparison_v1.json",root/"v2_residual_fp_to_v3_transition_v1.csv",
             root/"v2_tp_to_v3_regression_v1.csv",root/"language_comparison_v3_v1.csv",root/"aspect_family_comparison_v3_v1.csv",root/"development_report_v3_v1.md",root/PREDICTIONS_PATH.name]
    if any(x.exists() for x in outputs):raise FileExistsError("V3 final output exists")
    v3.to_csv(root/PREDICTIONS_PATH.name,index=False,encoding="utf-8-sig");decision=_decision(m3)
    evaluation={"experiment_id":EXPERIMENT_ID,"development_evidence":True,"confirmatory_evidence":False,"decision":decision,
      "qualitative_safeguard_requires_review":True,"metrics":m3,"success_gates":spec["success_gates"],"input_hashes":spec["frozen_input_hashes"]}
    (root/"candidate_evaluation_v3_v1.json").write_text(json.dumps(evaluation,indent=2,ensure_ascii=False,default=float),encoding="utf-8")
    def summary(frame,metrics):
        a=metrics["aspect_detection"];d=metrics["document_diagnostics"];e=metrics["evidence_grounding"]
        return {"aspect_precision":a["precision"],"aspect_recall":a["recall"],"aspect_f1":a["f1"],"tp":a["tp"],"fp":a["fp"],"fn":a["fn"],
          "joint_precision":metrics["joint_aspect_sentiment"]["precision"],"joint_recall":metrics["joint_aspect_sentiment"]["recall"],"joint_f1":metrics["joint_aspect_sentiment"]["f1"],
          "gold_zero_predicted_docs":d["zero_mention_confusion_gold_rows_predicted_columns"]["gold_zero_predicted_mentions"],
          "predicted_multi_aspect_docs":d["predicted_multi_aspect_documents"],"predicted_mentions":len(frame),
          "matched_sentiment_accuracy":metrics["matched_sentiment"]["accuracy"],"sentiment_macro_f1":metrics["matched_sentiment"]["macro_f1"],
          "strict_grounding":e["strict_raw_model_evidence_grounding_rate"],"recoverable_grounding":e["final_recoverable_exact_span_grounding_rate"]}
    comparison={"v1":summary(v1,m1),"v2":summary(v2,m2),"v3":summary(v3,m3)}
    (root/"v1_v2_v3_comparison_v1.json").write_text(json.dumps(comparison,indent=2),encoding="utf-8")
    s2=_statuses(v2,gold);s3=_statuses(v3,gold);residual=pd.read_csv(V2_RESIDUAL).fillna("");residual.document_id=residual.document_id.astype(str)
    transition=[]
    for row in residual.itertuples():
        same=s3[(s3.document_id.eq(row.document_id))&(s3.aspect.eq(row.aspect))&(s3.evaluation_status.eq("FP"))]
        exact_tp=s3[(s3.document_id.eq(row.document_id))&(s3.evidence_text.eq(row.evidence_text))&(s3.evaluation_status.eq("TP"))]
        other=s3[(s3.document_id.eq(row.document_id))&(s3.evaluation_status.eq("FP"))]
        behavior="remapped_to_correct_TP" if len(exact_tp) else ("still_same_FP" if len(same) else ("changed_to_different_FP" if len(other) else "removed"))
        transition.append({"residual_fp_id":row.residual_fp_id,"document_id":row.document_id,"v2_category":row.candidate_fp_category,
                           "v2_aspect":row.aspect,"v2_evidence":row.evidence_text,"v3_behavior":behavior})
    pd.DataFrame(transition).to_csv(root/"v2_residual_fp_to_v3_transition_v1.csv",index=False)
    c2=Counter(zip(v2.document_id,v2.aspect));c3=Counter(zip(v3.document_id,v3.aspect)); regress=[]
    for key,group in gold.sort_values("mention_id").groupby(["document_id","aspect"],sort=False):
        retained2=min(len(group),c2[key]);retained3=min(len(group),c3[key])
        for _,g in group.sort_values("mention_id").iloc[retained3:retained2].iterrows():
            regress.append({"document_id":key[0],"language":languages[key[0]],"raw_caption":captions[key[0]],"gold_aspect":key[1],
              "gold_sentiment":g.sentiment,"gold_evidence":g.evidence_text,"gold_expression_type":g.expression_type,
              "v2_prediction":" || ".join(v2[(v2.document_id.eq(key[0]))&(v2.aspect.eq(key[1]))].evidence_text.astype(str)),
              "v3_prediction":" || ".join(v3[v3.document_id.eq(key[0])].apply(lambda x:f"{x.aspect}: {x.evidence_text}",axis=1))})
    pd.DataFrame(regress,columns=["document_id","language","raw_caption","gold_aspect","gold_sentiment","gold_evidence","gold_expression_type","v2_prediction","v3_prediction"]).to_csv(root/"v2_tp_to_v3_regression_v1.csv",index=False)
    groups={x:{d for d,l in languages.items() if l==x} for x in ["English","Malay","Chinese","Malay/Indonesian uncertain"]};groups["non-English/uncertain"]={d for d,l in languages.items() if l!="English"}
    language=[]
    for name,docs in groups.items():
        language.append({"language":name,"documents":len(docs),**{f"v{i}_{k}":v for i,frame in [(1,v1),(2,v2),(3,v3)] for k,v in _metric_row(frame,gold,docs).items()}})
    pd.DataFrame(language).to_csv(root/"language_comparison_v3_v1.csv",index=False)
    aspect=[]
    for a in sorted(set(gold.aspect)|set(v1.aspect)|set(v2.aspect)|set(v3.aspect)):
        docs=set(sample.document_id); row={"aspect":a,"gold_mentions":int(gold.aspect.eq(a).sum())}
        for label,frame in [("v1",v1),("v2",v2),("v3",v3)]:
            gg=gold[gold.aspect.eq(a)];pp=frame[frame.aspect.eq(a)];gc=Counter(gg.document_id);pc=Counter(pp.document_id);tp=sum((gc&pc).values())
            row.update({f"{label}_predictions":len(pp),f"{label}_fp":len(pp)-tp,f"{label}_precision":tp/len(pp) if len(pp) else 0,f"{label}_recall":tp/len(gg) if len(gg) else 0})
        aspect.append(row)
    pd.DataFrame(aspect).to_csv(root/"aspect_family_comparison_v3_v1.csv",index=False)
    emotional={label:int(((status.aspect=="emotional_experience")&(status.evaluation_status=="FP")).sum()) for label,status in [("v1",_statuses(v1,gold)),("v2",s2),("v3",s3)]}
    report=f"""# ABSA V1 Aspect-Ownership Development Report\n\nDevelopment evidence: **true**. Confirmatory evidence: **false**.\n\nDecision: **{decision}** (qualitative safeguard review required before any pass is final).\n\nV3 aspect precision/recall/F1: {m3['aspect_detection']['precision']:.4f}/{m3['aspect_detection']['recall']:.4f}/{m3['aspect_detection']['f1']:.4f}; TP/FP/FN {m3['aspect_detection']['tp']}/{m3['aspect_detection']['fp']}/{m3['aspect_detection']['fn']}.\n\nEmotional-experience FP V1/V2/V3: {emotional['v1']}/{emotional['v2']}/{emotional['v3']}. Predicted multi-aspect documents gold/V1/V2/V3: 27/43/35/{m3['document_diagnostics']['predicted_multi_aspect_documents']}.\n\nNo V4 was created. This reused 80-document result is not confirmatory.\n"""
    (root/"development_report_v3_v1.md").write_text(report,encoding="utf-8");return root/"development_report_v3_v1.md"
