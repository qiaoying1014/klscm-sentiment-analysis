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
    ABSA_V1_INSTRUCTIONS, AUDIT_ROOT, CACHE_STAGE, CORPUS, DEFAULT_MODEL, GOLD_PATH,
    ONTOLOGY_PATH, PROMPT_VERSION, SAMPLE_PATH, SCHEMA_VERSION, SENTIMENTS,
    canonicalize_evidence, load_ontology, mention_schema, stable_mention_id,
    validation_metrics,
)

EXPERIMENT_ID = "absa_v1_precision_prompt_development_v1"
CANDIDATE_PROMPT_VERSION = "absa_v1_instructions_2_precision"
CANDIDATE_CACHE_STAGE = "absa_v1_mentions_precision_development_v1"
ROOT = Path("data/processed/absa_v1/development") / EXPERIMENT_ID
PROMPT_PATH = ROOT / "absa_v1_instructions_2_precision.txt"
SPEC_PATH = ROOT / "experiment_manifest_v1.json"
REQUEST_PATH = ROOT / "batch" / "absa_v1_precision_development_requests.jsonl"
REQUEST_MANIFEST_PATH = ROOT / "batch" / "request_manifest_v1.json"
SUBMISSION_PATH = ROOT / "batch" / "submission.json"
STATUS_PATH = ROOT / "batch" / "status.json"
RAW_PATH = ROOT / "batch" / "absa_v1_precision_development_raw_results.jsonl"
PREDICTIONS_PATH = ROOT / "candidate_predictions_v1.csv"

HYPOTHESIS = """ABSA v1 has adequate sensitivity to evaluative language but insufficient
local aspect-target attribution.

The model frequently detects real affect but:
1. assigns it to an adjacent ontology family;
2. infers additional aspects not locally entailed;
3. converts contextual/background affect into an aspect;
4. duplicates a single evaluation;
5. transfers overall document sentiment to neutrally mentioned entities.

A prompt-level local entailment and attribution policy should substantially
reduce false-positive aspects without removing legitimate explicit or implicit
evaluation."""

SUCCESS_GATES = {
    "aspect_precision_minimum": 0.55,
    "aspect_recall_minimum": 0.75,
    "joint_aspect_sentiment_precision_strictly_greater_than": 0.3302752,
    "joint_aspect_sentiment_f1_minimum": 0.4528302,
    "gold_zero_predicted_mentions_documents_maximum": 7,
    "qualitative_safeguard": "No obvious collapse of legitimate implicit or multilingual evaluations.",
}

PRECISION_POLICY = r"""

PRECISION-FOCUSED LOCAL ATTRIBUTION POLICY

An ontology-related concept being present is not sufficient for extraction. Extract an aspect only when
local evidence supports a qualifying evaluation or experience attributable to that specific aspect. Follow
this order: (1) identify evaluative evidence; (2) identify what it is actually about; (3) map that target to
the best-supported ontology family; (4) assign sentiment toward that target; (5) extract only when all four
are supported. Do not start by listing ontology concepts that merely appear in the caption.

Every extracted aspect must be locally entailed by its evidence span. The evidence must support both that a
qualifying evaluation exists and that it targets the predicted family. Emotion alone does not evaluate every
nearby entity, and an aspect mentioned elsewhere is not automatically the target of local affect. Choose the
family actually evaluated, not an adjacent family: distinguish community encouragement from achieved race
performance, general emotion from its object, overall worth from monetary value, physical strain from
performance, training effort from bodily condition, and atmosphere from personal emotion.

Distinguish experienced or observed evaluation from a wish, hope, encouragement, plan, intention, strategy,
or prospective outcome. "Good luck" or "finish strong" does not prove that a performance occurred. A plan,
preference, or strategy may still qualify when the strategy itself is evaluated; do not infer its eventual
result. Do not infer hidden causes or related aspects merely because they could explain an experience.
Struggling does not by itself establish hydration, weather, organization, pacing, or injury.

Extract the smallest sufficient set of mentions representing distinct evaluations. Each additional mention
requires distinct evidence or a clearly distinct evaluated target. Do not emit multiple mentions for the same
evaluation of the same aspect unless separate evaluative propositions justify them. Overall document sentiment
must not be copied onto neutral references such as race packs, sponsors, timing, photographs, locations, shoes,
transport, distances, training, or facilities. Bind polarity locally to its evaluated target.

Pure participation, categories, distances, dates, locations, finish times/results without evaluation, neutral
event descriptions, sponsor/product references, training logs, registration/expo facts, photography references,
transport facts, and weather reported only as context normally yield no mention. This is semantic policy, not
a keyword blacklist: the same subject qualifies when framed as achievement, disappointment, difficulty,
preference, improvement, complaint, praise, satisfaction, or affective reaction.

Preserve legitimate implicit evaluation. Do not require explicit adjectives. Achievement, difficulty,
impairment, insufficient preparation, comparison with expectation, preference, struggle, apprehension,
gratitude, encouragement, celebration, pride, reflective disappointment, and contextually evaluative DNF may
qualify. Apply identical semantic policy to English, Malay, Chinese, Indonesian, mixed/code-switched language,
hashtags, emoji, abbreviations, particles, comparative wording, and colloquial expressions. Sparse text is not
automatically non-evaluative, but generic happy/event hashtags do not establish an aspect without a target.
Community praise or encouragement may be evaluative; map its actual target under the frozen ontology rather
than suppressing it or treating a wished-for result as achieved performance.

Keep neutral available only for an existing-policy qualifying aspect-level judgment, selection, or experience
whose polarity is genuinely neutral; never create neutral from a mere fact. Use mixed only for opposing
sentiment toward the same aspect, not different sentiments across different aspects.

Contrastive principles (synthetic examples):
- "Finished the half marathon in 2:15." -> no evaluative mention. "Finally improved my PB!" -> positive
  race_performance.
- "Good luck; finish strong!" -> do not infer an achieved result. "Great job breaking your PB!" -> positive
  race_performance.
- "Fantastic weekend. Collected the race pack Friday." -> overall positivity does not make race_pack_expo
  positive.
- "I struggled badly near the end." -> represent only the directly supported evaluation; do not invent its
  hydration, weather, operational, pacing, or injury cause.
- "Beautiful route; I really loved the course." -> normally one route_course evaluation.
- "My legs completely gave up after 30 km." -> legitimate negative physical_experience.

Evidence remains the minimum sufficient exact original-caption span. Never paraphrase it or loosen any schema,
offset, multilingual, ontology, or output requirement above.
"""

CANDIDATE_INSTRUCTIONS = ABSA_V1_INSTRUCTIONS + PRECISION_POLICY

BASELINE = {
    "aspect_precision": 0.3853211, "aspect_recall": 0.84, "aspect_f1": 0.5283019,
    "tp": 84, "fp": 134, "fn": 16, "predicted_mentions": 218,
    "joint_precision": 0.3302752, "joint_recall": 0.72, "joint_f1": 0.4528302,
    "gold_zero_predicted_mentions": 11, "predicted_multi_aspect_documents": 43,
    "matched_sentiment_accuracy": 0.8571429, "strict_grounding_rate": 0.93578,
    "recoverable_grounding_rate": 0.94037,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prompt_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _frozen_inputs() -> dict[str, str]:
    marker = AUDIT_ROOT / "absa_v1_audit_gold_v1.frozen.json"
    baseline_predictions = Path("data/processed/absa_v1/absa_validation_predictions_v1.csv")
    baseline_evaluation = Path("data/processed/absa_v1/absa_validation_evaluation_v1.json")
    for path in [SAMPLE_PATH, GOLD_PATH, marker, ONTOLOGY_PATH, CORPUS, baseline_predictions, baseline_evaluation]:
        if not path.exists():
            raise FileNotFoundError(f"Required frozen input is missing: {path}")
    return {"sample_sha256": _sha(SAMPLE_PATH), "gold_sha256": _sha(GOLD_PATH),
            "gold_marker_sha256": _sha(marker), "ontology_sha256": _sha(ONTOLOGY_PATH),
            "corpus_sha256": _sha(CORPUS), "baseline_predictions_sha256": _sha(baseline_predictions),
            "baseline_evaluation_sha256": _sha(baseline_evaluation)}


def _source() -> pd.DataFrame:
    sample = pd.read_csv(SAMPLE_PATH)
    corpus = pd.read_csv(CORPUS, low_memory=False)
    sample_ids = sample.document_id.astype(str).tolist()
    if len(sample_ids) != 80 or len(set(sample_ids)) != 80:
        raise ValueError("Development experiment requires exactly 80 frozen document IDs")
    source = corpus.assign(document_id=corpus.document_id.astype(str))
    source = source[source.document_id.isin(sample_ids)].set_index("document_id").loc[sample_ids].reset_index()
    if source.document_id.tolist() != sample_ids:
        raise ValueError("Candidate population does not exactly reproduce frozen sample order")
    return source


def _estimate_cost(source: pd.DataFrame) -> dict[str, Any]:
    ontology_chars = len(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    input_tokens = sum(math.ceil((len(str(x)) + len(CANDIDATE_INSTRUCTIONS) + ontology_chars) / 4)
                       for x in source.original_text)
    output_tokens = len(source) * 260
    synchronous = (input_tokens * 1.25 + output_tokens * 10.0) / 1_000_000
    return {"estimated_input_tokens": input_tokens, "estimated_output_tokens": output_tokens,
            "estimated_synchronous_cost_usd": round(synchronous, 4),
            "estimated_batch_cost_usd": round(synchronous * 0.5, 4),
            "pricing_assumptions_usd_per_million": {"input": 1.25, "output": 10.0,
                                                     "batch_discount": 0.5}}


def create_experiment(root: Path = ROOT) -> dict[str, Any]:
    global ROOT, PROMPT_PATH, SPEC_PATH, REQUEST_PATH, REQUEST_MANIFEST_PATH
    # Alternate roots are supported for deterministic tests without changing production constants.
    prompt_path = root / PROMPT_PATH.name; spec_path = root / SPEC_PATH.name
    request_path = root / "batch" / REQUEST_PATH.name
    request_manifest_path = root / "batch" / REQUEST_MANIFEST_PATH.name
    inputs = _frozen_inputs(); source = _source(); ontology = load_ontology()
    prompt_hash = _prompt_sha(CANDIDATE_INSTRUCTIONS); baseline_hash = _prompt_sha(ABSA_V1_INSTRUCTIONS)
    if prompt_hash == baseline_hash:
        raise ValueError("Candidate prompt must differ from baseline")
    spec = {"experiment_id": EXPERIMENT_ID, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "hypothesis": HYPOTHESIS, "baseline_prompt": PROMPT_VERSION,
            "candidate_prompt": CANDIDATE_PROMPT_VERSION, "baseline_prompt_sha256": baseline_hash,
            "candidate_prompt_sha256": prompt_hash, "baseline_cache_stage": CACHE_STAGE,
            "candidate_cache_stage": CANDIDATE_CACHE_STAGE, "model": DEFAULT_MODEL,
            "schema": SCHEMA_VERSION, "development_evidence": True, "confirmatory_evidence": False,
            "model_unchanged": True, "ontology_unchanged": True, "schema_unchanged": True,
            "gold_unchanged": True, "validation_documents_unchanged": True,
            "matching_rules_unchanged": True, "preprocessing_unchanged": True,
            "only_prompt_policy_changed": True, "success_gates": SUCCESS_GATES,
            "decision_rule_frozen_before_candidate_evaluation": True, "frozen_input_hashes": inputs,
            "full_corpus_prohibited": True, "candidate_count": 1}
    if spec_path.exists():
        old = json.loads(spec_path.read_text(encoding="utf-8"))
        immutable = ["hypothesis", "baseline_prompt", "candidate_prompt", "candidate_prompt_sha256",
                     "success_gates", "frozen_input_hashes"]
        if any(old.get(k) != spec.get(k) for k in immutable):
            raise FileExistsError("Frozen experiment specification differs; refusing overwrite")
    else:
        root.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(CANDIDATE_INSTRUCTIONS, encoding="utf-8")
        spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    schema = mention_schema([a["id"] for a in ontology["aspects"]])
    lines = []
    for row in source.itertuples(index=False):
        context = {"document_id": str(row.document_id), "original_text": str(row.original_text),
                   "final_topic_name": str(row.final_topic_name),
                   "final_topic_group": str(row.final_topic_group), "topic_context_supplied": True,
                   "ontology": [a["id"] for a in ontology["aspects"]]}
        body = {"model": DEFAULT_MODEL, "instructions": CANDIDATE_INSTRUCTIONS,
                "input": json.dumps(context, ensure_ascii=False),
                "text": {"format": {"type": "json_schema", "name": SCHEMA_VERSION,
                                      "strict": True, "schema": schema}}}
        lines.append(json.dumps({"custom_id": f"absa1p2-{row.document_id}", "method": "POST",
                                 "url": "/v1/responses", "body": body}, ensure_ascii=False))
    content = "\n".join(lines) + "\n"; request_path.parent.mkdir(parents=True, exist_ok=True)
    if request_path.exists() and request_path.read_text(encoding="utf-8") != content:
        raise FileExistsError("Existing candidate request package differs; refusing overwrite")
    request_path.write_text(content, encoding="utf-8")
    costs = _estimate_cost(source)
    request_manifest = {"experiment_id": EXPERIMENT_ID, "document_count": 80, "request_count": 80,
        "document_ids_sha256": hashlib.sha256("\n".join(source.document_id).encode()).hexdigest(),
        "request_sha256": _sha(request_path), "model": DEFAULT_MODEL,
        "prompt_version": CANDIDATE_PROMPT_VERSION, "prompt_sha256": prompt_hash,
        "schema_version": SCHEMA_VERSION, "cache_stage": CANDIDATE_CACHE_STAGE,
        "submitted": False, "development_evidence": True, "confirmatory_evidence": False,
        "full_corpus": False, "cost_estimate": costs}
    request_manifest_path.write_text(json.dumps(request_manifest, indent=2), encoding="utf-8")
    return {"experiment_manifest": str(spec_path), "prompt_artifact": str(prompt_path),
            "request_path": str(request_path), **request_manifest}


def _validate_package(root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any], Path]:
    spec_path=root/SPEC_PATH.name; request_path=root/"batch"/REQUEST_PATH.name
    manifest_path=root/"batch"/REQUEST_MANIFEST_PATH.name
    spec=json.loads(spec_path.read_text(encoding="utf-8")); manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    if spec["success_gates"] != SUCCESS_GATES or not spec["decision_rule_frozen_before_candidate_evaluation"]:
        raise RuntimeError("Frozen success criteria changed")
    prompt_path=root/PROMPT_PATH.name
    if _sha(prompt_path) != spec["candidate_prompt_sha256"]:
        raise RuntimeError("Candidate prompt artifact hash changed")
    if spec["frozen_input_hashes"] != _frozen_inputs(): raise RuntimeError("A frozen input hash changed")
    if manifest["request_sha256"] != _sha(request_path) or manifest["request_count"] != 80:
        raise RuntimeError("Candidate request package failed integrity validation")
    lines=[json.loads(x) for x in request_path.read_text(encoding="utf-8").splitlines() if x]
    ids={x["custom_id"].removeprefix("absa1p2-") for x in lines}
    if len(lines) != 80 or ids != set(_source().document_id): raise RuntimeError("Candidate population changed")
    return spec, manifest, request_path


def submit_experiment(root: Path = ROOT) -> dict[str, Any]:
    from openai import OpenAI
    _, manifest, request_path = _validate_package(root)
    submission_path=root/"batch"/SUBMISSION_PATH.name
    if submission_path.exists(): raise FileExistsError("Candidate Batch was already submitted")
    client=OpenAI()
    with request_path.open("rb") as handle: uploaded=client.files.create(file=handle,purpose="batch")
    batch=client.batches.create(input_file_id=uploaded.id,endpoint="/v1/responses",completion_window="24h",
                                metadata={"stage":EXPERIMENT_ID,"prompt":CANDIDATE_PROMPT_VERSION})
    record={"batch_id":batch.id,"input_file_id":uploaded.id,"status":batch.status,
            "submitted_at_utc":datetime.now(timezone.utc).isoformat(),"request_count":manifest["request_count"]}
    submission_path.write_text(json.dumps(record,indent=2),encoding="utf-8"); return record


def experiment_status(root: Path = ROOT) -> dict[str, Any]:
    from openai import OpenAI
    submission=json.loads((root/"batch"/SUBMISSION_PATH.name).read_text(encoding="utf-8"))
    record=OpenAI().batches.retrieve(submission["batch_id"]).model_dump()
    (root/"batch"/STATUS_PATH.name).write_text(json.dumps(record,indent=2),encoding="utf-8"); return record


def import_experiment(root: Path = ROOT) -> Path:
    from openai import OpenAI
    status=json.loads((root/"batch"/STATUS_PATH.name).read_text(encoding="utf-8"))
    if status.get("status") != "completed" or not status.get("output_file_id"):
        raise RuntimeError("Candidate Batch must be completed before import")
    path=root/"batch"/RAW_PATH.name
    if path.exists(): raise FileExistsError("Candidate raw results already exist")
    path.write_bytes(OpenAI().files.content(status["output_file_id"]).content); return path


def _candidate_predictions(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_path=root/"batch"/RAW_PATH.name; corpus=pd.read_csv(CORPUS,low_memory=False)
    corpus.document_id=corpus.document_id.astype(str); corpus=corpus.set_index("document_id")
    allowed=set(_source().document_id); aspects=[a["id"] for a in load_ontology()["aspects"]]
    rows=[]; processed=set(); raw_exact=0; recoverable=0
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        item=json.loads(line); document_id=item["custom_id"].removeprefix("absa1p2-")
        if document_id not in allowed or document_id in processed: raise ValueError("Unknown or duplicate candidate document")
        processed.add(document_id); body=item.get("response",{}).get("body",{})
        texts=[c.get("text","") for o in body.get("output",[]) for c in o.get("content",[]) if c.get("type")=="output_text"]
        if len(texts)!=1: raise ValueError("Candidate result lacks one output_text")
        mentions=json.loads(texts[0]).get("mentions"); caption=str(corpus.loc[document_id].original_text)
        if not isinstance(mentions,list): raise ValueError("mentions must be an array")
        for index,m in enumerate(mentions):
            if m.get("aspect") not in aspects or m.get("sentiment") not in SENTIMENTS: raise ValueError("Invalid candidate output")
            rec=canonicalize_evidence(caption,m.get("evidence_text",""),m.get("evidence_start"),m.get("evidence_end"))
            raw_exact += int(rec["raw_exact_match"]); recoverable += int(rec["raw_exact_match"] or rec["evidence_repaired"])
            evidence=rec["final_exact_evidence_text"]
            rows.append({"mention_id":stable_mention_id(document_id,index,m["aspect"],evidence or rec["original_model_evidence_text"]),
                "document_id":document_id,**m,"evidence_text":evidence,"evidence_start":rec["final_evidence_start"],
                "evidence_end":rec["final_evidence_end"],"raw_exact_match":rec["raw_exact_match"],
                "evidence_repaired":rec["evidence_repaired"],"original_model_evidence_text":rec["original_model_evidence_text"],
                "prompt_version":CANDIDATE_PROMPT_VERSION,"schema_version":SCHEMA_VERSION,"model":body.get("model",DEFAULT_MODEL)})
    if processed != allowed: raise ValueError("Candidate results do not cover exactly 80 frozen documents")
    columns=["mention_id","document_id","aspect","target","sentiment","confidence","evidence_text",
             "expression_type","language","english_gloss","contributing_hashtags","contributing_emoji",
             "emerging_aspect_name","analysis_notes","evidence_start","evidence_end","raw_exact_match",
             "evidence_repaired","original_model_evidence_text","prompt_version","schema_version","model"]
    return pd.DataFrame(rows,columns=columns),{"total":len(rows),"raw_exact":raw_exact,"recoverable":recoverable}


def _decision(metrics: dict[str, Any]) -> str:
    a=metrics["aspect_detection"]; j=metrics["joint_aspect_sentiment"]
    zero=metrics["document_diagnostics"]["zero_mention_confusion_gold_rows_predicted_columns"]["gold_zero_predicted_mentions"]
    if a["precision"]>=.55 and a["recall"]<.75: return "TOO_CONSERVATIVE"
    if (a["precision"]>=.55 and a["recall"]>=.75 and j["precision"]>.3302752 and j["f1"]>=.4528302 and zero<=7):
        return "PASS_FOR_CONFIRMATORY_TESTING_PENDING_QUALITATIVE_SAFEGUARD"
    if a["precision"]>BASELINE["aspect_precision"] and a["precision"]<.55:
        return "INSUFFICIENT_PRECISION_IMPROVEMENT"
    return "FAILED"


def finalize_experiment(root: Path = ROOT) -> Path:
    spec,_,_=_validate_package(root); predictions,grounding=_candidate_predictions(root)
    predictions_path=root/PREDICTIONS_PATH.name
    outputs=[predictions_path,root/"candidate_evaluation_v1.json",root/"baseline_vs_candidate_v1.json",
             root/"fp_transition_analysis_v1.csv",root/"recall_regression_analysis_v1.csv",
             root/"reviewed_fp_category_transition_v1.csv", root/"language_comparison_v1.csv",
             root/"aspect_family_comparison_v1.csv",root/"development_report_v1.md"]
    if any(p.exists() for p in outputs): raise FileExistsError("Development final outputs already exist; refusing overwrite")
    sample=pd.read_csv(SAMPLE_PATH); gold=pd.read_csv(GOLD_PATH).fillna("")
    indexed=sample.assign(document_id=sample.document_id.astype(str)).set_index("document_id")
    captions=indexed.original_text.astype(str).to_dict(); languages=indexed.primary_language.astype(str).to_dict()
    metrics=validation_metrics(gold,predictions,captions,languages); decision=_decision(metrics)
    predictions.to_csv(predictions_path,index=False,encoding="utf-8-sig")
    evaluation={"experiment_id":EXPERIMENT_ID,"development_evidence":True,"confirmatory_evidence":False,
                "metrics":metrics,"decision":decision,"qualitative_safeguard_requires_researcher_confirmation":True,
                "success_gates":spec["success_gates"],"input_hashes":spec["frozen_input_hashes"]}
    (root/"candidate_evaluation_v1.json").write_text(json.dumps(evaluation,indent=2,ensure_ascii=False,default=float),encoding="utf-8")
    a=metrics["aspect_detection"]; j=metrics["joint_aspect_sentiment"]; d=metrics["document_diagnostics"]
    candidate={"aspect_precision":a["precision"],"aspect_recall":a["recall"],"aspect_f1":a["f1"],"tp":a["tp"],"fp":a["fp"],"fn":a["fn"],
      "predicted_mentions":len(predictions),"joint_precision":j["precision"],"joint_recall":j["recall"],"joint_f1":j["f1"],
      "gold_zero_predicted_mentions":d["zero_mention_confusion_gold_rows_predicted_columns"]["gold_zero_predicted_mentions"],
      "predicted_multi_aspect_documents":d["predicted_multi_aspect_documents"],"matched_sentiment_accuracy":metrics["matched_sentiment"]["accuracy"],
      "strict_grounding_rate":grounding["raw_exact"]/grounding["total"] if grounding["total"] else 1,
      "recoverable_grounding_rate":grounding["recoverable"]/grounding["total"] if grounding["total"] else 1}
    comparison={k:{"baseline":BASELINE[k],"candidate":candidate[k],"delta":candidate[k]-BASELINE[k]} for k in BASELINE}
    (root/"baseline_vs_candidate_v1.json").write_text(json.dumps(comparison,indent=2),encoding="utf-8")
    baseline=pd.read_csv("data/processed/absa_v1/absa_validation_predictions_v1.csv").fillna("")
    baseline.document_id=baseline.document_id.astype(str); predictions.document_id=predictions.document_id.astype(str)
    gold.document_id=gold.document_id.astype(str)
    bkeys=Counter(zip(baseline.document_id,baseline.aspect)); ckeys=Counter(zip(predictions.document_id,predictions.aspect))
    gkeys=Counter(zip(gold.document_id,gold.aspect))
    transition=[]
    fp_rows=[]
    for (doc,aspect),group in baseline.sort_values("mention_id").groupby(["document_id","aspect"],sort=False):
        fp_rows.extend(group.iloc[min(len(group),gkeys[(doc,aspect)]):].to_dict("records"))
    if len(fp_rows)!=134: raise ValueError("Baseline FP reconstruction must reproduce 134")
    candidate_remaining=Counter({key:max(0,count-gkeys[key]) for key,count in ckeys.items()})
    candidate_doc_aspects={doc:set(group.aspect) for doc,group in predictions.groupby("document_id")}
    for row in fp_rows:
        doc,aspect=str(row["document_id"]),row["aspect"]
        if candidate_remaining[(doc,aspect)]>0:
            status="still_present"; candidate_remaining[(doc,aspect)]-=1
        elif candidate_doc_aspects.get(doc,set())-{aspect}:
            status="changed_to_another_aspect"
        else: status="removed"
        transition.append({"mention_id":row.get("mention_id",""),"document_id":doc,
                           "baseline_aspect":aspect,"baseline_sentiment":row.get("sentiment",""),"status":status})
    transition_df=pd.DataFrame(transition); transition_df.to_csv(root/"fp_transition_analysis_v1.csv",index=False)
    review_path=Path("data/processed/absa_v1/development/absa_v1_false_positive_diagnostic_v1/absa_v1_fp_manual_review_sample_v1.csv")
    review=pd.read_csv(review_path).fillna("")
    reviewed_transition=review[["mention_id","review_row_id","fp_error_category"]].merge(transition_df,on="mention_id",how="left")
    reviewed_transition.to_csv(root/"reviewed_fp_category_transition_v1.csv",index=False)
    # Exact baseline-TP losses under the unchanged stable matching rule.
    regression=[]
    for row in gold.itertuples():
        doc=str(row.document_id); aspect=row.aspect
        if bkeys[(doc,aspect)]>0 and ckeys[(doc,aspect)]==0:
            regression.append({"document_id":doc,"language":languages.get(doc,""),"gold_aspect":aspect,
                "gold_sentiment":row.sentiment,"gold_evidence":row.evidence_text,
                "baseline_prediction":aspect,"candidate_prediction":""})
    pd.DataFrame(regression,columns=["document_id","language","gold_aspect","gold_sentiment","gold_evidence","baseline_prediction","candidate_prediction"]).to_csv(root/"recall_regression_analysis_v1.csv",index=False)
    lang=[]
    for name,v in metrics["document_diagnostics"]["multilingual_subgroups"].items():
        lang.append({"language":name,"documents":v["documents"],"metrics_reported":v["metrics_reported"],
                     "aspect_precision":v.get("aspect_detection",{}).get("precision"),"aspect_recall":v.get("aspect_detection",{}).get("recall")})
    pd.DataFrame(lang).to_csv(root/"language_comparison_v1.csv",index=False)
    aspect_rows=[]
    for aspect in sorted(set(gold.aspect)|set(baseline.aspect)|set(predictions.aspect)):
        def scores(frame):
            gg=gold[gold.aspect.eq(aspect)]; pp=frame[frame.aspect.eq(aspect)]
            gm=Counter(gg.document_id); pm=Counter(pp.document_id); tp=sum((gm&pm).values())
            precision=tp/len(pp) if len(pp) else 0.0; recall=tp/len(gg) if len(gg) else 0.0
            return precision,recall,len(pp)
        bp,br,bn=scores(baseline); cp,cr,cn=scores(predictions)
        aspect_rows.append({"aspect":aspect,"gold_mentions":int(gold.aspect.eq(aspect).sum()),
                            "baseline_precision":bp,"baseline_recall":br,"baseline_predictions":bn,
                            "candidate_precision":cp,"candidate_recall":cr,"candidate_predictions":cn})
    pd.DataFrame(aspect_rows).to_csv(root/"aspect_family_comparison_v1.csv",index=False)
    report=f"""# ABSA V1 Precision Prompt Development Report\n\nDevelopment evidence: **true**. Confirmatory evidence: **false**.\n\nDecision: **{decision}**\n\nCandidate prompt: `{CANDIDATE_PROMPT_VERSION}`; model/schema/ontology/gold/sample/matching/preprocessing unchanged.\n\nAspect precision/recall/F1: {a['precision']:.4f}/{a['recall']:.4f}/{a['f1']:.4f}. TP/FP/FN: {a['tp']}/{a['fp']}/{a['fn']}.\nJoint precision/recall/F1: {j['precision']:.4f}/{j['recall']:.4f}/{j['f1']:.4f}.\n\nThis reused development sample is not confirmatory. Do not automatically create instructions_3.\n"""
    (root/"development_report_v1.md").write_text(report,encoding="utf-8")
    return root/"development_report_v1.md"
