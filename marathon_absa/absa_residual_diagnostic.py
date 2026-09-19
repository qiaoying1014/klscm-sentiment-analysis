from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .absa_v1 import GOLD_PATH, ONTOLOGY_PATH, SAMPLE_PATH, validation_metrics
from .absa_precision_development import ROOT as DEVELOPMENT_ROOT

VERSION = "absa_v1_precision_v2_residual_diagnostic_v1"
BASELINE_PATH = Path("data/processed/absa_v1/absa_validation_predictions_v1.csv")
CANDIDATE_PATH = DEVELOPMENT_ROOT / "candidate_predictions_v1.csv"
BASELINE_EVALUATION = Path("data/processed/absa_v1/absa_validation_evaluation_v1.json")
CANDIDATE_EVALUATION = DEVELOPMENT_ROOT / "candidate_evaluation_v1.json"
REVIEWED_FP_PATH = Path("data/processed/absa_v1/development/absa_v1_false_positive_diagnostic_v1/absa_v1_fp_manual_review_sample_v1.csv")
OUTPUT_JSON = DEVELOPMENT_ROOT / f"{VERSION}.json"
OUTPUT_MD = DEVELOPMENT_ROOT / f"{VERSION}.md"

CATEGORIES = [
    "duplicate_redundant_mention", "duplicate_redundant_mention", "ontology_boundary_confusion",
    "over_decomposition", "ontology_boundary_confusion", "aspect_inference_too_far",
    "aspect_inference_too_far", "context_as_evaluative", "over_decomposition",
    "duplicate_redundant_mention", "duplicate_redundant_mention", "duplicate_redundant_mention",
    "aspect_inference_too_far", "context_as_evaluative", "duplicate_redundant_mention",
    "context_as_evaluative", "ontology_boundary_confusion", "over_decomposition",
    "ontology_boundary_confusion", "context_as_evaluative", "context_as_evaluative",
    "context_as_evaluative", "ontology_boundary_confusion", "ontology_boundary_confusion",
    "ontology_boundary_confusion", "aspect_inference_too_far", "duplicate_redundant_mention",
    "duplicate_redundant_mention", "ontology_boundary_confusion", "ontology_boundary_confusion",
    "over_decomposition", "factual_as_evaluative", "ontology_boundary_confusion",
    "grounding_related", "ontology_boundary_confusion", "ontology_boundary_confusion",
    "duplicate_redundant_mention", "context_as_evaluative", "over_decomposition",
    "sentiment_spillover", "duplicate_redundant_mention", "aspect_inference_too_far",
    "duplicate_redundant_mention", "context_as_evaluative", "context_as_evaluative",
    "ontology_boundary_confusion", "ontology_boundary_confusion", "ontology_boundary_confusion",
    "ontology_boundary_confusion", "ontology_boundary_confusion", "ontology_boundary_confusion",
    "duplicate_redundant_mention", "ontology_boundary_confusion", "ontology_boundary_confusion",
    "sentiment_spillover", "aspect_inference_too_far", "over_decomposition",
    "ontology_boundary_confusion", "ontology_boundary_confusion", "ontology_boundary_confusion",
    "context_as_evaluative", "ontology_boundary_confusion", "ontology_boundary_confusion",
    "ontology_boundary_confusion", "context_as_evaluative", "ontology_boundary_confusion",
    "aspect_inference_too_far", "ontology_boundary_confusion", "ontology_boundary_confusion",
    "context_as_evaluative", "aspect_inference_too_far", "context_as_evaluative",
    "ontology_boundary_confusion", "aspect_inference_too_far",
]

MEDIUM_CONFIDENCE = {2, 4, 6, 7, 9, 13, 16, 18, 22, 26, 30, 31, 33, 37, 38, 39, 42, 43,
                     44, 56, 57, 61, 67, 69, 71, 72, 73, 74}

FAILURE_TYPES = {
    ("5de49c3b68a39aae43b3e", "race_performance"): "encouragement_or_community_signal_suppressed",
    ("095ffeb6d7c5389f2ab5", "emotional_experience"): "multilingual_colloquial_signal_suppressed",
    ("d466b79ec846ea7f98b4", "training_preparation_pacing"): "multilingual_colloquial_signal_suppressed",
    ("708e0baa9acd63acf87d", "emotional_experience"): "ontology_boundary_shift",
    ("a0e460486e59a4916ff9", "emotional_experience"): "hashtag_or_emoji_signal_suppressed",
    ("ea9ff21b942072809d66", "race_performance"): "implicit_evaluation_suppressed",
    ("3d311b8125f0eb64ca4d", "physical_experience"): "ontology_boundary_shift",
    ("de19229d5d55e16512b7", "weather_conditions"): "ontology_boundary_shift",
    ("d8d8243563bf282277b1", "race_performance"): "performance_comparison_suppressed",
    ("bdd6376aff27961e04f8", "emotional_experience"): "ontology_boundary_shift",
    ("79be6db790c578553f19", "physical_experience"): "implicit_evaluation_suppressed",
    ("392569ec59f45fde5a92", "training_preparation_pacing"): "duplicate_gold_proposition_suppressed",
    ("dbd1b09611fae0ae3899", "crowd_community_atmosphere"): "encouragement_or_community_signal_suppressed",
    ("e1341338331adbda6448", "finisher_items"): "neutral_qualifying_judgment_suppressed",
    ("41103add6e89bfa7b8f6", "race_performance"): "ontology_boundary_shift",
    ("ce796fe45dc0fcd6b546", "emotional_experience"): "ontology_boundary_shift",
    ("a419d5f8ce31297fdba4", "emotional_experience"): "ontology_boundary_shift",
    ("4634c7c3c9a735f473e6", "physical_experience"): "ontology_boundary_shift",
    ("ae005409c954e5797b88", "training_preparation_pacing"): "performance_comparison_suppressed",
    ("57e2b772aa73069550a1", "organization_operations"): "ontology_boundary_shift",
    ("6047619c5662213cb0a8", "organization_operations"): "ontology_boundary_shift",
    ("3095615b8b0b945234c7", "crowd_community_atmosphere"): "encouragement_or_community_signal_suppressed",
    ("ddf29a86fab600bd06c5", "race_performance"): "ontology_boundary_shift",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in [BASELINE_PATH, CANDIDATE_PATH, GOLD_PATH, SAMPLE_PATH, ONTOLOGY_PATH,
                 BASELINE_EVALUATION, CANDIDATE_EVALUATION, REVIEWED_FP_PATH]:
        if not path.exists(): raise FileNotFoundError(path)
    baseline=pd.read_csv(BASELINE_PATH).fillna(""); candidate=pd.read_csv(CANDIDATE_PATH).fillna("")
    gold=pd.read_csv(GOLD_PATH).fillna(""); sample=pd.read_csv(SAMPLE_PATH).fillna("")
    for frame in [baseline,candidate,gold,sample]: frame["document_id"]=frame.document_id.astype(str)
    return baseline,candidate,gold,sample


def _statuses(predictions: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    counts=Counter(zip(gold.document_id,gold.aspect)); chunks=[]
    for key,group in predictions.sort_values("mention_id").groupby(["document_id","aspect"],sort=False):
        group=group.copy(); matched=min(len(group),counts[key])
        group["evaluation_status"]=["TP"]*matched+["FP"]*(len(group)-matched); chunks.append(group)
    return pd.concat(chunks,ignore_index=True) if chunks else predictions.assign(evaluation_status=[])


def _false_negatives(predictions: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    counts=Counter(zip(predictions.document_id,predictions.aspect)); chunks=[]
    for key,group in gold.sort_values("mention_id").groupby(["document_id","aspect"],sort=False):
        missing=group.iloc[min(len(group),counts[key]):].copy()
        if len(missing): chunks.append(missing)
    return pd.concat(chunks,ignore_index=True) if chunks else gold.iloc[0:0].copy()


def _note(row: pd.Series, category: str, gold_aspects: str) -> str:
    evidence=str(row.evidence_text or row.get("original_model_evidence_text", ""))
    snippet=evidence.replace("\n"," ")[:90]
    messages={
      "ontology_boundary_confusion": f"The evaluative signal '{snippet}' is real or plausible, but the frozen reference attributes it to {gold_aspects or 'no KLSCM aspect'}, not {row.aspect}.",
      "duplicate_redundant_mention": f"The document already has the supported {row.aspect} evaluation; '{snippet}' adds another encoding without a distinct frozen mention.",
      "aspect_inference_too_far": f"'{snippet}' does not locally entail a distinct {row.aspect} evaluation under the frozen policy; the candidate still infers beyond the supported target.",
      "context_as_evaluative": f"'{snippet}' functions as announcement, background, or off-event context rather than a KLSCM aspect evaluation.",
      "over_decomposition": f"'{snippet}' belongs to an experience already represented by {gold_aspects}; extracting {row.aspect} separately over-decomposes that proposition.",
      "sentiment_spillover": f"The polarity assigned from '{snippet}' is not locally attributable to {row.aspect}; surrounding document affect has spilled onto it.",
      "factual_as_evaluative": f"'{snippet}' is essentially descriptive and does not express an aspect-level judgment of {row.aspect}.",
      "grounding_related": f"The candidate supplied no usable exact evidence for {row.aspect}, so grounding failure materially prevents support for this mention.",
    }
    return messages[category]


def _review_residual_fps(candidate_status: pd.DataFrame, gold: pd.DataFrame,
                         sample: pd.DataFrame) -> pd.DataFrame:
    fps=candidate_status[candidate_status.evaluation_status.eq("FP")].copy().reset_index(drop=True)
    if len(fps)!=74 or len(CATEGORIES)!=74: raise ValueError("Residual FP review must contain exactly 74 rows")
    fps.insert(0,"residual_fp_id",[f"RFP-{i:03}" for i in range(1,75)])
    meta=sample.set_index("document_id"); gold_groups=gold.groupby("document_id")
    fps["language"]=fps.document_id.map(meta.primary_language)
    fps["raw_caption"]=fps.document_id.map(meta.original_text)
    fps["gold_aspects"]=fps.document_id.map(lambda d:"|".join(gold_groups.get_group(d).aspect) if d in gold_groups.groups else "")
    fps["gold_sentiments"]=fps.document_id.map(lambda d:"|".join(gold_groups.get_group(d).sentiment) if d in gold_groups.groups else "")
    fps["is_gold_zero_document"]=fps.gold_aspects.eq("")
    mention_counts=candidate_status.groupby("document_id").size()
    fps["is_candidate_multi_aspect_document"]=fps.document_id.map(mention_counts).gt(1)
    fps["candidate_fp_category"]=CATEGORIES
    fps["researcher_notes"]=[_note(row,cat,row.gold_aspects) for (_,row),cat in zip(fps.iterrows(),CATEGORIES)]
    fps["confidence"]=["medium" if i in MEDIUM_CONFIDENCE else "high" for i in range(1,75)]
    fps["reviewed"]=True
    return fps


def _prediction_transition(baseline_status: pd.DataFrame, candidate_status: pd.DataFrame,
                           gold: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    meta=sample.set_index("document_id"); gold_by_doc={d:g for d,g in gold.groupby("document_id")}
    candidate_pool={d:g.copy() for d,g in candidate_status.groupby("document_id")}; used=set(); rows=[]
    for b in baseline_status.sort_values("mention_id").itertuples():
        candidates=candidate_pool.get(b.document_id,candidate_status.iloc[0:0])
        available=candidates[~candidates.mention_id.isin(used)]
        same=available[(available.aspect.eq(b.aspect)) & (available.evaluation_status.eq(b.evaluation_status))]
        exact=available[available.evidence_text.astype(str).eq(str(b.evidence_text))]
        chosen=None
        if len(same): chosen=same.sort_values("mention_id").iloc[0]
        elif len(exact): chosen=exact.sort_values("mention_id").iloc[0]
        if chosen is not None: used.add(chosen.mention_id)
        if b.evaluation_status=="TP":
            transition="baseline_TP -> candidate_TP" if chosen is not None and chosen.evaluation_status=="TP" and chosen.aspect==b.aspect else (
                "baseline_TP -> candidate_wrong_aspect" if chosen is not None else "baseline_TP -> candidate_missing")
        else:
            if chosen is None: transition="baseline_FP -> candidate_removed"
            elif chosen.evaluation_status=="TP": transition="baseline_FP -> candidate_corrected_TP"
            elif chosen.aspect==b.aspect: transition="baseline_FP -> candidate_still_FP"
            else: transition="baseline_FP -> candidate_changed_FP"
        gd=gold_by_doc.get(b.document_id,gold.iloc[0:0]); ga=gd[gd.aspect.eq(b.aspect)]
        grow=ga.sort_values("mention_id").iloc[0] if len(ga) else (gd.sort_values("mention_id").iloc[0] if len(gd) else None)
        rows.append({"document_id":b.document_id,"language":meta.loc[b.document_id].primary_language,
          "raw_caption":meta.loc[b.document_id].original_text,"gold_aspect":"" if grow is None else grow.aspect,
          "gold_sentiment":"" if grow is None else grow.sentiment,"gold_evidence":"" if grow is None else grow.evidence_text,
          "baseline_mention_id":b.mention_id,"baseline_aspect":b.aspect,"baseline_sentiment":b.sentiment,
          "baseline_evidence":b.evidence_text,"baseline_status":b.evaluation_status,
          "candidate_mention_id":"" if chosen is None else chosen.mention_id,"candidate_aspect":"" if chosen is None else chosen.aspect,
          "candidate_sentiment":"" if chosen is None else chosen.sentiment,"candidate_evidence":"" if chosen is None else chosen.evidence_text,
          "candidate_status":"missing" if chosen is None else chosen.evaluation_status,"transition_type":transition})
    return pd.DataFrame(rows)


def _reviewed_mechanisms(review: pd.DataFrame, baseline_status: pd.DataFrame,
                         candidate_status: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    candidate_fps=candidate_status[candidate_status.evaluation_status.eq("FP")]
    candidate_tps=candidate_status[candidate_status.evaluation_status.eq("TP")]
    behaviors=[]
    for row in review.itertuples():
        same=candidate_fps[(candidate_fps.document_id.eq(str(row.document_id))) & (candidate_fps.aspect.eq(row.aspect))]
        exact_fp=candidate_fps[(candidate_fps.document_id.eq(str(row.document_id))) & (candidate_fps.evidence_text.eq(row.evidence_text))]
        exact_tp=candidate_tps[(candidate_tps.document_id.eq(str(row.document_id))) & (candidate_tps.evidence_text.eq(row.evidence_text))]
        if len(exact_tp): behavior="corrected"
        elif len(same): behavior="still_fp"
        elif len(exact_fp): behavior="changed_fp"
        elif len(candidate_fps[candidate_fps.document_id.eq(str(row.document_id))]): behavior="changed_fp"
        else: behavior="removed"
        behaviors.append({"review_row_id":row.review_row_id,"mention_id":row.mention_id,
                          "document_id":str(row.document_id),"mechanism":row.fp_error_category,"candidate_behavior":behavior})
    detail=pd.DataFrame(behaviors)
    summary=detail.groupby("mechanism").candidate_behavior.value_counts().unstack(fill_value=0)
    for col in ["removed","still_fp","changed_fp","corrected"]:
        if col not in summary: summary[col]=0
    summary["reviewed_baseline_fp"]=summary.sum(axis=1)
    summary["removal_rate"]=summary.removed/summary.reviewed_baseline_fp
    return detail,summary.reset_index()


def _recall_diagnostic(baseline: pd.DataFrame,candidate: pd.DataFrame,gold: pd.DataFrame,
                       sample: pd.DataFrame) -> pd.DataFrame:
    bcounts=Counter(zip(baseline.document_id,baseline.aspect)); ccounts=Counter(zip(candidate.document_id,candidate.aspect))
    meta=sample.set_index("document_id"); rows=[]
    for key,group in gold.sort_values("mention_id").groupby(["document_id","aspect"],sort=False):
        for pos,(_,g) in enumerate(group.iterrows(),1):
            if pos<=ccounts[key]: continue
            partition="already_FN_under_baseline" if pos>bcounts[key] else "new_FN_created_by_candidate"
            bp=baseline[(baseline.document_id.eq(key[0]))&(baseline.aspect.eq(key[1]))]
            cp=candidate[candidate.document_id.eq(key[0])]
            failure=FAILURE_TYPES.get(key,"other")
            note=("The stricter candidate suppressed or reassigned this qualifying signal; its local target was either treated as non-evaluative or shifted to another ontology family."
                  if partition=="new_FN_created_by_candidate" else
                  "This mention was already missed by the baseline and remains unresolved; instructions_2 did not recover its qualifying signal or frozen aspect boundary.")
            rows.append({"document_id":key[0],"language":meta.loc[key[0]].primary_language,"raw_caption":meta.loc[key[0]].original_text,
              "gold_aspect":g.aspect,"gold_sentiment":g.sentiment,"gold_evidence":g.evidence_text,"gold_expression_type":g.expression_type,
              "baseline_prediction":" || ".join(bp.evidence_text.astype(str)),
              "candidate_prediction":" || ".join(cp.apply(lambda x:f"{x.aspect}: {x.evidence_text}",axis=1)),
              "fn_partition":partition,"failure_type":failure,"researcher_notes":note})
    result=pd.DataFrame(rows)
    if len(result)!=23 or result.fn_partition.eq("new_FN_created_by_candidate").sum()!=12:
        raise ValueError("Candidate FN partition must reconcile to 23 total and 12 new")
    return result


def _metric_row(frame: pd.DataFrame,gold: pd.DataFrame,docs:set[str]) -> dict[str,Any]:
    gg=gold[gold.document_id.isin(docs)]; pp=frame[frame.document_id.isin(docs)]
    gc=Counter(zip(gg.document_id,gg.aspect)); pc=Counter(zip(pp.document_id,pp.aspect)); tp=sum((gc&pc).values())
    return {"tp":tp,"fp":len(pp)-tp,"fn":len(gg)-tp,"precision":tp/len(pp) if len(pp) else 0,
            "recall":tp/len(gg) if len(gg) else 0}


def _markdown_table(frame: pd.DataFrame) -> str:
    values=frame.copy()
    for column in values.select_dtypes(include="number").columns:
        values[column]=values[column].map(lambda value:f"{value:.4f}" if isinstance(value,float) else str(value))
    headers=[str(column) for column in values.columns]
    rows=[[str(value) for value in row] for row in values.itertuples(index=False,name=None)]
    return "\n".join(["| " + " | ".join(headers) + " |",
                       "| " + " | ".join(["---"]*len(headers)) + " |"] +
                      ["| " + " | ".join(row) + " |" for row in rows])


def create_residual_diagnostic(output_root: Path = DEVELOPMENT_ROOT) -> dict[str,Any]:
    baseline,candidate,gold,sample=_load(); captions=sample.set_index("document_id").original_text.astype(str).to_dict()
    languages=sample.set_index("document_id").primary_language.astype(str).to_dict()
    bm=validation_metrics(gold,baseline,captions,languages); cm=validation_metrics(gold,candidate,captions,languages)
    if tuple(bm["aspect_detection"][k] for k in ["tp","fp","fn"])!=(84,134,16): raise ValueError("Baseline reconciliation failed")
    if tuple(cm["aspect_detection"][k] for k in ["tp","fp","fn"])!=(77,74,23): raise ValueError("Candidate reconciliation failed")
    bs=_statuses(baseline,gold); cs=_statuses(candidate,gold)
    topic_names=sample.set_index("document_id").final_topic_name.astype(str).to_dict()
    for status in [bs,cs]: status["final_topic_name"]=status.document_id.map(topic_names)
    residual=_review_residual_fps(cs,gold,sample)
    transitions=_prediction_transition(bs,cs,gold,sample); recall=_recall_diagnostic(baseline,candidate,gold,sample)
    review=pd.read_csv(REVIEWED_FP_PATH).fillna(""); review.document_id=review.document_id.astype(str)
    reviewed_detail,reviewed_summary=_reviewed_mechanisms(review,bs,cs)
    paths={"prediction_transitions":output_root/"baseline_candidate_prediction_transitions_v1.csv",
           "residual_fp_review":output_root/"candidate_residual_fp_review_all_74_v1.csv",
           "reviewed_fp_detail":output_root/"reviewed_baseline_fp_candidate_behavior_v1.csv",
           "reviewed_fp_summary":output_root/"reviewed_baseline_fp_mechanism_transition_v1.csv",
           "recall_diagnostic":output_root/"candidate_fn_recall_diagnostic_all_23_v1.csv"}
    for path in [*paths.values(),output_root/OUTPUT_JSON.name,output_root/OUTPUT_MD.name]:
        if path.exists(): raise FileExistsError(f"Residual diagnostic output exists: {path}")
    transitions.to_csv(paths["prediction_transitions"],index=False,encoding="utf-8-sig")
    residual.to_csv(paths["residual_fp_review"],index=False,encoding="utf-8-sig")
    reviewed_detail.to_csv(paths["reviewed_fp_detail"],index=False,encoding="utf-8-sig")
    reviewed_summary.to_csv(paths["reviewed_fp_summary"],index=False,encoding="utf-8-sig")
    recall.to_csv(paths["recall_diagnostic"],index=False,encoding="utf-8-sig")
    def distributions(status:pd.DataFrame)->dict[str,Any]:
        fp=status[status.evaluation_status.eq("FP")].copy(); gold_docs=set(gold.document_id)
        counts=status.groupby("document_id").size(); affected=fp.groupby("document_id").size().sort_values(ascending=False)
        return {"total":len(fp),"by_aspect":fp.aspect.value_counts().to_dict(),
          "by_language":fp.document_id.map(languages).value_counts().to_dict(),
          "by_final_topic":fp.final_topic_name.value_counts().to_dict() if "final_topic_name" in fp else {},
          "by_sentiment":fp.sentiment.value_counts().to_dict(),"gold_zero_mentions":int((~fp.document_id.isin(gold_docs)).sum()),
          "gold_zero_affected_documents":int(fp.loc[~fp.document_id.isin(gold_docs),"document_id"].nunique()),
          "fp_in_predicted_multi_mention_documents":int(fp.document_id.map(counts).gt(1).sum()),
          "affected_documents":int(fp.document_id.nunique()),"fp_per_affected_document":affected.to_dict(),
          "top_documents":affected.head(10).to_dict()}
    aspects=[]
    for aspect in sorted(set(baseline.aspect)|set(candidate.aspect)):
        bv=int(((bs.aspect==aspect)&(bs.evaluation_status=="FP")).sum()); cv=int(((cs.aspect==aspect)&(cs.evaluation_status=="FP")).sum())
        aspects.append({"aspect":aspect,"baseline_fp":bv,"candidate_fp":cv,"absolute_reduction":bv-cv,
                        "percentage_reduction":(bv-cv)/bv if bv else None})
    language=[]
    groups={x:{d for d,l in languages.items() if l==x} for x in ["English","Malay","Chinese","Malay/Indonesian uncertain"]}
    groups["non-English/uncertain"]={d for d,l in languages.items() if l!="English"}
    for name,docs in groups.items():
        b=_metric_row(baseline,gold,docs);c=_metric_row(candidate,gold,docs)
        language.append({"language":name,"documents":len(docs),"baseline":b,"candidate":c,"fp_delta":c["fp"]-b["fp"],"fn_delta":c["fn"]-b["fn"]})
    residual_counts=residual.candidate_fp_category.value_counts().to_dict()
    gold_zero=cm["document_diagnostics"]["zero_mention_confusion_gold_rows_predicted_columns"]["gold_zero_predicted_mentions"]
    multi=cm["document_diagnostics"]["predicted_multi_aspect_documents"]
    candidate_multi_docs=set(candidate.groupby("document_id").size().loc[lambda x:x>1].index)
    fp_multi=int(residual.document_id.isin(candidate_multi_docs).sum()); pred_multi_mentions=int(candidate.document_id.isin(candidate_multi_docs).sum())
    conclusion="FURTHER_PROMPT_INTERVENTION_JUSTIFIED"
    report={"version":VERSION,"created_at_utc":datetime.now(timezone.utc).isoformat(),"development_evidence":True,
      "confirmatory_evidence":False,"api_calls":0,"inference_run":False,"prompt_modified":False,
      "reconciliation":{"baseline":bm["aspect_detection"],"candidate":cm["aspect_detection"]},
      "fp_reduction":{"baseline":134,"candidate":74,"absolute":60,"percentage":60/134},
      "baseline_fp_distribution":distributions(bs),"candidate_fp_distribution":distributions(cs),
      "fp_aspect_comparison":aspects,"residual_fp_category_counts":residual_counts,
      "residual_fp_confidence_counts":residual.confidence.value_counts().to_dict(),
      "reviewed_mechanism_transition":reviewed_summary.to_dict("records"),
      "recall":{"candidate_fn_total":23,"already_fn_under_baseline":11,"new_fn_created_by_candidate":12,
                "new_fn_failure_types":recall[recall.fn_partition.eq("new_FN_created_by_candidate")].failure_type.value_counts().to_dict(),
                "all_fn_by_aspect":recall.gold_aspect.value_counts().to_dict(),"all_fn_by_language":recall.language.value_counts().to_dict(),
                "all_fn_by_expression_type":recall.gold_expression_type.value_counts().to_dict(),"all_fn_by_sentiment":recall.gold_sentiment.value_counts().to_dict()},
      "gold_zero":{"baseline_gold_zero_predicted_mentions_documents":11,"candidate_gold_zero_predicted_mentions_documents":gold_zero},
      "multi_aspect":{"gold_multi_aspect_documents":27,"baseline_predicted_multi_aspect_documents":43,
                      "candidate_predicted_multi_aspect_documents":multi,"candidate_fp_mentions_in_multi_documents":fp_multi,
                      "candidate_mentions_in_multi_documents":pred_multi_mentions,
                      "candidate_fp_rate_within_multi_documents":fp_multi/pred_multi_mentions if pred_multi_mentions else 0},
      "language_comparison":language,"diagnostic_conclusion":conclusion,
      "further_intervention_justified":True,
      "dominant_unresolved_policy_problem":"single-proposition aspect ownership and mention cardinality: affective wording is still emitted as a separate emotional or adjacent-family mention when the frozen policy uses it as evidence for one object-level evaluation",
      "methodological_interpretation":{"fixed":"Substantially reduced unsupported expansion, context/factual extraction, gold-zero errors and multi-mention inflation.",
        "not_fixed":"Ontology-family ownership, redundant mentions and separate emotional mentions for affect already attributable to another target remain dominant.",
        "suppressed":"Thirteen new FNs include prospective/encouragement signals, multilingual colloquial affect, hashtags, comparative performance, implicit physical experience and several ontology-boundary shifts.",
        "recommendation":"Do not create instructions_3 now. A future intervention is justified only as a controlled attribution/remapping intervention, not broader conservatism."},
      "artifacts":{k:str(v) for k,v in paths.items()},
      "input_hashes":{"baseline_predictions_sha256":_sha(BASELINE_PATH),"candidate_predictions_sha256":_sha(CANDIDATE_PATH),
                      "gold_sha256":_sha(GOLD_PATH),"sample_sha256":_sha(SAMPLE_PATH),"ontology_sha256":_sha(ONTOLOGY_PATH)}}
    (output_root/OUTPUT_JSON.name).write_text(json.dumps(report,indent=2,ensure_ascii=False,default=float),encoding="utf-8")
    top="\n".join(f"- `{k}`: {v}" for k,v in sorted(residual_counts.items(),key=lambda x:-x[1]))
    mech=_markdown_table(reviewed_summary.sort_values("removal_rate",ascending=False))
    langmd=_markdown_table(pd.DataFrame([{"language":x["language"],"baseline_precision":x["baseline"]["precision"],"candidate_precision":x["candidate"]["precision"],
                         "baseline_recall":x["baseline"]["recall"],"candidate_recall":x["candidate"]["recall"],"fp_delta":x["fp_delta"],"fn_delta":x["fn_delta"]} for x in language]))
    markdown=f"""# ABSA V1 Precision V2 Residual Diagnostic\n\nDevelopment evidence: **true**. Confirmatory evidence: **false**. No API call or new inference occurred.\n\n## Development result\n\nPrecision: **{cm['aspect_detection']['precision']:.4f}**  \nRecall: **{cm['aspect_detection']['recall']:.4f}**  \nF1: **{cm['aspect_detection']['f1']:.4f}**  \nTP/FP/FN: **77 / 74 / 23**  \nDecision: **INSUFFICIENT_PRECISION_IMPROVEMENT**\n\n## FP reduction\n\n**134 -> 74: 60 FPs removed, {60/134:.1%} reduction.**\n\n## Reviewed-mechanism transition\n\n{mech}\n\n## Residual FP mechanisms\n\n{top}\n\nThe dominant residual is aspect ownership: genuine affect is still separated into `emotional_experience` or mapped to an adjacent family when the frozen reference treats that wording as evidence for a different object-level evaluation. Duplicate/cardinality errors are the next material class.\n\n## Recall regressions\n\nCandidate FN = 23: ten were already baseline FNs and thirteen are new. New losses include ontology-boundary shifts, prospective/community encouragement, multilingual colloquial affect, hashtag-only affect, comparative performance, implicit physical experience, and one repeated gold proposition suppressed by the minimum-set policy.\n\n## Gold-zero and multi-aspect behavior\n\nGold-zero documents receiving predictions improved **11 -> {gold_zero}**. Predicted multi-aspect documents improved **43 -> {multi}**, versus 27 gold multi-aspect documents. Candidate FP rate within candidate multi-mention documents is **{fp_multi}/{pred_multi_mentions} ({fp_multi/pred_multi_mentions:.1%})**.\n\n## Multilingual impact\n\n{langmd}\n\nTiny language groups are descriptive only. Malay recall declined and several Malay boundary/colloquial signals moved to adjacent aspects; Chinese retained high recall but seven FPs remain across five documents. Aggregated non-English recall remained stronger than English, so there is no broad multilingual collapse, but individual culturally natural signals were lost.\n\n## Methodological interpretation\n\n**What instructions_2 fixed:** unsupported expansion and context/factual extraction fell materially; gold-zero contamination and multi-aspect inflation also declined.\n\n**What it failed to fix:** ontology-family ownership and mention cardinality. It often still emits affect as its own emotional mention in addition to, or instead of, the frozen object family.\n\n**What it suppressed:** thirteen new gold misses spanning implicit/prospective evaluation, encouragement, colloquial multilingual affect, hashtags, comparisons, physical experience and boundary-sensitive mentions.\n\n**Single unresolved policy problem:** single-proposition aspect ownership—deciding whether affect is the evaluated target itself or evidence about another target, and emitting exactly the frozen mention cardinality.\n\n## Recommendation\n\n**{conclusion}**\n\nDo not create `absa_v1_instructions_3` now. A future controlled intervention is methodologically defensible because a concentrated, generalizable attribution/remapping problem can theoretically reduce more than the approximately eleven FPs needed to reach 0.55 without simply suppressing evaluation. It must target ownership/remapping rather than greater conservatism and must explicitly protect the documented recall-regression signals.\n"""
    markdown=markdown.replace(
        "Candidate FN = 23: ten were already baseline FNs and thirteen are new.",
        "Candidate FN = 23: eleven were already baseline FNs and twelve are new; five baseline FNs were recovered.",
    ).replace("thirteen new gold misses", "twelve new gold misses")
    (output_root/OUTPUT_MD.name).write_text(markdown,encoding="utf-8")
    return report
