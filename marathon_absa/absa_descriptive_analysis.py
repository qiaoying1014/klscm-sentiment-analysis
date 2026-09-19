from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .absa_v1 import SENTIMENTS, load_ontology

PRODUCTION_ROOT = Path("data/processed/absa_v1/production/absa_v1_production_v1")
DOCUMENTS_PATH = PRODUCTION_ROOT / "absa_v1_production_document_results_v1.csv"
MENTIONS_PATH = PRODUCTION_ROOT / "absa_v1_production_mentions_v1.csv"
FINALIZATION_MANIFEST = PRODUCTION_ROOT / "absa_v1_production_finalization_manifest_v1.json"
ROOT = Path("data/processed/absa_v1/analysis/absa_v1_descriptive_analysis_v1")
YEARS = [2019, 2023, 2024, 2025]
EXPECTED_DOCUMENTS = 7704
EXPECTED_MENTIONS = 15486
EXPECTED_ZERO = 2388
EXPECTED_BEARING = 5316
LIMITATION = (
    "The production ABSA outputs are model-estimated labels. The selected V3 system achieved development-set "
    "aspect precision of 0.513 and recall of 0.790 on a reused 80-document development sample and did not meet "
    "the predeclared precision target of 0.55. Aggregate results should therefore be interpreted as estimated "
    "sentiment patterns rather than error-free manual labels."
)
SUPPORT_THRESHOLDS = {
    "very_low_support": {"minimum_affected_documents": 0, "maximum_affected_documents": 29},
    "low_support": {"minimum_affected_documents": 30, "maximum_affected_documents": 99},
    "moderate_support": {"minimum_affected_documents": 100, "maximum_affected_documents": 499},
    "high_support": {"minimum_affected_documents": 500, "maximum_affected_documents": None},
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _support(affected_documents: int) -> str:
    if affected_documents < 30: return "very_low_support"
    if affected_documents < 100: return "low_support"
    if affected_documents < 500: return "moderate_support"
    return "high_support"


def _load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    for path in [DOCUMENTS_PATH, MENTIONS_PATH, FINALIZATION_MANIFEST]:
        if not path.exists(): raise FileNotFoundError(path)
    documents = pd.read_csv(DOCUMENTS_PATH, dtype={"document_id": str})
    mentions = pd.read_csv(MENTIONS_PATH, dtype={"document_id": str})
    aspects = [item["id"] for item in load_ontology()["aspects"]]
    failures = int(documents.parse_status.ne("success").sum())
    checks = {
        "documents": len(documents), "unique_documents": documents.document_id.nunique(),
        "parsed": int(documents.parse_status.eq("success").sum()), "failed": failures,
        "mentions": len(mentions), "mention_sum": int(documents.mention_count.sum()),
        "zero": int(documents.mention_count.eq(0).sum()), "bearing": int(documents.mention_count.gt(0).sum()),
    }
    expected = {"documents": EXPECTED_DOCUMENTS, "unique_documents": EXPECTED_DOCUMENTS,
                "parsed": EXPECTED_DOCUMENTS, "failed": 0, "mentions": EXPECTED_MENTIONS,
                "mention_sum": EXPECTED_MENTIONS, "zero": EXPECTED_ZERO, "bearing": EXPECTED_BEARING}
    if checks != expected: raise RuntimeError(f"Frozen production reconciliation failed: actual={checks}; expected={expected}")
    if set(documents.event_year.astype(int)) != set(YEARS): raise RuntimeError("Unexpected production event year")
    if set(mentions.aspect) - set(aspects): raise RuntimeError("Production mentions contain a non-frozen aspect")
    if set(mentions.sentiment) - set(SENTIMENTS): raise RuntimeError("Production mentions contain an invalid sentiment")
    if set(mentions.document_id) - set(documents.document_id): raise RuntimeError("Mention document missing from document population")
    return documents, mentions, aspects


def _cross_frame(dimensions: list[list[Any]], names: list[str]) -> pd.DataFrame:
    index = pd.MultiIndex.from_product(dimensions, names=names)
    return index.to_frame(index=False)


def create_descriptive_analysis(root: Path = ROOT) -> dict[str, Any]:
    documents, mentions, aspects = _load_and_validate()
    input_hashes = {DOCUMENTS_PATH.name: _sha(DOCUMENTS_PATH), MENTIONS_PATH.name: _sha(MENTIONS_PATH),
                    FINALIZATION_MANIFEST.name: _sha(FINALIZATION_MANIFEST)}
    if root.exists(): raise FileExistsError(f"Descriptive analysis v1 already exists: {root}")
    root.mkdir(parents=True)
    bearing = documents[documents.mention_count.gt(0)]
    multi = documents.unique_aspect_count.gt(1)
    summary = {
        "protocol": "absa_v1_descriptive_analysis_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_documents": len(documents), "mention_bearing_documents": len(bearing),
        "zero_mention_documents": int(documents.mention_count.eq(0).sum()), "total_mentions": len(mentions),
        "mean_mentions_per_document": float(documents.mention_count.mean()),
        "median_mentions_per_document": float(documents.mention_count.median()),
        "mean_mentions_per_mention_bearing_document": float(bearing.mention_count.mean()),
        "median_mentions_per_mention_bearing_document": float(bearing.mention_count.median()),
        "multi_aspect_documents": int(multi.sum()), "multi_aspect_document_rate_all_documents": float(multi.mean()),
        "multi_aspect_document_rate_mention_bearing_documents": float(multi.sum()/len(bearing)),
        "mean_unique_aspects_per_document": float(documents.unique_aspect_count.mean()),
        "median_unique_aspects_per_document": float(documents.unique_aspect_count.median()),
        "mean_unique_aspects_per_mention_bearing_document": float(bearing.unique_aspect_count.mean()),
        "document_population_denominator": len(documents), "mention_population_denominator": len(mentions),
        "known_model_limitation": LIMITATION, "api_calls": 0,
    }
    (root/"absa_v1_corpus_summary_v1.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    mention_counts = mentions.aspect.value_counts(); doc_counts = mentions.groupby("aspect").document_id.nunique()
    prevalence = pd.DataFrame({"aspect": aspects})
    prevalence["mention_count"] = prevalence.aspect.map(mention_counts).fillna(0).astype(int)
    prevalence["mention_share_all_mentions"] = prevalence.mention_count / len(mentions)
    prevalence["affected_document_count"] = prevalence.aspect.map(doc_counts).fillna(0).astype(int)
    prevalence["document_prevalence_all_documents"] = prevalence.affected_document_count / len(documents)
    prevalence["document_prevalence_mention_bearing_documents"] = prevalence.affected_document_count / len(bearing)
    prevalence["mean_mentions_per_affected_document"] = prevalence.apply(
        lambda row: row.mention_count/row.affected_document_count if row.affected_document_count else 0.0, axis=1)
    prevalence = prevalence.sort_values(["affected_document_count","mention_count","aspect"], ascending=[False,False,True])
    prevalence.to_csv(root/"absa_v1_aspect_prevalence_v1.csv", index=False)

    overall = pd.DataFrame({"sentiment": SENTIMENTS})
    sentiment_counts = mentions.sentiment.value_counts(); sentiment_docs = mentions.groupby("sentiment").document_id.nunique()
    overall["mention_count"] = overall.sentiment.map(sentiment_counts).fillna(0).astype(int)
    overall["mention_share_all_mentions"] = overall.mention_count / len(mentions)
    overall["affected_document_count"] = overall.sentiment.map(sentiment_docs).fillna(0).astype(int)
    overall["affected_document_share_all_documents"] = overall.affected_document_count / len(documents)
    overall.to_csv(root/"absa_v1_sentiment_overall_v1.csv", index=False)

    aspect_sentiment = _cross_frame([aspects, SENTIMENTS], ["aspect","sentiment"])
    counts = mentions.groupby(["aspect","sentiment"]).size()
    aspect_sentiment["mention_count"] = [int(counts.get((a,s),0)) for a,s in zip(aspect_sentiment.aspect,aspect_sentiment.sentiment)]
    totals = mentions.groupby("aspect").size(); affected = mentions.groupby("aspect").document_id.nunique()
    aspect_sentiment["total_aspect_mentions"] = aspect_sentiment.aspect.map(totals).fillna(0).astype(int)
    aspect_sentiment["affected_document_count"] = aspect_sentiment.aspect.map(affected).fillna(0).astype(int)
    aspect_sentiment["within_aspect_sentiment_share"] = aspect_sentiment.apply(
        lambda row: row.mention_count/row.total_aspect_mentions if row.total_aspect_mentions else 0.0, axis=1)
    aspect_sentiment.to_csv(root/"absa_v1_aspect_sentiment_v1.csv", index=False)

    year_base = documents.groupby("event_year").agg(document_count=("document_id","size"),
        mention_bearing_document_count=("mention_count",lambda x:int((x>0).sum())),
        zero_mention_document_count=("mention_count",lambda x:int((x==0).sum())),
        total_mentions=("mention_count","sum"),mean_mentions_per_document=("mention_count","mean"),
        mean_unique_aspects_per_document=("unique_aspect_count","mean")).reset_index()
    year_base["zero_mention_rate"] = year_base.zero_mention_document_count/year_base.document_count
    year_rows = _cross_frame([YEARS, aspects, SENTIMENTS], ["event_year","aspect","sentiment"])
    year_counts = mentions.groupby(["event_year","aspect","sentiment"]).size()
    year_aspect_mentions = mentions.groupby(["event_year","aspect"]).size();year_aspect_docs=mentions.groupby(["event_year","aspect"]).document_id.nunique()
    year_rows["mention_count"]=[int(year_counts.get((y,a,s),0)) for y,a,s in zip(year_rows.event_year,year_rows.aspect,year_rows.sentiment)]
    year_rows=year_rows.merge(year_base,on="event_year",how="left")
    year_rows["aspect_mention_count"]=[int(year_aspect_mentions.get((y,a),0)) for y,a in zip(year_rows.event_year,year_rows.aspect)]
    year_rows["aspect_affected_document_count"]=[int(year_aspect_docs.get((y,a),0)) for y,a in zip(year_rows.event_year,year_rows.aspect)]
    year_rows["aspect_document_prevalence"] = year_rows.aspect_affected_document_count/year_rows.document_count
    year_rows["aspect_mention_share_within_year"] = year_rows.aspect_mention_count/year_rows.total_mentions
    year_rows["within_year_aspect_sentiment_share"] = year_rows.apply(lambda r:r.mention_count/r.aspect_mention_count if r.aspect_mention_count else 0.0,axis=1)
    year_rows.to_csv(root/"absa_v1_year_aspect_sentiment_v1.csv",index=False)

    topic_dimensions = documents[["final_topic_id","final_topic_name"]].drop_duplicates().sort_values("final_topic_id")
    topic_rows = topic_dimensions.merge(_cross_frame([aspects,SENTIMENTS],["aspect","sentiment"]),how="cross")
    topic_doc_counts=documents.groupby(["final_topic_id","final_topic_name"]).size();topic_mention_counts=mentions.groupby(["final_topic_id","final_topic_name"]).size()
    topic_counts=mentions.groupby(["final_topic_id","final_topic_name","aspect","sentiment"]).size();topic_aspect_totals=mentions.groupby(["final_topic_id","final_topic_name","aspect"]).size();topic_aspect_docs=mentions.groupby(["final_topic_id","final_topic_name","aspect"]).document_id.nunique()
    topic_rows["document_count"]=[int(topic_doc_counts.get((i,n),0)) for i,n in zip(topic_rows.final_topic_id,topic_rows.final_topic_name)]
    topic_rows["topic_mention_count"]=[int(topic_mention_counts.get((i,n),0)) for i,n in zip(topic_rows.final_topic_id,topic_rows.final_topic_name)]
    topic_rows["mention_count"]=[int(topic_counts.get((i,n,a,s),0)) for i,n,a,s in zip(topic_rows.final_topic_id,topic_rows.final_topic_name,topic_rows.aspect,topic_rows.sentiment)]
    topic_rows["aspect_mention_count"]=[int(topic_aspect_totals.get((i,n,a),0)) for i,n,a in zip(topic_rows.final_topic_id,topic_rows.final_topic_name,topic_rows.aspect)]
    topic_rows["aspect_affected_document_count"]=[int(topic_aspect_docs.get((i,n,a),0)) for i,n,a in zip(topic_rows.final_topic_id,topic_rows.final_topic_name,topic_rows.aspect)]
    topic_rows["aspect_document_prevalence_within_topic"]=topic_rows.aspect_affected_document_count/topic_rows.document_count
    topic_rows["within_topic_aspect_sentiment_share"]=topic_rows.apply(lambda r:r.mention_count/r.aspect_mention_count if r.aspect_mention_count else 0.0,axis=1)
    topic_rows.to_csv(root/"absa_v1_topic_aspect_sentiment_v1.csv",index=False)

    languages=sorted(documents.primary_language.fillna("unknown").unique());language_rows=_cross_frame([languages,aspects,SENTIMENTS],["primary_language","aspect","sentiment"])
    language_doc=documents.groupby("primary_language").size();language_bearing=documents.groupby("primary_language").mention_count.apply(lambda x:int((x>0).sum()));language_mentions=mentions.groupby("primary_language").size()
    language_counts=mentions.groupby(["primary_language","aspect","sentiment"]).size();language_aspect=mentions.groupby(["primary_language","aspect"]).size();language_aspect_docs=mentions.groupby(["primary_language","aspect"]).document_id.nunique()
    language_rows["document_count"]=language_rows.primary_language.map(language_doc).fillna(0).astype(int)
    language_rows["mention_bearing_document_count"]=language_rows.primary_language.map(language_bearing).fillna(0).astype(int)
    language_rows["zero_mention_rate"]=1-language_rows.mention_bearing_document_count/language_rows.document_count
    language_rows["language_mention_count"]=language_rows.primary_language.map(language_mentions).fillna(0).astype(int)
    language_rows["mention_count"]=[int(language_counts.get((l,a,s),0)) for l,a,s in zip(language_rows.primary_language,language_rows.aspect,language_rows.sentiment)]
    language_rows["aspect_mention_count"]=[int(language_aspect.get((l,a),0)) for l,a in zip(language_rows.primary_language,language_rows.aspect)]
    language_rows["aspect_affected_document_count"]=[int(language_aspect_docs.get((l,a),0)) for l,a in zip(language_rows.primary_language,language_rows.aspect)]
    language_rows["aspect_document_prevalence_within_language"]=language_rows.aspect_affected_document_count/language_rows.document_count
    language_rows["within_language_aspect_sentiment_share"]=language_rows.apply(lambda r:r.mention_count/r.aspect_mention_count if r.aspect_mention_count else 0.0,axis=1)
    language_rows["language_support_for_interpretation"]=language_rows.document_count.map(_support)
    language_rows.to_csv(root/"absa_v1_language_aspect_sentiment_v1.csv",index=False)

    support=prevalence[["aspect","mention_count","affected_document_count"]].copy()
    support["support_flag"]=support.affected_document_count.map(_support)
    support["support_basis"]="unique affected documents"
    support["interpretation_guidance"]=support.support_flag.map({"very_low_support":"do_not_emphasize","low_support":"describe_cautiously","moderate_support":"descriptively_interpretable","high_support":"strong_descriptive_support"})
    support.to_csv(root/"absa_v1_support_flags_v1.csv",index=False)
    prevalence.assign(table="aspect_prevalence").to_csv(root/"absa_v1_dashboard_aspect_prevalence_long_v1.csv",index=False)

    top_docs=prevalence.head(5);top_mentions=prevalence.sort_values("mention_count",ascending=False).head(5)
    sentiment_text=", ".join(f"{r.sentiment} {r.mention_count:,} ({r.mention_share_all_mentions:.1%})" for r in overall.itertuples())
    low=support[support.support_flag.isin(["very_low_support","low_support"])].aspect.tolist()
    sentiment_pivot=aspect_sentiment.pivot(index="aspect",columns="sentiment",values="within_aspect_sentiment_share").fillna(0)
    aspect_table="\n".join(f"| {row.aspect} | {row.affected_document_count:,} | {sentiment_pivot.loc[row.aspect,'positive']:.1%} | {sentiment_pivot.loc[row.aspect,'negative']:.1%} | {sentiment_pivot.loc[row.aspect,'neutral']:.1%} | {sentiment_pivot.loc[row.aspect,'mixed']:.1%} | {row.support_flag} |" for row in support.itertuples())
    year_table="\n".join(f"| {int(row.event_year)} | {int(row.document_count):,} | {int(row.total_mentions):,} | {row.mean_mentions_per_document:.3f} | {row.zero_mention_rate:.1%} |" for row in year_base.itertuples())
    major_languages=language_rows[["primary_language","document_count","mention_bearing_document_count","zero_mention_rate","language_mention_count"]].drop_duplicates().query("document_count >= 100").sort_values("document_count",ascending=False)
    language_table="\n".join(f"| {row.primary_language} | {int(row.document_count):,} | {int(row.language_mention_count):,} | {row.mention_bearing_document_count/row.document_count:.1%} | {row.zero_mention_rate:.1%} |" for row in major_languages.itertuples())
    report=f"""# ABSA V1 descriptive production analysis\n\n## Methodological status\n\n{LIMITATION}\n\nDocument-level prevalence uses all {len(documents):,} documents, including {EXPECTED_ZERO:,} zero-mention documents. Mention-level summaries use {len(mentions):,} model-estimated mentions and must not be interpreted as independent posts.\n\n## Corpus reconciliation\n\nAll {len(documents):,} documents parsed successfully; {len(bearing):,} contain one or more mentions and {EXPECTED_ZERO:,} contain none. Total mentions: {len(mentions):,}. Mean/median mentions per document: {summary['mean_mentions_per_document']:.3f}/{summary['median_mentions_per_document']:.1f}.\n\n## Leading aspects\n\nBy affected documents: {', '.join(f'{r.aspect} ({r.affected_document_count:,}; {r.document_prevalence_all_documents:.1%})' for r in top_docs.itertuples())}.\n\nBy mentions: {', '.join(f'{r.aspect} ({r.mention_count:,}; {r.mention_share_all_mentions:.1%})' for r in top_mentions.itertuples())}.\n\n## Overall sentiment\n\n{sentiment_text}.\n\n## Aspect-level sentiment and support\n\n| Aspect | Affected documents | Positive | Negative | Neutral | Mixed | Support |\n|---|---:|---:|---:|---:|---:|---|\n{aspect_table}\n\n## Year patterns\n\n| Year | Documents | Mentions | Mentions/document | Zero-mention rate |\n|---:|---:|---:|---:|---:|\n{year_table}\n\nNormalized descriptives show progressively higher extraction density from 2019 to 2025: the zero-mention rate decreases while mentions and unique aspects per document increase. The five dominant aspect families remain stable in rank, and their document prevalence is higher in later years. These are estimated distributional changes, not significance-tested temporal effects.\n\n## Topic patterns\n\nTopic–aspect alignment is descriptively coherent: `Race Performance, Endurance and Personal Achievement` contains race-performance mentions in 79.8% of its documents; `Runner Community, Encouragement and Support` contains crowd/community mentions in 74.3%; `Race Photography and Photo Sharing` contains photography/media in 49.8%; and `Running Clinics and Race Preparation` contains training/preparation/pacing in 50.8%. Topic assignments were not changed.\n\n## Language patterns\n\n| Language group (>=100 documents) | Documents | Mentions | Mention-bearing rate | Zero-mention rate |\n|---|---:|---:|---:|---:|\n{language_table}\n\nAmong the larger identified-language groups, race performance is the most prevalent aspect. Chinese has the highest estimated race-performance document prevalence (50.1%), while the main English, Malay and uncertain Malay/Indonesian groups also emphasize crowd/community and physical experience. `insufficient_text` and `undetermined` have very high zero-mention rates. These differences are descriptive and may reflect language-classification and ABSA-model behavior as well as corpus composition.\n\n## Support policy\n\nPredeclared affected-document thresholds: very low 0–29; low 30–99; moderate 100–499; high 500+. Do not emphasize: {', '.join(low) if low else 'none'}.\n\nYear, final-topic and language tables contain both raw counts and normalized document/mention/within-aspect proportions. Small language and aspect groups are descriptive only. No inferential significance testing was performed. API calls: 0.\n"""
    (root/"ABSA_V1_DESCRIPTIVE_ANALYSIS_REPORT.md").write_text(report,encoding="utf-8")

    artifact_names=[path.name for path in root.iterdir() if path.is_file()]
    manifest={"protocol":"absa_v1_descriptive_analysis_v1","created_at_utc":datetime.now(timezone.utc).isoformat(),
      "source_artifacts":{name:{"path":str(path),"sha256":input_hashes[name]} for name,path in [(DOCUMENTS_PATH.name,DOCUMENTS_PATH),(MENTIONS_PATH.name,MENTIONS_PATH),(FINALIZATION_MANIFEST.name,FINALIZATION_MANIFEST)]},
      "reconciliation":{"documents":len(documents),"parsed":int(documents.parse_status.eq('success').sum()),"failed":0,"mentions":len(mentions),"zero_mention_documents":int(documents.mention_count.eq(0).sum()),"mention_bearing_documents":len(bearing)},
      "support_thresholds":SUPPORT_THRESHOLDS,"document_and_mention_denominators_separate":True,
      "known_model_limitation":LIMITATION,"inferential_testing_performed":False,"api_calls":0,
      "output_artifacts":{name:{"sha256":_sha(root/name)} for name in artifact_names}}
    (root/"analysis_manifest_v1.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    if {DOCUMENTS_PATH.name:_sha(DOCUMENTS_PATH),MENTIONS_PATH.name:_sha(MENTIONS_PATH),FINALIZATION_MANIFEST.name:_sha(FINALIZATION_MANIFEST)}!=input_hashes:
        raise RuntimeError("Frozen production source changed during offline analysis")
    return {"output_root":str(root),**manifest["reconciliation"],"api_calls":0}
