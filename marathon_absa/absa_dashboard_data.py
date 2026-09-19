from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .absa_descriptive_analysis import (
    DOCUMENTS_PATH, EXPECTED_BEARING, EXPECTED_DOCUMENTS, EXPECTED_MENTIONS,
    EXPECTED_ZERO, MENTIONS_PATH, SENTIMENTS, SUPPORT_THRESHOLDS, YEARS, _sha,
)
from .absa_v1 import load_ontology

DESCRIPTIVE = Path("data/processed/absa_v1/analysis/absa_v1_descriptive_analysis_v1")
INFERENTIAL = Path("data/processed/absa_v1/analysis/absa_v1_inferential_analysis_v1")
ROOT = Path("data/processed/absa_v1/dashboard/absa_v1_dashboard_data_v1")
FROZEN_ELIGIBLE = [
    "race_performance", "crowd_community_atmosphere", "physical_experience",
    "emotional_experience", "training_preparation_pacing", "photography_media",
    "emerging_other", "route_course", "organization_operations",
    "weather_conditions", "value_cost", "safety_medical",
]
WARNINGS = {
    "model": "Aspect and sentiment labels are model-estimated. The selected ABSA system achieved development precision of 0.513 and recall of 0.790 and did not meet the predeclared 0.55 precision target.",
    "statistical": "Statistical tests quantify associations in model-estimated labels and do not account for ABSA classification error.",
    "temporal": "Cross-year comparisons should be interpreted alongside changing corpus composition and increasing extraction density across editions.",
    "language": "Language comparisons are descriptive and should not be interpreted as evidence of cultural or linguistic differences.",
    "topic": "Topic-level sentiment patterns are descriptive; no topic-wide inferential testing was performed.",
}
SUPPORT_WARNINGS = {
    "very_low_support": "Very limited support; do not emphasize.",
    "low_support": "Limited support; interpret cautiously.",
    "moderate_support": "Moderate support.",
    "high_support": "High corpus support.",
}


def _effect(value: float | None) -> str:
    if value is None or pd.isna(value): return "not_tested"
    if value < .10: return "trivial"
    if value < .20: return "small"
    if value < .30: return "moderate"
    return "strong"


def _sources() -> dict[str, Path]:
    names = {
        "corpus_summary": DESCRIPTIVE / "absa_v1_corpus_summary_v1.json",
        "aspect_prevalence": DESCRIPTIVE / "absa_v1_aspect_prevalence_v1.csv",
        "sentiment_overall": DESCRIPTIVE / "absa_v1_sentiment_overall_v1.csv",
        "aspect_sentiment": DESCRIPTIVE / "absa_v1_aspect_sentiment_v1.csv",
        "year_aspect_sentiment": DESCRIPTIVE / "absa_v1_year_aspect_sentiment_v1.csv",
        "topic_aspect_sentiment": DESCRIPTIVE / "absa_v1_topic_aspect_sentiment_v1.csv",
        "language_aspect_sentiment": DESCRIPTIVE / "absa_v1_language_aspect_sentiment_v1.csv",
        "support": DESCRIPTIVE / "absa_v1_support_flags_v1.csv",
        "descriptive_manifest": DESCRIPTIVE / "analysis_manifest_v1.json",
        "document_matrix": INFERENTIAL / "absa_v1_document_analysis_matrix_v1.csv",
        "inferential_prevalence": INFERENTIAL / "absa_v1_inferential_aspect_prevalence_v1.csv",
        "inferential_positive": INFERENTIAL / "absa_v1_inferential_positive_year_v1.csv",
        "inferential_negative": INFERENTIAL / "absa_v1_inferential_negative_year_v1.csv",
        "pairwise": INFERENTIAL / "absa_v1_inferential_pairwise_year_v1.csv",
        "eligibility": INFERENTIAL / "absa_v1_inferential_eligibility_v1.csv",
        "inferential_manifest": INFERENTIAL / "absa_v1_inferential_analysis_manifest_v1.json",
        "production_documents": DOCUMENTS_PATH,
        "production_mentions": MENTIONS_PATH,
    }
    missing = [str(path) for path in names.values() if not path.exists()]
    if missing: raise FileNotFoundError(f"Missing frozen sources: {missing}")
    return names


def create_dashboard_data(root: Path = ROOT, created_at: str | None = None) -> dict[str, Any]:
    if root.exists(): raise FileExistsError(f"Dashboard data mart already exists: {root}")
    sources = _sources(); source_hashes = {name: _sha(path) for name, path in sources.items()}
    created_at = created_at or datetime.now(timezone.utc).isoformat()
    documents = pd.read_csv(DOCUMENTS_PATH, dtype={"document_id": str})
    mentions = pd.read_csv(MENTIONS_PATH, dtype={"document_id": str})
    matrix = pd.read_csv(sources["document_matrix"], dtype={"document_id": str})
    summary = json.loads(sources["corpus_summary"].read_text(encoding="utf-8"))
    checks = (len(documents), documents.document_id.nunique(), len(mentions),
              int(documents.mention_count.gt(0).sum()), int(documents.mention_count.eq(0).sum()))
    expected = (EXPECTED_DOCUMENTS, EXPECTED_DOCUMENTS, EXPECTED_MENTIONS, EXPECTED_BEARING, EXPECTED_ZERO)
    if checks != expected: raise RuntimeError(f"Production reconciliation failed: {checks} != {expected}")
    expected_years = {2019: 2279, 2023: 1873, 2024: 1724, 2025: 1828}
    if documents.event_year.value_counts().sort_index().to_dict() != expected_years:
        raise RuntimeError("Frozen year counts changed")
    eligibility = pd.read_csv(sources["eligibility"])
    eligible = eligibility.loc[eligibility.primary_inference_eligible, "aspect"].tolist()
    if eligible != FROZEN_ELIGIBLE: raise RuntimeError(f"Inferential eligibility changed: {eligible}")
    ontology = load_ontology(); aspects = [row["id"] for row in ontology["aspects"]]
    labels = {row["id"]: row["name"].title() for row in ontology["aspects"]}
    labels.update({"crowd_community_atmosphere": "Community & Atmosphere",
                   "training_preparation_pacing": "Training, Preparation & Pacing",
                   "race_performance": "Race Performance", "physical_experience": "Physical Experience"})
    if set(aspects) != set(eligibility.aspect): raise RuntimeError("Ontology/aspect mart mismatch")
    root.mkdir(parents=True)

    corpus = {
        "total_documents": EXPECTED_DOCUMENTS, "mention_bearing_documents": EXPECTED_BEARING,
        "zero_mention_documents": EXPECTED_ZERO, "zero_mention_rate": EXPECTED_ZERO/EXPECTED_DOCUMENTS,
        "total_mentions": EXPECTED_MENTIONS, "mean_mentions_per_document": summary["mean_mentions_per_document"],
        "median_mentions_per_document": summary["median_mentions_per_document"],
        "mean_mentions_per_mention_bearing_document": summary["mean_mentions_per_mention_bearing_document"],
        "median_mentions_per_mention_bearing_document": summary["median_mentions_per_mention_bearing_document"],
        "multi_aspect_documents": summary["multi_aspect_documents"],
        "multi_aspect_rate_all_documents": summary["multi_aspect_document_rate_all_documents"],
        "multi_aspect_rate_mention_bearing_documents": summary["multi_aspect_document_rate_mention_bearing_documents"],
        "mean_unique_aspects_per_document": summary["mean_unique_aspects_per_document"],
    }
    (root / "dashboard_corpus_overview_v1.json").write_text(json.dumps(corpus, indent=2), encoding="utf-8")

    overall = pd.read_csv(sources["sentiment_overall"]).rename(columns={
        "mention_share_all_mentions": "mention_share", "affected_document_count": "documents_containing_sentiment",
        "affected_document_share_all_documents": "document_share_all"})
    overall["document_share_mention_bearing"] = overall.documents_containing_sentiment / EXPECTED_BEARING
    overall.to_csv(root / "dashboard_sentiment_overall_v1.csv", index=False)

    prevalence = pd.read_csv(sources["aspect_prevalence"])
    support = pd.read_csv(sources["support"])[["aspect", "support_flag"]].rename(columns={"support_flag": "support_class"})
    aspect_sentiment = pd.read_csv(sources["aspect_sentiment"])
    sentiment_wide = aspect_sentiment.pivot(index="aspect", columns="sentiment", values=["mention_count", "within_aspect_sentiment_share"])
    sentiment_wide.columns = [f"{s}_{'mentions' if metric == 'mention_count' else 'share_within_aspect'}" for metric, s in sentiment_wide.columns]
    sentiment_wide = sentiment_wide.reset_index()
    inferential = pd.read_csv(sources["inferential_prevalence"]).drop_duplicates("aspect")
    inferential = inferential[["aspect", "adjusted_p", "effect_size", "significant_fdr"]].rename(columns={
        "adjusted_p": "year_association_adjusted_p", "effect_size": "cramers_v",
        "significant_fdr": "year_association_fdr_significant"})
    aspect_overview = prevalence.rename(columns={"mention_share_all_mentions": "mention_share",
        "document_prevalence_all_documents": "document_prevalence_all",
        "document_prevalence_mention_bearing_documents": "document_prevalence_mention_bearing"})
    aspect_overview = aspect_overview.merge(sentiment_wide, on="aspect").merge(support, on="aspect").merge(inferential, on="aspect", how="left")
    aspect_overview["display_label"] = aspect_overview.aspect.map(labels)
    aspect_overview["inferential_eligible"] = aspect_overview.aspect.isin(eligible)
    aspect_overview["year_association_fdr_significant"] = aspect_overview.year_association_fdr_significant.astype("boolean")
    aspect_overview["effect_category"] = aspect_overview.cramers_v.map(_effect)
    aspect_overview["support_warning"] = aspect_overview.support_class.map(SUPPORT_WARNINGS)
    aspect_overview["recommended_for_emphasis"] = (aspect_overview.inferential_eligible &
        aspect_overview.year_association_fdr_significant.fillna(False) & aspect_overview.cramers_v.ge(.10) &
        aspect_overview.support_class.isin(["moderate_support", "high_support"]))
    aspect_overview["interpretation_note"] = np.where(aspect_overview.recommended_for_emphasis,
        "FDR-detectable year association with at least a small effect; suitable for restrained emphasis.",
        np.where(~aspect_overview.inferential_eligible, "Descriptive only; not in the frozen inferential set.",
                 "Do not promote solely on statistical significance; inspect effect magnitude and support."))
    aspect_columns = ["aspect", "display_label", "mention_count", "mention_share", "affected_document_count",
        "document_prevalence_all", "document_prevalence_mention_bearing", "mean_mentions_per_affected_document",
        "positive_mentions", "negative_mentions", "mixed_mentions", "neutral_mentions",
        "positive_share_within_aspect", "negative_share_within_aspect", "mixed_share_within_aspect",
        "neutral_share_within_aspect", "support_class", "support_warning", "inferential_eligible",
        "year_association_fdr_significant", "year_association_adjusted_p", "cramers_v", "effect_category",
        "recommended_for_emphasis", "interpretation_note"]
    aspect_overview = aspect_overview[aspect_columns].sort_values("document_prevalence_all", ascending=False)
    aspect_overview.to_csv(root / "dashboard_aspect_overview_v1.csv", index=False)

    aspect_sentiment_mart = aspect_sentiment.rename(columns={"within_aspect_sentiment_share": "share_within_aspect"})
    sentiment_docs = mentions.groupby(["aspect", "sentiment"]).document_id.nunique().rename("affected_document_count").reset_index()
    aspect_sentiment_mart = aspect_sentiment_mart.drop(columns=["affected_document_count"]).merge(sentiment_docs, on=["aspect", "sentiment"], how="left").merge(support, on="aspect")
    aspect_sentiment_mart["display_label"] = aspect_sentiment_mart.aspect.map(labels)
    aspect_sentiment_mart[["aspect", "display_label", "sentiment", "mention_count", "share_within_aspect", "affected_document_count", "support_class"]].to_csv(root / "dashboard_aspect_sentiment_v1.csv", index=False)

    year_source = pd.read_csv(sources["year_aspect_sentiment"])
    year_base = year_source[["event_year", "document_count", "mention_bearing_document_count", "zero_mention_document_count",
        "total_mentions", "mean_mentions_per_document", "mean_unique_aspects_per_document", "zero_mention_rate"]].drop_duplicates()
    year_sentiment = mentions.groupby(["event_year", "sentiment"]).size().unstack(fill_value=0).reindex(columns=SENTIMENTS, fill_value=0)
    year_overview = year_base.rename(columns={"event_year": "year", "document_count": "documents",
        "mention_bearing_document_count": "mention_bearing_documents",
        "zero_mention_document_count": "zero_mention_documents", "total_mentions": "mentions",
        "mean_mentions_per_document": "mentions_per_document"})
    for sentiment in SENTIMENTS:
        year_overview[f"{sentiment}_mentions"] = year_overview.year.map(year_sentiment[sentiment]).astype(int)
        year_overview[f"{sentiment}_share"] = year_overview[f"{sentiment}_mentions"] / year_overview.mentions
    year_overview["extraction_density_warning"] = WARNINGS["temporal"]
    year_overview.to_csv(root / "dashboard_year_overview_v1.csv", index=False)

    year_aspect = year_source.drop_duplicates(["event_year", "aspect"])[["event_year", "aspect", "document_count",
        "aspect_affected_document_count", "aspect_document_prevalence", "aspect_mention_count", "aspect_mention_share_within_year"]]
    year_aspect = year_aspect.rename(columns={"event_year": "year", "document_count": "total_documents_year",
        "aspect_affected_document_count": "affected_documents", "aspect_document_prevalence": "document_prevalence",
        "aspect_mention_count": "mention_count", "aspect_mention_share_within_year": "mention_share_within_year"})
    sentiment_doc = matrix[["aspect", "event_year", "positive_present", "negative_present", "mixed_present", "neutral_present"]].groupby(["aspect", "event_year"]).sum().reset_index().rename(columns={"event_year": "year"})
    sentiment_doc = sentiment_doc.rename(columns={f"{s}_present": f"{s}_document_count" for s in SENTIMENTS})
    year_aspect = year_aspect.merge(sentiment_doc, on=["aspect", "year"]).merge(support, on="aspect")
    for sentiment in ["positive", "negative"]:
        year_aspect[f"{sentiment}_document_share_within_aspect_year"] = np.where(year_aspect.affected_documents.gt(0), year_aspect[f"{sentiment}_document_count"] / year_aspect.affected_documents, np.nan)
    frozen_ci = pd.read_csv(sources["inferential_prevalence"])[["aspect", "year", "ci_lower", "ci_upper", "adjusted_p", "effect_size", "significant_fdr"]]
    year_aspect = year_aspect.merge(frozen_ci, on=["aspect", "year"], how="left").rename(columns={"ci_lower": "document_prevalence_ci_lower",
        "ci_upper": "document_prevalence_ci_upper", "adjusted_p": "omnibus_adjusted_p", "effect_size": "cramers_v"})
    year_aspect["inferential_eligible"] = year_aspect.aspect.isin(eligible)
    year_aspect["display_label"] = year_aspect.aspect.map(labels)
    year_aspect["effect_category"] = year_aspect.cramers_v.map(_effect)
    emphasis_map = aspect_overview.set_index("aspect").recommended_for_emphasis
    year_aspect["recommended_for_emphasis"] = year_aspect.aspect.map(emphasis_map)
    year_aspect.to_csv(root / "dashboard_year_aspect_v1.csv", index=False)

    yas = year_source[["event_year", "aspect", "sentiment", "mention_count", "within_year_aspect_sentiment_share"]].rename(columns={
        "event_year": "year", "within_year_aspect_sentiment_share": "share_within_aspect_year"})
    yas_docs = mentions.groupby(["event_year", "aspect", "sentiment"]).document_id.nunique().rename("document_count").reset_index().rename(columns={"event_year": "year"})
    yas = yas.merge(yas_docs, on=["year", "aspect", "sentiment"], how="left")
    ci_parts = []
    for sentiment, key in [("positive", "inferential_positive"), ("negative", "inferential_negative")]:
        part = pd.read_csv(sources[key])[["aspect", "year", "ci_lower", "ci_upper"]].copy(); part["sentiment"] = sentiment; ci_parts.append(part)
    yas = yas.merge(pd.concat(ci_parts), on=["year", "aspect", "sentiment"], how="left")
    yas["display_label"] = yas.aspect.map(labels)
    yas.to_csv(root / "dashboard_year_aspect_sentiment_v1.csv", index=False)

    pairwise = pd.read_csv(sources["pairwise"])
    outcome_map = {"A_aspect_prevalence": "aspect_prevalence", "B_positive_sentiment": "positive_presence",
                   "C_negative_sentiment": "negative_presence"}
    parent_effect = {}
    for family, key in [("A_aspect_prevalence", "inferential_prevalence"), ("B_positive_sentiment", "inferential_positive"), ("C_negative_sentiment", "inferential_negative")]:
        table = pd.read_csv(sources[key]).drop_duplicates("aspect")
        parent_effect.update({(family, row.aspect): row.effect_size for row in table.itertuples()})
    pair_out = pd.DataFrame({"aspect": pairwise.aspect, "outcome": pairwise.analysis_family.map(outcome_map),
        "year_a": pairwise.year_1, "year_b": pairwise.year_2, "prevalence_a": pairwise.prevalence_1,
        "prevalence_b": pairwise.prevalence_2, "percentage_point_difference": pairwise.difference * 100,
        "odds_ratio": pairwise.odds_ratio_year_2_vs_year_1, "ci_lower": pairwise.ci_lower,
        "ci_upper": pairwise.ci_upper, "raw_p": pairwise.raw_p, "holm_adjusted_p": pairwise.adjusted_p,
        "significant_holm": pairwise.significant_holm})
    effects = [parent_effect[(f, a)] for f, a in zip(pairwise.analysis_family, pairwise.aspect)]
    pair_out["display_priority"] = np.where(pair_out.significant_holm & (np.asarray(effects) >= .10), "high", "standard")
    pair_out["display_label"] = pair_out.aspect.map(labels)
    pair_out.to_csv(root / "dashboard_pairwise_year_v1.csv", index=False)

    topic = pd.read_csv(sources["topic_aspect_sentiment"])
    topic_group = documents[["final_topic_id", "final_topic_group"]].drop_duplicates()
    topic_long = topic.groupby(["final_topic_id", "final_topic_name", "aspect"], as_index=False).agg(
        document_count=("document_count", "first"), affected_documents=("aspect_affected_document_count", "first"),
        document_prevalence=("aspect_document_prevalence_within_topic", "first"), mention_count=("aspect_mention_count", "first"))
    topic_sent = topic.pivot_table(index=["final_topic_id", "final_topic_name", "aspect"], columns="sentiment", values="within_topic_aspect_sentiment_share", aggfunc="first").reset_index()
    topic_long = topic_long.merge(topic_sent, on=["final_topic_id", "final_topic_name", "aspect"]).rename(columns={"final_topic_id": "topic_id", "final_topic_name": "topic_label", **{s: f"{s}_share" for s in SENTIMENTS}})
    topic_long["topic_id"] = topic_long.topic_id.astype(int)
    topic_long["aspect_display_label"] = topic_long.aspect.map(labels)
    topic_long.to_csv(root / "dashboard_topic_aspect_long_v1.csv", index=False)
    topic_mentions = mentions.groupby("final_topic_id").size(); topic_sentiments = mentions.groupby(["final_topic_id", "sentiment"]).size().unstack(fill_value=0)
    overview_rows = []
    for (topic_id, topic_label), group in topic_long.groupby(["topic_id", "topic_label"]):
        ranked = group.sort_values(["document_prevalence", "mention_count"], ascending=False).head(3)
        row = {"topic_id": topic_id, "topic_label": topic_label,
               "topic_group": topic_group.loc[topic_group.final_topic_id.eq(topic_id), "final_topic_group"].iloc[0],
               "document_count": int(group.document_count.iloc[0]), "mention_count": int(topic_mentions.get(topic_id, 0))}
        for i, item in enumerate(ranked.itertuples(), 1): row[f"top_aspect_{i}"], row[f"top_aspect_{i}_document_prevalence"] = item.aspect, item.document_prevalence
        for sentiment in SENTIMENTS: row[f"{sentiment}_share"] = topic_sentiments.get(sentiment, pd.Series()).get(topic_id, 0) / row["mention_count"] if row["mention_count"] else 0
        overview_rows.append(row)
    topic_overview = pd.DataFrame(overview_rows)
    if topic_overview.document_count.sum() != EXPECTED_DOCUMENTS: raise RuntimeError("Topic documents do not reconcile")
    topic_overview.to_csv(root / "dashboard_topic_overview_v1.csv", index=False)

    language = pd.read_csv(sources["language_aspect_sentiment"])
    language_long = language[["primary_language", "aspect", "document_count", "aspect_affected_document_count",
        "aspect_document_prevalence_within_language", "aspect_mention_count", "sentiment", "within_language_aspect_sentiment_share"]].rename(columns={
        "primary_language": "language", "aspect_affected_document_count": "affected_documents",
        "aspect_document_prevalence_within_language": "document_prevalence", "aspect_mention_count": "mention_count",
        "within_language_aspect_sentiment_share": "sentiment_share"})
    language_long["inferential_comparison_allowed"] = False
    language_long["aspect_display_label"] = language_long.aspect.map(labels)
    language_long.to_csv(root / "dashboard_language_aspect_long_v1.csv", index=False)
    lang_base = language[["primary_language", "document_count", "mention_bearing_document_count", "zero_mention_rate", "language_mention_count"]].drop_duplicates()
    lang_sent = mentions.groupby(["primary_language", "sentiment"]).size().unstack(fill_value=0)
    lang_aspect = language.drop_duplicates(["primary_language", "aspect"]).sort_values(["primary_language", "aspect_document_prevalence_within_language"], ascending=[True, False]).groupby("primary_language").first()
    lang_rows = []
    for row in lang_base.itertuples():
        total_mentions = int(row.language_mention_count); top = lang_aspect.loc[row.primary_language]
        item = {"language": row.primary_language, "documents": int(row.document_count),
                "mention_bearing_documents": int(row.mention_bearing_document_count), "zero_mention_rate": row.zero_mention_rate,
                "mentions": total_mentions, "top_aspect": top.aspect,
                "top_aspect_display_label": labels[top.aspect],
                "top_aspect_document_prevalence": top.aspect_document_prevalence_within_language,
                "inferential_comparison_allowed": False, "language_quality_warning": row.primary_language in {"insufficient_text", "undetermined"}}
        for sentiment in SENTIMENTS: item[f"{sentiment}_share"] = lang_sent.get(sentiment, pd.Series()).get(row.primary_language, 0)/total_mentions if total_mentions else 0
        lang_rows.append(item)
    language_overview = pd.DataFrame(lang_rows)
    if language_overview.documents.sum() != EXPECTED_DOCUMENTS: raise RuntimeError("Language documents do not reconcile")
    language_overview.to_csv(root / "dashboard_language_overview_v1.csv", index=False)

    a = aspect_overview.set_index("aspect")
    ya = year_aspect.set_index(["aspect", "year"])
    findings = []
    headlines = {
        "race_performance": "Race-performance discussion became more prevalent across observed editions.",
        "crowd_community_atmosphere": "Community/atmosphere mentions increased across observed editions.",
        "physical_experience": "Physical-experience mentions became more prevalent in later editions.",
    }
    for index, aspect in enumerate(["race_performance", "crowd_community_atmosphere", "physical_experience"], 1):
        findings.append({"finding_id": f"prevalence_{index}", "finding_type": "aspect_prevalence", "aspect": aspect,
            "metric": "document_prevalence", "headline": headlines[aspect],
            "short_interpretation": "A supported association across observed editions in model-estimated labels; not a causal claim.",
            **{f"{year}_value": ya.loc[(aspect, year), "document_prevalence"] for year in YEARS},
            "adjusted_p": a.loc[aspect, "year_association_adjusted_p"], "effect_size": a.loc[aspect, "cramers_v"],
            "effect_category": a.loc[aspect, "effect_category"], "support_class": a.loc[aspect, "support_class"],
            "recommended_for_emphasis": True, "caution_note": WARNINGS["temporal"]})
    pos = pd.read_csv(sources["inferential_positive"]); neg = pd.read_csv(sources["inferential_negative"])
    tp_pos = pos[pos.aspect.eq("training_preparation_pacing")].set_index("year")
    tp_neg = neg[neg.aspect.eq("training_preparation_pacing")].set_index("year")
    findings.append({"finding_id": "training_sentiment_1", "finding_type": "aspect_sentiment", "aspect": "training_preparation_pacing",
        "metric": "positive_document_share_with_negative_context", "headline": "Training/preparation sentiment shifted toward more positive and less negative discussion.",
        "short_interpretation": "Positive presence rose while negative presence declined among training/preparation-bearing documents.",
        **{f"{year}_value": tp_pos.loc[year, "prevalence"] for year in YEARS}, "adjusted_p": tp_pos.adjusted_p.iloc[0],
        "effect_size": tp_pos.effect_size.iloc[0], "effect_category": _effect(tp_pos.effect_size.iloc[0]),
        "support_class": a.loc["training_preparation_pacing", "support_class"], "recommended_for_emphasis": True,
        "caution_note": "Values show positive presence; negative shares are available in the sentiment mart. Documents may contain both."})
    pd.DataFrame(findings).to_csv(root / "dashboard_research_findings_v1.csv", index=False)

    low = ", ".join(a[a.support_class.isin(["low_support", "very_low_support"])].index.tolist())
    caution = [
        {"item": "photography_media", "reason": "Omnibus association is statistically detectable but effect is trivial and pairwise contrasts are not robust.", "dashboard_behavior": "Show descriptively; do not headline."},
        {"item": "weather_conditions", "reason": "Large sentiment variation but low yearly support and irregular pattern.", "dashboard_behavior": "Show only with caution badge."},
        {"item": "low_and_very_low_support_aspects", "reason": f"Limited affected-document support: {low}.", "dashboard_behavior": "Retain in explorers; never auto-promote."},
        {"item": "neutral_and_mixed_temporal_outcomes", "reason": "Sparse outcomes were not formally tested.", "dashboard_behavior": "Show descriptively without inferential badges."},
        {"item": "language_comparisons", "reason": WARNINGS["language"], "dashboard_behavior": "Descriptive explorer only."},
        {"item": "topic_wide_inferential_comparisons", "reason": WARNINGS["topic"], "dashboard_behavior": "Descriptive explorer only."},
    ]
    pd.DataFrame(caution).to_csv(root / "dashboard_caution_findings_v1.csv", index=False)

    metadata = {"dataset_version": "absa_v1_dashboard_data_v1", "created_at": created_at,
        "production_documents": EXPECTED_DOCUMENTS, "production_mentions": EXPECTED_MENTIONS, "years": YEARS,
        "aspect_families": aspects, "aspect_display_labels": labels, "sentiment_labels": SENTIMENTS,
        "support_thresholds": SUPPORT_THRESHOLDS, "effect_size_thresholds": {"trivial": "V < 0.10", "small": "0.10 <= V < 0.20", "moderate": "0.20 <= V < 0.30", "strong": "V >= 0.30"},
        "inferential_unit": "document", "descriptive_mention_unit": "mention", "proportion_representation": "numeric 0-1",
        "model_precision": .513, "model_recall": .790, "development_precision_target": .55,
        "development_target_met": False, "production_is_validation": False, "language_inference_allowed": False,
        "topic_inference_allowed": False, "document_counts_can_overlap_by_sentiment": True,
        "inferential_eligible_aspects": eligible, "warnings": WARNINGS, "support_warnings": SUPPORT_WARNINGS,
        "frozen_frontend_policy": "All hypothesis testing, confidence intervals, multiple-testing correction, effect sizes, and research-emphasis decisions are frozen preprocessing outputs. The eventual dashboard is a presentation/exploration layer only."}
    (root / "dashboard_metadata_v1.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    chart_spec = {"version": "dashboard_chart_spec_v1", "charts": [
        {"id": "overview_sentiment", "type": "donut_or_stacked_bar", "source": "dashboard_sentiment_overall_v1.csv", "metric": "mention_share"},
        {"id": "aspect_prevalence", "type": "horizontal_bar", "source": "dashboard_aspect_overview_v1.csv", "metric": "document_prevalence_all", "sort": "descending"},
        {"id": "aspect_sentiment", "type": "100_percent_stacked_horizontal_bar", "source": "dashboard_aspect_sentiment_v1.csv", "metric": "share_within_aspect"},
        {"id": "year_prevalence_trends", "type": "line", "source": "dashboard_year_aspect_v1.csv", "metric": "document_prevalence", "include_ci": True},
        {"id": "aspect_explorer", "filter": "aspect", "sources": ["dashboard_year_aspect_v1.csv", "dashboard_aspect_sentiment_v1.csv", "dashboard_pairwise_year_v1.csv"], "charts": ["yearly_document_prevalence", "sentiment_composition", "pairwise_significant_changes"]},
        {"id": "topic_explorer", "type": "explorer", "source": "dashboard_topic_aspect_long_v1.csv", "inferential": False},
        {"id": "language_explorer", "type": "explorer", "source": "dashboard_language_aspect_long_v1.csv", "inferential": False},
    ]}
    (root / "dashboard_chart_spec_v1.json").write_text(json.dumps(chart_spec, indent=2), encoding="utf-8")

    top = aspect_overview.head(5)
    report = f"""# ABSA V1 dashboard data report\n\n## Corpus\n\nThe frozen corpus contains {EXPECTED_DOCUMENTS:,} documents, {EXPECTED_BEARING:,} mention-bearing documents, {EXPECTED_ZERO:,} zero-mention documents and {EXPECTED_MENTIONS:,} model-estimated mentions.\n\n## Primary dashboard aspects\n\nTop aspects by all-document prevalence: {', '.join(f'{r.display_label} ({r.document_prevalence_all:.1%})' for r in top.itertuples())}.\n\n## Overall sentiment\n\nModel-estimated mention distribution: {', '.join(f'{r.sentiment} {r.mention_share:.1%}' for r in overall.itertuples())}. Document sentiment counts overlap and must not be summed to 100%.\n\n## Temporal findings\n\nEmphasis-worthy prevalence findings are restricted deterministically to Race Performance, Community & Atmosphere, and Physical Experience: each is inferentially eligible, FDR-significant, adequately supported and has Cramer's V >= .10.\n\n## Training/preparation sentiment result\n\nPositive presence among training/preparation-bearing documents increases from {tp_pos.loc[2019, 'prevalence']:.1%} in 2019 to {tp_pos.loc[2025, 'prevalence']:.1%} in 2025; negative presence changes from {tp_neg.loc[2019, 'prevalence']:.1%} to {tp_neg.loc[2025, 'prevalence']:.1%}. Both frozen omnibus families are FDR-significant with small effects.\n\n## Cautionary findings\n\nWeather sentiment has large but irregular variation and limited yearly support. Photography/media has a trivial omnibus effect and no robust pairwise contrasts. Low-support aspects remain visible but cannot be promoted. Topic and language views are descriptive only; neutral/mixed temporal outcomes are not tested.\n\n## Available dashboard filters\n\nYear, aspect, sentiment, final topic, language and support class. Raw identifiers and stable display labels are retained.\n\n## Statistical interpretation\n\nUse `recommended_for_emphasis`, `effect_category`, support fields, inferential-status flags, adjusted p-values and frozen confidence intervals directly. Do not calculate tests or research emphasis in the frontend. {WARNINGS['model']} {WARNINGS['statistical']}\n\nOpenAI calls = 0; Hugging Face inference = 0; ABSA inference = 0; manual review = 0; new statistical hypothesis tests = 0.\n"""
    (root / "ABSA_V1_DASHBOARD_DATA_REPORT.md").write_text(report, encoding="utf-8")

    # Cross-mart safeguards.
    if overall.mention_count.sum() != EXPECTED_MENTIONS or aspect_overview.mention_count.sum() != EXPECTED_MENTIONS or year_overview.mentions.sum() != EXPECTED_MENTIONS:
        raise RuntimeError("Mention totals do not reconcile across marts")
    if len(aspect_overview) != 20 or len(year_aspect) != 80 or set(year_aspect.year) != set(YEARS):
        raise RuntimeError("Aspect/year mart dimensions changed")
    if not year_aspect.groupby("year").total_documents_year.nunique().eq(1).all(): raise RuntimeError("Year denominators conflict")
    if bool(a.loc["photography_media", "recommended_for_emphasis"]): raise RuntimeError("Photography cannot be emphasized")
    if not any(row["item"] == "weather_conditions" for row in caution): raise RuntimeError("Weather caution missing")
    artifacts = [path for path in root.iterdir() if path.is_file()]
    manifest = {"protocol": "absa_v1_dashboard_data_v1", "created_at": created_at,
        "source_artifacts": {name: {"path": str(sources[name]), "sha256": source_hashes[name]} for name in sources},
        "reconciliation": {"documents": EXPECTED_DOCUMENTS, "mention_bearing_documents": EXPECTED_BEARING,
            "zero_mention_documents": EXPECTED_ZERO, "mentions": EXPECTED_MENTIONS, "year_counts": expected_years,
            "aspects": len(aspects), "eligible_aspects": eligible},
        "execution": {"openai_calls": 0, "hugging_face_inference": 0, "absa_inference": 0,
            "manual_review": 0, "new_statistical_hypothesis_tests": 0, "production_predictions_modified": False},
        "python_version": platform.python_version(), "output_artifacts": {path.name: {"sha256": _sha(path)} for path in artifacts}}
    (root / "dashboard_data_manifest_v1.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if {name: _sha(path) for name, path in sources.items()} != source_hashes: raise RuntimeError("Frozen source changed")
    return {"output_root": str(root), "documents": EXPECTED_DOCUMENTS, "mentions": EXPECTED_MENTIONS,
            "eligible_aspects": eligible, "recommended_for_emphasis": a[a.recommended_for_emphasis].index.tolist(),
            "artifacts": len(list(root.iterdir())), "api_calls": 0, "new_statistical_tests": 0}
