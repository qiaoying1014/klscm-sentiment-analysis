from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import pandas as pd

from .absa_v1 import GOLD_PATH, ONTOLOGY_PATH, ROOT, SAMPLE_PATH


PROTOCOL = "absa_v1_false_positive_diagnostic_v1"
TAXONOMY_VERSION = "absa_v1_fp_error_taxonomy_v1"
SEED = 20260819
FP_SAMPLE_SIZE = 45
TP_SAMPLE_SIZE = 15
DEVELOPMENT_ROOT = ROOT / "development" / PROTOCOL
PREDICTIONS_PATH = ROOT / "absa_validation_predictions_v1.csv"
EVALUATION_PATH = ROOT / "absa_validation_evaluation_v1.json"

FP_CATEGORIES = [
    "factual_as_evaluative", "context_as_evaluative", "over_decomposition",
    "aspect_inference_too_far", "sentiment_spillover", "ontology_boundary_confusion",
    "duplicate_redundant_mention", "grounding_related", "other_unclear",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    for path in [SAMPLE_PATH, GOLD_PATH, PREDICTIONS_PATH, EVALUATION_PATH, ONTOLOGY_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required frozen artifact is missing: {path}")
    sample = pd.read_csv(SAMPLE_PATH).fillna("")
    gold = pd.read_csv(GOLD_PATH).fillna("")
    predictions = pd.read_csv(PREDICTIONS_PATH).fillna("")
    evaluation = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    for frame in [sample, gold, predictions]:
        frame["document_id"] = frame.document_id.astype(str)
    expected = evaluation["metrics"]["aspect_detection"]
    if (len(sample), sample.document_id.nunique(), len(gold), gold.document_id.nunique(),
            len(predictions), predictions.document_id.nunique()) != (80, 80, 100, 56, 218, 65):
        raise ValueError("Frozen ABSA validation population does not match the declared baseline")
    if (expected["tp"], expected["fp"], expected["fn"]) != (84, 134, 16):
        raise ValueError("Frozen evaluation TP/FP/FN totals do not match the declared baseline")
    return sample, gold, predictions, evaluation


def reconstruct_matching(gold: pd.DataFrame, predictions: pd.DataFrame
                         ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply the frozen one-to-one document+aspect, stable-mention-ID matching rule."""
    gold = gold.copy(); predictions = predictions.copy()
    gold["document_id"] = gold.document_id.astype(str)
    predictions["document_id"] = predictions.document_id.astype(str)
    tp_indices: list[int] = []; fp_indices: list[int] = []; fn_indices: list[int] = []
    keys = sorted(set(zip(gold.document_id, gold.aspect)) | set(zip(predictions.document_id, predictions.aspect)))
    for document_id, aspect in keys:
        gold_group = gold[(gold.document_id == document_id) & (gold.aspect == aspect)].sort_values("mention_id")
        pred_group = predictions[(predictions.document_id == document_id) & (predictions.aspect == aspect)].sort_values("mention_id")
        matched = min(len(gold_group), len(pred_group))
        tp_indices.extend(pred_group.index[:matched]); fp_indices.extend(pred_group.index[matched:])
        fn_indices.extend(gold_group.index[matched:])
    tp = predictions.loc[tp_indices].copy(); tp["matching_status"] = "TP"
    fp = predictions.loc[fp_indices].copy(); fp["matching_status"] = "FP"
    fn = gold.loc[fn_indices].copy(); fn["matching_status"] = "FN"
    if (len(tp), len(fp), len(fn)) != (84, 134, 16):
        raise ValueError(f"Reconstructed totals differ from frozen baseline: {len(tp)}/{len(fp)}/{len(fn)}")
    return tp, fp, fn


def _document_context(sample: pd.DataFrame, gold: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    gold_groups = gold.groupby("document_id", sort=False)
    pred_groups = predictions.groupby("document_id", sort=False)
    rows = []
    for row in sample.itertuples(index=False):
        document_id = str(row.document_id)
        g = gold_groups.get_group(document_id) if document_id in gold_groups.groups else gold.iloc[0:0]
        p = pred_groups.get_group(document_id) if document_id in pred_groups.groups else predictions.iloc[0:0]
        rows.append({"document_id": document_id, "gold_document_has_mentions": bool(len(g)),
            "gold_document_mention_count": len(g), "gold_aspects": "|".join(g.aspect.astype(str)),
            "gold_sentiments": "|".join(g.sentiment.astype(str)), "gold_unique_aspect_count": g.aspect.nunique(),
            "predicted_document_mention_count": len(p), "predicted_document_unique_aspect_count": p.aspect.nunique(),
            "is_gold_zero_document": not bool(len(g)), "is_predicted_multi_aspect_document": len(p) > 1,
            "predicted_aspects": "|".join(p.aspect.astype(str)),
            "predicted_sentiments": "|".join(p.sentiment.astype(str))})
    return pd.DataFrame(rows)


def _enrich_predictions(frame: pd.DataFrame, sample: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    metadata = sample.rename(columns={"original_text": "raw_caption", "event_year": "year",
        "primary_language": "recorded_language", "original_topic_id": "raw_c1_topic_id",
        "final_topic_name": "final_topic_label"})
    keep = [column for column in ["document_id", "source", "year", "recorded_language", "raw_caption",
        "raw_c1_topic_id", "final_topic_id", "final_topic_label", "final_topic_group"] if column in metadata]
    enriched = frame.merge(metadata[keep], on="document_id", how="left", suffixes=("", "_sample"), validate="many_to_one")
    enriched = enriched.merge(context, on="document_id", how="left", validate="many_to_one")
    enriched["prediction_mention_index"] = enriched.groupby("document_id").cumcount()
    enriched["diagnostic_row_id"] = [f"FP-{i:04d}" for i in range(1, len(enriched) + 1)]
    enriched["source_record_id"] = ""
    enriched["normalized_caption_if_existing"] = ""
    enriched["raw_c1_topic_label"] = ""
    enriched["predicted_aspect"] = enriched.aspect
    enriched["predicted_sentiment"] = enriched.sentiment
    enriched["predicted_language"] = enriched.get("language", "")
    enriched["language"] = enriched.recorded_language
    enriched["predicted_evidence"] = enriched.get("original_model_evidence_text", enriched.evidence_text)
    enriched["predicted_evidence_start"] = enriched.evidence_start
    enriched["predicted_evidence_end"] = enriched.evidence_end
    enriched["predicted_explicit_implicit"] = enriched.expression_type
    enriched["grounding_status"] = enriched.get("raw_exact_match", False).map(
        lambda value: "strict_exact" if bool(value) else "raw_mismatch")
    enriched["recoverable_grounding_status"] = enriched.get("evidence_grounding_valid", False).map(
        lambda value: "recoverable_exact" if bool(value) else "unrecoverable")
    return enriched


def _matching_counts_by_aspect(ontology_ids: list[str], gold: pd.DataFrame, predictions: pd.DataFrame,
                               tp: pd.DataFrame, fp: pd.DataFrame, fn: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for aspect in ontology_ids:
        tp_n = int((tp.aspect == aspect).sum()); fp_n = int((fp.aspect == aspect).sum())
        fn_n = int((fn.aspect == aspect).sum()); predicted = int((predictions.aspect == aspect).sum())
        gold_n = int((gold.aspect == aspect).sum())
        rows.append({"aspect": aspect, "fp_count": fp_n, "fp_percentage": fp_n / len(fp) if len(fp) else 0,
            "affected_documents": int(fp.loc[fp.aspect.eq(aspect), "document_id"].nunique()),
            "tp_count": tp_n, "fn_count": fn_n, "aspect_precision": tp_n / predicted if predicted else None,
            "aspect_recall": tp_n / gold_n if gold_n else None, "predicted_mention_count": predicted,
            "gold_mention_count": gold_n})
    return sorted(rows, key=lambda row: (-row["fp_count"], row["aspect"]))


def _group_diagnostics(group_name: str, sample: pd.DataFrame, tp: pd.DataFrame, fp: pd.DataFrame,
                       fn: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for value, documents in sample.groupby(group_name).document_id.apply(set).items():
        tp_n = int(tp.document_id.isin(documents).sum()); fp_n = int(fp.document_id.isin(documents).sum())
        fn_n = int(fn.document_id.isin(documents).sum())
        rows.append({group_name: str(value), "document_count": len(documents), "fp_count": fp_n,
            "fp_per_document": fp_n / len(documents), "tp_count": tp_n, "fn_count": fn_n,
            "aspect_precision": tp_n / (tp_n + fp_n) if tp_n + fp_n else None,
            "aspect_recall": tp_n / (tp_n + fn_n) if tp_n + fn_n else None})
    return sorted(rows, key=lambda row: (-row["fp_count"], row[group_name]))


def _greedy_sample(frame: pd.DataFrame, size: int, seed: int, prefix: str) -> pd.DataFrame:
    rng = random.Random(seed); work = frame.copy()
    work["_tie"] = [rng.random() for _ in range(len(work))]
    dimensions = [column for column in ["predicted_aspect", "language", "final_topic_label",
        "predicted_sentiment", "grounding_status", "is_gold_zero_document",
        "is_predicted_multi_aspect_document"] if column in work]
    uncovered = {(column, str(value)) for column in dimensions for value in work[column].unique()}
    selected: list[int] = []; document_counts: Counter[str] = Counter()
    while len(selected) < min(size, len(work)):
        candidates = work.loc[~work.index.isin(selected) & work.document_id.map(lambda x: document_counts[str(x)] < 2)].copy()
        if candidates.empty:
            candidates = work.loc[~work.index.isin(selected)].copy()
        def score(row: pd.Series) -> tuple[float, float]:
            coverage = sum((column, str(row[column])) in uncovered for column in dimensions)
            return coverage - document_counts[str(row.document_id)] * 0.75, row._tie
        index = max(candidates.index, key=lambda idx: score(candidates.loc[idx]))
        selected.append(index); document_counts[str(work.loc[index, "document_id"])] += 1
        for column in dimensions: uncovered.discard((column, str(work.loc[index, column])))
    result = work.loc[selected].drop(columns="_tie").reset_index(drop=True)
    result["review_row_id"] = [f"{prefix}-{i:03d}" for i in range(1, len(result) + 1)]
    return result


def _write_review_guide(path: Path) -> None:
    definitions = {
        "factual_as_evaluative": "A factual statement, result, distance, time, location, participation fact, training fact, registration fact, or product/sponsor reference was treated as evaluative without an expressed evaluation.",
        "context_as_evaluative": "Event background or context was converted into an aspect mention despite lacking evaluation of that aspect.",
        "over_decomposition": "One underlying evaluative statement was expanded into more aspect families than are directly supported.",
        "aspect_inference_too_far": "The predicted aspect requires inference beyond what the caption sufficiently expresses or grounds.",
        "sentiment_spillover": "Overall positive or negative affect was transferred to an aspect mentioned neutrally.",
        "ontology_boundary_confusion": "Evaluative content exists, but the predicted family is not the supported aspect under the frozen ontology.",
        "duplicate_redundant_mention": "Multiple predictions represent essentially the same aspect evaluation without justification.",
        "grounding_related": "Unsupported, malformed, or unrecoverable evidence contributes materially to the false positive.",
        "other_unclear": "Use only when none of the other categories reasonably describes the failure.",
    }
    lines = ["# ABSA V1 False-Positive Diagnostic Review Guide", "", f"Protocol: `{PROTOCOL}`. Development evidence only.", "",
        "> The purpose of this audit is to identify generalizable error mechanisms for one controlled precision-focused prompt revision. It is not to hand-engineer rules for individual validation captions.", "",
        "## Review procedure", "", "1. Review the original raw caption.",
        "2. Inspect the predicted aspect, sentiment, and evidence.",
        "3. Treat the frozen human gold as authoritative.", "4. Assign exactly one primary FP category.",
        "5. Do not change the gold or redesign the ontology.",
        "6. Focus on why the model generated an unsupported mention.",
        "7. For over-decomposition, check whether one evaluation became multiple related families.",
        "8. For sentiment spillover, check whether general affect attached to a neutral entity.",
        "9. For factual-as-evaluative, ask whether removing affect leaves only factual event information.",
        "10. Use other_unclear sparingly.",
        "11. Preserve multilingual text and judge its original-language meaning where possible.", "", "## Frozen categories", ""]
    for category, definition in definitions.items(): lines.extend([f"### `{category}`", "", definition, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def create_fp_diagnostic(output_dir: Path = DEVELOPMENT_ROOT) -> dict[str, Any]:
    sample, gold, predictions, evaluation = _load_inputs()
    source_hashes = {"validation_sample": _sha256(SAMPLE_PATH), "gold": _sha256(GOLD_PATH),
                     "predictions": _sha256(PREDICTIONS_PATH), "evaluation": _sha256(EVALUATION_PATH),
                     "ontology": _sha256(ONTOLOGY_PATH)}
    manifest_path = output_dir / "absa_v1_fp_diagnostic_manifest_v1.json"
    if output_dir.exists():
        if not manifest_path.exists(): raise FileExistsError("Partial FP diagnostic directory exists; refusing overwrite")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("source_hashes") != source_hashes:
            raise FileExistsError("Existing FP diagnostic uses different frozen inputs; refusing overwrite")
        return existing
    output_dir.mkdir(parents=True)
    tp, fp, fn = reconstruct_matching(gold, predictions)
    context = _document_context(sample, gold, predictions)
    fp_full = _enrich_predictions(fp, sample, context)
    tp_full = _enrich_predictions(tp, sample, context)
    fp_full["diagnostic_row_id"] = [f"FP-{i:04d}" for i in range(1, len(fp_full) + 1)]
    tp_full["diagnostic_row_id"] = [f"TP-{i:04d}" for i in range(1, len(tp_full) + 1)]
    fp_path = output_dir / "absa_v1_fp_diagnostic_mentions_v1.csv"
    fp_full.to_csv(fp_path, index=False, encoding="utf-8-sig")
    fp_by_doc = fp_full.groupby("document_id").size()
    aspect_ids = [aspect["id"] for aspect in json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8"))["aspects"]]
    by_aspect = _matching_counts_by_aspect(aspect_ids, gold, predictions, tp, fp, fn)
    zero_ids = set(sample.document_id) - set(gold.document_id)
    predicted_ids = set(predictions.document_id)
    zero_predictions = predictions[predictions.document_id.isin(zero_ids)]
    gold_unique = gold.groupby("document_id").aspect.nunique().reindex(sample.document_id, fill_value=0)
    pred_unique = predictions.groupby("document_id").aspect.nunique().reindex(sample.document_id, fill_value=0)
    expansion = pd.DataFrame({"document_id": sample.document_id, "gold_unique_aspects": gold_unique.values,
                              "predicted_unique_aspects": pred_unique.values})
    expansion["unique_aspect_difference"] = expansion.predicted_unique_aspects - expansion.gold_unique_aspects
    expansion = expansion.merge(sample[["document_id", "primary_language", "final_topic_name"]], on="document_id")
    multi_ids = set(context.loc[context.is_predicted_multi_aspect_document, "document_id"])
    fp_multi = int(fp.document_id.isin(multi_ids).sum()); pred_multi = int(predictions.document_id.isin(multi_ids).sum())
    single_ids = set(predictions.document_id) - multi_ids
    fp_single = int(fp.document_id.isin(single_ids).sum()); pred_single = int(predictions.document_id.isin(single_ids).sum())
    language_sample = sample.rename(columns={"primary_language": "language"})
    by_language = _group_diagnostics("language", language_sample, tp, fp, fn)
    english_ids = set(sample.loc[sample.primary_language.eq("English"), "document_id"])
    for label, ids in [("non-English/uncertain", set(sample.document_id) - english_ids)]:
        tp_n = int(tp.document_id.isin(ids).sum()); fp_n = int(fp.document_id.isin(ids).sum()); fn_n = int(fn.document_id.isin(ids).sum())
        by_language.append({"language": label, "document_count": len(ids), "fp_count": fp_n,
            "fp_per_document": fp_n / len(ids), "tp_count": tp_n, "fn_count": fn_n,
            "aspect_precision": tp_n / (tp_n + fp_n), "aspect_recall": tp_n / (tp_n + fn_n)})
    topic_rows = []
    for topic, docs in sample.groupby("final_topic_name").document_id.apply(set).items():
        topic_fp = fp_full[fp_full.document_id.isin(docs)]
        topic_rows.append({"final_topic_label": topic, "validation_document_count": len(docs),
            "fp_count": len(topic_fp), "affected_document_count": topic_fp.document_id.nunique(),
            "fp_per_document": len(topic_fp) / len(docs),
            "most_common_fp_aspects": dict(Counter(topic_fp.predicted_aspect).most_common(5))})
    grounding_fp = {"strict_grounded": int(fp_full.grounding_status.eq("strict_exact").sum()),
                    "recoverable_grounded": int(fp_full.recoverable_grounding_status.eq("recoverable_exact").sum()),
                    "unrecoverable": int(fp_full.recoverable_grounding_status.eq("unrecoverable").sum())}
    grounding_tp = {"strict_grounded": int(tp_full.grounding_status.eq("strict_exact").sum()),
                    "recoverable_grounded": int(tp_full.recoverable_grounding_status.eq("recoverable_exact").sum()),
                    "unrecoverable": int(tp_full.recoverable_grounding_status.eq("unrecoverable").sum())}
    summary = {"protocol_name": PROTOCOL, "development_evidence": True,
        "baseline_reconciliation": {"tp": len(tp), "fp": len(fp), "fn": len(fn),
            "gold_mentions": len(gold), "gold_mention_documents": gold.document_id.nunique(),
            "gold_zero_documents": len(zero_ids), "predicted_mentions": len(predictions),
            "predicted_mention_documents": predictions.document_id.nunique(),
            "predicted_zero_documents": len(set(sample.document_id) - predicted_ids)},
        "overall": {"total_fp_mentions": len(fp), "affected_documents": fp.document_id.nunique(),
            "fp_per_affected_document_mean": mean(fp_by_doc), "fp_per_affected_document_median": median(fp_by_doc),
            "fp_per_affected_document_max": int(fp_by_doc.max()),
            "documents_by_fp_count": {"1": int((fp_by_doc == 1).sum()), "2": int((fp_by_doc == 2).sum()),
                                      "3": int((fp_by_doc == 3).sum()), "4+": int((fp_by_doc >= 4).sum())}},
        "by_aspect": by_aspect,
        "gold_zero_behavior": {"gold_zero_documents": len(zero_ids), "predicted_zero": len(zero_ids - predicted_ids),
            "receiving_predictions": len(zero_ids & predicted_ids), "predicted_mentions": len(zero_predictions),
            "fp_mentions": int(fp.document_id.isin(zero_ids).sum()),
            "fp_aspects": dict(Counter(fp.loc[fp.document_id.isin(zero_ids), "aspect"])),
            "predicted_sentiments": dict(Counter(zero_predictions.sentiment)),
            "documents_by_prediction_count": {"1": int((zero_predictions.groupby("document_id").size() == 1).sum()),
                "2": int((zero_predictions.groupby("document_id").size() == 2).sum()),
                "3+": int((zero_predictions.groupby("document_id").size() >= 3).sum())},
            "matrix": evaluation["metrics"]["document_diagnostics"]["zero_mention_confusion_gold_rows_predicted_columns"]},
        "over_extraction": {"gold_multi_aspect_documents": int((context.gold_document_mention_count > 1).sum()),
            "predicted_multi_aspect_documents": int((context.predicted_document_mention_count > 1).sum()),
            "average_gold_mentions_per_document": len(gold) / len(sample),
            "average_predicted_mentions_per_document": len(predictions) / len(sample),
            "fp_rate_predicted_multi_aspect_documents": fp_multi / pred_multi if pred_multi else None,
            "fp_rate_predicted_single_aspect_documents": fp_single / pred_single if pred_single else None,
            "documents_predicted_aspects_exceed_gold": int((expansion.unique_aspect_difference > 0).sum()),
            "highest_over_expansion_documents": expansion.sort_values(
                ["unique_aspect_difference", "predicted_unique_aspects"], ascending=False).head(15).to_dict("records")},
        "by_language": by_language, "by_final_topic": sorted(topic_rows, key=lambda row: (-row["fp_count"], row["final_topic_label"])),
        "fp_by_predicted_sentiment": dict(Counter(fp.predicted_sentiment if "predicted_sentiment" in fp else fp.sentiment)),
        "grounding_relationship": {"fp": grounding_fp, "tp": grounding_tp,
            "note": "Grounding correctness is not aspect correctness."}}
    summary_path = output_dir / "absa_v1_fp_diagnostic_summary_v1.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    report_path = output_dir / "absa_v1_fp_diagnostic_report_v1.md"
    top_aspects = ", ".join(f"{row['aspect']} ({row['fp_count']})" for row in by_aspect[:8])
    report_path.write_text("\n".join(["# ABSA V1 False-Positive Diagnostic", "", "Development/error-analysis artifact; the first frozen evaluation is unchanged.", "",
        "## Frozen reconciliation", "", f"TP **{len(tp)}**, FP **{len(fp)}**, FN **{len(fn)}**; {len(predictions)} predictions versus {len(gold)} gold mentions.", "",
        "## FP concentration", "", f"FPs affect **{fp.document_id.nunique()}** documents. Mean/median/max per affected document: {mean(fp_by_doc):.2f}/{median(fp_by_doc):.1f}/{int(fp_by_doc.max())}.",
        f"Highest-FP aspects: {top_aspects}.", "", "## Gold-zero behavior", "", f"Of 24 gold-zero documents, 13 were predicted zero and 11 received {len(zero_predictions)} mentions, all false positives.", "",
        "## Interpretation boundary", "", "These counts identify where errors concentrate; they do not assign causal FP categories. Causal mechanisms require completion of the stratified manual review."]), encoding="utf-8")
    fp_review = _greedy_sample(fp_full, FP_SAMPLE_SIZE, SEED, "FPREV")
    for column in ["fp_error_category", "fp_error_subcategory", "researcher_notes", "confidence", "reviewed"]: fp_review[column] = ""
    fp_review_path = output_dir / "absa_v1_fp_manual_review_sample_v1.csv"
    fp_review.to_csv(fp_review_path, index=False, encoding="utf-8-sig")
    tp_review = _greedy_sample(tp_full, TP_SAMPLE_SIZE, SEED + 1, "TPREV")
    for column in ["why_legitimate_evaluation", "researcher_notes", "reviewed"]: tp_review[column] = ""
    tp_path = output_dir / "absa_v1_tp_contrast_sample_v1.csv"; tp_review.to_csv(tp_path, index=False, encoding="utf-8-sig")
    fn_metadata = fn.merge(sample.rename(columns={"original_text":"raw_caption", "primary_language":"language",
        "final_topic_name":"final_topic_label"})[["document_id", "raw_caption", "language", "original_topic_id",
        "final_topic_id", "final_topic_label", "final_topic_group"]], on="document_id", how="left", validate="many_to_one")
    if "language_x" in fn_metadata:
        fn_metadata["language"] = fn_metadata.language_y.where(fn_metadata.language_y.astype(str).str.strip().ne(""),
                                                               fn_metadata.language_x)
        fn_metadata = fn_metadata.drop(columns=["language_x", "language_y"])
    fn_metadata = fn_metadata.merge(context[["document_id", "predicted_aspects", "predicted_sentiments"]], on="document_id", validate="many_to_one")
    fn_metadata = fn_metadata.rename(columns={"aspect":"gold_aspect", "sentiment":"gold_sentiment",
        "evidence_text":"gold_evidence", "expression_type":"gold_explicit_implicit"})
    fn_metadata["review_row_id"] = [f"FNREV-{i:03d}" for i in range(1, len(fn_metadata) + 1)]
    fn_metadata["researcher_notes"] = ""; fn_metadata["reviewed"] = ""
    fn_path = output_dir / "absa_v1_fn_contrast_sample_v1.csv"; fn_metadata.to_csv(fn_path, index=False, encoding="utf-8-sig")
    guide_path = output_dir / "ABSA_V1_FP_DIAGNOSTIC_REVIEW_GUIDE.md"; _write_review_guide(guide_path)
    coverage = {"selected_fp_mentions": len(fp_review), "unique_documents": fp_review.document_id.nunique(),
        "aspects": fp_review.predicted_aspect.nunique(), "languages": fp_review.language.value_counts().to_dict(),
        "topics": fp_review.final_topic_label.nunique(), "sentiments": fp_review.predicted_sentiment.value_counts().to_dict(),
        "gold_zero_rows": int(fp_review.is_gold_zero_document.sum()),
        "predicted_multi_aspect_rows": int(fp_review.is_predicted_multi_aspect_document.sum()),
        "raw_grounding_failures": int(fp_review.grounding_status.eq("raw_mismatch").sum())}
    module_path = Path(__file__)
    manifest = {"protocol_name": PROTOCOL, "created_at": datetime.now(timezone.utc).isoformat(),
        "source_paths": {"validation_sample": str(SAMPLE_PATH), "gold": str(GOLD_PATH),
            "predictions": str(PREDICTIONS_PATH), "evaluation": str(EVALUATION_PATH), "ontology": str(ONTOLOGY_PATH)},
        "source_hashes": source_hashes, "matching_evaluation_code_identity": _sha256(module_path),
        "manual_sample_seed": SEED, "manual_fp_sample_size": len(fp_review),
        "tp_contrast_sample_seed": SEED + 1, "tp_contrast_sample_size": len(tp_review),
        "fn_count": len(fn_metadata), "diagnostic_taxonomy_version": TAXONOMY_VERSION,
        "git_commit": None, "manual_sample_coverage": coverage,
        "first_frozen_evaluation_modified": False, "human_gold_modified": False,
        "ontology_modified": False, "model_modified": False, "prompt_modified": False,
        "matching_rules_modified": False, "paid_api_calls_performed": False, "development_evidence": True,
        "artifacts": {path.name: _sha256(path) for path in [fp_path, summary_path, report_path, fp_review_path,
                                                             tp_path, fn_path, guide_path]}}
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def finalize_fp_diagnostic(output_dir: Path = DEVELOPMENT_ROOT) -> dict[str, Any]:
    manifest_path = output_dir / "absa_v1_fp_diagnostic_manifest_v1.json"
    review_path = output_dir / "absa_v1_fp_manual_review_sample_v1.csv"
    if not manifest_path.exists() or not review_path.exists():
        raise FileNotFoundError("Create the FP diagnostic package before finalization")
    final_json = output_dir / "absa_v1_fp_diagnostic_final_v1.json"
    final_md = output_dir / "absa_v1_fp_diagnostic_final_v1.md"
    if final_json.exists() or final_md.exists():
        if final_json.exists() and final_md.exists(): return json.loads(final_json.read_text(encoding="utf-8"))
        raise FileExistsError("Partial final diagnostic exists; refusing overwrite")
    review = pd.read_csv(review_path).fillna("")
    reviewed = review.reviewed.astype(str).str.lower().isin(["true", "1", "yes"])
    if not reviewed.all(): raise ValueError("Every selected FP row must have reviewed=true")
    invalid = sorted(set(review.fp_error_category) - set(FP_CATEGORIES))
    if invalid or review.fp_error_category.eq("").any():
        raise ValueError(f"Unknown or missing FP categories: {invalid}")
    counts = review.fp_error_category.value_counts()
    crosstabs = {"by_aspect": pd.crosstab(review.fp_error_category, review.predicted_aspect).to_dict(),
        "by_language": pd.crosstab(review.fp_error_category, review.language).to_dict(),
        "by_gold_zero": pd.crosstab(review.fp_error_category, review.is_gold_zero_document).to_dict(),
        "by_predicted_multi_aspect": pd.crosstab(review.fp_error_category, review.is_predicted_multi_aspect_document).to_dict()}
    dominant = counts.head(5)
    mechanism_text = {
        "factual_as_evaluative": "Require an explicit or text-defensible evaluation; factual participation, times and event facts alone yield zero mentions.",
        "context_as_evaluative": "Separate event/topic context from the evaluated target and forbid context-only aspect extraction.",
        "over_decomposition": "Extract the minimum directly supported aspect set and avoid expanding one evaluative clause across related families.",
        "aspect_inference_too_far": "Tighten the evidence-to-aspect entailment requirement and prohibit background inference.",
        "sentiment_spillover": "Bind polarity locally to the evaluated aspect rather than transferring document-level affect.",
        "ontology_boundary_confusion": "Reinforce frozen aspect boundary rules with contrastive boundary examples.",
        "duplicate_redundant_mention": "Return one mention per distinct aspect evaluation unless separate evidence justifies repetition.",
        "grounding_related": "Require one contiguous exact evidence span for every mention.",
        "other_unclear": "Inspect unclear cases before specifying a general prompt-policy mechanism.",
    }
    examples = []
    for category in dominant.index:
        rows = review[review.fp_error_category.eq(category)].head(3)
        examples.append({"category": category, "evidence_snippets": [str(x)[:180] for x in rows.predicted_evidence]})
    def contrast_summary(filename: str, note_column: str) -> dict[str, Any]:
        path = output_dir / filename
        frame = pd.read_csv(path).fillna("")
        done = frame.reviewed.astype(str).str.lower().isin(["true", "1", "yes"]) if "reviewed" in frame else pd.Series(False, index=frame.index)
        notes = frame.loc[done, note_column].astype(str).loc[lambda x: x.str.strip().ne("")].tolist() if note_column in frame else []
        return {"rows": len(frame), "reviewed": int(done.sum()), "notes": notes}
    tp_summary = contrast_summary("absa_v1_tp_contrast_sample_v1.csv", "why_legitimate_evaluation")
    fn_summary = contrast_summary("absa_v1_fn_contrast_sample_v1.csv", "researcher_notes")
    result = {"protocol_name": PROTOCOL, "finalized_at": datetime.now(timezone.utc).isoformat(),
        "reviewed_fp_rows": len(review), "category_counts": counts.to_dict(),
        "category_percentages": (counts / len(review)).to_dict(), "crosstabs": crosstabs,
        "dominant_error_mechanisms": dominant.index.tolist(), "representative_examples": examples,
        "tp_contrast": tp_summary, "fn_contrast": fn_summary,
        "recommended_prompt_policy_mechanisms_to_address": [mechanism_text[category] for category in dominant.index],
        "recall_preservation_mechanisms_observed_in_tp_fn_contrast": (tp_summary["notes"] + fn_summary["notes"])[:5],
        "prompt_generated": False, "api_calls": 0}
    final_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=int), encoding="utf-8")
    lines = ["# ABSA V1 FP Diagnostic Final", "", f"Reviewed FP rows: **{len(review)}**.", "",
        "## Dominant error mechanisms", ""]
    for category, count in dominant.items(): lines.append(f"- `{category}`: {count} ({count/len(review):.1%})")
    lines.extend(["", "## Recommended prompt-policy mechanisms to address", ""])
    for i, category in enumerate(dominant.index, 1): lines.append(f"{i}. {mechanism_text[category]}")
    lines.extend(["", "## Recall-preservation mechanisms observed in TP/FN contrast", ""])
    contrast_notes = (tp_summary["notes"] + fn_summary["notes"])[:5]
    if contrast_notes:
        for i, note in enumerate(contrast_notes, 1): lines.append(f"{i}. {note}")
    else:
        lines.append("1. Contrast review is incomplete; no mechanism is asserted yet.")
    final_md.write_text("\n".join(lines), encoding="utf-8")
    return result
