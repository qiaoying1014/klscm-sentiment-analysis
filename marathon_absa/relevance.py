from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

RELEVANCE_LABELS = {"relevant", "ambiguous", "irrelevant"}
FINAL_HUMAN_LABELS = {"relevant", "irrelevant"}
RELEVANCE_POLICY_VERSION = "feeling-inclusive-v7"
REASON_CODES = {
    "event_registration", "event_preparation", "event_logistics", "event_information",
    "event_participation", "event_result_achievement", "event_experience_evaluation",
    "event_spectator_support", "unrelated_event", "generic_running",
    "commercial_promotion", "lifestyle_or_spam", "no_event_connection",
    "insufficient_text", "hashtag_only", "image_dependent", "weak_event_connection",
    "conflicting_evidence",
}

INITIAL_RELEVANT_THRESHOLD = 0.80
INITIAL_IRRELEVANT_THRESHOLD = 0.90
ADJUDICATED_RELEVANT_THRESHOLD = 0.90
ADJUDICATED_IRRELEVANT_THRESHOLD = 0.90
MANDATORY_ADJUDICATION_REASON_CODES = {
    "generic_running", "no_event_connection", "weak_event_connection",
    "hashtag_only", "conflicting_evidence",
}


def deterministic_result(row) -> dict | None:
    linguistic = str(getattr(row, "linguistic_text", "") or "").strip()
    status = str(getattr(row, "language_status", "") or "")
    if not linguistic:
        hashtags = getattr(row, "hashtags", []) or []
        if len(hashtags):
            return _excluded_result("hashtag_only")
        return _rule_result("insufficient_text")
    if status in {"no_text", "insufficient_text"}:
        return _rule_result("insufficient_text")
    return None


def _rule_result(reason_code: str) -> dict:
    return {
        "initial_relevance": "ambiguous", "initial_confidence": 1.0,
        "adjudicated_relevance": "", "adjudicated_confidence": None,
        "relevance": "ambiguous", "confidence": 1.0, "reason_code": reason_code,
        "evidence": "", "english_gloss": "", "model_observed_language": "",
        "code_switching_note": "", "review_status": "pending",
        "adjudication_method": "deterministic_rule", "include_in_topics": False,
    }



def _excluded_result(reason_code: str) -> dict:
    """Return an auditable final exclusion that bypasses human review."""
    return {
        "initial_relevance": "irrelevant", "initial_confidence": 1.0,
        "adjudicated_relevance": "", "adjudicated_confidence": None,
        "relevance": "irrelevant", "confidence": 1.0, "reason_code": reason_code,
        "evidence": "", "english_gloss": "", "model_observed_language": "",
        "code_switching_note": "", "review_status": "auto_excluded",
        "adjudication_method": "deterministic_exclusion",
        "include_in_topics": False, "policy_version": RELEVANCE_POLICY_VERSION,
        "exclusion_status": "excluded", "exclusion_reason": reason_code,
    }

def needs_adjudication(
    label: str, confidence: float, reason_code: str = "",
) -> bool:
    """Route uncertain or calibration-sensitive decisions to the stronger model."""
    if label == "relevant":
        return True
    if reason_code in MANDATORY_ADJUDICATION_REASON_CODES:
        return True
    return not (
        (label == "relevant" and confidence >= INITIAL_RELEVANT_THRESHOLD)
        or (label == "irrelevant" and confidence >= INITIAL_IRRELEVANT_THRESHOLD)
    )


def route_verified_result(initial: dict, verification: dict) -> dict:
    """Apply the auditable v6 evidence gate and keep routing separate."""
    confidence = float(verification["confidence"])
    verified_relevant = (
        verification["substantive_relevance"] == "relevant"
        and bool(verification["klscm_is_actual_subject"])
        and bool(verification["concrete_event_relation"])
        and verification["exclusion_trigger"] == "none"
    )
    if verified_relevant and confidence >= 0.80:
        final_label, status = "relevant", "auto_resolved"
    elif verification["substantive_relevance"] == "irrelevant" and confidence >= 0.80:
        final_label, status = "irrelevant", "auto_resolved"
    else:
        final_label, status = "ambiguous", "pending"
    return {
        "initial_relevance": initial["relevance"],
        "initial_confidence": float(initial["confidence"]),
        "adjudicated_relevance": verification["substantive_relevance"],
        "adjudicated_confidence": confidence,
        "relevance": final_label,
        "substantive_relevance": verification["substantive_relevance"],
        "confidence": confidence,
        "reason_code": verification["reason_code"],
        "evidence": verification["evidence"],
        "english_gloss": verification["english_gloss"],
        "model_observed_language": verification["model_observed_language"],
        "code_switching_note": verification["code_switching_note"],
        "klscm_is_actual_subject": bool(verification["klscm_is_actual_subject"]),
        "concrete_event_relation": bool(verification["concrete_event_relation"]),
        "exclusion_trigger": verification["exclusion_trigger"],
        "routing_status": status,
        "review_status": status,
        "adjudication_method": "structured_verification",
        "include_in_topics": final_label == "relevant" and status == "auto_resolved",
    }

def route_results(initial: dict, adjudicated: dict | None = None) -> dict:
    initial_label = initial["relevance"]
    initial_confidence = float(initial["confidence"])
    chosen = initial
    method = "initial_model"
    review_status = "auto_resolved"
    final_label = initial_label

    if needs_adjudication(
        initial_label, initial_confidence, initial.get("reason_code", "")
    ):
        if adjudicated is None:
            chosen = initial
            method = "awaiting_adjudication"
            review_status = "pending"
            final_label = "ambiguous"
        else:
            chosen = adjudicated
            method = "stronger_model"
            label = adjudicated["relevance"]
            confidence = float(adjudicated["confidence"])
            accepted = (
                label == "relevant" and confidence >= ADJUDICATED_RELEVANT_THRESHOLD
            ) or (
                label == "irrelevant" and confidence >= ADJUDICATED_IRRELEVANT_THRESHOLD
            )
            final_label = label if accepted else "ambiguous"
            review_status = "auto_resolved" if accepted else "pending"

    return {
        "initial_relevance": initial_label,
        "initial_confidence": initial_confidence,
        "adjudicated_relevance": adjudicated["relevance"] if adjudicated else "",
        "adjudicated_confidence": float(adjudicated["confidence"]) if adjudicated else None,
        "relevance": final_label,
        "confidence": float(chosen["confidence"]),
        "reason_code": chosen["reason_code"],
        "evidence": chosen["evidence"],
        "english_gloss": chosen["english_gloss"],
        "model_observed_language": chosen["model_observed_language"],
        "code_switching_note": chosen["code_switching_note"],
        "review_status": review_status,
        "adjudication_method": method,
        "include_in_topics": final_label == "relevant" and review_status == "auto_resolved",
    }


def stratified_pilot(documents: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n >= len(documents):
        return documents.copy()
    frame = documents.copy()
    had_length_band = "length_band" in frame.columns
    frame["length_band"] = pd.qcut(
        frame.linguistic_text.fillna("").str.len().rank(method="first"),
        q=min(4, len(frame)), labels=False, duplicates="drop",
    )
    groups = ["event_year", "primary_language", "language_status", "length_band"]
    sampled = frame.groupby(groups, group_keys=False, dropna=False).sample(n=1, random_state=seed)
    if len(sampled) < n:
        remaining = frame.drop(index=sampled.index)
        sampled = pd.concat([sampled, remaining.sample(min(n - len(sampled), len(remaining)), random_state=seed)])
    elif len(sampled) > n:
        sampled = sampled.sample(n, random_state=seed)
    sampled = sampled.head(n)
    return sampled if had_length_band else sampled.drop(columns=["length_band"])


def build_review_queue(relevance: pd.DataFrame, documents: pd.DataFrame, existing_csv: Path | None = None) -> pd.DataFrame:
    pending = relevance[relevance.review_status.eq("pending")].copy()
    context = documents[["document_id", "event_year", "primary_language", "original_text", "linguistic_text"]]
    queue = pending.merge(context, on="document_id", how="left")
    queue["human_relevance"] = ""
    queue["review_notes"] = ""
    if existing_csv and existing_csv.exists():
        old = pd.read_csv(existing_csv, dtype=str, keep_default_na=False)
        if {"document_id", "human_relevance", "review_notes"}.issubset(old.columns):
            saved = old[["document_id", "human_relevance", "review_notes"]].drop_duplicates("document_id", keep="last")
            queue = queue.drop(columns=["human_relevance", "review_notes"]).merge(saved, on="document_id", how="left")
            queue[["human_relevance", "review_notes"]] = queue[["human_relevance", "review_notes"]].fillna("")
    return queue


def apply_human_reviews(relevance: pd.DataFrame, review_queue: pd.DataFrame) -> pd.DataFrame:
    output = relevance.copy()
    if review_queue.empty:
        return output
    labels = review_queue.human_relevance.fillna("").astype(str).str.strip().str.lower()
    invalid = sorted(set(labels) - FINAL_HUMAN_LABELS - {""})
    if invalid:
        raise ValueError(f"Invalid human_relevance values: {invalid}; use relevant or irrelevant")
    for row, label in zip(review_queue.itertuples(index=False), labels):
        if not label:
            continue
        mask = output.document_id.eq(row.document_id)
        if not mask.any():
            continue
        output.loc[mask, "relevance"] = label
        output.loc[mask, "confidence"] = 1.0
        output.loc[mask, "review_status"] = "human_resolved"
        output.loc[mask, "adjudication_method"] = "human_review"
        output.loc[mask, "include_in_topics"] = label == "relevant"
        output.loc[mask, "human_relevance"] = label
        output.loc[mask, "review_notes"] = str(getattr(row, "review_notes", "") or "")
    return output


def validate_topic_gate(documents: pd.DataFrame, relevance: pd.DataFrame) -> None:
    expected = set(documents.loc[(documents.source == "instagram") & (documents.processing_status == "ready"), "document_id"])
    observed = set(relevance.document_id)
    missing = expected - observed
    pending = relevance[relevance.document_id.isin(expected) & relevance.review_status.eq("pending")]
    if missing:
        raise RuntimeError(f"Relevance filtering is incomplete for {len(missing):,} ready Instagram records")
    if len(pending):
        raise RuntimeError(f"Resolve {len(pending):,} pending relevance reviews before topic discovery")


def build_validation_sample(relevance: pd.DataFrame, documents: pd.DataFrame, n: int = 300, repeats: int = 30, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = relevance.merge(documents[["document_id", "event_year", "primary_language", "language_status", "original_text", "linguistic_text"]], on="document_id", how="inner")
    if merged.empty:
        raise ValueError("No relevance results are available for validation sampling")
    merged["length_band"] = pd.qcut(merged.linguistic_text.fillna("").str.len().rank(method="first"), q=min(4, len(merged)), labels=False, duplicates="drop")
    per_label = max(1, n // max(1, merged.relevance.nunique()))
    chosen = []
    for label, group in merged.groupby("relevance"):
        chosen.append(stratified_pilot(group, min(per_label, len(group)), seed))
    sample = pd.concat(chosen).drop_duplicates("document_id")
    if len(sample) < n:
        remaining = merged[~merged.document_id.isin(sample.document_id)]
        sample = pd.concat([sample, remaining.sample(min(n - len(sample), len(remaining)), random_state=seed)])
    sample = sample.head(n).sample(frac=1, random_state=seed).reset_index(drop=True)
    sample["validation_split"] = ["calibration" if i < n // 2 else "holdout" for i in range(n)]
    sample["review_id"] = [f"rel-{i + 1:04d}" for i in range(n)]
    key = sample[["review_id", "document_id", "relevance", "confidence", "reason_code", "validation_split", "event_year", "primary_language", "language_status", "length_band"]].rename(columns={"relevance": "model_relevance", "confidence": "model_confidence", "reason_code": "model_reason_code"})
    reviewer = sample[["review_id", "document_id", "event_year", "primary_language", "language_status", "original_text"]].copy()
    reviewer["repeat_of"] = ""
    reviewer["human_relevance"] = ""
    reviewer["review_notes"] = ""
    repeat_rows = reviewer.sample(min(repeats, len(reviewer)), random_state=seed + 1).copy()
    repeat_rows["repeat_of"] = repeat_rows.review_id
    repeat_rows["review_id"] = [f"repeat-{i + 1:03d}" for i in range(len(repeat_rows))]
    reviewer = pd.concat([reviewer, repeat_rows], ignore_index=True).sample(frac=1, random_state=seed + 2).reset_index(drop=True)
    return reviewer, key


def build_fresh_holdout_sample(
    relevance: pd.DataFrame,
    documents: pd.DataFrame,
    excluded_document_ids: set[str],
    n: int = 150,
    repeats: int = 15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a blind holdout with no documents used by earlier validation."""
    eligible = relevance[~relevance.document_id.isin(excluded_document_ids)].copy()
    if len(eligible) < n:
        raise ValueError(
            f"Only {len(eligible)} eligible records remain after validation exclusions; "
            f"{n} requested"
        )
    reviewer, key = build_validation_sample(
        eligible, documents, n=n, repeats=repeats, seed=seed + 1000,
    )
    key["validation_split"] = "fresh_holdout"
    id_map = {old: f"fresh-{index + 1:04d}" for index, old in enumerate(key.review_id.tolist())}
    key["review_id"] = key.review_id.map(id_map)
    originals = reviewer.repeat_of.eq("")
    reviewer.loc[originals, "review_id"] = reviewer.loc[originals, "review_id"].map(id_map)
    reviewer.loc[~originals, "repeat_of"] = reviewer.loc[~originals, "repeat_of"].map(id_map)
    reviewer.loc[~originals, "review_id"] = [f"fresh-repeat-{index + 1:03d}" for index in range((~originals).sum())]
    if set(key.document_id) & excluded_document_ids:
        raise RuntimeError("Fresh holdout contains an excluded validation document")
    return reviewer, key

def fresh_holdout_report(reviewer: pd.DataFrame, key: pd.DataFrame, output_dir: Path) -> dict:
    """Evaluate frozen relevance decisions with ambiguous treated as routing."""
    report = validation_report(reviewer, key, output_dir, evaluation_split="fresh_holdout")
    originals = reviewer[reviewer.repeat_of.fillna("").eq("")].copy()
    originals["human_relevance"] = originals.human_relevance.fillna("").str.strip().str.lower()
    evaluated = key.merge(originals[["review_id", "human_relevance"]], on="review_id", how="left")
    substantive = evaluated[evaluated.human_relevance.ne("ambiguous")].copy()
    human_binary = substantive.human_relevance.eq("relevant")
    model_binary = substantive.model_relevance.eq("relevant")
    precision, recall, _, _ = precision_recall_fscore_support(human_binary, model_binary, average="binary", zero_division=0)
    binary_macro_f1 = precision_recall_fscore_support(human_binary, model_binary, average="macro", zero_division=0)[2]
    repeats = reviewer[reviewer.repeat_of.fillna("").ne("")].merge(originals[["review_id", "human_relevance"]].rename(columns={"review_id": "repeat_of", "human_relevance": "original_label"}), on="repeat_of", how="left")
    repeat_agreements = int(repeats.human_relevance.str.strip().str.lower().eq(repeats.original_label).sum())
    final = {**report, "substantive_binary_n": int(len(substantive)), "substantive_binary_relevant_precision": float(precision), "substantive_binary_relevant_recall": float(recall), "substantive_binary_macro_f1": float(binary_macro_f1), "gold_ambiguous_n": int(evaluated.human_relevance.eq("ambiguous").sum()), "predicted_ambiguous_n": int(evaluated.model_relevance.eq("ambiguous").sum()), "ambiguous_routing_rate": float(evaluated.model_relevance.eq("ambiguous").mean()), "repeat_agreements": repeat_agreements, "repeat_agreement_rate": float(repeat_agreements / max(1, len(repeats))), "passes_targets": bool(precision >= 0.85 and recall >= 0.90 and binary_macro_f1 >= 0.80)}
    (output_dir / "relevance_fresh_holdout_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    return final

def binary_metrics_with_bootstrap(
    frame: pd.DataFrame,
    human_column: str,
    prediction_column: str,
    seed: int = 42,
    bootstrap_samples: int = 1000,
    weight_column: str | None = None,
) -> dict:
    """Return reproducible binary metrics and percentile confidence intervals."""
    human = frame[human_column].eq("relevant").to_numpy()
    predicted = frame[prediction_column].eq("relevant").to_numpy()
    weights = frame[weight_column].astype(float).to_numpy() if weight_column else None
    def calculate(index):
        sample_weight = weights[index] if weights is not None else None
        precision, recall, _, _ = precision_recall_fscore_support(human[index], predicted[index], average="binary", zero_division=0, sample_weight=sample_weight)
        macro = precision_recall_fscore_support(human[index], predicted[index], average="macro", zero_division=0, sample_weight=sample_weight)[2]
        return precision, recall, macro
    full = calculate(np.arange(len(frame)))
    rng = np.random.default_rng(seed)
    draws = np.array([calculate(rng.integers(0, len(frame), len(frame))) for _ in range(bootstrap_samples)])
    names = ["relevant_precision", "relevant_recall", "binary_macro_f1"]
    result = {name: float(value) for name, value in zip(names, full)}
    for index, name in enumerate(names):
        result[f"{name}_ci95"] = [float(x) for x in np.percentile(draws[:, index], [2.5, 97.5])]
    result["bootstrap_samples"] = bootstrap_samples
    result["bootstrap_seed"] = seed
    result["weighted"] = weight_column is not None
    return result

def validation_report(
    reviewer: pd.DataFrame, key: pd.DataFrame, output_dir: Path,
    evaluation_split: str = "holdout",
) -> dict:
    human = reviewer[reviewer.repeat_of.fillna("").eq("")].copy()
    human.human_relevance = human.human_relevance.fillna("").str.strip().str.lower()
    invalid = sorted(set(human.human_relevance) - RELEVANCE_LABELS - {""})
    if invalid:
        raise ValueError(f"Invalid validation labels: {invalid}")
    merged = key.merge(human[["review_id", "human_relevance"]], on="review_id", how="left")
    holdout_expected = merged[merged.validation_split == evaluation_split]
    missing_holdout = ~holdout_expected.human_relevance.isin(RELEVANCE_LABELS)
    if missing_holdout.any():
        raise ValueError(f"Complete all holdout labels before validation ({int(missing_holdout.sum())} missing)")
    holdout = holdout_expected.copy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        holdout.human_relevance.eq("relevant"), holdout.model_relevance.eq("relevant"), average="binary", zero_division=0,
    )
    macro = precision_recall_fscore_support(holdout.human_relevance, holdout.model_relevance, average="macro", zero_division=0)[2]
    originals = reviewer[["review_id", "human_relevance"]].rename(columns={"review_id": "repeat_of", "human_relevance": "original_label"})
    repeated = reviewer[reviewer.repeat_of.fillna("").ne("")].merge(originals, on="repeat_of", how="left")
    repeated["human_relevance"] = repeated.human_relevance.fillna("").str.strip().str.lower()
    repeated["original_label"] = repeated.original_label.fillna("").str.strip().str.lower()
    repeated = repeated[repeated.human_relevance.isin(RELEVANCE_LABELS) & repeated.original_label.isin(RELEVANCE_LABELS)]
    intra_reviewer_kappa = float(cohen_kappa_score(repeated.original_label, repeated.human_relevance)) if len(repeated) else None
    report = {
        f"{evaluation_split}_n": int(len(holdout)), "relevant_precision": float(precision),
        "relevant_recall": float(recall), "macro_f1": float(macro),
        "intra_reviewer_n": int(len(repeated)), "intra_reviewer_kappa": intra_reviewer_kappa,
        "passes_targets": bool(recall >= 0.90 and precision >= 0.85 and macro >= 0.80),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = ["relevant", "ambiguous", "irrelevant"]
    pd.DataFrame(confusion_matrix(holdout.human_relevance, holdout.model_relevance, labels=labels), index=labels, columns=labels).to_csv(output_dir / "relevance_confusion_matrix.csv")
    pd.DataFrame(classification_report(holdout.human_relevance, holdout.model_relevance, output_dict=True, zero_division=0)).T.to_csv(output_dir / "relevance_classification_report.csv")
    subgroup_rows = []
    for dimension in ["event_year", "primary_language", "language_status", "length_band"]:
        for value, group in holdout.groupby(dimension, dropna=False):
            relevant_total = int(group.human_relevance.eq("relevant").sum())
            subgroup_rows.append({
                "dimension": dimension, "group": str(value), "n": len(group),
                "accuracy": float(group.human_relevance.eq(group.model_relevance).mean()),
                "relevant_recall": float(((group.human_relevance == "relevant") & (group.model_relevance == "relevant")).sum() / max(1, relevant_total)),
            })
    pd.DataFrame(subgroup_rows).to_csv(output_dir / "relevance_subgroup_metrics.csv", index=False)
    (output_dir / "relevance_validation_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

