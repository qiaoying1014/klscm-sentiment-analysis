from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix


def classification_metrics(frame: pd.DataFrame, truth: str, prediction: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = frame.dropna(subset=[truth, prediction])
    if valid.empty:
        return pd.DataFrame(), pd.DataFrame()
    report = pd.DataFrame(classification_report(valid[truth], valid[prediction], output_dict=True, zero_division=0)).T.reset_index(names="label")
    labels = sorted(set(valid[truth]) | set(valid[prediction]))
    matrix = pd.DataFrame(confusion_matrix(valid[truth], valid[prediction], labels=labels), index=labels, columns=labels)
    return report, matrix


def language_summary(frame: pd.DataFrame) -> dict:
    valid = frame.dropna(subset=["human_language", "detected_language"])
    return {
        "n": len(valid),
        "accuracy": accuracy_score(valid.human_language, valid.detected_language) if len(valid) else None,
        "undetermined_rate": float((frame.detected_language == "undetermined").mean()) if len(frame) else None,
        "low_confidence_rate": float((frame.language_status == "low_confidence").mean()) if len(frame) else None,
    }


def inter_coder_kappa(frame: pd.DataFrame, coder_a: str, coder_b: str) -> float | None:
    valid = frame.dropna(subset=[coder_a, coder_b])
    return float(cohen_kappa_score(valid[coder_a], valid[coder_b])) if len(valid) else None


def make_validation_sample(documents: pd.DataFrame, n: int = 300, seed: int = 42) -> pd.DataFrame:
    pool = documents[documents.processing_status == "ready"].copy()
    pool["length_band"] = pd.qcut(pool.normalized_text.str.len().rank(method="first"), q=min(4, len(pool)), labels=False, duplicates="drop")
    per_group = max(1, n // max(1, pool.groupby(["source", "language_status"], dropna=False).ngroups))
    sampled = pool.groupby(["source", "language_status"], group_keys=False, dropna=False).apply(lambda x: x.sample(min(len(x), per_group), random_state=seed), include_groups=False)
    sampled = sampled.head(n).copy()
    for column in ["human_language", "coder_1_relevance", "coder_2_relevance", "human_aspects", "human_sentiment", "review_notes"]:
        sampled[column] = ""
    return sampled

