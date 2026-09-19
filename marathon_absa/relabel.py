from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .relevance import stratified_pilot

POLICY_VERSION = "feeling-inclusive-v7"
RELABEL_LABELS = {"relevant", "needs_review", "irrelevant"}
FEELING_RE = re.compile(
    r"\b(?:excited|excitement|nervous|energetic|energy|tired|tiring|exhausted|"
    r"fatigue|pain|painful|relieved|relief|happy|happiness|sad|disappointed|"
    r"proud|pride|achievement|syukur|bangga|penat|letih|gembira|teruja|"
    r"semangat|lega)\b",
    re.IGNORECASE,
)


def eligible_relabel_documents(documents: pd.DataFrame) -> pd.DataFrame:
    """Return ready Instagram captions containing natural-language text."""
    mask = (
        documents.source.eq("instagram")
        & documents.processing_status.eq("ready")
        & documents.linguistic_text.fillna("").astype(str).str.strip().ne("")
    )
    return documents.loc[mask].copy()


def build_relabel_sample(
    documents: pd.DataFrame,
    excluded_document_ids: set[str],
    stage: str,
    n: int,
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a blind, stratified v7 sample with feeling-cue coverage."""
    eligible = eligible_relabel_documents(documents)
    eligible = eligible[~eligible.document_id.isin(excluded_document_ids)].copy()
    if len(eligible) < n:
        raise ValueError(f"Only {len(eligible)} eligible unused documents; {n} requested")
    eligible["challenge_category"] = "representative"
    feeling_mask = eligible.original_text.fillna("").str.contains(FEELING_RE, na=False)
    eligible.loc[feeling_mask, "challenge_category"] = "feeling_or_physical_cue"
    target_n = min(n // 3, int(feeling_mask.sum()))
    selected = []
    if target_n:
        selected.append(stratified_pilot(eligible[feeling_mask], target_n, seed))
    used = set(selected[0].document_id) if selected else set()
    remainder = eligible[~eligible.document_id.isin(used)]
    selected.append(stratified_pilot(remainder, n - len(used), seed + 1))
    sample = (
        pd.concat(selected, ignore_index=True)
        .drop_duplicates("document_id")
        .head(n)
        .sample(frac=1, random_state=seed + 2)
        .reset_index(drop=True)
    )
    sample["review_id"] = [f"v7-{stage}-{i + 1:04d}" for i in range(len(sample))]
    sample["policy_version"] = POLICY_VERSION
    sample["label_stage"] = stage
    reviewer_columns = [
        "review_id", "document_id", "event_year", "primary_language",
        "language_status", "original_text", "policy_version", "label_stage",
        "challenge_category",
    ]
    reviewer = sample[reviewer_columns].copy()
    reviewer["repeat_of"] = ""
    reviewer["human_relevance"] = ""
    reviewer["review_notes"] = ""
    key = sample[[
        "review_id", "document_id", "event_year", "primary_language",
        "language_status", "policy_version", "label_stage", "challenge_category",
    ]].copy()
    repeated = reviewer.sample(min(repeats, len(reviewer)), random_state=seed + 3).copy()
    repeated["repeat_of"] = repeated.review_id
    repeated["review_id"] = [f"v7-{stage}-repeat-{i + 1:03d}" for i in range(len(repeated))]
    reviewer = pd.concat([reviewer, repeated], ignore_index=True)
    reviewer = reviewer.sample(frac=1, random_state=seed + 4).reset_index(drop=True)
    return reviewer, key


def finalize_relabel_sample(reviewer: pd.DataFrame, stage: str) -> dict:
    """Validate a completed v7 sample and report repeat consistency."""
    labels = reviewer.human_relevance.fillna("").str.strip().str.lower()
    invalid = sorted(set(labels) - RELABEL_LABELS)
    if invalid or labels.eq("").any():
        raise ValueError(f"Complete all labels; invalid values: {invalid}")
    if labels.eq("needs_review").any():
        raise ValueError("Adjudicate every needs_review row before finalization")
    originals = reviewer[reviewer.repeat_of.fillna("").eq("")]
    repeats = reviewer[reviewer.repeat_of.fillna("").ne("")].merge(
        originals[["review_id", "human_relevance"]].rename(
            columns={"review_id": "repeat_of", "human_relevance": "original_label"}
        ), on="repeat_of", how="left",
    )
    repeat_labels = repeats.human_relevance.str.strip().str.lower()
    original_labels = repeats.original_label.fillna("").str.strip().str.lower()
    agreements = int(repeat_labels.eq(original_labels).sum())
    return {
        "policy_version": POLICY_VERSION,
        "stage": stage,
        "base_records": int(len(originals)),
        "repeat_records": int(len(repeats)),
        "repeat_agreements": agreements,
        "repeat_agreement_rate": float(agreements / max(1, len(repeats))),
        "relevant": int(originals.human_relevance.str.lower().eq("relevant").sum()),
        "irrelevant": int(originals.human_relevance.str.lower().eq("irrelevant").sum()),
    }


def build_repeat_disagreement_audit(reviewer: pd.DataFrame) -> pd.DataFrame:
    """Preserve inconsistent blind repeats for policy review without relabelling."""
    originals = reviewer[reviewer.repeat_of.fillna("").eq("")].copy()
    repeats = reviewer[reviewer.repeat_of.fillna("").ne("")].copy()
    original_columns = originals[
        ["review_id", "document_id", "human_relevance", "review_notes"]
    ].rename(columns={
        "review_id": "repeat_of", "human_relevance": "original_label",
        "review_notes": "original_notes",
    })
    paired = repeats.merge(original_columns, on=["repeat_of", "document_id"], how="left")
    paired["repeat_label"] = paired.human_relevance.str.strip().str.lower()
    paired["original_label"] = paired.original_label.str.strip().str.lower()
    disagreements = paired[paired.repeat_label.ne(paired.original_label)].copy()
    disagreements["adjudicated_label"] = ""
    disagreements["adjudication_reason"] = ""
    return disagreements[[
        "document_id", "repeat_of", "review_id", "original_text",
        "original_label", "repeat_label", "original_notes", "review_notes",
        "primary_language", "event_year", "challenge_category",
        "adjudicated_label", "adjudication_reason",
    ]]

def write_summary(report: dict, path: Path) -> None:
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")