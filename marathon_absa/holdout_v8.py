from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


HOLDOUT_VERSION = "relevance_v8_blind_holdout_v1"
HOLDOUT_SEED = 91827
REVIEWER_A_SEED = 91828
REVIEWER_B_SEED = 91829
ANNOTATION_COLUMNS = ["holdout_id", "caption", "permitted_metadata", "language",
                      "label", "evidence_span", "rationale", "confidence", "comments"]
HIDDEN_TOKENS = {"prediction", "model_confidence", "routing", "reason_code",
                 "historical", "gold", "content_type", "event_connection"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def leakage_normalize(text: object) -> str:
    value = str(text).lower()
    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    value = re.sub(r"#\w+", " ", value)
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def collect_development_ids(root: Path) -> tuple[set[str], dict[str, int]]:
    """Collect IDs from all identifiable human/development evaluation artifacts."""
    patterns = [
        "relevance*sample*.csv", "relevance*key*.csv", "*adjudication*.csv",
        "*audit*.csv", "*error*.csv", "*review_queue*.csv",
        "*policy_alignment*.csv", "*threshold_route_diff*.csv",
        "*false_inclusion*.csv", "*auto_exclusion*.csv",
    ]
    skip = {"relevance_v8_holdout_sampling_frame_v1.csv",
            "relevance_v8_blind_holdout_v1.csv"}
    ids: set[str] = set()
    by_file: dict[str, int] = {}
    seen_paths: set[Path] = set()
    for pattern in patterns:
        for path in root.rglob(pattern):
            if path.name in skip or path in seen_paths:
                continue
            seen_paths.add(path)
            try:
                columns = pd.read_csv(path, nrows=0).columns
                id_column = "document_id" if "document_id" in columns else (
                    "record_id" if "record_id" in columns else None)
                if not id_column:
                    continue
                values = set(pd.read_csv(path, usecols=[id_column], dtype=str,
                                         keep_default_na=False)[id_column]) - {""}
                ids.update(values)
                by_file[str(path)] = len(values)
            except (UnicodeDecodeError, pd.errors.ParserError, ValueError):
                continue
    return ids, by_file


def build_sampling_frame(documents: pd.DataFrame, development_ids: set[str],
                         few_shot_texts: list[str]) -> tuple[pd.DataFrame, dict]:
    frame = documents.copy()
    total = len(frame)
    development = frame.document_id.isin(development_ids)
    few_norm = {leakage_normalize(text) for text in few_shot_texts}
    fewshot = frame.original_text.map(leakage_normalize).isin(few_norm)
    excluded_development = int((development | fewshot).sum())
    development_hashes = set(frame.loc[development, "text_hash"].dropna().astype(str))
    development_near = set(frame.loc[development, "original_text"].map(leakage_normalize)) - {""}
    eligible = frame.loc[~(development | fewshot)].copy()
    eligible_before_dedup = len(eligible)
    exact = (eligible.text_hash.astype(str).isin(development_hashes) |
             eligible.duplicated("text_hash", keep="first") |
             eligible.duplicate_of.fillna("").ne(""))
    exact_removed = int(exact.sum())
    eligible = eligible.loc[~exact].copy()
    eligible["near_duplicate_key"] = eligible.original_text.map(leakage_normalize)
    usable_near = eligible.near_duplicate_key.str.len().ge(8)
    near = usable_near & (eligible.near_duplicate_key.isin(development_near) |
                          eligible.duplicated("near_duplicate_key", keep="first"))
    near_removed = int(near.sum())
    eligible = eligible.loc[~near].copy()
    # Language is independent OpenLID preprocessing, not relevance-model output.
    counts = eligible.primary_language.fillna("undetermined").value_counts()
    eligible["sampling_language"] = eligible.primary_language.fillna("undetermined").where(
        eligible.primary_language.fillna("undetermined").map(counts).ge(30), "other_or_rare")
    eligible["sampling_stratum"] = (eligible.source.fillna("unknown").astype(str) + "|" +
        eligible.event_year.fillna("unknown").astype(str) + "|" + eligible.sampling_language)
    report = {"total_corpus_records": total,
        "excluded_development_records": excluded_development,
        "eligible_records_before_deduplication": eligible_before_dedup,
        "exact_duplicates_removed": exact_removed, "near_duplicates_removed": near_removed,
        "final_eligible_population": len(eligible)}
    return eligible, report


def stratified_sample(frame: pd.DataFrame, size: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if size > len(frame):
        raise ValueError("Holdout size exceeds eligible population")
    counts = frame.sampling_stratum.value_counts().sort_index()
    exact = counts / counts.sum() * size
    allocation = exact.astype(int)
    remainder = size - int(allocation.sum())
    order = (exact-allocation).sort_values(ascending=False, kind="mergesort").index
    for stratum in order[:remainder]:
        allocation[stratum] += 1
    parts = []
    for index, (stratum, n) in enumerate(allocation.items()):
        if n:
            parts.append(frame[frame.sampling_stratum.eq(stratum)].sample(
                n=int(n), random_state=seed + index))
    sample = pd.concat(parts).sort_values("document_id").reset_index(drop=True)
    if len(sample) != size or sample.document_id.nunique() != size:
        raise AssertionError("Stratified sample size or identity failure")
    table = pd.DataFrame({"sampling_stratum": counts.index,
                          "population_n": counts.values,
                          "sample_n": allocation.reindex(counts.index).values})
    return sample, table


def make_reviewer_file(holdout: pd.DataFrame, seed: int) -> pd.DataFrame:
    reviewer = pd.DataFrame({"holdout_id": holdout.holdout_id,
        "caption": holdout.caption,
        "permitted_metadata": holdout.permitted_metadata,
        "language": holdout.language,
        "label": "", "evidence_span": "", "rationale": "",
        "confidence": "", "comments": ""})
    reviewer = reviewer.sample(frac=1, random_state=seed).reset_index(drop=True)
    if list(reviewer.columns) != ANNOTATION_COLUMNS:
        raise AssertionError("Reviewer schema mismatch")
    if any(any(token in column.lower() for token in HIDDEN_TOKENS)
           for column in reviewer.columns):
        raise ValueError("Hidden model field leaked into reviewer file")
    return reviewer


def assert_holdout_scoring_allowed(candidate_ids: set[str], output_dir: Path,
                                   dedicated_evaluation: bool = False) -> None:
    """Block normal scoring of frozen holdout until gold is finalized."""
    holdout = output_dir / "relevance_v8_holdout" / "relevance_v8_blind_holdout_v1.csv"
    if not holdout.exists():
        return
    membership = set(pd.read_csv(holdout, usecols=["source_record_id"],
                                 dtype=str)["source_record_id"])
    if not (candidate_ids & membership):
        return
    state_path = output_dir / "relevance_v8_holdout" / "holdout_temporal_state_v1.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if not dedicated_evaluation or not state.get("gold_finalized_at"):
        raise RuntimeError("Frozen blind holdout scoring is blocked until gold finalization and a dedicated evaluation command")
