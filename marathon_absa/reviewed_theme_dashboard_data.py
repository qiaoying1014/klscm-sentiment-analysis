from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REVIEW_ROOT = Path("data/processed/absa_v1/aspect_level_themes_review_v1")
EXPECTED_SOURCE_CLUSTERS = 101
EXPECTED_REVIEWED_THEMES = 64
EXPECTED_THEME_ASPECTS = 16
EXPECTED_YEARS = {2019, 2023, 2024, 2025}
INSUFFICIENT_ASPECTS = {"facilities", "transport_access", "race_pack_expo", "event_information"}

SUMMARY_COLUMNS = {
    "aspect", "reviewed_theme_id", "reviewed_theme_key", "reviewed_theme_label", "source_cluster_ids",
    "support_mentions", "support_documents", "share_of_aspect_documents", "theme_quality", "review_status",
    "positive_documents", "negative_documents", "mixed_documents", "neutral_documents",
    "positive_share", "negative_share", "mixed_share", "neutral_share", "years_present", "top_targets",
}
SENTIMENT_COLUMNS = {"aspect", "reviewed_theme_id", "document_sentiment", "support_documents", "share_of_theme_documents"}
YEAR_COLUMNS = {"aspect", "reviewed_theme_id", "year", "support_documents", "aspect_documents_that_year", "within_aspect_theme_prevalence"}
EVIDENCE_COLUMNS = {"aspect", "reviewed_theme_id", "reviewed_theme_label", "mention_id", "document_id", "target", "evidence_text", "english_gloss", "sentiment", "event_year", "primary_language"}


@dataclass(frozen=True)
class ReviewedThemeDashboardData:
    manifest: dict[str, Any]
    taxonomy: pd.DataFrame
    summary: pd.DataFrame
    sentiment: pd.DataFrame
    years: pd.DataFrame
    evidence: pd.DataFrame
    insufficient_support: pd.DataFrame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(root: Path, name: str) -> pd.DataFrame:
    path = root / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Finalized reviewed-theme artifact is missing: {path}")
    return pd.read_parquet(path)


def load_reviewed_theme_dashboard_data(root: Path = REVIEW_ROOT) -> ReviewedThemeDashboardData:
    from .cloud_bundle import load_cloud_bundle
    bundle = load_cloud_bundle("themes") if root == REVIEW_ROOT else None
    if bundle is not None:
        data = ReviewedThemeDashboardData(**bundle)
        validate_reviewed_theme_dashboard_data(data)
        return data
    manifest_path = root / "review_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Finalized reviewed-theme manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("lifecycle_status") != "reviewed_taxonomy_finalized":
        raise ValueError("Reviewed-theme package is not finalized; provisional ALTA clusters will not be displayed")
    for name, expected in manifest.get("output_hashes", {}).items():
        path = root / name
        if not path.exists() or _sha256(path) != expected:
            raise ValueError(f"Finalized reviewed-theme artifact hash mismatch: {name}")
    data = ReviewedThemeDashboardData(
        manifest=manifest,
        taxonomy=_read(root, "reviewed_theme_taxonomy"),
        summary=_read(root, "reviewed_theme_summary"),
        sentiment=_read(root, "reviewed_theme_sentiment_summary"),
        years=_read(root, "reviewed_theme_year_summary"),
        evidence=_read(root, "reviewed_theme_evidence"),
        insufficient_support=_read(root, "insufficient_support_summary"),
    )
    validate_reviewed_theme_dashboard_data(data)
    return data


def validate_reviewed_theme_dashboard_data(data: ReviewedThemeDashboardData) -> None:
    errors: list[str] = []
    manifest = data.manifest
    if manifest.get("source_clusters") != EXPECTED_SOURCE_CLUSTERS: errors.append("source clusters != 101")
    if manifest.get("number_reviewed") != EXPECTED_SOURCE_CLUSTERS: errors.append("reviewed source clusters != 101")
    if manifest.get("final_reviewed_theme_count") != EXPECTED_REVIEWED_THEMES: errors.append("reviewed themes != 64")
    if len(data.taxonomy) != EXPECTED_REVIEWED_THEMES or len(data.summary) != EXPECTED_REVIEWED_THEMES: errors.append("taxonomy/summary rows != 64")
    if data.summary.reviewed_theme_id.duplicated().any() or data.taxonomy.reviewed_theme_id.duplicated().any(): errors.append("reviewed theme IDs are not unique")
    if data.summary.aspect.nunique() != EXPECTED_THEME_ASPECTS: errors.append("cluster-bearing aspects != 16")
    if not SUMMARY_COLUMNS.issubset(data.summary.columns): errors.append("reviewed summary schema")
    if not SENTIMENT_COLUMNS.issubset(data.sentiment.columns): errors.append("reviewed sentiment schema")
    if not YEAR_COLUMNS.issubset(data.years.columns): errors.append("reviewed year schema")
    if not EVIDENCE_COLUMNS.issubset(data.evidence.columns): errors.append("reviewed evidence schema")
    identities = data.summary[["aspect", "reviewed_theme_id"]]
    for name, frame in [("taxonomy", data.taxonomy), ("sentiment", data.sentiment), ("year", data.years), ("evidence", data.evidence)]:
        joined = frame[["aspect", "reviewed_theme_id"]].drop_duplicates().merge(identities, on=["aspect", "reviewed_theme_id"], how="left", indicator=True)
        if not joined._merge.eq("both").all(): errors.append(f"{name} cross-aspect or unknown-theme lineage")
    if len(data.sentiment) != EXPECTED_REVIEWED_THEMES * 4 or set(data.sentiment.document_sentiment) != {"positive", "negative", "mixed", "neutral"}: errors.append("sentiment rows/categories")
    if len(data.years) != EXPECTED_REVIEWED_THEMES * 4 or set(data.years.year.astype(int)) != EXPECTED_YEARS: errors.append("year rows/categories")
    if len(data.evidence) != EXPECTED_REVIEWED_THEMES * 5 or data.evidence.mention_id.duplicated().any(): errors.append("representative evidence rows/identity")
    if set(data.insufficient_support.aspect) != INSUFFICIENT_ASPECTS or not data.insufficient_support.status.eq("insufficient_support").all(): errors.append("insufficient-support aspects")
    if not data.summary.review_status.eq("reviewed").all() or data.summary.reviewed_theme_label.astype(str).str.strip().eq("").any(): errors.append("unreviewed/provisional summary rows")
    sentiment_counts = data.sentiment.groupby("reviewed_theme_id").support_documents.sum()
    summary_counts = data.summary.set_index("reviewed_theme_id").support_documents
    if not sentiment_counts.reindex(summary_counts.index).equals(summary_counts.astype(sentiment_counts.dtype)): errors.append("sentiment document support mismatch")
    if not np.allclose(data.sentiment.groupby("reviewed_theme_id").share_of_theme_documents.sum(), 1): errors.append("sentiment shares do not sum to one")
    if data.summary.support_documents.lt(1).any() or data.summary.share_of_aspect_documents.lt(0).any() or data.summary.share_of_aspect_documents.gt(1).any(): errors.append("reviewed support values")
    if errors:
        raise ValueError("Finalized reviewed-theme dashboard validation failed: " + "; ".join(errors))
