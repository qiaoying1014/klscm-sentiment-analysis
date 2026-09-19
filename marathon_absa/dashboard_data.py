from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

DASHBOARD_ROOT = Path("data/processed/absa_v1/dashboard/absa_v1_dashboard_data_v1")
EXPECTED_DOCUMENTS = 7704
EXPECTED_MENTIONS = 15486
EXPECTED_ASPECTS = 20
EXPECTED_TOPICS = 32
EXPECTED_YEARS = {2019, 2023, 2024, 2025}


@dataclass(frozen=True)
class DashboardData:
    metadata: dict[str, Any]
    corpus: dict[str, Any]
    sentiment_overall: pd.DataFrame
    aspects: pd.DataFrame
    aspect_sentiment: pd.DataFrame
    years: pd.DataFrame
    year_aspect: pd.DataFrame
    year_aspect_sentiment: pd.DataFrame
    pairwise: pd.DataFrame
    findings: pd.DataFrame
    cautions: pd.DataFrame
    topics: pd.DataFrame
    topic_aspect: pd.DataFrame
    languages: pd.DataFrame
    language_aspect: pd.DataFrame


def _read_csv(root: Path, name: str) -> pd.DataFrame:
    path = root / name
    if not path.exists():
        raise FileNotFoundError(f"Frozen dashboard artifact is missing: {path}")
    return pd.read_csv(path)


def load_dashboard_data(root: Path = DASHBOARD_ROOT) -> DashboardData:
    from .cloud_bundle import load_cloud_bundle
    bundle = load_cloud_bundle("dashboard") if root == DASHBOARD_ROOT else None
    if bundle is not None:
        data = DashboardData(**bundle)
        validate_dashboard_data(data)
        return data
    metadata_path = root / "dashboard_metadata_v1.json"
    corpus_path = root / "dashboard_corpus_overview_v1.json"
    if not metadata_path.exists() or not corpus_path.exists():
        raise FileNotFoundError(f"Frozen dashboard metadata is unavailable under {root}")
    data = DashboardData(
        metadata=json.loads(metadata_path.read_text(encoding="utf-8")),
        corpus=json.loads(corpus_path.read_text(encoding="utf-8")),
        sentiment_overall=_read_csv(root, "dashboard_sentiment_overall_v1.csv"),
        aspects=_read_csv(root, "dashboard_aspect_overview_v1.csv"),
        aspect_sentiment=_read_csv(root, "dashboard_aspect_sentiment_v1.csv"),
        years=_read_csv(root, "dashboard_year_overview_v1.csv"),
        year_aspect=_read_csv(root, "dashboard_year_aspect_v1.csv"),
        year_aspect_sentiment=_read_csv(root, "dashboard_year_aspect_sentiment_v1.csv"),
        pairwise=_read_csv(root, "dashboard_pairwise_year_v1.csv"),
        findings=_read_csv(root, "dashboard_research_findings_v1.csv"),
        cautions=_read_csv(root, "dashboard_caution_findings_v1.csv"),
        topics=_read_csv(root, "dashboard_topic_overview_v1.csv"),
        topic_aspect=_read_csv(root, "dashboard_topic_aspect_long_v1.csv"),
        languages=_read_csv(root, "dashboard_language_overview_v1.csv"),
        language_aspect=_read_csv(root, "dashboard_language_aspect_long_v1.csv"),
    )
    validate_dashboard_data(data)
    return data


def validate_dashboard_data(data: DashboardData) -> None:
    errors: list[str] = []
    if data.corpus.get("total_documents") != EXPECTED_DOCUMENTS: errors.append("documents != 7,704")
    if data.corpus.get("total_mentions") != EXPECTED_MENTIONS: errors.append("mentions != 15,486")
    if len(data.aspects) != EXPECTED_ASPECTS or data.aspects.aspect.nunique() != EXPECTED_ASPECTS: errors.append("aspects != 20")
    if set(data.years.year.astype(int)) != EXPECTED_YEARS: errors.append("event years changed")
    if len(data.topics) != EXPECTED_TOPICS or data.topics.topic_id.nunique() != EXPECTED_TOPICS: errors.append("topics != 32")
    if int(data.years.documents.sum()) != EXPECTED_DOCUMENTS: errors.append("year document total changed")
    if int(data.years.mentions.sum()) != EXPECTED_MENTIONS: errors.append("year mention total changed")
    if int(data.sentiment_overall.mention_count.sum()) != EXPECTED_MENTIONS: errors.append("sentiment total changed")
    if set(data.metadata.get("years", [])) != EXPECTED_YEARS: errors.append("metadata years changed")
    if data.metadata.get("inferential_unit") != "document": errors.append("inferential unit is not document")
    if data.metadata.get("language_inference_allowed") is not False: errors.append("language inference unexpectedly enabled")
    if data.metadata.get("topic_inference_allowed") is not False: errors.append("topic inference unexpectedly enabled")
    if errors:
        raise ValueError("Frozen dashboard data validation failed: " + "; ".join(errors))


def display_label_map(data: DashboardData) -> dict[str, str]:
    return dict(data.metadata["aspect_display_labels"])
