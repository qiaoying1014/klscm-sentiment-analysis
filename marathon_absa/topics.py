from __future__ import annotations

import json

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP

from .config import Settings


def fit_topics(units: pd.DataFrame, embeddings: list[list[float]], settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame, BERTopic]:
    if len(units) < 10:
        raise ValueError("At least 10 relevant units are required for topic discovery.")
    min_cluster = max(5, min(30, len(units) // 50))
    model = BERTopic(
        umap_model=UMAP(n_neighbors=min(15, len(units) - 1), n_components=5, min_dist=0.0, metric="cosine", random_state=settings.random_seed),
        hdbscan_model=HDBSCAN(min_cluster_size=min_cluster, metric="euclidean", cluster_selection_method="eom", prediction_data=True),
        language="multilingual", calculate_probabilities=False, verbose=False,
    )
    topics, probabilities = model.fit_transform(units["text"].tolist(), np.asarray(embeddings))
    assignments = units[["unit_id", "document_id", "source", "event_year", "detected_language", "text"]].copy()
    assignments["topic_id"] = topics
    assignments["topic_probability"] = probabilities if probabilities is not None else np.nan
    info = model.get_topic_info().rename(columns={"Topic": "topic_id", "Count": "count", "Name": "topic_name", "Representation": "representation", "Representative_Docs": "representative_docs"})
    for column in ["representation", "representative_docs"]:
        if column in info:
            info[column] = info[column].map(lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value)
    return assignments, info, model


def initial_taxonomy(topic_info: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in topic_info.itertuples(index=False):
        if int(row.topic_id) == -1:
            continue
        rows.append({"topic_id": int(row.topic_id), "discovered_label": row.topic_name, "approved_aspect": "", "description": "", "approved": False, "researcher_notes": ""})
    return pd.DataFrame(rows)

