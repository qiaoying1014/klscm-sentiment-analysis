from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .text import extract_emojis, extract_hashtags, linguistic_text, normalized_hash, repair_and_normalize, semantic_text, stable_id

DOCUMENT_COLUMNS = [
    "document_id", "source", "author", "url", "event_year", "timestamp",
    "original_text", "normalized_text", "linguistic_text", "semantic_text",
    "hashtags", "emojis", "emoji_aliases", "text_hash", "duplicate_of",
    "processing_status",
]


def _year(value: object, fallback: object = "") -> str:
    text = str(value or fallback or "")
    match = pd.Series([text]).str.extract(r"((?:19|20)\d{2})", expand=False).iloc[0]
    return "" if pd.isna(match) else str(match)


def _record(source: str, author: object, url: object, year: object, timestamp: object, text: object, ordinal: int) -> dict:
    original = "" if text is None else str(text)
    normalized = repair_and_normalize(original)
    tags = extract_hashtags(normalized)
    emojis, aliases = extract_emojis(normalized)
    return {
        "document_id": stable_id(source, url, author, timestamp, ordinal),
        "source": source,
        "author": str(author or ""),
        "url": str(url or ""),
        "event_year": _year(year, timestamp),
        "timestamp": str(timestamp or ""),
        "original_text": original,
        "normalized_text": normalized,
        "linguistic_text": linguistic_text(normalized),
        "semantic_text": semantic_text(normalized),
        "hashtags": tags,
        "emojis": emojis,
        "emoji_aliases": aliases,
        "text_hash": normalized_hash(normalized),
        "duplicate_of": "",
        "processing_status": "ready" if normalized else "empty",
    }


def load_documents(instagram_path: Path, blog_path: Path) -> pd.DataFrame:
    insta = pd.read_csv(instagram_path, dtype=str, keep_default_na=False)
    with blog_path.open(encoding="utf-8") as handle:
        blogs = json.load(handle)

    records: list[dict] = []
    for i, row in insta.iterrows():
        records.append(_record("instagram", row.get("author_id"), row.get("post_url"), row.get("hashtag"), row.get("timestamp"), row.get("caption"), int(i)))
    for i, row in enumerate(blogs):
        records.append(_record("blog", row.get("author"), row.get("url"), row.get("year"), row.get("year"), row.get("review"), i))

    frame = pd.DataFrame(records, columns=DOCUMENT_COLUMNS)
    first_by_hash: dict[str, str] = {}
    for idx, row in frame.iterrows():
        if row["processing_status"] == "empty":
            continue
        if row["text_hash"] in first_by_hash:
            frame.at[idx, "duplicate_of"] = first_by_hash[row["text_hash"]]
            frame.at[idx, "processing_status"] = "duplicate"
        else:
            first_by_hash[row["text_hash"]] = row["document_id"]
    return frame

