from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

LIST_COLUMNS = {"hashtags", "emojis", "emoji_aliases", "contributing_hashtags", "contributing_emojis", "detected_languages"}


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serial = frame.copy()
    for column in serial.columns.intersection(LIST_COLUMNS):
        serial[column] = serial[column].map(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x)
    serial.to_parquet(path, index=False)
    serial.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def read_table(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    for column in frame.columns.intersection(LIST_COLUMNS):
        frame[column] = frame[column].map(lambda x: json.loads(x) if isinstance(x, str) and x.startswith("[") else ([] if x is None else x))
    return frame

