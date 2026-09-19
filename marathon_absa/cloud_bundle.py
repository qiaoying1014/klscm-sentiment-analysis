"""Load the portable, prevalidated dashboard release used by Streamlit Cloud."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


def load_cloud_bundle(section: str) -> dict[str, Any] | None:
    """Verify every selected file; local research loaders remain the default."""
    location = os.environ.get("KLSCM_DASHBOARD_BUNDLE")
    if not location:
        return None
    root = Path(location).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("validation") != "passed":
        raise ValueError("Dashboard deployment bundle is not validated")
    result = {}
    for key, item in manifest["sections"][section].items():
        path = (root / item["file"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Deployment artifact escapes bundle directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Deployment artifact hash mismatch: {item['file']}")
        if item["format"] == "parquet":
            result[key] = pd.read_parquet(path)
        elif item["format"] == "json":
            result[key] = json.loads(path.read_text(encoding="utf-8"))
        else:
            raise ValueError("Unknown deployment artifact format")
    return result
