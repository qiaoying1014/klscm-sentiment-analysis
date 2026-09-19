"""Export a portable release only after all local dashboard validators pass.

Run from the repository root: python scripts/export_dashboard_bundle.py
"""
from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from marathon_absa.blog_dashboard_data import load_blog_dashboard_data
from marathon_absa.dashboard_data import load_dashboard_data
from marathon_absa.participant_experience_dashboard_data import load_participant_experience_dashboard_data
from marathon_absa.reviewed_theme_dashboard_data import load_reviewed_theme_dashboard_data
from marathon_absa.wordcloud_data import load_wordcloud_sources


def main() -> None:
    os.chdir(ROOT)
    if os.environ.get("KLSCM_DASHBOARD_BUNDLE"):
        raise RuntimeError("Unset KLSCM_DASHBOARD_BUNDLE: export must validate original research artifacts")
    sections = {}
    for name, loader in [("dashboard", load_dashboard_data),
                         ("themes", load_reviewed_theme_dashboard_data),
                         ("blog", load_blog_dashboard_data)]:
        value = loader()
        sections[name] = {field.name: getattr(value, field.name) for field in fields(value)}
    sections["experience"] = load_participant_experience_dashboard_data()
    documents, mentions = load_wordcloud_sources()
    sections["wordcloud"] = {"documents": documents, "mentions": mentions}
    destination = ROOT / "deploy" / "dashboard_data"
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "validation": "passed",
                "exported_at_utc": datetime.now(timezone.utc).isoformat(),
                "validation_scope": "Original dashboard, reviewed-theme, blog, participant-experience and wordcloud loaders; full local source checks before export",
                "sections": {}}
    for section, values in sections.items():
        manifest["sections"][section] = {}
        for key, value in values.items():
            extension = "parquet" if isinstance(value, pd.DataFrame) else "json"
            path = destination / f"{section}_{key}.{extension}"
            if extension == "parquet":
                value.to_parquet(path, index=False)
            else:
                path.write_bytes(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))
            manifest["sections"][section][key] = {
                "file": path.name, "format": extension,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
    # Publish the manifest last; partial exports fail the Cloud hash checks.
    (destination / "manifest.json").write_bytes(json.dumps(manifest, indent=2).encode("utf-8"))
    count = sum(len(items) for items in manifest["sections"].values())
    size = sum(item["bytes"] for items in manifest["sections"].values() for item in items.values())
    print(f"Validated and exported {count} artifacts ({size / 1024**2:.2f} MiB) to {destination}")


if __name__ == "__main__":
    main()
