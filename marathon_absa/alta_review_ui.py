from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from . import aspect_level_themes_review as review


WORKBOOK_NAME = "cluster_review_workbook.csv"
MAPPING_NAME = "cluster_review_mapping.csv"
EDITABLE_COLUMNS = [
    "theme_quality",
    "review_decision",
    "researcher_theme_label",
    "merge_target_theme_key",
    "researcher_notes",
    "review_status",
]
IDENTITY_COLUMNS = ["aspect", "theme_id", "cluster_key", "provisional_label"]


def assert_review_path(path: Path, review_root: Path = review.OUTPUT_ROOT) -> Path:
    """Return a resolved path only when it is inside the review namespace."""
    resolved = path.resolve()
    root = review_root.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Refusing write outside researcher-review namespace: {resolved}")
    return resolved


def load_review_data(
    review_root: Path = review.OUTPUT_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    workbook_path = review_root / WORKBOOK_NAME
    mapping_path = review_root / MAPPING_NAME
    if not workbook_path.exists() or not mapping_path.exists():
        missing = [str(p) for p in (workbook_path, mapping_path) if not p.exists()]
        raise FileNotFoundError("Missing review artifact(s): " + ", ".join(missing))
    workbook = pd.read_csv(workbook_path, keep_default_na=False, encoding="utf-8-sig")
    mapping = pd.read_csv(mapping_path, keep_default_na=False, encoding="utf-8-sig")
    if not set(IDENTITY_COLUMNS + EDITABLE_COLUMNS).issubset(mapping.columns):
        raise ValueError("Review mapping schema is incomplete")
    if workbook[["aspect", "theme_id", "cluster_key"]].duplicated().any():
        raise ValueError("Review workbook contains duplicate cluster identities")
    joined = workbook.drop(columns=[c for c in EDITABLE_COLUMNS if c in workbook], errors="ignore").merge(
        mapping[IDENTITY_COLUMNS + EDITABLE_COLUMNS],
        on=IDENTITY_COLUMNS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if len(joined) != len(workbook) or not joined.pop("_merge").eq("both").all():
        raise ValueError("Workbook and mapping cluster identities do not align")
    return workbook, mapping, joined


def aspect_order(joined: pd.DataFrame) -> list[str]:
    totals = joined.groupby("aspect", sort=False).support_documents.sum()
    first_seen = {aspect: index for index, aspect in enumerate(joined.aspect.unique())}
    return sorted(totals.index, key=lambda aspect: (-float(totals[aspect]), first_seen[aspect]))


def clusters_for_aspect(joined: pd.DataFrame, aspect: str) -> pd.DataFrame:
    return joined[joined.aspect.eq(aspect)].sort_values(
        ["support_documents", "theme_id"], ascending=[False, True]
    ).reset_index(drop=True)


def progress_summary(mapping: pd.DataFrame) -> dict[str, Any]:
    reviewed = mapping.review_status.eq("reviewed")
    by_aspect = (
        mapping.assign(_reviewed=reviewed.astype(int))
        .groupby("aspect", sort=False)
        .agg(total=("cluster_key", "size"), reviewed=("_reviewed", "sum"))
        .reset_index()
    )
    by_aspect["pending"] = by_aspect.total - by_aspect.reviewed
    return {
        "total": len(mapping),
        "reviewed": int(reviewed.sum()),
        "pending": int((~reviewed).sum()),
        "percent": (100.0 * reviewed.mean()) if len(mapping) else 0.0,
        "by_aspect": by_aspect,
        "decisions": mapping.loc[reviewed, "review_decision"].value_counts().to_dict(),
        "qualities": mapping.loc[reviewed, "theme_quality"].value_counts().to_dict(),
    }


def merge_options(joined: pd.DataFrame, cluster_key: str) -> pd.DataFrame:
    current = joined.loc[joined.cluster_key.eq(cluster_key)]
    if len(current) != 1:
        raise ValueError(f"Unknown or duplicate cluster key: {cluster_key}")
    aspect = current.iloc[0].aspect
    return clusters_for_aspect(joined, aspect).loc[
        lambda frame: frame.cluster_key.ne(cluster_key)
    ]


def validate_row_values(
    mapping: pd.DataFrame, cluster_key: str, values: Mapping[str, Any]
) -> pd.DataFrame:
    candidate = mapping.copy()
    matches = candidate.index[candidate.cluster_key.eq(cluster_key)]
    if len(matches) != 1:
        raise ValueError(f"Expected one mapping row for {cluster_key}")
    quality = str(values.get("theme_quality", "")).strip()
    decision = str(values.get("review_decision", "")).strip()
    label = str(values.get("researcher_theme_label", "")).strip()
    target = str(values.get("merge_target_theme_key", "")).strip()
    notes = str(values.get("researcher_notes", "")).strip()
    if quality not in review.QUALITIES:
        raise ValueError("Choose a theme quality")
    if decision not in review.DECISIONS:
        raise ValueError("Choose a review decision")
    if decision == "RENAME" and not label:
        raise ValueError("RENAME requires a researcher theme label")
    if decision == "MERGE":
        source_aspect = str(candidate.loc[matches[0], "aspect"])
        targets = candidate.loc[candidate.cluster_key.eq(target)]
        if target == cluster_key:
            raise ValueError("A cluster cannot merge into itself")
        if len(targets) != 1 or str(targets.iloc[0].aspect) != source_aspect:
            raise ValueError("MERGE target must be another cluster in the same aspect")
    elif target:
        raise ValueError("Only MERGE may have a merge target")
    candidate.loc[matches[0], EDITABLE_COLUMNS] = [
        quality, decision, label, target, notes, "reviewed"
    ]
    review.validate_review_mapping(candidate, require_complete=False)
    return candidate


def save_review_row(
    mapping_path: Path,
    cluster_key: str,
    values: Mapping[str, Any],
    review_root: Path = review.OUTPUT_ROOT,
    create_backup: bool = False,
) -> tuple[pd.DataFrame, Path | None]:
    mapping_path = assert_review_path(mapping_path, review_root)
    mapping = pd.read_csv(mapping_path, keep_default_na=False, encoding="utf-8-sig")
    updated = validate_row_values(mapping, cluster_key, values)
    backup_path: Path | None = None
    if create_backup:
        backup_dir = assert_review_path(review_root / "backups", review_root)
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        backup_path = backup_dir / f"cluster_review_mapping_{stamp}.csv"
        shutil.copy2(mapping_path, backup_path)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8-sig", newline="", suffix=".csv",
            prefix=".cluster_review_mapping_", dir=mapping_path.parent, delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            updated.to_csv(temporary, index=False, lineterminator="\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, mapping_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return updated, backup_path


def validation_issues(mapping: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    pending = mapping.loc[~mapping.review_status.eq("reviewed")]
    if len(pending):
        issues.append(f"{len(pending)} pending clusters")
    for row in mapping.loc[mapping.review_status.eq("reviewed")].itertuples():
        if row.review_decision == "RENAME" and not str(row.researcher_theme_label).strip():
            issues.append(f"RENAME cluster {row.cluster_key} is missing a researcher label")
        if row.review_decision == "MERGE" and not str(row.merge_target_theme_key).strip():
            issues.append(f"MERGE cluster {row.cluster_key} is missing a merge target")
    try:
        review.validate_review_mapping(mapping, require_complete=False)
    except ValueError as exc:
        issues.append(str(exc))
    return list(dict.fromkeys(issues))


def finalization_ready(mapping: pd.DataFrame) -> tuple[bool, str]:
    try:
        review.validate_review_mapping(mapping, require_complete=True)
    except ValueError as exc:
        return False, str(exc)
    return True, "Review mapping validation passed."
