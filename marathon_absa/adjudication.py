from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

WORKFLOW_VERSION = "v8_gold_adjudication_v1"
FINAL_LABELS = {"include", "exclude"}
ANNOTATION_COLUMNS = [
    "record_id",
    "caption",
    "language",
    "event_year",
    "reviewer_label",
    "evidence_span",
    "rationale",
    "reviewer_confidence",
    "optional_comments",
]
REVIEWER_FIELDS = [
    "reviewer_label",
    "evidence_span",
    "rationale",
    "reviewer_confidence",
    "optional_comments",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def build_blinded_annotation(
    candidates: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """Return only permitted fields in reproducibly randomized order."""
    required = {
        "record_id", "original_caption", "language", "event_year",
        "disagreement_reason",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate file is missing columns: {sorted(missing)}")
    disputed = candidates[candidates.disagreement_reason.str.strip().ne("")].copy()
    if disputed.record_id.duplicated().any():
        raise ValueError("Disputed candidate IDs must be unique")
    blinded = disputed[
        ["record_id", "original_caption", "language", "event_year"]
    ].rename(columns={"original_caption": "caption"})
    blinded = blinded.sample(frac=1, random_state=seed).reset_index(drop=True)
    for column in REVIEWER_FIELDS:
        blinded[column] = ""
    return blinded[ANNOTATION_COLUMNS]


def initialize_adjudication_package(
    candidates_path: Path,
    output_dir: Path,
    seed: int = 8042,
) -> dict:
    """Create a non-overwriting blind package and provenance manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    template_path = output_dir / "relevance_v8_blinded_annotation.csv"
    reviewer_a_path = output_dir / "relevance_v8_reviewer_a.csv"
    reviewer_b_path = output_dir / "relevance_v8_reviewer_b.csv"
    manifest_path = output_dir / "adjudication_manifest.json"
    targets = [template_path, reviewer_a_path, reviewer_b_path, manifest_path]
    if any(path.exists() for path in targets):
        raise FileExistsError("Adjudication package exists; refusing to overwrite")

    candidates = _read_csv(candidates_path)
    blinded = build_blinded_annotation(candidates, seed)
    blinded.to_csv(template_path, index=False, encoding="utf-8-sig")
    # Separate physical files prevent one reviewer from seeing another's edits.
    blinded.to_csv(reviewer_a_path, index=False, encoding="utf-8-sig")
    blinded.to_csv(reviewer_b_path, index=False, encoding="utf-8-sig")
    created_at = utc_now()
    manifest = {
        "workflow_version": WORKFLOW_VERSION,
        "created_at_utc": created_at,
        "random_seed": seed,
        "candidate_source": str(candidates_path),
        "candidate_source_sha256": sha256_file(candidates_path),
        "disputed_records": len(blinded),
        "annotation_columns": ANNOTATION_COLUMNS,
        "hidden_fields": [
            "historical label", "model prediction", "confidence",
            "model explanation", "routing", "reason code",
            "event connection", "content type",
        ],
        "files": {
            path.name: sha256_file(path)
            for path in [template_path, reviewer_a_path, reviewer_b_path]
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return manifest


def validate_completed_review(
    review: pd.DataFrame,
    expected_ids: set[str],
) -> None:
    missing = set(ANNOTATION_COLUMNS) - set(review.columns)
    if missing:
        raise ValueError(f"Reviewer file is missing columns: {sorted(missing)}")
    if len(review) != len(expected_ids) or set(review.record_id) != expected_ids:
        raise ValueError("Reviewer file does not match the blinded record set")
    if review.record_id.duplicated().any():
        raise ValueError("Reviewer file contains duplicate record IDs")
    labels = review.reviewer_label.str.strip().str.lower()
    if not labels.isin(FINAL_LABELS).all():
        raise ValueError("Every reviewer label must be include or exclude")
    if review.evidence_span.str.strip().eq("").any():
        raise ValueError("Every review requires an evidence span")
    if review.rationale.str.strip().eq("").any():
        raise ValueError("Every review requires a rationale")
    confidence = pd.to_numeric(review.reviewer_confidence, errors="coerce")
    if confidence.isna().any() or ~confidence.between(0, 1).all():
        raise ValueError("Reviewer confidence must be numeric from 0 to 1")


def import_reviewer_file(
    completed_path: Path,
    reviewer_id: str,
    reviewer_version: str,
    package_dir: Path,
) -> Path:
    """Import a complete reviewer file as an immutable, append-only snapshot."""
    reviewer_id = reviewer_id.strip()
    reviewer_version = reviewer_version.strip()
    if not reviewer_id or not reviewer_version:
        raise ValueError("Reviewer identity and version are required")
    template = _read_csv(package_dir / "relevance_v8_blinded_annotation.csv")
    completed = _read_csv(completed_path)
    validate_completed_review(completed, set(template.record_id))

    imports_dir = package_dir / "reviewer_imports"
    imports_dir.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(c for c in reviewer_id if c.isalnum() or c in "-_")
    safe_version = "".join(
        c for c in reviewer_version if c.isalnum() or c in "-_."
    )
    destination = imports_dir / f"{safe_id}__{safe_version}.csv"
    metadata_path = destination.with_suffix(".metadata.json")
    if destination.exists() or metadata_path.exists():
        raise FileExistsError("Reviewer/version import exists; refusing to overwrite")

    imported_at = utc_now()
    snapshot = completed[ANNOTATION_COLUMNS].copy()
    snapshot["reviewer_label"] = (
        snapshot.reviewer_label.str.strip().str.lower()
    )
    snapshot["reviewer_id"] = reviewer_id
    snapshot["reviewer_version"] = reviewer_version
    snapshot["imported_at_utc"] = imported_at
    snapshot["original_file"] = str(completed_path)
    snapshot["original_file_sha256"] = sha256_file(completed_path)
    snapshot.to_csv(destination, index=False, encoding="utf-8-sig")
    metadata_path.write_text(json.dumps({
        "workflow_version": WORKFLOW_VERSION,
        "reviewer_id": reviewer_id,
        "reviewer_version": reviewer_version,
        "imported_at_utc": imported_at,
        "original_file": str(completed_path),
        "original_file_sha256": sha256_file(completed_path),
        "imported_snapshot": str(destination),
        "imported_snapshot_sha256": sha256_file(destination),
        "records": len(snapshot),
    }, indent=2), encoding="utf-8")
    return destination


def _metric_row(
    reviewer_a: pd.Series,
    reviewer_b: pd.Series,
    dimension: str,
    group: str,
) -> dict:
    agreement = reviewer_a.eq(reviewer_b)
    kappa = (
        float(cohen_kappa_score(reviewer_a, reviewer_b))
        if len(reviewer_a) > 1 else None
    )
    return {
        "dimension": dimension,
        "group": group,
        "n": len(reviewer_a),
        "percent_agreement": float(agreement.mean()),
        "cohen_kappa": kappa,
    }


def analyze_agreement(
    reviewer_a_path: Path,
    reviewer_b_path: Path,
    candidates_path: Path,
    output_dir: Path,
) -> dict:
    """Compute overall/subgroup agreement and write disagreement artifacts."""
    a = _read_csv(reviewer_a_path)
    b = _read_csv(reviewer_b_path)
    candidates = _read_csv(candidates_path)
    required_import = {
        "record_id", "reviewer_label", "reviewer_id", "reviewer_version",
        "evidence_span", "rationale", "optional_comments",
    }
    if not required_import.issubset(a.columns) or not required_import.issubset(b.columns):
        raise ValueError("Use immutable imported reviewer snapshots")
    if a.reviewer_id.nunique() != 1 or b.reviewer_id.nunique() != 1:
        raise ValueError("Each imported snapshot must contain one reviewer")
    if a.reviewer_id.iloc[0] == b.reviewer_id.iloc[0]:
        raise ValueError("Agreement requires two independent reviewer identities")
    if set(a.record_id) != set(b.record_id):
        raise ValueError("Reviewer snapshots do not contain the same records")

    a_fields = a[[
        "record_id", "reviewer_label", "evidence_span", "rationale",
        "reviewer_confidence", "optional_comments", "reviewer_id",
        "reviewer_version", "imported_at_utc", "original_file",
        "original_file_sha256",
    ]].rename(columns={column: f"reviewer_a_{column}" for column in a.columns if column != "record_id"})
    b_fields = b[[
        "record_id", "reviewer_label", "evidence_span", "rationale",
        "reviewer_confidence", "optional_comments", "reviewer_id",
        "reviewer_version", "imported_at_utc", "original_file",
        "original_file_sha256",
    ]].rename(columns={column: f"reviewer_b_{column}" for column in b.columns if column != "record_id"})
    merged = a_fields.merge(b_fields, on="record_id", validate="one_to_one")
    hidden = candidates[[
        "record_id", "original_caption", "language", "event_year",
        "event_connection", "primary_content_type", "human_label_v7",
    ]]
    merged = merged.merge(hidden, on="record_id", validate="one_to_one")
    merged["agrees"] = merged.reviewer_a_reviewer_label.eq(
        merged.reviewer_b_reviewer_label
    )

    rows = [_metric_row(
        merged.reviewer_a_reviewer_label,
        merged.reviewer_b_reviewer_label,
        "overall",
        "all",
    )]
    for dimension in ["language", "event_connection", "primary_content_type"]:
        for group, frame in merged.groupby(dimension, dropna=False):
            rows.append(_metric_row(
                frame.reviewer_a_reviewer_label,
                frame.reviewer_b_reviewer_label,
                dimension,
                str(group),
            ))
    stats = pd.DataFrame(rows)
    matrix = confusion_matrix(
        merged.reviewer_a_reviewer_label,
        merged.reviewer_b_reviewer_label,
        labels=["include", "exclude"],
    )
    disagreements = merged[~merged.agrees].copy()
    report = disagreements[[
        "record_id", "original_caption", "language", "event_year",
        "reviewer_a_reviewer_label", "reviewer_b_reviewer_label",
        "reviewer_a_evidence_span", "reviewer_b_evidence_span",
        "reviewer_a_rationale", "reviewer_b_rationale",
        "reviewer_a_optional_comments", "reviewer_b_optional_comments",
    ]].rename(columns={"original_caption": "caption"})
    report["adjudicated_v8_label"] = ""
    report["adjudicator"] = ""
    report["adjudication_rationale"] = ""
    report["adjudicated_at_utc"] = ""

    output_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(output_dir / "agreement_statistics.csv", index=False)
    pd.DataFrame(
        matrix,
        index=["reviewer_a_include", "reviewer_a_exclude"],
        columns=["reviewer_b_include", "reviewer_b_exclude"],
    ).to_csv(output_dir / "agreement_confusion_matrix.csv")
    disagreements.to_csv(
        output_dir / "reviewer_disagreements.csv", index=False,
        encoding="utf-8-sig",
    )
    report.to_csv(
        output_dir / "relevance_v8_adjudication_report.csv", index=False,
        encoding="utf-8-sig",
    )
    overall = rows[0]
    summary = {
        "workflow_version": WORKFLOW_VERSION,
        "reviewer_a": a.reviewer_id.iloc[0],
        "reviewer_b": b.reviewer_id.iloc[0],
        "records": len(merged),
        "agreements": int(merged.agrees.sum()),
        "disagreements": int((~merged.agrees).sum()),
        "percent_agreement": overall["percent_agreement"],
        "cohen_kappa": overall["cohen_kappa"],
    }
    (output_dir / "agreement_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    return summary


def finalize_gold_standard(
    reviewer_a_path: Path,
    reviewer_b_path: Path,
    candidates_path: Path,
    adjudication_report_path: Path,
    output_dir: Path,
    gold_version: str,
) -> Path:
    """Create a new immutable gold version from consensus plus adjudication."""
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / f"relevance_v8_gold_{gold_version}.csv"
    manifest_path = final_path.with_suffix(".manifest.json")
    if final_path.exists() or manifest_path.exists():
        raise FileExistsError("Gold version exists; refusing to overwrite")
    a = _read_csv(reviewer_a_path)
    b = _read_csv(reviewer_b_path)
    candidates = _read_csv(candidates_path)
    report = _read_csv(adjudication_report_path)
    if set(a.record_id) != set(b.record_id):
        raise ValueError("Reviewer snapshots do not align")
    merged = a.merge(
        b,
        on="record_id",
        suffixes=("_reviewer1", "_reviewer2"),
        validate="one_to_one",
    ).merge(
        candidates[[
            "record_id", "human_label_v7", "original_caption", "language",
            "event_year",
        ]],
        on="record_id",
        validate="one_to_one",
    )
    disagreements = merged.reviewer_label_reviewer1.ne(
        merged.reviewer_label_reviewer2
    )
    expected = set(merged.loc[disagreements, "record_id"])
    if set(report.record_id) != expected:
        raise ValueError("Adjudication report must contain exactly the disagreements")
    if expected:
        labels = report.adjudicated_v8_label.str.strip().str.lower()
        if not labels.isin(FINAL_LABELS).all():
            raise ValueError("Every disagreement requires include or exclude")
        for column in [
            "adjudicator", "adjudication_rationale", "adjudicated_at_utc",
        ]:
            if report[column].str.strip().eq("").any():
                raise ValueError(f"Every disagreement requires {column}")
    decisions = dict(zip(
        report.record_id,
        report.adjudicated_v8_label.str.strip().str.lower(),
    ))
    merged["adjudicated_v8_label"] = merged.apply(
        lambda row: (
            row.reviewer_label_reviewer1
            if row.reviewer_label_reviewer1 == row.reviewer_label_reviewer2
            else decisions[row.record_id]
        ),
        axis=1,
    )
    adjudicator = dict(zip(report.record_id, report.adjudicator))
    adjudicated_at = dict(zip(report.record_id, report.adjudicated_at_utc))
    rationale = dict(zip(report.record_id, report.adjudication_rationale))
    merged["adjudicator"] = merged.record_id.map(adjudicator).fillna(
        "reviewer_consensus"
    )
    merged["adjudicated_at_utc"] = merged.record_id.map(
        adjudicated_at
    ).fillna(utc_now())
    merged["adjudication_rationale"] = merged.record_id.map(
        rationale
    ).fillna("Independent reviewers agreed.")
    final = merged.rename(columns={
        "human_label_v7": "historical_v7_label",
        "reviewer_label_reviewer1": "reviewer1_label",
        "reviewer_label_reviewer2": "reviewer2_label",
        "reviewer_id_reviewer1": "reviewer1_identity",
        "reviewer_id_reviewer2": "reviewer2_identity",
        "reviewer_version_reviewer1": "reviewer1_version",
        "reviewer_version_reviewer2": "reviewer2_version",
    })
    keep = [
        "record_id", "original_caption", "language", "event_year",
        "historical_v7_label", "reviewer1_label", "reviewer2_label",
        "adjudicated_v8_label", "reviewer1_identity", "reviewer2_identity",
        "reviewer1_version", "reviewer2_version", "adjudicator",
        "adjudicated_at_utc", "adjudication_rationale",
        "evidence_span_reviewer1", "evidence_span_reviewer2",
        "rationale_reviewer1", "rationale_reviewer2",
        "optional_comments_reviewer1", "optional_comments_reviewer2",
        "original_file_reviewer1", "original_file_reviewer2",
        "original_file_sha256_reviewer1", "original_file_sha256_reviewer2",
    ]
    final[keep].to_csv(final_path, index=False, encoding="utf-8-sig")
    manifest_path.write_text(json.dumps({
        "workflow_version": WORKFLOW_VERSION,
        "gold_version": gold_version,
        "created_at_utc": utc_now(),
        "records": len(final),
        "disagreements_adjudicated": len(expected),
        "reviewer_a_snapshot": str(reviewer_a_path),
        "reviewer_a_sha256": sha256_file(reviewer_a_path),
        "reviewer_b_snapshot": str(reviewer_b_path),
        "reviewer_b_sha256": sha256_file(reviewer_b_path),
        "adjudication_report": str(adjudication_report_path),
        "adjudication_report_sha256": sha256_file(adjudication_report_path),
        "candidate_source": str(candidates_path),
        "candidate_source_sha256": sha256_file(candidates_path),
        "gold_file": str(final_path),
        "gold_file_sha256": sha256_file(final_path),
    }, indent=2), encoding="utf-8")
    return final_path
