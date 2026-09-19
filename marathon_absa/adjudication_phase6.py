from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from .adjudication import ANNOTATION_COLUMNS, FINAL_LABELS, WORKFLOW_VERSION

HIDDEN_FIELDS = {
    "human_label_v7", "historical_v7_label", "model_prediction_v8",
    "model_confidence", "confidence", "routing_status", "reason_code",
    "short_explanation", "model_explanation", "proposed_v8_label",
    "event_connection", "primary_content_type",
}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _submission_checks(
    source: Path,
    template: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    review = _read(source)
    expected = set(template.record_id)
    report = {
        "source_file": str(source),
        "source_sha256": _sha256(source),
        "rows": len(review),
        "unique_record_ids": int(review.record_id.nunique())
        if "record_id" in review else 0,
        "duplicate_count": int(review.record_id.duplicated().sum())
        if "record_id" in review else 0,
        "missing_record_count": len(expected - set(review.record_id))
        if "record_id" in review else len(expected),
        "unexpected_record_count": len(set(review.record_id) - expected)
        if "record_id" in review else 0,
        "invalid_label_count": 0,
        "blank_evidence_count": 0,
        "blank_rationale_count": 0,
        "invalid_confidence_count": 0,
        "hidden_field_count": len(HIDDEN_FIELDS & set(review.columns)),
        "modified_caption_count": 0,
        "modified_language_count": 0,
        "modified_event_year_count": 0,
        "integrity_check_status": "failed",
        "errors": [],
    }
    missing_columns = set(ANNOTATION_COLUMNS) - set(review.columns)
    extra_columns = set(review.columns) - set(ANNOTATION_COLUMNS)
    if missing_columns:
        report["errors"].append(
            f"missing columns: {sorted(missing_columns)}"
        )
    if extra_columns:
        report["errors"].append(
            f"unexpected columns: {sorted(extra_columns)}"
        )
    if report["hidden_field_count"]:
        report["errors"].append("hidden classifier/history fields detected")
    if missing_columns or "record_id" not in review:
        return review, report

    labels = review.reviewer_label.str.strip().str.lower()
    report["invalid_label_count"] = int((~labels.isin(FINAL_LABELS)).sum())
    report["blank_evidence_count"] = int(
        review.evidence_span.str.strip().eq("").sum()
    )
    report["blank_rationale_count"] = int(
        review.rationale.str.strip().eq("").sum()
    )
    confidence = pd.to_numeric(review.reviewer_confidence, errors="coerce")
    report["invalid_confidence_count"] = int(
        (confidence.isna() | ~confidence.between(0, 1)).sum()
    )
    if set(review.record_id) == expected and not review.record_id.duplicated().any():
        aligned = review.set_index("record_id").loc[
            template.record_id
        ].reset_index()
        for column in ["caption", "language", "event_year"]:
            count = int(aligned[column].ne(template[column]).sum())
            report[f"modified_{column}_count"] = count
    numeric_failures = [
        key for key, value in report.items()
        if key.endswith("_count") and isinstance(value, int) and value != 0
    ]
    if len(review) != len(template):
        report["errors"].append("row count does not match template")
    if report["unique_record_ids"] != len(template):
        report["errors"].append("unique record count does not match template")
    if numeric_failures:
        report["errors"].append(
            f"nonzero validation counters: {numeric_failures}"
        )
    if not report["errors"]:
        report["integrity_check_status"] = "passed"
    return review, report


def preflight_two_reviewers(
    reviewer_a_path: Path,
    reviewer_b_path: Path,
    package_dir: Path,
    reviewer_a_id: str,
    reviewer_b_id: str,
    reviewer_a_version: str,
    reviewer_b_version: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Validate both submissions fully before either can be imported."""
    if reviewer_a_id.strip() == reviewer_b_id.strip():
        raise ValueError("Reviewer identities must be distinct")
    manifest_path = package_dir / "adjudication_manifest.json"
    template_path = package_dir / "relevance_v8_blinded_annotation.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    template_hash = _sha256(template_path)
    expected_hash = manifest["files"]["relevance_v8_blinded_annotation.csv"]
    if template_hash != expected_hash:
        raise ValueError("Canonical template hash does not match manifest")
    candidates_path = Path(manifest["candidate_source"])
    if _sha256(candidates_path) != manifest["candidate_source_sha256"]:
        raise ValueError("Phase 4 candidate source hash does not match manifest")
    template = _read(template_path)
    a, a_report = _submission_checks(reviewer_a_path, template)
    b, b_report = _submission_checks(reviewer_b_path, template)
    imports = package_dir / "reviewer_imports"
    target_a = imports / f"{reviewer_a_id}__{reviewer_a_version}.csv"
    target_b = imports / f"{reviewer_b_id}__{reviewer_b_version}.csv"
    append_only_clear = not any([
        target_a.exists(), target_a.with_suffix(".metadata.json").exists(),
        target_b.exists(), target_b.with_suffix(".metadata.json").exists(),
    ])
    report = {
        "workflow_version": WORKFLOW_VERSION,
        "validated_at_utc": _utc_now(),
        "canonical_template": str(template_path),
        "canonical_template_sha256": template_hash,
        "canonical_template_manifest_match": True,
        "candidate_source": str(candidates_path),
        "candidate_source_manifest_match": True,
        "reviewer_identities_distinct": True,
        "append_only_targets_available": append_only_clear,
        "reviewer_a": {
            "reviewer_id": reviewer_a_id,
            "reviewer_role": "independent_human_reviewer_a",
            "annotation_version": reviewer_a_version,
            **a_report,
        },
        "reviewer_b": {
            "reviewer_id": reviewer_b_id,
            "reviewer_role": "independent_human_reviewer_b",
            "annotation_version": reviewer_b_version,
            **b_report,
        },
    }
    report["overall_status"] = (
        "passed"
        if a_report["integrity_check_status"] == "passed"
        and b_report["integrity_check_status"] == "passed"
        and append_only_clear else "failed"
    )
    if report["overall_status"] != "passed":
        raise ValueError(json.dumps(report, ensure_ascii=False))
    return a, b, report


def import_validated_submission(
    review: pd.DataFrame,
    source_path: Path,
    reviewer_id: str,
    reviewer_role: str,
    annotation_version: str,
    package_dir: Path,
) -> Path:
    """Write a normalized immutable snapshot without changing source rows."""
    imports = package_dir / "reviewer_imports"
    imports.mkdir(parents=True, exist_ok=True)
    destination = imports / f"{reviewer_id}__{annotation_version}.csv"
    metadata_path = destination.with_suffix(".metadata.json")
    if destination.exists() or metadata_path.exists():
        raise FileExistsError("Reviewer import exists; refusing to overwrite")
    timestamp = _utc_now()
    snapshot = review[ANNOTATION_COLUMNS].copy()
    snapshot["reviewer_label"] = (
        snapshot.reviewer_label.str.strip().str.lower()
    )
    for column in [
        "evidence_span", "rationale", "reviewer_confidence",
        "optional_comments",
    ]:
        snapshot[column] = snapshot[column].str.strip()
    snapshot["reviewer_id"] = reviewer_id
    snapshot["reviewer_role"] = reviewer_role
    snapshot["reviewer_version"] = annotation_version
    snapshot["imported_at_utc"] = timestamp
    snapshot["original_file"] = str(source_path)
    snapshot["original_file_sha256"] = _sha256(source_path)
    snapshot.to_csv(destination, index=False, encoding="utf-8-sig")
    metadata = {
        "workflow_version": WORKFLOW_VERSION,
        "reviewer_id": reviewer_id,
        "reviewer_role": reviewer_role,
        "annotation_version": annotation_version,
        "imported_at_utc": timestamp,
        "original_file": str(source_path),
        "original_file_sha256": _sha256(source_path),
        "imported_file": str(destination),
        "imported_file_sha256": _sha256(destination),
        "record_count": len(snapshot),
        "rejected_row_count": 0,
        "duplicate_count": 0,
        "missing_record_count": 0,
        "invalid_label_count": 0,
        "integrity_check_status": "passed",
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8",
    )
    return destination


def _agreement_row(frame: pd.DataFrame, dimension: str, group: str) -> dict:
    a = frame.reviewer_a_label
    b = frame.reviewer_b_label
    agrees = a.eq(b)
    observed_labels = set(a) | set(b)
    kappa = (
        float(cohen_kappa_score(a, b))
        if len(frame) > 1 and len(observed_labels) > 1 else None
    )
    include_a = float(a.eq("include").mean())
    include_b = float(b.eq("include").mean())
    extreme_prevalence = (
        min(include_a, 1 - include_a, include_b, 1 - include_b) < 0.10
    )
    flags = []
    if len(frame) < 10:
        flags.append("small_n")
    if len(observed_labels) == 1:
        flags.append("one_class_only")
    if kappa is None:
        flags.append("kappa_undefined")
    if extreme_prevalence:
        flags.append("extreme_prevalence")
    return {
        "dimension": dimension,
        "group": group,
        "n": len(frame),
        "agreements": int(agrees.sum()),
        "disagreements": int((~agrees).sum()),
        "percent_agreement": float(agrees.mean()),
        "cohen_kappa": kappa,
        "reviewer_a_include_rate": include_a,
        "reviewer_b_include_rate": include_b,
        "stability_flag": "|".join(flags) if flags else "stable",
    }


def _length_bands(captions: pd.Series) -> pd.Series:
    lengths = captions.fillna("").str.len()
    ranked = lengths.rank(method="first")
    quartile = pd.qcut(ranked, q=4, labels=False, duplicates="drop")
    labels = {0: "shortest", 1: "short", 2: "long", 3: "longest"}
    return quartile.map(labels)


def analyze_phase6_agreement(
    reviewer_a_path: Path,
    reviewer_b_path: Path,
    candidates_path: Path,
    output_dir: Path,
    random_seed: int = 16084,
) -> dict:
    a = _read(reviewer_a_path)
    b = _read(reviewer_b_path)
    if a.reviewer_id.iloc[0] == b.reviewer_id.iloc[0]:
        raise ValueError("Agreement requires distinct reviewer identities")
    candidates = _read(candidates_path)
    a_fields = a[[
        "record_id", "reviewer_label", "evidence_span", "rationale",
        "reviewer_confidence", "optional_comments", "reviewer_id",
        "reviewer_role", "reviewer_version", "imported_at_utc",
        "original_file", "original_file_sha256",
    ]].rename(columns={
        column: f"reviewer_a_{column}"
        for column in a.columns if column != "record_id"
    })
    b_fields = b[[
        "record_id", "reviewer_label", "evidence_span", "rationale",
        "reviewer_confidence", "optional_comments", "reviewer_id",
        "reviewer_role", "reviewer_version", "imported_at_utc",
        "original_file", "original_file_sha256",
    ]].rename(columns={
        column: f"reviewer_b_{column}"
        for column in b.columns if column != "record_id"
    })
    metadata = candidates[[
        "record_id", "original_caption", "language", "event_year",
        "candidate_scope", "disagreement_category",
    ]].rename(columns={
        "language": "recorded_language",
        "candidate_scope": "phase4_candidate_origin",
        "disagreement_category": "phase4_disagreement_category",
    })
    merged = (
        a_fields.merge(b_fields, on="record_id", validate="one_to_one")
        .merge(metadata, on="record_id", validate="one_to_one")
    )
    merged["caption_length_band"] = _length_bands(
        merged.original_caption
    )
    merged["hashtag_presence"] = merged.original_caption.str.contains(
        r"(?<!\w)#[\w]+", regex=True, na=False,
    ).map({True: "has_hashtag", False: "no_hashtag"})
    merged["agrees"] = merged.reviewer_a_reviewer_label.eq(
        merged.reviewer_b_reviewer_label
    )
    analysis = merged.rename(columns={
        "reviewer_a_reviewer_label": "reviewer_a_label",
        "reviewer_b_reviewer_label": "reviewer_b_label",
    })
    rows = [_agreement_row(analysis, "overall", "all")]
    dimensions = [
        "recorded_language", "caption_length_band", "hashtag_presence",
        "event_year", "phase4_disagreement_category",
        "phase4_candidate_origin",
    ]
    for dimension in dimensions:
        for group, frame in analysis.groupby(dimension, dropna=False):
            rows.append(_agreement_row(frame, dimension, str(group)))
    stats = pd.DataFrame(rows)
    overall = rows[0]
    pabak = 2 * overall["percent_agreement"] - 1
    matrix = confusion_matrix(
        analysis.reviewer_a_label,
        analysis.reviewer_b_label,
        labels=["include", "exclude"],
    )
    label_distribution = {
        "reviewer_a": analysis.reviewer_a_label.value_counts().to_dict(),
        "reviewer_b": analysis.reviewer_b_label.value_counts().to_dict(),
    }

    disagreements = analysis[~analysis.agrees].sample(
        frac=1, random_state=random_seed,
    ).reset_index(drop=True)
    supervisor = pd.DataFrame({
        "record_id": disagreements.record_id,
        "original_caption": disagreements.original_caption,
        "permitted_metadata": disagreements.apply(
            lambda row: json.dumps({
                "event_year": row.event_year,
                "hashtag_presence": row.hashtag_presence,
            }, ensure_ascii=False),
            axis=1,
        ),
        "recorded_language": disagreements.recorded_language,
        "reviewer_a_label": disagreements.reviewer_a_label,
        "reviewer_a_evidence_span": disagreements.reviewer_a_evidence_span,
        "reviewer_a_rationale": disagreements.reviewer_a_rationale,
        "reviewer_a_confidence": disagreements.reviewer_a_reviewer_confidence,
        "reviewer_a_comments": disagreements.reviewer_a_optional_comments,
        "reviewer_b_label": disagreements.reviewer_b_label,
        "reviewer_b_evidence_span": disagreements.reviewer_b_evidence_span,
        "reviewer_b_rationale": disagreements.reviewer_b_rationale,
        "reviewer_b_confidence": disagreements.reviewer_b_reviewer_confidence,
        "reviewer_b_comments": disagreements.reviewer_b_optional_comments,
        "phase4_candidate_origin": disagreements.phase4_candidate_origin,
        "phase4_disagreement_category":
            disagreements.phase4_disagreement_category,
        "adjudicator_label": "",
        "adjudicator_rationale": "",
        "adjudicator_name": "",
        "adjudication_timestamp": "",
    })
    forbidden = HIDDEN_FIELDS & set(supervisor.columns)
    if forbidden:
        raise RuntimeError(f"Supervisor package leaked fields: {forbidden}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "subgroup_agreement.csv"
    matrix_path = output_dir / "overall_confusion_matrix.csv"
    metrics_path = output_dir / "overall_agreement_metrics.json"
    supervisor_path = output_dir / "relevance_v8_supervisor_adjudication.csv"
    stats.to_csv(stats_path, index=False)
    pd.DataFrame(
        matrix,
        index=["reviewer_a_include", "reviewer_a_exclude"],
        columns=["reviewer_b_include", "reviewer_b_exclude"],
    ).to_csv(matrix_path)
    supervisor.to_csv(supervisor_path, index=False, encoding="utf-8-sig")
    metrics = {
        "workflow_version": WORKFLOW_VERSION,
        "records": len(analysis),
        "agreements": overall["agreements"],
        "disagreements": overall["disagreements"],
        "percent_agreement": overall["percent_agreement"],
        "cohen_kappa": overall["cohen_kappa"],
        "pabak": pabak,
        "label_order": ["include", "exclude"],
        "label_distribution": label_distribution,
        "confusion_matrix": matrix.tolist(),
        "annotation_quality_passes": bool(
            overall["cohen_kappa"] is not None
            and overall["cohen_kappa"] >= 0.80
        ),
        "reviewer_a": {
            "identity": a.reviewer_id.iloc[0],
            "role": a.reviewer_role.iloc[0],
            "version": a.reviewer_version.iloc[0],
            "imported_at_utc": a.imported_at_utc.iloc[0],
        },
        "reviewer_b": {
            "identity": b.reviewer_id.iloc[0],
            "role": b.reviewer_role.iloc[0],
            "version": b.reviewer_version.iloc[0],
            "imported_at_utc": b.imported_at_utc.iloc[0],
        },
    }
    metrics_path.write_text(
        json.dumps(metrics, indent=2), encoding="utf-8",
    )
    return {
        "metrics": metrics,
        "analysis": analysis,
        "subgroups": stats,
        "supervisor": supervisor,
        "paths": {
            "metrics": metrics_path,
            "confusion_matrix": matrix_path,
            "subgroups": stats_path,
            "supervisor_csv": supervisor_path,
        },
    }


def write_supervisor_package(
    result: dict,
    output_dir: Path,
    guideline_path: Path,
    validation_report_path: Path,
) -> dict:
    metrics = result["metrics"]
    supervisor = result["supervisor"]
    instructions_path = output_dir / "ADJUDICATOR_INSTRUCTIONS.md"
    report_path = output_dir / "SUPERVISOR_DISAGREEMENT_REPORT.md"
    iaa_path = output_dir / "RELEVANCE_V8_INTER_ANNOTATOR_AGREEMENT_REPORT.md"
    manifest_path = output_dir / "supervisor_package_manifest.json"
    instructions_path.write_text(
        "# Adjudicator instructions\n\n"
        "Adjudicate only the disagreement records in "
        "`relevance_v8_supervisor_adjudication.csv`.\n\n"
        "- Use only `include` or `exclude`.\n"
        "- Apply the official v8 relevance policy.\n"
        "- Provide a brief grounded rationale and your name and timestamp.\n"
        "- Do not optimize agreement with either reviewer.\n"
        "- Do not consult classifier predictions, confidence, routing, "
        "historical labels, or Phase 4 proposed labels.\n"
        "- Do not alter agreed records; they are intentionally absent.\n\n"
        f"Guideline: `{guideline_path}`\n",
        encoding="utf-8",
    )
    lines = [
        "# Supervisor disagreement report", "",
        f"Records requiring adjudication: **{len(supervisor)}**", "",
        "The order below was independently randomized. Stable record IDs "
        "preserve linkage.", "",
    ]
    for index, row in supervisor.iterrows():
        lines.extend([
            f"## {index + 1}. `{row.record_id}`", "",
            f"**Language:** {row.recorded_language}  ",
            f"**Permitted metadata:** `{row.permitted_metadata}`", "",
            str(row.original_caption), "",
            f"- Reviewer A: **{row.reviewer_a_label}** â€” "
            f"{row.reviewer_a_rationale}",
            f"- Evidence A: `{row.reviewer_a_evidence_span}`",
            f"- Confidence A: {row.reviewer_a_confidence}",
            f"- Comments A: {row.reviewer_a_comments or '-'}",
            f"- Reviewer B: **{row.reviewer_b_label}** â€” "
            f"{row.reviewer_b_rationale}",
            f"- Evidence B: `{row.reviewer_b_evidence_span}`",
            f"- Confidence B: {row.reviewer_b_confidence}",
            f"- Comments B: {row.reviewer_b_comments or '-'}", "",
        ])
    report_path.write_text("\n".join(lines), encoding="utf-8")

    interpretation = (
        "passes" if metrics["annotation_quality_passes"] else "does not pass"
    )
    iaa = f"""# Relevance v8 inter-annotator agreement report

## Dataset

- Records: {metrics['records']} disputed calibration captions
- Reviewer A: `{metrics['reviewer_a']['identity']}` ({metrics['reviewer_a']['role']})
- Reviewer B: `{metrics['reviewer_b']['identity']}` ({metrics['reviewer_b']['role']})
- Annotation versions: {metrics['reviewer_a']['version']} and {metrics['reviewer_b']['version']}
- Import dates: {metrics['reviewer_a']['imported_at_utc']} and {metrics['reviewer_b']['imported_at_utc']}
- Package integrity: passed

## Overall agreement

- Agreements: {metrics['agreements']} of {metrics['records']}
- Disagreements: {metrics['disagreements']} ({metrics['disagreements'] / metrics['records']:.2%})
- Percent agreement: {metrics['percent_agreement']:.4f}
- Cohen's kappa: {metrics['cohen_kappa']:.4f}
- PABAK: {metrics['pabak']:.4f}
- Reviewer A distribution: {metrics['label_distribution']['reviewer_a']}
- Reviewer B distribution: {metrics['label_distribution']['reviewer_b']}
- Confusion matrix (rows A, columns B; include, exclude): {metrics['confusion_matrix']}

Observed agreement is the raw proportion of matching labels. Cohen's kappa
adjusts for chance agreement and is strongly affected here by the reviewers'
different include prevalences. PABAK is supplementary and does not replace the
predeclared Cohen's-kappa criterion.

## Subgroup agreement

Detailed results are in `subgroup_agreement.csv`. Small language and category
groups are flagged as unstable; they must not be interpreted as reliable
population differences. Caption-length, hashtag, year, Phase 4 category, and
candidate-origin patterns are descriptive diagnostics only.

## Disagreement analysis

Disagreement is concentrated in {metrics['disagreements']} records. Reviewer B
uses `include` substantially more often than Reviewer A, indicating a systematic
decision-boundary difference rather than random noise alone. This may eventually
justify clarification after adjudication, but the guideline is not changed in
this phase.

## Methodological status

`annotation_quality_passes = {str(metrics['annotation_quality_passes']).lower()}`

The observed kappa {interpretation} the predeclared 0.80 annotation-quality
criterion. This is not a classifier failure, and no annotation is discarded.

## Next action

A qualified adjudicator must complete only the randomized disagreement CSV
before final v8 gold generation. No final gold labels have been created.
"""
    iaa_path.write_text(iaa, encoding="utf-8")
    files = [
        result["paths"]["metrics"], result["paths"]["confusion_matrix"],
        result["paths"]["subgroups"], result["paths"]["supervisor_csv"],
        instructions_path, report_path, iaa_path, guideline_path,
        validation_report_path,
    ]
    manifest = {
        "workflow_version": WORKFLOW_VERSION,
        "created_at_utc": _utc_now(),
        "randomized_disagreement_records": len(supervisor),
        "adjudicator_fields_blank": True,
        "hidden_classifier_fields_absent": True,
        "files": {
            str(path): _sha256(path) for path in files
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    return {
        "instructions": instructions_path,
        "readable_report": report_path,
        "iaa_report": iaa_path,
        "manifest": manifest_path,
    }

