"""Create the Phase 9 blind holdout without model scoring."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from marathon_absa.holdout_v8 import (
    HOLDOUT_SEED, HOLDOUT_VERSION, REVIEWER_A_SEED, REVIEWER_B_SEED,
    build_sampling_frame, collect_development_ids, make_reviewer_file,
    sha256, stratified_sample, utc_now,
)
from marathon_absa.openai_service import RELEVANCE_V8_FEW_SHOTS
from marathon_absa.text import stable_id


ROOT = Path("data/processed")
OUT = ROOT/"relevance_v8_holdout"


def verify_phase8() -> dict:
    manifest_path = ROOT/"relevance_v8_calibration_results/threshold_selection/relevance_v8_threshold_selection_v1.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    failures = [raw for raw, expected in manifest["files"].items()
                if not Path(raw).exists() or sha256(Path(raw)) != expected]
    config_path = ROOT/"relevance_v8_calibration_results/threshold_selection/relevance_v8_operating_point_v1.json"
    config = json.loads(config_path.read_text())
    gold = ROOT/"relevance_v8_adjudication/final/relevance_v8_adjudicated_gold_v1.csv"
    calibration = ROOT/"relevance_v8_calibration_results/relevance_v8_calibration_adjudicated_v1.csv"
    if failures or sha256(gold) != manifest["source_gold_hash"]:
        raise RuntimeError(f"Frozen Phase 8 integrity failure: {failures}")
    if sha256(calibration) != config["calibration_dataset_hash"]:
        raise RuntimeError("Frozen calibration dataset hash changed")
    if config["relevance_operating_point_version"] != "v8_op1":
        raise RuntimeError("Unexpected operating point")
    return {"phase8_manifest": str(manifest_path), "phase8_manifest_sha256": sha256(manifest_path),
            "operating_config": str(config_path), "operating_config_sha256": sha256(config_path),
            "adjudicated_gold": str(gold), "adjudicated_gold_sha256": sha256(gold),
            "calibration_dataset": str(calibration), "calibration_dataset_sha256": sha256(calibration)}


def main() -> None:
    paths = {
        "frame": OUT/"relevance_v8_holdout_sampling_frame_v1.csv",
        "holdout": OUT/"relevance_v8_blind_holdout_v1.csv",
        "a": OUT/"relevance_v8_holdout_reviewer_a.csv",
        "b": OUT/"relevance_v8_holdout_reviewer_b.csv",
        "manifest": OUT/"relevance_v8_holdout_freeze_manifest_v1.json",
        "report": OUT/"relevance_v8_holdout_sampling_report_v1.json",
        "strata": OUT/"relevance_v8_holdout_strata_v1.csv",
        "metrics": OUT/"relevance_v8_holdout_predeclared_metrics_v1.json",
        "state": OUT/"holdout_temporal_state_v1.json",
        "protocol": Path("RELEVANCE_V8_HOLDOUT_PROTOCOL.md"),
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Phase 9 holdout v1 is append-only and already exists")
    frozen = verify_phase8()
    documents_path = ROOT/"documents.csv"
    documents = pd.read_csv(documents_path, low_memory=False)
    development_ids, development_files = collect_development_ids(ROOT)
    fewshots = [text for text, _, _ in RELEVANCE_V8_FEW_SHOTS]
    frame, counts = build_sampling_frame(documents, development_ids, fewshots)
    sample_size = 300
    if len(frame) < sample_size:
        raise RuntimeError("A 300-record holdout is not feasible")
    sample, strata = stratified_sample(frame, sample_size, HOLDOUT_SEED)
    sample["holdout_id"] = ["v8h1-" + stable_id(str(i), row.document_id)[:12]
                              for i, row in sample.iterrows()]
    sample["caption"] = sample.original_text
    sample["language"] = sample.primary_language.fillna("undetermined")
    sample["permitted_metadata"] = sample.apply(lambda r: json.dumps({
        "source": r.source, "event_year": str(r.event_year)}, ensure_ascii=False), axis=1)
    holdout = sample[["holdout_id", "document_id", "caption", "permitted_metadata",
        "language", "source", "event_year", "text_hash", "sampling_stratum"]].rename(
        columns={"document_id": "source_record_id"})
    forbidden = [c for c in holdout.columns if any(x in c.lower() for x in
        ["prediction", "confidence", "routing", "reason_code", "gold", "relevance", "content_type", "event_connection"])]
    if forbidden:
        raise RuntimeError(f"Hidden fields in holdout: {forbidden}")
    reviewer_a = make_reviewer_file(holdout, REVIEWER_A_SEED)
    reviewer_b = make_reviewer_file(holdout, REVIEWER_B_SEED)
    if reviewer_a.holdout_id.tolist() == reviewer_b.holdout_id.tolist():
        raise RuntimeError("Reviewer randomizations are not independent")

    OUT.mkdir(parents=True, exist_ok=True)
    frame_columns = ["document_id", "source", "event_year", "primary_language",
                     "text_hash", "near_duplicate_key", "sampling_stratum"]
    frame[frame_columns].to_csv(paths["frame"], index=False, encoding="utf-8-sig")
    holdout.to_csv(paths["holdout"], index=False, encoding="utf-8-sig")
    reviewer_a.to_csv(paths["a"], index=False, encoding="utf-8-sig")
    reviewer_b.to_csv(paths["b"], index=False, encoding="utf-8-sig")
    strata.to_csv(paths["strata"], index=False)
    created = utc_now()
    metrics = {"frozen_before_predictions": True,
        "automatic_only": ["automatic_coverage", "automatic_inclusion_precision",
            "automatic_inclusion_recall", "automatic_exclusion_precision",
            "included_class_f1", "excluded_class_f1", "binary_macro_f1",
            "automatic_false_inclusions", "automatic_false_exclusions"],
        "routing": ["review_rate", "relevant_routed_to_review",
            "irrelevant_routed_to_review", "reviews_per_1000"],
        "simulated_human_assisted": "reported separately and explicitly as simulated",
        "gold_reliability": ["percent_agreement", "cohen_kappa"],
        "success_criteria": {"inclusion_precision_min": .85,
            "inclusion_recall_min": .90, "binary_macro_f1_min": .80},
        "additional_reporting": ["false_exclusion_safety",
            "automatic_exclusion_precision", "review_workload"],
        "confirmatory_rule": "The first evaluation of relevance_v8_blind_holdout_v1 is the confirmatory evaluation of frozen v8_op1. Any post-result modification creates a new system version; this holdout becomes development evidence and a new untouched holdout is required."}
    paths["metrics"].write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    state = {"holdout_created_at": created, "gold_finalized_at": None,
             "classifier_first_scored_at": None, "temporal_order_required": True}
    paths["state"].write_text(json.dumps(state, indent=2), encoding="utf-8")
    sampling_report = {**counts, "holdout_size_predeclared": sample_size,
        "sample_size_rationale": "A fresh 300-record sample is feasible from the eligible population, matches the calibration scale, supports binary precision/recall estimation under expected imbalance, and is practical for two independent reviewers.",
        "random_seed": HOLDOUT_SEED, "reviewer_a_seed": REVIEWER_A_SEED,
        "reviewer_b_seed": REVIEWER_B_SEED,
        "sampling_algorithm": "proportional stratified random sampling with largest-remainder allocation and deterministic per-stratum pandas sampling",
        "stratification_variables": ["source", "event_year", "independent OpenLID primary_language (rare values pooled)"],
        "sampling_timestamp": created, "holdout_version": HOLDOUT_VERSION,
        "development_id_union_n": len(development_ids),
        "development_artifact_counts": development_files,
        "model_fields_used_for_eligibility_or_sampling": [], "api_calls": 0}
    paths["report"].write_text(json.dumps(sampling_report, indent=2), encoding="utf-8")
    paths["protocol"].write_text(f"""# Relevance v8 blind holdout protocol

## Frozen design

Holdout version: `{HOLDOUT_VERSION}`. Sample size 300 was declared before sampling because the eligible corpus is substantially larger, 300 matches the calibration scale, permits meaningful binary precision/recall estimates under expected imbalance, and remains feasible for two independent reviewers.

Membership was drawn with seed `{HOLDOUT_SEED}` by proportional stratified probability sampling over source, event year, and independently generated OpenLID language metadata. Rare languages were pooled for allocation. No relevance prediction, confidence, routing, event connection, content type, historical label, or difficulty judgment was used.

All identifiable calibration, validation, audit, adjudication, review, policy-alignment, previous holdout, threshold-error inspection, and repeated-record IDs were excluded. Exact `text_hash` duplicates, project-marked duplicates, and conservative normalized near duplicates (case/punctuation/URL/hashtag/whitespace variation) were removed against development records and within the eligible pool.

## Annotation

Use the unchanged `RELEVANCE_V8_ANNOTATION_GUIDELINE.md`. Reviewer A and Reviewer B work independently from separately randomized files. Permitted fields are holdout ID, caption, source/year metadata, and independently generated language. Labels are only `include` or `exclude`, with evidence span, rationale, confidence, and optional comments. Reviewers must not inspect one another's decisions or any classifier output.

After both reviews: validate submissions; compute agreement; route only disagreements to a qualified supervisor; finalize immutable holdout gold. Do not score before gold finalization.

## Leakage and temporal rule

Normal v8 scoring rejects these source IDs. A future dedicated evaluation command may operate only after `gold_finalized_at` exists. Required order: `holdout_created_at < gold_finalized_at < classifier_first_scored_at`.

## Predeclared evaluation

Metrics and unchanged targets are frozen in `{paths['metrics']}`. Automatic-only, routing, simulated human-assisted, and annotation-reliability views remain separate. Targets remain inclusion precision >=0.85, inclusion recall >=0.90, and binary macro-F1 >=0.80, with false-exclusion safety, exclusion precision, and workload also reported.

> The first evaluation of `relevance_v8_blind_holdout_v1` is the confirmatory evaluation of frozen `v8_op1`.

If it fails, do not tune and reuse it as blind evidence. Any modification creates a new system version; this holdout becomes development evidence, and another untouched holdout is required.
""", encoding="utf-8")
    freeze_files = [documents_path, Path("marathon_absa/openai_service.py"),
        Path("marathon_absa/schemas.py"), Path("marathon_absa/relevance_v8.py"),
        Path("marathon_absa/config.py"), Path("marathon_absa/pipeline.py"),
        Path("RELEVANCE_V8_ANNOTATION_GUIDELINE.md"), *[Path(v) for k,v in frozen.items() if not k.endswith("sha256")],
        paths["frame"], paths["holdout"], paths["a"], paths["b"], paths["strata"],
        paths["report"], paths["metrics"], paths["state"], paths["protocol"]]
    manifest = {"holdout_version": HOLDOUT_VERSION, "created_at_utc": created,
        "membership_frozen": True, "classifier_has_scored_holdout": False,
        "gold_finalized": False, "operating_point_version": "v8_op1",
        "policy_version": "v8_event_experience_binary", "prompt_version": "v8",
        "schema_version": "v8", "routing_version": "v8", "model_version": ["gpt-5.6-luna"],
        "few_shot_examples_location": "marathon_absa/openai_service.py:RELEVANCE_V8_FEW_SHOTS",
        "files": {str(path): sha256(path) for path in freeze_files}, "api_calls": 0}
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"counts": counts, "sample_size": len(holdout),
        "strata": len(strata), "seeds": [HOLDOUT_SEED, REVIEWER_A_SEED, REVIEWER_B_SEED],
        "paths": {k:str(v) for k,v in paths.items()}}, indent=2))


if __name__ == "__main__":
    main()
