from __future__ import annotations

import hashlib
import json
import os
import tempfile
import statistics
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import SETTINGS, Settings
from .openai_service import (CachedOpenAI, RELEVANCE_V8_INSTRUCTIONS,
                             RELEVANCE_V8_VERIFICATION_INSTRUCTIONS)
from .relevance import deterministic_result
from .relevance_v8 import material_model_disagreement, route_assessment
from .schemas import RELEVANCE_V8_ASSESSMENT_SCHEMA, V8_EXCLUDED_CONTENT_TYPES, V8_INCLUDED_CONTENT_TYPES
from .storage import read_table, write_table


WORKFLOW_VERSION = "v8_single_researcher_human_in_loop_v1"
OPERATING_POINT = "v8_op1"
INCLUSION_THRESHOLD = 0.80
EXCLUSION_THRESHOLD = 0.99
AUDIT_SEED = 104729
ARCHITECTURE_AUDIT_VERSION = "single_researcher_architecture_audit_v2"
CURRENT_AUDIT_DIR = Path("data/processed/relevance_v8_single_researcher_architecture_audit_v2")
CURRENT_AUDIT_BLIND = CURRENT_AUDIT_DIR / "relevance_v8_architecture_audit_blind_v2.csv"
CURRENT_AUDIT_KEY = CURRENT_AUDIT_DIR / "relevance_v8_architecture_audit_key_v2.csv"
CURRENT_AUDIT_FINAL = CURRENT_AUDIT_DIR / "final_v2"
AUDIT_COLUMNS = [
    "audit_id", "document_id", "original_text", "source", "event_year",
    "primary_language", "language_status", "length_group", "researcher_label",
    "evidence_span", "rationale", "confidence", "comments",
]
ANNOTATION_COLUMNS = ["researcher_label", "evidence_span", "rationale", "confidence", "comments"]
HIDDEN_TOKENS = ("automatic", "prediction", "routing", "reason", "model", "gold", "historical")
VALID_LABELS = {"include", "exclude"}
STANDARD_PRICING_USD_PER_MILLION = {
    # Verified 2026-08-14 against OpenAI's 2026-07-30 pricing announcement.
    "gpt-5.6-luna": {"input": 0.20, "output": 1.20},
    "gpt-5.6-terra": {"input": 2.00, "output": 12.00},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame_hash(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = frame[columns].fillna("").astype(str).agg("\x1f".join, axis=1)
    return values.map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())


def _length_group(text: object) -> str:
    n = len(str(text).split())
    return "very_short" if n < 8 else "short" if n < 25 else "medium" if n < 75 else "long"


def _normalize_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "caption": "original_text", "language": "primary_language",
        "v8_automatic_decision": "automatic_decision",
    }
    output = frame.rename(columns={k: v for k, v in aliases.items() if k in frame and v not in frame}).copy()
    required = {"document_id", "original_text", "automatic_decision"}
    missing = required.difference(output.columns)
    if missing:
        raise ValueError(f"Candidate decisions missing columns: {', '.join(sorted(missing))}")
    if output.document_id.astype(str).duplicated().any():
        raise ValueError("Candidate decisions contain duplicate document IDs")
    for column in ["source", "event_year", "primary_language", "language_status"]:
        if column not in output:
            output[column] = "unknown"
    output["automatic_decision"] = output.automatic_decision.fillna("").astype(str).str.lower()
    output["length_group"] = output.original_text.map(_length_group)
    output["sampling_stratum"] = output[["source", "event_year", "primary_language", "length_group"]].fillna("unknown").astype(str).agg("|".join, axis=1)
    return output


def _sample_one_stratum(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(frame) < n:
        raise ValueError(f"Insufficient {frame.automatic_decision.iloc[0] if len(frame) else 'decision'} candidates: need {n}, found {len(frame)}")
    counts = frame.sampling_stratum.value_counts().sort_index()
    exact = counts / counts.sum() * n
    allocation = exact.astype(int)
    for stratum in (exact - allocation).sort_values(ascending=False, kind="mergesort").index[:n-int(allocation.sum())]:
        allocation[stratum] += 1
    parts = []
    for offset, (stratum, amount) in enumerate(allocation.items()):
        if amount:
            parts.append(frame[frame.sampling_stratum.eq(stratum)].sample(int(amount), random_state=seed + offset))
    return pd.concat(parts, ignore_index=True)


def build_audit_sample(candidates: pd.DataFrame, per_decision: int = 75,
                       seed: int = AUDIT_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a blind, balanced audit of automatic decisions only."""
    frame = _normalize_candidates(candidates)
    if per_decision <= 0:
        raise ValueError("per_decision must be positive")
    selected = []
    for offset, decision in enumerate(["include", "exclude"]):
        pool = frame[frame.automatic_decision.eq(decision)].copy()
        selected.append(_sample_one_stratum(pool, per_decision, seed + offset * 10000))
    key = pd.concat(selected, ignore_index=True)
    key["source_row_hash"] = _frame_hash(key, ["document_id", "original_text", "source", "event_year", "primary_language", "language_status", "length_group"])
    key["stratum_population_n"] = key.apply(lambda row: int(((frame.automatic_decision == row.automatic_decision) & (frame.sampling_stratum == row.sampling_stratum)).sum()), axis=1)
    key["stratum_sample_n"] = key.groupby(["automatic_decision", "sampling_stratum"]).document_id.transform("size")
    key = key.sort_values(["automatic_decision", "document_id"]).reset_index(drop=True)
    key["audit_id"] = [f"SRA-{i:04d}" for i in range(1, len(key) + 1)]
    visible = key[AUDIT_COLUMNS[:-5]].copy()
    for column in ANNOTATION_COLUMNS:
        visible[column] = ""
    visible = visible.sample(frac=1, random_state=seed + 2).reset_index(drop=True)
    if visible.document_id.duplicated().any() or len(visible) != per_decision * 2:
        raise AssertionError("Audit identity or cardinality failure")
    if any(any(token in c.lower() for token in HIDDEN_TOKENS) for c in visible.columns):
        raise AssertionError("Hidden model information leaked into blind audit")
    return visible, key


def build_architecture_audit_sample(candidates: pd.DataFrame, include_n: int = 75,
                                    deterministic_exclude_n: int = 50,
                                    seed: int = AUDIT_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample model-assisted includes and deterministic exclusions as distinct arms."""
    frame = _normalize_candidates(candidates)
    if include_n <= 0 or deterministic_exclude_n <= 0:
        raise ValueError("Audit allocations must be positive")
    if "v8_routing_status" not in frame:
        raise ValueError("Architecture audit requires v8_routing_status")
    include_pool = frame[frame.automatic_decision.eq("include")].copy()
    deterministic_pool = frame[
        frame.automatic_decision.eq("exclude")
        & frame.v8_routing_status.eq("automatic_exclude_deterministic")
    ].copy()
    included = _sample_one_stratum(include_pool, include_n, seed)
    included["audit_arm"] = "automatic_include"
    excluded = _sample_one_stratum(deterministic_pool, deterministic_exclude_n, seed + 10000)
    excluded["audit_arm"] = "deterministic_auto_exclude"
    key = pd.concat([included, excluded], ignore_index=True)
    key["source_row_hash"] = _frame_hash(key, ["document_id", "original_text", "source", "event_year", "primary_language", "language_status", "length_group"])
    key["arm_population_n"] = key.audit_arm.map({"automatic_include": len(include_pool), "deterministic_auto_exclude": len(deterministic_pool)})
    key["stratum_population_n"] = key.apply(lambda row: int(((frame.automatic_decision == row.automatic_decision) & (frame.sampling_stratum == row.sampling_stratum)).sum()), axis=1)
    key["stratum_sample_n"] = key.groupby(["audit_arm", "sampling_stratum"]).document_id.transform("size")
    key = key.sort_values(["audit_arm", "document_id"]).reset_index(drop=True)
    key["audit_id"] = [f"SRA2-{i:04d}" for i in range(1, len(key) + 1)]
    visible = key[AUDIT_COLUMNS[:-5]].copy()
    for column in ANNOTATION_COLUMNS:
        visible[column] = ""
    visible = visible.sample(frac=1, random_state=seed + 2).reset_index(drop=True)
    if visible.document_id.duplicated().any() or len(visible) != include_n + deterministic_exclude_n:
        raise AssertionError("Architecture audit identity or cardinality failure")
    if any(any(token in c.lower() for token in HIDDEN_TOKENS) for c in visible.columns):
        raise AssertionError("Hidden routing information leaked into blind audit")
    return visible, key


def build_operational_review_queue(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Export every non-automatic v8 route for final binary researcher resolution."""
    frame = _normalize_candidates(candidates)
    queue = frame[~frame.automatic_decision.isin(VALID_LABELS)].copy().sort_values("document_id").reset_index(drop=True)
    queue["source_row_hash"] = _frame_hash(queue, ["document_id", "original_text", "source", "event_year", "primary_language", "language_status", "length_group"])
    queue["audit_id"] = [f"SRA-REVIEW-{i:05d}" for i in range(1, len(queue) + 1)]
    visible = queue[AUDIT_COLUMNS[:-5]].copy()
    for column in ANNOTATION_COLUMNS:
        visible[column] = ""
    return visible, queue


def write_audit_package(candidates_path: Path, package_dir: Path,
                        per_decision: int = 75, seed: int = AUDIT_SEED) -> dict:
    if package_dir.exists():
        raise FileExistsError(f"Audit package already exists: {package_dir}")
    candidates = pd.read_parquet(candidates_path) if candidates_path.suffix == ".parquet" else pd.read_csv(candidates_path, dtype=str, keep_default_na=False)
    blind, key = build_audit_sample(candidates, per_decision, seed)
    review, review_key = build_operational_review_queue(candidates)
    package_dir.mkdir(parents=True)
    blind_path = package_dir / "relevance_v8_single_researcher_audit_blind_v1.csv"
    key_path = package_dir / "relevance_v8_single_researcher_audit_key_v1.csv"
    blind.to_csv(blind_path, index=False, encoding="utf-8-sig")
    key.to_csv(key_path, index=False, encoding="utf-8-sig")
    review_path = package_dir / "relevance_v8_operational_review_queue_v1.csv"
    review_key_path = package_dir / "relevance_v8_operational_review_key_v1.csv"
    review.to_csv(review_path, index=False, encoding="utf-8-sig")
    review_key.to_csv(review_key_path, index=False, encoding="utf-8-sig")
    operational_versions = set(candidates.get("operational_config_version", pd.Series(dtype=str)).dropna().astype(str))
    operational_version = next(iter(operational_versions)) if len(operational_versions) == 1 else ""
    manifest = {
        "workflow_version": WORKFLOW_VERSION, "created_at_utc": utc_now(),
        "reviewer_mode": "single_researcher", "source_artifact": str(candidates_path),
        "source_artifact_sha256": sha256(candidates_path), "v8_policy_version": "v8_event_experience_binary",
        "v8_prompt_version": "v8", "v8_schema_version": "v8", "v8_routing_version": "v8",
        "operating_point": OPERATING_POINT, "inclusion_threshold": INCLUSION_THRESHOLD,
        "exclusion_threshold": EXCLUSION_THRESHOLD, "audit_sample_size": len(blind),
        "automatic_include_allocation": per_decision, "automatic_exclude_allocation": per_decision,
        "random_seed": seed, "api_calls_occurred": False,
        "phase_9_status": "superseded_before_execution",
        "phase_9_classifier_scored_before_supersession": False,
        "phase_9_gold_finalized": False,
        "operational_config_version": operational_version,
        "acceptance_protocol_version": "cost_conservative_audit_v1" if operational_version == "v8_single_researcher_cost_conservative_v1" else "single_researcher_audit_v1",
        "maximum_false_exclusions": 2 if operational_version == "v8_single_researcher_cost_conservative_v1" else 0,
        "operational_review_records": len(review),
        "output_hashes": {p.name: sha256(p) for p in [blind_path, key_path, review_path, review_key_path]},
    }
    manifest_path = package_dir / "manifest_v1.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def write_architecture_audit_package(candidates_path: Path, package_dir: Path,
                                     include_n: int = 75, deterministic_exclude_n: int = 50,
                                     seed: int = AUDIT_SEED) -> dict:
    """Write the additive v2 audit package; historical v1 artifacts are untouched."""
    if package_dir.exists():
        raise FileExistsError(f"Audit package already exists: {package_dir}")
    candidates = pd.read_parquet(candidates_path) if candidates_path.suffix == ".parquet" else pd.read_csv(candidates_path, dtype=str, keep_default_na=False)
    blind, key = build_architecture_audit_sample(candidates, include_n, deterministic_exclude_n, seed)
    review, review_key = build_operational_review_queue(candidates)
    package_dir.mkdir(parents=True)
    blind_path = package_dir / "relevance_v8_architecture_audit_blind_v2.csv"
    key_path = package_dir / "relevance_v8_architecture_audit_key_v2.csv"
    review_path = package_dir / "relevance_v8_operational_review_queue_v2.csv"
    review_key_path = package_dir / "relevance_v8_operational_review_key_v2.csv"
    blind.to_csv(blind_path, index=False, encoding="utf-8-sig")
    key.to_csv(key_path, index=False, encoding="utf-8-sig")
    review.to_csv(review_path, index=False, encoding="utf-8-sig")
    review_key.to_csv(review_key_path, index=False, encoding="utf-8-sig")
    manifest = {
        "workflow_version": ARCHITECTURE_AUDIT_VERSION, "created_at_utc": utc_now(),
        "reviewer_mode": "single_researcher", "source_artifact": str(candidates_path),
        "source_artifact_sha256": sha256(candidates_path), "operational_config_version": "v8_single_researcher_cost_conservative_v1",
        "acceptance_protocol_version": "architecture_aligned_audit_v2",
        "automatic_include_allocation": include_n,
        "deterministic_auto_exclude_allocation": deterministic_exclude_n,
        "operational_review_records": len(review), "random_seed": seed,
        "api_calls_occurred": False,
        "sampling_note": "Automatic includes and deterministic hashtag-only exclusions are separate audit arms; operational human-review records are excluded.",
        "output_hashes": {p.name: sha256(p) for p in [blind_path, key_path, review_path, review_key_path]},
    }
    (package_dir / "manifest_v2.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_v8_production(settings: Settings = SETTINGS) -> pd.DataFrame:
    """Run the explicitly authorized paid v8_op1 full-corpus classification."""
    output_dir = settings.output_dir / "relevance_v8_production"
    if output_dir.exists():
        raise FileExistsError("v8 production output exists; refusing to overwrite")
    status_path = settings.output_dir / "relevance_v8_holdout" / "phase9_protocol_supersession_v1.json"
    if not status_path.exists() or json.loads(status_path.read_text(encoding="utf-8")).get("protocol_status") != "superseded_before_execution":
        raise RuntimeError("Phase 9 supersession status is required before production scoring")
    documents = read_table(settings.output_dir / "documents.parquet")
    candidates = documents[(documents.source == "instagram") & (documents.processing_status == "ready")].copy()
    deterministic = candidates.apply(deterministic_result, axis=1)
    api_candidates = candidates[deterministic.isna()].copy()
    service = CachedOpenAI(settings.cache_dir, settings.chat_model)
    verification_service = CachedOpenAI(settings.cache_dir, settings.stronger_model)
    rows = []
    for row in api_candidates.itertuples(index=False):
        context = json.dumps({"document_id": row.document_id, "event_year": row.event_year,
            "original_text": row.original_text, "linguistic_text": row.linguistic_text,
            "hashtags": row.hashtags, "primary_language": row.primary_language,
            "language_status": row.language_status}, ensure_ascii=False)
        assessment, meta = service.structured("relevance_v8_initial", "v8", RELEVANCE_V8_INSTRUCTIONS, context, RELEVANCE_V8_ASSESSMENT_SCHEMA)
        verify = (assessment["event_connection"] in {"weak", "none"} or
                  assessment["primary_content_type"] in V8_EXCLUDED_CONTENT_TYPES or
                  assessment["confidence"] < .93 or assessment["contradiction_present"] or
                  assessment["image_dependent"] or assessment["requires_review_recommendation"])
        verification = None; verification_meta = {}
        if verify:
            verification, verification_meta = verification_service.structured(
                "relevance_v8_verification", "v8", RELEVANCE_V8_VERIFICATION_INSTRUCTIONS,
                json.dumps({"caption": json.loads(context), "first_assessment": assessment}, ensure_ascii=False),
                RELEVANCE_V8_ASSESSMENT_SCHEMA)
        routed = route_assessment(assessment, verification, include_threshold=.80, exclude_threshold=.99)
        rows.append({"document_id": row.document_id, **routed,
            "v8_initial_model": meta.get("model", ""), "v8_initial_prompt_version": meta.get("prompt_version", ""),
            "v8_initial_schema_version": "v8", "v8_initial_cached": meta.get("cached", False),
            "v8_initial_input_tokens": meta.get("input_tokens"), "v8_initial_output_tokens": meta.get("output_tokens"),
            "v8_verification_model": verification_meta.get("model", ""),
            "v8_verification_prompt_version": verification_meta.get("prompt_version", ""),
            "v8_verification_cached": verification_meta.get("cached", False),
            "v8_model_agreement": "not_checked" if verification is None else ("material_disagreement" if material_model_disagreement(assessment, verification) else "no_material_disagreement")})
    for index, result in deterministic[deterministic.notna()].items():
        row = candidates.loc[index]; excluded = result["reason_code"] == "hashtag_only"
        rows.append({"document_id": row.document_id, "v8_automatic_decision": "exclude" if excluded else "review",
            "v8_routing_status": "automatic_exclude_deterministic" if excluded else "pending_review",
            "v8_routing_reason": result["reason_code"], "v8_reason_code": result["reason_code"],
            "v8_final_inclusion": "exclude" if excluded else "unresolved",
            "v8_final_decision_source": "deterministic_exclusion" if excluded else "pending_human_review",
            "v8_include_in_topics": False, "v8_include_in_sentiment": False,
            "v8_policy_version": "v8_event_experience_binary", "v8_schema_version": "v8", "v8_routing_version": "v8",
            "v8_human_final_inclusion": "", "v8_review_notes": ""})
    routed = pd.DataFrame(rows).merge(candidates[["document_id", "original_text", "source", "event_year", "primary_language", "language_status", "linguistic_text"]], on="document_id", validate="one_to_one")
    if len(routed) != len(candidates):
        raise RuntimeError("Production routing failed to preserve candidate cardinality")
    output_dir.mkdir(parents=True)
    write_table(routed, output_dir / "relevance_v8_decisions.parquet")
    manifest = {"workflow_version": WORKFLOW_VERSION, "created_at_utc": utc_now(), "operating_point": OPERATING_POINT,
        "inclusion_threshold": .80, "exclusion_threshold": .99, "eligible_records": len(candidates),
        "v8_policy_version": "v8_event_experience_binary", "v8_prompt_version": "v8",
        "v8_schema_version": "v8", "v8_routing_version": "v8",
        "api_scored_records": len(api_candidates), "routing": routed.v8_automatic_decision.value_counts().to_dict(),
        "api_calls_occurred": bool(len(api_candidates)),
        "source_artifact_sha256": sha256(settings.output_dir / "documents.parquet"),
        "output_hashes": {name: sha256(output_dir / name) for name in ["relevance_v8_decisions.parquet", "relevance_v8_decisions.csv"]}}
    manifest_path = output_dir / "manifest_v1.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return routed


def estimate_v8_production_cost(settings: Settings = SETTINGS) -> dict:
    """Inspect local population/caches and estimate cost without creating an API client."""
    documents = read_table(settings.output_dir / "documents.parquet")
    candidates = documents[(documents.source == "instagram") & (documents.processing_status == "ready")].copy()
    deterministic = candidates.apply(deterministic_result, axis=1)
    api_candidates = candidates[deterministic.isna()].copy()

    def cache_key(stage: str, model: str, text: str) -> str:
        raw = "|".join([stage, "v8", model, text, ""])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    initial_hits = verification_needed = verification_hits = terra_eligible_known = 0
    current_review_known = variant_review_known = 0
    for row in api_candidates.itertuples(index=False):
        context = json.dumps({"document_id": row.document_id, "event_year": row.event_year,
            "original_text": row.original_text, "linguistic_text": row.linguistic_text,
            "hashtags": row.hashtags, "primary_language": row.primary_language,
            "language_status": row.language_status}, ensure_ascii=False)
        initial_path = settings.cache_dir / "relevance_v8_initial" / f"{cache_key('relevance_v8_initial', settings.chat_model, context)}.json"
        if not initial_path.exists():
            continue
        initial_hits += 1
        assessment = json.loads(initial_path.read_text(encoding="utf-8"))["result"]
        verify = (assessment["event_connection"] in {"weak", "none"} or
                  assessment["primary_content_type"] in V8_EXCLUDED_CONTENT_TYPES or
                  assessment["confidence"] < .93 or assessment["contradiction_present"] or
                  assessment["image_dependent"] or assessment["requires_review_recommendation"])
        if verify:
            verification_needed += 1
            relevant_candidate = (assessment["event_connection"] in {"explicit", "supported"}
                                  and assessment["meaningful_content_present"]
                                  and assessment["primary_content_type"] in V8_INCLUDED_CONTENT_TYPES)
            terra_eligible_known += int(relevant_candidate)
            verification_text = json.dumps({"caption": json.loads(context), "first_assessment": assessment}, ensure_ascii=False)
            verification_path = settings.cache_dir / "relevance_v8_verification" / f"{cache_key('relevance_v8_verification', settings.stronger_model, verification_text)}.json"
            if verification_path.exists():
                verification_hits += 1
                verification = json.loads(verification_path.read_text(encoding="utf-8"))["result"]
                current_review_known += int(route_assessment(assessment, verification, .80, .99)["v8_automatic_decision"] == "review")
                variant_review_known += int((not relevant_candidate) or route_assessment(assessment, verification, .80, .99)["v8_automatic_decision"] == "review")
        else:
            routed = route_assessment(assessment, None, .80, .99)
            current_review_known += int(routed["v8_automatic_decision"] == "review")
            variant_review_known += int(routed["v8_automatic_decision"] == "review")

    def historical(stage: str, cutoff: str = "2026-07-30") -> dict:
        records = []
        for path in (settings.cache_dir / stage).glob("*.json"):
            if datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).date().isoformat() >= cutoff:
                continue
            metadata = json.loads(path.read_text(encoding="utf-8"))["metadata"]
            if metadata.get("input_tokens") is not None and metadata.get("output_tokens") is not None:
                records.append(metadata)
        def summary(field: str) -> dict:
            values = sorted(float(record[field]) for record in records)
            return {"total": int(sum(values)), "mean": statistics.mean(values),
                    "median": statistics.median(values), "p95": values[min(len(values)-1, int((len(values)-1)*.95+.999999))]}
        return {"calls": len(records), "input": summary("input_tokens"), "output": summary("output_tokens"),
                "cached_input_tokens": None, "cached_input_note": "input_tokens_details were not persisted by CachedOpenAI"}

    initial_history = historical("relevance_v8_initial")
    verification_history = historical("relevance_v8_verification")
    new_initial = len(api_candidates)-initial_hits
    observed_rate = verification_needed/initial_hits if initial_hits else verification_history["calls"]/initial_history["calls"]
    expected_verification = round(new_initial*observed_rate)+(verification_needed-verification_hits)
    variant_rate = terra_eligible_known/initial_hits if initial_hits else 0
    expected_variant_verification = round(new_initial*variant_rate)
    prices = {settings.chat_model: STANDARD_PRICING_USD_PER_MILLION[settings.chat_model], settings.stronger_model: STANDARD_PRICING_USD_PER_MILLION[settings.stronger_model]}
    def scenario(n_initial: int, n_verification: int, percentile: str,
                 input_multiplier: float = 1.0) -> dict:
        ii=initial_history["input"][percentile]; io=initial_history["output"][percentile]
        vi=verification_history["input"][percentile]; vo=verification_history["output"][percentile]
        cost=(n_initial*(ii*prices[settings.chat_model]["input"]*input_multiplier+io*prices[settings.chat_model]["output"])+n_verification*(vi*prices[settings.stronger_model]["input"]*input_multiplier+vo*prices[settings.stronger_model]["output"]))/1_000_000
        return {"initial_calls": n_initial, "verification_calls": n_verification,
                "input_tokens": round(n_initial*ii+n_verification*vi), "output_tokens": round(n_initial*io+n_verification*vo), "input_price_multiplier": input_multiplier, "cost_usd": cost}
    expected=scenario(new_initial, expected_verification, "mean")
    conservative=scenario(new_initial, expected_verification, "p95", 1.25)
    worst=scenario(new_initial, new_initial+(verification_needed-verification_hits), "p95", 1.25)
    no_cache=scenario(len(api_candidates), round(len(api_candidates)*observed_rate), "mean")
    no_cache_high=scenario(len(api_candidates), round(len(api_candidates)*observed_rate), "p95", 1.25)
    saved=max(0, no_cache["cost_usd"]-expected["cost_usd"])
    variant_expected=scenario(new_initial, expected_variant_verification, "mean")
    variant_p95=scenario(new_initial, expected_variant_verification, "p95", 1.25)
    return {"api_called": False, "models": {"initial": settings.chat_model, "verification": settings.stronger_model},
        "population": {"eligible": len(candidates), "deterministic_total": int(deterministic.notna().sum()),
            "deterministic_exclusions": int(sum((value or {}).get("reason_code") == "hashtag_only" for value in deterministic if value is not None)),
            "api_candidates": len(api_candidates), "reusable_initial": initial_hits, "new_initial": new_initial,
            "known_verification_required": verification_needed, "reusable_verification": verification_hits,
            "known_missing_verification": verification_needed-verification_hits, "estimated_remaining_verification": expected_verification,
            "observed_production_verification_rate": observed_rate,
            "cost_conservative_terra_eligible_known": terra_eligible_known,
            "cost_conservative_estimated_remaining_terra": expected_variant_verification,
            "current_projected_remaining_manual_review": round(new_initial*current_review_known/initial_hits),
            "cost_conservative_projected_remaining_manual_review": round(new_initial*variant_review_known/initial_hits)},
        "historical": {"initial": initial_history, "verification": verification_history,
            "verification_rate": verification_history["calls"]/initial_history["calls"]},
        "pricing": {"as_of": "2026-08-14", "mode": "standard", "usd_per_million_tokens": prices,
            "source": "https://openai.com/index/advancing-the-price-performance-frontier-with-gpt-5-6/",
            "verification_note": "Verify current Luna and Terra standard rates before purchase/rerun. Conservative scenarios price every input token at the documented 1.25x cache-write multiplier because input token details were not saved."},
        "estimates": {"expected_remaining": expected, "conservative_p95": conservative,
            "worst_practical_all_remaining_verify_p95": worst, "no_cache_expected": no_cache,
            "no_cache_p95": no_cache_high, "estimated_cache_savings_usd": saved,
            "cost_conservative_expected": variant_expected,
            "cost_conservative_batch_expected": {**variant_expected, "cost_usd": variant_expected["cost_usd"]/2},
            "cost_conservative_p95": variant_p95,
            "v8_op1_batch_expected": {**expected, "cost_usd": expected["cost_usd"]/2}}}


def _validate_submission(blind: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    missing = set(AUDIT_COLUMNS).difference(blind.columns)
    if missing:
        raise ValueError(f"Blind audit missing columns: {', '.join(sorted(missing))}")
    if blind.audit_id.duplicated().any() or blind.document_id.duplicated().any():
        raise ValueError("Audit submission contains duplicates")
    if set(blind.audit_id) != set(key.audit_id) or set(blind.document_id) != set(key.document_id):
        raise ValueError("Blind audit and hidden key do not align")
    merged = blind.merge(key, on=["audit_id", "document_id"], suffixes=("", "_key"), validate="one_to_one")
    for column in ["original_text", "source", "event_year", "primary_language", "language_status", "length_group"]:
        if not merged[column].fillna("").astype(str).equals(merged[f"{column}_key"].fillna("").astype(str)):
            raise ValueError(f"Audit metadata was changed: {column}")
    labels = merged.researcher_label.fillna("").str.strip().str.lower()
    if not labels.isin(VALID_LABELS).all():
        raise ValueError("Complete all audit labels using include or exclude")
    if merged.rationale.fillna("").str.strip().eq("").any():
        raise ValueError("Every audit row requires a rationale")
    confidence = pd.to_numeric(merged.confidence, errors="coerce")
    if not confidence.between(0, 1).all():
        raise ValueError("Every confidence value must be between 0 and 1")
    merged["researcher_label"] = labels
    return merged


def calculate_metrics(merged: pd.DataFrame, maximum_false_exclusions: int = 0) -> dict:
    table = pd.crosstab(merged.researcher_label, merged.automatic_decision).reindex(index=["include", "exclude"], columns=["include", "exclude"], fill_value=0)
    include_rows = merged[merged.automatic_decision.eq("include")]
    exclude_rows = merged[merged.automatic_decision.eq("exclude")]
    include_confirmed = int(include_rows.researcher_label.eq("include").sum())
    exclude_confirmed = int(exclude_rows.researcher_label.eq("exclude").sum())
    false_exclusions = int(exclude_rows.researcher_label.eq("include").sum())
    weights = merged.stratum_population_n.astype(float) / merged.stratum_sample_n.astype(float)
    weighted = merged.assign(_weight=weights).groupby(["researcher_label", "automatic_decision"])._weight.sum()
    tp = float(weighted.get(("include", "include"), 0)); fn = float(weighted.get(("include", "exclude"), 0))
    metrics = {
        "automatic_include_audit": {"audited_n": len(include_rows), "researcher_confirmed_include": include_confirmed, "false_inclusions": len(include_rows)-include_confirmed, "confirmation_rate": include_confirmed/len(include_rows)},
        "automatic_exclude_audit": {"audited_n": len(exclude_rows), "researcher_confirmed_exclude": exclude_confirmed, "false_exclusions": false_exclusions, "confirmation_rate": exclude_confirmed/len(exclude_rows), "false_exclusion_rate": false_exclusions/len(exclude_rows)},
        "combined_automatic_audit": {"audited_n": len(merged), "agreement_accuracy": float((merged.researcher_label == merged.automatic_decision).mean()), "confusion_matrix_researcher_rows_automatic_columns": table.values.tolist(), "include_precision": include_confirmed/len(include_rows), "design_weighted_include_recall": tp/(tp+fn) if tp+fn else None, "prevalence_note": "The 75/75 audit is disproportionate; unweighted combined results do not estimate corpus prevalence. Recall is design-weighted using recorded stratum sampling fractions."},
    }
    include_ok = metrics["automatic_include_audit"]["confirmation_rate"] >= .90
    exclude_ok = metrics["automatic_exclude_audit"]["confirmation_rate"] >= .90
    if maximum_false_exclusions == 0:
        safety = "passes_zero_observed_false_exclusions" if false_exclusions == 0 else "requires_methodological_review"
    else:
        safety = "within_predeclared_limit" if false_exclusions <= maximum_false_exclusions else "exceeds_predeclared_limit"
    metrics["acceptance"] = {"include_confirmation_at_least_0_90": include_ok, "exclude_confirmation_at_least_0_90": exclude_ok,
        "maximum_false_exclusions": maximum_false_exclusions, "observed_false_exclusions": false_exclusions,
        "false_exclusion_safety_status": safety, "all_false_exclusions_require_qualitative_review": True,
        "audit_acceptable": include_ok and exclude_ok and false_exclusions <= maximum_false_exclusions,
        "rule_note": ("The cost-conservative protocol permits at most 2/75 audited automatic exclusions (about 2.67%); this is not corpus prevalence."
                      if maximum_false_exclusions == 2 else "The historical protocol requires zero observed false exclusions.")}
    return metrics


def calculate_architecture_metrics(merged: pd.DataFrame) -> dict:
    """Report the two production audit arms without pooling their semantics."""
    include_rows = merged[merged.audit_arm.eq("automatic_include")]
    deterministic_rows = merged[merged.audit_arm.eq("deterministic_auto_exclude")]
    if len(include_rows) != 75 or len(deterministic_rows) != 50:
        raise ValueError(
            "Architecture audit requires exactly 75 automatic_include and "
            "50 deterministic_auto_exclude records"
        )
    include_confirmed = int(include_rows.researcher_label.eq("include").sum())
    exclusion_confirmed = int(deterministic_rows.researcher_label.eq("exclude").sum())
    reversals = int(deterministic_rows.researcher_label.eq("include").sum())
    include_rate = include_confirmed / len(include_rows)
    exclusion_rate = exclusion_confirmed / len(deterministic_rows)
    include_ok = include_rate >= .90
    exclusion_ok = exclusion_rate >= .90
    return {
        "audit_protocol_version": "architecture_aligned_audit_v2",
        "automatic_include_audit": {"audited_n": len(include_rows), "researcher_confirmed_include": include_confirmed, "not_confirmed": len(include_rows)-include_confirmed, "confirmation_rate": include_rate},
        "deterministic_exclusion_audit": {"audited_n": len(deterministic_rows), "researcher_confirmed_exclude": exclusion_confirmed, "researcher_reversed_to_include": reversals, "confirmation_rate": exclusion_rate, "zero_error_rule_of_three_upper_95_percent": 3/len(deterministic_rows) if reversals == 0 else None, "classification_note": "These are deterministic hashtag-only exclusions, not model classifier exclusions."},
        "acceptance": {"automatic_include_confirmation_at_least_0_90": include_ok, "deterministic_exclusion_confirmation_at_least_0_90": exclusion_ok, "zero_deterministic_exclusion_reversals": reversals == 0, "all_reversals_require_qualitative_review": True, "audit_acceptable": include_ok and exclusion_ok and reversals == 0},
        "denominator_note": "Operational human-review records are separate and excluded from both quality-audit arms. No combined balanced-audit accuracy or model-exclusion metric is reported.",
    }


def finalize_audit(blind_path: Path, key_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Final audit output already exists: {output_dir}")
    blind = pd.read_csv(blind_path, dtype=str, keep_default_na=False)
    key = pd.read_csv(key_path, dtype=str, keep_default_na=False)
    merged = _validate_submission(blind, key)
    package_manifest_path = key_path.parent / "manifest_v1.json"
    package_manifest = json.loads(package_manifest_path.read_text(encoding="utf-8")) if package_manifest_path.exists() else {}
    metrics = calculate_metrics(merged, int(package_manifest.get("maximum_false_exclusions", 0)))
    review_path = key_path.parent / "relevance_v8_operational_review_queue_v1.csv"
    if package_manifest_path.exists():
        source_path = Path(package_manifest["source_artifact"])
        if source_path.exists():
            source = pd.read_parquet(source_path) if source_path.suffix == ".parquet" else pd.read_csv(source_path, dtype=str, keep_default_na=False)
            normalized = _normalize_candidates(source)
            counts = normalized.automatic_decision.value_counts()
            total = len(normalized)
            review = pd.read_csv(review_path, dtype=str, keep_default_na=False) if review_path.exists() else pd.DataFrame()
            review_labels = review.researcher_label.str.lower() if len(review) else pd.Series(dtype=str)
            manual_n = int(total-counts.get("include", 0)-counts.get("exclude", 0))
            metrics["routing"] = {
                "total_records": total, "automatically_included": int(counts.get("include", 0)),
                "automatically_excluded": int(counts.get("exclude", 0)), "routed_to_manual_review": manual_n,
                "automatic_include_proportion": float(counts.get("include", 0)/total) if total else 0,
                "automatic_exclude_proportion": float(counts.get("exclude", 0)/total) if total else 0,
                "manual_review_proportion": float(manual_n/total) if total else 0,
                "review_researcher_included": int(review_labels.eq("include").sum()),
                "review_researcher_excluded": int(review_labels.eq("exclude").sum()),
                "review_unresolved": int((~review_labels.isin(VALID_LABELS)).sum()),
                "denominator_note": "Manual-review rows are excluded from automatic audit metrics.",
            }
    output_dir.mkdir(parents=True)
    errors = merged[merged.researcher_label.ne(merged.automatic_decision)].copy()
    errors.to_csv(output_dir / "relevance_v8_single_audit_errors_v1.csv", index=False, encoding="utf-8-sig")
    (output_dir / "relevance_v8_single_audit_metrics_v1.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    provenance = {"workflow_version": WORKFLOW_VERSION, "finalized_at_utc": utc_now(), "blind_submission_sha256": sha256(blind_path), "hidden_key_sha256": sha256(key_path), "api_calls_occurred": False, "output_hashes": {p.name: sha256(p) for p in output_dir.iterdir()}}
    (output_dir / "manifest_v1.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return metrics


def finalize_architecture_audit(blind_path: Path, key_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Final audit output already exists: {output_dir}")
    blind = pd.read_csv(blind_path, dtype=str, keep_default_na=False)
    key = pd.read_csv(key_path, dtype=str, keep_default_na=False)
    merged = _validate_submission(blind, key)
    if "audit_arm" not in merged:
        raise ValueError("Architecture audit key is missing audit_arm")
    metrics = calculate_architecture_metrics(merged)
    output_dir.mkdir(parents=True)
    errors = merged[
        ((merged.audit_arm == "automatic_include") & (merged.researcher_label != "include"))
        | ((merged.audit_arm == "deterministic_auto_exclude") & (merged.researcher_label != "exclude"))
    ].copy()
    errors.to_csv(output_dir / "relevance_v8_architecture_audit_errors_v2.csv", index=False, encoding="utf-8-sig")
    metrics_path = output_dir / "relevance_v8_architecture_audit_metrics_v2.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    provenance = {"workflow_version": ARCHITECTURE_AUDIT_VERSION, "finalized_at_utc": utc_now(), "blind_submission_sha256": sha256(blind_path), "hidden_key_sha256": sha256(key_path), "api_calls_occurred": False, "output_hashes": {p.name: sha256(p) for p in output_dir.iterdir()}}
    (output_dir / "manifest_v2.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return metrics


def finalize_single_audit(blind_path: Path, key_path: Path,
                          output_dir: Path) -> dict:
    """Finalize v2 by default while retaining explicit historical v1 support."""
    if not key_path.exists():
        raise FileNotFoundError(key_path)
    columns = pd.read_csv(key_path, nrows=0).columns
    if "audit_arm" in columns:
        return finalize_architecture_audit(blind_path, key_path, output_dir)
    return finalize_audit(blind_path, key_path, output_dir)


def validate_topic_unlock(decisions: pd.DataFrame, review: pd.DataFrame,
                          audit_metrics: dict) -> None:
    frame = _normalize_candidates(decisions)
    pending = frame[~frame.automatic_decision.isin(VALID_LABELS)]
    if len(review) != len(pending) or set(pending.document_id.astype(str)) != set(review.document_id.astype(str)):
        raise ValueError("Operational review queue does not align with all routed review records")
    if not review.researcher_label.fillna("").str.lower().isin(VALID_LABELS).all():
        raise RuntimeError("Topic discovery remains blocked: unresolved review records exist")
    if not audit_metrics.get("acceptance", {}).get("audit_acceptable", False):
        raise RuntimeError("Topic discovery remains blocked: audit acceptance is not established")


def atomic_save_annotations(path: Path, frame: pd.DataFrame) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8-sig", newline="", suffix=".csv", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            frame.to_csv(handle, index=False, lineterminator="\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()
