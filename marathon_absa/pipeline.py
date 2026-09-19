from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime, timezone
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from huggingface_hub import hf_hub_download

from .chunking import SentenceChunker
from .config import SETTINGS, Settings
from .ingest import load_documents
from .language import HybridLanguageService
from .holdout_v8 import assert_holdout_scoring_allowed
from .openai_service import (
    CachedOpenAI, RELEVANCE_INSTRUCTIONS, RELEVANCE_VERIFICATION_INSTRUCTIONS,
    RELEVANCE_V8_INSTRUCTIONS, RELEVANCE_V8_VERIFICATION_INSTRUCTIONS,
    absa_instructions,
)
from .relabel import (
    build_relabel_sample, build_repeat_disagreement_audit,
    finalize_relabel_sample, write_summary,
)
from .relevance import (
    apply_human_reviews, build_fresh_holdout_sample, build_review_queue, build_validation_sample,
    deterministic_result, needs_adjudication, route_results, route_verified_result, stratified_pilot,
    fresh_holdout_report, validate_topic_gate, validation_report,
)
from .schemas import (
    ABSA_SCHEMA, ASPECTS, LANGUAGE_REVIEW_SCHEMA, RELEVANCE_SCHEMA,
    RELEVANCE_VERIFICATION_SCHEMA, RELEVANCE_V8_ASSESSMENT_SCHEMA,
    V8_EXCLUDED_CONTENT_TYPES,
)
from .storage import read_table, write_table
from .relevance_v8 import (
    RELEVANCE_POLICY_VERSION as RELEVANCE_V8_POLICY_VERSION,
    build_error_audit, material_model_disagreement, recommend_operating_point,
    route_assessment, subgroup_performance,
    threshold_analysis,
    validation_report_v8,
)
from .text import stable_id
from .topics import fit_topics, initial_taxonomy

LANGUAGE_INSTRUCTIONS = """Identify the natural languages actually used in this social-media or blog text. Ignore URLs, usernames, event names, hashtags used only as metadata, and personal names. Return the primary language and every language with a meaningful span. Mark mixed only when more than one language contributes actual linguistic content. Distinguish Malay from Indonesian only when evidence supports it; otherwise use Malay/Indonesian uncertain. Do not infer language from location."""


class Pipeline:
    def __init__(self, settings: Settings = SETTINGS): self.settings = settings; settings.ensure_dirs()

    def download_language_model(self) -> str:
        path = hf_hub_download(repo_id=self.settings.openlid_repo, filename=self.settings.openlid_filename,
                               revision=self.settings.openlid_revision, local_dir=self.settings.model_dir)
        return path

    def prepare(self):
        documents = load_documents(self.settings.instagram_path, self.settings.blog_path)
        documents["event_year"] = documents.event_year.replace("", "Unknown")
        documents = HybridLanguageService(self.settings).apply(documents)
        units = SentenceChunker(self.settings).build_units(documents)
        write_table(documents, self.settings.output_dir / "documents.parquet")
        write_table(units, self.settings.output_dir / "units.parquet")
        self._manifest("prepare", {"documents": len(documents), "units": len(units), "language_model": self.settings.openlid_repo,
                                   "language_model_revision": self.settings.openlid_revision, "openlid_available": self.settings.openlid_path.exists()})
        return documents, units

    def language_review(self, limit=None, source=None):
        documents = read_table(self.settings.output_dir / "documents.parquet")
        candidates = documents[(documents.processing_status == "ready") & documents.language_adjudication_status.eq("pending")]
        if source: candidates = candidates[candidates.source.eq(source)]
        if limit: candidates = candidates.head(limit)
        service = CachedOpenAI(self.settings.cache_dir, self.settings.chat_model); rows=[]
        for row in candidates.itertuples(index=False):
            context = json.dumps({"local_primary": row.primary_language, "local_components": row.detected_languages,
                                  "local_reason": row.language_review_reason, "text": row.original_text}, ensure_ascii=False)
            result, meta = service.structured("language_review", "language-review-v1", LANGUAGE_INSTRUCTIONS, context, LANGUAGE_REVIEW_SCHEMA)
            rows.append({"document_id": row.document_id, **result, **meta})
        reviews = pd.DataFrame(rows)
        if len(reviews):
            for review in reviews.itertuples(index=False):
                mask = documents.document_id.eq(review.document_id)
                documents.loc[mask, ["primary_language", "detected_language", "language_iso", "is_mixed_language",
                                     "language_confidence", "language_method", "language_status", "language_adjudication_status",
                                     "language_review_reason"]] = [review.primary_language, review.primary_language, review.language_iso,
                                     review.is_mixed_language, review.confidence, "openai_adjudicated",
                                     "mixed" if review.is_mixed_language else "ok", "complete", review.reason]
                idx = documents.index[mask][0]; documents.at[idx, "detected_languages"] = review.detected_languages
            self._merge_incremental("language_reviews", reviews, "document_id")
            write_table(documents, self.settings.output_dir / "documents.parquet")
            write_table(SentenceChunker(self.settings).build_units(documents), self.settings.output_dir / "units.parquet")
        self._manifest("language_review", {"processed": len(reviews), "model": self.settings.chat_model})
        return reviews

    def _score_relevance_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        initial_service = CachedOpenAI(self.settings.cache_dir, self.settings.chat_model)
        adjudication_service = CachedOpenAI(self.settings.cache_dir, self.settings.stronger_model)
        rows = []
        for row in candidates.itertuples(index=False):
            rule = deterministic_result(row)
            if rule is not None:
                rows.append({
                    "document_id": row.document_id, **rule,
                    "initial_model": "rule", "initial_prompt_version": "relevance-rule-v7",
                    "initial_cached": False, "initial_input_tokens": None, "initial_output_tokens": None,
                    "adjudication_model": "", "adjudication_prompt_version": "",
                    "adjudication_cached": False, "adjudication_input_tokens": None,
                    "adjudication_output_tokens": None, "human_relevance": "", "review_notes": "",
                })
                continue
            context = json.dumps({
                "document_id": row.document_id, "event_year": row.event_year,
                "original_text": row.original_text, "linguistic_text": row.linguistic_text,
                "hashtags": row.hashtags, "primary_language": row.primary_language,
                "language_status": row.language_status,
            }, ensure_ascii=False)
            initial, initial_meta = initial_service.structured(
                "relevance_initial", "relevance-v7", RELEVANCE_INSTRUCTIONS, context, RELEVANCE_SCHEMA,
            )
            adjudicated = None
            adjudication_meta = {}
            if needs_adjudication(
                initial["relevance"], float(initial["confidence"]),
                initial.get("reason_code", ""),
            ):
                adjudication_context = json.dumps({
                    "caption": json.loads(context), "first_pass": initial,
                }, ensure_ascii=False)
                adjudicated, adjudication_meta = adjudication_service.structured(
                    "relevance_adjudication", "relevance-verification-v7",
                    RELEVANCE_VERIFICATION_INSTRUCTIONS, adjudication_context, RELEVANCE_VERIFICATION_SCHEMA,
                )
            routed = (
                route_verified_result(initial, adjudicated)
                if adjudicated is not None else route_results(initial)
            )
            rows.append({
                "document_id": row.document_id, **routed,
                "initial_model": initial_meta.get("model", ""),
                "initial_prompt_version": initial_meta.get("prompt_version", ""),
                "initial_cached": initial_meta.get("cached", False),
                "initial_input_tokens": initial_meta.get("input_tokens"),
                "initial_output_tokens": initial_meta.get("output_tokens"),
                "adjudication_model": adjudication_meta.get("model", ""),
                "adjudication_prompt_version": adjudication_meta.get("prompt_version", ""),
                "adjudication_cached": adjudication_meta.get("cached", False),
                "adjudication_input_tokens": adjudication_meta.get("input_tokens"),
                "adjudication_output_tokens": adjudication_meta.get("output_tokens"),
                "human_relevance": "", "review_notes": "",
            })
        fresh = pd.DataFrame(rows)
        return fresh

    def _score_relevance_v8_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Score v8 assessments into an isolated schema/cache namespace."""
        assert_holdout_scoring_allowed(
            set(candidates.document_id.astype(str)), self.settings.output_dir,
        )
        service = CachedOpenAI(self.settings.cache_dir, self.settings.chat_model)
        rows = []
        for row in candidates.itertuples(index=False):
            context = json.dumps({
                "document_id": row.document_id, "event_year": row.event_year,
                "original_text": row.original_text, "linguistic_text": row.linguistic_text,
                "hashtags": row.hashtags, "primary_language": row.primary_language,
                "language_status": row.language_status,
            }, ensure_ascii=False)
            assessment, meta = service.structured(
                "relevance_v8_initial", "v8", RELEVANCE_V8_INSTRUCTIONS,
                context, RELEVANCE_V8_ASSESSMENT_SCHEMA,
            )
            needs_verification = (
                assessment["event_connection"] in {"weak", "none"}
                or assessment["primary_content_type"] in V8_EXCLUDED_CONTENT_TYPES
                or assessment["confidence"] < 0.93
                or assessment["contradiction_present"]
                or assessment["image_dependent"]
                or assessment["requires_review_recommendation"]
            )
            verification = None
            verification_meta = {}
            if needs_verification:
                verification_service = CachedOpenAI(
                    self.settings.cache_dir, self.settings.stronger_model)
                verification_context = json.dumps({
                    "caption": json.loads(context), "first_assessment": assessment,
                }, ensure_ascii=False)
                verification, verification_meta = verification_service.structured(
                    "relevance_v8_verification", "v8",
                    RELEVANCE_V8_VERIFICATION_INSTRUCTIONS, verification_context,
                    RELEVANCE_V8_ASSESSMENT_SCHEMA,
                )
            routed = route_assessment(assessment, verification)
            rows.append({
                "document_id": row.document_id, **routed,
                "v8_initial_model": meta.get("model", ""),
                "v8_initial_prompt_version": meta.get("prompt_version", ""),
                "v8_initial_schema_version": "v8",
                "v8_initial_cached": meta.get("cached", False),
                "v8_initial_input_tokens": meta.get("input_tokens"),
                "v8_initial_output_tokens": meta.get("output_tokens"),
                "v8_verification_model": verification_meta.get("model", ""),
                "v8_verification_prompt_version": verification_meta.get("prompt_version", ""),
                "v8_verification_cached": verification_meta.get("cached", False),
                "v8_model_agreement": "not_checked" if verification is None else (
                    "material_disagreement" if material_model_disagreement(assessment, verification)
                    else "no_material_disagreement"),
                "v8_human_final_inclusion": "", "v8_review_notes": "",
            })
        return pd.DataFrame(rows)

    def relevance_v8_score_calibration(self):
        """Controlled paid v8 calibration; never mutates production or v7 artifacts."""
        output = self.settings.output_dir
        sample_path = output / "relevance_v7_calibration_sample.csv"
        key_path = output / "relevance_v7_calibration_key.parquet"
        result_dir = output / "relevance_v8_calibration_results"
        if result_dir.exists():
            raise FileExistsError("v8 calibration results exist; refusing to overwrite")
        if not sample_path.exists() or not key_path.exists():
            raise FileNotFoundError("Completed v7 calibration labels and key are required")
        reviewer = pd.read_csv(sample_path, dtype=str, keep_default_na=False)
        originals = reviewer[reviewer.repeat_of.fillna("").eq("")].copy()
        labels = originals.human_relevance.str.strip().str.lower()
        if not labels.isin({"relevant", "irrelevant"}).all():
            raise ValueError("Calibration requires final binary human labels")
        reviewer["human_final_inclusion"] = reviewer.human_relevance.str.lower().map(
            {"relevant": "include", "irrelevant": "exclude"}).fillna("")
        originals = reviewer[reviewer.repeat_of.fillna("").eq("")].copy()
        key = read_table(key_path)
        documents = read_table(output / "documents.parquet")
        candidates = documents[documents.document_id.isin(set(key.document_id))].copy()
        if set(candidates.document_id) != set(key.document_id):
            raise ValueError("Some calibration documents are missing")
        scored = self._score_relevance_v8_candidates(candidates)
        context_columns = ["document_id", "event_year", "primary_language", "language_status", "original_text"]
        evaluated = scored.merge(documents[context_columns], on="document_id", how="left")
        lengths = candidates.set_index("document_id").linguistic_text.fillna("").str.len()
        evaluated["length_band"] = pd.qcut(
            evaluated.document_id.map(lengths).rank(method="first"), q=4,
            labels=False, duplicates="drop")
        result_dir.mkdir(parents=True)
        write_table(evaluated, result_dir / "relevance_v8_calibration_predictions.parquet")
        report = validation_report_v8(reviewer, evaluated, result_dir, "calibration")
        gold = originals[["document_id", "human_final_inclusion"]]
        thresholds = threshold_analysis(scored, gold)
        thresholds.to_csv(result_dir / "relevance_v8_threshold_analysis.csv", index=False)
        recommendation = recommend_operating_point(thresholds)
        (result_dir / "relevance_v8_threshold_recommendation.json").write_text(
            json.dumps({"recommended_operating_point": recommendation}, indent=2), encoding="utf-8")
        merged = evaluated.merge(gold, on="document_id", how="left")
        build_error_audit(merged).to_csv(result_dir / "relevance_v8_error_audit.csv", index=False)
        subgroup_performance(merged).to_csv(result_dir / "relevance_v8_subgroup_metrics.csv", index=False)
        self._manifest("relevance_v8_score_calibration", {
            "policy_version": RELEVANCE_V8_POLICY_VERSION, "scored": len(scored),
            "output_dir": str(result_dir), **report,
        })
        return report
    def relevance(self, limit=None, pilot_size=None):
        documents = read_table(self.settings.output_dir / "documents.parquet")
        candidates = documents[(documents.source == "instagram") & (documents.processing_status == "ready")].copy()
        if pilot_size:
            candidates = stratified_pilot(candidates, pilot_size, self.settings.random_seed)
        elif limit:
            candidates = candidates.head(limit)
        fresh = self._score_relevance_candidates(candidates)
        path = self.settings.output_dir / "relevance.parquet"
        preserved_reviews = pd.DataFrame()
        if path.exists() and len(fresh):
            old = read_table(path)
            if {"document_id", "human_relevance", "review_notes"}.issubset(old.columns):
                preserved_reviews = old[
                    old.human_relevance.fillna("").astype(str).str.lower().isin(["relevant", "irrelevant"])
                ][["document_id", "human_relevance", "review_notes"]].copy()
            combined = pd.concat([old[~old.document_id.isin(fresh.document_id)], fresh], ignore_index=True)
        else:
            combined = fresh
        if len(preserved_reviews):
            combined = apply_human_reviews(combined, preserved_reviews)
        queue_csv = self.settings.output_dir / "relevance_review_queue.csv"
        if queue_csv.exists():
            existing_queue = pd.read_csv(queue_csv, dtype=str, keep_default_na=False)
            combined = apply_human_reviews(combined, existing_queue)
        write_table(combined, path)
        queue = build_review_queue(combined, documents, queue_csv)
        write_table(queue, self.settings.output_dir / "relevance_review_queue.parquet")
        counts = combined.relevance.value_counts().to_dict() if len(combined) else {}
        self._manifest("relevance", {
            "processed": len(fresh), "classified_total": len(combined),
            "pending_review": int(combined.review_status.eq("pending").sum()) if len(combined) else 0,
            "counts": counts, "initial_model": self.settings.chat_model,
            "adjudication_model": self.settings.stronger_model,
        })
        return fresh

    def relevance_review_sample(self, size=300, repeats=30):
        documents = read_table(self.settings.output_dir / "documents.parquet")
        relevance = read_table(self.settings.output_dir / "relevance.parquet")
        reviewer, key = build_validation_sample(relevance, documents, size, repeats, self.settings.random_seed)
        reviewer.to_csv(self.settings.output_dir / "relevance_validation_sample.csv", index=False, encoding="utf-8-sig")
        write_table(key, self.settings.output_dir / "relevance_validation_key.parquet")
        return reviewer

    def relevance_fresh_holdout(self, size=150, repeats=15):
        """Create a new blind holdout while excluding prior validation IDs."""
        output_dir = self.settings.output_dir
        sample_path = output_dir / "relevance_fresh_holdout_sample.csv"
        key_csv_path = output_dir / "relevance_fresh_holdout_key.csv"
        key_parquet_path = output_dir / "relevance_fresh_holdout_key.parquet"
        if sample_path.exists() or key_csv_path.exists() or key_parquet_path.exists():
            raise FileExistsError("Fresh holdout artifacts already exist; refusing to overwrite them")
        excluded_ids: set[str] = set()
        for path in [output_dir / "relevance_validation_sample_before_recalibration.csv", output_dir / "relevance_validation_sample.csv"]:
            if path.exists():
                prior = pd.read_csv(path, dtype=str, keep_default_na=False)
                if "document_id" in prior:
                    excluded_ids.update(prior.document_id[prior.document_id.ne("")])
        documents = read_table(output_dir / "documents.parquet")
        relevance = read_table(output_dir / "relevance.parquet")
        reviewer, key = build_fresh_holdout_sample(relevance, documents, excluded_ids, size, repeats, self.settings.random_seed)
        reviewer.to_csv(sample_path, index=False, encoding="utf-8-sig")
        key.to_csv(key_csv_path, index=False, encoding="utf-8-sig")
        write_table(key, key_parquet_path)
        self._manifest("relevance_fresh_holdout", {"base_records": len(key), "reviewer_rows": len(reviewer), "repeats": int(reviewer.repeat_of.ne("").sum()), "excluded_prior_document_ids": len(excluded_ids), "seed": self.settings.random_seed + 1000})
        return reviewer
    def relevance_v7_sample(self, stage: str, size: int, repeats: int):
        """Create a versioned blind sample without exposing prior/model labels."""
        if stage not in {"pilot", "calibration", "fresh_holdout"}:
            raise ValueError("stage must be pilot, calibration, or fresh_holdout")
        output_dir = self.settings.output_dir
        stem = f"relevance_v7_{stage}"
        sample_path = output_dir / f"{stem}_sample.csv"
        key_csv_path = output_dir / f"{stem}_key.csv"
        key_parquet_path = output_dir / f"{stem}_key.parquet"
        if sample_path.exists() or key_csv_path.exists() or key_parquet_path.exists():
            raise FileExistsError(f"{stage} v7 artifacts already exist; refusing to overwrite")
        excluded_ids: set[str] = set()
        for prior_stage in ["pilot", "calibration", "fresh_holdout"]:
            prior = output_dir / f"relevance_v7_{prior_stage}_key.csv"
            if prior.exists():
                prior_frame = pd.read_csv(prior, dtype=str, keep_default_na=False)
                excluded_ids.update(prior_frame.document_id)
        documents = read_table(output_dir / "documents.parquet")
        stage_offset = {"pilot": 7000, "calibration": 8000, "fresh_holdout": 9000}[stage]
        reviewer, key = build_relabel_sample(
            documents, excluded_ids, stage, size, repeats,
            self.settings.random_seed + stage_offset,
        )
        reviewer.to_csv(sample_path, index=False, encoding="utf-8-sig")
        key.to_csv(key_csv_path, index=False, encoding="utf-8-sig")
        write_table(key, key_parquet_path)
        self._manifest("relevance_v7_sample", {
            "relabel_stage": stage, "base_records": len(key), "reviewer_rows": len(reviewer),
            "repeats": int(reviewer.repeat_of.ne("").sum()),
            "excluded_prior_document_ids": len(excluded_ids),
        })
        return reviewer

    def relevance_v7_finalize(self, stage: str):
        """Require final binary labels and report single-reviewer consistency."""
        output_dir = self.settings.output_dir
        sample_path = output_dir / f"relevance_v7_{stage}_sample.csv"
        summary_path = output_dir / f"relevance_v7_{stage}_summary.json"
        if not sample_path.exists():
            raise FileNotFoundError(f"Missing v7 {stage} sample: {sample_path}")
        if summary_path.exists():
            raise FileExistsError(f"v7 {stage} summary already exists; refusing to overwrite")
        reviewer = pd.read_csv(sample_path, dtype=str, keep_default_na=False)
        report = finalize_relabel_sample(reviewer, stage)
        write_summary(report, summary_path)
        self._manifest("relevance_v7_finalize", {**report, "stage": "relevance_v7_finalize", "relabel_stage": stage})
        return report
    def relevance_v7_repeat_audit(self, stage: str):
        """Write an append-only audit of inconsistent blind repeat labels."""
        output_dir = self.settings.output_dir
        sample_path = output_dir / f"relevance_v7_{stage}_sample.csv"
        summary_path = output_dir / f"relevance_v7_{stage}_summary.json"
        audit_path = output_dir / f"relevance_v7_{stage}_repeat_disagreements.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Finalize v7 {stage} before creating its repeat audit")
        if audit_path.exists():
            raise FileExistsError(f"v7 {stage} repeat audit already exists; refusing to overwrite")
        reviewer = pd.read_csv(sample_path, dtype=str, keep_default_na=False)
        audit = build_repeat_disagreement_audit(reviewer)
        audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
        queue = audit.copy()
        queue["human_relevance"] = ""
        queue["review_notes"] = ""
        queue["review_status"] = "repeat_adjudication"
        queue.to_csv(
            output_dir / f"relevance_v7_{stage}_repeat_adjudication_queue.csv",
            index=False, encoding="utf-8-sig",
        )
        self._manifest("relevance_v7_repeat_audit", {
            "relabel_stage": stage, "disagreements": len(audit),
            "audit_path": str(audit_path),
        })
        return audit

    def relevance_v7_score_calibration(self):
        """Score the completed v7 calibration set without touching production data."""
        output_dir = self.settings.output_dir
        sample_path = output_dir / "relevance_v7_calibration_sample.csv"
        key_path = output_dir / "relevance_v7_calibration_key.parquet"
        summary_path = output_dir / "relevance_v7_calibration_summary.json"
        adjudication_path = output_dir / "relevance_v7_calibration_repeat_adjudication_queue.csv"
        result_dir = output_dir / "relevance_v7_calibration_results"
        if result_dir.exists():
            raise FileExistsError("v7 calibration results already exist; refusing to overwrite")
        if not sample_path.exists() or not key_path.exists() or not summary_path.exists():
            raise FileNotFoundError("Completed and finalized v7 calibration artifacts are required")
        reviewer = pd.read_csv(sample_path, dtype=str, keep_default_na=False)
        finalize_relabel_sample(reviewer, "calibration")
        if not adjudication_path.exists():
            raise FileNotFoundError("Complete the v7 calibration repeat adjudication queue first")
        adjudication = pd.read_csv(adjudication_path, dtype=str, keep_default_na=False)
        adjudicated = adjudication.human_relevance.str.strip().str.lower()
        if not adjudicated.isin({"relevant", "irrelevant"}).all():
            raise ValueError("Complete all repeat adjudication labels before calibration scoring")
        decisions = dict(zip(adjudication.document_id, adjudicated))
        original_mask = reviewer.repeat_of.fillna("").eq("")
        reviewer.loc[original_mask, "human_relevance"] = reviewer.loc[
            original_mask, "document_id"
        ].map(decisions).fillna(reviewer.loc[original_mask, "human_relevance"])
        key = read_table(key_path)
        documents = read_table(output_dir / "documents.parquet")
        requested_ids = set(key.document_id)
        candidates = documents[documents.document_id.isin(requested_ids)].copy()
        if set(candidates.document_id) != requested_ids:
            raise ValueError("Some v7 calibration documents are missing from prepared data")
        scored = self._score_relevance_candidates(candidates)
        score_columns = scored[[
            "document_id", "relevance", "confidence", "reason_code",
            "initial_prompt_version", "adjudication_prompt_version",
        ]].rename(columns={
            "relevance": "model_relevance", "confidence": "model_confidence",
            "reason_code": "model_reason_code",
        })
        rescored_key = key.merge(score_columns, on="document_id", how="left")
        if rescored_key.model_relevance.isna().any():
            raise RuntimeError("Some v7 calibration documents were not scored")
        lengths = candidates.set_index("document_id").linguistic_text.fillna("").str.len()
        ranked = rescored_key.document_id.map(lengths).rank(method="first")
        rescored_key["length_band"] = pd.qcut(
            ranked, q=min(4, len(rescored_key)), labels=False, duplicates="drop",
        )
        rescored_key["validation_split"] = "calibration"
        result_dir.mkdir(parents=True)
        reviewer.to_csv(result_dir / "relevance_validation_sample.csv", index=False, encoding="utf-8-sig")
        write_table(rescored_key, result_dir / "relevance_validation_key.parquet")
        report = validation_report(
            reviewer, rescored_key, result_dir, evaluation_split="calibration",
        )
        self._manifest("relevance_v7_score_calibration", {
            "scored": len(rescored_key), "output_dir": str(result_dir), **report,
        })
        return report
    def relevance_policy_alignment_sample(self):
        """Create the blind 60-record second-coder policy-alignment package."""
        output_dir = self.settings.output_dir
        reviewer_path = output_dir / "relevance_policy_alignment_second_coder.csv"
        key_path = output_dir / "relevance_policy_alignment_key.csv"
        rubric_path = output_dir / "relevance_policy_alignment_rubric.md"
        if reviewer_path.exists() or key_path.exists() or rubric_path.exists():
            raise FileExistsError("Policy-alignment artifacts already exist; refusing to overwrite")
        sample = pd.read_csv(output_dir / "relevance_validation_sample_before_recalibration.csv", dtype=str, keep_default_na=False)
        v5 = pd.read_csv(output_dir / "relevance_v5_calibration" / "relevance_validation_key.csv", dtype=str, keep_default_na=False)
        audit = pd.read_csv(output_dir / "relevance_calibration_disagreement_audit_v4.csv", dtype=str, keep_default_na=False)
        originals = sample[sample.repeat_of.fillna("").eq("")].copy()
        merged = v5.merge(originals[["review_id", "document_id", "original_text", "human_relevance"]], on=["review_id", "document_id"], how="inner")
        proposals = dict(zip(audit.review_id, audit.proposed_label))
        merged["policy_overlay_label"] = merged.apply(lambda row: proposals.get(row.review_id, row.human_relevance), axis=1)
        rng = self.settings.random_seed + 2000
        groups = []
        definitions = [
            ("human_relevant_v5_irrelevant", merged.human_relevance.eq("relevant") & merged.model_relevance.eq("irrelevant"), 20),
            ("human_relevant_v5_ambiguous", merged.human_relevance.eq("relevant") & merged.model_relevance.eq("ambiguous"), 20),
            ("nonrelevant_v5_disputed", ~merged.human_relevance.eq("relevant") & merged.model_relevance.isin(["relevant", "ambiguous"]), 10),
        ]
        selected_ids: set[str] = set()
        for index, (name, mask, count) in enumerate(definitions):
            pool = merged[mask & ~merged.review_id.isin(selected_ids)]
            if len(pool) < count:
                raise ValueError(f"Policy-alignment stratum {name} has only {len(pool)} rows; {count} required")
            chosen = pool.sample(count, random_state=rng + index).copy()
            chosen["alignment_stratum"] = name
            groups.append(chosen)
            selected_ids.update(chosen.review_id)
        controls = []
        for index, label in enumerate(["relevant", "irrelevant"]):
            pool = merged[merged.human_relevance.eq(label) & merged.model_relevance.eq(label) & ~merged.review_id.isin(selected_ids)]
            if len(pool) < 5:
                raise ValueError(f"Only {len(pool)} {label} agreement controls; 5 required")
            chosen = pool.sample(5, random_state=rng + 10 + index).copy()
            chosen["alignment_stratum"] = f"agreement_control_{label}"
            controls.append(chosen)
        chosen = pd.concat(groups + controls, ignore_index=True).sample(frac=1, random_state=rng + 20).reset_index(drop=True)
        chosen["alignment_id"] = [f"align-{i + 1:03d}" for i in range(len(chosen))]
        reviewer = chosen[["alignment_id", "document_id", "event_year", "primary_language", "language_status", "original_text"]].copy()
        reviewer["second_coder_label"] = ""
        reviewer["visual_context_required"] = ""
        reviewer["reason_category"] = ""
        reviewer["evidence_span"] = ""
        reviewer["coder_notes"] = ""
        hidden = chosen[["alignment_id", "document_id", "review_id", "alignment_stratum", "human_relevance", "policy_overlay_label", "model_relevance", "model_confidence", "model_reason_code", "initial_prompt_version", "adjudication_prompt_version"]].rename(columns={"human_relevance": "original_human_label", "model_relevance": "v5_model_relevance", "model_confidence": "v5_model_confidence", "model_reason_code": "v5_model_reason_code"})
        reviewer.to_csv(reviewer_path, index=False, encoding="utf-8-sig")
        hidden.to_csv(key_path, index=False, encoding="utf-8-sig")
        rubric_path.write_text("""# KLSCM relevance policy-alignment rubric\n\nLabel `relevant` only when the available caption text makes KLSCM the actual event subject and states a concrete KLSCM registration, preparation, logistics, information, participation, experience, result, support, or evaluation relationship. Label `irrelevant` when another event/activity, generic running, lifestyle, promotion, or metadata is the actual subject. A KLSCM hashtag alone, or beside generic motivation, dates, distances, apparel, celebration, or photo credit, is insufficient.\n\nSet `visual_context_required` to `yes` only when the text cannot decide the substantive label without the unavailable image; otherwise 
o`. Choose one reason category: `concrete_klscm_journey`, `another_event`, `generic_running`, `commercial_promotion`, `lifestyle_or_spam`, `incidental_hashtag`, `image_dependent`, or `insufficient_text`. Copy a short exact evidence span. Do not open the hidden key or consult prior/model labels.\n""", encoding="utf-8")
        self._manifest("relevance_policy_alignment_sample", {"records": len(reviewer), "seed": rng, "strata": hidden.alignment_stratum.value_counts().to_dict()})
        return reviewer
    def relevance_policy_alignment_finalize(self):
        """Validate second-coder labels and create append-only adjudication artifacts."""
        output_dir = self.settings.output_dir
        reviewer_path = output_dir / "relevance_policy_alignment_second_coder.csv"
        key_path = output_dir / "relevance_policy_alignment_key.csv"
        audit_path = output_dir / "relevance_policy_alignment_audit.csv"
        queue_path = output_dir / "relevance_policy_alignment_adjudication_queue.csv"
        summary_path = output_dir / "relevance_policy_alignment_agreement.json"
        if audit_path.exists() or queue_path.exists() or summary_path.exists():
            raise FileExistsError("Policy-alignment finalization artifacts already exist; refusing to overwrite")
        reviewer = pd.read_csv(reviewer_path, dtype=str, keep_default_na=False)
        key = pd.read_csv(key_path, dtype=str, keep_default_na=False)
        if len(reviewer) != 60 or set(reviewer.alignment_id) != set(key.alignment_id):
            raise ValueError("Policy-alignment reviewer/key mismatch")
        labels = reviewer.second_coder_label.str.strip().str.lower()
        visuals = reviewer.visual_context_required.str.strip().str.lower()
        valid_reasons = {"concrete_klscm_journey", "another_event", "generic_running", "commercial_promotion", "lifestyle_or_spam", "incidental_hashtag", "image_dependent", "insufficient_text"}
        if not labels.isin({"relevant", "irrelevant"}).all():
            raise ValueError("Complete all second-coder substantive labels")
        if not visuals.isin({"yes", "no"}).all():
            raise ValueError("visual_context_required must be yes or no")
        if not reviewer.reason_category.str.strip().str.lower().isin(valid_reasons).all():
            raise ValueError("Invalid or missing second-coder reason category")
        if reviewer.evidence_span.str.strip().eq("").any():
            raise ValueError("Every second-coder row requires an evidence span")
        merged = key.merge(reviewer, on=["alignment_id", "document_id"], how="inner")
        comparable = merged.original_human_label.isin(["relevant", "irrelevant"])
        agreement = float((merged.loc[comparable, "original_human_label"] == merged.loc[comparable, "second_coder_label"]).mean())
        kappa = float(cohen_kappa_score(merged.loc[comparable, "original_human_label"], merged.loc[comparable, "second_coder_label"]))
        merged["adjudicated_label"] = ""
        merged["adjudication_reason"] = ""
        merged["adjudicated_at"] = ""
        merged.to_csv(audit_path, index=False, encoding="utf-8-sig")
        disagreements = merged[~merged.original_human_label.eq(merged.second_coder_label)].copy()
        disagreements.to_csv(queue_path, index=False, encoding="utf-8-sig")
        summary_path.write_text(json.dumps({"records": len(merged), "comparable_records": int(comparable.sum()), "agreements": int((merged.loc[comparable, "original_human_label"] == merged.loc[comparable, "second_coder_label"]).sum()), "agreement_rate": agreement, "cohen_kappa": kappa, "adjudication_required": len(disagreements)}, indent=2), encoding="utf-8")
        return disagreements
    def relevance_finalize(self):
        documents = read_table(self.settings.output_dir / "documents.parquet")
        relevance_path = self.settings.output_dir / "relevance.parquet"
        queue_path = self.settings.output_dir / "relevance_review_queue.csv"
        relevance = read_table(relevance_path)
        if not queue_path.exists():
            raise FileNotFoundError("relevance_review_queue.csv does not exist")
        queue = pd.read_csv(queue_path, dtype=str, keep_default_na=False)
        relevance = apply_human_reviews(relevance, queue)
        write_table(relevance, relevance_path)
        pending = build_review_queue(relevance, documents, queue_path)
        write_table(pending, self.settings.output_dir / "relevance_review_queue.parquet")
        self._manifest("relevance_finalize", {
            "human_resolved": int(relevance.adjudication_method.eq("human_review").sum()),
            "pending_review": int(relevance.review_status.eq("pending").sum()),
            "counts": relevance.relevance.value_counts().to_dict(),
        })
        return relevance

    def relevance_rescore_validation(self, split: str = "calibration"):
        """Rescore the archived sample with v4 without changing production data."""
        sample_path = (
            self.settings.output_dir
            / "relevance_validation_sample_before_recalibration.csv"
        )
        key_path = (
            self.settings.output_dir
            / "relevance_validation_key_before_recalibration.csv"
        )
        if not sample_path.exists() or not key_path.exists():
            raise FileNotFoundError(
                "Completed pre-recalibration validation sample and key are required"
            )
        reviewer = pd.read_csv(sample_path, dtype=str, keep_default_na=False)
        key = pd.read_csv(key_path, dtype=str, keep_default_na=False)
        originals = reviewer[reviewer.repeat_of.fillna("").eq("")]
        labels = originals.human_relevance.fillna("").str.strip().str.lower()
        missing_labels = ~labels.isin({"relevant", "ambiguous", "irrelevant"})
        if missing_labels.any():
            raise ValueError(
                f"Archived validation sample has {int(missing_labels.sum())} "
                "missing or invalid original labels"
            )

        documents = read_table(self.settings.output_dir / "documents.parquet")
        if split not in {"calibration", "holdout"}:
            raise ValueError("split must be calibration or holdout")
        selected_key = key[key.validation_split.eq(split)].copy()
        requested_ids = set(selected_key.document_id)
        candidates = documents[documents.document_id.isin(requested_ids)].copy()
        missing_documents = requested_ids - set(candidates.document_id)
        if missing_documents:
            raise ValueError(
                f"Validation documents are missing from prepared data: "
                f"{len(missing_documents)}"
            )
        scored = self._score_relevance_candidates(candidates)
        score_columns = scored[[
            "document_id", "relevance", "confidence", "reason_code",
            "initial_prompt_version", "adjudication_prompt_version",
        ]].rename(columns={
            "relevance": "v4_model_relevance",
            "confidence": "v4_model_confidence",
            "reason_code": "v4_model_reason_code",
        })
        rescored_key = selected_key.drop(columns=[
            "model_relevance", "model_confidence", "model_reason_code",
        ], errors="ignore").merge(score_columns, on="document_id", how="left")
        rescored_key = rescored_key.rename(columns={
            "v4_model_relevance": "model_relevance",
            "v4_model_confidence": "model_confidence",
            "v4_model_reason_code": "model_reason_code",
        })
        if rescored_key.model_relevance.isna().any():
            raise RuntimeError("Some validation documents were not rescored")

        output_dir = self.settings.output_dir / f"relevance_v6_{split}"
        output_dir.mkdir(parents=True, exist_ok=True)
        reviewer.to_csv(
            output_dir / "relevance_validation_sample.csv",
            index=False,
            encoding="utf-8-sig",
        )
        write_table(
            rescored_key,
            output_dir / "relevance_validation_key.parquet",
        )
        report = validation_report(
            reviewer, rescored_key, output_dir, evaluation_split=split,
        )
        self._manifest("relevance_rescore_validation", {
            "rescored": len(rescored_key),
            "output_dir": str(output_dir),
            **report,
        })
        return report
    def relevance_rescore_fresh_holdout(self):
        """Score the completed independent holdout once with frozen v4."""
        output_dir = self.settings.output_dir
        reviewer_path = output_dir / "relevance_fresh_holdout_sample.csv"
        key_path = output_dir / "relevance_fresh_holdout_key.parquet"
        result_dir = output_dir / "relevance_v4_fresh_holdout"
        if result_dir.exists():
            raise FileExistsError("Fresh holdout result directory already exists; refusing to overwrite")
        if not reviewer_path.exists() or not key_path.exists():
            raise FileNotFoundError("Fresh holdout reviewer and key are required")
        reviewer = pd.read_csv(reviewer_path, dtype=str, keep_default_na=False)
        key = read_table(key_path)
        base = reviewer[reviewer.repeat_of.fillna("").eq("")].copy()
        repeats = reviewer[reviewer.repeat_of.fillna("").ne("")].copy()
        valid_labels = {"relevant", "irrelevant", "ambiguous"}
        labels = reviewer.human_relevance.fillna("").str.strip().str.lower()
        if len(base) != 150 or len(repeats) != 15:
            raise ValueError("Fresh holdout must contain 150 base rows and 15 repeats")
        if not labels.isin(valid_labels).all():
            raise ValueError("Fresh holdout contains missing or invalid labels")
        if len(key) != 150 or not key.validation_split.eq("fresh_holdout").all():
            raise ValueError("Fresh holdout key must contain 150 fresh_holdout rows")
        if set(base.review_id) != set(key.review_id) or set(base.document_id) != set(key.document_id):
            raise ValueError("Fresh holdout reviewer and key do not align")
        prior_ids: set[str] = set()
        for path in [output_dir / "relevance_validation_sample_before_recalibration.csv", output_dir / "relevance_validation_sample.csv"]:
            if path.exists():
                prior = pd.read_csv(path, dtype=str, keep_default_na=False)
                prior_ids.update(prior.document_id[prior.document_id.ne("")])
        overlap = set(key.document_id) & prior_ids
        if overlap:
            raise ValueError(f"Fresh holdout leaks {len(overlap)} prior validation documents")
        documents = read_table(output_dir / "documents.parquet")
        candidates = documents[documents.document_id.isin(set(key.document_id))].copy()
        if set(candidates.document_id) != set(key.document_id):
            raise ValueError("Some fresh holdout documents are missing")
        hashes = {"reviewer_sha256": hashlib.sha256(reviewer_path.read_bytes()).hexdigest(), "key_parquet_sha256": hashlib.sha256(key_path.read_bytes()).hexdigest()}
        key_csv_path = output_dir / "relevance_fresh_holdout_key.csv"
        if key_csv_path.exists():
            hashes["key_csv_sha256"] = hashlib.sha256(key_csv_path.read_bytes()).hexdigest()
        scored = self._score_relevance_candidates(candidates)
        scored = scored[["document_id", "relevance", "confidence", "reason_code", "initial_prompt_version", "adjudication_prompt_version"]].rename(columns={"relevance": "model_relevance", "confidence": "model_confidence", "reason_code": "model_reason_code"})
        if len(scored) != 150 or scored.document_id.nunique() != 150:
            raise RuntimeError("Fresh holdout scorer did not return 150 unique records")
        rescored_key = key.drop(columns=["model_relevance", "model_confidence", "model_reason_code"], errors="ignore").merge(scored, on="document_id", how="left")
        if rescored_key.model_relevance.isna().any():
            raise RuntimeError("Some fresh holdout documents were not rescored")
        result_dir.mkdir(parents=True)
        write_table(rescored_key, result_dir / "relevance_fresh_holdout_key.parquet")
        (result_dir / "input_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
        report = fresh_holdout_report(reviewer, rescored_key, result_dir)
        self._manifest("relevance_rescore_fresh_holdout", {"rescored": len(rescored_key), "output_dir": str(result_dir), **hashes, **report})
        return report
    def relevance_validate(self):
        sample_path = self.settings.output_dir / "relevance_validation_sample.csv"
        key_path = self.settings.output_dir / "relevance_validation_key.parquet"
        if not sample_path.exists() or not key_path.exists():
            raise FileNotFoundError("Create and complete the relevance validation sample first")
        reviewer = pd.read_csv(sample_path, dtype=str, keep_default_na=False)
        key = read_table(key_path)
        return validation_report(reviewer, key, self.settings.output_dir)
    def topics(self, limit=None):
        units=read_table(self.settings.output_dir/"units.parquet"); relevance=read_table(self.settings.output_dir/"relevance.parquet"); documents=read_table(self.settings.output_dir/"documents.parquet")
        validate_topic_gate(documents,relevance)
        selected=units[units.document_id.isin(set(relevance.loc[relevance.include_in_topics.astype(bool),"document_id"]) | set(documents.loc[(documents.source=="blog")&(documents.processing_status=="ready"),"document_id"]))].copy()
        if limit: selected=selected.head(limit)
        embedder=CachedOpenAI(self.settings.cache_dir,self.settings.embedding_model); embeddings=[]
        for start in range(0,len(selected),256): embeddings.extend(embedder.embeddings(selected.text.iloc[start:start+256].tolist()))
        assignments,info,model=fit_topics(selected,embeddings,self.settings); write_table(assignments,self.settings.output_dir/"topic_assignments.parquet"); write_table(info,self.settings.output_dir/"topic_info.parquet")
        taxonomy=self.settings.output_dir/"taxonomy.csv"
        if not taxonomy.exists(): initial_taxonomy(info).to_csv(taxonomy,index=False,encoding="utf-8-sig")
        with (self.settings.output_dir/"bertopic_model.pkl").open("wb") as handle: pickle.dump(model,handle)
        return assignments,info

    def absa(self, limit=None):
        units=read_table(self.settings.output_dir/"units.parquet"); relevance=read_table(self.settings.output_dir/"relevance.parquet"); documents=read_table(self.settings.output_dir/"documents.parquet")
        validate_topic_gate(documents,relevance)
        selected=units[units.document_id.isin(set(relevance.loc[relevance.include_in_topics.astype(bool),"document_id"]) | set(documents.loc[(documents.source=="blog")&(documents.processing_status=="ready"),"document_id"]))].copy()
        if limit: selected=selected.head(limit)
        aspects=self._approved_aspects(); service=CachedOpenAI(self.settings.cache_dir,self.settings.chat_model); rows=[]; logs=[]
        for unit in selected.itertuples(index=False):
            result,meta=service.structured("absa","absa-v1",absa_instructions(aspects),unit.text,ABSA_SCHEMA,extra=json.dumps(aspects)); logs.append({"unit_id":unit.unit_id,**meta})
            for i,m in enumerate(result["mentions"]): rows.append({"mention_id":stable_id(unit.unit_id,i,m["aspect"],m["evidence"]),"unit_id":unit.unit_id,"document_id":unit.document_id,"source":unit.source,"event_year":unit.event_year,"detected_language":unit.detected_language,"url":unit.url,**m})
        output=pd.DataFrame(rows)
        if len(output): self._merge_incremental("aspect_mentions",output,"mention_id")
        write_table(pd.DataFrame(logs),self.settings.output_dir/"absa_processing_log.parquet"); return output

    def _approved_aspects(self):
        path=self.settings.output_dir/"taxonomy.csv"
        if not path.exists(): return ASPECTS
        t=pd.read_csv(path,keep_default_na=False); a=t[t.approved.astype(str).str.lower().isin(["true","1","yes"])]; values=a.approved_aspect.astype(str).str.strip().replace("",pd.NA).dropna().unique().tolist()
        return values+(["other_emerging"] if "other_emerging" not in values else []) if values else ASPECTS

    def _merge_incremental(self,name,fresh,key):
        path=self.settings.output_dir/f"{name}.parquet"
        if path.exists(): old=read_table(path); fresh=pd.concat([old[~old[key].isin(fresh[key])],fresh],ignore_index=True)
        write_table(fresh,path)

    def _manifest(self,stage,values):
        payload={"timestamp_utc":datetime.now(timezone.utc).isoformat(),"stage":stage,**values}
        with (self.settings.output_dir/"run_manifest.jsonl").open("a",encoding="utf-8") as h: h.write(json.dumps(payload,ensure_ascii=False)+"\n")












