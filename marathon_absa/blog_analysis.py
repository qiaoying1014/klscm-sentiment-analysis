from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .absa_v1 import (
    DEFAULT_MODEL, SCHEMA_VERSION, SENTIMENTS, canonicalize_evidence,
    load_ontology, mention_schema,
)
from .absa_aspect_ownership_development import INSTRUCTIONS, PROMPT_VERSION
from .absa_production import EXPECTED_PROMPT_SHA256

ROOT = Path("data/processed/blog_analysis_v1")
RAW = Path("Online Review Blog/raw-data.json")
DOCUMENTS = Path("data/processed/documents.csv")
UNITS = Path("data/processed/units.csv")
ONTOLOGY = Path("data/processed/absa_v1/absa_aspect_ontology_v1.json")
THEMES = Path("data/processed/absa_v1/aspect_level_themes_review_v1")
STAGE = "absa_v1_blog_mentions_v1"
REQUESTS = ROOT / "batch/blog_absa_requests_v1.jsonl"
REQUEST_MANIFEST = ROOT / "batch/blog_absa_request_manifest_v1.json"
RAW_OUTPUT = ROOT / "batch/blog_absa_raw_output_v1.jsonl"
SUBMISSION = ROOT / "batch/blog_absa_submission_v1.json"
STATUS = ROOT / "batch/blog_absa_status_v1.json"
MENTIONS = ROOT / "blog_absa_mentions_v1.csv"
CHUNK_RESULTS = ROOT / "blog_absa_chunk_results_v1.csv"
PROCESSING_LOG = ROOT / "blog_absa_processing_log_v1.csv"
GROUNDING_AUDIT = ROOT / "blog_absa_grounding_audit_v1.csv"
MAPPING = ROOT / "blog_theme_mapping_candidates_v1.csv"
DECISIONS = {"MATCH_EXISTING", "EMERGENT_BLOG_THEME", "UNCLEAR", "EXCLUDE_FROM_THEME_COMPARISON"}


def verify_frozen_upstream(root: Path = ROOT) -> dict[str, Any]:
    topic_manifest_path = Path("data/processed/topic_discovery_v1/topic_discovery_corpus_manifest_v1.json")
    taxonomy_manifest_path = Path("data/processed/topic_discovery_v1/final_taxonomy_v1/final_topic_taxonomy_manifest_v1.json")
    production_root = Path("data/processed/absa_v1/production/absa_v1_production_v1")
    review_manifest_path = THEMES / "review_manifest.json"
    topic = json.loads(topic_manifest_path.read_text(encoding="utf-8")); taxonomy = json.loads(taxonomy_manifest_path.read_text(encoding="utf-8")); review = json.loads(review_manifest_path.read_text(encoding="utf-8"))
    docs_path = production_root / "absa_v1_production_document_results_v1.csv"; mentions_path = production_root / "absa_v1_production_mentions_v1.csv"
    docs = pd.read_csv(docs_path, usecols=["document_id"]); mentions = pd.read_csv(mentions_path, usecols=["document_id", "aspect"])
    checks = {"topic_population": topic["total_original_records"] == 13799, "fitting_eligible": topic["unique_eligible_semantic_texts"] == 13743,
              "selected_c1": taxonomy["selected_model"] == "c1", "final_topics": taxonomy["final_consolidated_substantive_topics"] == 32,
              "absa_documents": len(docs) == 7704, "absa_mentions": len(mentions) == 15486,
              "mention_bearing_documents": mentions.document_id.nunique() == 5316, "zero_mention_documents": len(docs) - mentions.document_id.nunique() == 2388,
              "aspect_families": mentions.aspect.nunique() == 20, "alta_source_clusters": review["source_clusters"] == 101,
              "reviewed_themes": review["final_reviewed_theme_count"] == 64, "theme_bearing_aspects": len(review["aspect_reviewed_theme_counts"]) == 16,
              "insufficient_support_aspects": 20 - len(review["aspect_reviewed_theme_counts"]) == 4,
              "production_documents_hash": _sha(docs_path) == review["absa_source_identity"][str(docs_path).replace("/", "\\")],
              "production_mentions_hash": _sha(mentions_path) == review["absa_source_identity"][str(mentions_path).replace("/", "\\")],
              "reviewed_taxonomy_hash": _sha(THEMES / "reviewed_theme_taxonomy.csv") == review["output_hashes"]["reviewed_theme_taxonomy.csv"]}
    if not all(checks.values()): raise RuntimeError(f"Frozen upstream integrity failure: {checks}")
    result = {"stage": "blog_frozen_upstream_integrity_v1", "verified_at_utc": datetime.now(timezone.utc).isoformat(), "checks": checks, "all_passed": True, "api_calls": 0}
    root.mkdir(parents=True, exist_ok=True); (root / "blog_frozen_upstream_integrity_v1.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_table(frame: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(stem.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    frame.to_parquet(stem.with_suffix(".parquet"), index=False)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.DataFrame(json.loads(RAW.read_text(encoding="utf-8-sig")))
    documents = pd.read_csv(DOCUMENTS, low_memory=False)
    units = pd.read_csv(UNITS, low_memory=False)
    return raw, documents[documents.source.eq("blog")].copy(), units[units.source.eq("blog")].copy()


def included_reviews() -> tuple[pd.DataFrame, pd.DataFrame]:
    _, docs, units = _inputs()
    docs["document_id"] = docs.document_id.astype(str)
    usable = docs[docs.original_text.fillna("").str.strip().ne("") & docs.processing_status.eq("ready") & docs.duplicate_of.isna()].copy()
    chunks = units[units.document_id.astype(str).isin(set(usable.document_id))].copy()
    chunks["document_id"] = chunks.document_id.astype(str)
    chunks["unit_id"] = chunks.unit_id.astype(str)
    if usable.document_id.duplicated().any() or chunks.unit_id.duplicated().any():
        raise RuntimeError("Blog parent/chunk identifiers are not unique")
    if set(chunks.document_id) != set(usable.document_id):
        raise RuntimeError("Every included review must have at least one inference chunk")
    return usable, chunks


def audit_blog_source(root: Path = ROOT) -> dict[str, Any]:
    raw, docs, all_chunks = _inputs()
    reviews, chunks = included_reviews()
    lineage = docs[["document_id", "text_hash", "duplicate_of", "processing_status"]].copy()
    lineage["raw_row_number"] = range(1, len(lineage) + 1)
    lineage["usable"] = lineage.document_id.astype(str).isin(set(reviews.document_id))
    lineage["exclusion_reason"] = ""
    lineage.loc[docs.original_text.fillna("").str.strip().eq(""), "exclusion_reason"] = "empty_review"
    lineage.loc[docs.processing_status.eq("duplicate") | docs.duplicate_of.notna(), "exclusion_reason"] = "exact_duplicate"
    counts = chunks.groupby("document_id").size().rename("chunk_count")
    review_audit = reviews[["document_id", "event_year", "primary_language", "text_hash", "processing_status"]].merge(counts, on="document_id")
    chunk_audit = chunks[["document_id", "unit_id", "chunk_index", "sentence_start", "sentence_end", "token_count", "event_year", "primary_language"]].rename(columns={"unit_id": "chunk_id"})
    duplicate_count = int((docs.processing_status.eq("duplicate") | docs.duplicate_of.notna()).sum())
    summary = {
        "stage": "blog_analysis_audit_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_review_count": int(len(raw)), "usable_review_count": int(len(reviews)),
        "duplicate_review_count": duplicate_count, "chunk_count": int(len(chunks)),
        "reviews_with_multiple_chunks": int((counts > 1).sum()),
        "min_chunks_per_review": int(counts.min()), "max_chunks_per_review": int(counts.max()),
        "year_counts": {str(k): int(v) for k, v in reviews.event_year.value_counts(dropna=False).sort_index().items()},
        "language_counts": {str(k): int(v) for k, v in reviews.primary_language.value_counts(dropna=False).items()},
        "historical_documents_blog_rows": int(len(docs)), "historical_units_blog_rows": int(len(all_chunks)),
        "existing_document_ids_reused": True, "existing_chunk_ids_reused": True,
        "excluded_records": lineage[~lineage.usable].to_dict("records"),
        "source_hashes": {str(p): _sha(p) for p in [RAW, DOCUMENTS, UNITS]},
        "provenance": {key: "DOCUMENTATION_REQUIRED" for key in ["source_platforms", "search_strategy", "collection_procedure", "inclusion_criteria", "exclusion_criteria", "year_selection_rationale", "terms_ethics_considerations"]},
        "statistical_unit": "parent blog review", "inference_unit": "chunk", "api_calls": 0,
    }
    root.mkdir(parents=True, exist_ok=True)
    review_audit.to_csv(root / "blog_source_audit_v1.csv", index=False, encoding="utf-8-sig")
    chunk_audit.to_csv(root / "blog_chunk_audit_v1.csv", index=False, encoding="utf-8-sig")
    (root / "blog_analysis_audit_v1.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {**summary, "artifacts": {"source_audit": _sha(root / "blog_source_audit_v1.csv"), "chunk_audit": _sha(root / "blog_chunk_audit_v1.csv")}}
    (root / "blog_analysis_manifest_v1.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _verify_frozen() -> tuple[list[str], str]:
    prompt_hash = hashlib.sha256(INSTRUCTIONS.encode("utf-8")).hexdigest()
    if PROMPT_VERSION != "absa_v1_instructions_3_aspect_ownership" or prompt_hash != EXPECTED_PROMPT_SHA256:
        raise RuntimeError("Frozen V3 prompt identity/hash mismatch")
    aspects = [x["id"] for x in load_ontology()["aspects"]]
    if len(aspects) != 20:
        raise RuntimeError("Frozen ontology must contain 20 aspects")
    return aspects, prompt_hash


def create_blog_absa_requests(root: Path = ROOT) -> dict[str, Any]:
    audit = audit_blog_source(root)
    aspects, prompt_hash = _verify_frozen()
    reviews, chunks = included_reviews()
    review_map = reviews.set_index("document_id")
    schema = mention_schema(aspects)
    lines, input_tokens, output_tokens = [], 0, len(chunks) * 260
    for row in chunks.sort_values(["document_id", "chunk_index"]).itertuples(index=False):
        payload = {"document_id": row.unit_id, "parent_review_id": row.document_id, "original_text": row.text,
                   "document_type": "long_form_review_inference_chunk", "ontology": aspects}
        body = {"model": DEFAULT_MODEL, "instructions": INSTRUCTIONS, "input": json.dumps(payload, ensure_ascii=False),
                "text": {"format": {"type": "json_schema", "name": SCHEMA_VERSION, "strict": True, "schema": schema}}}
        input_tokens += math.ceil(len(json.dumps(body, ensure_ascii=False)) / 4)
        lines.append(json.dumps({"custom_id": f"absa1blog-{row.unit_id}", "method": "POST", "url": "/v1/responses", "body": body}, ensure_ascii=True))
    REQUESTS.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + "\n"
    if REQUESTS.exists() and REQUESTS.read_text(encoding="utf-8") != content:
        raise FileExistsError("Existing blog request package differs; refusing overwrite")
    REQUESTS.write_text(content, encoding="utf-8")
    synchronous = (input_tokens * 1.25 + output_tokens * 10) / 1_000_000
    manifest = {"stage": STAGE, "selected_prompt_identity": PROMPT_VERSION, "prompt_sha256": prompt_hash,
                "model": DEFAULT_MODEL, "schema": SCHEMA_VERSION, "ontology_sha256": _sha(ONTOLOGY),
                "review_count": len(reviews), "request_count": len(lines), "chunk_count": len(chunks),
                "request_sha256": _sha(REQUESTS), "source_hashes": audit["source_hashes"],
                "estimated_input_tokens": input_tokens, "estimated_output_tokens": output_tokens,
                "estimated_batch_cost_usd": round(synchronous * .5, 4), "submitted": False, "api_calls": 0}
    REQUEST_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def submit_blog_absa(root: Path = ROOT) -> dict[str, Any]:
    from openai import OpenAI
    manifest = json.loads(REQUEST_MANIFEST.read_text(encoding="utf-8"))
    if manifest["request_sha256"] != _sha(REQUESTS):
        raise RuntimeError("Blog request package hash mismatch")
    client = OpenAI()
    with REQUESTS.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(input_file_id=uploaded.id, endpoint="/v1/responses", completion_window="24h", metadata={"stage": STAGE})
    record = {"batch_id": batch.id, "input_file_id": uploaded.id, "status": batch.status, "api_calls": 2}
    SUBMISSION.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def blog_absa_status(root: Path = ROOT) -> dict[str, Any]:
    from openai import OpenAI
    sub = json.loads(SUBMISSION.read_text(encoding="utf-8"))
    result = OpenAI().batches.retrieve(sub["batch_id"]).model_dump()
    STATUS.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def import_blog_absa(root: Path = ROOT) -> dict[str, Any]:
    from openai import OpenAI
    status = blog_absa_status(root)
    if status.get("status") != "completed" or not status.get("output_file_id"):
        raise RuntimeError(f"Blog Batch is not complete: {status.get('status')}")
    content = OpenAI().files.content(status["output_file_id"]).content
    RAW_OUTPUT.write_bytes(content)
    return {"raw_output": str(RAW_OUTPUT), "sha256": _sha(RAW_OUTPUT), "api_calls": 2}


def _response_payload(item: dict[str, Any]) -> dict[str, Any]:
    response = item.get("response", {}).get("body", {})
    if response.get("status") == "failed" or item.get("error"):
        raise ValueError(str(item.get("error") or response.get("error")))
    text = response.get("output_text")
    if not text:
        for output in response.get("output", []):
            for content in output.get("content", []):
                if content.get("type") == "output_text": text = content.get("text")
    return json.loads(text)


def _unique_literal_span(text: str, value: str) -> tuple[int, int] | None:
    """Return a unique exact or case-insensitive source span."""
    starts = [match.start() for match in re.finditer(re.escape(value), text)]
    if len(starts) == 1:
        return starts[0], starts[0] + len(value)
    starts = [match.start() for match in re.finditer(re.escape(value), text, re.IGNORECASE)]
    if len(starts) == 1:
        start = starts[0]
        return start, start + len(value)
    return None


def _anchored_enclosing_span(text: str, evidence: str) -> tuple[int, int] | None:
    """Recover one exact enclosing span from strong, unique ordered anchors.

    This handles model evidence that joins source clauses with an ellipsis or
    accidentally omits an intervening sentence. It never synthesizes wording:
    the returned value is always sliced directly from the immutable chunk.
    """
    pieces = [piece.strip() for piece in re.split(r"\s*(?:\.\.\.|…)+\s*", evidence) if piece.strip()]
    if len(pieces) >= 2:
        spans = [_unique_literal_span(text, piece) for piece in pieces]
        if all(spans) and all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1)):
            return spans[0][0], spans[-1][1]

    boundaries = [match.end() for match in re.finditer(r"\S+(?:\s+|$)", evidence)]
    prefix = None
    for end in reversed(boundaries):
        candidate = evidence[:end].rstrip()
        if len(candidate) < 32:
            break
        prefix = _unique_literal_span(text, candidate)
        if prefix:
            break
    suffix = None
    starts = [match.start() for match in re.finditer(r"(?:^|\s+)\S+", evidence)]
    for start in starts:
        candidate = evidence[start:].lstrip()
        if len(candidate) < 32:
            continue
        suffix = _unique_literal_span(text, candidate)
        if suffix:
            break
    if prefix and suffix and prefix[1] <= suffix[0]:
        anchored = prefix[1] - prefix[0] + suffix[1] - suffix[0]
        if anchored >= 64 and anchored >= len(evidence) / 2:
            return prefix[0], suffix[1]
    return None


def ground_blog_evidence(text: str, evidence: str) -> dict[str, Any]:
    """Convert model evidence to an auditable exact source span or fail closed."""
    result = canonicalize_evidence(text, evidence)
    if result["normalized_match_status"] != "unrecoverable":
        return result
    literal_span = _unique_literal_span(text, str(evidence or ""))
    if literal_span:
        start, end = literal_span
        return {
            "raw_exact_match": False, "normalized_match_status": "recoverable",
            "evidence_repaired": True, "evidence_repair_method": "unique_case_insensitive_match",
            "original_model_evidence_text": str(evidence or ""),
            "final_exact_evidence_text": text[start:end], "final_evidence_start": start,
            "final_evidence_end": end, "likely_mismatch_category": "case_difference",
        }
    span = _anchored_enclosing_span(text, str(evidence or ""))
    if not span:
        return result
    start, end = span
    return {
        "raw_exact_match": False,
        "normalized_match_status": "recoverable",
        "evidence_repaired": True,
        "evidence_repair_method": "unique_ordered_anchor_enclosing_span",
        "original_model_evidence_text": str(evidence or ""),
        "final_exact_evidence_text": text[start:end],
        "final_evidence_start": start,
        "final_evidence_end": end,
        "likely_mismatch_category": "discontinuous_model_evidence",
    }


def finalize_blog_absa(root: Path = ROOT) -> dict[str, Any]:
    aspects, prompt_hash = _verify_frozen()
    reviews, chunks = included_reviews()
    expected = set(chunks.unit_id.astype(str)); by_chunk = chunks.set_index(chunks.unit_id.astype(str))
    seen, mention_ids, mentions, chunk_rows, logs, grounding_rows = set(), set(), [], [], [], []
    for line_no, line in enumerate(RAW_OUTPUT.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip(): continue
        item = json.loads(line); chunk_id = str(item.get("custom_id", "")).removeprefix("absa1blog-")
        if chunk_id not in expected: raise RuntimeError(f"Unknown chunk ID: {chunk_id}")
        if chunk_id in seen: raise RuntimeError(f"Duplicate chunk response: {chunk_id}")
        seen.add(chunk_id); row = by_chunk.loc[chunk_id]; text = str(row.text)
        try:
            payload = _response_payload(item); values = payload.get("mentions")
            if not isinstance(values, list): raise ValueError("mentions is not a list")
            for index, mention in enumerate(values):
                aspect, sentiment = mention.get("aspect"), mention.get("sentiment")
                evidence = str(mention.get("evidence_text", ""))
                if aspect not in aspects or sentiment not in SENTIMENTS: raise ValueError("invalid controlled label")
                grounding = ground_blog_evidence(text, evidence)
                if grounding["normalized_match_status"] == "unrecoverable":
                    raise ValueError("evidence cannot be recovered as an unambiguous exact chunk span")
                evidence = grounding["final_exact_evidence_text"]
                mention = {**mention, "evidence_text": evidence,
                           "original_model_evidence_text": grounding["original_model_evidence_text"],
                           "raw_exact_match": grounding["raw_exact_match"],
                           "evidence_repaired": grounding["evidence_repaired"],
                           "evidence_repair_method": grounding["evidence_repair_method"],
                           "evidence_start": grounding["final_evidence_start"],
                           "evidence_end": grounding["final_evidence_end"]}
                grounding_rows.append({"chunk_id": chunk_id, "mention_index": index, "aspect": aspect,
                    "status": grounding["normalized_match_status"], "raw_exact_match": grounding["raw_exact_match"],
                    "evidence_repaired": grounding["evidence_repaired"], "repair_method": grounding["evidence_repair_method"],
                    "original_model_evidence_text": grounding["original_model_evidence_text"],
                    "final_exact_evidence_text": evidence, "evidence_start": grounding["final_evidence_start"],
                    "evidence_end": grounding["final_evidence_end"]})
                mention_id = "absa1blog_" + hashlib.sha256(f"{chunk_id}|{index}|{aspect}|{evidence}".encode()).hexdigest()[:16]
                if mention_id in mention_ids: raise ValueError("duplicate mention ID")
                mention_ids.add(mention_id); mentions.append({"review_id": row.document_id, "chunk_id": chunk_id,
                    "blog_mention_id": mention_id, "mention_index": index, **mention,
                    "event_year": row.event_year, "primary_language": row.primary_language,
                    "prompt_version": PROMPT_VERSION, "model": DEFAULT_MODEL, "schema_version": SCHEMA_VERSION})
            chunk_rows.append({"review_id": row.document_id, "chunk_id": chunk_id, "chunk_index": row.chunk_index,
                               "processing_status": "success", "mention_count": len(values)})
            logs.append({"chunk_id": chunk_id, "line_number": line_no, "status": "parsed", "error": "",
                         "repaired_mentions": sum(bool(x["evidence_repaired"]) for x in grounding_rows if x["chunk_id"] == chunk_id)})
        except Exception as exc:
            logs.append({"chunk_id": chunk_id, "line_number": line_no, "status": "failed", "error": str(exc), "repaired_mentions": 0})
    missing = expected - seen
    for chunk_id in sorted(missing): logs.append({"chunk_id": chunk_id, "line_number": "", "status": "missing", "error": "no response"})
    pd.DataFrame(logs).to_csv(PROCESSING_LOG, index=False, encoding="utf-8-sig")
    if missing or any(x["status"] == "failed" for x in logs):
        raise RuntimeError(f"Fail-closed blog finalization: missing={len(missing)}, failed={sum(x['status']=='failed' for x in logs)}")
    pd.DataFrame(grounding_rows).to_csv(GROUNDING_AUDIT, index=False, encoding="utf-8-sig")
    mention_columns = ["review_id", "chunk_id", "blog_mention_id", "mention_index", "aspect", "target", "sentiment", "confidence", "evidence_text", "original_model_evidence_text", "raw_exact_match", "evidence_repaired", "evidence_repair_method", "evidence_start", "evidence_end", "expression_type", "language", "english_gloss", "contributing_hashtags", "contributing_emoji", "emerging_aspect_name", "analysis_notes", "event_year", "primary_language", "prompt_version", "model", "schema_version"]
    mention_frame = pd.DataFrame(mentions).reindex(columns=mention_columns)
    chunk_frame = pd.DataFrame(chunk_rows)
    _write_table(mention_frame, MENTIONS.with_suffix("")); _write_table(chunk_frame, CHUNK_RESULTS.with_suffix(""))
    result = aggregate_blog_absa(mention_frame, reviews, chunks, root)
    manifest = {"stage": STAGE, "finalized_at_utc": datetime.now(timezone.utc).isoformat(), "prompt_sha256": prompt_hash,
                "expected_chunks": len(expected), "successful_chunks": len(chunk_frame), "zero_mention_chunks": int((chunk_frame.mention_count == 0).sum()),
                "mention_count": len(mention_frame), "review_count": len(reviews),
                "raw_exact_evidence_mentions": int(mention_frame.raw_exact_match.sum()),
                "repaired_evidence_mentions": int(mention_frame.evidence_repaired.sum()), "api_calls": 0,
                "hashes": {p.name: _sha(p) for p in [MENTIONS, CHUNK_RESULTS, PROCESSING_LOG, GROUNDING_AUDIT]}}
    (root / "blog_absa_manifest_v1.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {**manifest, **result}


def _review_sentiment(values: pd.Series) -> str:
    substantive = set(values.dropna().astype(str))
    return next(iter(substantive)) if len(substantive) == 1 else "mixed"


def aggregate_blog_absa(mentions: pd.DataFrame, reviews: pd.DataFrame, chunks: pd.DataFrame, root: Path = ROOT) -> dict[str, Any]:
    total = len(reviews)
    summary = reviews[["document_id", "event_year", "primary_language"]].rename(columns={"document_id": "review_id"}).copy()
    chunk_counts = chunks.groupby("document_id").size(); mention_counts = mentions.groupby("review_id").size() if len(mentions) else pd.Series(dtype=int)
    summary["chunk_count"] = summary.review_id.map(chunk_counts).fillna(0).astype(int); summary["mention_count"] = summary.review_id.map(mention_counts).fillna(0).astype(int)
    if len(mentions):
        presence = mentions.groupby(["review_id", "aspect"], as_index=False).agg(review_aspect_sentiment=("sentiment", _review_sentiment), support_mentions=("blog_mention_id", "nunique"))
        presence = presence.merge(summary[["review_id", "event_year"]], on="review_id", how="left")
        aspect = presence.groupby("aspect", as_index=False).agg(support_reviews=("review_id", "nunique"), support_mentions=("support_mentions", "sum"))
        aspect["total_included_reviews"] = total; aspect["review_prevalence"] = aspect.support_reviews / total; aspect["denominator_unit"] = "included_parent_reviews"
        sentiment = presence.groupby(["aspect", "review_aspect_sentiment"], as_index=False).agg(support_reviews=("review_id", "nunique")); sentiment["total_included_reviews"] = total
        years = presence.groupby(["aspect", "event_year"], as_index=False).agg(support_reviews=("review_id", "nunique")); year_den = summary.groupby("event_year").size(); years["total_reviews_in_year"] = years.event_year.map(year_den); years["review_prevalence"] = years.support_reviews / years.total_reviews_in_year
    else:
        presence = pd.DataFrame(columns=["review_id", "aspect", "review_aspect_sentiment", "support_mentions", "event_year"])
        aspect = pd.DataFrame(columns=["aspect", "support_reviews", "support_mentions", "total_included_reviews", "review_prevalence", "denominator_unit"])
        sentiment = pd.DataFrame(columns=["aspect", "review_aspect_sentiment", "support_reviews", "total_included_reviews"])
        years = pd.DataFrame(columns=["aspect", "event_year", "support_reviews", "total_reviews_in_year", "review_prevalence"])
    for frame, name in [(summary, "blog_review_summary_v1"), (presence, "blog_review_aspect_presence_v1"), (aspect, "blog_aspect_summary_v1"), (sentiment, "blog_aspect_sentiment_summary_v1"), (years, "blog_aspect_year_summary_v1")]: _write_table(frame, root / name)
    return {"included_reviews": total, "inference_chunks": len(chunks), "mentions": len(mentions), "aspect_rows": len(aspect)}


def generate_theme_candidates(root: Path = ROOT, top_k: int = 3) -> dict[str, Any]:
    mentions = pd.read_csv(MENTIONS); taxonomy = pd.read_csv(THEMES / "reviewed_theme_taxonomy.csv")
    if len(taxonomy) != 64 or not taxonomy.review_status.eq("reviewed").all(): raise RuntimeError("Final reviewed taxonomy is not the frozen 64-theme taxonomy")
    evidence = pd.read_csv(THEMES / "reviewed_theme_evidence.csv")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", local_files_only=True)
    def semantic(row: pd.Series) -> str: return " | ".join(str(row.get(x, "")) for x in ["target", "evidence_text", "english_gloss"] if pd.notna(row.get(x)) and str(row.get(x)).strip())
    rows = []
    for _, mention in mentions.iterrows():
        candidates = taxonomy[taxonomy.aspect.eq(mention.aspect)]
        representations = []
        for theme in candidates.itertuples(index=False):
            members = evidence[evidence.reviewed_theme_id.eq(theme.reviewed_theme_id)]
            representations.append(" || ".join(members.apply(semantic, axis=1).tolist()))
        base = mention.to_dict(); base.update({"mapping_decision": "", "selected_reviewed_theme_id": "", "emergent_blog_theme_label": "", "researcher_note": "", "review_status": "pending"})
        if representations:
            import numpy as np
            vectors = model.encode([semantic(mention)] + representations, normalize_embeddings=True)
            order = np.argsort(vectors[1:] @ vectors[0])[::-1][:top_k]
            for rank, idx in enumerate(order, 1):
                theme = candidates.iloc[int(idx)]; base[f"candidate_theme_id_{rank}"] = theme.reviewed_theme_id; base[f"candidate_theme_label_{rank}"] = theme.reviewed_theme_label; base[f"candidate_similarity_{rank}"] = float(vectors[int(idx)+1] @ vectors[0])
        rows.append(base)
    result = pd.DataFrame(rows)
    result.to_csv(MAPPING, index=False, encoding="utf-8-sig")
    return {"mapping_rows": len(result), "review_status": "pending", "automatic_assignments": 0, "api_calls": 0}


def finalize_theme_mappings(mapping_path: Path = MAPPING, root: Path = ROOT) -> dict[str, Any]:
    mapping = pd.read_csv(mapping_path, keep_default_na=False); taxonomy = pd.read_csv(THEMES / "reviewed_theme_taxonomy.csv")
    if not mapping.review_status.eq("reviewed").all() or mapping.mapping_decision.isin(["", "UNCLEAR"]).any(): raise RuntimeError("Unresolved mapping rows remain")
    if not set(mapping.mapping_decision).issubset(DECISIONS): raise ValueError("Invalid mapping decision")
    lookup = taxonomy.set_index("reviewed_theme_id").aspect.to_dict()
    matched = mapping[mapping.mapping_decision.eq("MATCH_EXISTING")].copy()
    if matched.selected_reviewed_theme_id.eq("").any() or any(lookup.get(r.selected_reviewed_theme_id) != r.aspect for r in matched.itertuples()): raise ValueError("MATCH_EXISTING must select a same-aspect reviewed theme")
    emergent = mapping[mapping.mapping_decision.eq("EMERGENT_BLOG_THEME")].copy()
    if emergent.emergent_blog_theme_label.eq("").any(): raise ValueError("Emergent mappings require a label")
    total = pd.read_csv(root / "blog_review_summary_v1.csv").review_id.nunique()
    matched["theme_id"] = matched.selected_reviewed_theme_id; emergent["theme_id"] = "blog_emergent__" + emergent.aspect + "__" + emergent.emergent_blog_theme_label.map(lambda x: hashlib.sha256(x.encode()).hexdigest()[:8])
    assignments = pd.concat([matched, emergent], ignore_index=True); assignments["theme_origin"] = assignments.mapping_decision.map({"MATCH_EXISTING": "instagram_reviewed_taxonomy", "EMERGENT_BLOG_THEME": "blog_emergent"})
    presence = assignments.drop_duplicates(["review_id", "theme_id"])
    theme_summary = presence.groupby(["theme_id", "aspect", "theme_origin"], as_index=False).agg(support_reviews=("review_id", "nunique")); theme_summary["total_included_reviews"] = total; theme_summary["review_prevalence"] = theme_summary.support_reviews / total
    mentions = assignments.groupby("theme_id").size().rename("support_mentions"); theme_summary = theme_summary.merge(mentions, on="theme_id")
    sent = presence.groupby(["theme_id", "sentiment"], as_index=False).agg(support_reviews=("review_id", "nunique")); years = presence.groupby(["theme_id", "event_year"], as_index=False).agg(support_reviews=("review_id", "nunique"))
    for frame, name in [(assignments, "blog_reviewed_theme_assignments_v1"), (theme_summary, "blog_reviewed_theme_summary_v1"), (sent, "blog_reviewed_theme_sentiment_summary_v1"), (years, "blog_reviewed_theme_year_summary_v1"), (assignments[["review_id", "chunk_id", "blog_mention_id", "theme_id", "evidence_text", "english_gloss"]], "blog_reviewed_theme_evidence_v1"), (emergent, "blog_emergent_themes_v1")]: _write_table(frame, root / name)
    manifest = {"stage": "blog_theme_v1", "lifecycle_status": "researcher_mapping_finalized", "assignment_mentions": len(assignments), "unique_review_theme_assignments": len(presence), "api_calls": 0, "frozen_taxonomy_modified": False}
    (root / "blog_theme_manifest_v1.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
