from __future__ import annotations

import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


VERSION = "absa_v1"
PROMPT_VERSION = "absa_v1_instructions_1"
SCHEMA_VERSION = "absa_mention_schema_v1"
CACHE_STAGE = "absa_v1_mentions"
DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_SEED = 20260819
DEFAULT_SAMPLE_SIZE = 80
AUDIT_VERSION = "absa_v1_single_researcher_audit_v1"
AUDIT_SEED = 20260823
ROOT = Path("data/processed/absa_v1")
AUDIT_ROOT = ROOT / AUDIT_VERSION
CORPUS = Path("data/processed/topic_discovery_v1/final_taxonomy_v1/final_substantive_topic_corpus_v1.csv")
ONTOLOGY_PATH = ROOT / "absa_aspect_ontology_v1.json"
HISTORICAL_SAMPLE_PATH = ROOT / "absa_validation_sample_v1.csv"
HISTORICAL_MANIFEST_PATH = ROOT / "absa_validation_manifest_v1.json"
SAMPLE_PATH = AUDIT_ROOT / "absa_v1_audit_sample_v1.csv"
MANIFEST_PATH = AUDIT_ROOT / "absa_v1_audit_manifest_v1.json"
ANNOTATIONS_PATH = AUDIT_ROOT / "absa_v1_audit_annotations_working_v1.csv"
GOLD_PATH = AUDIT_ROOT / "absa_v1_audit_gold_v1.csv"
PROGRESS_PATH = AUDIT_ROOT / "absa_v1_audit_progress_v1.csv"
DRAFT_ANNOTATIONS_PATH = AUDIT_ROOT / "absa_v1_ai_draft_annotations_v1.csv"
DRAFT_PROGRESS_PATH = AUDIT_ROOT / "absa_v1_ai_draft_progress_v1.csv"
DRAFT_MANIFEST_PATH = AUDIT_ROOT / "absa_v1_ai_draft_manifest_v1.json"

SENTIMENTS = ["positive", "negative", "neutral", "mixed"]
EXPRESSION_TYPES = ["explicit", "implicit"]

EVIDENCE_DIAGNOSTIC_VERSION = "absa_v1_evidence_diagnostic_v1"

ABSA_V1_INSTRUCTIONS = """Analyze the original multilingual KLSCM caption directly. Extract only evaluative
aspect mentions; factual non-evaluative references produce zero mentions. Every mention must quote exact,
non-empty evidence from the caption. Separate different evaluated aspects, even when their sentiments differ.
Use mixed only when the same aspect receives both positive and negative evaluation in the same local context.
Do not hallucinate a target or infer from background knowledge. The supplied final BERTopic topic is context,
not an aspect label and must not force extraction. Use emerging_other only when no controlled aspect fits and
give a concise emerging_aspect_name. Hashtags and emoji may contribute only when they materially support an
evaluation; preserve them exactly. Return the observed language and an auxiliary English gloss without
replacing original evidence. Mark implicit only where the text itself defensibly conveys evaluation.
Return JSON matching the strict schema and return {\"mentions\": []} when there is no evaluative content."""


def load_ontology(path: Path = ONTOLOGY_PATH) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_ontology(data)
    return data


def validate_ontology(data: dict[str, Any]) -> None:
    if data.get("version") != "absa_aspect_ontology_v1":
        raise ValueError("Unexpected ontology version")
    aspects = data.get("aspects", [])
    ids = [a.get("id") for a in aspects]
    if not 12 <= len(ids) <= 20 or len(ids) != len(set(ids)):
        raise ValueError("Ontology must contain 12-20 unique aspects")
    required = {"id", "name", "definition", "inclusion_examples", "exclusion_boundaries", "related_aspects", "topic_links"}
    for aspect in aspects:
        if not required.issubset(aspect) or not aspect["definition"] or not aspect["inclusion_examples"]:
            raise ValueError(f"Incomplete aspect: {aspect.get('id')}")
    if "emerging_other" not in ids:
        raise ValueError("emerging_other is required")


def mention_schema(aspect_ids: list[str]) -> dict[str, Any]:
    item = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "aspect": {"type": "string", "enum": aspect_ids},
            "target": {"type": "string"},
            "sentiment": {"type": "string", "enum": SENTIMENTS},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence_text": {"type": "string"},
            "expression_type": {"type": "string", "enum": EXPRESSION_TYPES},
            "language": {"type": "string"}, "english_gloss": {"type": "string"},
            "contributing_hashtags": {"type": "array", "items": {"type": "string"}},
            "contributing_emoji": {"type": "array", "items": {"type": "string"}},
            "emerging_aspect_name": {"type": "string"}, "analysis_notes": {"type": "string"},
        },
        "required": ["aspect", "target", "sentiment", "confidence", "evidence_text", "expression_type",
                     "language", "english_gloss", "contributing_hashtags", "contributing_emoji",
                     "emerging_aspect_name", "analysis_notes"],
    }
    return {"type": "object", "additionalProperties": False,
            "properties": {"mentions": {"type": "array", "items": item}}, "required": ["mentions"]}


def validate_mentions(result: dict[str, Any], caption: str, aspect_ids: list[str]) -> list[dict[str, Any]]:
    mentions = result.get("mentions")
    if not isinstance(mentions, list):
        raise ValueError("mentions must be an array")
    output = []
    for mention in mentions:
        if mention.get("aspect") not in aspect_ids or mention.get("sentiment") not in SENTIMENTS:
            raise ValueError("Invalid aspect or sentiment")
        evidence = mention.get("evidence_text", "")
        if not evidence or evidence not in caption:
            raise ValueError("Every evidence_text must be an exact caption substring")
        if mention.get("expression_type") not in EXPRESSION_TYPES:
            raise ValueError("Invalid expression_type")
        if mention["aspect"] == "emerging_other" and not mention.get("emerging_aspect_name", "").strip():
            raise ValueError("emerging_aspect_name is required for emerging_other")
        start = caption.find(evidence)
        output.append({**mention, "evidence_start": start, "evidence_end": start + len(evidence)})
    return output


def _normalize_evidence(text: str, mode: str) -> str:
    value = unicodedata.normalize("NFC", str(text))
    if mode == "punctuation":
        value = value.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                                              "–": "-", "—": "-", "…": "..."}))
        value = re.sub(r"[\"']", "", value)
    if mode in {"whitespace", "punctuation"}:
        value = re.sub(r"\s+", " ", value)
    if mode == "case":
        value = value.casefold()
    return value.strip()


def _unique_normalized_span(caption: str, evidence: str, mode: str) -> tuple[int, int] | None:
    needle = _normalize_evidence(evidence, mode)
    haystack = _normalize_evidence(caption, mode)
    if not needle:
        return None
    starts = [match.start() for match in re.finditer(re.escape(needle), haystack)]
    if len(starts) != 1:
        return None
    normalized_start, normalized_end = starts[0], starts[0] + len(needle)
    boundaries: dict[int, list[int]] = {}
    for index in range(len(caption) + 1):
        boundaries.setdefault(len(_normalize_evidence(caption[:index], mode)), []).append(index)
    candidates = set()
    for start in boundaries.get(normalized_start, []):
        for end in boundaries.get(normalized_end, []):
            if start < end and _normalize_evidence(caption[start:end], mode) == needle:
                candidates.add((start, end))
    return next(iter(candidates)) if len(candidates) == 1 else None


def _likely_mismatch_category(caption: str, evidence: str) -> str:
    if _normalize_evidence(evidence, "whitespace") in _normalize_evidence(caption, "whitespace"):
        return "whitespace_difference"
    if unicodedata.normalize("NFC", evidence) in unicodedata.normalize("NFC", caption):
        return "unicode_normalization"
    if evidence.casefold() in caption.casefold():
        return "case_difference"
    if any(mark in evidence for mark in ["...", "…", " / "]):
        return "truncated_evidence"
    if any(ord(char) > 0xFFFF for char in evidence):
        return "emoji_alias_difference"
    evidence_tokens = set(re.findall(r"\w+", evidence.casefold()))
    caption_tokens = set(re.findall(r"\w+", caption.casefold()))
    overlap = len(evidence_tokens & caption_tokens) / len(evidence_tokens) if evidence_tokens else 0
    if overlap >= 0.35:
        return "translated_or_paraphrased_evidence"
    if evidence_tokens:
        return "hallucinated_evidence"
    return "other"


def canonicalize_evidence(caption: str, evidence: str, predicted_start: Any = None,
                          predicted_end: Any = None) -> dict[str, Any]:
    """Recover an exact original span only through deterministic, unique transformations."""
    evidence = str(evidence or "")
    exact_start = caption.find(evidence) if evidence else -1
    if exact_start >= 0:
        return {"raw_exact_match": True, "normalized_match_status": "exact_as_returned",
                "evidence_repaired": False, "evidence_repair_method": "none",
                "original_model_evidence_text": evidence, "final_exact_evidence_text": evidence,
                "final_evidence_start": exact_start, "final_evidence_end": exact_start + len(evidence),
                "likely_mismatch_category": "exact"}
    try:
        start, end = int(predicted_start), int(predicted_end)
    except (TypeError, ValueError):
        start = end = -1
    if 0 <= start < end <= len(caption):
        span = caption[start:end]
        plausible = any(_normalize_evidence(span, mode) == _normalize_evidence(evidence, mode)
                        for mode in ["unicode", "whitespace", "punctuation", "case"])
        if plausible:
            return {"raw_exact_match": False, "normalized_match_status": "recoverable",
                    "evidence_repaired": True, "evidence_repair_method": "model_offsets",
                    "original_model_evidence_text": evidence, "final_exact_evidence_text": span,
                    "final_evidence_start": start, "final_evidence_end": end,
                    "likely_mismatch_category": _likely_mismatch_category(caption, evidence)}
    for mode, method, category in [
        ("unicode", "unique_unicode_nfc_match", "unicode_normalization"),
        ("whitespace", "unique_whitespace_normalized_match", "whitespace_difference"),
        ("punctuation", "unique_trivial_punctuation_match", "punctuation_difference"),
        ("case", "unique_case_insensitive_match", "case_difference"),
    ]:
        match = _unique_normalized_span(caption, evidence, mode)
        if match:
            start, end = match; span = caption[start:end]
            return {"raw_exact_match": False, "normalized_match_status": "recoverable",
                    "evidence_repaired": True, "evidence_repair_method": method,
                    "original_model_evidence_text": evidence, "final_exact_evidence_text": span,
                    "final_evidence_start": start, "final_evidence_end": end,
                    "likely_mismatch_category": category}
    return {"raw_exact_match": False, "normalized_match_status": "unrecoverable",
            "evidence_repaired": False, "evidence_repair_method": "none",
            "original_model_evidence_text": evidence, "final_exact_evidence_text": "",
            "final_evidence_start": -1, "final_evidence_end": -1,
            "likely_mismatch_category": _likely_mismatch_category(caption, evidence)}


def stable_mention_id(document_id: str, index: int, aspect: str, evidence: str) -> str:
    digest = hashlib.sha256(f"{VERSION}|{document_id}|{index}|{aspect}|{evidence}".encode()).hexdigest()[:16]
    return f"absa1_{digest}"


def _cue_score(row: pd.Series) -> int:
    text = str(row.get("original_text", ""))
    lower = text.lower()
    cues = ["i ", "my ", "we ", "best", "love", "good", "great", "amazing", "bad", "worst",
            "sakit", "penat", "panas", "bangga", "kecewa", "queue", "worth", "pb", "dnf"]
    return int(len(text) >= 180) + int(any(c in lower for c in cues)) + int(bool(re.search(r"[😀-🙏❤🔥😭😡]", text)))


def _audit_cues(row: pd.Series) -> list[str]:
    text = str(row.get("original_text", "")); lower = text.lower()
    language = " ".join(str(row.get(k, "")) for k in ["primary_language", "detected_language", "language_status"]).lower()
    cues = []
    if any(x in language for x in ["malay", "chinese", "indones", "mixed", "uncertain"]): cues.append("multilingual_or_uncertain")
    if len(text) >= 180: cues.append("long_caption")
    if len(re.findall(r"[.!?;]|\b(?:but|tapi|tetapi|dan|and)\b", lower)) >= 2: cues.append("multiple_clauses")
    if re.search(r"[😀-🙏❤🔥😭😡]|#\w+", text): cues.append("emoji_or_hashtag")
    if re.search(r"\b(i|my|me|we|our|saya|aku|kami|kita|我|我的)\b", lower): cues.append("first_person")
    return cues


def create_single_researcher_audit(corpus_path: Path = CORPUS, output_dir: Path = AUDIT_ROOT,
                                    seed: int = AUDIT_SEED) -> pd.DataFrame:
    frame = pd.read_csv(corpus_path, low_memory=False)
    if len(frame) != 7704 or set(frame.final_taxonomy_action.astype(str)) != {"keep"} or frame.document_id.duplicated().any():
        raise ValueError("Audit requires 7,704 unique frozen substantive documents only")
    data = frame.copy()
    data["audit_cues"] = data.apply(lambda row: "|".join(_audit_cues(row)), axis=1)
    data["first_person_cue"] = data.audit_cues.str.contains("first_person")
    data["emoji_hashtag_cue"] = data.audit_cues.str.contains("emoji_or_hashtag")
    data["text_length_quartile"] = pd.qcut(data.text_length_chars.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    rng = random.Random(seed); chosen: list[int] = []; component: dict[int, str] = {}
    for topic_id in sorted(data.final_topic_id.unique(), key=str):
        index = rng.choice(data.index[data.final_topic_id == topic_id].tolist())
        chosen.append(index); component[index] = "topic_floor"
    pool = data.drop(index=chosen)
    difficult = pool[pool.audit_cues.ne("")].copy()
    difficult["cue_count"] = difficult.audit_cues.str.count(r"\|") + 1
    candidates = difficult.index.tolist(); weights = difficult.cue_count.astype(float).tolist()
    enriched = []
    while len(enriched) < 24:
        pick = rng.choices(candidates, weights=weights, k=1)[0]
        pos = candidates.index(pick); candidates.pop(pos); weights.pop(pos)
        enriched.append(pick); component[pick] = "difficult_content_enrichment"
    chosen.extend(enriched)
    remaining_pool = data.drop(index=chosen).index.tolist()
    random_indices = rng.sample(remaining_pool, 24)
    for index in random_indices: component[index] = "random_component"
    chosen.extend(random_indices)
    sample = data.loc[chosen].copy()
    sample["sample_component"] = [component[i] for i in sample.index]
    sample["sample_order"] = range(1, 81)
    public = ["sample_order", "sample_component", "audit_cues", "document_id", "original_text", "source",
              "event_year", "primary_language", "language_status", "original_topic_id", "final_topic_id",
              "final_topic_name", "final_topic_group", "text_length_chars", "text_length_quartile",
              "first_person_cue", "emoji_hashtag_cue"]
    sample = sample[public]
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / SAMPLE_PATH.name
    manifest_path = output_dir / MANIFEST_PATH.name
    if manifest_path.exists():
        raise FileExistsError("Audit sample package already exists and will not be overwritten")
    if sample_path.exists():
        existing = pd.read_csv(sample_path)
        if existing.document_id.astype(str).tolist() != sample.document_id.astype(str).tolist():
            raise FileExistsError("An incompatible partial audit sample exists; refusing overwrite")
    else:
        sample.to_csv(sample_path, index=False, encoding="utf-8-sig")
    coverage = {column: sample[column].astype(object).where(sample[column].notna(), "unknown").astype(str).value_counts().sort_index().to_dict()
                for column in ["final_topic_name", "event_year", "primary_language", "text_length_quartile", "source",
                               "first_person_cue", "emoji_hashtag_cue"]}
    manifest = {"version": AUDIT_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
        "sample_size": 80, "population_size": 7704, "composition": {"topic_floor": 32,
        "difficult_content_enrichment": 24, "random_component": 24}, "all_topics_represented": sample.final_topic_id.nunique() == 32,
        "unique_documents": sample.document_id.nunique() == 80, "predictions_used": False, "coverage": coverage,
        "document_ids_sha256": hashlib.sha256("\n".join(sample.document_id.astype(str)).encode()).hexdigest()}
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return sample


def create_validation_sample(corpus_path: Path = CORPUS, output_dir: Path = ROOT,
                             size: int = DEFAULT_SAMPLE_SIZE, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    frame = pd.read_csv(corpus_path, low_memory=False)
    if len(frame) != 7704 or set(frame["final_taxonomy_action"].astype(str)) != {"keep"}:
        raise ValueError("ABSA v1 requires the frozen 7,704-document substantive corpus only")
    if size > len(frame):
        raise ValueError("Sample exceeds corpus")
    data = frame.copy()
    data["sampling_cue_score"] = data.apply(_cue_score, axis=1)
    data["length_band"] = pd.qcut(data["text_length_chars"].rank(method="first"), 3,
                                   labels=["short", "medium", "long"])
    rng = random.Random(seed)
    topic_ids = sorted(data["final_topic_id"].unique(), key=str)
    chosen: list[int] = []
    # One random record per final topic preserves broad coverage.
    for topic_id in topic_ids:
        candidates = data.index[data["final_topic_id"] == topic_id].tolist()
        chosen.append(rng.choice(candidates))
    remaining = size - len(chosen)
    pool = data.drop(index=chosen)
    enriched_n = min(round(size * 0.30), remaining)
    enriched_pool = pool[pool["sampling_cue_score"] >= 1]
    weights = (1 + enriched_pool["sampling_cue_score"]).tolist()
    enriched: list[int] = []
    candidates = enriched_pool.index.tolist()
    while candidates and len(enriched) < enriched_n:
        pick = rng.choices(candidates, weights=weights, k=1)[0]
        pos = candidates.index(pick); enriched.append(pick); candidates.pop(pos); weights.pop(pos)
    chosen.extend(enriched)
    pool = data.drop(index=chosen)
    random_remaining = rng.sample(pool.index.tolist(), remaining - len(enriched))
    chosen.extend(random_remaining)
    sample = data.loc[chosen].copy()
    sample["sample_order"] = list(range(1, len(sample) + 1))
    sample["annotation_status"] = "not_started"
    sample["no_evaluative_aspect_mention"] = ""
    public = ["sample_order", "document_id", "original_text", "source", "event_year", "primary_language",
              "language_status", "original_topic_id", "final_topic_id", "final_topic_name", "final_topic_group",
              "annotation_status", "no_evaluative_aspect_mention"]
    sample = sample[public].sort_values("sample_order")
    output_dir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_dir / SAMPLE_PATH.name, index=False, encoding="utf-8-sig")
    manifest = {
        "version": "absa_validation_sample_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed, "sample_size": len(sample), "population_size": len(frame),
        "population": "final substantive topics only", "blinded_to_model_predictions": True,
        "strategy": {"topic_floor": len(topic_ids), "evaluative_cue_enrichment": len(enriched),
                     "random_component": len(random_remaining), "enrichment_share_target": 0.30,
                     "cues": "text length, first-person/lexical cues, and emoji; no ABSA predictions"},
        "document_ids_sha256": hashlib.sha256("\n".join(sample.document_id.astype(str)).encode()).hexdigest(),
    }
    (output_dir / MANIFEST_PATH.name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return sample


def annotation_columns() -> list[str]:
    return ["mention_id", "document_id", "aspect", "target", "sentiment", "evidence_text",
            "evidence_start", "evidence_end", "expression_type", "language", "english_gloss",
            "contributing_hashtags", "contributing_emoji", "emerging_aspect_name", "analysis_notes"]


def save_annotations(rows: list[dict[str, Any]], path: Path = ANNOTATIONS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = path.with_name(f"{path.stem}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}")
        backup.write_bytes(path.read_bytes())
    pd.DataFrame(rows, columns=annotation_columns()).to_csv(path, index=False, encoding="utf-8-sig")


def validate_draft_package(sample_path: Path = SAMPLE_PATH,
                           annotations_path: Path = DRAFT_ANNOTATIONS_PATH,
                           progress_path: Path = DRAFT_PROGRESS_PATH) -> dict[str, int]:
    """Validate a complete AI-assisted draft without treating it as human gold."""
    sample = pd.read_csv(sample_path)
    progress = pd.read_csv(progress_path)
    annotations = (pd.read_csv(annotations_path).fillna("") if annotations_path.exists()
                   else pd.DataFrame(columns=annotation_columns()))
    sample_ids = set(sample.document_id.astype(str))
    progress["document_id"] = progress.document_id.astype(str)
    annotations["document_id"] = annotations.document_id.astype(str)
    if set(progress.document_id) != sample_ids or progress.document_id.duplicated().any():
        raise ValueError("Draft progress must contain every sampled document exactly once")
    if not progress.annotation_status.eq("ai_draft_complete").all():
        raise ValueError("Every draft document must be marked ai_draft_complete")
    if not set(annotations.document_id).issubset(sample_ids) or annotations.mention_id.duplicated().any():
        raise ValueError("Draft contains unknown document IDs or duplicate mention IDs")
    aspects = [a["id"] for a in load_ontology()["aspects"]]
    captions = sample.assign(document_id=sample.document_id.astype(str)).set_index("document_id").original_text.astype(str).to_dict()
    for document_id, group in annotations.groupby("document_id", sort=False):
        if not group.aspect.isin(aspects).all() or not group.sentiment.isin(SENTIMENTS).all():
            raise ValueError("Draft contains an invalid aspect or sentiment")
        for index, row in enumerate(group.itertuples(index=False)):
            if row.evidence_text not in captions[document_id]:
                raise ValueError(f"Draft evidence is not exact for {document_id}")
            if row.mention_id != stable_mention_id(document_id, index, row.aspect, row.evidence_text):
                raise ValueError(f"Draft mention ID is misaligned for {document_id}")
    no_mention = progress.ai_draft_no_evaluative_aspect_mention.astype(str).str.lower().eq("true")
    expected_positive = set(progress.loc[~no_mention, "document_id"])
    if expected_positive != set(annotations.document_id):
        raise ValueError("Draft mention/progress states do not reconcile")
    return {"documents": len(sample), "mentions": len(annotations),
            "zero_mention_documents": int(no_mention.sum())}


def save_document_annotations(document_id: str, rows: list[dict[str, Any]],
                              annotations_path: Path = ANNOTATIONS_PATH,
                              progress_path: Path = PROGRESS_PATH) -> None:
    if progress_path.exists():
        progress = pd.read_csv(progress_path)
        if str(document_id) in set(progress.loc[progress.annotation_status.eq("complete"), "document_id"].astype(str)):
            raise ValueError("Completed annotations are immutable")
    existing = pd.read_csv(annotations_path).fillna("").to_dict("records") if annotations_path.exists() else []
    retained = [row for row in existing if str(row["document_id"]) != str(document_id)]
    save_annotations(retained + rows, annotations_path)


def freeze_gold(sample_path: Path = SAMPLE_PATH, gold_path: Path = GOLD_PATH) -> Path:
    if gold_path.exists():
        raise FileExistsError("Versioned audit gold already exists and will not be overwritten")
    sample = pd.read_csv(sample_path)
    progress = pd.read_csv(PROGRESS_PATH) if PROGRESS_PATH.exists() else pd.DataFrame()
    completed = set(progress.loc[progress.get("annotation_status", pd.Series(dtype=str)).eq("complete"), "document_id"].astype(str)) if not progress.empty else set()
    if completed != set(sample.document_id.astype(str)):
        raise ValueError("All sampled documents must be complete before gold can be frozen")
    annotations = pd.read_csv(ANNOTATIONS_PATH).fillna("") if ANNOTATIONS_PATH.exists() else pd.DataFrame(columns=annotation_columns())
    valid_ids = set(sample.document_id.astype(str)); annotations["document_id"] = annotations.document_id.astype(str)
    if not set(annotations.document_id).issubset(valid_ids) or annotations.mention_id.duplicated().any():
        raise ValueError("Unknown document IDs or duplicate mention IDs")
    aspects = [a["id"] for a in load_ontology()["aspects"]]
    captions = sample.set_index(sample.document_id.astype(str)).original_text.astype(str).to_dict()
    for document_id, group in annotations.groupby("document_id", sort=False):
        if not group.aspect.isin(aspects).all() or not group.sentiment.isin(SENTIMENTS).all():
            raise ValueError("Invalid aspect or sentiment")
        for index, row in enumerate(group.itertuples(index=False)):
            if row.evidence_text not in captions[document_id]: raise ValueError("Evidence is not in the original caption")
            if row.mention_id != stable_mention_id(document_id, index, row.aspect, row.evidence_text):
                raise ValueError("Mention ID does not align with document and mention content")
    positive_docs = set(progress.loc[progress.no_evaluative_aspect_mention.astype(str).str.lower().ne("true"), "document_id"].astype(str))
    if positive_docs - set(annotations.document_id): raise ValueError("Mention-required documents have no annotations")
    annotations.to_csv(gold_path, index=False, encoding="utf-8-sig")
    marker = AUDIT_ROOT / "absa_v1_audit_gold_v1.frozen.json"
    marker.write_text(json.dumps({"version": AUDIT_VERSION, "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                                  "sample_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
                                  "gold_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest()}, indent=2), encoding="utf-8")
    return marker


def cost_estimate(corpus_path: Path = CORPUS, sample_size: int = DEFAULT_SAMPLE_SIZE,
                  input_per_million: float = 1.25, output_per_million: float = 10.0,
                  batch_discount: float = 0.5) -> dict[str, Any]:
    data = pd.read_csv(corpus_path, usecols=["document_id", "original_text"], low_memory=False)
    ontology_chars = len(ABSA_V1_INSTRUCTIONS) + len(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    input_tokens = sum(math.ceil((len(str(x)) + ontology_chars) / 4) for x in data.original_text)
    expected_output = len(data) * 260
    p95_in, p95_out = math.ceil(input_tokens * 1.25), math.ceil(expected_output * 1.5)
    def price(i: int, o: int, discount: float = 1.0) -> float:
        return round((i * input_per_million + o * output_per_million) / 1_000_000 * discount, 2)
    ratio = sample_size / len(data)
    return {"version": VERSION, "model": DEFAULT_MODEL, "substantive_documents": len(data),
            "compatible_cache_hits": 0, "new_api_calls_required": len(data),
            "estimated_input_tokens": input_tokens, "estimated_output_tokens": expected_output,
            "expected_cost_usd": price(input_tokens, expected_output),
            "conservative_p95_cost_usd": price(p95_in, p95_out),
            "batch_supported": True, "batch_expected_cost_usd": price(input_tokens, expected_output, batch_discount),
            "validation_documents": sample_size,
            "validation_expected_cost_usd": price(math.ceil(input_tokens * ratio), math.ceil(expected_output * ratio)),
            "pricing": {"input_per_million_usd": input_per_million, "output_per_million_usd": output_per_million,
                        "batch_discount": batch_discount}, "api_calls_made": 0}


def require_gold_frozen() -> None:
    if not (AUDIT_ROOT / "absa_v1_audit_gold_v1.frozen.json").exists():
        raise RuntimeError("Full-corpus ABSA is blocked until human validation gold is frozen and evaluated")


def create_batch_requests(scope: str, output_dir: Path = ROOT) -> Path:
    if scope not in {"validation", "full"}:
        raise ValueError("scope must be validation or full")
    if scope == "full":
        require_gold_frozen()
        source = pd.read_csv(CORPUS, low_memory=False)
    else:
        if not (AUDIT_ROOT / "absa_v1_audit_gold_v1.frozen.json").exists():
            raise RuntimeError("Freeze human gold before exposing the validation sample to model scoring")
        sample = pd.read_csv(SAMPLE_PATH)
        corpus = pd.read_csv(CORPUS, low_memory=False)
        source = corpus[corpus.document_id.astype(str).isin(sample.document_id.astype(str))]
    ontology = load_ontology()
    ids = [a["id"] for a in ontology["aspects"]]
    schema = mention_schema(ids)
    batch_dir = output_dir / "batch" / scope
    batch_dir.mkdir(parents=True, exist_ok=True)
    path = batch_dir / f"absa_v1_{scope}_requests.jsonl"
    lines = []
    for row in source.itertuples(index=False):
        context = {"document_id": str(row.document_id), "original_text": str(row.original_text),
                   "final_topic_name": str(row.final_topic_name), "final_topic_group": str(row.final_topic_group),
                   "topic_context_supplied": True, "ontology": ids}
        body = {"model": DEFAULT_MODEL, "instructions": ABSA_V1_INSTRUCTIONS,
                "input": json.dumps(context, ensure_ascii=False),
                "text": {"format": {"type": "json_schema", "name": SCHEMA_VERSION,
                                      "strict": True, "schema": schema}}}
        lines.append(json.dumps({"custom_id": f"absa1-{row.document_id}", "method": "POST",
                                 "url": "/v1/responses", "body": body}, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (batch_dir / "manifest.json").write_text(json.dumps({"scope": scope, "requests": len(lines),
        "model": DEFAULT_MODEL, "prompt_version": PROMPT_VERSION, "schema_version": SCHEMA_VERSION,
        "topic_context_supplied": True, "submitted": False}, indent=2), encoding="utf-8")
    return path


def submit_batch(scope: str, output_dir: Path = ROOT) -> dict[str, Any]:
    from openai import OpenAI
    batch_dir = output_dir / "batch" / scope
    request_path = batch_dir / f"absa_v1_{scope}_requests.jsonl"
    if not request_path.exists():
        raise FileNotFoundError("Create and review the request package first")
    client = OpenAI()
    with request_path.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(input_file_id=uploaded.id, endpoint="/v1/responses",
                                  completion_window="24h", metadata={"stage": VERSION, "scope": scope})
    record = {"batch_id": batch.id, "input_file_id": uploaded.id, "status": batch.status,
              "submitted_at_utc": datetime.now(timezone.utc).isoformat()}
    (batch_dir / "submission.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def batch_status(scope: str, output_dir: Path = ROOT) -> dict[str, Any]:
    from openai import OpenAI
    batch_dir = output_dir / "batch" / scope
    submission = json.loads((batch_dir / "submission.json").read_text(encoding="utf-8"))
    batch = OpenAI().batches.retrieve(submission["batch_id"])
    record = batch.model_dump()
    (batch_dir / "status.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def import_batch(scope: str, output_dir: Path = ROOT) -> Path:
    from openai import OpenAI
    batch_dir = output_dir / "batch" / scope
    status = json.loads((batch_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("status") != "completed" or not status.get("output_file_id"):
        raise RuntimeError("Batch must be completed before import")
    content = OpenAI().files.content(status["output_file_id"]).content
    path = batch_dir / f"absa_v1_{scope}_raw_results.jsonl"
    path.write_bytes(content)
    return path


def finalize_batch(scope: str, output_dir: Path = ROOT) -> Path:
    batch_dir = output_dir / "batch" / scope
    raw_path = batch_dir / f"absa_v1_{scope}_raw_results.jsonl"
    name = "absa_validation_predictions_v1.csv" if scope == "validation" else "absa_mentions_v1.csv"
    path = output_dir / name
    diagnostic_path = output_dir / f"absa_{scope}_evidence_diagnostic_v1.csv"
    summary_path = output_dir / f"absa_{scope}_evidence_diagnostic_summary_v1.json"
    error_path = output_dir / f"absa_{scope}_evidence_errors_v1.csv"
    finalization_path = output_dir / f"absa_{scope}_finalization_manifest_v1.json"
    if path.exists() or finalization_path.exists():
        if not (path.exists() and finalization_path.exists()):
            raise FileExistsError("Partial finalization artifacts exist; refusing overwrite")
        manifest = json.loads(finalization_path.read_text(encoding="utf-8"))
        if manifest.get("raw_batch_sha256") != hashlib.sha256(raw_path.read_bytes()).hexdigest():
            raise FileExistsError("Existing predictions belong to a different raw Batch; refusing overwrite")
        return path
    corpus = pd.read_csv(CORPUS, low_memory=False).set_index("document_id", drop=False)
    aspect_ids = [a["id"] for a in load_ontology()["aspects"]]
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    document_errors: list[dict[str, Any]] = []
    processed_documents: set[str] = set()
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line); custom_id = item["custom_id"]
            document_id = custom_id.removeprefix("absa1-")
            body = item.get("response", {}).get("body", {})
            texts = [c.get("text", "") for out in body.get("output", []) for c in out.get("content", [])
                     if c.get("type") == "output_text"]
            if not texts or document_id not in corpus.index:
                raise ValueError(f"Missing result or unknown document: {document_id}")
            if document_id in processed_documents:
                raise ValueError(f"Duplicate Batch result: {document_id}")
            processed_documents.add(document_id)
            doc = corpus.loc[document_id]; caption = str(doc.original_text)
            result = json.loads(texts[0]); mentions = result.get("mentions")
            if not isinstance(mentions, list):
                raise ValueError("mentions must be an array")
            for index, mention in enumerate(mentions):
                aspect = mention.get("aspect", ""); sentiment = mention.get("sentiment", "")
                expression = mention.get("expression_type", "")
                structural_error = ""
                if aspect not in aspect_ids or sentiment not in SENTIMENTS or expression not in EXPRESSION_TYPES:
                    structural_error = "invalid controlled aspect, sentiment, or expression_type"
                recovery = canonicalize_evidence(caption, mention.get("evidence_text", ""),
                                                 mention.get("evidence_start"), mention.get("evidence_end"))
                final_evidence = recovery["final_exact_evidence_text"]
                diagnostics.append({"document_id": document_id, "original_caption": caption,
                    "mention_index": index, "aspect": aspect, "sentiment": sentiment,
                    "predicted_evidence_text": mention.get("evidence_text", ""),
                    "exact_substring_match": recovery["raw_exact_match"],
                    "predicted_evidence_start": mention.get("evidence_start", ""),
                    "predicted_evidence_end": mention.get("evidence_end", ""),
                    "normalized_match_status": recovery["normalized_match_status"],
                    "likely_mismatch_category": recovery["likely_mismatch_category"],
                    "evidence_repaired": recovery["evidence_repaired"],
                    "evidence_repair_method": recovery["evidence_repair_method"],
                    "original_model_evidence_text": recovery["original_model_evidence_text"],
                    "final_exact_evidence_text": final_evidence,
                    "final_evidence_start": recovery["final_evidence_start"],
                    "final_evidence_end": recovery["final_evidence_end"],
                    "structural_error": structural_error})
                row = {"mention_id": stable_mention_id(document_id, index, aspect,
                                                         final_evidence or recovery["original_model_evidence_text"]),
                    "document_id": document_id, "source": doc.source, "event_year": doc.event_year,
                    "raw_c1_topic_id": doc.original_topic_id, "final_topic_id": doc.final_topic_id,
                    "final_topic_name": doc.final_topic_name, "final_topic_group": doc.final_topic_group,
                    **mention, "evidence_text": final_evidence, "evidence_start": recovery["final_evidence_start"],
                    "evidence_end": recovery["final_evidence_end"], "raw_exact_match": recovery["raw_exact_match"],
                    "evidence_repaired": recovery["evidence_repaired"],
                    "evidence_repair_method": recovery["evidence_repair_method"],
                    "original_model_evidence_text": recovery["original_model_evidence_text"],
                    "final_exact_evidence_text": final_evidence,
                    "evidence_grounding_valid": bool(final_evidence) and not structural_error,
                    "prediction_structurally_valid": not structural_error,
                    "prediction_error": structural_error or ("unrecoverable_evidence" if not final_evidence else ""),
                    "topic_context_supplied": True, "prompt_version": PROMPT_VERSION,
                    "schema_version": SCHEMA_VERSION, "model": body.get("model", DEFAULT_MODEL)}
                rows.append(row)
        except Exception as exc:
            document_errors.append({"document_id": locals().get("document_id", "unknown"),
                                    "error": type(exc).__name__, "message": str(exc)})
    diagnostic = pd.DataFrame(diagnostics)
    predictions = pd.DataFrame(rows)
    errors = diagnostic[(~diagnostic.exact_substring_match) | diagnostic.structural_error.astype(bool)].copy()
    total = len(diagnostic); raw_exact = int(diagnostic.exact_substring_match.sum()) if total else 0
    recoverable = int(diagnostic.evidence_repaired.sum()) if total else 0
    unrecoverable = int(diagnostic.normalized_match_status.eq("unrecoverable").sum()) if total else 0
    summary = {"version": EVIDENCE_DIAGNOSTIC_VERSION, "failed_first_finalization_preserved": True,
        "failed_first_finalization_error": "ValueError: Every evidence_text must be an exact caption substring",
        "total_documents": len(processed_documents), "total_predicted_mentions": total,
        "raw_exact_evidence_matches": raw_exact, "deterministically_recoverable_mismatches": recoverable,
        "unrecoverable_grounding_failures": unrecoverable,
        "failed_raw_grounding_mentions": total - raw_exact,
        "documents_affected": int(diagnostic.loc[~diagnostic.exact_substring_match, "document_id"].nunique()) if total else 0,
        "raw_exact_grounding_rate": raw_exact / total if total else 1.0,
        "recoverable_exact_span_grounding_rate": (raw_exact + recoverable) / total if total else 1.0,
        "mismatch_category_counts": diagnostic.loc[~diagnostic.exact_substring_match,
            "likely_mismatch_category"].value_counts().to_dict() if total else {},
        "document_level_errors": document_errors, "api_calls": 0,
        "raw_batch_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "gold_sha256": hashlib.sha256(GOLD_PATH.read_bytes()).hexdigest() if GOLD_PATH.exists() else None,
        "ontology_sha256": hashlib.sha256(ONTOLOGY_PATH.read_bytes()).hexdigest()}
    diagnostic.to_csv(diagnostic_path, index=False, encoding="utf-8-sig")
    errors.to_csv(error_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(path, index=False, encoding="utf-8-sig")
    if scope == "validation":
        gold = pd.read_csv(GOLD_PATH).fillna("")
        sample = pd.read_csv(SAMPLE_PATH)
        captions = sample.assign(document_id=sample.document_id.astype(str)).set_index("document_id").original_text.astype(str).to_dict()
        languages = sample.assign(document_id=sample.document_id.astype(str)).set_index("document_id").primary_language.astype(str).to_dict()
        metrics = validation_metrics(gold, predictions, captions, languages)
        metrics["evidence_diagnostic"] = summary
        (output_dir / "absa_validation_metrics_v1.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    finalization_path.write_text(json.dumps({"version": "absa_v1_safe_finalization_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "scope": scope,
        "raw_batch_sha256": summary["raw_batch_sha256"], "predictions_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "diagnostic_sha256": hashlib.sha256(diagnostic_path.read_bytes()).hexdigest(), "api_calls": 0,
        "raw_batch_modified": False, "gold_modified": False, "ontology_modified": False}, indent=2), encoding="utf-8")
    return path


def validation_metrics(gold: pd.DataFrame, predictions: pd.DataFrame,
                       captions: dict[str, str], languages: dict[str, str] | None = None) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    gold = gold.copy(); predictions = predictions.copy()
    for frame in (gold, predictions):
        frame["document_id"] = frame["document_id"].astype(str)
    gold_keys = Counter(zip(gold.document_id, gold.aspect))
    pred_keys = Counter(zip(predictions.document_id, predictions.aspect))
    matched = gold_keys & pred_keys
    tp = sum(matched.values()); fp = len(predictions) - tp; fn = len(gold) - tp
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    gold_sent, pred_sent = [], []
    joint_tp = 0
    for key, count in matched.items():
        g = gold[(gold.document_id == key[0]) & (gold.aspect == key[1])].sort_values("mention_id").head(count)
        q = predictions[(predictions.document_id == key[0]) & (predictions.aspect == key[1])].sort_values("mention_id").head(count)
        gold_sent.extend(g.sentiment.tolist()); pred_sent.extend(q.sentiment.tolist())
        joint_tp += sum(a == b for a, b in zip(g.sentiment, q.sentiment))
    labels = SENTIMENTS
    if gold_sent:
        pr, rc, fs, support = precision_recall_fscore_support(gold_sent, pred_sent, labels=labels, zero_division=0)
        per_class = {label: {"precision": pr[i], "recall": rc[i], "f1": fs[i], "support": int(support[i])}
                     for i, label in enumerate(labels)}
        sent = {"accuracy": accuracy_score(gold_sent, pred_sent), "macro_f1": float(fs.mean()), "per_class": per_class}
    else:
        sent = {"accuracy": None, "macro_f1": None, "per_class": {}}
    joint_p = joint_tp / len(predictions) if len(predictions) else 0.0
    joint_r = joint_tp / len(gold) if len(gold) else 0.0
    joint_f = 2 * joint_p * joint_r / (joint_p + joint_r) if joint_p + joint_r else 0.0
    raw_valid = []
    recoverable_valid = []
    spans = []
    for row in predictions.itertuples():
        caption = captions.get(str(row.document_id), "")
        evidence = str(row.evidence_text)
        raw = bool(getattr(row, "raw_exact_match", bool(evidence) and evidence in caption))
        recovered = bool(evidence) and evidence in caption
        raw_valid.append(raw); recoverable_valid.append(recovered)
        try:
            start, end = int(row.evidence_start), int(row.evidence_end)
        except (TypeError, ValueError):
            start = end = -1
        spans.append(recovered and 0 <= start < end <= len(caption) and caption[start:end] == evidence)
    all_docs = set(captions); gold_docs = set(gold.document_id); pred_docs = set(predictions.document_id)
    gold_zero, pred_zero = all_docs - gold_docs, all_docs - pred_docs
    zero_agreement = sum((doc in gold_zero) == (doc in pred_zero) for doc in all_docs) / len(all_docs) if all_docs else 1.0
    multi_gold = set(gold.groupby("document_id").size().loc[lambda x: x > 1].index)
    multi_pred = set(predictions.groupby("document_id").size().loc[lambda x: x > 1].index)
    diagnostics = {"zero_mention_agreement": zero_agreement, "gold_zero_documents": len(gold_zero),
                   "predicted_zero_documents": len(pred_zero), "gold_multi_aspect_documents": len(multi_gold),
                   "predicted_multi_aspect_documents": len(multi_pred),
                   "zero_mention_confusion_gold_rows_predicted_columns": {
                       "gold_zero_predicted_zero": len(gold_zero & pred_zero),
                       "gold_zero_predicted_mentions": len(gold_zero - pred_zero),
                       "gold_mentions_predicted_zero": len(pred_zero - gold_zero),
                       "gold_mentions_predicted_mentions": len((all_docs - gold_zero) & (all_docs - pred_zero)),
                   },
                   "multi_aspect_document_agreement": len(multi_gold & multi_pred) / len(multi_gold | multi_pred) if multi_gold | multi_pred else 1.0}
    if languages:
        subgroup_definitions = {language: {str(d) for d, lang in languages.items() if lang == language}
                                for language in sorted(set(languages.values()))}
        subgroup_definitions["English"] = {str(d) for d, lang in languages.items() if lang == "English"}
        subgroup_definitions["non-English/uncertain"] = {str(d) for d, lang in languages.items() if lang != "English"}
        subgroup_metrics = {}
        for language, document_ids in subgroup_definitions.items():
            entry: dict[str, Any] = {"documents": len(document_ids), "minimum_documents_for_metrics": 5}
            if len(document_ids) >= 5:
                subset_captions = {doc: captions[doc] for doc in document_ids if doc in captions}
                subset_gold = gold[gold.document_id.isin(document_ids)]
                subset_predictions = predictions[predictions.document_id.isin(document_ids)]
                subset = validation_metrics(subset_gold, subset_predictions, subset_captions)
                entry.update({"metrics_reported": True, "gold_mentions": len(subset_gold),
                              "predicted_mentions": len(subset_predictions),
                              "aspect_detection": subset["aspect_detection"],
                              "matched_sentiment": subset["matched_sentiment"],
                              "joint_aspect_sentiment": subset["joint_aspect_sentiment"],
                              "evidence_grounding": subset["evidence_grounding"],
                              "zero_mention_agreement": subset["document_diagnostics"]["zero_mention_agreement"]})
            else:
                entry.update({"metrics_reported": False, "reason": "sample_size_below_5"})
            subgroup_metrics[language] = entry
        diagnostics["multilingual_subgroups"] = subgroup_metrics
    return {"aspect_detection": {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn},
            "matched_sentiment": sent,
            "joint_aspect_sentiment": {"precision": joint_p, "recall": joint_r, "f1": joint_f},
            "evidence_grounding": {
                "strict_raw_model_exact_count": sum(raw_valid),
                "strict_raw_model_evidence_grounding_rate": sum(raw_valid) / len(raw_valid) if raw_valid else 1.0,
                "final_recoverable_exact_span_count": sum(recoverable_valid),
                "final_recoverable_exact_span_grounding_rate": sum(recoverable_valid) / len(recoverable_valid) if recoverable_valid else 1.0,
                "unrecoverable_grounding_failures": len(recoverable_valid) - sum(recoverable_valid),
                "exact_substring_rate": sum(recoverable_valid) / len(recoverable_valid) if recoverable_valid else 1.0,
                "valid_span_rate": sum(spans) / len(spans) if spans else 1.0},
            "document_diagnostics": diagnostics,
            "matching_rule": "one-to-one within document+aspect, stable mention_id order"}


def evaluate_validation(output_dir: Path = ROOT) -> Path:
    """Evaluate existing validation predictions against frozen gold without network access."""
    predictions_path = output_dir / "absa_validation_predictions_v1.csv"
    json_path = output_dir / "absa_validation_evaluation_v1.json"
    csv_path = output_dir / "absa_validation_evaluation_summary_v1.csv"
    subgroup_path = output_dir / "absa_validation_evaluation_subgroups_v1.csv"
    if not predictions_path.exists():
        raise FileNotFoundError("Finalize the imported validation Batch before evaluation")
    marker_path = AUDIT_ROOT / "absa_v1_audit_gold_v1.frozen.json"
    if not GOLD_PATH.exists() or not marker_path.exists():
        raise RuntimeError("Frozen 80-document human gold is required")
    source_hashes = {"gold_sha256": hashlib.sha256(GOLD_PATH.read_bytes()).hexdigest(),
                     "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
                     "sample_sha256": hashlib.sha256(SAMPLE_PATH.read_bytes()).hexdigest(),
                     "ontology_sha256": hashlib.sha256(ONTOLOGY_PATH.read_bytes()).hexdigest()}
    if json_path.exists() or csv_path.exists() or subgroup_path.exists():
        if not (json_path.exists() and csv_path.exists() and subgroup_path.exists()):
            raise FileExistsError("Partial validation-evaluation artifacts exist; refusing overwrite")
        existing = json.loads(json_path.read_text(encoding="utf-8"))
        if existing.get("input_hashes") != source_hashes:
            raise FileExistsError("Existing evaluation uses different frozen inputs; refusing overwrite")
        return json_path
    sample = pd.read_csv(SAMPLE_PATH)
    if len(sample) != 80 or sample.document_id.astype(str).nunique() != 80:
        raise ValueError("Evaluation requires the frozen 80-document audit sample")
    gold = pd.read_csv(GOLD_PATH).fillna("")
    predictions = pd.read_csv(predictions_path).fillna("")
    sample_ids = set(sample.document_id.astype(str))
    if not set(gold.document_id.astype(str)).issubset(sample_ids):
        raise ValueError("Gold contains documents outside the frozen sample")
    if not set(predictions.document_id.astype(str)).issubset(sample_ids):
        raise ValueError("Predictions contain documents outside the frozen sample")
    indexed = sample.assign(document_id=sample.document_id.astype(str)).set_index("document_id")
    captions = indexed.original_text.astype(str).to_dict()
    languages = indexed.primary_language.astype(str).to_dict()
    metrics = validation_metrics(gold, predictions, captions, languages)
    report = {"version": "absa_v1_validation_evaluation_v1",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "first_frozen_non_paid_evaluation", "api_calls": 0,
        "inference_rerun": False, "matching_rules_changed": False,
        "matching_methodology": metrics["matching_rule"],
        "population": {"gold_documents_total": len(sample),
            "gold_documents_with_mentions": int(gold.document_id.astype(str).nunique()),
            "gold_zero_mention_documents": len(sample_ids - set(gold.document_id.astype(str))),
            "gold_mentions": len(gold), "prediction_documents_total": len(sample),
            "predicted_documents_with_mentions": int(predictions.document_id.astype(str).nunique()),
            "predicted_zero_mention_documents": len(sample_ids - set(predictions.document_id.astype(str))),
            "predicted_mentions": len(predictions)},
        "metrics": metrics, "input_hashes": source_hashes,
        "immutability": {"gold_modified": False, "predictions_modified": False,
                         "ontology_modified": False, "prompt_modified": False,
                         "schema_modified": False}}
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    flat_rows = []
    def add(section: str, metric: str, value: Any) -> None:
        flat_rows.append({"section": section, "metric": metric, "value": value})
    for key, value in report["population"].items(): add("population", key, value)
    for section in ["aspect_detection", "matched_sentiment", "joint_aspect_sentiment", "evidence_grounding"]:
        for key, value in metrics[section].items():
            if key != "per_class": add(section, key, value)
    for label, values in metrics["matched_sentiment"].get("per_class", {}).items():
        for key, value in values.items(): add(f"sentiment_{label}", key, value)
    for key, value in metrics["document_diagnostics"].items():
        if key not in {"multilingual_subgroups", "zero_mention_confusion_gold_rows_predicted_columns"}:
            add("document_diagnostics", key, value)
    for key, value in metrics["document_diagnostics"]["zero_mention_confusion_gold_rows_predicted_columns"].items():
        add("zero_mention_confusion", key, value)
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    subgroup_rows = []
    for language, entry in metrics["document_diagnostics"]["multilingual_subgroups"].items():
        row = {"subgroup": language, "documents": entry["documents"],
               "metrics_reported": entry["metrics_reported"], "reason": entry.get("reason", "")}
        if entry["metrics_reported"]:
            row.update({"gold_mentions": entry["gold_mentions"], "predicted_mentions": entry["predicted_mentions"],
                        "aspect_precision": entry["aspect_detection"]["precision"],
                        "aspect_recall": entry["aspect_detection"]["recall"],
                        "aspect_f1": entry["aspect_detection"]["f1"],
                        "sentiment_accuracy": entry["matched_sentiment"]["accuracy"],
                        "sentiment_macro_f1": entry["matched_sentiment"]["macro_f1"],
                        "joint_precision": entry["joint_aspect_sentiment"]["precision"],
                        "joint_recall": entry["joint_aspect_sentiment"]["recall"],
                        "joint_f1": entry["joint_aspect_sentiment"]["f1"],
                        "raw_grounding_rate": entry["evidence_grounding"]["strict_raw_model_evidence_grounding_rate"],
                        "recoverable_grounding_rate": entry["evidence_grounding"]["final_recoverable_exact_span_grounding_rate"],
                        "zero_mention_agreement": entry["zero_mention_agreement"]})
        subgroup_rows.append(row)
    pd.DataFrame(subgroup_rows).to_csv(subgroup_path, index=False, encoding="utf-8-sig")
    return json_path


def aggregate_mentions(mentions: pd.DataFrame, group_by: list[str]) -> pd.DataFrame:
    mention_counts = mentions.groupby(group_by, dropna=False).size().rename("aspect_mentions").reset_index()
    doc_counts = mentions.groupby(group_by, dropna=False)["document_id"].nunique().rename("documents_with_mention").reset_index()
    return mention_counts.merge(doc_counts, on=group_by)


def annotation_paths(historical_150: bool = False) -> dict[str, Path]:
    if historical_150:
        return {"sample": HISTORICAL_SAMPLE_PATH, "annotations": ROOT / "absa_validation_annotations_working_v1.csv",
                "progress": ROOT / "absa_validation_progress_v1.csv", "gold": ROOT / "absa_validation_gold_v1.csv"}
    return {"sample": SAMPLE_PATH, "annotations": ANNOTATIONS_PATH, "progress": PROGRESS_PATH, "gold": GOLD_PATH}
