"""Local, non-destructive multilingual topic discovery (version 1)."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import ROOT
from .storage import read_table
from .text import URL_RE, MENTION_RE, HASHTAG_RE, repair_and_normalize

VERSION = "topic_discovery_corpus_v1"
OUTPUT_DIR = ROOT / "data" / "processed" / "topic_discovery_v1"
DECISIONS_PATH = ROOT / "data" / "processed" / "relevance_v8_single_researcher_cost_conservative" / "production_v1" / "relevance_v8_cost_conservative_decisions_v1.parquet"
DOCUMENTS_PATH = ROOT / "data" / "processed" / "documents.parquet"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CHINESE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
TOKEN_RE = re.compile(r"(?u)\b\w[\w'+-]*\b")
ONLY_PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)
REVIEW_LABELS = (
    "substantive_klsm_topic", "generic_running", "commercial_or_promotion",
    "other_event", "noise_or_uninterpretable", "mixed_topic",
    "needs_topic_refinement",
)
TOPIC_QUALITY_LABELS = ("coherent", "somewhat_mixed", "highly_mixed")
REVIEW_PATH = OUTPUT_DIR / "topic_review_c1_v1.csv"
REFINED_DIR = OUTPUT_DIR / "c1_refined_v1"
MODEL_SELECTION_DIR = OUTPUT_DIR / "model_selection_review_v1"
INTERPRETABILITY_CHANGE_LABELS = ("improved", "similar", "worse")
SEMANTIC_RELATION_LABELS = (
    "preserves_theme", "merges_related_topics", "splits_theme_usefully",
    "mixes_unrelated_topics", "recovers_useful_outliers", "unclear",
)

# Curated aliases observed during the completed c1 topic review. These are
# demojized style markers, not ordinary lexical words.
EMOJI_STYLE_BASES = {
    "face_with_tears_of_joy", "rolling_on_the_floor_laughing", "flexed_biceps",
    "woman_running", "man_running", "person_running", "camera_with_flash",
    "fire", "red_heart", "smiling_face_with_heart_eyes", "clapping_hands",
    "raising_hands", "thumbs_up", "folded_hands", "party_popper",
}
SKIN_TONE_SUFFIX_RE = re.compile(r"_(?:light|medium_light|medium|medium_dark|dark)_skin_tone$")
SPACED_KLSCM_RE = re.compile(r"(?i)(?<!\w)K\s+L\s+S\s+C\s+M(?:\s+([12])\s+([0-9])\s+([0-9])\s+([0-9]))?(?!\w)")
MALAY_INDONESIAN_STOPWORDS = {
    "ada", "adalah", "aja", "akan", "aku", "anda", "atau", "awak", "bagi",
    "bahawa", "banyak", "baru", "boleh", "buat", "dah", "dalam", "dan", "dari",
    "daripada", "dekat", "dengan", "dia", "di", "diorang", "dulu", "gak", "guna",
    "hanya", "hari", "ini", "itu", "jadi", "je", "juga", "kat", "kau", "ke",
    "kita", "kami", "korang", "lagi", "lah", "la", "macam", "mau", "memang",
    "mereka", "nak", "ni", "nya", "oleh", "pada", "pun", "saja", "saya", "sebab",
    "sudah", "tak", "tapi", "telah", "tidak", "tu", "untuk", "utk", "yang", "yg",
    "ya", "you",
}
# Evidence-coded from the completed c1 review's top terms and human names.
C1_ARTIFACT_TOPIC_GROUPS = {
    "emoji_token_driven": {4, 5, 8, 9, 11, 13, 19, 21, 22, 25, 26, 29, 32, 38},
    "english_malay_stopword_dominated": {0, 1, 2, 7, 10, 12, 14, 18, 24, 26, 27, 30, 31, 34},
    "formatting_or_repetitive_hashtag": {2, 3, 9, 18, 28, 35, 36, 38, 39},
    "fragmented_photography": {11, 13, 17, 36},
    "generic_running_incidental_klscm": {19, 28, 35, 39},
}

CANDIDATES = [
    {"candidate_id": "c1", "umap_n_neighbors": 15, "hdbscan_min_cluster_size": 30, "hdbscan_min_samples": 10},
    {"candidate_id": "c2", "umap_n_neighbors": 30, "hdbscan_min_cluster_size": 30, "hdbscan_min_samples": 10},
    {"candidate_id": "c3", "umap_n_neighbors": 15, "hdbscan_min_cluster_size": 50, "hdbscan_min_samples": 15},
    {"candidate_id": "c4", "umap_n_neighbors": 30, "hdbscan_min_cluster_size": 50, "hdbscan_min_samples": 15},
]


def normalize_hashtag(tag: str) -> str:
    """Remove # and conservatively split separators/camel case and letter-number edges."""
    value = tag.lstrip("#").strip()
    value = re.sub(r"[_-]+", " ", value)
    value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    value = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def semantic_topic_text(text: object) -> tuple[str, list[str]]:
    original = repair_and_normalize(text)
    hashtags = [m.group(1).rstrip(".,!?:;)]}") for m in HASHTAG_RE.finditer(original)]
    value = URL_RE.sub(" ", original)
    value = MENTION_RE.sub(" ", value)
    value = HASHTAG_RE.sub(lambda m: " " + normalize_hashtag(m.group(1)) + " ", value)
    try:
        import emoji
        value = emoji.demojize(value, delimiters=(" ", " "))
    except ImportError:  # pragma: no cover - base project installs emoji
        pass
    return re.sub(r"\s+", " ", value).strip(), hashtags


def _normalize_spaced_klscm(text: str) -> str:
    def replace(match: re.Match) -> str:
        year = "" if match.group(1) is None else " " + "".join(match.groups())
        return "KLSCM" + year
    return SPACED_KLSCM_RE.sub(replace, text)


def _is_emoji_style_token(token: str) -> bool:
    canonical = token.casefold().strip("_").replace("-", "_")
    base = SKIN_TONE_SUFFIX_RE.sub("", canonical)
    collapsed = base.replace("_", "")
    aliases, collapsed_aliases = _multiword_emoji_alias_forms()
    return base in aliases or collapsed in collapsed_aliases


@lru_cache(maxsize=1)
def _multiword_emoji_aliases() -> frozenset[str]:
    """Installed emoji aliases with >=2 components; single words stay semantic."""
    import emoji
    aliases = set(EMOJI_STYLE_BASES)
    for character in emoji.EMOJI_DATA:
        alias = emoji.demojize(character).strip(":").casefold().replace("-", "_")
        alias = SKIN_TONE_SUFFIX_RE.sub("", alias)
        if "_" in alias:
            aliases.add(alias)
    return frozenset(aliases)


@lru_cache(maxsize=1)
def _multiword_emoji_alias_forms() -> tuple[frozenset[str], frozenset[str]]:
    aliases = _multiword_emoji_aliases()
    return aliases, frozenset(item.replace("_", "") for item in aliases)


def refined_text_fields(original_text: object, max_hashtags: int = 12) -> tuple[str, str]:
    """Return minimally cleaned embedding text and cleaner c-TF-IDF text."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    original = _normalize_spaced_klscm(repair_and_normalize(original_text))
    hashtags = [normalize_hashtag(m.group(1)) for m in HASHTAG_RE.finditer(original)]
    # Stable case-insensitive de-duplication prevents repeated hashtag blocks.
    unique_hashtags, seen = [], set()
    for tag in hashtags:
        key = tag.casefold()
        if key and key not in seen:
            unique_hashtags.append(tag); seen.add(key)
    domain_re = re.compile(r"(?i)klscm|scklm|marathon|half|full|race|runner|running|finish|training|pacer|clinic|route|water|(?:10|21|42)\s*km")
    if len(unique_hashtags) > max_hashtags:
        important = [tag for tag in unique_hashtags if domain_re.search(tag)]
        remainder = [tag for tag in unique_hashtags if tag not in important]
        unique_hashtags = (important + remainder)[:max_hashtags]
    body = HASHTAG_RE.sub(" ", original)
    body = URL_RE.sub(" ", MENTION_RE.sub(" ", body))
    import emoji
    body = emoji.demojize(body, delimiters=(" ", " "))
    raw_tokens = re.findall(r"(?u)\b[\w'+-]+\b|[^\w\s]", body)
    body_tokens = [token for token in raw_tokens if not _is_emoji_style_token(token)]
    embedding = re.sub(r"\s+", " ", " ".join([*body_tokens, *unique_hashtags])).strip()
    stopwords = set(ENGLISH_STOP_WORDS) | MALAY_INDONESIAN_STOPWORDS
    representation_tokens = [
        token for token in multilingual_tokenizer(embedding)
        if token.casefold() not in stopwords and not _is_emoji_style_token(token)
    ]
    representation = " ".join(representation_tokens)
    return embedding, representation


def multilingual_tokenizer(text: str) -> list[str]:
    """Latin-aware tokens plus overlapping Chinese characters/bigrams."""
    lowered = text.casefold()
    without_chinese = CHINESE_RE.sub(" ", lowered)
    tokens = TOKEN_RE.findall(without_chinese)
    for span in CHINESE_RE.findall(lowered):
        chars = list(span)
        tokens.extend(chars)
        tokens.extend("".join(chars[i:i + 2]) for i in range(len(chars) - 1))
    return tokens


def quality_reason(original: str, semantic: str) -> str:
    if not original.strip():
        return "empty_text"
    without_urls = URL_RE.sub("", original).strip()
    if not without_urls:
        return "url_only"
    if not MENTION_RE.sub("", original).strip():
        return "mention_only"
    if not semantic.strip():
        return "effectively_empty_after_normalization"
    try:
        import emoji
        if emoji.replace_emoji(original, replace="").strip() == "":
            return "emoji_only"
    except ImportError:  # pragma: no cover
        pass
    if ONLY_PUNCT_RE.fullmatch(semantic):
        return "punctuation_only"
    if not any(char.isalnum() for char in semantic):
        return "unusable_or_corrupt_text"
    return ""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decision(row: pd.Series) -> str:
    value = str(row.get("v8_automatic_decision", "")).strip().lower()
    return value if value in {"include", "review", "exclude"} else "review"


def build_topic_corpus(decisions: pd.DataFrame, documents: pd.DataFrame) -> pd.DataFrame:
    """Return one row per finalized record; relevance never determines eligibility."""
    document_columns = [
        "document_id", "normalized_text", "linguistic_text", "semantic_text",
        "hashtags", "text_hash", "duplicate_of", "processing_status",
        "detected_language", "detected_languages", "language_iso",
        "language_confidence", "is_mixed_language", "possible_code_switching",
    ]
    available = [c for c in document_columns if c in documents.columns]
    merged = decisions.merge(documents[available], on="document_id", how="left", suffixes=("", "_prepared"), validate="one_to_one")
    rows = []
    for row in merged.to_dict("records"):
        original = row.get("original_text", "")
        semantic, hashtags = semantic_topic_text(original)
        reason = quality_reason(repair_and_normalize(original), semantic)
        route = row.get("review_route_source") or row.get("v8_routing_status") or ""
        rows.append({
            **row,
            "original_caption_text": original,
            "semantic_text_for_embeddings": semantic,
            "topic_hashtags": json.dumps(hashtags, ensure_ascii=False),
            "text_length_chars": len(semantic),
            "text_length_tokens": len(multilingual_tokenizer(semantic)),
            "relevance_production_route": route,
            "relevance_decision": _decision(pd.Series(row)),
            "data_quality_eligibility": reason == "",
            "data_quality_exclusion_reason": reason,
        })
    result = pd.DataFrame(rows)
    result["semantic_text_hash"] = result.semantic_text_for_embeddings.map(lambda x: hashlib.sha256(x.casefold().encode("utf-8")).hexdigest())
    eligible = result.data_quality_eligibility
    result["duplicate_representative_id"] = ""
    for _, group in result[eligible].groupby("semantic_text_hash", sort=False):
        representative = str(group.iloc[0].document_id)
        result.loc[group.index, "duplicate_representative_id"] = representative
        if len(group) > 1:
            duplicate_indices = group.index[1:]
            result.loc[duplicate_indices, "data_quality_eligibility"] = False
            result.loc[duplicate_indices, "data_quality_exclusion_reason"] = "exact_duplicate_semantic_text"
    result["include_in_topic_discovery"] = result.data_quality_eligibility.astype(bool)
    return result


def default_config() -> dict:
    return {
        "version": "bertopic_baseline_v1", "random_seed": 42,
        "embedding": {"model": EMBEDDING_MODEL, "local_only_after_download": True, "expected_dimensions": 768},
        "umap": {"n_components": 5, "min_dist": 0.0, "metric": "cosine", "random_state": 42},
        "hdbscan": {"metric": "euclidean", "cluster_selection_method": "eom", "prediction_data": True},
        "vectorizer": {"ngram_range": [1, 2], "min_df": 2, "max_features": 50000, "lowercase": True, "tokenizer": "multilingual_tokenizer_v1", "stopword_strategy": "none_initially; inspect multilingual/domain frequency first"},
        "bertopic": {"calculate_probabilities": False, "verbose": True, "nr_topics": None, "top_n_words": 20},
        "long_documents": {"strategy": "no_chunking_in_v1_preflight", "rationale": "inspect distribution before introducing parent/chunk semantics"},
        "candidates": CANDIDATES,
    }


def prepare_topic_corpus(output_dir: Path = OUTPUT_DIR) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions, documents = read_table(DECISIONS_PATH), read_table(DOCUMENTS_PATH)
    corpus = build_topic_corpus(decisions, documents)
    if len(corpus) != 13799 or corpus.document_id.nunique() != 13799:
        raise ValueError("Finalized production population must contain 13,799 unique IDs")
    included = corpus[corpus.include_in_topic_discovery].copy()
    excluded = corpus[~corpus.include_in_topic_discovery].copy()
    corpus_path = output_dir / "topic_discovery_corpus_v1.csv"
    excluded_path = output_dir / "topic_discovery_excluded_data_quality_v1.csv"
    config_path = output_dir / "bertopic_configuration_v1.json"
    corpus.to_csv(corpus_path, index=False, encoding="utf-8-sig")
    excluded.to_csv(excluded_path, index=False, encoding="utf-8-sig")
    config_path.write_text(json.dumps(default_config(), indent=2, ensure_ascii=False), encoding="utf-8")
    lengths = included.text_length_chars
    token_lengths = included.text_length_tokens
    manifest = {
        "version": VERSION, "generated_at_utc": datetime.now(timezone.utc).isoformat(), "api_calls": 0,
        "source_decisions": str(DECISIONS_PATH.relative_to(ROOT)), "source_decisions_sha256": _sha256(DECISIONS_PATH),
        "total_original_records": len(corpus), "eligible_records": len(included), "excluded_records": len(excluded),
        "exclusion_reasons": excluded.data_quality_exclusion_reason.value_counts().sort_index().to_dict(),
        "exact_duplicate_records_excluded_from_fitting": int(excluded.data_quality_exclusion_reason.eq("exact_duplicate_semantic_text").sum()),
        "unique_eligible_semantic_texts": int(included.semantic_text_hash.nunique()),
        "language_distribution": corpus.primary_language.fillna("unknown").replace("", "unknown").value_counts().to_dict(),
        "text_length_chars": {k: float(v) for k, v in lengths.describe(percentiles=[.25, .5, .75, .9, .95, .99]).to_dict().items()},
        "text_length_tokens": {k: float(v) for k, v in token_lengths.describe(percentiles=[.25, .5, .75, .9, .95, .99]).to_dict().items()},
        "source_distribution": corpus.source.fillna("unknown").replace("", "unknown").value_counts().to_dict(),
        "event_year_distribution": {str(k): int(v) for k, v in corpus.event_year.value_counts().sort_index().items()},
        "long_text_diagnostics_chars": {f"greater_than_{threshold}": int((lengths > threshold).sum()) for threshold in (1000, 1500, 2000, 2500, 3000)},
        "relevance_decision_distribution": corpus.relevance_decision.value_counts().to_dict(),
        "full_bertopic_executed": False, "absa_status": "blocked_pending_topic_review",
        "artifacts": {},
    }
    manifest_path = output_dir / "topic_discovery_corpus_manifest_v1.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest["artifacts"] = {p.name: _sha256(p) for p in (corpus_path, excluded_path, config_path)}
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def validate_embedding_cache(ids: Iterable[str], embeddings: np.ndarray) -> None:
    ids = list(ids)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(ids):
        raise ValueError("Embedding cache rows do not align with record IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("Embedding cache record IDs are not unique")


def _json_counts(series: pd.Series) -> str:
    counts = series.fillna("unknown").astype(str).replace("", "unknown").value_counts()
    return json.dumps({str(k): int(v) for k, v in counts.items()}, ensure_ascii=False)


def _examples(frame: pd.DataFrame) -> list[dict]:
    return [
        {"document_id": str(row.document_id), "caption": str(row.original_caption_text)}
        for row in frame.itertuples(index=False)
    ]


def build_c1_review_frames(
    corpus: pd.DataFrame,
    assignments: pd.DataFrame,
    topic_info: pd.DataFrame,
    representative_documents: dict[int, list[str]],
    seed: int = 104729,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create normal-topic, outlier, and largest-five review tables."""
    eligible = corpus[corpus.include_in_topic_discovery.astype(str).str.lower().isin({"true", "1"})]
    joined = assignments.merge(eligible, on="document_id", how="left", validate="one_to_one")
    if joined.original_caption_text.isna().any():
        raise ValueError("Assignments do not align with the eligible corpus")
    normal_ids = sorted(joined.loc[joined.topic_id.ne(-1), "topic_id"].unique().tolist())
    if normal_ids != list(range(40)):
        raise ValueError(f"Expected exactly c1 topics 0-39; found {normal_ids}")
    total = len(joined)
    rows = []
    info_by_topic = topic_info.set_index("Topic")
    for topic_id in normal_ids:
        group = joined[joined.topic_id.eq(topic_id)]
        random_docs = group.sample(n=min(5, len(group)), random_state=seed + topic_id)
        reps = representative_documents.get(topic_id, [])[:5]
        rows.append({
            "topic_id": topic_id, "topic_size": len(group),
            "corpus_percentage": len(group) / total * 100,
            "bertopic_name": str(info_by_topic.loc[topic_id, "Name"]),
            "top_terms": json.dumps(list(info_by_topic.loc[topic_id, "Representation"]) if isinstance(info_by_topic.loc[topic_id, "Representation"], list) else [], ensure_ascii=False),
            "representative_documents": json.dumps(reps, ensure_ascii=False),
            "random_documents": json.dumps(_examples(random_docs), ensure_ascii=False),
            "language_distribution": _json_counts(group.primary_language),
            "year_distribution": _json_counts(group.event_year),
            "relevance_distribution": _json_counts(group.relevance_decision),
            "source_distribution": _json_counts(group.source),
            "topic_relevance": "", "topic_name": "", "topic_quality": "",
            "notes": "", "reviewed_at": "",
        })
    review = pd.DataFrame(rows)
    outliers = joined[joined.topic_id.eq(-1)]
    outlier_sample = outliers.sample(n=min(30, len(outliers)), random_state=seed)
    outlier = pd.DataFrame([{
        "topic_id": -1, "topic_size": len(outliers),
        "corpus_percentage": len(outliers) / total * 100,
        "interpretation": "HDBSCAN outliers/unassigned documents; not an irrelevance label",
        "language_distribution": _json_counts(outliers.primary_language),
        "year_distribution": _json_counts(outliers.event_year),
        "relevance_distribution": _json_counts(outliers.relevance_decision),
        "source_distribution": _json_counts(outliers.source),
        "random_documents": json.dumps(_examples(outlier_sample), ensure_ascii=False),
    }])
    largest_rows = []
    for topic_id in review.nlargest(5, "topic_size").topic_id:
        group = joined[joined.topic_id.eq(topic_id)]
        largest_rows.append({
            "topic_id": int(topic_id), "topic_size": len(group),
            "top_terms": review.loc[review.topic_id.eq(topic_id), "top_terms"].iloc[0],
            "representative_documents": json.dumps(representative_documents.get(int(topic_id), [])[:10], ensure_ascii=False),
            "random_documents": json.dumps(_examples(group.sample(n=min(20, len(group)), random_state=seed + 1000 + int(topic_id))), ensure_ascii=False),
            "language_distribution": _json_counts(group.primary_language),
            "year_distribution": _json_counts(group.event_year),
        })
    return review, outlier, pd.DataFrame(largest_rows)


def _parse_topic_terms(value: object) -> list[str]:
    import ast
    parsed = ast.literal_eval(str(value))
    return [str(item) for item in parsed][:20]


def preserve_topic_annotations(fresh: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    """Refresh diagnostics while retaining every researcher-entered field."""
    annotation_columns = ["topic_relevance", "topic_name", "topic_quality", "notes", "reviewed_at"]
    if set(existing.topic_id) != set(fresh.topic_id):
        raise ValueError("Existing review topics do not align with the refreshed package")
    preserved = existing[["topic_id", *annotation_columns]]
    return fresh.drop(columns=annotation_columns).merge(preserved, on="topic_id", how="left", validate="one_to_one")


def prepare_c1_topic_review(output_dir: Path = OUTPUT_DIR, seed: int = 104729) -> dict:
    """Build review artifacts from the existing c1 model without fitting anything."""
    from bertopic import BERTopic

    corpus = pd.read_csv(output_dir / "topic_discovery_corpus_v1.csv", keep_default_na=False, low_memory=False)
    assignments = pd.read_csv(output_dir / "topic_assignments_c1_v1.csv")
    topic_info = pd.read_csv(output_dir / "topic_info_c1_v1.csv")
    topic_info["Representation"] = topic_info.Representation.map(_parse_topic_terms)
    model = BERTopic.load(output_dir / "bertopic_c1_v1")
    eligible = corpus[corpus.include_in_topic_discovery.astype(str).str.lower().isin({"true", "1"})]
    documents = assignments.merge(eligible[["document_id", "semantic_text_for_embeddings"]], on="document_id", validate="one_to_one")
    extraction_frame = pd.DataFrame({"Document": documents.semantic_text_for_embeddings, "Topic": documents.topic_id, "ID": range(len(documents))})
    representatives, _, _, _ = model._extract_representative_docs(
        model.c_tf_idf_, extraction_frame, model.topic_representations_,
        nr_samples=500, nr_repr_docs=10,
    )
    review, outlier, largest = build_c1_review_frames(corpus, assignments, topic_info, representatives, seed)
    review_path = output_dir / "topic_review_c1_v1.csv"
    if review_path.exists():
        existing = pd.read_csv(review_path, keep_default_na=False)
        review = preserve_topic_annotations(review, existing)
    review.to_csv(review_path, index=False, encoding="utf-8-sig")
    outlier.to_csv(output_dir / "topic_outlier_c1_v1.csv", index=False, encoding="utf-8-sig")
    largest.to_csv(output_dir / "topic_largest_five_diagnostics_c1_v1.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "version": "topic_review_c1_v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_calls": 0, "candidate_status": "provisional_baseline_not_final",
        "random_seed": seed, "normal_topics": len(review),
        "outlier_records": int(outlier.iloc[0].topic_size),
        "outlier_percentage": float(outlier.iloc[0].corpus_percentage),
        "largest_five": {str(row.topic_id): int(row.topic_size) for row in largest.itertuples()},
    }
    (output_dir / "topic_review_c1_manifest_v1.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def topic_review_summary(review_path: Path = REVIEW_PATH, assignments_path: Path | None = None) -> dict:
    assignments_path = assignments_path or OUTPUT_DIR / "topic_assignments_c1_v1.csv"
    review = pd.read_csv(review_path, keep_default_na=False)
    assignments = pd.read_csv(assignments_path)
    if set(review.topic_id) != set(range(40)):
        raise ValueError("Review file must contain exactly normal topics 0-39")
    valid = review.topic_relevance.isin(REVIEW_LABELS)
    sizes = assignments[assignments.topic_id.ne(-1)].topic_id.value_counts()
    categories = {}
    for label in REVIEW_LABELS:
        topic_ids = review.loc[review.topic_relevance.eq(label), "topic_id"]
        documents = int(sizes.reindex(topic_ids, fill_value=0).sum())
        categories[label] = {"topics": int(len(topic_ids)), "documents": documents, "document_percentage_of_normal_topics": documents / int(sizes.sum()) * 100}
    quality = {label: int(review.topic_quality.eq(label).sum()) for label in TOPIC_QUALITY_LABELS}
    outlier_count = int(assignments.topic_id.eq(-1).sum())
    return {"topics_reviewed": int(valid.sum()), "topics_total": 40, "categories": categories, "topic_quality": quality, "outliers": {"documents": outlier_count, "percentage_of_corpus": outlier_count / len(assignments) * 100}}


def freeze_c1_baseline(output_dir: Path = OUTPUT_DIR) -> dict:
    """Hash immutable c1 evidence before the controlled refinement."""
    names = [
        "bertopic_c1_v1", "topic_assignments_c1_v1.csv", "topic_info_c1_v1.csv",
        "topic_review_c1_v1.csv", "embeddings_v1.npy", "embedding_record_ids_v1.json",
        "bertopic_candidate_metrics_v1.csv", "bertopic_configuration_v1.json",
    ]
    artifacts = {name: _sha256(output_dir / name) for name in names}
    summary = topic_review_summary(output_dir / "topic_review_c1_v1.csv", output_dir / "topic_assignments_c1_v1.csv")
    payload = {
        "version": "c1_frozen_baseline_v1", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "immutable_baseline": True, "api_calls": 0, "artifacts": artifacts,
        "completed_human_review_summary": summary,
    }
    path = output_dir / "c1_frozen_baseline_manifest_v1.json"
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior["artifacts"] != artifacts:
            raise ValueError("Frozen c1 artifact hashes changed")
        return prior
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "topic_review_c1_summary_v1.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return payload


def _topic_metrics(labels: pd.Series) -> dict:
    counts = labels.value_counts()
    normal = counts.drop(index=-1, errors="ignore")
    outliers = int(counts.get(-1, 0))
    return {
        "topic_count": int(len(normal)), "outlier_count": outliers,
        "outlier_proportion": outliers / len(labels),
        "normal_topic_documents": int(normal.sum()),
        "normal_topic_proportion": float(normal.sum() / len(labels)),
        "median_topic_size": float(normal.median()) if len(normal) else 0,
        "largest_topic_size": int(normal.max()) if len(normal) else 0,
        "smallest_topic_size": int(normal.min()) if len(normal) else 0,
        "topic_size_distribution": {str(int(k)): int(v) for k, v in normal.sort_index().items()},
        "very_small_topics_below_30": int((normal < 30).sum()),
        "giant_topic_over_half_corpus": bool((normal > len(labels) / 2).any()),
    }


def run_c1_refined(output_dir: Path = OUTPUT_DIR, seed: int = 104729) -> dict:
    """Run one controlled representation refinement with unchanged c1 parameters."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    from umap import UMAP

    freeze_c1_baseline(output_dir)
    refined_dir = output_dir / "c1_refined_v1"
    refined_dir.mkdir(parents=True, exist_ok=True)
    corpus = pd.read_csv(output_dir / "topic_discovery_corpus_v1.csv", keep_default_na=False, low_memory=False)
    fit = corpus[corpus.include_in_topic_discovery.astype(str).str.lower().isin({"true", "1"})].copy()
    fields = fit.original_caption_text.map(refined_text_fields)
    fit["refined_embedding_text"] = fields.map(lambda value: value[0])
    fit["refined_representation_text"] = fields.map(lambda value: value[1])
    if fit.refined_embedding_text.eq("").any() or fit.refined_representation_text.eq("").any():
        raise ValueError("Refinement produced empty text for an eligible document")
    refined_corpus_path = refined_dir / "topic_discovery_corpus_c1_refined_v1.csv"
    fit.to_csv(refined_corpus_path, index=False, encoding="utf-8-sig")
    ids = fit.document_id.tolist()
    embeddings_path = refined_dir / "embeddings_c1_refined_v1.npy"
    ids_path = refined_dir / "embedding_record_ids_c1_refined_v1.json"
    if embeddings_path.exists() and ids_path.exists():
        embeddings = np.load(embeddings_path)
        if json.loads(ids_path.read_text(encoding="utf-8")) != ids:
            raise ValueError("Refined embedding cache IDs do not align")
    else:
        encoder = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
        embeddings = encoder.encode(fit.refined_embedding_text.tolist(), batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        validate_embedding_cache(ids, embeddings)
        np.save(embeddings_path, embeddings)
        ids_path.write_text(json.dumps(ids), encoding="utf-8")
    umap = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
    clusterer = HDBSCAN(min_cluster_size=30, min_samples=10, metric="euclidean", cluster_selection_method="eom", prediction_data=True)
    # Text is already tokenized after multilingual stopword removal.
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w[\w'+-]*\b", ngram_range=(1, 2), min_df=2, max_features=50000, lowercase=False)
    model = BERTopic(umap_model=umap, hdbscan_model=clusterer, vectorizer_model=vectorizer, top_n_words=20, calculate_probabilities=False, verbose=True)
    refined_labels, _ = model.fit_transform(fit.refined_representation_text.tolist(), embeddings)
    assignments = pd.DataFrame({"document_id": ids, "topic_id": refined_labels})
    assignments_path = refined_dir / "topic_assignments_c1_refined_v1.csv"
    assignments.to_csv(assignments_path, index=False)
    model.save(refined_dir / "bertopic_c1_refined_v1", serialization="pickle")
    model.get_topic_info().to_csv(refined_dir / "topic_info_c1_refined_v1.csv", index=False)

    original = pd.read_csv(output_dir / "topic_assignments_c1_v1.csv").rename(columns={"topic_id": "c1_topic_id"})
    aligned = original.merge(assignments.rename(columns={"topic_id": "refined_topic_id"}), on="document_id", validate="one_to_one")
    crosswalk_counts = aligned.groupby(["refined_topic_id", "c1_topic_id"]).size().rename("document_count").reset_index()
    refined_sizes = crosswalk_counts.groupby("refined_topic_id").document_count.sum().rename("refined_topic_size")
    crosswalk_counts = crosswalk_counts.merge(refined_sizes, on="refined_topic_id")
    crosswalk_counts["percentage_of_refined_topic"] = crosswalk_counts.document_count / crosswalk_counts.refined_topic_size * 100
    crosswalk_counts.sort_values(["refined_topic_id", "document_count"], ascending=[True, False]).to_csv(refined_dir / "c1_to_c1_refined_crosswalk_v1.csv", index=False)
    from scipy.optimize import linear_sum_assignment
    contingency = pd.crosstab(aligned.c1_topic_id, aligned.refined_topic_id)
    row_index, column_index = linear_sum_assignment(-contingency.to_numpy())
    optimally_matched = int(contingency.to_numpy()[row_index, column_index].sum())
    comparison = {
        "version": "c1_vs_c1_refined_v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "api_calls": 0,
        "decision_status": "unresolved_keep_c1_adopt_refined_or_neither",
        "original_c1": _topic_metrics(aligned.c1_topic_id),
        "c1_refined": _topic_metrics(aligned.refined_topic_id),
        "assignment_changes_after_optimal_label_mapping": len(aligned) - optimally_matched,
        "assignment_change_proportion_after_optimal_label_mapping": 1 - optimally_matched / len(aligned),
        "adjusted_rand_index": float(adjusted_rand_score(aligned.c1_topic_id, aligned.refined_topic_id)),
        "adjusted_mutual_information": float(adjusted_mutual_info_score(aligned.c1_topic_id, aligned.refined_topic_id)),
    }
    (refined_dir / "c1_vs_c1_refined_comparison_v1.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    topic_info = model.get_topic_info().copy()
    topic_info = topic_info.rename(columns={"Topic": "Topic", "Representation": "Representation"})
    representatives, _, _, representative_ids = model._extract_representative_docs(
        model.c_tf_idf_, pd.DataFrame({"Document": fit.refined_representation_text, "Topic": refined_labels, "ID": range(len(fit))}),
        model.topic_representations_, nr_samples=500, nr_repr_docs=10,
    )
    # Replace cleaned representation strings with original captions by matching
    # the deterministic selected frame indices returned by BERTopic.
    rep_captions = {}
    for topic, indices in zip(model.topic_representations_.keys(), representative_ids):
        rep_captions[int(topic)] = fit.loc[indices].original_caption_text.astype(str).tolist()[:10]
    review, outlier, largest = build_refined_review_frames(fit, assignments, topic_info, rep_captions, aligned, seed)
    review.to_csv(refined_dir / "topic_review_c1_refined_v1.csv", index=False, encoding="utf-8-sig")
    outlier.to_csv(refined_dir / "topic_outlier_c1_refined_v1.csv", index=False, encoding="utf-8-sig")
    largest.to_csv(refined_dir / "topic_largest_five_diagnostics_c1_refined_v1.csv", index=False, encoding="utf-8-sig")
    return finalize_c1_refined_reports(output_dir)


def build_refined_review_frames(
    fit: pd.DataFrame, assignments: pd.DataFrame, topic_info: pd.DataFrame,
    representatives: dict[int, list[str]], aligned: pd.DataFrame, seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    joined = assignments.merge(fit, on="document_id", validate="one_to_one")
    info = topic_info.set_index("Topic")
    normal_ids = sorted(joined.loc[joined.topic_id.ne(-1), "topic_id"].unique())
    rows = []
    for topic_id in normal_ids:
        group = joined[joined.topic_id.eq(topic_id)]
        contributions = aligned[aligned.refined_topic_id.eq(topic_id)].c1_topic_id.value_counts()
        rows.append({
            "topic_id": int(topic_id), "topic_size": len(group), "corpus_percentage": len(group) / len(joined) * 100,
            "bertopic_name": str(info.loc[topic_id, "Name"]),
            "top_terms": json.dumps(_parse_topic_terms(info.loc[topic_id, "Representation"]), ensure_ascii=False),
            "representative_documents": json.dumps(representatives.get(int(topic_id), [])[:5], ensure_ascii=False),
            "random_documents": json.dumps(_examples(group.sample(n=min(5, len(group)), random_state=seed + int(topic_id))), ensure_ascii=False),
            "c1_topic_contributions": json.dumps({str(int(k)): int(v) for k, v in contributions.items()}, ensure_ascii=False),
            "language_distribution": _json_counts(group.primary_language), "year_distribution": _json_counts(group.event_year),
            "relevance_distribution": _json_counts(group.relevance_decision), "source_distribution": _json_counts(group.source),
            "topic_relevance": "", "topic_name": "", "topic_quality": "", "notes": "", "reviewed_at": "",
        })
    review = pd.DataFrame(rows)
    outliers = joined[joined.topic_id.eq(-1)]
    outlier = pd.DataFrame([{
        "topic_id": -1, "topic_size": len(outliers), "corpus_percentage": len(outliers) / len(joined) * 100,
        "interpretation": "HDBSCAN outliers/unassigned documents; not an irrelevance label",
        "random_documents": json.dumps(_examples(outliers.sample(n=min(30, len(outliers)), random_state=seed)), ensure_ascii=False),
        "language_distribution": _json_counts(outliers.primary_language), "year_distribution": _json_counts(outliers.event_year),
        "relevance_distribution": _json_counts(outliers.relevance_decision),
    }])
    largest_rows = []
    for topic_id in review.nlargest(5, "topic_size").topic_id:
        group = joined[joined.topic_id.eq(topic_id)]
        largest_rows.append({
            "topic_id": int(topic_id), "topic_size": len(group),
            "top_terms": review.loc[review.topic_id.eq(topic_id), "top_terms"].iloc[0],
            "representative_documents": json.dumps(representatives.get(int(topic_id), [])[:10], ensure_ascii=False),
            "random_documents": json.dumps(_examples(group.sample(n=min(20, len(group)), random_state=seed + 1000 + int(topic_id))), ensure_ascii=False),
            "language_distribution": _json_counts(group.primary_language), "year_distribution": _json_counts(group.event_year),
        })
    return review, outlier, pd.DataFrame(largest_rows)


def finalize_c1_refined_reports(output_dir: Path = OUTPUT_DIR) -> dict:
    """Write reproducible QA, crosswalk summaries, configuration, and hashes."""
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    refined_dir = output_dir / "c1_refined_v1"
    corpus = pd.read_csv(refined_dir / "topic_discovery_corpus_c1_refined_v1.csv", keep_default_na=False, low_memory=False)
    original = pd.read_csv(output_dir / "topic_assignments_c1_v1.csv").rename(columns={"topic_id": "c1_topic_id"})
    refined = pd.read_csv(refined_dir / "topic_assignments_c1_refined_v1.csv").rename(columns={"topic_id": "refined_topic_id"})
    aligned = original.merge(refined, on="document_id", validate="one_to_one")
    contingency = pd.crosstab(aligned.c1_topic_id, aligned.refined_topic_id)
    row_index, column_index = linear_sum_assignment(-contingency.to_numpy())
    matched = int(contingency.to_numpy()[row_index, column_index].sum())
    comparison = {
        "version": "c1_vs_c1_refined_v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "api_calls": 0,
        "decision_status": "unresolved_keep_c1_adopt_refined_or_neither",
        "original_c1": _topic_metrics(aligned.c1_topic_id), "c1_refined": _topic_metrics(aligned.refined_topic_id),
        "assignment_changes_after_optimal_label_mapping": len(aligned) - matched,
        "assignment_change_proportion_after_optimal_label_mapping": 1 - matched / len(aligned),
        "adjusted_rand_index": float(adjusted_rand_score(aligned.c1_topic_id, aligned.refined_topic_id)),
        "adjusted_mutual_information": float(adjusted_mutual_info_score(aligned.c1_topic_id, aligned.refined_topic_id)),
        "metric_caution": "ARI/AMI and optimally mapped agreement describe similarity, not topic quality or model superiority.",
    }
    (refined_dir / "c1_vs_c1_refined_comparison_v1.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    crosswalk = pd.read_csv(refined_dir / "c1_to_c1_refined_crosswalk_v1.csv")
    old_review = pd.read_csv(output_dir / "topic_review_c1_v1.csv", keep_default_na=False)[["topic_id", "topic_name", "topic_relevance", "topic_quality"]]
    old_review = pd.concat([old_review, pd.DataFrame([{"topic_id": -1, "topic_name": "c1 outliers/unassigned", "topic_relevance": "outlier_unassigned", "topic_quality": "not_reviewed"}])], ignore_index=True)
    enriched = crosswalk.merge(old_review, left_on="c1_topic_id", right_on="topic_id", how="left").drop(columns="topic_id")
    enriched.to_csv(refined_dir / "c1_to_c1_refined_crosswalk_with_human_labels_v1.csv", index=False, encoding="utf-8-sig")
    normal = crosswalk[crosswalk.refined_topic_id.ne(-1)]
    dominant = normal.sort_values(["refined_topic_id", "document_count"], ascending=[True, False]).groupby("refined_topic_id").first()
    crosswalk_summary = {
        "refined_normal_topics": int(dominant.shape[0]),
        "dominant_c1_source_at_least_80_percent": int(dominant.percentage_of_refined_topic.ge(80).sum()),
        "dominant_c1_source_at_least_50_percent": int(dominant.percentage_of_refined_topic.ge(50).sum()),
        "median_dominant_c1_contribution_percent": float(dominant.percentage_of_refined_topic.median()),
        "refined_topics_dominated_by_original_outliers": int(dominant.c1_topic_id.eq(-1).sum()),
    }
    (refined_dir / "c1_to_c1_refined_crosswalk_summary_v1.json").write_text(json.dumps(crosswalk_summary, indent=2), encoding="utf-8")

    originals = corpus.original_caption_text.astype(str)
    duplicate_hashtag_docs = 0; capped_hashtag_docs = 0; emoji_alias_docs = 0
    for text in originals:
        tags = [normalize_hashtag(m.group(1)).casefold() for m in HASHTAG_RE.finditer(text)]
        unique = list(dict.fromkeys(tag for tag in tags if tag))
        duplicate_hashtag_docs += int(len(tags) > len(unique))
        capped_hashtag_docs += int(len(unique) > 12)
    for semantic in corpus.semantic_text_for_embeddings.astype(str):
        emoji_alias_docs += int(any(_is_emoji_style_token(token) for token in re.findall(r"(?u)\b[\w'+-]+\b", semantic)))
    preprocessing = {
        "version": "c1_refined_text_v1", "documents": len(corpus),
        "embedding_text_changed_vs_c1_semantic_text": int(corpus.semantic_text_for_embeddings.ne(corpus.refined_embedding_text).sum()),
        "documents_with_detected_multiword_emoji_aliases_before_refinement": emoji_alias_docs,
        "documents_with_spaced_KLSCM_artifact_normalized": int(originals.str.contains(r"(?i)K\s+L\s+S\s+C\s+M", regex=True).sum()),
        "documents_with_duplicate_hashtags_deduplicated": duplicate_hashtag_docs,
        "documents_with_more_than_12_unique_hashtags_capped": capped_hashtag_docs,
        "english_stopword_source": "sklearn.feature_extraction.text.ENGLISH_STOP_WORDS",
        "malay_indonesian_stopword_count": len(MALAY_INDONESIAN_STOPWORDS),
        "emoji_alias_rule": "remove installed emoji-library aliases containing two or more underscore-separated components, their collapsed forms, and skin-tone variants; preserve single-word aliases",
    }
    (refined_dir / "c1_refined_preprocessing_report_v1.json").write_text(json.dumps(preprocessing, indent=2), encoding="utf-8")
    config = {
        "version": "c1_refined_v1", "experiment": "one controlled text-representation refinement",
        "embedding_model": EMBEDDING_MODEL, "embedding_cache": "separate; original cache not reused because embedding text changed",
        "umap": {"n_neighbors": 15, "n_components": 5, "min_dist": 0.0, "metric": "cosine", "random_state": 42},
        "hdbscan": {"min_cluster_size": 30, "min_samples": 10, "metric": "euclidean", "cluster_selection_method": "eom", "prediction_data": True},
        "vectorizer": {"ngram_range": [1, 2], "min_df": 2, "max_features": 50000, "pretokenized_refined_representation_text": True},
        "decision_status": "unresolved_keep_c1_adopt_refined_or_neither", "api_calls": 0,
    }
    (refined_dir / "c1_refined_configuration_v1.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    artifact_names = [
        "bertopic_c1_refined_v1", "embeddings_c1_refined_v1.npy", "embedding_record_ids_c1_refined_v1.json",
        "topic_assignments_c1_refined_v1.csv", "topic_info_c1_refined_v1.csv", "topic_review_c1_refined_v1.csv",
        "topic_outlier_c1_refined_v1.csv", "topic_largest_five_diagnostics_c1_refined_v1.csv",
        "c1_to_c1_refined_crosswalk_v1.csv", "c1_to_c1_refined_crosswalk_with_human_labels_v1.csv",
        "c1_vs_c1_refined_comparison_v1.json", "c1_refined_configuration_v1.json", "c1_refined_preprocessing_report_v1.json",
    ]
    manifest = {
        "version": "c1_refined_manifest_v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_calls": 0, "final_decision": "unresolved", "artifacts": {name: _sha256(refined_dir / name) for name in artifact_names},
    }
    (refined_dir / "c1_refined_manifest_v1.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return comparison


def prepare_model_selection_review(
    output_dir: Path = OUTPUT_DIR, selection_dir: Path = MODEL_SELECTION_DIR,
) -> dict:
    """Select a compact, evidence-driven subset of refined topics for comparison."""
    selection_dir.mkdir(parents=True, exist_ok=True)
    refined_dir = output_dir / "c1_refined_v1"
    review = pd.read_csv(refined_dir / "topic_review_c1_refined_v1.csv", keep_default_na=False)
    crosswalk = pd.read_csv(refined_dir / "c1_to_c1_refined_crosswalk_with_human_labels_v1.csv", keep_default_na=False)
    old_review = pd.read_csv(output_dir / "topic_review_c1_v1.csv", keep_default_na=False)
    top_ten = set(review.nlargest(10, "topic_size").topic_id.astype(int))
    normal_crosswalk = crosswalk[crosswalk.refined_topic_id.ne(-1)].copy()
    dominant = normal_crosswalk.sort_values(
        ["refined_topic_id", "document_count"], ascending=[True, False],
    ).groupby("refined_topic_id").first()
    outlier_dominated = set(dominant[dominant.c1_topic_id.eq(-1)].index.astype(int))
    mixed_ids = set(old_review.loc[
        old_review.topic_relevance.eq("mixed_topic") | old_review.topic_quality.eq("highly_mixed"), "topic_id"
    ].astype(int))
    generic_ids = set(old_review.loc[old_review.topic_relevance.eq("generic_running"), "topic_id"].astype(int))
    substantial = normal_crosswalk[
        normal_crosswalk.percentage_of_refined_topic.ge(25) & normal_crosswalk.document_count.ge(10)
    ]
    mixed_contribution = set(substantial[substantial.c1_topic_id.isin(mixed_ids)].refined_topic_id.astype(int))
    generic_contribution = set(substantial[substantial.c1_topic_id.isin(generic_ids)].refined_topic_id.astype(int))

    direct_patterns: dict[int, list[str]] = {}
    stopwords = set(__import__("sklearn.feature_extraction.text", fromlist=["ENGLISH_STOP_WORDS"]).ENGLISH_STOP_WORDS) | MALAY_INDONESIAN_STOPWORDS
    for row in review.itertuples(index=False):
        terms = json.loads(row.top_terms); joined = " ".join(terms).casefold(); reasons = []
        if any(any(_is_emoji_style_token(token) for token in re.findall(r"(?u)\w+", term)) for term in terms):
            reasons.append("emoji_token_driven")
        if sum(term.casefold() in stopwords for term in terms) >= 3:
            reasons.append("english_malay_stopword_dominated")
        if any(keyword in joined for keyword in ("photo", "camera", "photograph", "gambar", "picture", "selfie")):
            reasons.append("fragmented_photography")
        if any(len(parts := term.split()) == 2 and parts[0] == parts[1] for term in terms):
            reasons.append("formatting_or_repetitive_hashtag")
        if reasons:
            direct_patterns[int(row.topic_id)] = reasons
    artifact_pattern = set(direct_patterns) | generic_contribution
    selected_ids = sorted(top_ten | outlier_dominated | mixed_contribution | artifact_pattern)

    rows = []
    for topic_id in selected_ids:
        topic = review[review.topic_id.eq(topic_id)].iloc[0]
        contributions = normal_crosswalk[normal_crosswalk.refined_topic_id.eq(topic_id)].sort_values("document_count", ascending=False)
        contribution_records = []
        for item in contributions.itertuples(index=False):
            contribution_records.append({
                "c1_topic_id": int(item.c1_topic_id), "document_count": int(item.document_count),
                "percentage_of_refined_topic": float(item.percentage_of_refined_topic),
                "human_topic_name": str(item.topic_name), "topic_relevance": str(item.topic_relevance),
                "topic_quality": str(item.topic_quality),
            })
        reasons = []
        if topic_id in top_ten: reasons.append("largest_10")
        if topic_id in outlier_dominated: reasons.append("dominated_by_c1_outliers")
        if topic_id in mixed_contribution: reasons.append("substantial_mixed_or_highly_mixed_c1_contribution")
        reasons.extend(direct_patterns.get(topic_id, []))
        if topic_id in generic_contribution: reasons.append("generic_running_incidental_klscm")
        outlier_row = contributions[contributions.c1_topic_id.eq(-1)]
        outlier_share = 0.0 if outlier_row.empty else float(outlier_row.iloc[0].percentage_of_refined_topic)
        rows.append({
            "refined_topic_id": topic_id, "topic_size": int(topic.topic_size),
            "corpus_percentage": float(topic.corpus_percentage),
            "selection_reasons": json.dumps(reasons), "top_terms": topic.top_terms,
            "representative_documents": topic.representative_documents,
            "random_documents": topic.random_documents,
            "c1_contributions": json.dumps(contribution_records, ensure_ascii=False),
            "c1_outlier_inheritance_percentage": outlier_share,
            "interpretability_change": "", "semantic_relation": "",
            "refined_topic_quality": "", "note": "", "reviewed_at": "",
        })
    compact = pd.DataFrame(rows)
    path = selection_dir / "model_selection_compact_review_v1.csv"
    if path.exists():
        existing = pd.read_csv(path, keep_default_na=False)
        annotation_cols = ["interpretability_change", "semantic_relation", "refined_topic_quality", "note", "reviewed_at"]
        if set(existing.refined_topic_id) != set(compact.refined_topic_id):
            raise ValueError("Existing compact-review selection does not align")
        compact = compact.drop(columns=annotation_cols).merge(
            existing[["refined_topic_id", *annotation_cols]], on="refined_topic_id", validate="one_to_one",
        )
    compact.to_csv(path, index=False, encoding="utf-8-sig")
    manifest = {
        "version": "model_selection_compact_review_v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_calls": 0, "selected_topics": len(compact), "selected_topic_ids": selected_ids,
        "selection_counts": {
            "largest_10": len(top_ten), "dominated_by_c1_outliers": len(outlier_dominated),
            "substantial_mixed_or_highly_mixed_contribution": len(mixed_contribution),
            "direct_artifact_pattern": len(set(direct_patterns)), "generic_running_incidental": len(generic_contribution),
        },
        "substantial_contribution_rule": "at least 25% of the refined topic and at least 10 documents",
        "artifact_detection": "direct refined top-term diagnostics plus substantial contribution from completed c1 generic-running topics",
        "review_csv_sha256": _sha256(path),
    }
    (selection_dir / "model_selection_compact_review_manifest_v1.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_model_selection_summary(path, selection_dir)
    return manifest


def write_model_selection_summary(
    review_path: Path | None = None, selection_dir: Path = MODEL_SELECTION_DIR,
) -> dict:
    review_path = review_path or selection_dir / "model_selection_compact_review_v1.csv"
    data = pd.read_csv(review_path, keep_default_na=False)
    complete = (
        data.interpretability_change.isin(INTERPRETABILITY_CHANGE_LABELS)
        & data.semantic_relation.isin(SEMANTIC_RELATION_LABELS)
        & data.refined_topic_quality.isin(TOPIC_QUALITY_LABELS)
    )
    total_docs = int(data.topic_size.sum())
    changes = {}
    for label in INTERPRETABILITY_CHANGE_LABELS:
        subset = data[data.interpretability_change.eq(label)]
        changes[label] = {"topics": len(subset), "documents": int(subset.topic_size.sum()), "document_share": float(subset.topic_size.sum() / total_docs) if total_docs else 0}
    relations = {label: int(data.semantic_relation.eq(label).sum()) for label in SEMANTIC_RELATION_LABELS}
    qualities = {label: int(data.refined_topic_quality.eq(label).sum()) for label in TOPIC_QUALITY_LABELS}
    improved_share = changes["improved"]["document_share"]
    worse_share = changes["worse"]["document_share"]
    useful_relation_share = float(data.semantic_relation.isin({"merges_related_topics", "splits_theme_usefully", "recovers_useful_outliers"}).mean())
    unrelated_share = float(data.semantic_relation.eq("mixes_unrelated_topics").mean())
    if not complete.all():
        recommendation = "pending_compact_review"
    elif improved_share >= .60 and improved_share - worse_share >= .30 and unrelated_share <= .10 and useful_relation_share >= .40:
        recommendation = "recommend_c1_refined"
    else:
        recommendation = "recommend_keep_frozen_c1"
    summary = {
        "version": "model_selection_comparison_final_v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "api_calls": 0,
        "reviewed_topics": int(complete.sum()), "selected_topics": len(data), "selected_documents_nonexclusive_scope": total_docs,
        "interpretability_change": changes, "semantic_relation": relations, "refined_topic_quality": qualities,
        "original_c1": {"coverage": .5850, "outliers": .4150, "topics": 40, "coherent": 24, "somewhat_mixed": 10, "highly_mixed": 6, "substantive_topics": 27, "substantive_clustered_documents": 7112},
        "c1_refined": {"coverage": .5262, "outliers": .4738, "topics": 44, "additional_outliers": 808},
        "decision_rule": {"improved_document_share_min": .60, "improved_minus_worse_min": .30, "mixes_unrelated_topic_share_max": .10, "useful_relation_topic_share_min": .40},
        "recommendation": recommendation, "absa_status": "blocked",
    }
    selection_dir.mkdir(parents=True, exist_ok=True)
    (selection_dir / "model_selection_summary_v1.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_model_selection_markdown(summary, selection_dir / "MODEL_SELECTION_COMPARISON_FINAL.md")
    if selection_dir.resolve() == MODEL_SELECTION_DIR.resolve():
        _write_model_selection_markdown(summary, ROOT / "MODEL_SELECTION_COMPARISON_FINAL.md")
    return summary


def _write_model_selection_markdown(summary: dict, path: Path) -> None:
    recommendation = summary["recommendation"].replace("_", " ")
    text = f"""# Final Model-Selection Comparison

## Status

Recommendation: **{recommendation}**. ABSA remains blocked. No model rerun, API call, or record-level annotation is part of this review.

The compact evidence-driven review contains {summary['selected_topics']} refined topics; {summary['reviewed_topics']} are complete.

## Fixed objective comparison

| Evidence | Frozen c1 | c1-refined |
|---|---:|---:|
| Clustered coverage | 58.50% | 52.62% |
| Outliers | 41.50% | 47.38% |
| Topics | 40 | 44 |
| Additional outliers | — | 808 |

Frozen c1 has a completed full review: 24 coherent, 10 somewhat mixed, six highly mixed; 27 substantive topics cover 7,112 clustered documents.

## Compact comparative review

Interpretability counts are improved {summary['interpretability_change']['improved']['topics']}, similar {summary['interpretability_change']['similar']['topics']}, and worse {summary['interpretability_change']['worse']['topics']}. Refined-topic quality counts are coherent {summary['refined_topic_quality']['coherent']}, somewhat mixed {summary['refined_topic_quality']['somewhat_mixed']}, and highly mixed {summary['refined_topic_quality']['highly_mixed']}.

## Conservative decision rule

c1-refined is recommended only after complete review when improved topics cover at least 60% of selected-topic documents, improved exceeds worse by at least 30 percentage points, at most 10% of selected topics mix unrelated themes, and at least 40% usefully merge, split, or recover outliers. Otherwise frozen c1 is recommended. The rule explicitly weighs interpretability against 808 additional outliers, lower coverage, and four additional topics; it never favors the newer model automatically.
"""
    path.write_text(text, encoding="utf-8")


def run_topic_discovery(output_dir: Path = OUTPUT_DIR) -> None:
    """Run the deliberately explicit, potentially slow local baseline comparison."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    corpus = pd.read_csv(output_dir / "topic_discovery_corpus_v1.csv", keep_default_na=False)
    fit = corpus[corpus.include_in_topic_discovery.astype(str).str.lower().isin({"true", "1"})].copy()
    ids, texts = fit.document_id.tolist(), fit.semantic_text_for_embeddings.tolist()
    cache_path, ids_path = output_dir / "embeddings_v1.npy", output_dir / "embedding_record_ids_v1.json"
    if cache_path.exists() and ids_path.exists():
        embeddings, cached_ids = np.load(cache_path), json.loads(ids_path.read_text(encoding="utf-8"))
        if cached_ids != ids:
            raise ValueError("Embedding cache IDs/order do not match the eligible corpus")
    else:
        model = SentenceTransformer(EMBEDDING_MODEL)
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        validate_embedding_cache(ids, embeddings)
        np.save(cache_path, embeddings)
        ids_path.write_text(json.dumps(ids), encoding="utf-8")
    metrics = []
    for candidate in CANDIDATES:
        umap = UMAP(n_neighbors=candidate["umap_n_neighbors"], n_components=5, min_dist=0.0, metric="cosine", random_state=42)
        clusterer = HDBSCAN(min_cluster_size=candidate["hdbscan_min_cluster_size"], min_samples=candidate["hdbscan_min_samples"], metric="euclidean", cluster_selection_method="eom", prediction_data=True)
        vectorizer = CountVectorizer(tokenizer=multilingual_tokenizer, token_pattern=None, ngram_range=(1, 2), min_df=2, max_features=50000)
        topic_model = BERTopic(umap_model=umap, hdbscan_model=clusterer, vectorizer_model=vectorizer, top_n_words=20, calculate_probabilities=False, verbose=True)
        topics, _ = topic_model.fit_transform(texts, embeddings)
        counts = Counter(topics); sizes = [n for topic, n in counts.items() if topic != -1]
        metrics.append({**candidate, "topic_count": len(sizes), "outlier_count": counts.get(-1, 0), "outlier_proportion": counts.get(-1, 0) / len(texts), "median_topic_size": float(np.median(sizes)) if sizes else 0, "largest_topic_size": max(sizes, default=0), "smallest_topic_size": min(sizes, default=0)})
        topic_model.save(output_dir / f"bertopic_{candidate['candidate_id']}_v1", serialization="pickle")
        pd.DataFrame({"document_id": ids, "topic_id": topics}).to_csv(output_dir / f"topic_assignments_{candidate['candidate_id']}_v1.csv", index=False)
        topic_model.get_topic_info().to_csv(output_dir / f"topic_info_{candidate['candidate_id']}_v1.csv", index=False)
    pd.DataFrame(metrics).to_csv(output_dir / "bertopic_candidate_metrics_v1.csv", index=False)
