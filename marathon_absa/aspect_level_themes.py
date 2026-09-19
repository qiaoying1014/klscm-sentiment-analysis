from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PRODUCTION_ROOT = Path("data/processed/absa_v1/production/absa_v1_production_v1")
MENTIONS_PATH = PRODUCTION_ROOT / "absa_v1_production_mentions_v1.csv"
DOCUMENTS_PATH = PRODUCTION_ROOT / "absa_v1_production_document_results_v1.csv"
CORPUS_PATH = Path("data/processed/topic_discovery_v1/final_taxonomy_v1/final_substantive_topic_corpus_v1.csv")
ONTOLOGY_PATH = Path("data/processed/absa_v1/absa_aspect_ontology_v1.json")
METHODOLOGY_SOURCE = Path("ASPECT_LEVEL_THEMATIC_ANALYSIS.md")
OUTPUT_ROOT = Path("data/processed/absa_v1/aspect_level_themes_v1")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EXPECTED_DOCUMENTS = 7704
EXPECTED_MENTIONS = 15486
EXPECTED_BEARING = 5316
EXPECTED_ZERO = 2388
EXPECTED_YEARS = {2019, 2023, 2024, 2025}
RANDOM_SEED = 42
MIN_ASPECT_MENTIONS = 60
MIN_ASPECT_DOCUMENTS = 40
LOW_SUPPORT_THEME_DOCUMENTS = 15
SENTIMENTS = ["positive", "negative", "mixed", "neutral"]
REPRESENTATION_STOPWORDS = sorted(set(
    "a an and are as at be been but by for from had has have he her hers him his i if in into is it its me my of on or our ours she so than that the their them they this to was we were will with you your "
    "aku anda apa adalah akan bagi dalam dan dari dengan dia di ini itu juga kami kita kepada kerana ke lebih mereka pada saya sebagai seperti sudah tak tidak untuk yang "
    "acara event klscm marathon race running run runner runners".split()
))
LIMITATION = (
    "Aspect-level themes are derived from model-estimated ABSA mentions. The thematic analysis does "
    "not remove uncertainty in upstream aspect classification (development aspect precision about "
    "0.513; recall about 0.790)."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\x00", " ")).strip()


def construct_theme_semantic_text(row: pd.Series | dict[str, Any]) -> str:
    """Deterministically join target, grounded evidence and separate English gloss."""
    parts: list[str] = []
    for field in ("target", "evidence_text", "english_gloss"):
        value = _clean(row.get(field, ""))
        if value and value.casefold() not in {item.casefold() for item in parts}:
            parts.append(value)
    return " | ".join(parts)


def load_and_reconcile() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    for path in (MENTIONS_PATH, DOCUMENTS_PATH, CORPUS_PATH, ONTOLOGY_PATH):
        if not path.exists():
            raise FileNotFoundError(path)
    mentions = pd.read_csv(MENTIONS_PATH, keep_default_na=False, low_memory=False)
    documents = pd.read_csv(DOCUMENTS_PATH, keep_default_na=False, low_memory=False)
    corpus = pd.read_csv(CORPUS_PATH, keep_default_na=False, low_memory=False)
    ontology_payload = json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8-sig"))
    raw_aspects = ontology_payload.get("aspects", ontology_payload)
    aspects = [x["aspect"] if isinstance(x, dict) and "aspect" in x else x.get("id") if isinstance(x, dict) else x for x in raw_aspects]
    aspects = [str(x) for x in aspects if x]
    errors = []
    if len(documents) != EXPECTED_DOCUMENTS or documents.document_id.nunique() != EXPECTED_DOCUMENTS:
        errors.append("document count/identity")
    if len(mentions) != EXPECTED_MENTIONS or mentions.mention_id.nunique() != EXPECTED_MENTIONS:
        errors.append("mention count/identity")
    if mentions.document_id.nunique() != EXPECTED_BEARING:
        errors.append("mention-bearing documents")
    if int(documents.mention_count.astype(int).eq(0).sum()) != EXPECTED_ZERO:
        errors.append("zero-mention documents")
    if set(mentions.event_year.astype(int)) != EXPECTED_YEARS or set(documents.event_year.astype(int)) != EXPECTED_YEARS:
        errors.append("event years")
    if set(mentions.aspect) != set(aspects) or len(aspects) != 20:
        errors.append("frozen 20-aspect ontology")
    if not set(mentions.document_id).issubset(set(documents.document_id)) or set(documents.document_id) != set(corpus.document_id):
        errors.append("document lineage")
    if errors:
        raise ValueError("Frozen production reconciliation failed: " + ", ".join(errors))
    original = corpus[["document_id", "original_caption_text", "hashtags"]].copy()
    mentions = mentions.merge(original, on="document_id", how="left", validate="many_to_one")
    mentions["theme_semantic_text"] = mentions.apply(construct_theme_semantic_text, axis=1)
    if mentions.theme_semantic_text.eq("").any():
        raise ValueError("Derived semantic text is empty")
    return mentions, documents, corpus, aspects


def adaptive_cluster_policy(n_mentions: int, n_documents: int) -> dict[str, Any]:
    sufficient = n_mentions >= MIN_ASPECT_MENTIONS and n_documents >= MIN_ASPECT_DOCUMENTS
    minimum = min(50, max(10, int(np.ceil(np.sqrt(n_mentions))))) if sufficient else None
    return {
        "sufficient_support": sufficient,
        "min_cluster_size": minimum,
        "min_samples": max(3, minimum // 3) if minimum else None,
        "umap_n_neighbors": min(15, n_mentions - 1) if sufficient else None,
    }


def embed_mentions(mentions: pd.DataFrame, output_root: Path, batch_size: int = 32) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    cache_path = output_root / "mention_embeddings_v1.npy"
    ids_path = output_root / "mention_embedding_ids_v1.json"
    ids = mentions.mention_id.astype(str).tolist()
    if cache_path.exists() and ids_path.exists():
        cached_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        vectors = np.load(cache_path)
        if cached_ids != ids or vectors.shape != (len(ids), 768):
            raise ValueError("ALTA embedding cache does not align with frozen mention IDs")
        return vectors
    model = SentenceTransformer(MODEL_NAME, local_files_only=True)
    vectors = model.encode(
        mentions.theme_semantic_text.tolist(), batch_size=batch_size,
        show_progress_bar=True, normalize_embeddings=True,
    )
    if vectors.shape != (len(mentions), 768) or not np.isfinite(vectors).all():
        raise ValueError("Unexpected ALTA embedding matrix")
    np.save(cache_path, vectors)
    ids_path.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    return vectors


def assign_themes(mentions: pd.DataFrame, vectors: np.ndarray, aspects: list[str]) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    from hdbscan import HDBSCAN
    from umap import UMAP

    assignments = mentions.copy()
    assignments["theme_id"] = -1
    assignments["theme_status"] = "noise"
    policies: dict[str, dict[str, Any]] = {}
    for aspect in aspects:
        positions = np.flatnonzero(assignments.aspect.to_numpy() == aspect)
        policy = adaptive_cluster_policy(len(positions), assignments.iloc[positions].document_id.nunique())
        policies[aspect] = policy
        if not policy["sufficient_support"]:
            assignments.loc[assignments.index[positions], "theme_status"] = "insufficient_support"
            continue
        reduced = UMAP(
            n_neighbors=policy["umap_n_neighbors"], n_components=5, min_dist=0.0,
            metric="cosine", random_state=RANDOM_SEED,
        ).fit_transform(vectors[positions])
        labels = HDBSCAN(
            min_cluster_size=policy["min_cluster_size"], min_samples=policy["min_samples"],
            metric="euclidean", cluster_selection_method="eom", prediction_data=False,
        ).fit_predict(reduced)
        idx = assignments.index[positions]
        assignments.loc[idx, "theme_id"] = labels.astype(int)
        assignments.loc[idx, "theme_status"] = np.where(labels == -1, "noise", "stable_cluster")
    return assignments, policies


def _terms(texts: list[str], top_n: int = 8) -> list[str]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not texts:
        return []
    fitted = TfidfVectorizer(
        token_pattern=r"(?u)\b[^\W_][\w'+-]*\b", ngram_range=(1, 2), min_df=2,
        max_features=20000, stop_words=REPRESENTATION_STOPWORDS,
    )
    try:
        matrix = fitted.fit_transform(texts)
    except ValueError:
        return []
    scores = np.asarray(matrix.mean(axis=0)).ravel()
    names = fitted.get_feature_names_out()
    return [str(names[i]) for i in scores.argsort()[::-1][:top_n] if scores[i] > 0]


def _representatives(group: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    ordered = group.assign(_length=group.evidence_text.astype(str).str.len()).sort_values(
        ["_length", "mention_id"], ascending=[False, True]
    )
    seen_docs: set[str] = set()
    seen_text: set[str] = set()
    rows = []
    for _, row in ordered.iterrows():
        normalized = re.sub(r"\W+", " ", str(row.evidence_text).casefold()).strip()
        if str(row.document_id) in seen_docs or normalized in seen_text:
            continue
        rows.append(row)
        seen_docs.add(str(row.document_id)); seen_text.add(normalized)
        if len(rows) == limit:
            break
    return pd.DataFrame(rows).drop(columns=["_length"], errors="ignore")


def build_outputs(assignments: pd.DataFrame, documents: pd.DataFrame) -> dict[str, pd.DataFrame]:
    clustered = assignments[assignments.theme_id.astype(int).ge(0)].copy()
    summary_rows, evidence_rows, year_rows, sentiment_rows = [], [], [], []
    aspect_docs = assignments.groupby("aspect").document_id.nunique().to_dict()
    for (aspect, theme_id), group in clustered.groupby(["aspect", "theme_id"], sort=True):
        doc_sent = group.groupby("document_id").sentiment.agg(lambda x: set(x))
        doc_class = doc_sent.map(lambda s: next(iter(s)) if len(s) == 1 else "mixed")
        terms = _terms(group.theme_semantic_text.tolist())
        targets = [x for x, _ in Counter(_clean(x) for x in group.target if _clean(x)).most_common(8)]
        label_parts = targets[:1] + [x for x in terms if x.casefold() not in {t.casefold() for t in targets[:1]}][:2]
        label = " / ".join(label_parts) if label_parts else f"{aspect} theme {int(theme_id)}"
        support_docs = int(group.document_id.nunique())
        composition = {s: int((doc_class == s).sum()) for s in SENTIMENTS}
        row = {
            "aspect": aspect, "theme_id": int(theme_id),
            "provisional_theme_key": f"{aspect}__theme_{int(theme_id):02d}",
            "provisional_label": label, "support_mentions": int(len(group)),
            "support_documents": support_docs,
            "share_of_aspect_documents": support_docs / aspect_docs[aspect],
            "share_of_all_7704_documents": support_docs / EXPECTED_DOCUMENTS,
            "representative_targets": " | ".join(targets),
            "characteristic_keyphrases": " | ".join(terms),
            "positive_document_presence": composition["positive"],
            "negative_document_presence": composition["negative"],
            "mixed_document_presence": composition["mixed"],
            "neutral_document_presence": composition["neutral"],
            "years_present": " | ".join(map(str, sorted(group.event_year.astype(int).unique()))),
            "language_composition": json.dumps(group.primary_language.value_counts().to_dict(), ensure_ascii=False, sort_keys=True),
            "cluster_status": "low_support_theme" if support_docs < LOW_SUPPORT_THEME_DOCUMENTS else "stable_cluster",
            "machine_descriptive_summary": f"Online discourse within this model-estimated {aspect} theme includes {label}; it appears in {support_docs} unique documents.",
        }
        summary_rows.append(row)
        for _, ev in _representatives(group).iterrows():
            evidence_rows.append({
                "aspect": aspect, "theme_id": int(theme_id), "provisional_label": label,
                "mention_id": ev.mention_id, "document_id": ev.document_id,
                "target": ev.target, "evidence_text": ev.evidence_text,
                "english_gloss": ev.english_gloss, "sentiment": ev.sentiment,
                "event_year": int(ev.event_year), "primary_language": ev.primary_language,
            })
        for year, yg in group.groupby("event_year"):
            yd = int(yg.document_id.nunique())
            denom = int(assignments[(assignments.aspect == aspect) & (assignments.event_year == year)].document_id.nunique())
            year_rows.append({"aspect": aspect, "theme_id": int(theme_id), "event_year": int(year), "support_mentions": len(yg), "support_documents": yd, "year_specific_within_aspect_prevalence": yd / denom if denom else 0.0})
        for sentiment in SENTIMENTS:
            sentiment_rows.append({"aspect": aspect, "theme_id": int(theme_id), "document_sentiment": sentiment, "support_documents": composition[sentiment], "share_of_theme_documents": composition[sentiment] / support_docs})
    summary = pd.DataFrame(summary_rows)
    review = summary[["aspect", "theme_id", "provisional_label"]].copy()
    review["researcher_label"] = ""; review["review_status"] = "pending"; review["review_notes"] = ""
    candidates = summary[summary.support_documents.ge(LOW_SUPPORT_THEME_DOCUMENTS)].copy()
    if not candidates.empty:
        candidates.insert(0, "candidate_id", [f"alta_candidate_{i:04d}" for i in range(1, len(candidates) + 1)])
        candidates["sentiment_composition"] = candidates.apply(lambda r: json.dumps({s: int(r[f"{s}_document_presence"]) for s in SENTIMENTS}, sort_keys=True), axis=1)
        candidates["representative_evidence_ids"] = candidates.apply(lambda r: " | ".join(pd.DataFrame(evidence_rows).query("aspect == @r.aspect and theme_id == @r.theme_id").mention_id.astype(str)), axis=1)
        candidates["interview_candidate_status"] = "candidate_pending_researcher_review"
        candidates["researcher_interview_statement"] = ""; candidates["researcher_notes"] = ""
        candidates = candidates[["candidate_id", "aspect", "theme_id", "provisional_label", "support_documents", "share_of_aspect_documents", "sentiment_composition", "years_present", "representative_evidence_ids", "machine_descriptive_summary", "interview_candidate_status", "researcher_interview_statement", "researcher_notes"]]
    return {"theme_summary": summary, "theme_year_summary": pd.DataFrame(year_rows), "theme_sentiment_summary": pd.DataFrame(sentiment_rows), "theme_representative_evidence": pd.DataFrame(evidence_rows), "theme_review": review, "interview_theme_candidates": candidates}


def _write_frame(frame: pd.DataFrame, stem: Path) -> dict[str, str]:
    csv_path, parquet_path = stem.with_suffix(".csv"), stem.with_suffix(".parquet")
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    frame.to_parquet(parquet_path, index=False)
    return {csv_path.name: sha256_file(csv_path), parquet_path.name: sha256_file(parquet_path)}


def create_aspect_level_themes(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    mentions, documents, corpus, aspects = load_and_reconcile()
    source_hashes = {str(p): sha256_file(p) for p in (MENTIONS_PATH, DOCUMENTS_PATH, CORPUS_PATH, ONTOLOGY_PATH)}
    vectors = embed_mentions(mentions, output_root)
    assignments, policies = assign_themes(mentions, vectors, aspects)
    outputs = build_outputs(assignments, documents)
    hashes = _write_frame(assignments, output_root / "mention_theme_assignments")
    for name, frame in outputs.items():
        hashes.update(_write_frame(frame, output_root / name))
    methodology_output = output_root / "README.md"
    methodology_output.write_text(METHODOLOGY_SOURCE.read_text(encoding="utf-8"), encoding="utf-8")
    hashes[methodology_output.name] = sha256_file(methodology_output)
    hashes["mention_embeddings_v1.npy"] = sha256_file(output_root / "mention_embeddings_v1.npy")
    hashes["mention_embedding_ids_v1.json"] = sha256_file(output_root / "mention_embedding_ids_v1.json")
    manifest = {
        "stage": "aspect_level_themes_v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analytical_status": "exploratory_descriptive_machine_induced_requires_researcher_interpretation",
        "source_artifact_hashes": source_hashes,
        "source_row_counts": {"documents": len(documents), "mentions": len(mentions), "mention_bearing_documents": mentions.document_id.nunique(), "zero_mention_documents": int(documents.mention_count.astype(int).eq(0).sum())},
        "source_schema": {"mentions": list(mentions.columns), "documents": list(documents.columns), "corpus": list(corpus.columns)},
        "semantic_text": "whitespace-normalized nonempty target + exact evidence_text + english_gloss, pipe-delimited with case-insensitive exact duplicates removed; source fields remain unchanged",
        "embedding": {"model": MODEL_NAME, "dimensions": 768, "local_files_only": True, "normalize_embeddings": True, "batch_size": 32, "cache": "mention_embeddings_v1.npy"},
        "clustering": {"scope": "separate within each frozen aspect", "dimension_reduction": "UMAP(n_components=5,min_dist=0,metric=cosine,random_state=42)", "method": "HDBSCAN(metric=euclidean,cluster_selection_method=eom)", "policy_by_aspect": policies},
        "thresholds": {"minimum_aspect_mentions": MIN_ASPECT_MENTIONS, "minimum_aspect_documents": MIN_ASPECT_DOCUMENTS, "low_support_theme_documents": LOW_SUPPORT_THEME_DOCUMENTS},
        "aspects": aspects, "years": sorted(EXPECTED_YEARS), "output_hashes": hashes,
        "code_identity": {"module": str(Path(__file__)), "module_sha256": sha256_file(Path(__file__))},
        "openai_calls": 0, "paid_api_calls": 0, "network_calls": 0,
        "upstream_artifacts_modified": False, "upstream_statement": "Frozen ABSA, topic, dashboard, inferential, and thesis artifacts were read only and were not modified.",
        "limitation": LIMITATION,
    }
    manifest_path = output_root / "theme_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"output_root": str(output_root), "mentions": len(assignments), "aspects": len(aspects), "themes": len(outputs["theme_summary"]), "noise_mentions": int(assignments.theme_status.eq("noise").sum()), "insufficient_support_mentions": int(assignments.theme_status.eq("insufficient_support").sum()), "manifest": str(manifest_path)}
