"""Final c1 selection and non-paid topic-taxonomy consolidation."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import ROOT
from .topic_discovery import OUTPUT_DIR

FINAL_SELECTION_DIR = OUTPUT_DIR / "final_selection"
TAXONOMY_DIR = OUTPUT_DIR / "final_taxonomy_v1"
TAXONOMY_ACTIONS = (
    "keep", "merge", "exclude_from_substantive_taxonomy",
    "retain_as_contextual_topic",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_final_c1_selection(output_dir: Path = OUTPUT_DIR) -> dict:
    frozen = json.loads((output_dir / "c1_frozen_baseline_manifest_v1.json").read_text(encoding="utf-8"))
    summary_path = output_dir / "model_selection_review_v1" / "model_selection_summary_v1.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["reviewed_topics"] != 28 or summary["recommendation"] != "recommend_keep_frozen_c1":
        raise ValueError("Completed compact review must recommend frozen c1")
    refined_comparison = output_dir / "c1_refined_v1" / "c1_vs_c1_refined_comparison_v1.json"
    compact_review = output_dir / "model_selection_review_v1" / "model_selection_compact_review_v1.csv"
    payload = {
        "version": "final_topic_model_selection_v1", "selected_model": "c1",
        "status": "final_selected", "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "c1_artifact_hashes": frozen["artifacts"],
        "c1_human_review_sha256": frozen["artifacts"]["topic_review_c1_v1.csv"],
        "refined_comparison_sha256": sha256(refined_comparison),
        "compact_review_sha256": sha256(compact_review),
        "compact_review_summary_sha256": sha256(summary_path),
        "objective_metrics": {
            "eligible_corpus": 13743, "normal_topics": 40,
            "normal_topic_documents": 8040, "coverage": 0.5850251036891508,
            "outliers": 5703, "outlier_proportion": 0.41497489631084916,
        },
        "human_review_metrics": frozen["completed_human_review_summary"],
        "compact_comparison_metrics": summary,
        "selection_reason": "c1-refined mixed unrelated themes in 6/28 reviewed topics (21.43%), exceeding the predeclared 10% ceiling, and reduced clustered coverage by 808 documents despite cleaner representation and substantial improvement in selected documents.",
        "c1_refined_status": "evaluated_not_selected", "api_calls": 0,
        "absa_ready": False, "absa_blocker": "final taxonomy decisions and mapping are incomplete",
    }
    FINAL_SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    path = FINAL_SELECTION_DIR / "final_topic_model_selection_v1.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable = {k: v for k, v in payload.items() if k != "selected_at_utc"}
        prior = {k: v for k, v in existing.items() if k != "selected_at_utc"}
        if comparable != prior:
            raise ValueError("Existing final-selection manifest conflicts with current evidence")
        return existing
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _parse_terms(value: str) -> list[str]:
    return [str(term) for term in json.loads(value)]


def build_merge_candidates(output_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    review = pd.read_csv(output_dir / "topic_review_c1_v1.csv", keep_default_na=False)
    assignments = pd.read_csv(output_dir / "topic_assignments_c1_v1.csv")
    ids = json.loads((output_dir / "embedding_record_ids_v1.json").read_text(encoding="utf-8"))
    embeddings = np.load(output_dir / "embeddings_v1.npy", mmap_mode="r")
    if len(ids) != embeddings.shape[0] or ids != assignments.document_id.tolist():
        raise ValueError("c1 embedding IDs and assignments do not align")
    centroids = {}
    labels = assignments.topic_id.to_numpy()
    for topic_id in range(40):
        centroid = np.asarray(embeddings[labels == topic_id]).mean(axis=0)
        centroids[topic_id] = centroid / np.linalg.norm(centroid)
    names = review.sort_values("topic_id").human_topic_name.tolist() if "human_topic_name" in review else review.sort_values("topic_id").topic_name.tolist()
    name_vectors = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(names)
    name_similarity = cosine_similarity(name_vectors)
    terms = {int(row.topic_id): set(_parse_terms(row.top_terms)) for row in review.itertuples(index=False)}
    known_pairs = {(11, 13): "completed review identifies overlapping race-photography themes"}
    rows = []
    by_topic = review.set_index("topic_id")
    for left in range(40):
        for right in range(left + 1, 40):
            centroid_score = float(np.dot(centroids[left], centroids[right]))
            union = terms[left] | terms[right]
            term_score = len(terms[left] & terms[right]) / len(union) if union else 0.0
            name_score = float(name_similarity[left, right])
            known = (left, right) in known_pairs
            score = min(1.0, .55 * centroid_score + .20 * term_score + .25 * name_score + (.15 if known else 0))
            rows.append({
                "topic_id_a": left, "topic_name_a": by_topic.loc[left, "topic_name"],
                "topic_id_b": right, "topic_name_b": by_topic.loc[right, "topic_name"],
                "embedding_centroid_cosine": centroid_score, "top_term_jaccard": term_score,
                "human_name_cosine": name_score, "known_review_overlap": known,
                "review_observation": known_pairs.get((left, right), ""), "candidate_score": score,
            })
    return pd.DataFrame(rows).sort_values(
        ["known_review_overlap", "candidate_score"], ascending=[False, False], ignore_index=True,
    )


def prepare_final_taxonomy(output_dir: Path = OUTPUT_DIR, taxonomy_dir: Path = TAXONOMY_DIR) -> dict:
    freeze_final_c1_selection(output_dir)
    taxonomy_dir.mkdir(parents=True, exist_ok=True)
    review = pd.read_csv(output_dir / "topic_review_c1_v1.csv", keep_default_na=False)
    if set(review.topic_id) != set(range(40)):
        raise ValueError("Completed c1 review must contain topics 0-39")
    candidates = build_merge_candidates(output_dir)
    candidates.to_csv(taxonomy_dir / "topic_merge_candidates_v1.csv", index=False, encoding="utf-8-sig")
    top_candidates = candidates.head(40)
    suggestion_map = {}
    for topic_id in range(40):
        relevant = top_candidates[(top_candidates.topic_id_a.eq(topic_id)) | (top_candidates.topic_id_b.eq(topic_id))].head(3)
        suggestion_map[topic_id] = json.dumps([
            {
                "other_topic_id": int(row.topic_id_b if row.topic_id_a == topic_id else row.topic_id_a),
                "other_topic_name": row.topic_name_b if row.topic_id_a == topic_id else row.topic_name_a,
                "score": round(float(row.candidate_score), 4),
                "known_review_overlap": bool(row.known_review_overlap),
            } for row in relevant.itertuples(index=False)
        ], ensure_ascii=False)
    substantive_ids = set(review.loc[review.topic_relevance.eq("substantive_klsm_topic"), "topic_id"])
    related_substantive = {}
    for topic_id in review.loc[review.topic_relevance.eq("mixed_topic"), "topic_id"]:
        related = candidates[
            ((candidates.topic_id_a.eq(topic_id)) & candidates.topic_id_b.isin(substantive_ids))
            | ((candidates.topic_id_b.eq(topic_id)) & candidates.topic_id_a.isin(substantive_ids))
        ].head(5)
        related_substantive[int(topic_id)] = json.dumps([
            int(row.topic_id_b if row.topic_id_a == topic_id else row.topic_id_a)
            for row in related.itertuples(index=False)
        ])
    worksheet = review.rename(columns={"topic_name": "human_topic_name"}).copy()
    worksheet["suggested_merge_candidates"] = worksheet.topic_id.map(suggestion_map)
    worksheet["possible_related_substantive_topics"] = worksheet.topic_id.map(related_substantive).fillna("[]")
    for column in ["final_taxonomy_action", "final_topic_name", "merge_target", "final_topic_group", "taxonomy_note"]:
        worksheet[column] = ""
    path = taxonomy_dir / "final_taxonomy_worksheet_v1.csv"
    if path.exists():
        existing = pd.read_csv(path, keep_default_na=False)
        fields = ["final_taxonomy_action", "final_topic_name", "merge_target", "final_topic_group", "taxonomy_note"]
        worksheet = worksheet.drop(columns=fields).merge(existing[["topic_id", *fields]], on="topic_id", validate="one_to_one")
    worksheet.to_csv(path, index=False, encoding="utf-8-sig")
    mixed = worksheet[worksheet.topic_relevance.eq("mixed_topic")][[
        "topic_id", "topic_size", "human_topic_name", "representative_documents",
        "random_documents", "relevance_distribution", "possible_related_substantive_topics",
    ]]
    mixed.to_csv(taxonomy_dir / "mixed_topic_diagnostics_v1.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "version": "final_taxonomy_preparation_v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": "c1", "topics_requiring_decisions": len(worksheet),
        "merge_pairs_ranked": len(candidates), "known_merge_candidate": [11, 13],
        "outliers": {"topic_id": -1, "status": "unassigned_outlier", "documents": 5703, "proportion": 0.41497489631084916},
        "api_calls": 0, "absa_ready": False,
    }
    (taxonomy_dir / "final_taxonomy_preparation_manifest_v1.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def validate_taxonomy_actions(worksheet: pd.DataFrame) -> dict[int, int]:
    if set(worksheet.topic_id.astype(int)) != set(range(40)) or len(worksheet) != 40:
        raise ValueError("Taxonomy worksheet must contain exactly topics 0-39")
    if not worksheet.final_taxonomy_action.isin(TAXONOMY_ACTIONS).all():
        raise ValueError("Every topic requires a valid final_taxonomy_action")
    actions = worksheet.set_index("topic_id").final_taxonomy_action.to_dict()
    targets = {}
    for row in worksheet.itertuples(index=False):
        if row.final_taxonomy_action == "merge":
            try:
                target = int(str(row.merge_target).strip())
            except ValueError as exc:
                raise ValueError(f"Topic {row.topic_id} requires a valid merge target") from exc
            if target not in actions or target == row.topic_id:
                raise ValueError(f"Invalid merge target for topic {row.topic_id}")
            targets[int(row.topic_id)] = target
    roots = {}
    for topic_id in range(40):
        seen, current = set(), topic_id
        while actions[current] == "merge":
            if current in seen:
                raise ValueError(f"Circular merge involving topic {current}")
            seen.add(current); current = targets[current]
        roots[topic_id] = current
    return roots


def finalize_taxonomy(output_dir: Path = OUTPUT_DIR, taxonomy_dir: Path = TAXONOMY_DIR) -> dict:
    selection = json.loads((FINAL_SELECTION_DIR / "final_topic_model_selection_v1.json").read_text(encoding="utf-8"))
    if selection["selected_model"] != "c1" or selection["status"] != "final_selected":
        raise ValueError("Frozen c1 final-selection manifest is required")
    worksheet_path = taxonomy_dir / "final_taxonomy_worksheet_v1.csv"
    worksheet = pd.read_csv(worksheet_path, keep_default_na=False)
    roots = validate_taxonomy_actions(worksheet)
    by_topic = worksheet.set_index("topic_id")
    mapping_rows = []
    for topic_id in range(40):
        root = roots[topic_id]; root_row = by_topic.loc[root]
        effective_name = root_row.final_topic_name.strip() or root_row.human_topic_name
        mapping_rows.append({
            "original_topic_id": topic_id, "final_topic_id": root,
            "original_human_topic_name": by_topic.loc[topic_id].human_topic_name,
            "final_topic_name": effective_name, "final_topic_group": root_row.final_topic_group,
            "final_taxonomy_action": root_row.final_taxonomy_action,
            "was_merged": topic_id != root, "merge_target": root if topic_id != root else "",
            "original_topic_relevance": by_topic.loc[topic_id].topic_relevance,
            "original_topic_quality": by_topic.loc[topic_id].topic_quality,
        })
    mapping = pd.DataFrame(mapping_rows)
    mapping.to_csv(taxonomy_dir / "final_topic_mapping_v1.csv", index=False, encoding="utf-8-sig")
    roots_frame = mapping[~mapping.was_merged].copy()
    roots_frame.to_csv(taxonomy_dir / "final_topic_taxonomy_v1.csv", index=False, encoding="utf-8-sig")
    assignments = pd.read_csv(output_dir / "topic_assignments_c1_v1.csv")
    corpus = pd.read_csv(output_dir / "topic_discovery_corpus_v1.csv", keep_default_na=False, low_memory=False)
    eligible = corpus[corpus.include_in_topic_discovery.astype(str).str.lower().isin({"true", "1"})]
    final_assignments = assignments.merge(mapping, left_on="topic_id", right_on="original_topic_id", how="left")
    outlier = final_assignments.topic_id.eq(-1)
    final_assignments.loc[outlier, ["final_topic_id", "final_topic_name", "final_topic_group", "final_taxonomy_action"]] = [-1, "unassigned_outlier", "", "unassigned_outlier"]
    if len(final_assignments) != 13743 or final_assignments.document_id.nunique() != 13743:
        raise ValueError("Final assignments lost or duplicated document IDs")
    final_assignments.to_csv(taxonomy_dir / "final_topic_assignments_v1.csv", index=False, encoding="utf-8-sig")
    enriched = final_assignments.merge(eligible, on="document_id", validate="one_to_one")
    substantive = enriched[enriched.final_taxonomy_action.eq("keep")]
    contextual = enriched[enriched.final_taxonomy_action.isin({"retain_as_contextual_topic", "exclude_from_substantive_taxonomy"})]
    outliers = enriched[enriched.final_taxonomy_action.eq("unassigned_outlier")]
    substantive.to_csv(taxonomy_dir / "final_substantive_topic_corpus_v1.csv", index=False, encoding="utf-8-sig")
    contextual.to_csv(taxonomy_dir / "final_contextual_or_excluded_topic_corpus_v1.csv", index=False, encoding="utf-8-sig")
    outliers.to_csv(taxonomy_dir / "final_outlier_corpus_v1.csv", index=False, encoding="utf-8-sig")
    if set(substantive.document_id) | set(contextual.document_id) | set(outliers.document_id) != set(eligible.document_id):
        raise ValueError("Final corpus partitions do not preserve every eligible document")
    counts = worksheet.topic_relevance.value_counts().to_dict()
    manifest = {
        "version": "final_topic_taxonomy_v1", "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": "c1", "raw_c1_topics": 40,
        "final_consolidated_substantive_topics": int(roots_frame.final_taxonomy_action.eq("keep").sum()),
        "merged_topics": int(mapping.was_merged.sum()),
        "contextual_topics": int(roots_frame.final_taxonomy_action.eq("retain_as_contextual_topic").sum()),
        "excluded_topics": int(roots_frame.final_taxonomy_action.eq("exclude_from_substantive_taxonomy").sum()),
        "original_review_categories": counts,
        "substantive_documents": len(substantive), "contextual_or_excluded_documents": len(contextual),
        "outliers": len(outliers), "outlier_status": "unassigned_outlier",
        "eligible_documents_reconciled": len(substantive) + len(contextual) + len(outliers),
        "api_calls": 0, "ABSA_READY": True,
        "artifacts": {},
    }
    artifact_names = ["final_topic_taxonomy_v1.csv", "final_topic_mapping_v1.csv", "final_topic_assignments_v1.csv", "final_substantive_topic_corpus_v1.csv", "final_contextual_or_excluded_topic_corpus_v1.csv", "final_outlier_corpus_v1.csv"]
    manifest["artifacts"] = {name: sha256(taxonomy_dir / name) for name in artifact_names}
    (taxonomy_dir / "final_topic_taxonomy_manifest_v1.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
