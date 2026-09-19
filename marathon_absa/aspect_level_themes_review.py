from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ALTA_ROOT = Path("data/processed/absa_v1/aspect_level_themes_v1")
OUTPUT_ROOT = Path("data/processed/absa_v1/aspect_level_themes_review_v1")
METHODOLOGY_SOURCE = Path("ASPECT_LEVEL_THEMES_REVIEW.md")
SUMMARY_PATH = ALTA_ROOT / "theme_summary.parquet"
EVIDENCE_PATH = ALTA_ROOT / "theme_representative_evidence.parquet"
ASSIGNMENTS_PATH = ALTA_ROOT / "mention_theme_assignments.parquet"
ALTA_MANIFEST_PATH = ALTA_ROOT / "theme_manifest.json"
EXPECTED_CLUSTERS = 101
EXPECTED_ASPECTS = 20
EXPECTED_MENTIONS = 15486
EXPECTED_NOISE = 3065
EXPECTED_INSUFFICIENT = 154
YEARS = [2019, 2023, 2024, 2025]
DECISIONS = {"KEEP", "RENAME", "MERGE", "UNCLEAR_OTHER", "EXCLUDE_FROM_INTERPRETATION"}
QUALITIES = {"coherent", "somewhat_mixed", "highly_mixed"}
REVIEW_STATUSES = {"pending", "reviewed"}
PERCEPTION_STATUSES = {"DRAFT", "APPROVED", "REVISE", "DO_NOT_USE"}
INTERVIEW_RELEVANCE = {"high", "medium", "low"}
INTERVIEW_STATUSES = {"PENDING", "SHORTLIST", "HOLD", "EXCLUDE"}
LIMITATION = (
    "Reviewed aspect-level discussion themes remain downstream of model-estimated ABSA assignments "
    "and therefore inherit uncertainty from the upstream classifier (development precision about "
    "0.513 and recall about 0.790). Researcher interpretation does not validate the classifier."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_frame(frame: pd.DataFrame, stem: Path) -> dict[str, str]:
    csv_path, parquet_path = stem.with_suffix(".csv"), stem.with_suffix(".parquet")
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    frame.to_parquet(parquet_path, index=False)
    return {csv_path.name: sha256_file(csv_path), parquet_path.name: sha256_file(parquet_path)}


def load_frozen_alta() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    for path in (SUMMARY_PATH, EVIDENCE_PATH, ASSIGNMENTS_PATH, ALTA_MANIFEST_PATH):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest = json.loads(ALTA_MANIFEST_PATH.read_text(encoding="utf-8"))
    for name, expected_hash in manifest["output_hashes"].items():
        path = ALTA_ROOT / name
        if not path.exists() or sha256_file(path) != expected_hash:
            raise ValueError(f"Frozen ALTA v1 artifact hash mismatch: {name}")
    summary = pd.read_parquet(SUMMARY_PATH)
    evidence = pd.read_parquet(EVIDENCE_PATH)
    assignments = pd.read_parquet(ASSIGNMENTS_PATH)
    errors = []
    if len(summary) != EXPECTED_CLUSTERS or summary[["aspect", "theme_id"]].duplicated().any(): errors.append("101 unique clusters")
    if len(assignments) != EXPECTED_MENTIONS or assignments.mention_id.nunique() != EXPECTED_MENTIONS: errors.append("15,486 unique assignments")
    if assignments.aspect.nunique() != EXPECTED_ASPECTS: errors.append("20 frozen aspects")
    if int(assignments.theme_status.eq("noise").sum()) != EXPECTED_NOISE: errors.append("noise reconciliation")
    if int(assignments.theme_status.eq("insufficient_support").sum()) != EXPECTED_INSUFFICIENT: errors.append("insufficient-support reconciliation")
    if set(summary.theme_id.astype(int)) == {-1} or summary.theme_id.astype(int).lt(0).any(): errors.append("cluster theme IDs")
    if len(evidence) != EXPECTED_CLUSTERS * 5 or not evidence.mention_id.isin(assignments.mention_id).all(): errors.append("representative evidence lineage")
    if errors:
        raise ValueError("Frozen ALTA v1 reconciliation failed: " + ", ".join(errors))
    return summary, evidence, assignments, manifest


def _sentiment_shares(row: pd.Series) -> dict[str, float]:
    total = int(row.support_documents)
    return {s: (int(row[f"{s}_document_presence"]) / total if total else 0.0) for s in ["positive", "negative", "mixed", "neutral"]}


def build_review_workbook(summary: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    aspect_order = summary.groupby("aspect").support_documents.sum().sort_values(ascending=False).index.tolist()
    order = {aspect: index for index, aspect in enumerate(aspect_order)}
    rows = []
    for _, cluster in summary.iterrows():
        examples = evidence[(evidence.aspect == cluster.aspect) & (evidence.theme_id == cluster.theme_id)].sort_values("mention_id")
        shares = _sentiment_shares(cluster)
        row = {
            "aspect": cluster.aspect, "theme_id": int(cluster.theme_id),
            "cluster_key": cluster.provisional_theme_key, "provisional_label": cluster.provisional_label,
            "support_mentions": int(cluster.support_mentions), "support_documents": int(cluster.support_documents),
            "share_of_aspect_documents": float(cluster.share_of_aspect_documents),
            "positive_share": shares["positive"], "negative_share": shares["negative"],
            "mixed_share": shares["mixed"], "neutral_share": shares["neutral"],
            "years_present": cluster.years_present, "top_targets": cluster.representative_targets,
            "top_keyphrases": cluster.characteristic_keyphrases,
            "representative_evidence_ids": " | ".join(examples.mention_id.astype(str)),
            "theme_quality": "", "review_decision": "", "researcher_theme_label": "",
            "merge_target_theme_key": "", "researcher_notes": "", "review_status": "pending",
            "_aspect_order": order[cluster.aspect],
        }
        for number, (_, example) in enumerate(examples.head(5).iterrows(), 1):
            row[f"representative_evidence_{number}"] = example.evidence_text
            row[f"representative_gloss_{number}"] = example.english_gloss
            row[f"representative_document_id_{number}"] = example.document_id
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["_aspect_order", "support_documents", "theme_id"], ascending=[True, False, True]).drop(columns="_aspect_order")
    return frame.reset_index(drop=True)


def build_insufficient_support(assignments: pd.DataFrame) -> pd.DataFrame:
    subset = assignments[assignments.theme_status.eq("insufficient_support")]
    rows = []
    for aspect, group in subset.groupby("aspect"):
        examples = group.sort_values(["document_id", "mention_id"]).drop_duplicates("document_id").head(5)
        rows.append({
            "aspect": aspect, "support_mentions": len(group), "support_documents": group.document_id.nunique(),
            "representative_evidence_ids": " | ".join(examples.mention_id.astype(str)),
            "representative_evidence": " || ".join(examples.evidence_text.astype(str)),
            "status": "insufficient_support",
            "interpretation": "Stable within-aspect thematic decomposition was not supported under frozen ALTA v1 rules.",
        })
    return pd.DataFrame(rows).sort_values("support_mentions", ascending=False).reset_index(drop=True)


def build_noise_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    noise = assignments[assignments.theme_status.eq("noise")]
    aspect_total = assignments.groupby("aspect").size()
    rows = []
    for aspect, group in noise.groupby("aspect"):
        rows.append({"aspect": aspect, "noise_mentions": len(group), "noise_documents": group.document_id.nunique(), "noise_share_of_aspect_mentions": len(group) / int(aspect_total[aspect])})
    return pd.DataFrame(rows).sort_values("noise_mentions", ascending=False).reset_index(drop=True)


def _source_hashes() -> dict[str, str]:
    return {str(path): sha256_file(path) for path in sorted(ALTA_ROOT.iterdir()) if path.is_file()}


def generate_review_package(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    summary, evidence, assignments, manifest = load_frozen_alta()
    output_root.mkdir(parents=True, exist_ok=True)
    workbook = build_review_workbook(summary, evidence)
    mapping = workbook[["aspect", "theme_id", "cluster_key", "provisional_label", "theme_quality", "review_decision", "researcher_theme_label", "merge_target_theme_key", "researcher_notes", "review_status"]].copy()
    hashes = {}
    for name, frame in {"cluster_review_workbook": workbook, "cluster_review_mapping": mapping, "insufficient_support_summary": build_insufficient_support(assignments), "noise_summary": build_noise_summary(assignments)}.items():
        hashes.update(_write_frame(frame, output_root / name))
    readme = output_root / "README.md"
    readme.write_text(METHODOLOGY_SOURCE.read_text(encoding="utf-8"), encoding="utf-8")
    hashes[readme.name] = sha256_file(readme)
    review_manifest = {
        "stage": "aspect_level_themes_review_v1", "lifecycle_status": "review_package_generated_pending_researcher_review",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "alta_v1_input_artifact_hashes": _source_hashes(),
        "absa_source_identity": manifest["source_artifact_hashes"], "source_clusters": len(summary), "source_aspects": assignments.aspect.nunique(),
        "review_counts": {"reviewed": 0, **{decision: 0 for decision in sorted(DECISIONS)}},
        "final_reviewed_theme_count": 0, "aspect_reviewed_theme_counts": {}, "mapping_hash": hashes["cluster_review_mapping.parquet"],
        "output_hashes": hashes, "code_identity": {"module": str(Path(__file__)), "module_sha256": sha256_file(Path(__file__))},
        "network_calls": 0, "openai_calls": 0, "paid_inference": 0, "frozen_upstream_modified": False,
        "limitation": LIMITATION,
    }
    (output_root / "review_manifest.json").write_text(json.dumps(review_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"output_root": str(output_root), "source_clusters": len(summary), "review_rows": len(workbook), "representative_examples_per_cluster": 5, "reviewed": 0, "finalized": False}


def _load_mapping(path: Path | None = None) -> pd.DataFrame:
    path = path or OUTPUT_ROOT / "cluster_review_mapping.csv"
    if not path.exists(): raise FileNotFoundError(path)
    return pd.read_csv(path, keep_default_na=False, encoding="utf-8-sig")


def _resolve_roots(mapping: pd.DataFrame) -> dict[str, str]:
    by_key = mapping.set_index("cluster_key").to_dict("index")
    roots: dict[str, str] = {}
    for key, row in by_key.items():
        if row["review_decision"] != "MERGE":
            roots[key] = key
            continue
        seen = {key}; current = str(row["merge_target_theme_key"]).strip()
        while True:
            if current not in by_key: raise ValueError(f"Invalid merge target for {key}: {current}")
            if current in seen: raise ValueError(f"Merge cycle detected from {key}")
            seen.add(current)
            target = by_key[current]
            if target["aspect"] != row["aspect"]: raise ValueError(f"Cross-aspect merge is prohibited: {key} -> {current}")
            if target["review_decision"] != "MERGE":
                if target["review_decision"] in {"UNCLEAR_OTHER", "EXCLUDE_FROM_INTERPRETATION"}: raise ValueError(f"Merge target is not an interpretable theme: {current}")
                roots[key] = current; break
            current = str(target["merge_target_theme_key"]).strip()
    return roots


def validate_review_mapping(mapping: pd.DataFrame, require_complete: bool = False) -> dict[str, Any]:
    summary, _, _, _ = load_frozen_alta()
    required = {"aspect", "theme_id", "cluster_key", "provisional_label", "theme_quality", "review_decision", "researcher_theme_label", "merge_target_theme_key", "researcher_notes", "review_status"}
    if not required.issubset(mapping.columns): raise ValueError(f"Missing review columns: {sorted(required - set(mapping.columns))}")
    source = summary[["aspect", "theme_id", "provisional_theme_key"]].rename(columns={"provisional_theme_key": "cluster_key"})
    aligned = mapping.merge(source, on=["aspect", "theme_id", "cluster_key"], how="outer", indicator=True)
    if len(mapping) != EXPECTED_CLUSTERS or mapping[["aspect", "theme_id", "cluster_key"]].duplicated().any() or not aligned._merge.eq("both").all():
        raise ValueError("Review mapping does not map each frozen ALTA cluster exactly once")
    completed = mapping.review_status.eq("reviewed")
    for _, row in mapping[completed].iterrows():
        if row.theme_quality not in QUALITIES: raise ValueError(f"Invalid theme_quality for {row.cluster_key}")
        if row.review_decision not in DECISIONS: raise ValueError(f"Invalid review_decision for {row.cluster_key}")
        if row.review_decision == "RENAME" and not str(row.researcher_theme_label).strip(): raise ValueError(f"RENAME requires a label: {row.cluster_key}")
        if row.review_decision == "MERGE" and not str(row.merge_target_theme_key).strip(): raise ValueError(f"MERGE requires a target: {row.cluster_key}")
        if row.review_decision != "MERGE" and str(row.merge_target_theme_key).strip(): raise ValueError(f"Only MERGE may have a target: {row.cluster_key}")
    if not mapping.review_status.isin(REVIEW_STATUSES).all(): raise ValueError("Invalid review_status")
    if require_complete and not completed.all(): raise ValueError(f"Researcher review incomplete: {int((~completed).sum())} clusters remain pending")
    roots = _resolve_roots(mapping.copy()) if completed.any() else {}
    return {"source_clusters": len(mapping), "reviewed": int(completed.sum()), "pending": int((~completed).sum()), "complete": bool(completed.all()), "merge_roots_resolved": len(roots)}


def validate_review_package(mapping_path: Path | None = None) -> dict[str, Any]:
    return validate_review_mapping(_load_mapping(mapping_path), require_complete=False)


def _document_sentiment(group: pd.DataFrame) -> pd.Series:
    return group.groupby("document_id").sentiment.agg(lambda values: next(iter(set(values))) if len(set(values)) == 1 else "mixed")


def _summary_sentence(label: str, documents: int, sentiments: dict[str, int], targets: str, years: list[int]) -> str:
    total = sum(sentiments.values())
    leader, count = max(sentiments.items(), key=lambda item: item[1])
    if total and count / total >= 0.60:
        sentiment_text = f"The associated model-estimated document sentiment is predominantly {leader}."
    else:
        sentiment_text = "The associated model-estimated document sentiment includes more than one evaluation pattern."
    target_text = f" Representative targets include {targets}." if targets else ""
    return f"Online discussion within this reviewed theme concerns {label} and appears in {documents} unique documents across {', '.join(map(str, years))}. {sentiment_text}{target_text}"


def validate_researcher_layers(perceptions: pd.DataFrame, candidates: pd.DataFrame) -> None:
    if not perceptions.perception_status.isin(PERCEPTION_STATUSES).all():
        raise ValueError("Invalid perception_status")
    if not candidates.empty:
        if not candidates.interview_candidate_status.isin(INTERVIEW_STATUSES).all():
            raise ValueError("Invalid interview_candidate_status")
        relevance = candidates.interview_relevance.astype(str).str.strip()
        if not relevance.eq("").where(relevance.eq(""), relevance.isin(INTERVIEW_RELEVANCE)).all():
            raise ValueError("Invalid interview_relevance")


def finalize_reviewed_taxonomy(mapping_path: Path | None = None, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    mapping = _load_mapping(mapping_path)
    validation = validate_review_mapping(mapping, require_complete=True)
    summary, evidence, assignments, manifest = load_frozen_alta()
    roots = _resolve_roots(mapping)
    interpretable = mapping[~mapping.review_decision.isin(["UNCLEAR_OTHER", "EXCLUDE_FROM_INTERPRETATION"])].copy()
    interpretable["root_key"] = interpretable.cluster_key.map(roots)
    root_rows = mapping.set_index("cluster_key")
    root_keys = sorted(interpretable.root_key.unique(), key=lambda key: (root_rows.loc[key, "aspect"], int(root_rows.loc[key, "theme_id"])))
    reviewed_ids = {key: f"{root_rows.loc[key, 'aspect']}__rt{index:02d}" for aspect in sorted(set(root_rows.loc[k, "aspect"] for k in root_keys)) for index, key in enumerate([k for k in root_keys if root_rows.loc[k, "aspect"] == aspect], 1)}
    label_for = lambda key: str(root_rows.loc[key, "researcher_theme_label"]).strip() or str(root_rows.loc[key, "provisional_label"])
    cluster_to_reviewed = {row.cluster_key: reviewed_ids[row.root_key] for _, row in interpretable.iterrows()}
    cluster_key_lookup = summary.set_index(["aspect", "theme_id"]).provisional_theme_key.to_dict()
    reviewed_assignments = assignments.copy()
    reviewed_assignments["cluster_key"] = [cluster_key_lookup.get((a, int(t)), "") if int(t) >= 0 else "" for a, t in zip(assignments.aspect, assignments.theme_id)]
    reviewed_assignments["reviewed_theme_id"] = reviewed_assignments.cluster_key.map(cluster_to_reviewed).fillna("")
    reviewed_assignments["reviewed_theme_key"] = reviewed_assignments.reviewed_theme_id
    reviewed_assignments["reviewed_theme_label"] = reviewed_assignments.cluster_key.map({k: label_for(roots[k]) for k in cluster_to_reviewed}).fillna("")
    reviewed_assignments["review_interpretation_status"] = reviewed_assignments.cluster_key.map(mapping.set_index("cluster_key").review_decision).fillna(reviewed_assignments.theme_status)
    taxonomy_rows, summary_rows, year_rows, sentiment_rows, evidence_rows, perception_rows = [], [], [], [], [], []
    aspect_denoms = assignments.groupby("aspect").document_id.nunique()
    for reviewed_id, group in reviewed_assignments[reviewed_assignments.reviewed_theme_id.ne("")].groupby("reviewed_theme_id"):
        aspect = group.aspect.iloc[0]
        source_ids = sorted(group.theme_id.astype(int).unique().tolist())
        root_key = interpretable[interpretable.cluster_key.isin([cluster_key_lookup[(aspect, i)] for i in source_ids])].root_key.iloc[0]
        label = label_for(root_key)
        doc_sent = _document_sentiment(group)
        sent = {name: int((doc_sent == name).sum()) for name in ["positive", "negative", "mixed", "neutral"]}
        support_documents = group.document_id.nunique(); years = sorted(group.event_year.astype(int).unique())
        targets = " | ".join(x for x, _ in Counter(str(x).strip() for x in group.target if str(x).strip()).most_common(5))
        source_reviews = mapping[mapping.cluster_key.isin([cluster_key_lookup[(aspect, i)] for i in source_ids])]
        quality = "highly_mixed" if source_reviews.theme_quality.eq("highly_mixed").any() else "somewhat_mixed" if source_reviews.theme_quality.eq("somewhat_mixed").any() else "coherent"
        base = {"aspect": aspect, "reviewed_theme_id": reviewed_id, "reviewed_theme_key": reviewed_id, "reviewed_theme_label": label, "source_cluster_ids": json.dumps(source_ids), "support_mentions": len(group), "support_documents": support_documents, "share_of_aspect_documents": support_documents / int(aspect_denoms[aspect]), "theme_quality": quality, "review_status": "reviewed", "researcher_notes": " | ".join(x for x in source_reviews.researcher_notes.astype(str) if x.strip())}
        taxonomy_rows.append(base); summary_rows.append({**base, **{f"{k}_documents": v for k, v in sent.items()}, **{f"{k}_share": v / support_documents for k, v in sent.items()}, "years_present": " | ".join(map(str, years)), "top_targets": targets})
        for year in YEARS:
            yg = group[group.event_year.astype(int).eq(year)]; denom = assignments[(assignments.aspect == aspect) & assignments.event_year.astype(int).eq(year)].document_id.nunique()
            year_rows.append({"aspect": aspect, "reviewed_theme_id": reviewed_id, "year": year, "support_documents": yg.document_id.nunique(), "aspect_documents_that_year": denom, "within_aspect_theme_prevalence": yg.document_id.nunique() / denom if denom else 0.0})
        for sentiment, count in sent.items(): sentiment_rows.append({"aspect": aspect, "reviewed_theme_id": reviewed_id, "document_sentiment": sentiment, "support_documents": count, "share_of_theme_documents": count / support_documents})
        examples = group.sort_values(["document_id", "mention_id"]).drop_duplicates("document_id").head(5)
        evidence_ids = " | ".join(examples.mention_id.astype(str))
        for _, ev in examples.iterrows(): evidence_rows.append({"aspect": aspect, "reviewed_theme_id": reviewed_id, "reviewed_theme_label": label, "mention_id": ev.mention_id, "document_id": ev.document_id, "target": ev.target, "evidence_text": ev.evidence_text, "english_gloss": ev.english_gloss, "sentiment": ev.sentiment, "event_year": int(ev.event_year), "primary_language": ev.primary_language})
        perception_rows.append({"aspect": aspect, "reviewed_theme_id": reviewed_id, "reviewed_theme_label": label, "support_documents": support_documents, "sentiment_summary": json.dumps(sent, sort_keys=True), "representative_evidence_ids": evidence_ids, "machine_descriptive_summary": _summary_sentence(label, support_documents, sent, targets, years), "researcher_perception": "", "perception_status": "DRAFT", "perception_notes": ""})
    frames = {"cluster_review_mapping": mapping, "reviewed_theme_taxonomy": pd.DataFrame(taxonomy_rows), "reviewed_theme_assignments": reviewed_assignments, "reviewed_theme_summary": pd.DataFrame(summary_rows), "reviewed_theme_year_summary": pd.DataFrame(year_rows), "reviewed_theme_sentiment_summary": pd.DataFrame(sentiment_rows), "reviewed_theme_evidence": pd.DataFrame(evidence_rows), "reviewed_theme_perceptions": pd.DataFrame(perception_rows), "insufficient_support_summary": build_insufficient_support(assignments), "noise_summary": build_noise_summary(assignments)}
    # Preserve researcher perception edits on re-finalization when identities still align.
    existing_perceptions = output_root / "reviewed_theme_perceptions.csv"
    if existing_perceptions.exists():
        old = pd.read_csv(existing_perceptions, keep_default_na=False, encoding="utf-8-sig")
        if {"reviewed_theme_id", "researcher_perception", "perception_status", "perception_notes"}.issubset(old.columns):
            edits = old[["reviewed_theme_id", "researcher_perception", "perception_status", "perception_notes"]]
            frames["reviewed_theme_perceptions"] = frames["reviewed_theme_perceptions"].drop(columns=["researcher_perception", "perception_status", "perception_notes"]).merge(edits, on="reviewed_theme_id", how="left", validate="one_to_one").fillna({"researcher_perception": "", "perception_status": "DRAFT", "perception_notes": ""})
    perceptions = frames["reviewed_theme_perceptions"]
    if not perceptions.perception_status.isin(PERCEPTION_STATUSES).all(): raise ValueError("Invalid perception_status")
    approved = perceptions[(perceptions.perception_status == "APPROVED") & perceptions.researcher_perception.astype(str).str.strip().ne("")]
    candidates = []
    summary_by_id = frames["reviewed_theme_summary"].set_index("reviewed_theme_id")
    for index, (_, perception) in enumerate(approved.iterrows(), 1):
        info = summary_by_id.loc[perception.reviewed_theme_id]
        statement = f"Some previous KLSCM social-media posts described or referred to {perception.researcher_perception.rstrip('.').lower()}."
        candidates.append({"candidate_id": f"review_interview_{index:04d}", "aspect": perception.aspect, "reviewed_theme_id": perception.reviewed_theme_id, "reviewed_theme_label": perception.reviewed_theme_label, "support_documents": perception.support_documents, "share_within_aspect": info.share_of_aspect_documents, "sentiment_summary": perception.sentiment_summary, "years_present": info.years_present, "researcher_perception": perception.researcher_perception, "representative_evidence_ids": perception.representative_evidence_ids, "interview_relevance": "", "interview_candidate_status": "PENDING", "draft_interview_statement": statement, "draft_primary_question": "Based on your own KLSCM experience, do you agree, disagree, or partly agree with this? Why?", "draft_follow_up_1": "What experiences or circumstances shaped your view?", "draft_follow_up_2": "How, if at all, did this affect your overall KLSCM experience?", "researcher_final_statement": "", "researcher_final_question": "", "researcher_notes": ""})
    candidate_columns = ["candidate_id", "aspect", "reviewed_theme_id", "reviewed_theme_label", "support_documents", "share_within_aspect", "sentiment_summary", "years_present", "researcher_perception", "representative_evidence_ids", "interview_relevance", "interview_candidate_status", "draft_interview_statement", "draft_primary_question", "draft_follow_up_1", "draft_follow_up_2", "researcher_final_statement", "researcher_final_question", "researcher_notes"]
    frames["interview_proposition_candidates"] = pd.DataFrame(candidates, columns=candidate_columns)
    validate_researcher_layers(frames["reviewed_theme_perceptions"], frames["interview_proposition_candidates"])
    hashes = {}
    for name, frame in frames.items(): hashes.update(_write_frame(frame, output_root / name))
    counts = mapping.review_decision.value_counts().to_dict()
    review_manifest = {"stage": "aspect_level_themes_review_v1", "lifecycle_status": "reviewed_taxonomy_finalized", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "alta_v1_input_artifact_hashes": _source_hashes(), "absa_source_identity": manifest["source_artifact_hashes"], "source_clusters": len(summary), "number_reviewed": validation["reviewed"], **{f"number_{d.lower()}": int(counts.get(d, 0)) for d in DECISIONS}, "final_reviewed_theme_count": len(frames["reviewed_theme_taxonomy"]), "aspect_reviewed_theme_counts": frames["reviewed_theme_taxonomy"].groupby("aspect").size().to_dict(), "mapping_hash": hashes["cluster_review_mapping.parquet"], "output_hashes": hashes, "code_identity": {"module": str(Path(__file__)), "module_sha256": sha256_file(Path(__file__))}, "network_calls": 0, "openai_calls": 0, "paid_inference": 0, "frozen_upstream_modified": False, "limitation": LIMITATION}
    (output_root / "review_manifest.json").write_text(json.dumps(review_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"source_clusters": len(summary), "reviewed_themes": len(frames["reviewed_theme_taxonomy"]), "approved_perceptions": len(approved), "interview_candidates": len(candidates), "output_root": str(output_root)}
