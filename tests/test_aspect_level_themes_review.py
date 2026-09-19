from __future__ import annotations

import hashlib
import inspect

import pandas as pd
import pytest

import marathon_absa.aspect_level_themes_review as review


def _hashes():
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in review.ALTA_ROOT.iterdir() if path.is_file()}


def _complete_mapping() -> pd.DataFrame:
    summary, _, _, _ = review.load_frozen_alta()
    return pd.DataFrame({
        "aspect": summary.aspect, "theme_id": summary.theme_id.astype(int),
        "cluster_key": summary.provisional_theme_key, "provisional_label": summary.provisional_label,
        "theme_quality": "coherent", "review_decision": "KEEP",
        "researcher_theme_label": "", "merge_target_theme_key": "",
        "researcher_notes": "", "review_status": "reviewed",
    })


def test_frozen_alta_reconciles_and_is_never_modified(tmp_path):
    before = _hashes()
    result = review.generate_review_package(tmp_path)
    after = _hashes()
    assert before == after
    assert result["source_clusters"] == result["review_rows"] == 101
    workbook = pd.read_parquet(tmp_path / "cluster_review_workbook.parquet")
    assert workbook[["aspect", "theme_id"]].drop_duplicates().shape[0] == 101
    assert workbook.review_status.eq("pending").all()
    assert all(workbook[f"representative_evidence_{i}"].astype(str).str.len().gt(0).all() for i in range(1, 6))


def test_pending_review_validates_but_cannot_finalize(tmp_path):
    review.generate_review_package(tmp_path)
    mapping = pd.read_csv(tmp_path / "cluster_review_mapping.csv", keep_default_na=False, encoding="utf-8-sig")
    status = review.validate_review_mapping(mapping)
    assert status == {"source_clusters": 101, "reviewed": 0, "pending": 101, "complete": False, "merge_roots_resolved": 0}
    with pytest.raises(ValueError, match="review incomplete"):
        review.validate_review_mapping(mapping, require_complete=True)


@pytest.mark.parametrize("field,value,message", [
    ("theme_quality", "automatic", "theme_quality"),
    ("review_decision", "SPLIT", "review_decision"),
    ("review_status", "done", "review_status"),
])
def test_review_vocabularies_are_enforced(field, value, message):
    mapping = _complete_mapping(); mapping.loc[0, field] = value
    with pytest.raises(ValueError, match=message): review.validate_review_mapping(mapping, require_complete=True)


def test_rename_requires_label_and_merge_requires_valid_target():
    mapping = _complete_mapping(); mapping.loc[0, "review_decision"] = "RENAME"
    with pytest.raises(ValueError, match="RENAME requires"): review.validate_review_mapping(mapping, True)
    mapping = _complete_mapping(); mapping.loc[0, "review_decision"] = "MERGE"; mapping.loc[0, "merge_target_theme_key"] = "missing"
    with pytest.raises(ValueError, match="Invalid merge target"): review.validate_review_mapping(mapping, True)


def test_cross_aspect_merge_self_merge_and_cycles_are_rejected():
    mapping = _complete_mapping()
    mapping.loc[0, "review_decision"] = "MERGE"; mapping.loc[0, "merge_target_theme_key"] = mapping.loc[1, "cluster_key"]
    if mapping.loc[0, "aspect"] == mapping.loc[1, "aspect"]:
        other = mapping.index[mapping.aspect.ne(mapping.loc[0, "aspect"])][0]; mapping.loc[0, "merge_target_theme_key"] = mapping.loc[other, "cluster_key"]
    with pytest.raises(ValueError, match="Cross-aspect"): review.validate_review_mapping(mapping, True)
    mapping = _complete_mapping(); mapping.loc[0, "review_decision"] = "MERGE"; mapping.loc[0, "merge_target_theme_key"] = mapping.loc[0, "cluster_key"]
    with pytest.raises(ValueError, match="cycle"): review.validate_review_mapping(mapping, True)
    mapping = _complete_mapping(); same = mapping.index[mapping.aspect.eq(mapping.loc[0, "aspect"])][:2]
    mapping.loc[same, "review_decision"] = "MERGE"; mapping.loc[same[0], "merge_target_theme_key"] = mapping.loc[same[1], "cluster_key"]; mapping.loc[same[1], "merge_target_theme_key"] = mapping.loc[same[0], "cluster_key"]
    with pytest.raises(ValueError, match="cycle"): review.validate_review_mapping(mapping, True)


def test_merge_finalization_deduplicates_documents_and_reaggregates_sentiment_year_and_lineage(tmp_path):
    mapping = _complete_mapping()
    same = mapping.index[mapping.aspect.eq("route_course")][:2]
    source_key, target_key = mapping.loc[same[0], "cluster_key"], mapping.loc[same[1], "cluster_key"]
    mapping.loc[same[0], "review_decision"] = "MERGE"; mapping.loc[same[0], "merge_target_theme_key"] = target_key
    path = tmp_path / "mapping.csv"; mapping.to_csv(path, index=False, encoding="utf-8-sig")
    result = review.finalize_reviewed_taxonomy(path, tmp_path)
    assignments = pd.read_parquet(review.ASSIGNMENTS_PATH)
    source_ids = mapping[mapping.cluster_key.isin([source_key, target_key])].theme_id.astype(int).tolist()
    raw = assignments[(assignments.aspect == "route_course") & assignments.theme_id.astype(int).isin(source_ids)]
    taxonomy = pd.read_parquet(tmp_path / "reviewed_theme_taxonomy.parquet")
    row = taxonomy[taxonomy.source_cluster_ids.eq(str(sorted(source_ids)).replace("'", '"'))]
    if row.empty: row = taxonomy[taxonomy.source_cluster_ids.map(lambda x: sorted(__import__('json').loads(x)) == sorted(source_ids))]
    assert len(row) == 1 and row.iloc[0].support_documents == raw.document_id.nunique() and row.iloc[0].support_mentions == len(raw)
    rid = row.iloc[0].reviewed_theme_id
    sentiment = pd.read_parquet(tmp_path / "reviewed_theme_sentiment_summary.parquet").query("reviewed_theme_id == @rid")
    assert sentiment.support_documents.sum() == raw.document_id.nunique()
    years = pd.read_parquet(tmp_path / "reviewed_theme_year_summary.parquet").query("reviewed_theme_id == @rid")
    assert set(years.year) == {2019, 2023, 2024, 2025}
    evidence = pd.read_parquet(tmp_path / "reviewed_theme_evidence.parquet")
    assert evidence.mention_id.isin(assignments.mention_id).all() and evidence.document_id.isin(assignments.document_id).all()
    assert result["source_clusters"] == 101


def test_insufficient_support_and_noise_are_preserved():
    _, _, assignments, _ = review.load_frozen_alta()
    insufficient = review.build_insufficient_support(assignments)
    noise = review.build_noise_summary(assignments)
    assert set(insufficient.aspect) == {"event_information", "race_pack_expo", "transport_access", "facilities"}
    assert insufficient.support_mentions.sum() == 154
    assert noise.noise_mentions.sum() == 3065


def test_perception_and_interview_status_validation():
    perceptions = pd.DataFrame({"perception_status": ["APPROVED"]})
    candidates = pd.DataFrame({"interview_candidate_status": ["PENDING"], "interview_relevance": [""]})
    review.validate_researcher_layers(perceptions, candidates)
    perceptions.loc[0, "perception_status"] = "VALIDATED"
    with pytest.raises(ValueError, match="perception_status"): review.validate_researcher_layers(perceptions, candidates)
    perceptions.loc[0, "perception_status"] = "DRAFT"; candidates.loc[0, "interview_candidate_status"] = "YES"
    with pytest.raises(ValueError, match="interview_candidate_status"): review.validate_researcher_layers(perceptions, candidates)


def test_manifest_generation_and_no_network_api_path(tmp_path):
    review.generate_review_package(tmp_path)
    manifest = __import__("json").loads((tmp_path / "review_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_clusters"] == 101 and manifest["network_calls"] == manifest["openai_calls"] == 0
    assert all(hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == value for name, value in manifest["output_hashes"].items())
    source = inspect.getsource(review).casefold()
    assert "import openai" not in source and "from openai" not in source and "requests." not in source and "urlopen(" not in source
