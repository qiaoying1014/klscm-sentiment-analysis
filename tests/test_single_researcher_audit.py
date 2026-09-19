from __future__ import annotations

import json

import pandas as pd
import pytest

from marathon_absa.single_researcher_audit import (
    AUDIT_COLUMNS, CURRENT_AUDIT_BLIND, CURRENT_AUDIT_FINAL, CURRENT_AUDIT_KEY,
    atomic_save_annotations, build_architecture_audit_sample,
    build_audit_sample, calculate_architecture_metrics, calculate_metrics,
    finalize_audit, finalize_single_audit, validate_topic_unlock, write_architecture_audit_package,
    write_audit_package,
)


def candidates(n_each: int = 90) -> pd.DataFrame:
    rows = []
    for decision in ["include", "exclude"]:
        for i in range(n_each):
            rows.append({"document_id": f"{decision}-{i}", "original_text": f"KLSCM caption {i} with useful words", "source": "instagram", "event_year": str(2023+i%3), "primary_language": ["English", "Malay"][i%2], "language_status": "ok", "v8_automatic_decision": decision})
    rows += [{"document_id": "review-1", "original_text": "uncertain", "source": "instagram", "event_year": "2024", "primary_language": "English", "language_status": "ok", "v8_automatic_decision": "review"}]
    return pd.DataFrame(rows)


def test_sampling_is_reproducible_balanced_unique_and_blind():
    first, key1 = build_audit_sample(candidates(), 75, 104729)
    second, key2 = build_audit_sample(candidates(), 75, 104729)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(key1, key2)
    assert len(first) == first.document_id.nunique() == 150
    assert key1.automatic_decision.value_counts().to_dict() == {"include": 75, "exclude": 75}
    assert "review-1" not in set(first.document_id)
    assert list(first.columns) == AUDIT_COLUMNS
    assert not any(token in column.lower() for column in first for token in ["automatic", "prediction", "routing", "model", "gold"])


def test_sampling_rejects_insufficient_stratum():
    frame = candidates(80)
    remove = {f"exclude-{i}" for i in range(50, 80)}
    frame = frame[~frame.document_id.isin(remove)]
    with pytest.raises(ValueError, match="Insufficient exclude candidates"):
        build_audit_sample(frame, 75)


def test_architecture_audit_separates_deterministic_exclusions_from_includes():
    frame = candidates(90)
    frame["v8_routing_status"] = frame.v8_automatic_decision.map({"include": "automatic_include", "exclude": "automatic_exclude_deterministic", "review": "pending_review"})
    blind, key = build_architecture_audit_sample(frame, 75, 50)
    assert len(blind) == 125
    assert key.audit_arm.value_counts().to_dict() == {"automatic_include": 75, "deterministic_auto_exclude": 50}
    assert "audit_arm" not in blind
    merged = key.copy(); merged["researcher_label"] = merged.automatic_decision
    metrics = calculate_architecture_metrics(merged)
    assert metrics["automatic_include_audit"]["audited_n"] == 75
    assert metrics["deterministic_exclusion_audit"]["audited_n"] == 50
    assert "combined_automatic_audit" not in metrics


def completed_architecture_files(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    frame = candidates(90)
    frame["v8_routing_status"] = frame.v8_automatic_decision.map({"include": "automatic_include", "exclude": "automatic_exclude_deterministic", "review": "pending_review"})
    blind, key = build_architecture_audit_sample(frame, 75, 50)
    labels = key.set_index("audit_id").automatic_decision
    blind["researcher_label"] = blind.audit_id.map(labels)
    blind["rationale"] = "completed rationale"
    blind["evidence_span"] = "caption evidence"
    blind["confidence"] = "0.90"
    blind_path, key_path = tmp_path / "blind_v2.csv", tmp_path / "key_v2.csv"
    blind.to_csv(blind_path, index=False); key.to_csv(key_path, index=False)
    return blind_path, key_path


def test_current_defaults_resolve_architecture_v2():
    assert CURRENT_AUDIT_BLIND.name == "relevance_v8_architecture_audit_blind_v2.csv"
    assert CURRENT_AUDIT_KEY.name == "relevance_v8_architecture_audit_key_v2.csv"
    assert CURRENT_AUDIT_FINAL.name == "final_v2"


def test_v2_finalization_never_overwrites_annotations_and_v1_stays_supported(tmp_path):
    blind_path, key_path = completed_architecture_files(tmp_path)
    before = blind_path.read_bytes()
    metrics = finalize_single_audit(blind_path, key_path, tmp_path / "final_v2")
    assert blind_path.read_bytes() == before
    assert metrics["audit_protocol_version"] == "architecture_aligned_audit_v2"
    (tmp_path / "v1").mkdir()
    _, _, v1_blind, v1_key = completed_files(tmp_path / "v1")
    v1_metrics = finalize_single_audit(v1_blind, v1_key, tmp_path / "final_v1")
    assert "combined_automatic_audit" in v1_metrics


def test_v2_enforces_75_50_and_hidden_key_alignment(tmp_path):
    blind_path, key_path = completed_architecture_files(tmp_path)
    key = pd.read_csv(key_path, dtype=str)
    key.iloc[:-1].to_csv(key_path, index=False)
    with pytest.raises(ValueError, match="align"):
        finalize_single_audit(blind_path, key_path, tmp_path / "bad_alignment")
    blind_path, key_path = completed_architecture_files(tmp_path / "counts")
    key = pd.read_csv(key_path, dtype=str)
    key.loc[key.audit_arm.eq("deterministic_auto_exclude").idxmax(), "audit_arm"] = "automatic_include"
    key.to_csv(key_path, index=False)
    with pytest.raises(ValueError, match="exactly 75"):
        finalize_single_audit(blind_path, key_path, tmp_path / "bad_counts")


def completed_files(tmp_path):
    blind, key = build_audit_sample(candidates(4), 3)
    blind["researcher_label"] = blind.document_id.str.split("-").str[0]
    blind["rationale"] = "reason"
    blind["evidence_span"] = "evidence"
    blind["confidence"] = "0.90"
    blind_path, key_path = tmp_path / "blind.csv", tmp_path / "key.csv"
    blind.to_csv(blind_path, index=False); key.to_csv(key_path, index=False)
    return blind, key, blind_path, key_path


def test_finalizer_blocks_incomplete_invalid_mutation_and_overwrite(tmp_path):
    blind, key, blind_path, key_path = completed_files(tmp_path)
    broken = blind.copy(); broken.loc[0, "researcher_label"] = ""
    broken.to_csv(blind_path, index=False)
    with pytest.raises(ValueError, match="Complete all"):
        finalize_audit(blind_path, key_path, tmp_path / "out")
    broken.loc[0, "researcher_label"] = "maybe"; broken.to_csv(blind_path, index=False)
    with pytest.raises(ValueError, match="Complete all"):
        finalize_audit(blind_path, key_path, tmp_path / "out")
    blind.loc[0, "original_text"] = "changed"; blind.to_csv(blind_path, index=False)
    with pytest.raises(ValueError, match="metadata was changed"):
        finalize_audit(blind_path, key_path, tmp_path / "out")
    blind, key, blind_path, key_path = completed_files(tmp_path)
    metrics = finalize_audit(blind_path, key_path, tmp_path / "out")
    assert metrics["acceptance"]["audit_acceptable"]
    with pytest.raises(FileExistsError):
        finalize_audit(blind_path, key_path, tmp_path / "out")


def test_metrics_and_false_exclusion_acceptance():
    _, key = build_audit_sample(candidates(4), 3)
    merged = key.copy()
    merged["researcher_label"] = merged.automatic_decision
    merged.loc[merged.automatic_decision.eq("exclude").idxmax(), "researcher_label"] = "include"
    result = calculate_metrics(merged)
    assert result["automatic_include_audit"]["confirmation_rate"] == 1
    assert result["automatic_exclude_audit"]["false_exclusions"] == 1
    assert not result["acceptance"]["audit_acceptable"]
    assert result["acceptance"]["false_exclusion_safety_status"] == "requires_methodological_review"


def test_package_refuses_overwrite_and_records_no_api(tmp_path):
    source = tmp_path / "decisions.csv"; candidates().to_csv(source, index=False)
    package = tmp_path / "package"
    manifest = write_audit_package(source, package)
    assert manifest["api_calls_occurred"] is False
    assert json.loads((package / "manifest_v1.json").read_text())["random_seed"] == 104729
    with pytest.raises(FileExistsError):
        write_audit_package(source, package)


def test_architecture_package_is_versioned_and_keeps_review_separate(tmp_path):
    frame = candidates(90)
    frame["v8_routing_status"] = frame.v8_automatic_decision.map({"include": "automatic_include", "exclude": "automatic_exclude_deterministic", "review": "pending_review"})
    source = tmp_path / "decisions.csv"; frame.to_csv(source, index=False)
    manifest = write_architecture_audit_package(source, tmp_path / "v2")
    assert manifest["automatic_include_allocation"] == 75
    assert manifest["deterministic_auto_exclude_allocation"] == 50
    assert manifest["operational_review_records"] == 1
    assert manifest["api_calls_occurred"] is False


def test_atomic_save_and_topic_gate(tmp_path):
    path = tmp_path / "annotations.csv"
    frame = pd.DataFrame([{"x": "before"}]); frame.to_csv(path, index=False)
    atomic_save_annotations(path, pd.DataFrame([{"x": "after"}]))
    assert pd.read_csv(path).loc[0, "x"] == "after"
    decisions = candidates(2)
    review = pd.DataFrame([{"document_id": "review-1", "researcher_label": ""}])
    passing = {"acceptance": {"audit_acceptable": True}}
    with pytest.raises(RuntimeError, match="unresolved"):
        validate_topic_unlock(decisions, review, passing)
    review["researcher_label"] = "include"
    validate_topic_unlock(decisions, review, passing)
    with pytest.raises(RuntimeError, match="acceptance"):
        validate_topic_unlock(decisions, review, {"acceptance": {"audit_acceptable": False}})
