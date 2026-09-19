from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from marathon_absa import aspect_level_themes_review as review
from marathon_absa import alta_review_ui as ui


def _copy_review_files(tmp_path: Path) -> Path:
    root = tmp_path / "aspect_level_themes_review_v1"
    root.mkdir()
    for name in [ui.WORKBOOK_NAME, ui.MAPPING_NAME]:
        (root / name).write_bytes((review.OUTPUT_ROOT / name).read_bytes())
    return root


def _values(**changes: str) -> dict[str, str]:
    values = {
        "theme_quality": "coherent",
        "review_decision": "KEEP",
        "researcher_theme_label": "",
        "merge_target_theme_key": "",
        "researcher_notes": "Evidence checked.",
    }
    values.update(changes)
    return values


def test_load_join_order_and_progress():
    workbook, mapping, joined = ui.load_review_data()
    assert len(workbook) == len(mapping) == len(joined) == 101
    assert joined.cluster_key.nunique() == 101
    aspects = ui.aspect_order(joined)
    totals = joined.groupby("aspect").support_documents.sum()
    assert totals.loc[aspects].tolist() == sorted(totals.tolist(), reverse=True)
    for aspect in aspects:
        supports = ui.clusters_for_aspect(joined, aspect).support_documents.tolist()
        assert supports == sorted(supports, reverse=True)
    summary = ui.progress_summary(mapping)
    expected_reviewed = int(mapping.review_status.eq("reviewed").sum())
    assert summary["total"] == 101
    assert summary["reviewed"] == expected_reviewed
    assert summary["pending"] == 101 - expected_reviewed


def test_decision_validation_and_merge_restrictions():
    _, mapping, joined = ui.load_review_data()
    key = mapping.iloc[0].cluster_key
    with pytest.raises(ValueError, match="RENAME requires"):
        ui.validate_row_values(mapping, key, _values(review_decision="RENAME"))
    same_aspect = ui.merge_options(joined, key)
    assert not same_aspect.empty
    assert same_aspect.aspect.eq(mapping.iloc[0].aspect).all()
    assert not same_aspect.cluster_key.eq(key).any()
    with pytest.raises(ValueError, match="itself"):
        ui.validate_row_values(
            mapping, key, _values(review_decision="MERGE", merge_target_theme_key=key)
        )
    cross = mapping.loc[mapping.aspect.ne(mapping.iloc[0].aspect)].iloc[0].cluster_key
    with pytest.raises(ValueError, match="same aspect"):
        ui.validate_row_values(
            mapping, key, _values(review_decision="MERGE", merge_target_theme_key=cross)
        )


def test_atomic_row_save_preserves_other_rows_utf8_and_creates_one_backup(tmp_path):
    root = _copy_review_files(tmp_path)
    path = root / ui.MAPPING_NAME
    before = pd.read_csv(path, keep_default_na=False, encoding="utf-8-sig")
    key = before.iloc[0].cluster_key
    updated, backup = ui.save_review_row(
        path, key,
        _values(researcher_theme_label="Hills – 山路", researcher_notes="Bukti asal dikekalkan."),
        review_root=root, create_backup=True,
    )
    assert backup is not None and backup.exists()
    assert updated.loc[updated.cluster_key.eq(key), "review_status"].iat[0] == "reviewed"
    assert updated.loc[updated.cluster_key.eq(key), "researcher_theme_label"].iat[0] == "Hills – 山路"
    untouched = before.cluster_key.ne(key)
    pd.testing.assert_frame_equal(
        updated.loc[untouched].reset_index(drop=True), before.loc[untouched].reset_index(drop=True)
    )
    reloaded = pd.read_csv(path, keep_default_na=False, encoding="utf-8-sig")
    assert "Hills – 山路" in reloaded.researcher_theme_label.tolist()


def test_write_path_safety_and_finalization_gate(tmp_path):
    root = _copy_review_files(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        ui.assert_review_path(tmp_path / "elsewhere.csv", root)
    mapping = pd.read_csv(root / ui.MAPPING_NAME, keep_default_na=False, encoding="utf-8-sig")
    mapping.loc[:, ui.EDITABLE_COLUMNS] = ["", "", "", "", "", "pending"]
    ready, message = ui.finalization_ready(mapping)
    assert not ready and "incomplete" in message


def test_ui_validation_reuses_review_validator(monkeypatch):
    _, mapping, _ = ui.load_review_data()
    called = {}

    def fake_validate(frame: pd.DataFrame, require_complete: bool = False):
        called["rows"] = len(frame)
        called["require_complete"] = require_complete
        return {"complete": False}

    monkeypatch.setattr(review, "validate_review_mapping", fake_validate)
    ui.validate_row_values(mapping, mapping.iloc[0].cluster_key, _values())
    assert called == {"rows": 101, "require_complete": False}


def test_frozen_alta_hashes_unchanged_by_ui_save(tmp_path):
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in review.ALTA_ROOT.iterdir() if path.is_file()
    }
    root = _copy_review_files(tmp_path)
    mapping = pd.read_csv(root / ui.MAPPING_NAME, keep_default_na=False, encoding="utf-8-sig")
    ui.save_review_row(
        root / ui.MAPPING_NAME, mapping.iloc[0].cluster_key, _values(), review_root=root
    )
    after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in review.ALTA_ROOT.iterdir() if path.is_file()
    }
    assert before == after
