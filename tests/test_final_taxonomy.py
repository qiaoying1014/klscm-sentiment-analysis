import hashlib
import inspect
import json

import pandas as pd
import pytest

import marathon_absa.final_taxonomy as module
from marathon_absa.final_taxonomy import (
    FINAL_SELECTION_DIR, TAXONOMY_ACTIONS, TAXONOMY_DIR,
    validate_taxonomy_actions,
)
from marathon_absa.topic_discovery import OUTPUT_DIR


def test_final_selection_freezes_c1_and_never_substitutes_refined():
    manifest = json.loads((FINAL_SELECTION_DIR / "final_topic_model_selection_v1.json").read_text(encoding="utf-8"))
    assert manifest["selected_model"] == "c1"
    assert manifest["status"] == "final_selected"
    assert manifest["c1_refined_status"] == "evaluated_not_selected"
    for name, expected in manifest["c1_artifact_hashes"].items():
        actual = hashlib.sha256((OUTPUT_DIR / name).read_bytes()).hexdigest()
        assert actual == expected


def test_taxonomy_worksheet_has_all_topics_and_preserves_human_annotations():
    worksheet = pd.read_csv(TAXONOMY_DIR / "final_taxonomy_worksheet_v1.csv", keep_default_na=False)
    review = pd.read_csv(OUTPUT_DIR / "topic_review_c1_v1.csv", keep_default_na=False)
    assert worksheet.topic_id.tolist() == list(range(40))
    assert worksheet.human_topic_name.tolist() == review.topic_name.tolist()
    assert worksheet.topic_relevance.tolist() == review.topic_relevance.tolist()
    assert worksheet.topic_quality.tolist() == review.topic_quality.tolist()
    assert worksheet.final_taxonomy_action.isin(TAXONOMY_ACTIONS).all()
    assert worksheet.final_taxonomy_action.eq("merge").sum() == 5


def _worksheet(actions=None, targets=None):
    actions = actions or {topic: "keep" for topic in range(40)}
    targets = targets or {}
    return pd.DataFrame({
        "topic_id": range(40),
        "final_taxonomy_action": [actions.get(topic, "keep") for topic in range(40)],
        "merge_target": [str(targets.get(topic, "")) for topic in range(40)],
    })


def test_merge_targets_validate_and_resolve():
    frame = _worksheet({1: "merge"}, {1: 2})
    roots = validate_taxonomy_actions(frame)
    assert roots[1] == 2 and roots[2] == 2
    invalid = _worksheet({1: "merge"}, {1: 99})
    with pytest.raises(ValueError, match="Invalid merge target"):
        validate_taxonomy_actions(invalid)


def test_circular_merges_are_rejected():
    frame = _worksheet({1: "merge", 2: "merge"}, {1: 2, 2: 1})
    with pytest.raises(ValueError, match="Circular merge"):
        validate_taxonomy_actions(frame)


def test_c1_assignments_preserve_every_eligible_id_and_outliers_separately():
    assignments = pd.read_csv(OUTPUT_DIR / "topic_assignments_c1_v1.csv")
    corpus = pd.read_csv(OUTPUT_DIR / "topic_discovery_corpus_v1.csv", usecols=["document_id", "include_in_topic_discovery"])
    eligible = corpus[corpus.include_in_topic_discovery]
    assert len(assignments) == 13743
    assert set(assignments.document_id) == set(eligible.document_id)
    assert assignments.topic_id.eq(-1).sum() == 5703
    assert set(assignments.loc[assignments.topic_id.ne(-1), "topic_id"]) == set(range(40))


def test_taxonomy_finalized_for_absa_without_api_code():
    preparation = json.loads((TAXONOMY_DIR / "final_taxonomy_preparation_manifest_v1.json").read_text(encoding="utf-8"))
    assert preparation["absa_ready"] is False
    final = json.loads((TAXONOMY_DIR / "final_topic_taxonomy_manifest_v1.json").read_text(encoding="utf-8"))
    assert final["ABSA_READY"] is True
    assert final["substantive_documents"] == 7704
    assert final["contextual_or_excluded_documents"] == 336
    assert final["outliers"] == 5703
    source = inspect.getsource(module)
    assert "CachedOpenAI" not in source and "openai_service" not in source
    assert set(TAXONOMY_ACTIONS) == {"keep", "merge", "exclude_from_substantive_taxonomy", "retain_as_contextual_topic"}
