import hashlib
import json
import shutil

import pandas as pd
import pytest

from marathon_absa.absa_precision_development import (
    ABSA_V1_INSTRUCTIONS, CANDIDATE_CACHE_STAGE, CANDIDATE_INSTRUCTIONS,
    CANDIDATE_PROMPT_VERSION, DEFAULT_MODEL, EXPERIMENT_ID, GOLD_PATH,
    ONTOLOGY_PATH, PROMPT_VERSION, ROOT, SAMPLE_PATH, SCHEMA_VERSION,
    SUCCESS_GATES, _validate_package, create_experiment,
)


def test_frozen_baseline_and_candidate_invariants(tmp_path):
    baseline = json.loads(
        (ROOT.parent.parent / "absa_validation_evaluation_v1.json").read_text(encoding="utf-8")
    )
    aspect = baseline["metrics"]["aspect_detection"]
    assert (aspect["tp"], aspect["fp"], aspect["fn"]) == (84, 134, 16)
    assert baseline["population"]["gold_mentions"] == 100
    assert baseline["population"]["predicted_mentions"] == 218

    root = tmp_path / EXPERIMENT_ID
    package = create_experiment(root)
    spec = json.loads((root / "experiment_manifest_v1.json").read_text(encoding="utf-8"))
    requests = [json.loads(line) for line in
                (root / "batch" / "absa_v1_precision_development_requests.jsonl").read_text(encoding="utf-8").splitlines()]
    sample_ids = set(pd.read_csv(SAMPLE_PATH).document_id.astype(str))
    request_ids = {row["custom_id"].removeprefix("absa1p2-") for row in requests}
    assert len(requests) == 80 and request_ids == sample_ids
    assert package["model"] == DEFAULT_MODEL == "gpt-5.6-luna"
    assert package["schema_version"] == SCHEMA_VERSION == "absa_mention_schema_v1"
    assert package["prompt_version"] == CANDIDATE_PROMPT_VERSION
    assert spec["baseline_prompt"] == PROMPT_VERSION
    assert spec["candidate_prompt_sha256"] != spec["baseline_prompt_sha256"]
    assert spec["candidate_cache_stage"] == CANDIDATE_CACHE_STAGE
    assert spec["candidate_cache_stage"] != spec["baseline_cache_stage"]
    assert spec["success_gates"] == SUCCESS_GATES
    assert spec["development_evidence"] is True
    assert spec["confirmatory_evidence"] is False
    assert spec["frozen_input_hashes"]["gold_sha256"] == hashlib.sha256(GOLD_PATH.read_bytes()).hexdigest()
    assert spec["frozen_input_hashes"]["ontology_sha256"] == hashlib.sha256(ONTOLOGY_PATH.read_bytes()).hexdigest()
    assert CANDIDATE_INSTRUCTIONS.startswith(ABSA_V1_INSTRUCTIONS)


def test_package_rejects_changed_success_gates(tmp_path):
    root = tmp_path / EXPERIMENT_ID
    shutil.copytree(ROOT, root)
    spec_path = root / "experiment_manifest_v1.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["success_gates"]["aspect_precision_minimum"] = 0.01
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(RuntimeError, match="success criteria"):
        _validate_package(root)


def test_candidate_directory_is_separate_from_frozen_outputs():
    assert ROOT.parent.name == "development"
    frozen_root = ROOT.parent.parent
    assert not ROOT.is_relative_to(frozen_root / "absa_v1_single_researcher_audit_v1")
    assert all(item["body"]["model"] == DEFAULT_MODEL for item in
               (json.loads(line) for line in
                (ROOT / "batch" / "absa_v1_precision_development_requests.jsonl").read_text(encoding="utf-8").splitlines()))
