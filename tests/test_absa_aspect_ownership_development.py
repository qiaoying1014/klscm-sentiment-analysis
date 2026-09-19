import hashlib
import json
import shutil

import pandas as pd
import pytest

from marathon_absa.absa_aspect_ownership_development import (
    CACHE_STAGE, DEFAULT_MODEL, EXPERIMENT_ID, GOLD_PATH, INSTRUCTIONS,
    ONTOLOGY_PATH, OWNERSHIP_POLICY, PROMPT_VERSION, ROOT, SAMPLE_PATH,
    SCHEMA_VERSION, SUCCESS_GATES, V2_INSTRUCTIONS, V2_RESIDUAL,
    V2_ROOT, _validate, create_experiment,
)


def _requests(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_v3_package_changes_only_prompt_and_uses_same_ordered_80_documents(tmp_path):
    root = tmp_path / EXPERIMENT_ID
    package=create_experiment(root); spec=json.loads((root/"experiment_manifest_v1.json").read_text(encoding="utf-8"))
    v3=_requests(root/"batch"/"absa_v1_aspect_ownership_requests.jsonl")
    v2=_requests(V2_ROOT/"batch"/"absa_v1_precision_development_requests.jsonl")
    sample_ids=pd.read_csv(SAMPLE_PATH).document_id.astype(str).tolist()
    assert [x["custom_id"].removeprefix("absa1own-") for x in v3]==sample_ids
    assert [x["custom_id"].removeprefix("absa1p2-") for x in v2]==sample_ids
    assert len(v3)==len(v2)==package["request_count"]==80
    for old,new in zip(v2,v3):
        old_body={k:v for k,v in old["body"].items() if k!="instructions"}
        new_body={k:v for k,v in new["body"].items() if k!="instructions"}
        assert old_body==new_body
        assert new["body"]["model"]==DEFAULT_MODEL=="gpt-5.6-luna"
        assert new["body"]["text"]["format"]["name"]==SCHEMA_VERSION
    assert INSTRUCTIONS.startswith(V2_INSTRUCTIONS)
    assert INSTRUCTIONS==V2_INSTRUCTIONS+OWNERSHIP_POLICY
    assert spec["only_prompt_changed"] is True
    assert spec["postprocessing_added"] is False
    assert spec["code_deduplication_added"] is False
    assert spec["matching_evaluator_unchanged"] is True
    assert spec["inference_parameters_unchanged"] is True
    assert spec["success_gates"]==SUCCESS_GATES
    assert spec["candidate_prompt"]==PROMPT_VERSION
    assert spec["candidate_cache_stage"]==CACHE_STAGE
    assert spec["candidate_cache_stage"]!=spec["parent_cache_stage"]
    assert spec["v3_prompt_sha256"]!=spec["v2_prompt_sha256"]


def test_v3_frozen_hashes_and_residual_input_are_preserved():
    spec=json.loads((ROOT/"experiment_manifest_v1.json").read_text(encoding="utf-8"))
    hashes=spec["frozen_input_hashes"]
    assert hashes["gold_sha256"]==hashlib.sha256(GOLD_PATH.read_bytes()).hexdigest()
    assert hashes["ontology_sha256"]==hashlib.sha256(ONTOLOGY_PATH.read_bytes()).hexdigest()
    assert hashes["sample_sha256"]==hashlib.sha256(SAMPLE_PATH.read_bytes()).hexdigest()
    assert hashes["v2_residual_74_sha256"]==hashlib.sha256(V2_RESIDUAL.read_bytes()).hexdigest()
    residual=pd.read_csv(V2_RESIDUAL)
    assert len(residual)==74 and residual.residual_fp_id.nunique()==74
    assert spec["development_evidence"] is True and spec["confirmatory_evidence"] is False
    assert spec["full_corpus_prohibited"] is True and spec["candidate_count"]==1


def test_v3_validation_rejects_changed_gates(tmp_path):
    root=tmp_path/EXPERIMENT_ID;shutil.copytree(ROOT,root)
    path=root/"experiment_manifest_v1.json";spec=json.loads(path.read_text(encoding="utf-8"))
    spec["success_gates"]["aspect_precision_minimum"]=0.01
    path.write_text(json.dumps(spec),encoding="utf-8")
    with pytest.raises(RuntimeError,match="success gates"):_validate(root)


def test_v3_validation_accepts_platform_line_endings_and_rejects_text_changes(tmp_path):
    root=tmp_path/EXPERIMENT_ID;shutil.copytree(ROOT,root)
    prompt=root/"absa_v1_instructions_3_aspect_ownership.txt"
    text=prompt.read_text(encoding="utf-8")
    prompt.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    _validate(root)
    prompt.write_text(text+"changed",encoding="utf-8")
    with pytest.raises(RuntimeError,match="prompt hash changed"):_validate(root)
