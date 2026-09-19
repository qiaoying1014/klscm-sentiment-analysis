"""Build a versioned rt08 evidence-selection correction without altering v1."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

from . import participant_experience as workflow
from . import participant_experience_review as review
from .participant_experience_dashboard_data import (
    SENTIMENTS, digest, read, sha, validate_frozen_references,
    validate_insight, validate_package,
)


TARGET = "training_preparation_pacing__rt08"
V1_ROOT = Path("data/processed/participant_experience_v1")
VERSIONED_ROOT = Path("data/processed/participant_experience_v2_rt08")
CORRECTION_VERSION = "participant_experience_v2_rt08"
REVIEWER = "AI-assisted researcher review (ChatGPT)"
CORRECTION_IDS = (
    "absa1_96e46ec812ce3f13",
    "absa1_f0699a167d078931",
    "absa1_1e8a1799b763707b",
    "absa1_2f3d510797e8b845",
)


def _now() -> str:
    return workflow.now()


def _records() -> list[dict]:
    path = Path("data/processed/absa_v1/aspect_level_themes_review_v1/reviewed_theme_assignments.csv")
    import csv
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["reviewed_theme_id"] == TARGET and row["mention_id"] in CORRECTION_IDS]
    found = {row["mention_id"] for row in rows}
    if found != set(CORRECTION_IDS):
        raise ValueError("rt08 correction lineage is incomplete")
    rows.sort(key=lambda row: CORRECTION_IDS.index(row["mention_id"]))
    return rows


def _evidence(row: dict) -> dict:
    return {
        "evidence_id": "instagram:" + row["mention_id"],
        "source": "instagram",
        "source_evidence_id": row["mention_id"],
        "parent_id": row["document_id"],
        "text": row["evidence_text"],
        "english_gloss": row["english_gloss"],
        "sentiment": row["sentiment"],
    }


def corrected_candidate() -> dict:
    """Targeted in-session AI-assisted candidate; no external API response is claimed."""
    work_disrupted = "instagram:absa1_96e46ec812ce3f13"
    working_life = "instagram:absa1_2f3d510797e8b845"
    schedule_plan = "instagram:absa1_1e8a1799b763707b"
    return {
        "theme_id": TARGET,
        "claims": {
            "participant_experience_summary": {
                "text": "The supplied Instagram excerpts describe training plans being adjusted around hectic work schedules and, for other participants, work or a new working lifestyle disrupting training.",
                "evidence_ids": [schedule_plan, work_disrupted, working_life],
            },
            "positive_experience_summary": {
                "text": "One participant described a training plan that worked around a hectic work schedule.",
                "evidence_ids": [schedule_plan],
            },
            "negative_experience_summary": {
                "text": "Other participants said work had greatly affected a training plan, that juggling work and training was not easy, or that adjustment to a new working lifestyle had affected training.",
                "evidence_ids": [work_disrupted, working_life],
            },
            "mixed_experience_summary": {
                "text": "The supplied examples include both a training plan that accommodated a hectic work schedule and accounts of work or a new working lifestyle affecting training. They do not establish how common either experience is.",
                "evidence_ids": [schedule_plan, work_disrupted, working_life],
            },
            "organizer_insight": {
                "text": "These examples show that work schedules and working-life adjustments can shape individual participants' training plans; they do not establish a shared operational problem or a population-level effect.",
                "evidence_ids": [schedule_plan, work_disrupted, working_life],
            },
            "organizer_implication": {
                "text": "Organizers may consider inviting feedback on whether training-planning information accommodates different work schedules, while recognising that these excerpts do not establish a specific service failure or remedy.",
                "evidence_ids": [schedule_plan, work_disrupted, working_life],
            },
            "evidence_scope_note": {
                "text": "This is a corrected candidate interpretation based on four legitimate Instagram records from rt08's existing frozen lineage, not all 45 supporting documents. The excerpts illustrate individual work or working-lifestyle contexts; there is no blog support, which does not imply disagreement.",
                "evidence_ids": [work_disrupted, working_life, schedule_plan],
            },
        },
    }


def _request_line(theme: dict, package: dict, model: str) -> str:
    payload = {"source_units": package["source_units"], "theme": theme,
               "model_limitation": package["model_limitation"]}
    payload = json.loads(json.dumps(payload))
    for evidence in payload["theme"]["representative_evidence"]:
        evidence.pop("parent_id")
        evidence.pop("source_evidence_id")
    return json.dumps({
        "custom_id": TARGET, "method": "POST", "url": "/v1/responses",
        "body": {"model": model, "instructions": workflow.PROMPT,
                 "input": json.dumps(payload, ensure_ascii=False), "max_output_tokens": 3000,
                 "text": {"format": {"type": "json_schema", "name": "participant_experience",
                 "strict": True, "schema": workflow.response_schema()}}},
    }, ensure_ascii=False)


def _claim_reviews() -> dict:
    return {field: {"state": "ACCEPT", "note": "ACCEPT: corrected rt08 evidence directly supports this cautious claim."}
            for field in corrected_candidate()["claims"]}


def _load_v2_evidence(root: Path) -> dict:
    """Retain v1 frozen checks while allowing only the documented rt08 correction IDs."""
    root = Path(root)
    package = read(root / "evidence.json")
    manifest = read(root / "evidence_manifest.json")
    if manifest["evidence_sha256"] != sha(root / "evidence.json") or manifest["input_hashes"] != package["input_hashes"]:
        raise ValueError("Versioned evidence manifest mismatch")
    validate_package(package)
    if any(sha(Path(name)) != expected for name, expected in package["input_hashes"].items()):
        raise ValueError("Frozen input changed")
    target = next(theme for theme in package["themes"] if theme["theme_id"] == TARGET)
    expected_ids = ["instagram:" + identity for identity in CORRECTION_IDS]
    if target.get("correction", {}).get("correction_version"):
        raise ValueError("Unexpected theme-level correction metadata")
    if package.get("correction", {}).get("correction_version") != CORRECTION_VERSION:
        raise ValueError("Missing v2 correction provenance")
    if [item["evidence_id"] for item in target["representative_evidence"]] != expected_ids:
        raise ValueError("Wrong rt08 correction evidence")
    # Run the original exhaustive frozen-reference validator with the v1 rt08
    # representative subset, then separately verify each v2 replacement against
    # the immutable full reviewed assignment lineage.
    v1_target = next(theme for theme in read(V1_ROOT / "evidence.json")["themes"] if theme["theme_id"] == TARGET)
    v1_shape = copy.deepcopy(package)
    v1_shape["themes"] = [v1_target if theme["theme_id"] == TARGET else theme for theme in package["themes"]]
    validate_frozen_references(v1_shape)
    records = {row["mention_id"]: row for row in _records()}
    for evidence in target["representative_evidence"]:
        record = records[evidence["source_evidence_id"]]
        if (evidence["source"] != "instagram" or evidence["parent_id"] != record["document_id"]
                or evidence["text"] != record["evidence_text"] or evidence["english_gloss"] != record["english_gloss"]
                or evidence["sentiment"] != record["sentiment"]):
            raise ValueError("rt08 corrected evidence differs from fixed lineage")
    return package


def _verify_requests(root: Path) -> tuple[dict, dict]:
    package = _load_v2_evidence(root)
    manifest = read(Path(root) / "request_manifest.json")
    if (manifest["input_artifact_hash"] != sha(Path(root) / "evidence.json")
            or manifest["requests_hash"] != sha(Path(root) / "requests.jsonl")
            or manifest["prompt_hash"] != digest(workflow.PROMPT)
            or manifest.get("correction", {}).get("regenerated_request_lines") != [TARGET]):
        raise ValueError("Versioned request package mismatch")
    return package, manifest


def _load_v2_review(root: Path, v1_root: Path) -> tuple[dict, dict, dict]:
    package, request = _verify_requests(root)
    candidates = read(Path(root) / "candidates.json")
    reviews = read(Path(root) / "review.json")
    themes = {theme["theme_id"]: theme for theme in package["themes"]}
    originals = {row["theme_id"]: row for row in candidates["insights"]}
    if (len(themes) != 99 or originals.keys() != themes.keys() or len(reviews) != 99
            or {row["theme_id"] for row in reviews} != themes.keys()
            or candidates["request_manifest_hash"] != sha(Path(root) / "request_manifest.json")
            or candidates.get("correction", {}).get("regenerated_candidate_rows") != [TARGET]):
        raise ValueError("Versioned candidate/review coverage mismatch")
    for row in reviews:
        original = originals[row["theme_id"]]
        validate_insight(original, themes[row["theme_id"]])
        validate_insight(row["insight"], themes[row["theme_id"]])
        if row["candidate_hash"] != digest(original):
            raise ValueError("Stale review candidate hash")
        if set(row.get("claim_reviews", {})) != set(original["claims"]):
            raise ValueError("Incomplete claim review")
    v1 = review.load_review(v1_root)
    for identity, original in v1["originals"].items():
        if identity != TARGET and digest(originals[identity]) != digest(original):
            raise ValueError("Unaffected candidate changed: " + identity)
    for v2_row in reviews:
        if v2_row["theme_id"] != TARGET:
            v1_row = next(row for row in v1["reviews"] if row["theme_id"] == v2_row["theme_id"])
            if v2_row != v1_row:
                raise ValueError("Unaffected review changed: " + v2_row["theme_id"])
    return package, candidates, {"reviews": reviews, "themes": themes, "request": request}


def build(root: Path = VERSIONED_ROOT, v1_root: Path = V1_ROOT) -> dict:
    """Create v2 exclusively; retain v1 untouched and copy unaffected content exactly."""
    root, v1_root = Path(root), Path(v1_root)
    if root.exists():
        raise FileExistsError("Versioned correction root already exists")
    source = review.load_review(v1_root)
    package = copy.deepcopy(source["package"])
    target = next(theme for theme in package["themes"] if theme["theme_id"] == TARGET)
    old_ids = [item["evidence_id"] for item in target["representative_evidence"]]
    target["representative_evidence"] = [_evidence(row) for row in _records()]
    target["unavailable_evidence_ids"] = []
    package["correction"] = {
        "correction_version": CORRECTION_VERSION,
        "target_theme_id": TARGET,
        "method": "targeted representative-evidence selection from fixed rt08 lineage",
        "v1_representative_evidence_ids": old_ids,
        "v2_representative_evidence_ids": [item["evidence_id"] for item in target["representative_evidence"]],
        "lineage_source": "reviewed_theme_assignments.csv",
        "external_api_calls": 0,
    }
    validate_package(package)
    root.mkdir(parents=True)
    workflow.write_new(root / "evidence.json", package)
    workflow.write_new(root / "evidence_manifest.json", {
        "evidence_sha256": sha(root / "evidence.json"), "input_hashes": package["input_hashes"],
        "correction_version": CORRECTION_VERSION, "v1_evidence_sha256": sha(v1_root / "evidence.json"),
    })

    v1_requests = (v1_root / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    request_lines = []
    for line in v1_requests:
        item = json.loads(line)
        request_lines.append(_request_line(target, package, source["request"]["model"])
                             if item["custom_id"] == TARGET else line)
    if len(request_lines) != 99 or sum(json.loads(line)["custom_id"] == TARGET for line in request_lines) != 1:
        raise ValueError("Request coverage changed")
    workflow.write_new(root / "requests.jsonl", "\n".join(request_lines) + "\n", raw=True)
    request = copy.deepcopy(source["request"])
    request.update(version=CORRECTION_VERSION, input_artifact_hash=sha(root / "evidence.json"),
                   requests_hash=sha(root / "requests.jsonl"), request_count=99,
                   request_bytes=(root / "requests.jsonl").stat().st_size,
                   created_at=_now(), generation_status="TARGETED_RT08_CORRECTION_READY",
                   correction={"v1_request_manifest_hash": sha(v1_root / "request_manifest.json"),
                               "preserved_request_lines": 98, "regenerated_request_lines": [TARGET],
                               "external_api_calls": 0})
    workflow.write_new(root / "request_manifest.json", request)
    shutil.copyfile(v1_root / "sample_request.json", root / "sample_request.json")

    candidate = corrected_candidate()
    validate_insight(candidate, target)
    candidates = copy.deepcopy(source["candidates"])
    candidates["insights"] = [candidate if item["theme_id"] == TARGET else item
                              for item in candidates["insights"]]
    provenance = candidates.setdefault("retry_provenance", {})
    themes = provenance.setdefault("themes", {})
    themes[TARGET] = {
        "accepted_version": "v2_targeted_in_session_ai_assisted_candidate",
        "retried": False,
        "response_id": "not_applicable_no_external_api_response",
        "batch_request_id": "not_applicable_targeted_v2_correction",
        "response_line_hash": "not_applicable",
        "response_file": "not_applicable",
        "response_file_hash": "not_applicable",
        "raw_insight_hash": digest(candidate), "insight_hash": digest(candidate),
        "mechanical_corrections": [],
        "v1_candidate_hash": source["originals"][TARGET] and digest(source["originals"][TARGET]),
        "evidence_selection_correction": "Four existing rt08 lineage records replaced v1's unrepresentative five excerpts.",
    }
    candidates.update(generation_status="generated_pending_review", generated_at=_now(),
                      request_manifest_hash=sha(root / "request_manifest.json"),
                      response_hash="not_applicable_targeted_in_session_ai_assisted_candidate",
                      model=request["model"], correction={
                          "correction_version": CORRECTION_VERSION,
                          "v1_candidates_sha256": sha(v1_root / "candidates.json"),
                          "preserved_candidate_rows": 98, "regenerated_candidate_rows": [TARGET],
                          "external_api_calls": 0,
                          "candidate_generation": "targeted in-session AI-assisted synthesis; not an external API response",
                      })
    if len(candidates["insights"]) != 99:
        raise ValueError("Candidate coverage changed")
    workflow.write_new(root / "candidates.json", candidates)

    reviews = copy.deepcopy(source["reviews"])
    for row in reviews:
        if row["theme_id"] != TARGET:
            continue
        v1_review = copy.deepcopy(row)
        row.update(theme_id=TARGET, decision="APPROVE", reviewer=REVIEWER,
                   reason=("AI-assisted substantive review; not independent human validation. "
                           "APPROVE: corrected rt08 evidence directly and cautiously supports work/life-training scheduling; "
                           "no prevalence, operational failure, or remedy is inferred."),
                   candidate_hash=digest(candidate), insight=copy.deepcopy(candidate),
                   claim_reviews=_claim_reviews(), reviewed_at=_now(), review_schema_version=2,
                   review_history=[{"saved_at": _now(), "v1_review_reference": v1_review,
                                    "reason": "Versioned rt08 evidence-selection correction; v1 remains unchanged."}])
    workflow.write_new(root / "review.json", reviews)
    correction = {
        "correction_version": CORRECTION_VERSION, "created_at": _now(), "target_theme_id": TARGET,
        "v1_root": str(v1_root.resolve()), "v1_hashes": {
            name: sha(v1_root / name) for name in ("evidence.json", "evidence_manifest.json", "requests.jsonl",
                                                     "request_manifest.json", "candidates.json", "review.json")},
        "v2_hashes": {name: sha(root / name) for name in ("evidence.json", "evidence_manifest.json", "requests.jsonl",
                                                             "request_manifest.json", "candidates.json", "review.json")},
        "preserved": {"themes": 98, "request_lines": 98, "candidate_rows": 98, "review_rows": 98},
        "regenerated": {"evidence_theme": TARGET, "request_line": TARGET, "candidate_row": TARGET,
                          "review_row": TARGET},
        "external_api_calls": 0,
    }
    workflow.write_new(root / "correction_manifest.json", correction)
    return verify(root, v1_root)


def verify(root: Path = VERSIONED_ROOT, v1_root: Path = V1_ROOT) -> dict:
    root, v1_root = Path(root), Path(v1_root)
    package, candidates_file, data = _load_v2_review(root, v1_root)
    v1 = review.load_review(v1_root)
    reviews = {row["theme_id"]: row for row in data["reviews"]}
    candidates = {row["theme_id"]: row for row in candidates_file["insights"]}
    if len(package["themes"]) != 99 or len(reviews) != 99 or set(reviews) != set(candidates):
        raise ValueError("Expected exactly 99 themes")
    if any(set(row.get("claim_reviews", {})) != set(candidates[row["theme_id"]]["claims"]) for row in data["reviews"]):
        raise ValueError("Incomplete claim judgments")
    if any(row["decision"] != "APPROVE" for row in data["reviews"]):
        raise ValueError("Not all themes are approved")
    target = next(theme for theme in package["themes"] if theme["theme_id"] == TARGET)
    expected_ids = ["instagram:" + identity for identity in CORRECTION_IDS]
    if [item["evidence_id"] for item in target["representative_evidence"]] != expected_ids:
        raise ValueError("Wrong rt08 correction evidence")
    validate_insight(candidates[TARGET], target)
    for identity, v1_candidate in v1["originals"].items():
        if identity != TARGET and digest(candidates[identity]) != digest(v1_candidate):
            raise ValueError("Unaffected candidate changed: " + identity)
    for v2_row in data["reviews"]:
        if v2_row["theme_id"] == TARGET:
            continue
        v1_row = next(row for row in v1["reviews"] if row["theme_id"] == v2_row["theme_id"])
        if v2_row != v1_row:
            raise ValueError("Unaffected review changed: " + v2_row["theme_id"])
    correction = read(root / "correction_manifest.json")
    if correction["v2_hashes"] != {name: sha(root / name) for name in correction["v2_hashes"]}:
        raise ValueError("Versioned artifact hash mismatch")
    if correction["v1_hashes"] != {name: sha(v1_root / name) for name in correction["v1_hashes"]}:
        raise ValueError("v1 artifact changed")
    return {"themes": 99, "approve": 99, "reject": 0, "pending": 0,
            "rt08": reviews[TARGET]["decision"], "request_manifest": data["request"]["requests_hash"],
            "v1_preserved": True, "external_api_calls": 0}


def finalize(root: Path = VERSIONED_ROOT, v1_root: Path = V1_ROOT, reviewer: str = "") -> dict:
    """Versioned finalizer; never routes v2 through v1's narrower evidence resolver."""
    if not reviewer.strip():
        raise ValueError("Named finalizer required")
    root = Path(root)
    result = verify(root, v1_root)
    if any((root / name).exists() for name in ("finalized.json", "finalized_manifest.json")):
        raise FileExistsError("Versioned package is already finalized or partially finalized")
    _, candidates, data = _load_v2_review(root, v1_root)
    lookup = {row["theme_id"]: row for row in candidates["insights"]}
    for row in data["reviews"]:
        if row["decision"] != "APPROVE" or not row["reviewer"].strip() or not row["reason"].strip():
            raise ValueError("Unresolved review")
        if row["candidate_hash"] != digest(lookup[row["theme_id"]]):
            raise ValueError("Stale candidate review")
    workflow.write_new(root / "finalized.json", [row["insight"] for row in data["reviews"]])
    names = ("evidence.json", "evidence_manifest.json", "requests.jsonl", "request_manifest.json",
             "candidates.json", "review.json", "correction_manifest.json", "finalized.json")
    workflow.write_new(root / "finalized_manifest.json", {
        "version": CORRECTION_VERSION, "review_status": "finalized", "reviewer": reviewer,
        "finalized_at": _now(), "hashes": {name: sha(root / name) for name in names},
        "v1_root": str(Path(v1_root).resolve()), "external_api_calls": 0,
    })
    return {**result, "review_status": "finalized", "insights": 99}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify", "finalize"))
    parser.add_argument("--root", type=Path, default=VERSIONED_ROOT)
    parser.add_argument("--v1-root", type=Path, default=V1_ROOT)
    parser.add_argument("--reviewer", default=REVIEWER)
    args = parser.parse_args(argv)
    if args.action == "build":
        result = build(args.root, args.v1_root)
    elif args.action == "verify":
        result = verify(args.root, args.v1_root)
    else:
        result = finalize(args.root, args.v1_root, args.reviewer)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
