import json

from marathon_absa import participant_experience_rt08_v2 as correction
from marathon_absa import participant_experience_review as review
from marathon_absa.participant_experience_dashboard_data import digest, read, sha


def test_rt08_v2_build_preserves_v1_and_reviews_all_themes(tmp_path):
    v1_root = correction.V1_ROOT
    before = {name: sha(v1_root / name) for name in ("evidence.json", "evidence_manifest.json", "requests.jsonl",
                                                   "request_manifest.json", "candidates.json", "review.json")}
    root = tmp_path / "participant_experience_v2_rt08"
    result = correction.build(root)
    assert result == {"themes": 99, "approve": 99, "reject": 0, "pending": 0,
                      "rt08": "APPROVE", "request_manifest": read(root / "request_manifest.json")["requests_hash"],
                      "v1_preserved": True, "external_api_calls": 0}
    assert before == {name: sha(v1_root / name) for name in before}
    v1 = review.load_review(v1_root)
    _, candidates_file, v2 = correction._load_v2_review(root, v1_root)
    assert len(v2["themes"]) == 99
    target = correction.TARGET
    assert [e["evidence_id"] for e in v2["themes"][target]["representative_evidence"]] == ["instagram:" + x for x in correction.CORRECTION_IDS]
    assert all(row["decision"] == "APPROVE" for row in v2["reviews"])
    candidates = {row["theme_id"]: row for row in candidates_file["insights"]}
    assert all(set(row["claim_reviews"]) == set(candidates[row["theme_id"]]["claims"]) for row in v2["reviews"])
    assert all(digest(candidates[theme]) == digest(v1["originals"][theme]) for theme in v1["originals"] if theme != target)
    assert all(next(row for row in v2["reviews"] if row["theme_id"] == theme) ==
               next(row for row in v1["reviews"] if row["theme_id"] == theme)
               for theme in v1["themes"] if theme != target)
    manifest = read(root / "correction_manifest.json")
    assert manifest["preserved"] == {"themes": 98, "request_lines": 98, "candidate_rows": 98, "review_rows": 98}
    assert manifest["external_api_calls"] == 0
    assert correction.verify(root) == result


def test_rt08_candidate_cites_only_corrected_lineage():
    candidate = correction.corrected_candidate()
    allowed = {"instagram:" + identity for identity in correction.CORRECTION_IDS}
    assert all(set(claim["evidence_ids"]) <= allowed for claim in candidate["claims"].values())
    assert candidate["claims"]["organizer_implication"]["text"].startswith("Organizers may consider")
