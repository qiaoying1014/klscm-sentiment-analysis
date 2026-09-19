import hashlib
import json
from pathlib import Path

import pandas as pd

from marathon_absa.adjudication_phase8 import (
    deterministic_scores, operating_point_metrics, route_diff, route_frame,
)


CAL = Path("data/processed/relevance_v8_calibration_results")
OUT = CAL/"threshold_selection"


def calibration():
    return pd.read_csv(CAL/"relevance_v8_calibration_adjudicated_v1.csv",
                       keep_default_na=False)


def test_current_threshold_simulation_reproduces_frozen_routes():
    frame = calibration()
    routed = route_frame(frame, .80, .93)
    assert routed.v8_routing_status.tolist() == frame.v8_routing_status.tolist()
    assert routed.v8_final_inclusion.tolist() == frame.v8_final_inclusion.tolist()


def test_only_three_permitted_exclusion_routes_change():
    diff = route_diff(calibration())
    assert len(diff) == 3
    assert set(diff.old_route) == {"automatic_exclude"}
    assert set(diff.new_route) == {"pending_review"}
    assert diff.gold_label.value_counts().to_dict() == {"include": 2, "exclude": 1}


def test_threshold_does_not_change_classifier_or_gold_fields():
    frame = calibration()
    before = frame.copy(deep=True)
    route_frame(frame, .80, .99)
    pd.testing.assert_frame_equal(frame, before)


def test_selected_metrics_and_zero_false_exclusions():
    metrics = operating_point_metrics(calibration(), .80, .99)
    assert metrics["automatic_decision_n"] == 259
    assert metrics["automatic_false_exclusions"] == 0
    assert metrics["automatic_exclusion_precision"] == 1
    assert metrics["review_n"] == 41


def test_independent_excluded_recall_f1_and_macro_f1():
    truth = pd.Series(["include"]*239 + ["exclude"]*20)
    pred = pd.Series(["include"]*239 + ["include"]*16 + ["exclude"]*4)
    result = deterministic_scores(truth, pred)
    assert result["confusion_matrix"] == [[239, 0], [16, 4]]
    assert result["excluded_class_recall"] == .2
    assert abs(result["excluded_class_f1"] - 1/3) < 1e-12
    assert abs(result["automatic_binary_macro_f1"] - .650472334682861) < 1e-12


def test_workload_projection_is_deterministic():
    current = operating_point_metrics(calibration(), .80, .93)
    selected = operating_point_metrics(calibration(), .80, .99)
    extra = selected["review_n"] - current["review_n"]
    assert extra == 3
    assert extra / 300 * 1000 == 10
    assert extra / 300 * 14000 == 140


def test_operating_point_version_and_append_only_manifest():
    config = json.loads((OUT/"relevance_v8_operating_point_v1.json").read_text())
    assert config["relevance_operating_point_version"] == "v8_op1"
    assert config["inclusion_threshold"] == .80
    assert config["exclusion_threshold"] == .99
    assert config["gold_labels_changed"] is False
    assert config["classifier_outputs_changed"] is False


def test_phase8_manifest_integrity_and_reports():
    manifest = json.loads((OUT/"relevance_v8_threshold_selection_v1.manifest.json").read_text())
    for raw, expected in manifest["files"].items():
        assert hashlib.sha256(Path(raw).read_bytes()).hexdigest() == expected
    assert Path("RELEVANCE_V8_THRESHOLD_SELECTION_REPORT.md").exists()
    assert len(pd.read_csv(OUT/"relevance_v8_false_inclusion_analysis_v1.csv")) == 16
    safety = pd.read_csv(OUT/"relevance_v8_auto_exclusion_safety_audit_v1.csv")
    assert len(safety) == 4 and safety.safe_gold_exclude.all()
