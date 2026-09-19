import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from marathon_absa.absa_final_outputs import ROOT, create_thesis_outputs
from marathon_absa.dashboard_data import DASHBOARD_ROOT, load_dashboard_data


def test_core_counts_and_final_table_dimensions_reconcile():
    corpus = pd.read_csv(ROOT / "table_01_corpus_summary.csv").set_index("metric")
    aspects = pd.read_csv(ROOT / "table_03_aspect_prevalence.csv")
    topics = pd.read_csv(ROOT / "table_06_topic_aspect_summary.csv")
    assert corpus.loc["Analyzed substantive posts", "value"] == 7704
    assert corpus.loc["Posts with >=1 model-estimated aspect", "value"] == 5316
    assert corpus.loc["Zero-mention posts", "value"] == 2388
    assert corpus.loc["Model-estimated aspect mentions", "value"] == 15486
    assert len(aspects) == aspects.aspect.nunique() == 20
    assert len(topics) == topics.topic_id.nunique() == 32
    assert topics.analysis_type.eq("descriptive_only").all()


def test_aspect_effects_and_fdr_values_are_unchanged():
    frozen = load_dashboard_data().aspects.set_index("aspect")
    final = pd.read_csv(ROOT / "table_03_aspect_prevalence.csv").set_index("aspect")
    assert np.allclose(final.loc[frozen.index, "cramers_v"], frozen.cramers_v, equal_nan=True)
    assert np.allclose(final.loc[frozen.index, "adjusted_p"], frozen.year_association_adjusted_p, equal_nan=True)
    assert final.loc[frozen.index, "effect_category"].equals(frozen.effect_category)
    assert final.loc[frozen.index, "recommended_for_emphasis"].equals(frozen.recommended_for_emphasis)


def test_figure_and_temporal_table_values_match_frozen_marts():
    data = load_dashboard_data()
    table = pd.read_csv(ROOT / "table_05_year_aspect_prevalence.csv").set_index("aspect")
    for aspect in table.index:
        frozen = data.year_aspect[data.year_aspect.aspect.eq(aspect)].set_index("year")
        for year in [2019, 2023, 2024, 2025]:
            assert table.loc[aspect, f"{year}_prevalence"] == frozen.loc[year, "document_prevalence"]
    for name in ["figure_01_primary_aspect_prevalence.png", "figure_02_training_sentiment_by_year.png",
                 "figure_03_aspect_sentiment_composition.png", "figure_04_overall_sentiment.png",
                 "figure_05_descriptive_topic_aspect_alignment.png"]:
        with Image.open(ROOT / name) as image: assert image.size == (1800, 1100)
    assert (ROOT / "figure_01_primary_aspect_prevalence.pdf").stat().st_size > 10000


def test_key_findings_and_cautions_remain_frozen():
    data = load_dashboard_data()
    prose = (ROOT / "THESIS_KEY_FINDINGS_V1.md").read_text(encoding="utf-8")
    for headline in data.findings.headline: assert headline in prose
    final = pd.read_csv(ROOT / "table_03_aspect_prevalence.csv").set_index("aspect")
    assert not final.loc["photography_media", "recommended_for_emphasis"]
    assert not final.loc["weather_conditions", "recommended_for_emphasis"]
    assert not final[final.support_class.isin(["low_support", "very_low_support"])].recommended_for_emphasis.any()
    assert "Photography/media" in prose and "weather sentiment" in prose


def test_manifest_hashes_every_frozen_source_without_modification():
    manifest = json.loads((ROOT / "final_output_manifest_v1.json").read_text(encoding="utf-8"))
    import hashlib
    for path_text, record in manifest["source_artifacts"].items():
        path = Path(path_text); digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert record["sha256"] == digest
    assert len(manifest["source_artifacts"]) == len([p for p in DASHBOARD_ROOT.iterdir() if p.is_file()])
    assert all(value == 0 for value in manifest["execution"].values())


def test_exporter_contains_no_api_inference_or_hypothesis_testing_code():
    source = inspect.getsource(create_thesis_outputs).lower()
    for forbidden in ["chi2_contingency", "multipletests(", "wilson_interval(", "openai.",
                      "from .openai", "huggingface_hub", "bertopic(", "requests."]:
        assert forbidden not in source


def test_outputs_regenerate_deterministically(tmp_path):
    manifest = json.loads((ROOT / "final_output_manifest_v1.json").read_text(encoding="utf-8"))
    regenerated = tmp_path / "thesis"
    create_thesis_outputs(regenerated, created_at=manifest["created_at"])
    for name in ["table_01_corpus_summary.csv", "table_03_aspect_prevalence.csv",
                 "table_07_inferential_summary.csv", "THESIS_KEY_FINDINGS_V1.md",
                 "figure_01_primary_aspect_prevalence.png", "figure_04_overall_sentiment.png"]:
        assert (ROOT / name).read_bytes() == (regenerated / name).read_bytes()
