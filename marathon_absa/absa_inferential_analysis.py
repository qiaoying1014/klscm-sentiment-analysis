from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.stats import chi2_contingency, norm

from .absa_descriptive_analysis import (
    DOCUMENTS_PATH, EXPECTED_BEARING, EXPECTED_DOCUMENTS, EXPECTED_MENTIONS,
    EXPECTED_ZERO, FINALIZATION_MANIFEST, MENTIONS_PATH, SUPPORT_THRESHOLDS,
    YEARS, _load_and_validate, _sha, _support,
)

DESCRIPTIVE_ROOT = Path("data/processed/absa_v1/analysis/absa_v1_descriptive_analysis_v1")
SUPPORT_PATH = DESCRIPTIVE_ROOT / "absa_v1_support_flags_v1.csv"
DESCRIPTIVE_MANIFEST = DESCRIPTIVE_ROOT / "analysis_manifest_v1.json"
ROOT = Path("data/processed/absa_v1/analysis/absa_v1_inferential_analysis_v1")
ALPHA = 0.05
MIN_YEAR_ASPECT_DOCUMENTS = 20
MODEL_LIMITATION = (
    "Statistical tests quantify associations in the model-estimated ABSA labels, not uncertainty in the ABSA "
    "classifier itself. The tests do not correct for development precision = 0.513 and recall = 0.790."
)


def wilson_interval(events: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    z = norm.ppf(1 - alpha / 2)
    p = events / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    lower, upper = max(0.0, centre - half), min(1.0, centre + half)
    if abs(lower) < 1e-15: lower = 0.0
    if abs(upper - 1.0) < 1e-15: upper = 1.0
    return lower, upper


def adjust_pvalues(values: list[float], method: str) -> list[float]:
    p = np.asarray(values, dtype=float)
    result = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return result.tolist()
    ordered = valid[np.argsort(p[valid])]
    m = len(ordered)
    if method == "bh":
        adjusted = p[ordered] * m / np.arange(1, m + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    elif method == "holm":
        adjusted = p[ordered] * (m - np.arange(m))
        adjusted = np.maximum.accumulate(adjusted)
    else:
        raise ValueError(method)
    result[ordered] = np.minimum(adjusted, 1.0)
    return result.tolist()


def cramers_v(table: np.ndarray, statistic: float) -> float:
    n = table.sum()
    return float(np.sqrt(statistic / (n * min(table.shape[0] - 1, table.shape[1] - 1)))) if n else np.nan


def _omnibus(frame: pd.DataFrame, outcome: str) -> dict[str, Any]:
    table = pd.crosstab(frame.event_year, frame[outcome]).reindex(index=YEARS, columns=[0, 1], fill_value=0)
    statistic, raw_p, df, expected = chi2_contingency(table.to_numpy(), correction=False)
    return {"test": "pearson_chi_square", "statistic": statistic, "df": int(df), "raw_p": raw_p,
            "effect_size": cramers_v(table.to_numpy(), statistic), "minimum_expected_count": float(expected.min()),
            "expected_counts_adequate": bool((expected >= 5).all())}


def _categorical_logit(frame: pd.DataFrame, outcome: str) -> list[dict[str, Any]]:
    year = frame.event_year.astype(int).to_numpy()
    y = frame[outcome].astype(float).to_numpy()
    x = np.column_stack([np.ones(len(frame))] + [(year == value).astype(float) for value in YEARS[1:]])

    def objective(beta: np.ndarray) -> float:
        eta = x @ beta
        return float(np.logaddexp(0, eta).sum() - y @ eta)

    fitted = minimize(objective, np.zeros(x.shape[1]), method="BFGS")
    eta = x @ fitted.x
    probabilities = 1 / (1 + np.exp(-np.clip(eta, -40, 40)))
    information = x.T @ (x * (probabilities * (1 - probabilities))[:, None])
    covariance = np.linalg.pinv(information)
    se = np.sqrt(np.diag(covariance))
    rows = []
    for index, comparison in enumerate(YEARS[1:], start=1):
        beta = fitted.x[index]
        rows.append({"reference_year": YEARS[0], "comparison_year": comparison, "coefficient": beta,
                     "odds_ratio": float(np.exp(beta)), "ci_lower": float(np.exp(beta - 1.96 * se[index])),
                     "ci_upper": float(np.exp(beta + 1.96 * se[index])),
                     "raw_p": float(2 * norm.sf(abs(beta / se[index]))), "converged": bool(fitted.success)})
    return rows


def _difference_ci(e1: int, n1: int, e2: int, n2: int) -> tuple[float, float, float]:
    p1, p2 = e1 / n1, e2 / n2
    difference = p2 - p1
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return difference, difference - 1.96 * se, difference + 1.96 * se


def _pairwise(frame: pd.DataFrame, outcome: str, aspect: str, family: str) -> list[dict[str, Any]]:
    rows = []
    for left_index, left in enumerate(YEARS):
        for right in YEARS[left_index + 1:]:
            a, b = frame[frame.event_year.eq(left)], frame[frame.event_year.eq(right)]
            e1, n1, e2, n2 = int(a[outcome].sum()), len(a), int(b[outcome].sum()), len(b)
            table = np.array([[e1, n1 - e1], [e2, n2 - e2]])
            statistic, p, _, _ = chi2_contingency(table, correction=False)
            difference, lower, upper = _difference_ci(e1, n1, e2, n2)
            odds_ratio = ((e2 + .5) * (n1 - e1 + .5)) / ((n2 - e2 + .5) * (e1 + .5))
            rows.append({"aspect": aspect, "analysis_family": family, "year_1": left, "year_2": right,
                         "n_1": n1, "events_1": e1, "prevalence_1": e1/n1, "n_2": n2,
                         "events_2": e2, "prevalence_2": e2/n2, "difference": difference,
                         "ci_lower": lower, "ci_upper": upper, "odds_ratio_year_2_vs_year_1": odds_ratio,
                         "test": "two_sample_proportion_chi_square", "statistic": statistic, "raw_p": p})
    adjusted = adjust_pvalues([row["raw_p"] for row in rows], "holm")
    for row, value in zip(rows, adjusted):
        row["adjusted_p"] = value
        row["significant_holm"] = value < ALPHA
    return rows


def build_document_matrix(documents: pd.DataFrame, mentions: pd.DataFrame, aspects: list[str]) -> pd.DataFrame:
    base = documents[["document_id", "event_year", "primary_language"]].copy()
    matrix = base.merge(pd.DataFrame({"aspect": aspects}), how="cross")
    grouped = mentions.groupby(["document_id", "aspect"])
    present = grouped.size().rename("mention_count").reset_index()
    matrix = matrix.merge(present, on=["document_id", "aspect"], how="left")
    matrix["mention_count"] = matrix.mention_count.fillna(0).astype(int)
    matrix["aspect_present"] = matrix.mention_count.gt(0).astype(int)
    for sentiment in ["positive", "negative", "mixed", "neutral"]:
        keys = mentions.loc[mentions.sentiment.eq(sentiment), ["document_id", "aspect"]].drop_duplicates()
        keys[f"{sentiment}_present"] = 1
        matrix = matrix.merge(keys, on=["document_id", "aspect"], how="left")
        matrix[f"{sentiment}_present"] = matrix[f"{sentiment}_present"].fillna(0).astype(int)
    sentiment_count = matrix[[f"{s}_present" for s in ["positive", "negative", "mixed", "neutral"]]].sum(axis=1)
    matrix["sentiment_label_count"] = sentiment_count
    matrix["dominant_sentiment_for_aspect"] = "absent"
    for sentiment in ["positive", "negative", "mixed", "neutral"]:
        matrix.loc[(sentiment_count.eq(1)) & matrix[f"{sentiment}_present"].eq(1), "dominant_sentiment_for_aspect"] = sentiment
    matrix.loc[sentiment_count.gt(1), "dominant_sentiment_for_aspect"] = "multiple_not_forced"
    return matrix


def _prevalence_rows(frame: pd.DataFrame, aspect: str, support_class: str, outcome: str,
                     family: str, omnibus: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for year in YEARS:
        subset = frame[frame.event_year.eq(year)]
        events, n = int(subset[outcome].sum()), len(subset)
        lower, upper = wilson_interval(events, n)
        rows.append({"aspect": aspect, "analysis_family": family, "support_class": support_class,
                     "year": year, "n": n, "events": events, "prevalence": events/n,
                     "ci_lower": lower, "ci_upper": upper, **omnibus})
    return rows


def _effect_label(value: float) -> str:
    if value < .10: return "no_detectable_year_association"
    if value < .20: return "small_year_association"
    if value < .30: return "moderate_year_association"
    return "strong_year_association"


def create_inferential_analysis(root: Path = ROOT) -> dict[str, Any]:
    documents, mentions, aspects = _load_and_validate()
    if root.exists():
        raise FileExistsError(f"Inferential analysis v1 already exists: {root}")
    if not SUPPORT_PATH.exists() or not DESCRIPTIVE_MANIFEST.exists():
        raise FileNotFoundError("Completed descriptive v1 artifacts are required")
    year_counts = documents.event_year.astype(int).value_counts().sort_index().to_dict()
    expected_years = {2019: 2279, 2023: 1873, 2024: 1724, 2025: 1828}
    if year_counts != expected_years:
        raise RuntimeError(f"Year reconciliation failed: {year_counts} != {expected_years}")
    source_paths = [DOCUMENTS_PATH, MENTIONS_PATH, FINALIZATION_MANIFEST, SUPPORT_PATH, DESCRIPTIVE_MANIFEST]
    source_hashes = {str(path): _sha(path) for path in source_paths}
    support = pd.read_csv(SUPPORT_PATH)
    matrix = build_document_matrix(documents, mentions, aspects)
    if len(matrix) != EXPECTED_DOCUMENTS * len(aspects) or matrix.document_id.nunique() != EXPECTED_DOCUMENTS:
        raise RuntimeError("Document x aspect matrix reconciliation failed")

    eligible = []
    for row in support.itertuples():
        counts = matrix.loc[matrix.aspect.eq(row.aspect)].groupby("event_year").aspect_present.sum().reindex(YEARS, fill_value=0)
        if row.support_flag in {"moderate_support", "high_support"} and int(counts.min()) >= MIN_YEAR_ASPECT_DOCUMENTS:
            eligible.append(row.aspect)
    eligibility = support.assign(
        minimum_year_affected_documents=support.aspect.map(lambda a: int(matrix.loc[matrix.aspect.eq(a)].groupby("event_year").aspect_present.sum().reindex(YEARS, fill_value=0).min())),
        primary_inference_eligible=support.aspect.isin(eligible),
    )
    # Eligibility is frozen here, before any hypothesis test is run.
    support_map = support.set_index("aspect").support_flag.to_dict()
    prevalence_rows, model_rows = [], []
    omnibus_by_family: dict[str, list[dict[str, Any]]] = {"A_aspect_prevalence": [], "B_positive_sentiment": [], "C_negative_sentiment": []}
    outcome_tables: dict[str, list[dict[str, Any]]] = {"positive": [], "negative": []}
    multi_sentiment = []
    for aspect in eligible:
        aspect_all = matrix[matrix.aspect.eq(aspect)].copy()
        test = _omnibus(aspect_all, "aspect_present")
        omnibus_by_family["A_aspect_prevalence"].append({"aspect": aspect, **test})
        prevalence_rows.extend(_prevalence_rows(aspect_all, aspect, support_map[aspect], "aspect_present", "A_aspect_prevalence", test))
        for item in _categorical_logit(aspect_all, "aspect_present"):
            model_rows.append({"aspect": aspect, "analysis_family": "A_aspect_prevalence", "support_class": support_map[aspect], **item})
        bearing = aspect_all[aspect_all.aspect_present.eq(1)].copy()
        multi_sentiment.append({"aspect": aspect, "aspect_bearing_document_count": len(bearing),
                                "multiple_sentiment_document_count": int(bearing.sentiment_label_count.gt(1).sum()),
                                "multiple_sentiment_rate": float(bearing.sentiment_label_count.gt(1).mean())})
        for sentiment, family in [("positive", "B_positive_sentiment"), ("negative", "C_negative_sentiment")]:
            outcome = f"{sentiment}_present"
            test = _omnibus(bearing, outcome)
            # Sparse tests are retained descriptively but excluded from the primary FDR family.
            test["primary_test_eligible"] = test["expected_counts_adequate"]
            omnibus_by_family[family].append({"aspect": aspect, **test})
            outcome_tables[sentiment].extend(_prevalence_rows(bearing, aspect, support_map[aspect], outcome, family, test))

    for family, tests in omnibus_by_family.items():
        eligible_tests = [row for row in tests if family == "A_aspect_prevalence" or row.get("primary_test_eligible")]
        adjusted = adjust_pvalues([row["raw_p"] for row in eligible_tests], "bh")
        for row in tests:
            row["adjusted_p"] = np.nan
            row["significant_fdr"] = False
        for row, value in zip(eligible_tests, adjusted):
            row["adjusted_p"] = value
            row["significant_fdr"] = value < ALPHA

    lookup = {(family, row["aspect"]): row for family, rows in omnibus_by_family.items() for row in rows}
    for row in prevalence_rows + outcome_tables["positive"] + outcome_tables["negative"]:
        result = lookup[(row["analysis_family"], row["aspect"])]
        row["adjusted_p"] = result["adjusted_p"]
        row["significant_fdr"] = result["significant_fdr"]
        if "primary_test_eligible" in result: row["primary_test_eligible"] = result["primary_test_eligible"]

    pairwise_rows = []
    family_specs = [("A_aspect_prevalence", "aspect_present", None),
                    ("B_positive_sentiment", "positive_present", 1),
                    ("C_negative_sentiment", "negative_present", 1)]
    for family, outcome, bearing_only in family_specs:
        for result in omnibus_by_family[family]:
            if result["significant_fdr"]:
                subset = matrix[matrix.aspect.eq(result["aspect"])]
                if bearing_only: subset = subset[subset.aspect_present.eq(1)]
                pairwise_rows.extend(_pairwise(subset, outcome, result["aspect"], family))

    root.mkdir(parents=True)
    matrix.to_csv(root / "absa_v1_document_analysis_matrix_v1.csv", index=False)
    prevalence = pd.DataFrame(prevalence_rows)
    positive = pd.DataFrame(outcome_tables["positive"])
    negative = pd.DataFrame(outcome_tables["negative"])
    models = pd.DataFrame(model_rows)
    pairwise = pd.DataFrame(pairwise_rows)
    prevalence.to_csv(root / "absa_v1_inferential_aspect_prevalence_v1.csv", index=False)
    models.to_csv(root / "absa_v1_inferential_aspect_year_models_v1.csv", index=False)
    positive.to_csv(root / "absa_v1_inferential_positive_year_v1.csv", index=False)
    negative.to_csv(root / "absa_v1_inferential_negative_year_v1.csv", index=False)
    pairwise.to_csv(root / "absa_v1_inferential_pairwise_year_v1.csv", index=False)
    prevalence.to_csv(root / "year_aspect_prevalence_ci_long_v1.csv", index=False)
    positive.to_csv(root / "year_aspect_positive_share_ci_long_v1.csv", index=False)
    negative.to_csv(root / "year_aspect_negative_share_ci_long_v1.csv", index=False)
    pairwise[pairwise.significant_holm].to_csv(root / "significant_pairwise_year_differences_v1.csv", index=False)
    pd.DataFrame(multi_sentiment).to_csv(root / "absa_v1_multiple_sentiment_diagnostic_v1.csv", index=False)
    eligibility.to_csv(root / "absa_v1_inferential_eligibility_v1.csv", index=False)

    summary_rows = []
    for aspect in eligible:
        rows = prevalence[prevalence.aspect.eq(aspect)].set_index("year")
        test = lookup[("A_aspect_prevalence", aspect)]
        significant_pairs = pairwise[(pairwise.aspect.eq(aspect)) & (pairwise.analysis_family.eq("A_aspect_prevalence")) & pairwise.significant_holm]
        largest = significant_pairs.iloc[significant_pairs.difference.abs().argmax()] if len(significant_pairs) else None
        interpretation = _effect_label(test["effect_size"]) if test["significant_fdr"] else "no_detectable_year_association"
        summary_rows.append({"aspect": aspect, **{f"{year}_prevalence": rows.loc[year, "prevalence"] for year in YEARS},
                             "omnibus_adjusted_p": test["adjusted_p"], "cramers_v": test["effect_size"],
                             "largest_pairwise_difference": float(largest.difference) if largest is not None else np.nan,
                             "largest_pairwise_years": f"{int(largest.year_1)}-{int(largest.year_2)}" if largest is not None else "",
                             "interpretation": interpretation})
    research_summary = pd.DataFrame(summary_rows)
    research_summary.to_csv(root / "absa_v1_inferential_research_summary_v1.csv", index=False)

    confidence = mentions.confidence.describe(percentiles=[.01, .05, .5, .95, .99]).to_dict()
    significant_a = [r["aspect"] for r in omnibus_by_family["A_aspect_prevalence"] if r["significant_fdr"]]
    significant_b = [r["aspect"] for r in omnibus_by_family["B_positive_sentiment"] if r["significant_fdr"]]
    significant_c = [r["aspect"] for r in omnibus_by_family["C_negative_sentiment"] if r["significant_fdr"]]
    prevalence_table = "\n".join(
        f"| {a} | " + " | ".join(f"{int(prevalence[(prevalence.aspect.eq(a)) & (prevalence.year.eq(y))].events.iloc[0])}/{expected_years[y]} ({prevalence[(prevalence.aspect.eq(a)) & (prevalence.year.eq(y))].prevalence.iloc[0]:.1%})" for y in YEARS)
        + f" | {lookup[('A_aspect_prevalence', a)]['effect_size']:.3f} | {lookup[('A_aspect_prevalence', a)]['adjusted_p']:.3g} |"
        for a in eligible)
    report = f"""# ABSA V1 inferential analysis\n\n## Data basis\n\nReconciled {len(documents):,} documents, {len(mentions):,} mentions, {EXPECTED_BEARING:,} mention-bearing documents, and {EXPECTED_ZERO:,} zero-mention documents. Year counts: {expected_years}. Every mention maps to a frozen production document.\n\n## Analysis design\n\nThe primary unit is the document. The document × aspect matrix includes zero-mention documents and binary presence/sentiment fields. Eligible aspects were frozen before hypothesis testing: {', '.join(eligible)}. Eligibility requires moderate/high corpus support and at least {MIN_YEAR_ASPECT_DOCUMENTS} aspect-bearing documents per year. Pearson chi-square omnibus tests use Cramer's V; BH-FDR is separate for prevalence, positive sentiment, and negative sentiment. Pairwise tests are generated only after an FDR-significant omnibus result and use Holm correction within aspect/outcome. Categorical logistic models use 2019 as reference.\n\nEffect interpretation uses Cramer's V <0.10 as trivial/no detectable magnitude, 0.10–<0.20 small, 0.20–<0.30 moderate, and ≥0.30 strong; statistical detectability and substantive magnitude remain distinct.\n\n## Aspect prevalence by year\n\n| Aspect | 2019 | 2023 | 2024 | 2025 | Cramer's V | BH-adjusted p |\n|---|---:|---:|---:|---:|---:|---:|\n{prevalence_table}\n\nFDR-significant prevalence associations: {', '.join(significant_a) if significant_a else 'none'}. Non-significance is reported in the CSV rather than treated as evidence of equality.\n\n## Sentiment within aspects by year\n\nAmong aspect-bearing documents, FDR-significant positive-sentiment associations: {', '.join(significant_b) if significant_b else 'none'}. FDR-significant negative-sentiment associations: {', '.join(significant_c) if significant_c else 'none'}. A document may retain multiple sentiment labels; these are quantified in `absa_v1_multiple_sentiment_diagnostic_v1.csv`, and no dominant label is forced for multi-sentiment cases. Neutral and mixed outcomes are not formally tested.\n\n## Pairwise differences\n\nOnly FDR-qualified omnibus results generated pairwise comparisons. Holm-significant contrasts and absolute percentage-point differences are in `significant_pairwise_year_differences_v1.csv`; all qualifying comparisons and confidence intervals are preserved in the full pairwise table.\n\n## Extraction-density diagnostic\n\nThe descriptive increase in mentions per document (1.501 in 2019 to 2.543 in 2025) and decline in zero-mention rate (37.8% to 23.3%) may reflect corpus composition and/or model extraction behavior. It is not automatically interpreted as changing runner expressiveness and may contribute to broad prevalence increases.\n\n## Sensitivity analysis\n\nThe model confidence field ranges from {confidence['min']:.2f} to {confidence['max']:.2f}, with median {confidence['50%']:.2f} and 95th percentile {confidence['95%']:.2f}. It is highly compressed, model-self-reported, and not calibrated against frozen gold. No post-hoc confidence threshold was selected. Document-level presence, rather than independent mention rows, is the implemented robustness protection.\n\n## Model limitation\n\n{MODEL_LIMITATION} Small differences, borderline results, and lower-support aspects require particular caution. P-values do not prove true population sentiment differences or causal temporal change.\n\n## Recommended research findings\n\nUse only rows in the research summary that combine adequate support, FDR significance, non-trivial Cramer's V (≥0.10), and coherent year-specific estimates. Phrase them as associations in model-estimated labels.\n\n## Findings NOT to emphasize\n\nDo not emphasize low/very-low-support aspects, trivial effects despite small p-values, borderline p-values, sparse neutral/mixed trends, language differences, or topic-wide inferential comparisons. Language and topic results remain descriptive.\n\n## Execution integrity\n\nOpenAI calls = 0; Hugging Face inference = 0; ABSA inference = 0; manual review = 0; production predictions modified = false.\n"""
    (root / "ABSA_V1_INFERENTIAL_ANALYSIS_REPORT.md").write_text(report, encoding="utf-8")

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip() or None
    except OSError:
        commit = None
    manifest = {
        "protocol": "absa_v1_inferential_analysis_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifacts": {path: {"sha256": digest} for path, digest in source_hashes.items()},
        "reconciliation": {"documents": len(documents), "mentions": len(mentions), "mention_bearing_documents": EXPECTED_BEARING,
                           "zero_mention_documents": EXPECTED_ZERO, "year_counts": expected_years},
        "analysis_code": "marathon_absa.absa_inferential_analysis", "git_commit": commit,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "statsmodels": "not_installed; logistic MLE implemented with scipy.optimize"},
        "alpha": ALPHA, "multiple_testing": {"omnibus": "Benjamini-Hochberg within each family", "pairwise": "Holm within aspect/outcome"},
        "analysis_families": {"A": "aspect prevalence by year", "B": "positive presence among aspect-bearing documents by year",
                              "C": "negative presence among aspect-bearing documents by year"},
        "support_eligibility_rule": f"moderate/high support and >= {MIN_YEAR_ASPECT_DOCUMENTS} aspect-bearing documents in each year",
        "eligible_aspects_frozen_before_testing": eligible, "confidence_sensitivity": "skipped_unvalidated_compressed_confidence",
        "effect_size_cutoffs": {"trivial": "<0.10", "small": "0.10-<0.20", "moderate": "0.20-<0.30", "strong": ">=0.30"},
        "openai_calls": 0, "hugging_face_inference": 0, "absa_inference": 0, "manual_review": 0,
        "production_predictions_modified": False, "model_limitation": MODEL_LIMITATION,
    }
    artifact_names = [path.name for path in root.iterdir() if path.is_file()]
    manifest["output_artifacts"] = {name: {"sha256": _sha(root / name)} for name in artifact_names}
    (root / "absa_v1_inferential_analysis_manifest_v1.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if {str(path): _sha(path) for path in source_paths} != source_hashes:
        raise RuntimeError("Frozen production/descriptive source changed during analysis")
    return {"output_root": str(root), "eligible_aspects": eligible, "significant_prevalence": significant_a,
            "significant_positive": significant_b, "significant_negative": significant_c,
            "documents": len(documents), "mentions": len(mentions), "api_calls": 0}
