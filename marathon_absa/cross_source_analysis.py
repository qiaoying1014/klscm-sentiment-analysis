from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("data/processed/cross_source_analysis_v1")
BLOG_ROOT = Path("data/processed/blog_analysis_v1")
INSTA_ROOT = Path("data/processed/absa_v1/production/absa_v1_production_v1")
THEME_ROOT = Path("data/processed/absa_v1/aspect_level_themes_review_v1")
EXPECTED_INSTAGRAM_DOCUMENTS = 7704
EXPECTED_INSTAGRAM_MENTIONS = 15486


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(frame: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(stem.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    frame.to_parquet(stem.with_suffix(".parquet"), index=False)


def create_cross_source_analysis(root: Path = ROOT) -> dict[str, Any]:
    docs_path = INSTA_ROOT / "absa_v1_production_document_results_v1.csv"
    mentions_path = INSTA_ROOT / "absa_v1_production_mentions_v1.csv"
    insta_docs, insta_mentions = pd.read_csv(docs_path, low_memory=False), pd.read_csv(mentions_path, low_memory=False)
    if len(insta_docs) != EXPECTED_INSTAGRAM_DOCUMENTS or len(insta_mentions) != EXPECTED_INSTAGRAM_MENTIONS:
        raise RuntimeError("Frozen Instagram ABSA quantities changed")
    blog_reviews = pd.read_csv(BLOG_ROOT / "blog_review_summary_v1.csv")
    blog_presence = pd.read_csv(BLOG_ROOT / "blog_review_aspect_presence_v1.csv")
    insta_presence = insta_mentions[["document_id", "aspect"]].drop_duplicates()
    ia = insta_presence.groupby("aspect").document_id.nunique(); ba = blog_presence.groupby("aspect").review_id.nunique()
    aspects = sorted(set(ia.index) | set(ba.index)); rows = []
    for aspect in aspects:
        i, b = int(ia.get(aspect, 0)), int(ba.get(aspect, 0))
        status = "observed_in_both" if i and b else "instagram_only_observed" if i else "blog_only_observed"
        rows.append({"aspect": aspect, "instagram_support_documents": i, "instagram_total_documents": len(insta_docs),
                     "instagram_prevalence": i / len(insta_docs), "blog_support_reviews": b, "blog_total_reviews": len(blog_reviews),
                     "blog_prevalence": b / len(blog_reviews), "comparison_status": status})
    aspect_comparison = pd.DataFrame(rows)
    blog_themes = pd.read_csv(BLOG_ROOT / "blog_reviewed_theme_summary_v1.csv")
    instagram_themes = pd.read_csv(THEME_ROOT / "reviewed_theme_summary.csv")
    reviewed_blog = blog_themes[blog_themes.theme_origin.eq("instagram_reviewed_taxonomy")]
    theme_comparison = instagram_themes[["aspect", "reviewed_theme_id", "reviewed_theme_label", "support_documents"]].rename(columns={"support_documents": "instagram_support_documents"})
    theme_comparison = theme_comparison.merge(reviewed_blog[["theme_id", "support_reviews"]].rename(columns={"theme_id": "reviewed_theme_id", "support_reviews": "blog_support_reviews"}), on="reviewed_theme_id", how="left")
    theme_comparison["blog_support_reviews"] = theme_comparison.blog_support_reviews.fillna(0).astype(int)
    theme_comparison["present_in_instagram"] = theme_comparison.instagram_support_documents.gt(0); theme_comparison["present_in_blogs"] = theme_comparison.blog_support_reviews.gt(0)
    from .blog_emergent_themes import finalized_summary, public_text
    emergent = finalized_summary(BLOG_ROOT)
    if len(blog_reviews) != 25 or blog_reviews.review_id.nunique() != 25:
        raise ValueError("Blog denominator must be 25 parent reviews")
    if len(theme_comparison) != 64 or int(theme_comparison.present_in_blogs.sum()) != 47:
        raise ValueError("Frozen matched-theme comparison changed")
    if len(aspect_comparison) != 20 or not aspect_comparison.comparison_status.eq("observed_in_both").all():
        raise ValueError("Frozen aspect comparison changed")
    for column in ["theme_label", "representative_evidence", "evidence_gloss_en"]:
        emergent[column] = emergent[column].map(public_text)
    for frame, name in [(aspect_comparison, "cross_source_aspect_comparison_v1"), (theme_comparison, "cross_source_theme_comparison_v1"), (emergent, "blog_emergent_theme_summary_v1")]: _write(frame, root / name)
    sources = [docs_path, mentions_path, THEME_ROOT / "review_manifest.json", BLOG_ROOT / "blog_theme_manifest_v1.json",
               BLOG_ROOT / "blog_emergent_theme_taxonomy_v1.csv",
               BLOG_ROOT / "blog_emergent_theme_finalization_v1.json"]
    manifest = {"stage": "cross_source_analysis_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "comparison_type": "descriptive_triangulation", "pooled_prevalence": False, "inferential_tests": 0,
                "instagram_total_documents": len(insta_docs), "blog_total_reviews": len(blog_reviews),
                "source_hashes": {str(p): _sha(p) for p in sources}, "api_calls": 0}
    root.mkdir(parents=True, exist_ok=True); (root / "cross_source_manifest_v1.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
