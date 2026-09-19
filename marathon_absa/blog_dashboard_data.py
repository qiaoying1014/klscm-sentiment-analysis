"""Read-only, validated long-form dashboard views with parent-review denominators."""
from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from .blog_emergent_themes import ROOT, finalized_summary, public_text, sha

CROSS_ROOT = Path('data/processed/cross_source_analysis_v1')


@dataclass(frozen=True)
class BlogDashboardData:
    counts: dict
    aspects: pd.DataFrame
    sentiment: pd.DataFrame
    matched: pd.DataFrame
    emergent: pd.DataFrame
    years: pd.DataFrame
    languages: pd.DataFrame


def load_blog_dashboard_data(blog_root: Path = ROOT, cross_root: Path = CROSS_ROOT) -> BlogDashboardData:
    """Fail closed on stale marts; never pool source populations or expose identity fields."""
    from .cloud_bundle import load_cloud_bundle
    bundle = load_cloud_bundle("blog") if blog_root == ROOT and cross_root == CROSS_ROOT else None
    if bundle is not None:
        return BlogDashboardData(**bundle)
    emergent = finalized_summary(blog_root)
    manifest = json.loads((cross_root / 'cross_source_manifest_v1.json').read_text(encoding='utf-8'))
    if (manifest.get('comparison_type') != 'descriptive_triangulation'
            or manifest.get('pooled_prevalence') is not False
            or manifest.get('inferential_tests') != 0
            or manifest.get('instagram_total_documents') != 7704
            or manifest.get('blog_total_reviews') != 25):
        raise ValueError('Cross-source analysis does not preserve separate descriptive denominators')
    for path, expected in manifest['source_hashes'].items():
        if sha(Path(path)) != expected:
            raise ValueError('Cross-source inputs changed; regenerate the comparison')
    absa = json.loads((blog_root / 'blog_absa_manifest_v1.json').read_text(encoding='utf-8'))
    for name, expected in absa['hashes'].items():
        if sha(blog_root / name) != expected:
            raise ValueError('Blog ABSA inputs changed')
    reviews = pd.read_csv(blog_root / 'blog_review_summary_v1.csv')
    aspects = pd.read_csv(cross_root / 'cross_source_aspect_comparison_v1.csv')
    matched = pd.read_csv(cross_root / 'cross_source_theme_comparison_v1.csv')
    sentiment = pd.read_csv(blog_root / 'blog_aspect_sentiment_summary_v1.csv')
    published = pd.read_csv(cross_root / 'blog_emergent_theme_summary_v1.csv')
    if len(reviews) != 25 or reviews.review_id.nunique() != 25 or reviews.mention_count.sum() != absa['mention_count']:
        raise ValueError('Blog review/mention counts do not reconcile')
    if reviews.chunk_count.sum() != absa['successful_chunks'] or absa['successful_chunks'] != absa['expected_chunks']:
        raise ValueError('Blog inference is incomplete')
    if len(aspects) != 20 or not aspects.instagram_total_documents.eq(7704).all() or not aspects.blog_total_reviews.eq(25).all():
        raise ValueError('Aspect comparison denominators changed')
    if len(matched) != 64 or matched.reviewed_theme_id.duplicated().any() or int(matched.present_in_blogs.sum()) != 47:
        raise ValueError('Matched-theme comparison changed')
    if not sentiment.total_included_reviews.eq(25).all():
        raise ValueError('Blog sentiment denominator changed')
    support = aspects.set_index('aspect').blog_support_reviews.sort_index()
    sentiment_support = sentiment.groupby('aspect').support_reviews.sum().sort_index()
    if not support.equals(sentiment_support):
        raise ValueError('Review-level sentiment does not reconcile to aspect support')
    expected = emergent[['theme_id','theme_label','aspect','support_reviews','total_included_reviews','review_prevalence','support_mentions','singleton_flag','low_support_flag']].copy()
    expected['theme_label'] = expected.theme_label.map(public_text)
    try:
        pd.testing.assert_frame_equal(expected.sort_values('theme_id').reset_index(drop=True),
            published[expected.columns].sort_values('theme_id').reset_index(drop=True), check_dtype=False)
    except AssertionError as error:
        raise ValueError('Published emergent summary is stale') from error
    lineage = emergent.set_index('theme_id').source_existing_theme_ids
    # Public display allowlists deliberately omit authors, URLs and parent/chunk identifiers.
    emergent = published[[*expected.columns, 'representative_evidence']].copy()
    decisions = pd.read_csv(blog_root / 'blog_emergent_theme_decisions_v1.csv', keep_default_na=False)
    notes = decisions.set_index('existing_theme_id').reason.to_dict()
    emergent['scope_note'] = emergent.theme_id.map(lambda key: ' '.join(dict.fromkeys(notes[x] for x in json.loads(lineage[key]))))
    for column in ['theme_label','representative_evidence','scope_note']:
        emergent[column] = emergent[column].map(public_text)
    matched = matched[['aspect','reviewed_theme_id','reviewed_theme_label','instagram_support_documents','blog_support_reviews','present_in_blogs']].copy()
    matched['reviewed_theme_label'] = matched.reviewed_theme_label.map(public_text)
    sentiment['review_prevalence'] = sentiment.support_reviews / 25
    sentiment = sentiment[['aspect','review_aspect_sentiment','support_reviews','review_prevalence']]
    years = reviews.groupby('event_year', as_index=False).agg(support_reviews=('review_id','nunique'))
    languages = reviews.groupby('primary_language', as_index=False).agg(support_reviews=('review_id','nunique'))
    counts = dict(reviews=len(reviews), mentions=int(reviews.mention_count.sum()), chunks=int(reviews.chunk_count.sum()),
        aspects=len(aspects), matched=int(matched.present_in_blogs.sum()), reference_themes=len(matched),
        emergent=len(emergent), singletons=int(emergent.singleton_flag.sum()))
    return BlogDashboardData(counts, aspects, sentiment, matched, emergent, years, languages)
