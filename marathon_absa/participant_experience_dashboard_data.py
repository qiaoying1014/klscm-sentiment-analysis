"""Read-only validation for the downstream participant-experience release."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path('data/processed/participant_experience_v2_rt08')
VERSION = 'participant_experience_v1'
RELEASE_VERSION = 'participant_experience_v2_rt08'
SENTIMENTS = ('positive', 'negative', 'mixed', 'neutral')
FIELDS = ('participant_experience_summary', 'positive_experience_summary',
          'negative_experience_summary', 'mixed_experience_summary',
          'organizer_insight', 'organizer_implication', 'evidence_scope_note')
CATEGORIES = ('Positive experience driver', 'Participant pain point',
              'Mixed / contested experience', 'Primarily descriptive / insufficient evaluative evidence')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def read(path: Path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def classify(sentiment: dict) -> str:
    """Presence rule, not dominance, estimated opinion, or statistical confidence."""
    labels = {s for source in sentiment.values() for s, n in source['counts'].items() if n > 0}
    if 'mixed' in labels or {'positive', 'negative'} <= labels:
        return CATEGORIES[2]
    if 'positive' in labels:
        return CATEGORIES[0]
    if 'negative' in labels:
        return CATEGORIES[1]
    return CATEGORIES[3]


def descriptor(ig: int, blog: int) -> str:
    if ig and blog:
        return 'Cross-source recurring evidence' if ig > 1 and blog > 1 else 'Cross-source evidence; limited within-source repetition'
    if ig:
        return 'Repeated Instagram evidence' if ig > 1 else 'Singleton observation'
    return 'Multiple blog-review support' if blog > 1 else 'Singleton blog-only emergent observation'


def validate_package(package: dict) -> None:
    if package.get('version') != VERSION or package.get('pooled_prevalence') is not False:
        raise ValueError('Invalid version or pooled denominator policy')
    if set(package['source_units']) != {'instagram', 'blog'}:
        raise ValueError('Separate source units required')
    if package['source_units']['instagram']['unit'] != 'documents' or package['source_units']['blog']['unit'] != 'parent_reviews':
        raise ValueError('Invalid source denominator units')
    aspects = package['aspects']
    if len(aspects) != 20 or not all(aspects.values()):
        raise ValueError('Existing 20-aspect ontology required')
    if not package.get('input_hashes'):
        raise ValueError('Missing input provenance')
    seen = set()
    for row in package['themes']:
        identity = row['theme_id']
        if identity in seen or row['aspect_id'] not in aspects or row['aspect_label'] != aspects[row['aspect_id']]:
            raise ValueError('Invalid aspect or duplicate theme reference')
        seen.add(identity)
        if not row.get('theme_label') or row['theme_origin'] not in {'instagram_reviewed_taxonomy', 'blog_emergent'}:
            raise ValueError('Invalid theme reference')
        ig, blog = row['instagram_support_documents'], row['blog_support_reviews']
        if any(type(n) is not int or n < 0 for n in (ig, blog)) or not (ig or blog):
            raise ValueError('Invalid support')
        expected = 'blog_only_emergent' if row['theme_origin'] == 'blog_emergent' else 'cross_source' if blog else 'instagram_only'
        if row['source_coverage'] != expected or (expected == 'blog_only_emergent' and ig) or (expected != 'blog_only_emergent' and not ig):
            raise ValueError('Invalid source coverage')
        if ig > package['source_units']['instagram']['total'] or blog > package['source_units']['blog']['total']:
            raise ValueError('Support exceeds source denominator')
        if set(row['sentiment_evidence']) != {'instagram', 'blog'}:
            raise ValueError('Missing sentiment source')
        for source, unit in [('instagram', 'theme_documents'), ('blog', 'assigned_mentions')]:
            part = row['sentiment_evidence'][source]
            if part['unit'] != unit or set(part['counts']) != set(SENTIMENTS) or any(type(n) is not int or n < 0 for n in part['counts'].values()):
                raise ValueError('Malformed sentiment evidence')
        if sum(row['sentiment_evidence']['instagram']['counts'].values()) != ig:
            raise ValueError('Instagram sentiment/support mismatch')
        evidence = row['representative_evidence']
        keys = [e['evidence_id'] for e in evidence]
        if not keys or len(set(keys)) != len(keys):
            raise ValueError('Missing or duplicate evidence references')
        for e in evidence:
            if e['source'] not in {'instagram', 'blog'} or e['sentiment'] not in SENTIMENTS or not e['text'] or not e['source_evidence_id'] or not e['parent_id']:
                raise ValueError('Malformed evidence reference')
        blog_evidence = [e for e in evidence if e['source'] == 'blog']
        if len({e['parent_id'] for e in blog_evidence}) != blog:
            raise ValueError('Blog parent review support mismatch')
        if {s: sum(e['sentiment'] == s for e in blog_evidence) for s in SENTIMENTS} != row['sentiment_evidence']['blog']['counts']:
            raise ValueError('Blog mention sentiment mismatch')
        if row['experience_classification'] != classify(row['sentiment_evidence']) or row['evidence_descriptor'] != descriptor(ig, blog):
            raise ValueError('Invalid deterministic interpretation')
        if not row['scope_note'] or (blog == 1 and 'single' not in row['scope_note'].lower()):
            raise ValueError('Singleton caution required')


def load_evidence(root: Path = ROOT) -> dict:
    manifest = read(root / 'evidence_manifest.json')
    if sha(root / 'evidence.json') != manifest['evidence_sha256']:
        raise ValueError('Evidence hash mismatch')
    package = read(root / 'evidence.json')
    if package.get('correction', {}).get('correction_version') == RELEASE_VERSION:
        # The versioned rt08 release intentionally uses existing assignment-lineage
        # evidence absent from v1's narrower representative-evidence catalog.
        # Its dedicated verifier retains the normal frozen-source checks plus that
        # explicit, fixed lineage check.
        from .participant_experience_rt08_v2 import _load_v2_evidence
        return _load_v2_evidence(root)
    validate_package(package)
    if package['input_hashes'] != manifest['input_hashes']:
        raise ValueError('Missing or inconsistent input provenance')
    for name, expected in package['input_hashes'].items():
        if sha(Path(name)) != expected:
            raise ValueError('Frozen source hash mismatch: ' + name)
    validate_frozen_references(package)
    return package


def validate_frozen_references(package: dict) -> None:
    """Check release identities against the actual unchanged reviewed catalogs."""
    def table(name):
        if name not in package['input_hashes']:
            raise ValueError('Required frozen input absent from provenance: ' + name)
        with Path(name).open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))
    ig_root = 'data/processed/absa_v1/aspect_level_themes_review_v1/'
    blog_root = 'data/processed/blog_analysis_v1/'
    # Paths are normalized by Path in the input manifest on each platform.
    def rows(name): return table(str(Path(name)))
    metadata_name = str(Path('data/processed/absa_v1/dashboard/absa_v1_dashboard_data_v1/dashboard_metadata_v1.json'))
    if metadata_name not in package['input_hashes'] or package['aspects'] != read(Path(metadata_name))['aspect_display_labels']:
        raise ValueError('Unknown aspect IDs or altered display mapping')
    if package['source_units']['instagram']['total'] != read(Path(metadata_name))['production_documents'] or package['source_units']['blog']['total'] != len(rows(blog_root + 'blog_review_summary_v1.csv')):
        raise ValueError('Changed source denominators')
    ig = {r['reviewed_theme_id']: r for r in rows(ig_root + 'reviewed_theme_summary.csv')}
    blog = {r['emergent_theme_id']: r for r in rows(blog_root + 'blog_emergent_theme_taxonomy_v1.csv')}
    if {r['theme_id'] for r in package['themes']} != ig.keys() | blog.keys():
        raise ValueError('Invalid theme references')
    original_evidence = {'instagram:' + r['mention_id']: r for r in rows(ig_root + 'reviewed_theme_evidence.csv')}
    original_evidence.update({'blog:' + r['blog_mention_id']: r for r in rows(blog_root + 'blog_absa_mentions_v1.csv')})
    blog_members = {}
    for r in rows(blog_root + 'blog_reviewed_theme_assignments_v1.csv'):
        if r['theme_origin'] == 'instagram_reviewed_taxonomy':
            blog_members.setdefault(r['theme_id'], set()).add('blog:' + r['blog_mention_id'])
    for r in rows(blog_root + 'blog_emergent_theme_lineage_v1.csv'):
        if r['emergent_theme_id']:
            blog_members.setdefault(r['emergent_theme_id'], set()).add('blog:' + r['blog_mention_id'])
    from .blog_emergent_themes import public_text
    for theme in package['themes']:
        identity = theme['theme_id']
        original = ig.get(identity, blog.get(identity))
        label = original['reviewed_theme_label'] if identity in ig else original['emergent_theme_label']
        if theme['aspect_id'] != original['aspect'] or theme['theme_label'] != public_text(label):
            raise ValueError('Theme label or aspect differs from reviewed taxonomy')
        expected_counts = {s: int(original[s + '_documents']) if identity in ig else 0 for s in SENTIMENTS}
        if theme['sentiment_evidence']['instagram']['counts'] != expected_counts:
            raise ValueError('Instagram sentiment differs from frozen theme evidence')
        if {e['evidence_id'] for e in theme['representative_evidence'] if e['source'] == 'blog'} != blog_members.get(identity, set()):
            raise ValueError('Blog evidence differs from finalized theme mapping')
        for e in theme['representative_evidence']:
            if e['evidence_id'] not in original_evidence:
                raise ValueError('Unresolved original evidence ID')
            o = original_evidence[e['evidence_id']]
            if o['aspect'] != theme['aspect_id'] or o['sentiment'] != e['sentiment'] or public_text(o['evidence_text']) != e['text']:
                raise ValueError('Evidence differs from frozen original')
            if e['source'] == 'instagram' and (o['reviewed_theme_id'] != identity or o['document_id'] != e['parent_id']):
                raise ValueError('Invalid Instagram evidence lineage')
            if e['source'] == 'blog' and o['review_id'] != e['parent_id']:
                raise ValueError('Invalid blog parent lineage')
        expected_missing = {key for key, value in original_evidence.items() if key.startswith('instagram:') and value['reviewed_theme_id'] == identity and not value['evidence_text'].strip()}
        if set(theme.get('unavailable_evidence_ids', [])) != expected_missing:
            raise ValueError('Missing excerpt provenance differs from original')


def validate_insight(row: dict, theme: dict) -> None:
    if row.get('theme_id') != theme['theme_id'] or set(row.get('claims', {})) != set(FIELDS):
        raise ValueError('Invalid theme or insight fields')
    available = {e['evidence_id']: e for e in theme['representative_evidence']}
    for field, claim in row['claims'].items():
        if set(claim) != {'text', 'evidence_ids'} or not isinstance(claim['text'], str) or not claim['text'].strip():
            raise ValueError('Malformed synthesis claim')
        ids = claim['evidence_ids']
        if not isinstance(ids, list) or len(ids) != len(set(ids)) or not set(ids) <= available.keys():
            raise ValueError('Unresolved evidence reference')
        relevant = {'positive_experience_summary': 'positive', 'negative_experience_summary': 'negative'}
        if field in relevant and ids and not any(available[x]['sentiment'] in {relevant[field], 'mixed'} for x in ids):
            raise ValueError('Sentiment claim lacks matching evidence')
        if not ids and (field not in {'positive_experience_summary', 'negative_experience_summary', 'mixed_experience_summary'} or claim['text'] != 'Insufficient evidence at this level.'):
            raise ValueError('Organizer implications and observations require evidence')
    if theme['blog_support_reviews'] == 1 and 'single' not in row['claims']['evidence_scope_note']['text'].lower():
        raise ValueError('Singleton blog caution required')
    if not row['claims']['organizer_implication']['text'].startswith('Organizers may consider'):
        raise ValueError('Cautious implication required')


def load_participant_experience_dashboard_data(root: Path = ROOT) -> dict:
    """Only released, fully reviewed synthesis is returned; no fallback to candidates."""
    from .cloud_bundle import load_cloud_bundle
    bundle = load_cloud_bundle("experience") if root == ROOT else None
    if bundle is not None:
        validate_package(bundle['evidence'])
        themes = {row['theme_id']: row for row in bundle['evidence']['themes']}
        for row in bundle['insights']:
            validate_insight(row, themes[row['theme_id']])
        return bundle
    final = read(root / 'finalized_manifest.json')
    if final.get('review_status') != 'finalized' or not final.get('reviewer') or not final.get('finalized_at'):
        raise ValueError('Unreviewed synthesis cannot be displayed')
    if final.get('version') not in {VERSION, RELEASE_VERSION}:
        raise ValueError('Unknown finalized release version')
    for name, expected in final['hashes'].items():
        if sha(root / name) != expected:
            raise ValueError('Finalized synthesis hash mismatch')
    required = {'evidence.json', 'evidence_manifest.json', 'requests.jsonl', 'request_manifest.json', 'candidates.json', 'review.json', 'finalized.json'}
    correction = read(root / 'evidence.json').get('correction', {})
    expected = required | ({'correction_manifest.json'} if correction.get('correction_version') == RELEASE_VERSION else set())
    if set(final['hashes']) != expected:
        raise ValueError('Incomplete release provenance')
    if correction.get('correction_version') == RELEASE_VERSION and final.get('version') != RELEASE_VERSION:
        raise ValueError('Versioned evidence/finalization mismatch')
    package = load_evidence(root)
    request = read(root / 'request_manifest.json')
    if not request.get('model') or not request.get('prompt_hash') or not request.get('prompt_version') or request['input_artifact_hash'] != sha(root / 'evidence.json'):
        raise ValueError('Missing generation provenance')
    candidates = read(root / 'candidates.json')
    if not candidates.get('generated_at') or not candidates.get('response_hash') or candidates.get('request_manifest_hash') != sha(root / 'request_manifest.json') or candidates.get('model') != request['model']:
        raise ValueError('Incomplete candidate provenance')
    candidate_map = {r['theme_id']: r for r in candidates['insights']}
    result = read(root / 'finalized.json')
    themes = {t['theme_id']: t for t in package['themes']}
    reviews = read(root / 'review.json')
    if len(reviews) != len(themes) or len({r['theme_id'] for r in reviews}) != len(themes) or any(r.get('decision') != 'APPROVE' or not r.get('reviewer') or not r.get('reason') for r in reviews):
        raise ValueError('Unreviewed synthesis')
    if len(result) != len(themes) or {r['theme_id'] for r in result} != themes.keys():
        raise ValueError('Invalid theme references or incomplete release')
    review_map = {r['theme_id']: r for r in reviews}
    for row in result:
        validate_insight(row, themes[row['theme_id']])
        if row != review_map[row['theme_id']]['insight']:
            raise ValueError('Finalized insight differs from reviewed content')
        if row['theme_id'] not in candidate_map or review_map[row['theme_id']]['candidate_hash'] != digest(candidate_map[row['theme_id']]):
            raise ValueError('Stale candidate review provenance')
    return {'evidence': package, 'insights': result, 'provenance': {**final, 'synthesis': request,
            'generated_at': candidates['generated_at'], 'response_hash': candidates['response_hash']}}


def filter_themes(themes: list, aspect=None, category=None, coverage=None) -> list:
    return copy.deepcopy([r for r in themes if (not aspect or r['aspect_id'] == aspect)
                          and (not category or r['experience_classification'] == category)
                          and (not coverage or r['source_coverage'] == coverage)])
