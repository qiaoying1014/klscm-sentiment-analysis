"""Offline, separately reviewed blog-emergent taxonomy. No upstream writes."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

ROOT = Path('data/processed/blog_analysis_v1')
AUDIT = 'blog_emergent_theme_audit_v1'
DECISION_FILE = 'blog_emergent_theme_decisions_v1.csv'
TAXONOMY = 'blog_emergent_theme_taxonomy_v1.csv'
MANIFEST = 'blog_emergent_theme_finalization_v1.json'
BASELINE = 'blog_emergent_frozen_baseline_v1.json'
MISSING = 'LABEL_MISSING_RESEARCHER_REVIEW_REQUIRED'
DECISIONS = {'KEEP_SEPARATE', 'MERGE_WITH_EXISTING_EMERGENT', 'RENAME_ONLY', 'EXCLUDE_AS_NON_THEME', 'UNCLEAR'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_write(path, text):
    """Atomic replacement with content-addressed backup of every prior version."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = path.parent / 'emergent_backups' / (path.name + '.' + sha(path) + '.bak')
        backup.parent.mkdir(exist_ok=True)
        if not backup.exists():
            shutil.copy2(path, backup)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read(path):
    return pd.read_csv(path, keep_default_na=False, dtype=str)


def normalize(label):
    return ' '.join(unicodedata.normalize('NFKC', str(label)).casefold().split())


def readable(label):
    return bool(str(label).strip()) and label != MISSING and not str(label).startswith('blog_emergent__') and bool(re.search(r'[A-Za-z]{3}', str(label)))


def stable_id(aspect, label, source_ids):
    sources = sorted(set(source_ids))
    if len(sources) == 1:
        return sources[0]
    value = json.dumps([aspect, normalize(label), sources], ensure_ascii=False)
    return 'blog_emergent__' + aspect + '__' + hashlib.sha256(value.encode()).hexdigest()[:16]


def load_source(root=ROOT):
    source = read(root / 'blog_emergent_themes_v1.csv')
    mapping = read(root / 'blog_theme_mapping_candidates_v1.csv')
    marked = mapping[mapping.mapping_decision.eq('EMERGENT_BLOG_THEME')]
    if source.blog_mention_id.duplicated().any() or set(source.blog_mention_id) != set(marked.blog_mention_id):
        raise ValueError('Source emergent mention coverage/uniqueness failure')
    columns = ['review_id', 'chunk_id', 'aspect', 'evidence_text', 'english_gloss', 'emergent_blog_theme_label']
    for col in columns:
        if source.set_index('blog_mention_id')[col].sort_index().to_dict() != marked.set_index('blog_mention_id')[col].sort_index().to_dict():
            raise ValueError('Source mapping lineage differs: ' + col)
    return source.sort_values(['aspect', 'theme_id', 'review_id', 'blog_mention_id'])


def verify_integrity(root=ROOT):
    baseline = json.loads((root / BASELINE).read_text(encoding='utf-8'))
    if not baseline:
        raise ValueError('Empty frozen baseline')
    changed = [p for p, expected in baseline.items() if not Path(p).exists() or sha(p) != expected]
    if changed:
        raise ValueError('Frozen upstream hashes differ: ' + ', '.join(changed))
    # Existing verifier writes its report only to an isolated temporary directory.
    from .blog_analysis import verify_frozen_upstream, INSTRUCTIONS, EXPECTED_PROMPT_SHA256
    with tempfile.TemporaryDirectory() as temporary:
        checks = verify_frozen_upstream(Path(temporary))['checks']
    if hashlib.sha256(INSTRUCTIONS.encode()).hexdigest() != EXPECTED_PROMPT_SHA256:
        raise ValueError('Frozen ABSA prompt differs')
    return {'all_passed': True, 'files_verified': len(baseline), 'checks': checks, 'prompt_sha256': EXPECTED_PROMPT_SHA256}


def audit(root=ROOT):
    integrity = verify_integrity(root)
    source = load_source(root)
    from .blog_analysis import included_reviews
    reviews, chunks = included_reviews()
    parents = read(root / 'blog_review_summary_v1.csv').review_id.tolist()
    chunk_parents = chunks.set_index('unit_id').document_id.to_dict()
    if len(parents) != 25 or len(set(parents)) != 25 or set(parents) != set(reviews.document_id):
        raise ValueError('Audit requires exactly 25 usable parent reviews')
    if any(chunk_parents.get(row.chunk_id) != row.review_id for row in source.itertuples()):
        raise ValueError('Audit chunk/parent lineage mismatch')
    rows = []
    for identity, group in source.groupby('theme_id', sort=True):
        labels = sorted(set(group.emergent_blog_theme_label))
        if group.aspect.nunique() != 1 or len(labels) != 1:
            raise ValueError('Inconsistent source concept identity')
        rows.append(dict(existing_theme_id=identity, aspect=group.aspect.iloc[0],
            researcher_label_original=labels[0] or MISSING,
            parent_review_count=group.review_id.nunique(), mention_count=len(group),
            parent_review_ids=json.dumps(sorted(set(group.review_id))),
            chunk_ids=json.dumps(sorted(set(group.chunk_id))),
            representative_evidence=group.evidence_text.iloc[0], evidence_gloss_en=group.english_gloss.iloc[0],
            suspected_duplicate_group='', audit_status='RESEARCHER_CONSOLIDATION_REVIEW_REQUIRED',
            notes='Original labels are delegated AI-assisted review, not independent human validation. Distinct labels require consolidation review.'))
    frame = pd.DataFrame(rows)
    scope_notes = {
        'Overall course quality and difficulty': 'Scope review: combines generic route approval and technical difficulty; do not merge with course width merely because both evaluate the route.',
        'Race-village space and ground conditions': 'Scope review: combines space and muddy ground. Keep frozen facilities aspect; weather-related mud has a different frozen aspect and cannot be merged.',
        'Contributing as an expo volunteer': 'Scope review: evidence explicitly mentions spreading the word on GCM. Researcher must check comparator-event scope before retaining or excluding; no exclusion inferred.',
        'Emotional response to non-completion': 'Related to injury/setback emotions but non-completion is a distinct trigger. Equivalence is not established.',
        'Emotional response to injury and setbacks': 'Related to non-completion emotions but injury/setback is a distinct trigger. Equivalence is not established.',
    }
    for label, note in scope_notes.items():
        mask = frame.researcher_label_original.eq(label)
        frame.loc[mask, 'notes'] = frame.loc[mask, 'notes'] + ' ' + note
    candidates = []
    for aspect, group in frame.groupby('aspect'):
        records = group.to_dict('records')
        for i, left in enumerate(records):
            for right in records[i+1:]:
                score = SequenceMatcher(None, normalize(left['researcher_label_original']), normalize(right['researcher_label_original'])).ratio()
                candidates.append(dict(aspect=aspect, source_id=left['existing_theme_id'], candidate_id=right['existing_theme_id'],
                    label_similarity=score, method='lexical_sequence_ratio_advisory_only', decision='NOT_REVIEWED'))
    pairs = pd.DataFrame(candidates, columns=['aspect','source_id','candidate_id','label_similarity','method','decision'])
    # All same-aspect pairs are supplied; no threshold implies a suspected duplicate or merge.
    for identity in frame.existing_theme_id:
        peers = pairs[(pairs.source_id == identity) | (pairs.candidate_id == identity)]
        frame.loc[frame.existing_theme_id == identity, 'suspected_duplicate_group'] = 'UNASSESSED' if len(peers) else 'NO_SAME_ASPECT_PEER'
    summary = dict(raw_emergent_records=len(source), unique_emergent_ids=len(frame), reviews_represented=source.review_id.nunique(),
        aspects_represented=source.aspect.nunique(), ids_per_aspect=frame.groupby('aspect').size().to_dict(),
        original_labels_available=int(frame.researcher_label_original.ne(MISSING).sum()), original_labels_missing=int(frame.researcher_label_original.eq(MISSING).sum()),
        singleton_count=int(frame.parent_review_count.eq(1).sum()), confirmed_duplicate_groups=0, suspected_duplicate_groups='No evidence-equivalent duplicate established; all 39 same-aspect pairs remain available for researcher screening',
        same_aspect_candidate_pairs=len(pairs), evidence_recoverable=bool(source.evidence_text.str.strip().ne('').all()),
        id_origin='aspect plus SHA256(exact original label)[:8]; per concept, not per observation',
        denominator=25, low_support_rule='support_reviews < 2', review_support_reconstructable=True,
        lifecycle_status='researcher_consolidation_review_required', frozen_hash_verification=integrity, api_calls=0)
    safe_write(root / (AUDIT+'.csv'), frame.to_csv(index=False))
    safe_write(root / (AUDIT+'.json'), json.dumps(summary, indent=2))
    safe_write(root / 'blog_emergent_theme_candidates_v1.csv', pairs.to_csv(index=False))
    if not (root / DECISION_FILE).exists():
        decisions = frame[['existing_theme_id']].copy()
        decisions['researcher_decision'] = 'UNCLEAR'
        decisions['target_existing_theme_id'] = ''
        decisions['final_label'] = frame.researcher_label_original
        decisions['reason'] = ''
        safe_write(root / DECISION_FILE, decisions.to_csv(index=False))
    return summary


def build_taxonomy(source, decisions, review_ids):
    """Pure validation/aggregation; every source mention remains in lineage, including exclusions."""
    if len(review_ids) != 25 or len(set(review_ids)) != 25:
        raise ValueError('Prevalence denominator must be exactly 25 unique parent reviews')
    if not set(source.review_id).issubset(set(review_ids)):
        raise ValueError('Unknown support review ID')
    if source.blog_mention_id.duplicated().any() or source.evidence_text.str.strip().eq('').any():
        raise ValueError('Duplicate mention lineage or missing evidence')
    ids = set(source.theme_id)
    if decisions.existing_theme_id.duplicated().any() or set(decisions.existing_theme_id) != ids:
        raise ValueError('Every source emergent ID must be accounted for exactly once')
    lookup = decisions.set_index('existing_theme_id').to_dict('index')
    aspects = source.groupby('theme_id').aspect.agg(lambda x: sorted(set(x))).to_dict()
    if any(len(x) != 1 for x in aspects.values()):
        raise ValueError('Source ID crosses aspect boundaries')
    groups = {}
    excluded = []
    for identity in sorted(ids):
        row = lookup[identity]
        decision = row['researcher_decision']
        if decision not in DECISIONS or decision == 'UNCLEAR':
            raise ValueError('Unresolved researcher decisions remain')
        target = row['target_existing_theme_id']
        if decision != 'MERGE_WITH_EXISTING_EMERGENT' and target:
            raise ValueError('Unexpected merge target')
        if decision == 'EXCLUDE_AS_NON_THEME':
            if not row['reason'].strip():
                raise ValueError('Exclusion requires a reason')
            excluded.append(identity)
            continue
        if not readable(row['final_label']):
            raise ValueError('Readable final label required')
        if decision == 'KEEP_SEPARATE' and set(source[source.theme_id.eq(identity)].emergent_blog_theme_label) != {row['final_label']}:
            raise ValueError('Use RENAME_ONLY to change a label')
        if decision == 'MERGE_WITH_EXISTING_EMERGENT':
            if target not in ids or target == identity or aspects[target] != aspects[identity]:
                raise ValueError('Merge requires a different same-aspect target')
            if lookup[target]['researcher_decision'] not in {'KEEP_SEPARATE', 'RENAME_ONLY'}:
                raise ValueError('Merge target must be retained; chains/cycles/exclusions are rejected')
            if row['final_label'] != lookup[target]['final_label'] or not row['reason'].strip():
                raise ValueError('Merge requires confirmed common label and substantive equivalence reason')
        groups.setdefault(target or identity, []).append(identity)
    rows, lineage = [], source[['theme_id','aspect','review_id','chunk_id','blog_mention_id','evidence_text','english_gloss']].copy()
    lineage['emergent_theme_id'] = ''
    lineage['researcher_decision'] = lineage.theme_id.map(lambda x: lookup[x]['researcher_decision'])
    lineage['reason'] = lineage.theme_id.map(lambda x: lookup[x]['reason'])
    for target, sources in sorted(groups.items()):
        group = source[source.theme_id.isin(sources)].sort_values(['review_id','blog_mention_id'])
        label, aspect = lookup[target]['final_label'], aspects[target][0]
        identity = stable_id(aspect, label, sources)
        n = group.review_id.nunique()
        rows.append(dict(emergent_theme_id=identity, emergent_theme_label=label, aspect=aspect, theme_origin='blog_emergent',
            support_reviews=n,total_included_reviews=25,review_prevalence=n/25,support_mentions=group.blog_mention_id.nunique(),
            representative_evidence=group.evidence_text.iloc[0],evidence_gloss_en=group.english_gloss.iloc[0],
            source_existing_theme_ids=json.dumps(sorted(sources)), researcher_decision='MERGE_WITH_EXISTING_EMERGENT' if len(sources)>1 else lookup[target]['researcher_decision'],
            low_support_flag=n<2,singleton_flag=n==1,finalized=True))
        lineage.loc[lineage.theme_id.isin(sources),'emergent_theme_id'] = identity
    columns = ['emergent_theme_id','emergent_theme_label','aspect','theme_origin','support_reviews','total_included_reviews','review_prevalence','support_mentions','representative_evidence','evidence_gloss_en','source_existing_theme_ids','researcher_decision','low_support_flag','singleton_flag','finalized']
    result = pd.DataFrame(rows, columns=columns)
    if result.emergent_theme_id.duplicated().any():
        raise ValueError('Duplicate finalized ID')
    summary = dict(source_emergent_ids=len(ids), finalized_emergent_themes=len(result), merged_groups=sum(len(g)>1 for g in groups.values()),
        retained_separate_themes=sum(len(g)==1 for g in groups.values()), excluded_non_theme_observations=len(excluded),
        singleton_themes=int(result.singleton_flag.sum()),low_support_themes=int(result.low_support_flag.sum()),
        reviews_represented=source[~source.theme_id.isin(excluded)].review_id.nunique(),denominator=25,unresolved_decisions=0,api_calls=0)
    return result, lineage, summary


def finalize(root=ROOT):
    integrity = verify_integrity(root)
    source = load_source(root)
    from .blog_analysis import included_reviews
    reviews, chunks = included_reviews()
    parents = read(root / 'blog_review_summary_v1.csv').review_id.tolist()
    if set(parents) != set(reviews.document_id):
        raise ValueError('Parent review summary differs from usable source reviews')
    chunk_parents = chunks.set_index('unit_id').document_id.to_dict()
    if any(chunk_parents.get(row.chunk_id) != row.review_id for row in source.itertuples()):
        raise ValueError('Chunk/parent lineage mismatch')
    taxonomy, lineage, summary = build_taxonomy(source, read(root / DECISION_FILE), parents)
    verify_integrity(root)
    safe_write(root / TAXONOMY, taxonomy.to_csv(index=False))
    safe_write(root / 'blog_emergent_theme_lineage_v1.csv', lineage.to_csv(index=False))
    summary.update(frozen_hash_verification=integrity, finalized=True,
        taxonomy_sha256=sha(root / TAXONOMY), decisions_sha256=sha(root / DECISION_FILE),
        lineage_sha256=sha(root / 'blog_emergent_theme_lineage_v1.csv'))
    safe_write(root / MANIFEST, json.dumps(summary, indent=2))
    return summary


def finalized_summary(root=ROOT):
    manifest = json.loads((root / MANIFEST).read_text(encoding='utf-8'))
    if not manifest.get('finalized') or manifest.get('unresolved_decisions') != 0:
        raise ValueError('Emergent finalization is pending')
    for file, key in [(TAXONOMY,'taxonomy_sha256'),(DECISION_FILE,'decisions_sha256'),('blog_emergent_theme_lineage_v1.csv','lineage_sha256')]:
        if sha(root/file) != manifest[key]:
            raise ValueError('Stale emergent finalization; rerun finalizer')
    verify_integrity(root)
    frame = pd.read_csv(root/TAXONOMY)
    return frame.rename(columns={'emergent_theme_id':'theme_id','emergent_theme_label':'theme_label'})


def save_decision(root, identity, decision, target, label, reason, expected_hash):
    path = root / DECISION_FILE
    lock = root / 'blog_emergent_review.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        if sha(path) != expected_hash:
            raise ValueError('Decisions changed in another session. Reload before saving.')
        frame = read(path)
        if identity not in set(frame.existing_theme_id) or decision not in DECISIONS:
            raise ValueError('Invalid source ID or decision')
        if decision == 'EXCLUDE_AS_NON_THEME' and not reason.strip():
            raise ValueError('Exclusion requires a reason')
        if decision != 'UNCLEAR' and decision != 'EXCLUDE_AS_NON_THEME' and not readable(label):
            raise ValueError('Readable label required')
        if decision == 'MERGE_WITH_EXISTING_EMERGENT':
            source = load_source(root).groupby('theme_id').aspect.first().to_dict()
            if target == identity or target not in source or source[target] != source[identity] or not reason.strip():
                raise ValueError('Merge requires same-aspect target and substantive equivalence reason')
        frame.loc[frame.existing_theme_id.eq(identity), ['researcher_decision','target_existing_theme_id','final_label','reason']] = [decision,target if decision=='MERGE_WITH_EXISTING_EMERGENT' else '',label,reason]
        safe_write(path, frame.to_csv(index=False))
    finally:
        os.close(fd)
        lock.unlink()

def public_text(value):
    """Redact known source authors and URLs from display copies, preserving audit evidence."""
    text = str(value)
    from .blog_analysis import RAW
    authors = {str(row.get('author', '')).strip() for row in json.loads(RAW.read_text(encoding='utf-8-sig'))}
    for author in sorted(authors, key=len, reverse=True):
        if author:
            text = re.sub(re.escape(author), '[author redacted]', text, flags=re.I)
    text = re.sub(r'(?i)(?:https?://|www\.)[^\s<>]+', '[URL redacted]', text)
    return text
