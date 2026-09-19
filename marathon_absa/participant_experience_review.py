"""Offline researcher review persistence; never generates or releases synthesis."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

from .participant_experience_dashboard_data import (
    ROOT, FIELDS, digest, load_evidence, read, sha, validate_insight,
)

CLAIM_STATES = ('PENDING', 'ACCEPT', 'REVISE', 'REJECT')
DECISIONS = ('PENDING', 'APPROVE', 'REJECT')


def load_review(root: Path = ROOT) -> dict:
    """Verify frozen evidence and candidate provenance without importing API code."""
    root = Path(root)
    names = ['evidence.json', 'evidence_manifest.json', 'request_manifest.json',
             'requests.jsonl', 'candidates.json', 'review.json']
    before = {name: sha(root / name) for name in names}
    package = load_evidence(root)
    candidates = read(root / 'candidates.json')
    request = read(root / 'request_manifest.json')
    if (candidates['generation_status'] != 'generated_pending_review'
            or candidates['request_manifest_hash'] != before['request_manifest.json']
            or request['input_artifact_hash'] != before['evidence.json']
            or request['requests_hash'] != before['requests.jsonl']
            or candidates['model'] != request['model']):
        raise ValueError('Candidate/request provenance changed')
    themes = {t['theme_id']: t for t in package['themes']}
    originals = {r['theme_id']: r for r in candidates['insights']}
    reviews = read(root / 'review.json')
    if (len(originals) != len(candidates['insights']) or originals.keys() != themes.keys()
            or len(reviews) != len(themes)
            or {r['theme_id'] for r in reviews} != themes.keys()):
        raise ValueError('Incomplete or duplicate review/candidate coverage')
    provenance = candidates.get('retry_provenance', {})
    for name, expected in provenance.get('source_hashes', {}).items():
        if sha(Path(name)) != expected:
            raise ValueError('Generation/retry source changed: ' + name)
    if provenance.get('mechanical_correction_audit'):
        audit = read(root / provenance['mechanical_correction_audit'])
        if digest(audit) != provenance['mechanical_correction_audit_hash']:
            raise ValueError('Mechanical correction audit changed')
    for row in reviews:
        identity = row['theme_id']
        original = originals[identity]
        validate_insight(original, themes[identity])
        if row['candidate_hash'] != digest(original):
            raise ValueError('Stale/tampered candidate hash')
        lineage = provenance.get('themes', {}).get(identity)
        if lineage and lineage['insight_hash'] != digest(original):
            raise ValueError('Candidate differs from generation provenance')
        validate_insight(row['insight'], themes[identity])
        if row['decision'] not in DECISIONS:
            raise ValueError('Unknown review decision')
    if before != {name: sha(root / name) for name in names}:
        raise ValueError('Artifacts changed during loading; reload')
    return dict(package=package, themes=themes, candidates=candidates,
                originals=originals, reviews=reviews, request=request,
                token=digest(before), hashes=before)


def claim_reviews(row: dict) -> dict:
    return copy.deepcopy(row.get('claim_reviews', {
        field: {'state': 'PENDING', 'note': ''} for field in FIELDS}))


def resolve_evidence(theme: dict, ids: list) -> list:
    lookup = {e['evidence_id']: e for e in theme['representative_evidence']}
    if len(ids) != len(set(ids)) or any(i not in lookup for i in ids):
        raise ValueError('Unresolved or duplicate evidence reference')
    return [copy.deepcopy(lookup[i]) for i in ids]


def save_review(root: Path, token: str, theme_id: str, insight: dict,
                claims: dict, decision: str, reviewer: str, reason: str) -> None:
    """Locked optimistic transaction, flushed temporary file, atomic replacement."""
    root = Path(root)
    lock = root / '.researcher_review.lock'
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError('Another save is active; reload and retry. A crash may leave a lock.') from exc
    temporary = None
    try:
        os.close(handle)
        if any((root / n).exists() for n in ('finalized.json', 'finalized_manifest.json')):
            raise ValueError('Released or partially finalized package is read-only')
        data = load_review(root)
        if token != data['token']:
            raise ValueError('Stale review session; reload before saving')
        if not reviewer.strip() or not reason.strip():
            raise ValueError('Reviewer identity and reason are required for every save')
        if decision not in DECISIONS or set(claims) != set(FIELDS):
            raise ValueError('Invalid theme decision or claim review coverage')
        original = data['originals'][theme_id]
        validate_insight(insight, data['themes'][theme_id])
        for field, review in claims.items():
            if set(review) != {'state', 'note'} or review['state'] not in CLAIM_STATES or not isinstance(review['note'], str):
                raise ValueError('Invalid claim review state')
            changed = insight['claims'][field] != original['claims'][field]
            if changed and review['state'] != 'REVISE':
                raise ValueError('Changed claims must be marked REVISE')
            if review['state'] in {'REVISE', 'REJECT'} and not review['note'].strip():
                raise ValueError('Revised/rejected claims require a note')
        if decision == 'APPROVE' and any(c['state'] not in {'ACCEPT', 'REVISE'} for c in claims.values()):
            raise ValueError('Approval requires every claim to be accepted or revised')
        row = next(r for r in data['reviews'] if r['theme_id'] == theme_id)
        previous = copy.deepcopy({k: v for k, v in row.items() if k != 'review_history'})
        timestamp = datetime.now(timezone.utc).isoformat()
        row.setdefault('review_history', []).append({'saved_at': timestamp, 'previous': previous})
        row.update(insight=copy.deepcopy(insight), claim_reviews=copy.deepcopy(claims),
                   decision=decision, reviewer=reviewer.strip(), reason=reason.strip(),
                   reviewed_at=timestamp, review_schema_version=1)
        payload = json.dumps(data['reviews'], ensure_ascii=False, indent=2, allow_nan=False)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                         dir=root, prefix='.review-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if data['hashes'] != {name: sha(root / name) for name in data['hashes']}:
            raise ValueError('Artifacts changed during save; reload')
        os.replace(temporary, root / 'review.json')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)
