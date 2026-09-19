"""Offline, audited exact-citation deduplication after the second retry."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from . import participant_experience_retry_round2 as previous
from . import participant_experience_dashboard_data as grounding
from . import participant_experience as workflow


def deduplicate(row: dict, theme: dict) -> tuple:
    """Remove only repeated identical allowed IDs; never repair unknown IDs or text."""
    result = copy.deepcopy(row)
    allowed = {e['evidence_id'] for e in theme['representative_evidence']}
    changes = []
    for field, claim in result['claims'].items():
        ids = claim['evidence_ids']
        if not isinstance(ids, list) or any(not isinstance(x, str) or x not in allowed for x in ids):
            raise ValueError('Unknown/malformed evidence IDs cannot be repaired')
        unique = list(dict.fromkeys(ids))
        if unique != ids:
            changes.append({'field': field, 'before': ids[:], 'after': unique,
                            'removed_count': len(ids) - len(unique)})
            claim['evidence_ids'] = unique
    grounding.validate_insight(result, theme)
    return result, changes


def build(root: Path) -> tuple:
    info, package, preserved = previous.verify(root)
    directory = root / 'retry_v2'
    previous.checked_collection(directory)
    latest = previous.prior.records(directory / 'responses.jsonl')
    if set(latest) != set(info['retry_theme_ids']):
        raise ValueError('Second retry coverage mismatch')
    sources = dict(preserved)
    sources.update({k: (v, 'retry_v2', directory / 'responses.jsonl') for k, v in latest.items()})
    themes = {t['theme_id']: t for t in package['themes']}
    if len(sources) != 99 or sources.keys() != themes.keys():
        raise ValueError('Exactly 99 unique source responses required')
    paths = [p for folder in (root, root / 'retry_v1', directory) for p in folder.iterdir() if p.is_file()]
    hashes = {str(p.resolve()): grounding.sha(p) for p in paths}
    rows, provenance, corrections = [], {}, []
    for identity in themes:
        record, version, path = sources[identity]
        original = previous.prior.parse_response(record[0])
        changes = []
        try:
            grounding.validate_insight(original, themes[identity])
            accepted = original
        except ValueError:
            # Never alter an already-preserved original or first-retry candidate.
            if version != 'retry_v2':
                raise
            accepted, changes = deduplicate(original, themes[identity])
            if not changes:
                raise ValueError('Invalid response has no permissible mechanical correction')
        grounding.validate_insight(accepted, themes[identity])
        rows.append(accepted)
        provenance[identity] = {'accepted_version': version + ('_exact_dedup' if changes else ''),
            'retried': version != 'original', 'response_id': record[0]['response']['body'].get('id'),
            'batch_request_id': record[0].get('id'), 'response_line_hash': previous.prior.byte_hash(record[1]),
            'response_file': str(path.resolve()), 'response_file_hash': grounding.sha(path),
            'raw_insight_hash': grounding.digest(original), 'insight_hash': grounding.digest(accepted),
            'mechanical_corrections': changes}
        if changes:
            corrections.append({'theme_id': identity, 'changes': changes,
                                'before_hash': grounding.digest(original), 'after_hash': grounding.digest(accepted)})
    audit = {'created_at': workflow.now(), 'operation': 'exact_valid_citation_deduplication',
        'implementation_hash': grounding.sha(Path(__file__)), 'source_hashes': hashes,
        'corrections': corrections, 'candidates': len(rows),
        'unchanged_candidates': len(rows) - len(corrections),
        'frozen_inputs_verified': len(package['input_hashes']), 'api_calls': 0}
    candidates = {'generation_status': 'generated_pending_review', 'generated_at': workflow.now(),
        'response_hash': grounding.sha(root / 'responses.jsonl'),
        'request_manifest_hash': grounding.sha(root / 'request_manifest.json'),
        'model': grounding.read(root / 'request_manifest.json')['model'],
        'retry_provenance': {'themes': provenance, 'source_hashes': hashes,
            'mechanical_correction_audit': 'mechanical_deduplication_audit.json',
            'mechanical_correction_audit_hash': grounding.digest(audit)}, 'insights': rows}
    reviews = [{'theme_id': row['theme_id'], 'decision': 'PENDING', 'reviewer': '', 'reason': '',
                'candidate_hash': grounding.digest(row), 'insight': row} for row in rows]
    return candidates, reviews, audit


def merge(root: Path) -> dict:
    names = ('mechanical_deduplication_audit.json', 'candidates.json', 'review.json')
    if any((root / n).exists() for n in (*names, 'finalized.json', 'finalized_manifest.json')):
        raise FileExistsError('Existing correction/review/release artifacts; never overwrite')
    candidates, reviews, audit = build(root)
    for name, expected in audit['source_hashes'].items():
        if grounding.sha(Path(name)) != expected:
            raise ValueError('Source changed during validation')
    for name, value in zip(names, (audit, candidates, reviews)):
        workflow.write_new(root / name, value)
    return {'candidates': len(reviews), 'mechanically_corrected': len(audit['corrections']),
            'unchanged_candidates': audit['unchanged_candidates'], 'review_status': 'pending', 'api_calls': 0}


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=grounding.ROOT)
    print(json.dumps(merge(parser.parse_args().root), indent=2))


if __name__ == '__main__':
    main()
