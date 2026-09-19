"""Four-theme follow-up to retry_v1; preserve all 95 validated responses."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from . import participant_experience_retry as prior
from . import participant_experience as workflow
from . import participant_experience_dashboard_data as grounding


def checked_collection(directory: Path) -> dict:
    submission = prior.submission_lineage(directory)
    collection = grounding.read(directory / 'collection.json')
    batch = collection['batch']
    if (collection['response_hash'] != grounding.sha(directory / 'responses.jsonl')
            or collection['retry_manifest_hash'] != grounding.sha(directory / 'retry_manifest.json')
            or batch['id'] != submission['id'] or batch['input_file_id'] != submission['input_file_id']
            or batch['status'] != 'completed' or not batch['output_file_id']):
        raise ValueError('Collection lineage mismatch')
    return collection


def base_state(root: Path) -> tuple:
    previous = root / 'retry_v1'
    info, package, originals, valid = prior.verify(root, previous)
    checked_collection(previous)
    responses = prior.records(previous / 'responses.jsonl')
    if set(responses) != set(info['retry_theme_ids']):
        raise ValueError('Previous retry coverage mismatch')
    themes = {t['theme_id']: t for t in package['themes']}
    accepted = {k: (originals[k], 'original', root / 'responses.jsonl') for k in valid}
    failures = {}
    for identity, record in responses.items():
        try:
            grounding.validate_insight(prior.parse_response(record[0]), themes[identity])
        except (ValueError, KeyError, TypeError) as error:
            failures[identity] = str(error)
        else:
            accepted[identity] = (record, 'retry_v1', previous / 'responses.jsonl')
    if len(themes) != 99 or len(accepted) + len(failures) != 99:
        raise ValueError('Incomplete predecessor coverage')
    paths = [root / name for name in info['source_hashes']]
    paths += [p for p in previous.iterdir() if p.is_file()]
    snapshots = {str(p.resolve()): grounding.sha(p) for p in paths}
    return package, accepted, failures, snapshots


def request_text(root: Path, package: dict, failures: dict) -> str:
    originals = prior.records(root / 'requests.jsonl')
    rows = []
    for theme in package['themes']:
        identity = theme['theme_id']
        if identity not in failures:
            continue
        row = copy.deepcopy(originals[identity][0])
        allowed = [e['evidence_id'] for e in theme['representative_evidence']]
        schema = row['body']['text']['format']['schema']
        schema['properties']['theme_id']['enum'] = [identity]
        for claim in schema['properties']['claims']['properties'].values():
            claim['properties']['evidence_ids']['items']['enum'] = allowed
        row['body']['instructions'] += prior.RETRY_PROMPT + '\nALLOWED_EVIDENCE_IDS: ' + json.dumps(allowed)
        row['body']['instructions'] += '\nBefore returning, remove repeated identical IDs within each claim. Never repeat an ID in one evidence_ids array.'
        rows.append(json.dumps(row, ensure_ascii=False))
    return '\n'.join(rows) + '\n'


def prepare(root: Path) -> dict:
    directory = root / 'retry_v2'
    if directory.exists():
        raise FileExistsError('Follow-up attempt already exists')
    if any((root / name).exists() for name in ('candidates.json', 'review.json', 'finalized.json')):
        raise ValueError('Existing candidate/review/release')
    package, accepted, failures, snapshots = base_state(root)
    if not failures:
        raise ValueError('No invalid themes')
    content = request_text(root, package, failures)
    preserved = b''.join(v[0][1] for v in accepted.values())
    directory.mkdir()
    (directory / 'preserved_valid_responses.jsonl').write_bytes(preserved)
    workflow.write_new(directory / 'requests.jsonl', content, raw=True)
    info = {'root': str(root.resolve()), 'created_at': workflow.now(),
        'version': 'participant_experience_retry_v2_enum', 'source_hashes': snapshots,
        'implementation_hash': grounding.sha(Path(__file__)), 'request_count': len(failures),
        'retry_theme_ids': list(failures), 'validation_errors': failures,
        'preserved_theme_ids': list(accepted), 'preserved_hash': prior.byte_hash(preserved),
        'requests_hash': grounding.sha(directory / 'requests.jsonl'),
        'status': 'PREPARED_AWAITING_EXPLICIT_API_APPROVAL', 'api_calls': 0}
    workflow.write_new(directory / 'retry_manifest.json', info)
    verify(root)
    return info


def verify(root: Path) -> tuple:
    directory = root / 'retry_v2'
    info = grounding.read(directory / 'retry_manifest.json')
    if info['root'] != str(root.resolve()) or info['implementation_hash'] != grounding.sha(Path(__file__)):
        raise ValueError('Follow-up root or implementation changed')
    for name, expected in info['source_hashes'].items():
        if grounding.sha(Path(name)) != expected:
            raise ValueError('Predecessor artifact changed: ' + name)
    package, accepted, failures, snapshots = base_state(root)
    if snapshots != info['source_hashes'] or list(accepted) != info['preserved_theme_ids'] or list(failures) != info['retry_theme_ids']:
        raise ValueError('Predecessor selection/provenance changed')
    expected = request_text(root, package, failures).encode('utf-8')
    preserved = b''.join(v[0][1] for v in accepted.values())
    if (directory / 'requests.jsonl').read_bytes() != expected or prior.byte_hash(expected) != info['requests_hash'] or info['request_count'] != len(failures):
        raise ValueError('Follow-up requests changed')
    if (directory / 'preserved_valid_responses.jsonl').read_bytes() != preserved or prior.byte_hash(preserved) != info['preserved_hash']:
        raise ValueError('Preserved response bytes changed')
    return info, package, accepted


def submit(root: Path, run_api=False) -> dict:
    if not run_api:
        raise ValueError('Explicit approval and --run-api required')
    verify(root)
    directory = root / 'retry_v2'
    if any((directory / n).exists() for n in ('submission_started.json', 'submission.json')):
        raise FileExistsError('Submission already attempted')
    if any((root / n).exists() for n in ('candidates.json', 'review.json', 'finalized.json')):
        raise ValueError('Existing candidate/review/release')
    client = workflow.create_api_client()
    workflow.write_new(directory / 'submission_started.json', {'started_at': workflow.now(),
        'retry_manifest_hash': grounding.sha(directory / 'retry_manifest.json')})
    with (directory / 'requests.jsonl').open('rb') as stream:
        upload = client.files.create(file=stream, purpose='batch')
    workflow.write_new(directory / 'upload.json', {'file_id': upload.id})
    batch = client.batches.create(input_file_id=upload.id, endpoint='/v1/responses', completion_window='24h')
    workflow.write_new(directory / 'submission.json', batch.model_dump())
    return {'batch_id': batch.id}


def collect(root: Path, run_api=False) -> dict:
    if not run_api:
        raise ValueError('Explicit approval and --run-api required')
    verify(root)
    directory = root / 'retry_v2'
    submission = prior.submission_lineage(directory)
    if (directory / 'responses.jsonl').exists() or (directory / 'collection.json').exists():
        raise FileExistsError('Already collected; use offline merge')
    client = workflow.create_api_client()
    batch = client.batches.retrieve(submission['id'])
    if batch.status != 'completed' or not batch.output_file_id:
        raise ValueError('Retry batch not completed')
    if batch.input_file_id != submission['input_file_id']:
        raise ValueError('Remote input file mismatch')
    workflow.write_new(directory / 'responses.jsonl', client.files.content(batch.output_file_id).text, raw=True)
    workflow.write_new(directory / 'collection.json', {'batch': batch.model_dump(),
        'response_hash': grounding.sha(directory / 'responses.jsonl'),
        'retry_manifest_hash': grounding.sha(directory / 'retry_manifest.json'), 'collected_at': workflow.now()})
    return merge(root)


def merge(root: Path) -> dict:
    info, package, preserved = verify(root)
    directory = root / 'retry_v2'
    checked_collection(directory)
    responses = prior.records(directory / 'responses.jsonl')
    if set(responses) != set(info['retry_theme_ids']):
        raise ValueError('Follow-up must cover only remaining invalid themes')
    themes = {t['theme_id']: t for t in package['themes']}
    records = dict(preserved)
    records.update({k: (v, 'retry_v2', directory / 'responses.jsonl') for k, v in responses.items()})
    rows, provenance, errors = [], {}, {}
    for identity, (record, version, path) in records.items():
        try:
            insight = prior.parse_response(record[0])
            grounding.validate_insight(insight, themes[identity])
        except (ValueError, KeyError, TypeError) as error:
            errors[identity] = str(error)
            continue
        rows.append(insight)
        provenance[identity] = {'accepted_version': version, 'retried': version != 'original',
            'response_id': record[0]['response']['body'].get('id'), 'batch_request_id': record[0].get('id'),
            'response_line_hash': prior.byte_hash(record[1]), 'insight_hash': grounding.digest(insight),
            'response_file': str(path.resolve()), 'response_file_hash': grounding.sha(path)}
    if errors:
        raise ValueError('Retry validation failed; no review created: ' + json.dumps(errors))
    if len(rows) != 99 or {r['theme_id'] for r in rows} != themes.keys():
        raise ValueError('Exactly 99 unique validated candidates required')
    for name in ('candidates.json', 'review.json'):
        if (root / name).exists():
            raise FileExistsError('Existing candidate/review artifact')
    verify(root)
    lookup = {r['theme_id']: r for r in rows}
    rows = [lookup[k] for k in themes]
    workflow.write_new(root / 'candidates.json', {'generation_status': 'generated_pending_review',
        'generated_at': workflow.now(), 'response_hash': grounding.sha(root / 'responses.jsonl'),
        'request_manifest_hash': grounding.sha(root / 'request_manifest.json'),
        'model': grounding.read(root / 'request_manifest.json')['model'],
        'retry_provenance': {'themes': provenance, 'manifest_hash': grounding.sha(directory / 'retry_manifest.json'),
            'manifest': str((directory / 'retry_manifest.json').resolve()),
            'collection_hash': grounding.sha(directory / 'collection.json'), 'source_hashes': info['source_hashes']},
        'insights': rows})
    workflow.write_new(root / 'review.json', [{'theme_id': row['theme_id'], 'decision': 'PENDING',
        'reviewer': '', 'reason': '', 'candidate_hash': grounding.digest(row), 'insight': row} for row in rows])
    return {'candidates': 99, 'preserved': len(preserved), 'retried': len(responses), 'review_status': 'pending'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'verify', 'submit', 'collect', 'merge'])
    parser.add_argument('--root', type=Path, default=grounding.ROOT)
    parser.add_argument('--run-api', action='store_true')
    args = parser.parse_args(argv)
    if args.action in {'submit', 'collect'}:
        result = globals()[args.action](args.root, args.run_api)
    elif args.action == 'verify':
        result = verify(args.root)[0]
    else:
        result = globals()[args.action](args.root)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
