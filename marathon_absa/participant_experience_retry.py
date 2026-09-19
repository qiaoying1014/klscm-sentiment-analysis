"""Invalid-only recovery. Offline prepare/verify/merge; API actions require approval."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from . import participant_experience as workflow
from . import participant_experience_dashboard_data as grounding

RETRY_PROMPT = '''
Retry citation contract (takes precedence over ambiguous wording above):
ONLY copy evidence IDs verbatim from ALLOWED_EVIDENCE_IDS below. Never construct,
shorten, infer, alter, or cite IDs outside this theme's allowed list. In particular,
unavailable_evidence_ids are provenance metadata for BLANK excerpts, NEVER citations,
including in evidence_scope_note. Deduplicate evidence_ids within every claim.
Every substantive organizer_insight and organizer_implication must have at least one
valid supporting evidence ID. Cite only excerpts that actually support the claim.
When evidence does not support a conclusion, explicitly state insufficient evidence
instead of inventing a citation. For unsupported positive/negative/mixed summaries,
return EXACTLY "Insufficient evidence at this level." with [] and NO added sentence.
Other fields still require supporting citations under the unchanged validator: make
only a limited observation supported by available excerpts, stating limitations;
organizer_implication must begin "Organizers may consider". If no such supported
observation is possible, return an explicit insufficient-evidence claim with [];
this remains invalid and blocked for researcher attention, never auto-accepted.
Do not treat this retry as approval or finalization.
'''


def byte_hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def records(path: Path) -> dict:
    result = {}
    for line in path.read_bytes().splitlines(keepends=True):
        item = json.loads(line)
        identity = item['custom_id']
        if identity in result:
            raise ValueError('Duplicate response/request identity')
        result[identity] = (item, line)
    return result


def parse_response(item: dict) -> dict:
    if item.get('error') or item['response']['status_code'] != 200:
        raise ValueError('Failed batch response')
    body = item['response']['body']
    if body.get('status') != 'completed':
        raise ValueError('Incomplete model response')
    output = [c['text'] for o in body['output'] if o.get('type') == 'message'
              for c in o.get('content', []) if c.get('type') == 'output_text']
    if len(output) != 1:
        raise ValueError('Missing, refused or ambiguous model response')
    return json.loads(output[0])


def audit(root: Path) -> tuple:
    package, manifest = workflow.verify_requests(root)
    themes = {t['theme_id']: t for t in package['themes']}
    requests = records(root / 'requests.jsonl')
    responses = records(root / 'responses.jsonl')
    if len(themes) != 99 or requests.keys() != themes.keys() or responses.keys() != themes.keys():
        raise ValueError('Expected exactly 99 unique themes, requests and responses')
    if manifest['request_count'] != 99:
        raise ValueError('Original request count differs')
    submission = grounding.read(root / 'submission.json')
    if submission['input_file_id'] != grounding.read(root / 'upload.json')['file_id']:
        raise ValueError('Submission/upload lineage mismatch')
    valid, invalid = {}, {}
    for identity, theme in themes.items():
        request = requests[identity][0]
        expected = copy.deepcopy(theme)
        for e in expected['representative_evidence']:
            e.pop('parent_id')
            e.pop('source_evidence_id')
        payload = {'source_units': package['source_units'], 'theme': expected,
                   'model_limitation': package['model_limitation']}
        if (json.loads(request['body']['input']) != payload
                or request['body']['instructions'] != workflow.PROMPT
                or request['body']['model'] != manifest['model']
                or request['body']['text']['format']['schema'] != workflow.response_schema()
                or request['method'] != 'POST' or request['url'] != '/v1/responses'):
            raise ValueError('Original request differs from finalized evidence/prompt/schema')
        try:
            insight = parse_response(responses[identity][0])
            grounding.validate_insight(insight, theme)
        except (ValueError, KeyError, TypeError) as error:
            invalid[identity] = str(error)
        else:
            valid[identity] = insight
    return package, manifest, requests, responses, valid, invalid


def retry_requests(package: dict, requests: dict, invalid: dict) -> str:
    rows = []
    for theme in package['themes']:
        identity = theme['theme_id']
        if identity not in invalid:
            continue
        row = copy.deepcopy(requests[identity][0])
        allowed = [e['evidence_id'] for e in theme['representative_evidence']]
        row['body']['instructions'] += RETRY_PROMPT + '\nALLOWED_EVIDENCE_IDS: ' + json.dumps(allowed)
        rows.append(json.dumps(row, ensure_ascii=False))
    return '\n'.join(rows) + '\n'


def source_hashes(root: Path) -> dict:
    # Includes submission and previous collection audit, not just model output.
    return {p.name: grounding.sha(p) for p in sorted(root.iterdir()) if p.is_file()}


def implementation_hashes() -> dict:
    return {p.name: grounding.sha(p) for p in
            (Path(__file__), Path(workflow.__file__), Path(grounding.__file__))}


def prepare(root: Path, retry: Path) -> dict:
    if retry.exists():
        raise FileExistsError('Use a new retry directory; never overwrite attempts')
    if any((root / name).exists() for name in ('candidates.json', 'review.json', 'finalized.json')):
        raise ValueError('Recovery requires an unreleased, unreviewed original package')
    before = source_hashes(root)
    package, manifest, requests, responses, valid, invalid = audit(root)
    if not invalid:
        raise ValueError('No invalid themes to retry')
    content = retry_requests(package, requests, invalid)
    preserved = b''.join(raw for identity, (_, raw) in responses.items() if identity in valid)
    retry.mkdir(parents=True)
    (retry / 'preserved_valid_responses.jsonl').write_bytes(preserved)
    workflow.write_new(retry / 'requests.jsonl', content, raw=True)
    info = {'version': 'participant_experience_invalid_retry_v1', 'created_at': workflow.now(),
            'root': str(root.resolve()), 'source_hashes': before,
            'implementation_hashes': implementation_hashes(), 'model': manifest['model'],
            'retry_theme_ids': list(invalid), 'validation_errors': invalid,
            'preserved_theme_ids': list(valid), 'request_count': len(invalid),
            'requests_hash': grounding.sha(retry / 'requests.jsonl'),
            'preserved_hash': byte_hash(preserved),
            'original_response_line_hashes': {k: byte_hash(v[1]) for k, v in responses.items()},
            'original_insight_hashes': {k: grounding.digest(v) for k, v in valid.items()},
            'frozen_input_count': len(package['input_hashes']),
            'status': 'PREPARED_AWAITING_EXPLICIT_API_APPROVAL', 'api_calls': 0}
    if source_hashes(root) != before:
        raise ValueError('Original artifacts changed during preparation')
    workflow.write_new(retry / 'retry_manifest.json', info)
    return info


def verify(root: Path, retry: Path) -> tuple:
    info = grounding.read(retry / 'retry_manifest.json')
    if info['root'] != str(root.resolve()) or info['implementation_hashes'] != implementation_hashes():
        raise ValueError('Retry root or implementation changed; audit a new attempt')
    for name, expected in info['source_hashes'].items():
        if grounding.sha(root / name) != expected:
            raise ValueError('Original artifact hash changed: ' + name)
    package, manifest, requests, responses, valid, invalid = audit(root)
    if list(valid) != info['preserved_theme_ids'] or list(invalid) != info['retry_theme_ids']:
        raise ValueError('Retry selection changed')
    if info['request_count'] != len(invalid) or info['model'] != manifest['model']:
        raise ValueError('Retry manifest count/model mismatch')
    if info['original_response_line_hashes'] != {k: byte_hash(v[1]) for k, v in responses.items()}:
        raise ValueError('Original response line hashes changed')
    if info['original_insight_hashes'] != {k: grounding.digest(v) for k, v in valid.items()}:
        raise ValueError('Preserved insight hashes changed')
    expected = retry_requests(package, requests, invalid).encode('utf-8')
    if (retry / 'requests.jsonl').read_bytes() != expected or byte_hash(expected) != info['requests_hash']:
        raise ValueError('Retry requests changed or include valid themes')
    preserved = b''.join(raw for identity, (_, raw) in responses.items() if identity in valid)
    if (retry / 'preserved_valid_responses.jsonl').read_bytes() != preserved or byte_hash(preserved) != info['preserved_hash']:
        raise ValueError('Preserved valid response bytes changed')
    return info, package, responses, valid


def submit(root: Path, retry: Path, run_api=False) -> dict:
    if not run_api:
        raise ValueError('Explicit approval and --run-api required')
    verify(root, retry)
    if (retry / 'submission_started.json').exists() or (retry / 'submission.json').exists():
        raise FileExistsError('Submission already attempted; inspect recorded remote state')
    if (root / 'candidates.json').exists() or (root / 'review.json').exists():
        raise ValueError('Candidates/review already exist')
    client = workflow.create_api_client()
    workflow.write_new(retry / 'submission_started.json', {'started_at': workflow.now(),
                       'retry_manifest_hash': grounding.sha(retry / 'retry_manifest.json')})
    with (retry / 'requests.jsonl').open('rb') as stream:
        upload = client.files.create(file=stream, purpose='batch')
    workflow.write_new(retry / 'upload.json', {'file_id': upload.id})
    batch = client.batches.create(input_file_id=upload.id, endpoint='/v1/responses', completion_window='24h')
    workflow.write_new(retry / 'submission.json', batch.model_dump())
    return {'batch_id': batch.id}


def submission_lineage(retry: Path) -> dict:
    started = grounding.read(retry / 'submission_started.json')
    submission = grounding.read(retry / 'submission.json')
    if (started['retry_manifest_hash'] != grounding.sha(retry / 'retry_manifest.json')
            or submission['input_file_id'] != grounding.read(retry / 'upload.json')['file_id']):
        raise ValueError('Retry submission lineage mismatch')
    return submission


def collect(root: Path, retry: Path, run_api=False) -> dict:
    if not run_api:
        raise ValueError('Explicit approval and --run-api required')
    info, _, _, _ = verify(root, retry)
    submission = submission_lineage(retry)
    if (retry / 'responses.jsonl').exists() or (retry / 'collection.json').exists():
        raise FileExistsError('Collection already recorded; use offline merge')
    client = workflow.create_api_client()
    batch = client.batches.retrieve(submission['id'])
    if batch.status != 'completed' or not batch.output_file_id:
        raise ValueError('Retry batch not completed')
    if batch.input_file_id != submission['input_file_id']:
        raise ValueError('Remote retry input file mismatch')
    workflow.write_new(retry / 'responses.jsonl', client.files.content(batch.output_file_id).text, raw=True)
    workflow.write_new(retry / 'collection.json', {'batch': batch.model_dump(),
                       'response_hash': grounding.sha(retry / 'responses.jsonl'),
                       'retry_manifest_hash': grounding.sha(retry / 'retry_manifest.json'),
                       'collected_at': workflow.now(), 'expected_count': info['request_count']})
    return merge(root, retry)


def merge(root: Path, retry: Path) -> dict:
    info, package, original, valid = verify(root, retry)
    submission = submission_lineage(retry)
    collection = grounding.read(retry / 'collection.json')
    if (collection['response_hash'] != grounding.sha(retry / 'responses.jsonl')
            or collection['retry_manifest_hash'] != grounding.sha(retry / 'retry_manifest.json')
            or collection['batch']['id'] != submission['id']
            or collection['batch']['input_file_id'] != submission['input_file_id']
            or collection['batch']['status'] != 'completed'
            or not collection['batch']['output_file_id']):
        raise ValueError('Retry collection lineage mismatch')
    responses = records(retry / 'responses.jsonl')
    if set(responses) != set(info['retry_theme_ids']):
        raise ValueError('Retry response coverage must be invalid themes only')
    themes = {t['theme_id']: t for t in package['themes']}
    accepted = dict(valid)
    for identity, (item, _) in responses.items():
        insight = parse_response(item)
        grounding.validate_insight(insight, themes[identity])
        accepted[identity] = insight
    if len(accepted) != 99 or accepted.keys() != themes.keys():
        raise ValueError('Exactly 99 unique validated candidates required')
    provenance = {}
    for identity, insight in accepted.items():
        retried = identity in responses
        item, raw = (responses if retried else original)[identity]
        grounding.validate_insight(insight, themes[identity])
        provenance[identity] = {'retried': retried, 'accepted_version': 'retry_v1' if retried else 'original',
            'response_id': item['response']['body'].get('id'),
            'batch_request_id': item.get('id'), 'response_line_hash': byte_hash(raw),
            'insight_hash': grounding.digest(insight),
            'response_file_hash': collection['response_hash'] if retried else info['source_hashes']['responses.jsonl']}
    # All validation completes before either review artifact is created.
    for name in ('candidates.json', 'review.json'):
        if (root / name).exists():
            raise FileExistsError('Existing candidate/review artifacts are never overwritten')
    verify(root, retry)
    rows = [accepted[k] for k in themes]
    workflow.write_new(root / 'candidates.json', {'generation_status': 'generated_pending_review',
        'generated_at': workflow.now(), 'response_hash': info['source_hashes']['responses.jsonl'],
        'request_manifest_hash': grounding.sha(root / 'request_manifest.json'), 'model': info['model'],
        'retry_provenance': {'manifest': str((retry / 'retry_manifest.json').resolve()),
            'manifest_hash': grounding.sha(retry / 'retry_manifest.json'),
            'collection_hash': grounding.sha(retry / 'collection.json'),
            'requests_hash': info['requests_hash'], 'themes': provenance}, 'insights': rows})
    workflow.write_new(root / 'review.json', [{'theme_id': r['theme_id'], 'decision': 'PENDING',
        'reviewer': '', 'reason': '', 'candidate_hash': grounding.digest(r), 'insight': r} for r in rows])
    return {'candidates': len(rows), 'preserved': len(valid), 'retried': len(responses), 'review_status': 'pending'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'verify', 'submit', 'collect', 'merge'])
    parser.add_argument('--root', type=Path, default=grounding.ROOT)
    parser.add_argument('--retry-dir', type=Path)
    parser.add_argument('--run-api', action='store_true')
    args = parser.parse_args(argv)
    retry = args.retry_dir or args.root / 'retry_v1'
    if args.action in {'submit', 'collect'}:
        result = globals()[args.action](args.root, retry, args.run_api)
    elif args.action == 'verify':
        result = verify(args.root, retry)[0]
    else:
        result = globals()[args.action](args.root, retry)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
