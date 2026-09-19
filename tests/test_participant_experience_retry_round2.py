"""Follow-up tests use temporary artifacts and never call an API."""
import copy
import json
import shutil

import pytest

from marathon_absa import participant_experience_retry_round2 as second
from marathon_absa import participant_experience_dashboard_data as grounding
from marathon_absa import participant_experience as workflow


@pytest.fixture(scope='module')
def production_state():
    return second.base_state(grounding.ROOT)


@pytest.fixture
def prepared(tmp_path, monkeypatch, production_state):
    root = tmp_path / 'release'
    root.mkdir()
    for name in ('requests.jsonl', 'request_manifest.json', 'responses.jsonl'):
        shutil.copyfile(grounding.ROOT / name, root / name)
    monkeypatch.setattr(second, 'base_state', lambda root: copy.deepcopy(production_state))
    def denied():
        raise AssertionError('API forbidden')
    monkeypatch.setattr(workflow, 'create_api_client', denied)
    second.prepare(root)
    return root


def collection(root, mutation=None):
    info, package, _ = second.verify(root)
    directory = root / 'retry_v2'
    rows = []
    for theme in package['themes']:
        if theme['theme_id'] not in info['retry_theme_ids']:
            continue
        eid = theme['representative_evidence'][0]['evidence_id']
        claims = {f: {'text': 'Synthetic test only.', 'evidence_ids': [eid]} for f in grounding.FIELDS}
        for f in ('positive_experience_summary', 'negative_experience_summary', 'mixed_experience_summary'):
            claims[f] = {'text': 'Insufficient evidence at this level.', 'evidence_ids': []}
        claims['organizer_implication']['text'] = 'Organizers may consider this synthetic test.'
        claims['evidence_scope_note']['text'] = 'Synthetic singleton scope.'
        if mutation == 'unknown': claims['organizer_insight']['evidence_ids'] = [eid[:-1]]
        if mutation == 'duplicate': claims['organizer_insight']['evidence_ids'] = [eid, eid]
        insight = {'theme_id': theme['theme_id'], 'claims': claims}
        rows.append({'custom_id': theme['theme_id'], 'response': {'status_code': 200, 'body': {
            'status': 'completed', 'id': 'synthetic', 'output': [{'type': 'message', 'content': [
                {'type': 'output_text', 'text': json.dumps(insight)}]}]}}})
    if mutation == 'coverage': rows.pop()
    batch = {'id': 'test-batch', 'input_file_id': 'test-input', 'output_file_id': 'test-output', 'status': 'completed'}
    workflow.write_new(directory / 'submission_started.json', {'retry_manifest_hash': grounding.sha(directory / 'retry_manifest.json')})
    workflow.write_new(directory / 'upload.json', {'file_id': 'test-input'})
    workflow.write_new(directory / 'submission.json', batch)
    workflow.write_new(directory / 'responses.jsonl', '\n'.join(json.dumps(r) for r in rows), raw=True)
    workflow.write_new(directory / 'collection.json', {'batch': batch,
        'response_hash': grounding.sha(directory / 'responses.jsonl'),
        'retry_manifest_hash': grounding.sha(directory / 'retry_manifest.json')})


def test_four_only_enums_and_95_byte_preserved(prepared):
    info, package, accepted = second.verify(prepared)
    assert info['request_count'] == 4
    assert len(accepted) == 95
    assert {v[1] for v in accepted.values()} == {'original', 'retry_v1'}
    requests = second.prior.records(prepared / 'retry_v2/requests.jsonl')
    originals = second.prior.records(prepared / 'requests.jsonl')
    themes = {t['theme_id']: t for t in package['themes']}
    assert set(requests) == set(info['retry_theme_ids'])
    for identity, (request, _) in requests.items():
        assert request['body']['input'] == originals[identity][0]['body']['input']
        for claim in request['body']['text']['format']['schema']['properties']['claims']['properties'].values():
            assert claim['properties']['evidence_ids']['items']['enum'] == [e['evidence_id'] for e in themes[identity]['representative_evidence']]
    assert (prepared / 'retry_v2/preserved_valid_responses.jsonl').read_bytes() == b''.join(v[0][1] for v in accepted.values())
    with pytest.raises(ValueError, match='approval'): second.submit(prepared)


def test_merge_99_pending_preserves_original_and_first_retry(prepared):
    _, _, accepted = second.verify(prepared)
    collection(prepared)
    assert second.merge(prepared)['candidates'] == 99
    result = grounding.read(prepared / 'candidates.json')
    rows = {r['theme_id']: r for r in result['insights']}
    assert len(rows) == 99
    for identity, (record, version, _) in accepted.items():
        assert rows[identity] == second.prior.parse_response(record[0])
        assert result['retry_provenance']['themes'][identity]['accepted_version'] == version
    assert all(r['decision'] == 'PENDING' for r in grounding.read(prepared / 'review.json'))
    assert not (prepared / 'finalized.json').exists()


@pytest.mark.parametrize('mutation', ['unknown', 'duplicate', 'coverage'])
def test_invalid_never_repaired_or_reviewed(prepared, mutation):
    collection(prepared, mutation)
    before = (prepared / 'retry_v2/responses.jsonl').read_bytes()
    with pytest.raises(ValueError): second.merge(prepared)
    assert not (prepared / 'candidates.json').exists()
    assert not (prepared / 'review.json').exists()
    assert (prepared / 'retry_v2/responses.jsonl').read_bytes() == before


def test_tampered_request_blocked_before_api(prepared):
    path = prepared / 'retry_v2/requests.jsonl'
    with path.open('ab') as stream: stream.write(b' ')
    with pytest.raises(ValueError, match='requests changed'): second.submit(prepared, True)


def test_repeated_submission_blocked(prepared):
    workflow.write_new(prepared / 'retry_v2/submission_started.json', {})
    with pytest.raises(FileExistsError): second.submit(prepared, True)
