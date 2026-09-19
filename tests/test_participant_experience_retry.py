"""Offline recovery tests; synthetic replacement claims never enter production."""
import copy
import json
import shutil
from pathlib import Path

import pytest

from marathon_absa import participant_experience_retry as recovery
from marathon_absa import participant_experience_dashboard_data as grounding
from marathon_absa import participant_experience as workflow


@pytest.fixture
def attempt(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('No API calls permitted')
    monkeypatch.setattr(workflow, 'create_api_client', denied)
    root = tmp_path / 'release'
    root.mkdir()
    # Reconstruct the pre-retry stage, even after production has advanced to
    # candidate review. Never copy later candidate/review outputs into recovery.
    original_names = grounding.read(
        grounding.ROOT / 'retry_v1' / 'retry_manifest.json')['source_hashes']
    for path in grounding.ROOT.iterdir():
        if path.is_file() and path.name in original_names:
            shutil.copyfile(path, root / path.name)
    retry = root / 'retry_v1'
    info = recovery.prepare(root, retry)
    return root, retry, info


def synthetic_collection(root, retry):
    info, package, _, _ = recovery.verify(root, retry)
    rows = []
    for theme in package['themes']:
        if theme['theme_id'] not in info['retry_theme_ids']:
            continue
        evidence = theme['representative_evidence'][0]['evidence_id']
        claims = {f: {'text': 'Synthetic test only.', 'evidence_ids': [evidence]} for f in grounding.FIELDS}
        for f in ('positive_experience_summary', 'negative_experience_summary', 'mixed_experience_summary'):
            claims[f] = {'text': 'Insufficient evidence at this level.', 'evidence_ids': []}
        claims['organizer_implication']['text'] = 'Organizers may consider this synthetic test only.'
        claims['evidence_scope_note']['text'] = 'Synthetic singleton scope.'
        insight = {'theme_id': theme['theme_id'], 'claims': claims}
        rows.append({'id': 'synthetic-request', 'custom_id': theme['theme_id'], 'response': {'status_code': 200,
            'body': {'id': 'synthetic-response', 'status': 'completed', 'output': [{'type': 'message',
            'content': [{'type': 'output_text', 'text': json.dumps(insight)}]}]}}})
    batch = {'id': 'synthetic-batch', 'input_file_id': 'synthetic-file', 'status': 'completed', 'output_file_id': 'synthetic-output'}
    workflow.write_new(retry / 'submission_started.json', {'retry_manifest_hash': grounding.sha(retry / 'retry_manifest.json')})
    workflow.write_new(retry / 'upload.json', {'file_id': 'synthetic-file'})
    workflow.write_new(retry / 'submission.json', batch)
    workflow.write_new(retry / 'responses.jsonl', '\n'.join(json.dumps(r) for r in rows) + '\n', raw=True)
    workflow.write_new(retry / 'collection.json', {'batch': batch,
        'response_hash': grounding.sha(retry / 'responses.jsonl'),
        'retry_manifest_hash': grounding.sha(retry / 'retry_manifest.json')})
    return rows


def test_invalid_only_preservation_and_frozen_integrity(attempt):
    root, retry, info = attempt
    assert info['request_count'] == 10
    assert len(info['preserved_theme_ids']) == 89
    assert len(info['source_hashes']) >= 9
    assert set(recovery.records(retry / 'requests.jsonl')) == set(info['retry_theme_ids'])
    assert not set(info['preserved_theme_ids']) & set(info['retry_theme_ids'])
    recovery.verify(root, retry)
    assert all(grounding.sha(root / n) == h for n, h in info['source_hashes'].items())
    package = grounding.load_evidence(root)
    assert len(package['input_hashes']) == 411
    assert all(grounding.sha(Path(n)) == h for n, h in package['input_hashes'].items())
    assert not (root / 'review.json').exists()
    with pytest.raises(FileExistsError):
        recovery.prepare(root, retry)


@pytest.mark.parametrize('mutation', ['unknown', 'truncated', 'duplicate', 'missing_implication', 'extended_sentinel'])
def test_strict_validator_unchanged_and_no_repairs(attempt, mutation):
    root, retry, _ = attempt
    _, package, _, valid = recovery.verify(root, retry)
    row = copy.deepcopy(next(iter(valid.values())))
    theme = next(t for t in package['themes'] if t['theme_id'] == row['theme_id'])
    claim = row['claims']['organizer_implication']
    if mutation == 'unknown': claim['evidence_ids'] = ['invented']
    if mutation == 'truncated': claim['evidence_ids'] = [claim['evidence_ids'][0][:-1]]
    if mutation == 'duplicate': claim['evidence_ids'] *= 2
    if mutation == 'missing_implication': claim['evidence_ids'] = []
    if mutation == 'extended_sentinel':
        row['claims']['mixed_experience_summary'] = {'text': 'Insufficient evidence at this level. Extra.', 'evidence_ids': []}
    before = copy.deepcopy(row)
    with pytest.raises(ValueError): grounding.validate_insight(row, theme)
    assert row == before


def test_merge_99_unique_provenance_pending_and_bytes(attempt):
    root, retry, info = attempt
    synthetic_collection(root, retry)
    assert recovery.merge(root, retry) == {'candidates': 99, 'preserved': 89, 'retried': 10, 'review_status': 'pending'}
    candidates = grounding.read(root / 'candidates.json')
    assert len({r['theme_id'] for r in candidates['insights']}) == 99
    for row in candidates['insights']:
        identity = row['theme_id']
        provenance = candidates['retry_provenance']['themes'][identity]
        assert provenance['retried'] == (identity in info['retry_theme_ids'])
        if not provenance['retried']:
            assert grounding.digest(row) == info['original_insight_hashes'][identity]
            assert provenance['response_line_hash'] == info['original_response_line_hashes'][identity]
    assert all(r['decision'] == 'PENDING' for r in grounding.read(root / 'review.json'))
    assert not (root / 'finalized.json').exists()
    assert all(grounding.sha(root / n) == h for n, h in info['source_hashes'].items())
    with pytest.raises(FileExistsError): recovery.merge(root, retry)


@pytest.mark.parametrize('mutation', ['invalid', 'missing', 'duplicate', 'valid_theme', 'wrong_batch', 'response_hash'])
def test_failed_retry_never_creates_review(attempt, mutation):
    root, retry, _ = attempt
    rows = synthetic_collection(root, retry)
    if mutation == 'invalid':
        insight = json.loads(rows[0]['response']['body']['output'][0]['content'][0]['text'])
        insight['claims']['organizer_insight']['evidence_ids'] = ['invented']
        rows[0]['response']['body']['output'][0]['content'][0]['text'] = json.dumps(insight)
    if mutation == 'missing': rows.pop()
    if mutation == 'duplicate': rows.append(rows[0])
    if mutation == 'valid_theme': rows[0]['custom_id'] = grounding.read(retry / 'retry_manifest.json')['preserved_theme_ids'][0]
    (retry / 'responses.jsonl').write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
    collection = grounding.read(retry / 'collection.json')
    if mutation != 'response_hash': collection['response_hash'] = grounding.sha(retry / 'responses.jsonl')
    if mutation == 'wrong_batch': collection['batch']['id'] = 'another-batch'
    (retry / 'collection.json').write_text(json.dumps(collection), encoding='utf-8')
    with pytest.raises(ValueError): recovery.merge(root, retry)
    assert not (root / 'review.json').exists()
    assert not (root / 'candidates.json').exists()


@pytest.mark.parametrize('target', ['original', 'requests', 'preserved', 'manifest'])
def test_lineage_tampering_blocks_submit_before_client(attempt, target):
    root, retry, _ = attempt
    path = {'original': root / 'responses.jsonl', 'requests': retry / 'requests.jsonl',
            'preserved': retry / 'preserved_valid_responses.jsonl', 'manifest': retry / 'retry_manifest.json'}[target]
    if target == 'manifest':
        info = grounding.read(path)
        info['retry_theme_ids'].append(info['preserved_theme_ids'][0])
        path.write_text(json.dumps(info), encoding='utf-8')
    else:
        with path.open('ab') as stream: stream.write(b' ')
    with pytest.raises(ValueError): recovery.submit(root, retry, run_api=True)


def test_api_gate_and_repeat_submission_guard(attempt):
    root, retry, _ = attempt
    for action in (recovery.submit, recovery.collect):
        with pytest.raises(ValueError, match='approval'): action(root, retry)
    workflow.write_new(retry / 'submission_started.json', {})
    with pytest.raises(FileExistsError): recovery.submit(root, retry, run_api=True)


def test_mock_submission_uploads_only_ten_original_theme_payloads(attempt, monkeypatch):
    from types import SimpleNamespace
    root, retry, info = attempt
    original = recovery.records(root / 'requests.jsonl')
    def upload(file, purpose):
        rows = [json.loads(line) for line in file.read().splitlines()]
        assert purpose == 'batch'
        assert len(rows) == 10
        assert {r['custom_id'] for r in rows} == set(info['retry_theme_ids'])
        for row in rows:
            assert row['body']['input'] == original[row['custom_id']][0]['body']['input']
            assert 'ALLOWED_EVIDENCE_IDS' in row['body']['instructions']
        return SimpleNamespace(id='mock-file')
    def batch(**kwargs):
        assert kwargs['input_file_id'] == 'mock-file'
        return SimpleNamespace(id='mock-batch', model_dump=lambda: {'id': 'mock-batch', 'input_file_id': 'mock-file'})
    monkeypatch.setattr(workflow, 'create_api_client', lambda: SimpleNamespace(
        files=SimpleNamespace(create=upload), batches=SimpleNamespace(create=batch)))
    assert recovery.submit(root, retry, run_api=True) == {'batch_id': 'mock-batch'}
    with pytest.raises(FileExistsError): recovery.submit(root, retry, run_api=True)
