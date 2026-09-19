"""Researcher workflow tests: production is read-only; all saves use tmp_path."""
import copy
import json
import shutil
from pathlib import Path
import socket

import pytest
from streamlit.testing.v1 import AppTest

from marathon_absa import participant_experience_review as review
from marathon_absa.participant_experience_dashboard_data import ROOT, FIELDS, read, sha


@pytest.fixture
def workspace(tmp_path):
    for name in ('evidence.json', 'evidence_manifest.json', 'requests.jsonl',
                 'request_manifest.json', 'candidates.json', 'review.json',
                 'mechanical_deduplication_audit.json'):
        shutil.copy2(ROOT / name, tmp_path / name)
    return tmp_path


def save(root, data, decision='PENDING', claims=None, insight=None, reviewer='Fixture researcher'):
    row = data['reviews'][0]
    review.save_review(root, data['token'], row['theme_id'], insight or row['insight'],
                       claims or review.claim_reviews(row), decision, reviewer, 'Temporary fixture only')


def test_round_trip_revision_history_and_candidate_preservation(workspace):
    data = review.load_review(workspace)
    candidate_hash = sha(workspace / 'candidates.json')
    row = data['reviews'][0]
    insight = copy.deepcopy(row['insight'])
    field = FIELDS[0]
    insight['claims'][field]['text'] = 'Synthetic researcher revision for persistence testing.'
    claims = {f: {'state': 'ACCEPT', 'note': ''} for f in FIELDS}
    claims[field] = {'state': 'REVISE', 'note': 'Synthetic fixture revision'}
    save(workspace, data, 'APPROVE', claims, insight)
    after = review.load_review(workspace)
    saved = after['reviews'][0]
    assert saved['decision'] == 'APPROVE'
    assert saved['claim_reviews'] == claims
    assert saved['insight'] == insight
    assert saved['reviewer'] == 'Fixture researcher'
    assert saved['review_history'][0]['previous'] == row
    assert sha(workspace / 'candidates.json') == candidate_hash
    save(workspace, after, 'REJECT', claims, insight)
    assert read(workspace / 'review.json')[0]['decision'] == 'REJECT'
    assert len(read(workspace / 'review.json')[0]['review_history']) == 2
    assert not (workspace / 'finalized.json').exists()


def test_stale_session_and_lock(workspace):
    data = review.load_review(workspace)
    save(workspace, data)
    with pytest.raises(ValueError, match='Stale'):
        save(workspace, data)
    (workspace / '.researcher_review.lock').touch()
    with pytest.raises(ValueError, match='Another save'):
        save(workspace, data)


@pytest.mark.parametrize('mutation', ['candidate', 'hash', 'evidence', 'audit'])
def test_tamper_blocks_save(workspace, mutation):
    data = review.load_review(workspace)
    name = {'candidate': 'candidates.json', 'hash': 'review.json',
            'evidence': 'evidence.json', 'audit': 'mechanical_deduplication_audit.json'}[mutation]
    value = read(workspace / name)
    if mutation == 'candidate': value['insights'][0]['claims'][FIELDS[0]]['text'] += ' changed'
    elif mutation == 'hash': value[0]['candidate_hash'] = 'tampered'
    elif mutation == 'evidence': value['themes'][0]['representative_evidence'][0]['text'] += ' changed'
    else: value['operation'] = 'changed'
    (workspace / name).write_text(json.dumps(value), encoding='utf-8')
    before = sha(workspace / 'review.json')
    with pytest.raises(ValueError): save(workspace, data)
    assert sha(workspace / 'review.json') == before


@pytest.mark.parametrize('failure', ['pending_claims', 'missing_reviewer', 'invalid_id', 'unmarked_revision', 'missing_note'])
def test_approval_gates(workspace, failure):
    data = review.load_review(workspace)
    insight = copy.deepcopy(data['reviews'][0]['insight'])
    claims = {f: {'state': 'ACCEPT', 'note': ''} for f in FIELDS}
    reviewer = 'Test researcher'
    if failure == 'pending_claims': claims[FIELDS[0]]['state'] = 'PENDING'
    if failure == 'missing_reviewer': reviewer = ''
    if failure == 'invalid_id': insight['claims'][FIELDS[0]]['evidence_ids'] = ['unknown']
    if failure == 'unmarked_revision': insight['claims'][FIELDS[0]]['text'] += ' revised'
    if failure == 'missing_note': claims[FIELDS[0]]['state'] = 'REVISE'
    before = sha(workspace / 'review.json')
    with pytest.raises(ValueError): save(workspace, data, 'APPROVE', claims, insight, reviewer)
    assert sha(workspace / 'review.json') == before


def test_atomic_failure_leaves_original(workspace, monkeypatch):
    data = review.load_review(workspace)
    before = sha(workspace / 'review.json')
    def fail(*args): raise OSError('simulated replace failure')
    monkeypatch.setattr(review.os, 'replace', fail)
    with pytest.raises(OSError, match='simulated'): save(workspace, data)
    assert sha(workspace / 'review.json') == before
    assert not list(workspace.glob('.review-*.tmp'))
    assert not (workspace / '.researcher_review.lock').exists()


def test_finalized_package_blocks_save(workspace):
    data = review.load_review(workspace)
    (workspace / 'finalized.json').write_text('[]')
    with pytest.raises(ValueError, match='read-only'): save(workspace, data)


def test_exact_source_distinguished_resolution():
    data = review.load_review()
    theme = next(t for t in data['themes'].values() if t['source_coverage'] == 'cross_source')
    ids = [e['evidence_id'] for e in theme['representative_evidence']]
    assert review.resolve_evidence(theme, ids) == theme['representative_evidence']
    assert {e['source'] for e in review.resolve_evidence(theme, ids)} == {'instagram', 'blog'}
    with pytest.raises(ValueError): review.resolve_evidence(theme, ['unknown'])


def test_ui_offline_navigation_and_save(workspace, monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError('Network/inference forbidden')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    import openai
    monkeypatch.setattr(openai, 'OpenAI', forbidden)
    source = "from pathlib import Path\nfrom marathon_absa.participant_experience_review_ui import render_review\nrender_review(Path(" + repr(str(workspace)) + "))"
    app = AppTest.from_string(source, default_timeout=60).run()
    assert not app.exception
    assert len(app.selectbox[2].options) == 99
    assert not any('finalize' in b.label.lower() for b in app.button)
    app.text_input[0].set_value('UI fixture researcher')
    app.text_area[-1].set_value('UI draft round trip')
    app.button[0].click().run()
    assert not app.exception
    assert app.success
    saved = read(workspace / 'review.json')[0]
    assert saved['reviewer'] == 'UI fixture researcher'
    assert saved['decision'] == 'PENDING'
    assert len(saved['claim_reviews']) == 7
    data = review.load_review(workspace)
    corrected = 'physical_experience__rt04'
    app.selectbox[2].select(corrected).run()
    assert not app.exception
    assert any('mechanical_corrections' in j.value for j in app.json)
    singleton = next(t for t in data['themes'].values() if t['blog_support_reviews'] == 1)
    app.selectbox[2].select(singleton['theme_id']).run()
    assert any('Singleton blog support' in w.value for w in app.warning)


def test_organizer_excludes_pending():
    from marathon_absa.participant_experience_dashboard_data import load_participant_experience_dashboard_data
    with pytest.raises(FileNotFoundError): load_participant_experience_dashboard_data()


def test_production_unchanged():
    baseline = read(Path('work/participant_review_before.json'))
    assert all(sha(Path(p)) == expected for p, expected in baseline['hashes'].items())
    assert all(sha(Path(p)) == expected for p, expected in baseline['frozen_hashes'].items())
    assert {r['decision'] for r in read(ROOT / 'review.json')} == {'PENDING'}

@pytest.mark.parametrize('state', review.CLAIM_STATES)
def test_each_claim_state_round_trips(workspace, state):
    data = review.load_review(workspace)
    claims = {f: {'state': state, 'note': 'Temporary review note'} for f in FIELDS}
    save(workspace, data, claims=claims)
    assert review.load_review(workspace)['reviews'][0]['claim_reviews'] == claims


def test_all_99_candidate_claims_resolve_and_provenance_is_present():
    data = review.load_review()
    assert len(data['originals']) == 99
    count = 0
    for identity, insight in data['originals'].items():
        for claim in insight['claims'].values():
            evidence = review.resolve_evidence(data['themes'][identity], claim['evidence_ids'])
            assert [e['evidence_id'] for e in evidence] == claim['evidence_ids']
            count += 1
    assert count == 693
    corrected = data['candidates']['retry_provenance']['themes']['physical_experience__rt04']
    assert corrected['accepted_version'] == 'retry_v2_exact_dedup'
    assert corrected['mechanical_corrections']


def test_ui_module_has_no_generation_inference_or_release_imports():
    import ast
    for filename in ('participant_experience_review_app.py',
                     'marathon_absa/participant_experience_review.py',
                     'marathon_absa/participant_experience_review_ui.py'):
        tree = ast.parse(Path(filename).read_text(encoding='utf-8-sig'))
        imports = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        assert not any(m.split('.')[0] in {'openai', 'transformers', 'sentence_transformers', 'requests', 'httpx'} for m in imports)
        assert 'participant_experience' not in imports
