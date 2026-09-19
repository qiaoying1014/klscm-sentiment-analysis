"""Offline synthesis contract, lineage, review and presentation regression tests."""
import copy
import json
import os
from pathlib import Path
import socket

import pytest

from marathon_absa import participant_experience as workflow
from marathon_absa import participant_experience_dashboard_data as loader


@pytest.mark.parametrize('action', ['submit', 'collect'])
@pytest.mark.parametrize('environment_key', [None, 'test-existing-environment-key'])
def test_api_actions_load_repository_dotenv(tmp_path, monkeypatch, action, environment_key):
    import openai

    repository = tmp_path / 'repository'
    repository.mkdir()
    (repository / '.env').write_text('OPENAI_API_KEY=test-dotenv-key\n', encoding='utf-8')
    monkeypatch.setattr(workflow, '__file__', str(repository / 'marathon_absa' / 'participant_experience.py'))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('PYTHON_DOTENV_DISABLED', raising=False)
    if environment_key:
        monkeypatch.setenv('OPENAI_API_KEY', environment_key)
    monkeypatch.setattr(workflow, 'verify_requests', lambda root: None)

    class ClientReached(Exception):
        pass

    def fake_client():
        assert os.environ['OPENAI_API_KEY'] == (environment_key or 'test-dotenv-key')
        raise ClientReached

    monkeypatch.setattr(openai, 'OpenAI', fake_client)
    with pytest.raises(ValueError, match='--run-api'):
        getattr(workflow, action)(tmp_path)
    assert os.environ.get('OPENAI_API_KEY') == environment_key
    with pytest.raises(ClientReached):
        getattr(workflow, action)(tmp_path, run_api=True)
    assert not (tmp_path / 'submission_started.json').exists()


@pytest.fixture(scope='module')
def package():
    return workflow.build_evidence()


def candidate(theme):
    evidence = theme['representative_evidence'][0]['evidence_id']
    claims = {f: {'text': 'Test-only grounded observation.', 'evidence_ids': [evidence]} for f in loader.FIELDS}
    for field in ('positive_experience_summary', 'negative_experience_summary', 'mixed_experience_summary'):
        claims[field] = {'text': 'Insufficient evidence at this level.', 'evidence_ids': []}
    claims['organizer_implication']['text'] = 'Organizers may consider reviewing this test-only observation.'
    claims['evidence_scope_note']['text'] = 'Test-only scope; singleton blog evidence where applicable.'
    return {'theme_id': theme['theme_id'], 'claims': claims}


def prepared(tmp_path, monkeypatch, package):
    monkeypatch.setattr(workflow, 'build_evidence', lambda: copy.deepcopy(package))
    root = tmp_path / 'release'
    workflow.prepare(root, model='test-model')
    return root


def import_fixture(root, package):
    rows = [{'custom_id': t['theme_id'], 'response': {'status_code': 200, 'body': {'status': 'completed',
             'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': json.dumps(candidate(t))}]}]}}} for t in package['themes']]
    path = root / 'test-responses.jsonl'
    workflow.write_new(path, '\n'.join(json.dumps(r) for r in rows), raw=True)
    workflow.import_responses(root, path)


def approve_fixture(root):
    rows = loader.read(root / 'review.json')
    for row in rows:
        row.update(decision='APPROVE', reviewer='Test fixture only', reason='Synthetic lifecycle test, never production content')
    (root / 'review.json').write_text(json.dumps(rows), encoding='utf-8')


def test_real_counts_units_and_frozen_lineage(package):
    assert len(package['aspects']) == 20
    assert package['research_counts']['source_clusters'] == 101
    assert package['research_counts']['instagram_mentions'] == 15486
    assert package['source_units'] == {'instagram': {'unit': 'documents', 'total': 7704}, 'blog': {'unit': 'parent_reviews', 'total': 25}}
    assert len(package['themes']) == 99
    assert sum(t['source_coverage'] == 'cross_source' for t in package['themes']) == 47
    assert sum(t['source_coverage'] == 'instagram_only' for t in package['themes']) == 17
    assert sum(t['source_coverage'] == 'blog_only_emergent' for t in package['themes']) == 35
    assert all(loader.sha(Path(p)) == h for p, h in package['input_hashes'].items())


@pytest.mark.parametrize('mutation', ['aspect', 'coverage', 'sentiment', 'provenance', 'denominator', 'pooled', 'support', 'evidence', 'label'])
def test_invalid_evidence_rejected(package, mutation):
    p = copy.deepcopy(package)
    t = p['themes'][0]
    if mutation == 'aspect': t['aspect_id'] = 'invented_aspect'
    if mutation == 'coverage': t['source_coverage'] = 'combined'
    if mutation == 'sentiment': t['sentiment_evidence']['blog']['counts']['invented'] = 1
    if mutation == 'provenance': p['input_hashes'] = {}
    if mutation == 'denominator': p['source_units']['blog']['unit'] = 'chunks'
    if mutation == 'pooled': p['pooled_prevalence'] = True
    if mutation == 'support': t['blog_support_reviews'] = 999
    if mutation == 'evidence': t['representative_evidence'] = []
    if mutation == 'label': t['experience_classification'] = '92% confidence'
    with pytest.raises(ValueError): loader.validate_package(p)


@pytest.mark.parametrize('counts,expected', [({'positive': 1}, 0), ({'negative': 1}, 1),
    ({'positive': 1000, 'negative': 1}, 2), ({'mixed': 1}, 2), ({'neutral': 2}, 3), ({}, 3)])
def test_presence_rule_without_arbitrary_threshold(counts, expected):
    assert loader.classify({'instagram': {'counts': counts}, 'blog': {'counts': {}}}) == loader.CATEGORIES[expected]


def test_singleton_parent_support_not_mentions(package):
    singletons = [t for t in package['themes'] if t['theme_origin'] == 'blog_emergent' and t['blog_support_reviews'] == 1]
    assert len(singletons) == 24
    assert all('singleton' in t['scope_note'].lower() for t in singletons)
    assert all(len({e['parent_id'] for e in t['representative_evidence']}) == 1 for t in singletons)


def test_prepare_offline_and_no_overwrite(tmp_path, monkeypatch, package):
    def denied(*a, **kw): raise AssertionError('Network forbidden')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    root = prepared(tmp_path, monkeypatch, package)
    assert workflow.verify_requests(root)[1]['request_count'] == 99
    assert not (root / 'finalized_manifest.json').exists()
    with pytest.raises(FileExistsError): workflow.prepare(root)
    with pytest.raises(ValueError, match='approval'): workflow.submit(root)


def test_complete_review_release_and_filters(tmp_path, monkeypatch, package):
    root = prepared(tmp_path, monkeypatch, package)
    import_fixture(root, package)
    with pytest.raises(FileNotFoundError): loader.load_participant_experience_dashboard_data(root)
    with pytest.raises(ValueError, match='Unresolved'): workflow.finalize(root, 'Test')
    approve_fixture(root)
    workflow.finalize(root, 'Test')
    def denied(*a, **kw): raise AssertionError('Runtime network forbidden')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    result = loader.load_participant_experience_dashboard_data(root)
    assert len(result['insights']) == 99
    before = loader.digest(result)
    rows = loader.filter_themes(result['evidence']['themes'], coverage='blog_only_emergent')
    rows[0]['theme_label'] = 'mutated copy'
    assert loader.digest(result) == before
    with pytest.raises(FileExistsError): workflow.finalize(root, 'Test')


@pytest.mark.parametrize('mutation', ['unknown_theme', 'unknown_evidence', 'uncited_implication', 'unsupported_sentiment', 'uncautious_implication'])
def test_invalid_claims_rejected(package, mutation):
    theme = package['themes'][0]
    row = candidate(theme)
    if mutation == 'unknown_theme': row['theme_id'] = 'invented'
    if mutation == 'unknown_evidence': row['claims']['organizer_insight']['evidence_ids'] = ['missing']
    if mutation == 'uncited_implication': row['claims']['organizer_implication']['evidence_ids'] = []
    if mutation == 'uncautious_implication': row['claims']['organizer_implication']['text'] = 'Add exactly ten counters.'
    if mutation == 'unsupported_sentiment':
        theme = copy.deepcopy(theme)
        theme['representative_evidence'][0]['sentiment'] = 'negative'
        row['claims']['positive_experience_summary'] = {'text': 'Praise', 'evidence_ids': [theme['representative_evidence'][0]['evidence_id']]}
    with pytest.raises(ValueError): loader.validate_insight(row, theme)


def test_modified_requests_and_partial_response_rejected(tmp_path, monkeypatch, package):
    root = prepared(tmp_path, monkeypatch, package)
    empty = root / 'empty.jsonl'
    empty.write_text('', encoding='utf-8')
    with pytest.raises(ValueError, match='Incomplete'): workflow.import_responses(root, empty)
    (root / 'requests.jsonl').write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='modified'): workflow.verify_requests(root)


def test_frozen_release_tampering_rejected(tmp_path, monkeypatch, package):
    root = prepared(tmp_path, monkeypatch, package)
    import_fixture(root, package)
    approve_fixture(root)
    workflow.finalize(root, 'Test')
    (root / 'finalized.json').write_text('[]', encoding='utf-8')
    with pytest.raises(ValueError, match='hash'): loader.load_participant_experience_dashboard_data(root)


def test_dashboard_verified_release_page_and_no_network(monkeypatch):
    from streamlit.testing.v1 import AppTest
    def denied(*a, **kw): raise AssertionError('Network forbidden')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    app = AppTest.from_file('absa_dashboard.py', default_timeout=45).run()
    app.sidebar.radio[0].set_value('Participant Experience & Organizer Insights').run(timeout=45)
    assert not app.exception
    assert any('Showing 99 of 99 finalized themes.' in item.value for item in app.caption)
    assert not any('this page presents the frozen Instagram analysis' in i.value for i in app.info)


def test_dashboard_finalized_fixture(tmp_path, monkeypatch, package):
    from streamlit.testing.v1 import AppTest
    root = prepared(tmp_path, monkeypatch, package)
    import_fixture(root, package)
    approve_fixture(root)
    workflow.finalize(root, 'Test')
    script = "from pathlib import Path\nfrom marathon_absa.participant_experience_page import render_participant_experience\nrender_participant_experience(Path(" + repr(str(root)) + "))"
    app = AppTest.from_string(script, default_timeout=60).run()
    assert not app.exception
    next(s for s in app.selectbox if s.key == 'experience_aspect').set_value('facilities').run(timeout=60)
    assert not app.exception
    assert len(next(s for s in app.selectbox if s.key == 'experience_aspect').options) == 21


def test_runtime_modules_have_no_inference():
    for name in ('participant_experience_dashboard_data.py', 'participant_experience_page.py'):
        text = (Path('marathon_absa') / name).read_text(encoding='utf-8')
        assert all(x not in text for x in ('OpenAI(', 'SentenceTransformer(', 'model.encode(', 'batches.create('))
